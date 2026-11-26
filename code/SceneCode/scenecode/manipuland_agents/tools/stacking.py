"""Stacking utilities using collision geometry and physics simulation.

This module provides functionality for:
- Computing initial stack transforms using collision geometry.
- Simulating stack stability using Drake physics.

The key insight is that collision geometry (CoACD convex hulls) is typically
inflated compared to visual geometry, so we must use collision bounds for
accurate stacking.
"""

import logging
import tempfile

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectives,
    MeshcatVisualizer,
    MultibodyPlant,
    ProcessModelDirectives,
    RigidTransform,
    SceneGraph,
    Simulator,
    StartMeshcat,
)

from scenecode.agent_utils.physical_feasibility import _effective_to_scene_transform
from scenecode.agent_utils.room import SceneObject
from scenecode.utils.mesh_loading import load_collision_meshes_from_sdf

console_logger = logging.getLogger(__name__)


def compute_initial_stack_transforms(
    collision_bounds_list: list[tuple[float, float]], base_transform: RigidTransform
) -> list[RigidTransform]:
    """Compute initial world transforms for stack using collision geometry.

    Objects are stacked bottom-to-top along Z-axis. Each object's position
    is computed so its collision z_min sits on top of the previous object's
    collision z_max.

    Args:
        collision_bounds_list: List of (z_min, z_max) for each asset in stack order.
        base_transform: Transform for the base position on the support surface.

    Returns:
        List of world transforms for each stack item.
    """
    transforms = []
    cumulative_z = 0.0
    for z_min, z_max in collision_bounds_list:
        # Offset so object's z_min sits at cumulative height.
        z_offset = cumulative_z - z_min
        translation = base_transform.translation() + np.array([0, 0, z_offset])
        transforms.append(RigidTransform(base_transform.rotation(), translation))
        # Update cumulative height to top of this object.
        cumulative_z = cumulative_z + (z_max - z_min)

    return transforms


def compute_actual_stack_height(
    transforms: list[RigidTransform], collision_bounds_list: list[tuple[float, float]]
) -> float:
    """Compute actual stack height from transforms and collision bounds.

    This function computes the height of the topmost point in the stack,
    taking into account the actual positions of objects (which may have
    settled or nested during simulation).

    Args:
        transforms: World transforms for each object (initial or final).
        collision_bounds_list: List of (z_min, z_max) for each object.

    Returns:
        Height of the topmost point in the stack (max z coordinate).
        Returns 0.0 if transforms is empty.
    """
    if not transforms:
        return 0.0

    top_heights = []
    for transform, (_, z_max) in zip(transforms, collision_bounds_list):
        # Top of object = transform z + z_max (z_max is relative to object origin).
        top_z = transform.translation()[2] + z_max
        top_heights.append(top_z)

    return max(top_heights)


@dataclass
class StackSimulationResult:
    """Result of physics simulation for stack stability."""

    is_stable: bool
    """Whether stack remained stable (all objects within position threshold)."""
    final_transforms: list[RigidTransform]
    """Settled positions after simulation."""
    stable_indices: list[int]
    """Indices of objects that remained stacked."""
    unstable_indices: list[int]
    """Indices of objects that fell or toppled."""
    error_message: str | None = None
    """Error message if simulation failed."""


def simulate_stack_stability(
    scene_objects: list[SceneObject],
    initial_transforms: list[RigidTransform],
    ground_xyz: tuple[float, float, float],
    simulation_time: float,
    simulation_time_step: float,
    position_threshold: float = 0.1,
    output_html_path: Path | None = None,
) -> StackSimulationResult:
    """Simulate stack on artificial surface to check stability.

    Creates temporary Drake simulation:
    1. Ground support surface sized to bottom object's XY footprint
    2. Catch floor at z=-5m to prevent simulation crashes from falling objects
    3. All stack objects placed at initial positions as free bodies
    4. Simulate for configurable duration
    5. Check final positions against initial (position only, not rotation)

    The limited ground surface ensures that unstable objects fall off and are
    reliably detected. The catch floor prevents infinite fall distances that
    would crash the simulation.

    Args:
        scene_objects: List of SceneObjects to simulate (must have sdf_path).
        initial_transforms: Initial world transforms for each object.
        ground_xyz: XYZ coordinates for ground plane (X, Y center; Z is top surface).
        simulation_time: Duration to simulate in seconds.
        simulation_time_step: Simulation time step in seconds.
        position_threshold: Maximum position displacement for stability (meters).
            Default 0.1m (10cm) which cleanly separates settled objects (<1cm)
            from fallen objects (>1m).
        output_html_path: If provided, record simulation and save as interactive
            HTML file. The file can be opened in a browser to replay the physics.

    Returns:
        StackSimulationResult with stability status and final transforms.
    """
    if len(scene_objects) != len(initial_transforms):
        return StackSimulationResult(
            is_stable=False,
            final_transforms=initial_transforms,
            stable_indices=[],
            unstable_indices=list(range(len(scene_objects))),
            error_message="Mismatch between objects and transforms count",
        )

    # Validate all objects have SDF paths.
    for i, obj in enumerate(scene_objects):
        if not obj.sdf_path or not obj.sdf_path.exists():
            return StackSimulationResult(
                is_stable=False,
                final_transforms=initial_transforms,
                stable_indices=[],
                unstable_indices=list(range(len(scene_objects))),
                error_message=f"Object {i} ({obj.name}) missing SDF path",
            )

    try:
        # Build Drake simulation.
        builder = DiagramBuilder()
        plant: MultibodyPlant
        scene_graph: SceneGraph
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=simulation_time_step
        )

        # Set up visualization if recording is requested.
        meshcat = None
        visualizer = None
        if output_html_path is not None:
            meshcat = StartMeshcat()
            console_logger.info(f"Meshcat URL: {meshcat.web_url()}")

        # Compute XY bounding box of bottom object for ground surface.
        # This ensures unstable objects fall off and are reliably detected.
        bottom_collision_meshes = load_collision_meshes_from_sdf(
            scene_objects[0].sdf_path
        )
        if not bottom_collision_meshes:
            # Clean up Meshcat before early return (see comment in finally block).
            if meshcat is not None:
                del meshcat
            return StackSimulationResult(
                is_stable=False,
                final_transforms=initial_transforms,
                stable_indices=[],
                unstable_indices=list(range(len(scene_objects))),
                error_message="Bottom object has no collision geometry",
            )

        # Apply bottom object's scale_factor to collision meshes.
        bottom_scale = scene_objects[0].scale_factor
        if bottom_scale != 1.0:
            for mesh in bottom_collision_meshes:
                mesh.vertices *= bottom_scale

        x_min = min(m.vertices[:, 0].min() for m in bottom_collision_meshes)
        x_max = max(m.vertices[:, 0].max() for m in bottom_collision_meshes)
        y_min = min(m.vertices[:, 1].min() for m in bottom_collision_meshes)
        y_max = max(m.vertices[:, 1].max() for m in bottom_collision_meshes)

        ground_size_x = x_max - x_min
        ground_size_y = y_max - y_min
        ground_box_thickness = 0.1
        ground_z_center = ground_xyz[2] - ground_box_thickness / 2

        # Create ground support surface sized to bottom object footprint.
        ground_sdf_content = f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="ground_plane">
    <static>true</static>
    <pose>{ground_xyz[0]} {ground_xyz[1]} {ground_z_center} 0 0 0</pose>
    <link name="ground_link">
      <collision name="ground_collision">
        <geometry>
          <box>
            <size>{ground_size_x} {ground_size_y} {ground_box_thickness}</size>
          </box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>"""

        # Create catch floor at z=-5m to prevent simulation crashes.
        catch_floor_sdf_content = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="catch_floor">
    <static>true</static>
    <pose>0 0 -5.0 0 0 0</pose>
    <link name="catch_link">
      <collision name="catch_collision">
        <geometry>
          <box><size>10 10 0.1</size></box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>"""

        # Write ground SDF to temp file.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sdf", delete=False
        ) as ground_file:
            ground_file.write(ground_sdf_content)
            ground_sdf_path = ground_file.name

        # Write catch floor SDF to temp file.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sdf", delete=False
        ) as catch_file:
            catch_file.write(catch_floor_sdf_content)
            catch_floor_sdf_path = catch_file.name

        # Build directive with ground, catch floor, and objects.
        # Note: Ground uses <static>true</static> which auto-welds to world.
        # We set the ground pose in the SDF directly to avoid duplicate welds.
        directive_parts = ["directives:"]
        directive_parts.append(
            f"""
- add_model:
    name: ground_plane
    file: file://{ground_sdf_path}"""
        )
        directive_parts.append(
            f"""
- add_model:
    name: catch_floor
    file: file://{catch_floor_sdf_path}"""
        )

        # Add each object as free body.
        model_names = []
        for i, (obj, transform) in enumerate(zip(scene_objects, initial_transforms)):
            model_name = f"stack_obj_{i}"
            model_names.append(model_name)

            effective_transform = transform @ obj.internal_model_pose
            translation = effective_transform.translation()
            angle_axis = effective_transform.rotation().ToAngleAxis()
            angle_deg = angle_axis.angle() * 180 / np.pi
            axis = angle_axis.axis()

            directive_parts.append(
                f"""
- add_model:
    name: {model_name}
    file: file://{obj.sdf_path.absolute()}
    default_free_body_pose:
      base_link:
        translation: [{translation[0]}, {translation[1]}, {translation[2]}]
        rotation: !AngleAxis
          angle_deg: {angle_deg}
          axis: [{axis[0]}, {axis[1]}, {axis[2]}]"""
            )

        directive_yaml = "\n".join(directive_parts)

        # Write directive to temp file.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as directive_file:
            directive_file.write(directive_yaml)
            directive_path = directive_file.name

        try:
            # Load directives into plant.
            directives = LoadModelDirectives(directive_path)
            ProcessModelDirectives(directives, plant, parser=None)
            plant.Finalize()

            # Add visualizer after plant is finalized.
            if meshcat is not None:
                visualizer = MeshcatVisualizer.AddToBuilder(
                    builder, scene_graph, meshcat
                )

            # Build diagram and create simulator.
            diagram = builder.Build()
            simulator = Simulator(diagram)
            context = simulator.get_mutable_context()

            # Start recording if visualizing.
            if visualizer is not None:
                visualizer.StartRecording()

            # Run simulation.
            simulator.AdvanceTo(simulation_time)

            # Stop recording and save HTML if visualizing.
            if visualizer is not None and meshcat is not None:
                visualizer.StopRecording()
                visualizer.PublishRecording()
                html = meshcat.StaticHtml()
                output_html_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_html_path, "w") as f:
                    f.write(html)
                console_logger.info(f"Saved simulation HTML to {output_html_path}")

            # Get final positions.
            plant_context = plant.GetMyContextFromRoot(context)
            final_transforms = []
            stable_indices = []
            unstable_indices = []

            for i, model_name in enumerate(model_names):
                model_instance = plant.GetModelInstanceByName(model_name)
                body_indices = plant.GetBodyIndices(model_instance)
                if body_indices:
                    if len(body_indices) > 1:
                        console_logger.warning(
                            f"Object {i} has multiple bodies: {body_indices}"
                        )

                    body = plant.get_body(body_indices[0])
                    final_effective_pose = plant.EvalBodyPoseInWorld(
                        plant_context, body
                    )
                    final_pose = _effective_to_scene_transform(
                        final_effective_pose, scene_objects[i].internal_model_pose
                    )
                    final_transforms.append(final_pose)

                    # Check position displacement.
                    initial_pos = initial_transforms[i].translation()
                    final_pos = final_pose.translation()
                    displacement = np.linalg.norm(final_pos - initial_pos)

                    if displacement <= position_threshold:
                        stable_indices.append(i)
                    else:
                        unstable_indices.append(i)
                        console_logger.info(
                            f"Object {i} unstable: displacement={displacement:.3f}m "
                            f"(threshold={position_threshold:.3f}m)"
                        )
                else:
                    final_transforms.append(initial_transforms[i])
                    unstable_indices.append(i)

            is_stable = len(unstable_indices) == 0

            return StackSimulationResult(
                is_stable=is_stable,
                final_transforms=final_transforms,
                stable_indices=stable_indices,
                unstable_indices=unstable_indices,
            )

        finally:
            # Clean up temp files.
            Path(directive_path).unlink(missing_ok=True)
            Path(ground_sdf_path).unlink(missing_ok=True)
            Path(catch_floor_sdf_path).unlink(missing_ok=True)

            # Explicitly delete Meshcat on the main thread to avoid threading issues.
            # Drake's Meshcat destructor asserts it must be called from the thread
            # that created it. Without explicit deletion, Python's GC might destroy
            # the Meshcat from a ThreadPoolExecutor worker thread, causing a crash.
            if meshcat is not None:
                del meshcat

    except Exception as e:
        console_logger.error(f"Stack simulation failed: {e}")
        return StackSimulationResult(
            is_stable=False,
            final_transforms=initial_transforms,
            stable_indices=[],
            unstable_indices=list(range(len(scene_objects))),
            error_message=str(e),
        )
