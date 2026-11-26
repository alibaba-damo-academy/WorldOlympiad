"""Fill container utilities using physics simulation.

This module provides functionality for:
- Computing container interior bounds using top rim heuristic.
- Computing fill object spawn transforms.
- Resolving initial fill object collisions using NLP projection.
- Simulating fill objects dropping into containers using Drake physics.
"""

import logging
import tempfile

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectives,
    MeshcatVisualizer,
    ProcessModelDirectives,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    Simulator,
    StartMeshcat,
)
from scipy.spatial import ConvexHull, QhullError

from scenecode.agent_utils.physical_feasibility import (
    _effective_to_scene_transform,
    solve_non_penetration_ik,
)
from scenecode.agent_utils.room import SceneObject
from scenecode.utils.sdf_utils import extract_base_link_name_from_sdf

console_logger = logging.getLogger(__name__)


@dataclass
class ContainerInteriorBounds:
    """Interior bounds of a container for fill object spawning."""

    hull_vertices_2d: np.ndarray
    """2D convex hull vertices of container opening in XY plane."""
    centroid_2d: np.ndarray
    """Centroid of the hull in XY."""
    top_z: float
    """Z coordinate of container top (rim)."""
    bottom_z: float
    """Z coordinate of container bottom (interior floor)."""


@dataclass
class FillSimulationResult:
    """Result of physics simulation for fill container operation."""

    inside_indices: list[int]
    """Indices of new fill objects that stayed inside the container."""
    outside_indices: list[int]
    """Indices of new fill objects that fell outside (to catch floor)."""
    final_transforms: list[RigidTransform]
    """Final transforms for new fill objects after simulation."""
    settled_final_transforms: list[RigidTransform] | None = None
    """Updated transforms for settled objects (may have shifted)."""
    settled_fell_out_indices: list[int] | None = None
    """Indices (into settled list) of settled objects pushed out of container."""
    error_message: str | None = None
    """Error message if simulation failed."""


def compute_container_interior_bounds(
    collision_meshes: list[trimesh.Trimesh],
    top_rim_height_fraction: float = 0.15,
    interior_scale: float = 0.95,
) -> ContainerInteriorBounds:
    """Compute interior bounds of a container using top rim heuristic.

    The algorithm:
    1. Find container's max Z (top of container).
    2. Select vertices within top N% of container height.
    3. Project those vertices to XY plane.
    4. Compute convex hull of projected points.
    5. Scale hull by interior_scale to stay away from walls.

    This gives the opening of the container, avoiding handles/decorations.

    Args:
        collision_meshes: Container collision mesh pieces.
        top_rim_height_fraction: Fraction of height for rim detection.
        interior_scale: Scale factor to shrink interior bounds.

    Returns:
        ContainerInteriorBounds with hull vertices and Z bounds.

    Raises:
        ValueError: If collision meshes are empty or hull computation fails.
    """
    if not collision_meshes:
        raise ValueError("No collision meshes provided")

    # Combine all vertices.
    all_vertices = np.vstack([m.vertices for m in collision_meshes])

    # Find Z extent.
    z_min = float(all_vertices[:, 2].min())
    z_max = float(all_vertices[:, 2].max())
    height = z_max - z_min

    if height <= 0:
        raise ValueError("Container has zero or negative height")

    # Select vertices in top N% of container height.
    z_threshold = z_max - (top_rim_height_fraction * height)
    top_vertices = all_vertices[all_vertices[:, 2] >= z_threshold]

    if len(top_vertices) < 3:
        # Fallback: use all vertices if not enough in top region.
        console_logger.warning(
            f"Only {len(top_vertices)} vertices in top rim region, using all vertices"
        )
        top_vertices = all_vertices

    # Project to XY plane.
    xy_vertices = top_vertices[:, :2]

    # Compute convex hull.
    try:
        hull = ConvexHull(xy_vertices)
    except QhullError as e:
        raise ValueError(f"Failed to compute container interior hull: {e}")

    hull_vertices = xy_vertices[hull.vertices]

    # Compute centroid.
    centroid = hull_vertices.mean(axis=0)

    # Scale hull toward centroid to create interior bounds.
    scaled_hull = centroid + interior_scale * (hull_vertices - centroid)

    return ContainerInteriorBounds(
        hull_vertices_2d=scaled_hull,
        centroid_2d=centroid,
        top_z=z_max,
        bottom_z=z_min,
    )


def _compute_object_extents(
    collision_meshes: list[trimesh.Trimesh],
) -> tuple[float, float, float]:
    """Compute axis-aligned bounding box extents from collision meshes.

    Returns:
        Tuple of (dx, dy, dz) extents along each axis.
    """
    all_vertices = np.vstack([m.vertices for m in collision_meshes])
    mins = all_vertices.min(axis=0)
    maxs = all_vertices.max(axis=0)
    return float(maxs[0] - mins[0]), float(maxs[1] - mins[1]), float(maxs[2] - mins[2])


def _should_flip_for_thick_end_up(
    vertices: np.ndarray, rotation: RotationMatrix
) -> bool:
    """Determine if object should be flipped 180° to put thick end up.

    After orienting an elongated object vertically, compares the XY footprint
    of the top half vs bottom half. If the bottom half has a larger footprint
    (thicker), the object should be flipped.

    This is important for asymmetric objects like utensils (pan flippers,
    spatulas, spoons) where the thick/business end should point upward when
    placed in a container like a utensil crock.

    Args:
        vertices: Original mesh vertices (Nx3).
        rotation: Rotation already applied to align longest axis with Z.

    Returns:
        True if object should be flipped 180° around X axis.
    """
    # Apply rotation to vertices.
    rotated = (rotation.matrix() @ vertices.T).T

    # Find center Z.
    z_min, z_max = rotated[:, 2].min(), rotated[:, 2].max()
    center_z = (z_min + z_max) / 2

    # Split into top and bottom halves.
    top_verts = rotated[rotated[:, 2] > center_z]
    bottom_verts = rotated[rotated[:, 2] <= center_z]

    # Handle edge cases.
    if len(top_verts) < 3 or len(bottom_verts) < 3:
        return False  # Not enough vertices to analyze.

    # Compute XY bounding box area for each half.
    def xy_area(verts: np.ndarray) -> float:
        x_extent = verts[:, 0].max() - verts[:, 0].min()
        y_extent = verts[:, 1].max() - verts[:, 1].min()
        return x_extent * y_extent

    top_area = xy_area(top_verts)
    bottom_area = xy_area(bottom_verts)

    # Flip if bottom is significantly thicker (10% threshold avoids flipping
    # symmetric objects).
    return bottom_area > top_area * 1.1


def _compute_fill_object_rotation(
    extents: tuple[float, float, float],
    container_interior: ContainerInteriorBounds,
    vertices: np.ndarray | None = None,
    aspect_ratio_threshold: float = 2.0,
) -> RotationMatrix:
    """Compute rotation for fill object based on shape and container geometry.

    Algorithm:
    1. Detect non-cubic objects using max/min ratio >= threshold.
    2. Stand up: rotate so shortest axis becomes horizontal.
    3. Align: apply yaw to align longest horizontal axis with container length.
    4. Apply thick-end-up flip if needed for asymmetric objects.

    Cubic objects (aspect ratio < threshold) return identity rotation.

    Args:
        extents: Object extents (dx, dy, dz).
        container_interior: Container interior bounds for alignment.
        vertices: Optional mesh vertices for thick-end-up analysis.
        aspect_ratio_threshold: Ratio of longest/shortest to be considered
            non-cubic and needing orientation.

    Returns:
        RotationMatrix to apply to the object.
    """
    dx, dy, dz = extents
    dims = [dx, dy, dz]

    # Sort dimensions to find shortest/middle/longest.
    sorted_with_idx = sorted(enumerate(dims), key=lambda x: x[1])
    shortest_idx, shortest_val = sorted_with_idx[0]
    middle_idx, middle_val = sorted_with_idx[1]
    longest_idx, longest_val = sorted_with_idx[2]

    # Check if non-cubic (needs special orientation).
    aspect_ratio = longest_val / shortest_val if shortest_val > 0 else 1.0
    if aspect_ratio < aspect_ratio_threshold:
        # Cubic object - no rotation needed.
        return RotationMatrix()

    # Step 1: Stand up - rotate so shortest axis becomes horizontal.
    # In Drake, Z is up by default. We want shortest axis in XY plane.
    if shortest_idx == 2:
        # Z is shortest (e.g., plate lying flat). Rotate 90° around X.
        # This makes: X→X, Y→Z (vertical), Z→-Y (horizontal).
        base_rotation = RotationMatrix(RollPitchYaw([np.pi / 2, 0.0, 0.0]))
        # After rotation: object's X is world X, object's Y is world Z,
        # object's Z is world -Y.
        # Horizontal axes are now: original X and original Z.
        horizontal_dims = [(dims[0], 0), (dims[2], 2)]  # X and Z extents.
    elif shortest_idx == 1:
        # Y is shortest. Rotate 90° around Z then 90° around X.
        # Simpler: rotate -90° around X to make Y horizontal.
        # Y→-Z (horizontal), Z→Y (still horizontal), X→X.
        # Actually: rotate 90° around Z to swap X and Y, then proceed.
        # Simplest: just rotate so Y ends up horizontal.
        # Rotate around Z by 90°: X→Y, Y→-X, Z→Z. Then around X by 90°.
        # Let's use: rotate around Z by 90° to make Y the new X (horizontal).
        base_rotation = RotationMatrix(RollPitchYaw([0.0, 0.0, np.pi / 2]))
        # After: X→Y, Y→-X, Z→Z. Shortest (Y) is now along -X (horizontal).
        # Vertical axis is Z. Horizontal axes are: original Y (now -X) and Z.
        horizontal_dims = [(dims[1], 1), (dims[2], 2)]
    else:
        # X is shortest. Already horizontal in default orientation.
        base_rotation = RotationMatrix()
        # Vertical axis is Z. Horizontal axes are X and Y.
        # Wait, if X is shortest, what's vertical? We need to stand up.
        # "Stand up" means the tall axis should be vertical.
        # If X is shortest, then Y or Z is longest. Z might already be vertical.
        # We want shortest horizontal, longest vertical if possible.
        # If X is shortest and Z is longest: Z is already vertical, good.
        # If X is shortest and Y is longest: rotate to make Y vertical.
        if longest_idx == 1:
            # Y is longest, X is shortest. Rotate 90° around X to make Y vertical.
            base_rotation = RotationMatrix(RollPitchYaw([np.pi / 2, 0.0, 0.0]))
            # After: X→X, Y→Z, Z→-Y. Horizontal axes: X and Z (original).
            horizontal_dims = [(dims[0], 0), (dims[2], 2)]
        else:
            # Z is longest (or middle). Z is vertical, X is horizontal.
            base_rotation = RotationMatrix()
            # Horizontal axes: X and Y.
            horizontal_dims = [(dims[0], 0), (dims[1], 1)]

    # Step 2: Align longest horizontal axis with container length direction.
    # Determine container length direction from hull.
    hull = container_interior.hull_vertices_2d
    x_extent = float(hull[:, 0].max() - hull[:, 0].min())
    y_extent = float(hull[:, 1].max() - hull[:, 1].min())
    container_length_along_x = x_extent >= y_extent

    # Find longest horizontal dimension after base rotation.
    longest_horiz_val, longest_horiz_orig_idx = max(horizontal_dims, key=lambda x: x[0])

    # Determine yaw to align longest horizontal with container length.
    # After base_rotation, we need to know where the longest horizontal axis ended up.
    # Apply base rotation to unit vectors to find current orientation.
    R = base_rotation.matrix()
    orig_axes = np.eye(3)
    rotated_axes = R @ orig_axes  # Columns are rotated X, Y, Z axes.

    # Find which world axis the longest original horizontal axis aligns with.
    longest_horiz_world_dir = rotated_axes[:, longest_horiz_orig_idx]

    # We want this direction to align with container length (X or Y).
    # Compute yaw angle needed.
    # Current direction in XY plane:
    current_angle = np.arctan2(longest_horiz_world_dir[1], longest_horiz_world_dir[0])
    # Target direction:
    target_angle = 0.0 if container_length_along_x else np.pi / 2

    yaw = target_angle - current_angle

    # Apply yaw rotation.
    yaw_rotation = RotationMatrix(RollPitchYaw([0.0, 0.0, yaw]))
    rotation = yaw_rotation @ base_rotation

    # Step 3: Apply thick-end-up check for asymmetric objects.
    if vertices is not None and _should_flip_for_thick_end_up(vertices, rotation):
        # Flip 180° around X axis to put thick end up.
        flip = RotationMatrix(RollPitchYaw([np.pi, 0.0, 0.0]))
        rotation = flip @ rotation

    return rotation


def compute_fill_spawn_transforms(
    fill_collision_meshes: list[list[trimesh.Trimesh]],
    container_interior: ContainerInteriorBounds,
    container_transform: RigidTransform,
    spawn_height_above_rim: float = 0.1,
    height_stagger_fraction: float = 0.75,
    min_height_stagger: float = 0.02,
    rng: np.random.Generator | None = None,
) -> list[RigidTransform]:
    """Compute initial spawn transforms for fill objects.

    Fill objects spawn at staggered heights above the container rim to prevent
    initial overlaps. Each subsequent object spawns higher based on its actual
    post-rotation height (not diagonal, since rotation is deterministic).

    Objects are oriented based on their shape and container geometry:
    - Non-cubic objects (max/min >= 2): rotated to stand up with shortest axis
      horizontal, longest horizontal axis aligned with container length.
    - Cubic objects: no rotation applied.

    Args:
        fill_collision_meshes: List of collision mesh lists for each fill object.
        container_interior: Container interior bounds.
        container_transform: Transform of container in world frame.
        spawn_height_above_rim: Base height above container top for first object.
        height_stagger_fraction: Fraction of post-rotation height for Z spacing.
        min_height_stagger: Minimum stagger between objects (meters).
        rng: Random number generator (uses default if None).

    Returns:
        List of world transforms for each fill object.
    """
    if rng is None:
        rng = np.random.default_rng()

    transforms = []
    hull = container_interior.hull_vertices_2d

    # Compute bounding box of hull for rejection sampling.
    hull_min = hull.min(axis=0)
    hull_max = hull.max(axis=0)

    # Track current spawn Z level (accumulates with per-object stagger).
    current_z = container_interior.top_z + spawn_height_above_rim

    for obj_index, meshes in enumerate(fill_collision_meshes):
        # Compute object extents.
        extents = _compute_object_extents(meshes)
        all_vertices = np.vstack([m.vertices for m in meshes])

        # Generate random XY within hull using rejection sampling.
        max_attempts = 100
        for _ in range(max_attempts):
            x = rng.uniform(hull_min[0], hull_max[0])
            y = rng.uniform(hull_min[1], hull_max[1])
            point = np.array([x, y])

            # Check if point is inside hull.
            if _point_in_polygon(point=point, polygon=hull):
                break
        else:
            # Fallback to centroid if rejection sampling fails.
            x, y = container_interior.centroid_2d
            console_logger.warning("Rejection sampling failed, using centroid")

        # Compute rotation based on object shape and container geometry.
        rotation = _compute_fill_object_rotation(
            extents=extents,
            container_interior=container_interior,
            vertices=all_vertices,
        )

        # Compute post-rotation bounding box.
        rotated_vertices = (rotation.matrix() @ all_vertices.T).T
        rotated_z_min = float(rotated_vertices[:, 2].min())
        rotated_z_max = float(rotated_vertices[:, 2].max())
        rotated_height = rotated_z_max - rotated_z_min

        # Compute spawn Z: object's z_min sits at current layer.
        spawn_z = current_z - rotated_z_min

        # Create local transform (in container frame).
        local_transform = RigidTransform(rotation, [float(x), float(y), float(spawn_z)])

        # Convert to world frame.
        world_transform = container_transform @ local_transform
        transforms.append(world_transform)

        # Stagger by actual height (rotation is deterministic, not random).
        stagger = max(min_height_stagger, rotated_height * height_stagger_fraction)
        current_z += stagger

    return transforms


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Check if a 2D point is inside a convex polygon using cross product.

    Args:
        point: 2D point [x, y].
        polygon: Nx2 array of polygon vertices (ordered).

    Returns:
        True if point is inside polygon.
    """
    n = len(polygon)
    sign = None

    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        # Compute cross product of (p2-p1) and (point-p1).
        d = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])

        if sign is None:
            sign = d >= 0
        elif (d >= 0) != sign:
            return False

    return True


def project_fill_objects_non_penetrating(
    fill_scene_objects: list[SceneObject],
    fill_initial_transforms: list[RigidTransform],
    influence_distance: float = 0.02,
    solver_name: str = "snopt",
    iteration_limit: int = 1000,
    time_limit_s: float = 30.0,
) -> tuple[list[RigidTransform], bool]:
    """Resolve penetrations between fill objects using IK projection.

    Creates a temporary Drake plant with only the fill objects and uses
    shared IK projection utility to push apart any overlapping objects.
    This prevents explosive contact forces when physics simulation starts.

    Uses the same projection logic as scene-level non-penetration projection,
    but with fix_rotation=False and fix_z=False to allow full 3D movement
    (objects will fall and rotate during physics simulation anyway).

    Args:
        fill_scene_objects: List of fill SceneObjects (must have sdf_path).
        fill_initial_transforms: Initial transforms for each fill object.
        influence_distance: Distance threshold for collision influence.
        solver_name: NLP solver name ("snopt" or "ipopt").
        iteration_limit: Maximum solver iterations.
        time_limit_s: Maximum solver time in seconds.

    Returns:
        Tuple of (projected_transforms, success_flag).
        On failure: returns (original_transforms, False).
    """
    if not fill_scene_objects:
        return fill_initial_transforms, True

    console_logger.info(
        f"Starting NLP projection for {len(fill_scene_objects)} fill objects"
    )

    try:
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

        # Build directive for fill objects as free bodies.
        directive_parts = ["directives:"]
        model_names = []
        for i, (obj, transform) in enumerate(
            zip(fill_scene_objects, fill_initial_transforms)
        ):
            if not obj.sdf_path or not obj.sdf_path.exists():
                continue

            model_name = f"fill_obj_{i}"
            model_names.append(model_name)

            effective_transform = transform @ obj.internal_model_pose
            translation = effective_transform.translation()
            angle_axis = effective_transform.rotation().ToAngleAxis()
            angle_deg = angle_axis.angle() * 180 / np.pi
            axis = angle_axis.axis()

            # Extract base link name.
            try:
                base_link_name = extract_base_link_name_from_sdf(obj.sdf_path)
            except ValueError:
                base_link_name = "base_link"

            directive_parts.append(
                f"""
- add_model:
    name: {model_name}
    file: file://{obj.sdf_path.absolute()}
    default_free_body_pose:
      {base_link_name}:
        translation: [{translation[0]}, {translation[1]}, {translation[2]}]
        rotation: !AngleAxis
          angle_deg: {angle_deg}
          axis: [{axis[0]}, {axis[1]}, {axis[2]}]"""
            )

        if not model_names:
            console_logger.warning("No valid SDF paths for fill objects")
            return fill_initial_transforms, False

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

            # Use shared IK projection utility.
            # fix_rotation=False, fix_z=False: allow full 3D movement since
            # fill objects will fall and rotate during physics simulation anyway.
            plant_context, success = solve_non_penetration_ik(
                builder=builder,
                plant=plant,
                scene_graph=scene_graph,
                influence_distance=influence_distance,
                fix_rotation=False,
                fix_z=False,
                solver_name=solver_name,
                iteration_limit=iteration_limit,
                time_limit_s=time_limit_s,
            )

            if not success or plant_context is None:
                console_logger.warning(
                    "NLP projection failed. Proceeding with original positions."
                )
                return fill_initial_transforms, False

            # Extract projected transforms.
            projected_transforms = []
            for i, model_name in enumerate(model_names):
                model_idx = plant.GetModelInstanceByName(model_name)
                body_indices = plant.GetBodyIndices(model_idx)

                if body_indices:
                    body = plant.get_body(body_indices[0])
                    effective_pose = plant.EvalBodyPoseInWorld(plant_context, body)
                    projected_transforms.append(
                        _effective_to_scene_transform(
                            effective_pose, fill_scene_objects[i].internal_model_pose
                        )
                    )
                else:
                    projected_transforms.append(fill_initial_transforms[i])

            console_logger.info(
                f"NLP projection completed successfully for {len(projected_transforms)} "
                "fill objects"
            )
            return projected_transforms, True

        finally:
            Path(directive_path).unlink(missing_ok=True)

    except Exception as e:
        console_logger.error(f"NLP projection failed with exception: {e}")
        return fill_initial_transforms, False


def simulate_fill_physics(
    container_scene_object: SceneObject,
    container_transform: RigidTransform,
    new_fill_objects: list[SceneObject],
    new_fill_transforms: list[RigidTransform],
    settled_fill_objects: list[SceneObject] | None = None,
    settled_fill_transforms: list[RigidTransform] | None = None,
    catch_floor_z: float = -5.0,
    inside_z_threshold: float = -2.0,
    simulation_time: float = 5.0,
    simulation_time_step: float = 0.001,
    output_html_path: Path | None = None,
) -> FillSimulationResult:
    """Simulate fill objects dropping into a container.

    Creates a Drake simulation with:
    - Container welded in the air at an elevated position.
    - Previously settled fill objects welded at their positions.
    - New fill objects as free bodies spawned above container.
    - Catch floor below to detect objects that fell out.

    Args:
        container_scene_object: Container SceneObject (must have sdf_path).
        container_transform: World transform for container.
        new_fill_objects: List of NEW fill SceneObjects to simulate (free bodies).
        new_fill_transforms: Initial transforms for new fill objects.
        settled_fill_objects: List of previously settled SceneObjects (welded).
        settled_fill_transforms: Transforms for previously settled fill objects.
        catch_floor_z: Z position of catch floor.
        inside_z_threshold: Z threshold for inside/outside classification.
        simulation_time: Duration to simulate.
        simulation_time_step: Simulation time step.
        output_html_path: If provided, record simulation as HTML.

    Returns:
        FillSimulationResult with inside/outside classification and final transforms
        for the NEW fill objects only.
    """
    if len(new_fill_objects) != len(new_fill_transforms):
        return FillSimulationResult(
            inside_indices=[],
            outside_indices=list(range(len(new_fill_objects))),
            final_transforms=new_fill_transforms,
            error_message="Mismatch between new fill objects and transforms count",
        )

    # Validate settled lists are consistent.
    if settled_fill_objects is None:
        settled_fill_objects = []
    if settled_fill_transforms is None:
        settled_fill_transforms = []
    if len(settled_fill_objects) != len(settled_fill_transforms):
        return FillSimulationResult(
            inside_indices=[],
            outside_indices=list(range(len(new_fill_objects))),
            final_transforms=new_fill_transforms,
            error_message="Mismatch between settled fill objects and transforms count",
        )

    # Validate container has SDF.
    if (
        not container_scene_object.sdf_path
        or not container_scene_object.sdf_path.exists()
    ):
        return FillSimulationResult(
            inside_indices=[],
            outside_indices=list(range(len(new_fill_objects))),
            final_transforms=new_fill_transforms,
            error_message="Container has no SDF path",
        )

    try:
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=simulation_time_step
        )

        # Set up visualization if recording.
        meshcat = None
        visualizer = None
        if output_html_path is not None:
            meshcat = StartMeshcat()
            console_logger.info(f"Meshcat URL: {meshcat.web_url()}")

        # Create catch floor SDF.
        catch_floor_sdf = f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="catch_floor">
    <static>true</static>
    <pose>0 0 {catch_floor_z} 0 0 0</pose>
    <link name="catch_link">
      <collision name="catch_collision">
        <geometry>
          <box><size>20 20 0.1</size></box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sdf", delete=False
        ) as catch_file:
            catch_file.write(catch_floor_sdf)
            catch_floor_path = catch_file.name

        # Build directive.
        directive_parts = ["directives:"]

        # Add catch floor.
        directive_parts.append(
            f"""
- add_model:
    name: catch_floor
    file: file://{catch_floor_path}"""
        )

        # Add container (welded).
        container_effective_transform = (
            container_transform @ container_scene_object.internal_model_pose
        )
        container_translation = container_effective_transform.translation()
        container_angle_axis = container_effective_transform.rotation().ToAngleAxis()
        container_angle_deg = container_angle_axis.angle() * 180 / np.pi
        container_axis = container_angle_axis.axis()

        try:
            container_base_link = extract_base_link_name_from_sdf(
                container_scene_object.sdf_path
            )
        except ValueError:
            container_base_link = "base_link"

        directive_parts.append(
            f"""
- add_model:
    name: container
    file: file://{container_scene_object.sdf_path.absolute()}
- add_weld:
    parent: world
    child: container::{container_base_link}
    X_PC:
      translation: [{container_translation[0]}, {container_translation[1]}, {container_translation[2]}]
      rotation: !AngleAxis
        angle_deg: {container_angle_deg}
        axis: [{container_axis[0]}, {container_axis[1]}, {container_axis[2]}]"""
        )

        # Add settled fill objects as free bodies (not welded) so new objects can
        # push them realistically. They start at their settled positions.
        settled_model_names: list[str] = []
        for i, (obj, transform) in enumerate(
            zip(settled_fill_objects, settled_fill_transforms)
        ):
            if not obj.sdf_path or not obj.sdf_path.exists():
                continue

            effective_transform = transform @ obj.internal_model_pose
            translation = effective_transform.translation()
            angle_axis = effective_transform.rotation().ToAngleAxis()
            angle_deg = angle_axis.angle() * 180 / np.pi
            axis = angle_axis.axis()

            try:
                base_link = extract_base_link_name_from_sdf(obj.sdf_path)
            except ValueError:
                base_link = "base_link"

            model_name = f"settled_fill_{i}"
            settled_model_names.append(model_name)
            directive_parts.append(
                f"""
- add_model:
    name: {model_name}
    file: file://{obj.sdf_path.absolute()}
    default_free_body_pose:
      {base_link}:
        translation: [{translation[0]}, {translation[1]}, {translation[2]}]
        rotation: !AngleAxis
          angle_deg: {angle_deg}
          axis: [{axis[0]}, {axis[1]}, {axis[2]}]"""
            )

        if settled_model_names:
            console_logger.info(
                f"Added {len(settled_model_names)} settled fill objects (free)"
            )

        # Add new fill objects as free bodies.
        free_model_names = []
        for i, (obj, transform) in enumerate(
            zip(new_fill_objects, new_fill_transforms)
        ):
            if not obj.sdf_path or not obj.sdf_path.exists():
                continue

            model_name = f"fill_obj_{i}"
            free_model_names.append((i, model_name))

            effective_transform = transform @ obj.internal_model_pose
            translation = effective_transform.translation()
            angle_axis = effective_transform.rotation().ToAngleAxis()
            angle_deg = angle_axis.angle() * 180 / np.pi
            axis = angle_axis.axis()

            try:
                base_link = extract_base_link_name_from_sdf(obj.sdf_path)
            except ValueError:
                base_link = "base_link"

            directive_parts.append(
                f"""
- add_model:
    name: {model_name}
    file: file://{obj.sdf_path.absolute()}
    default_free_body_pose:
      {base_link}:
        translation: [{translation[0]}, {translation[1]}, {translation[2]}]
        rotation: !AngleAxis
          angle_deg: {angle_deg}
          axis: [{axis[0]}, {axis[1]}, {axis[2]}]"""
            )

        directive_yaml = "\n".join(directive_parts)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as directive_file:
            directive_file.write(directive_yaml)
            directive_path = directive_file.name

        try:
            # Load directives.
            directives = LoadModelDirectives(directive_path)
            ProcessModelDirectives(directives, plant, parser=None)
            plant.Finalize()

            # Add visualizer after finalize.
            if meshcat is not None:
                visualizer = MeshcatVisualizer.AddToBuilder(
                    builder=builder, scene_graph=scene_graph, meshcat=meshcat
                )

            # Build and simulate.
            diagram = builder.Build()
            simulator = Simulator(diagram)
            context = simulator.get_mutable_context()

            if visualizer is not None:
                visualizer.StartRecording()

            simulator.AdvanceTo(simulation_time)

            if visualizer is not None and meshcat is not None:
                visualizer.StopRecording()
                visualizer.PublishRecording()
                html = meshcat.StaticHtml()
                output_html_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_html_path, "w") as f:
                    f.write(html)
                console_logger.info(f"Saved fill simulation HTML to {output_html_path}")

            # Get final positions and classify.
            plant_context = plant.GetMyContextFromRoot(context)
            final_transforms = []
            inside_indices = []
            outside_indices = []

            for i, (obj_idx, model_name) in enumerate(free_model_names):
                model_instance = plant.GetModelInstanceByName(model_name)
                body_indices = plant.GetBodyIndices(model_instance)

                if body_indices:
                    body = plant.get_body(body_indices[0])
                    final_effective_pose = plant.EvalBodyPoseInWorld(
                        plant_context, body
                    )
                    final_pose = _effective_to_scene_transform(
                        final_effective_pose,
                        new_fill_objects[obj_idx].internal_model_pose,
                    )
                    final_transforms.append(final_pose)

                    # Classify by Z position.
                    final_z = final_pose.translation()[2]
                    if final_z > inside_z_threshold:
                        inside_indices.append(obj_idx)
                        console_logger.debug(
                            f"Fill object {obj_idx} INSIDE: z={final_z:.3f}"
                        )
                    else:
                        outside_indices.append(obj_idx)
                        console_logger.debug(
                            f"Fill object {obj_idx} OUTSIDE: z={final_z:.3f}"
                        )
                else:
                    final_transforms.append(new_fill_transforms[i])
                    outside_indices.append(obj_idx)

            console_logger.info(
                f"Fill simulation: {len(inside_indices)} inside, "
                f"{len(outside_indices)} outside"
            )

            # Extract updated transforms for settled objects and check if any fell out.
            settled_final_transforms: list[RigidTransform] | None = None
            settled_fell_out_indices: list[int] | None = None
            if settled_model_names:
                settled_final_transforms = []
                settled_fell_out_indices = []
                for i, model_name in enumerate(settled_model_names):
                    model_instance = plant.GetModelInstanceByName(model_name)
                    body_indices = plant.GetBodyIndices(model_instance)
                    if body_indices:
                        body = plant.get_body(body_indices[0])
                        final_effective_pose = plant.EvalBodyPoseInWorld(
                            plant_context, body
                        )
                        final_pose = _effective_to_scene_transform(
                            final_effective_pose,
                            settled_fill_objects[i].internal_model_pose,
                        )
                        settled_final_transforms.append(final_pose)
                        # Check if this settled object was pushed out.
                        final_z = final_pose.translation()[2]
                        if final_z <= inside_z_threshold:
                            settled_fell_out_indices.append(i)
                            console_logger.warning(
                                f"Settled object {i} was pushed out: z={final_z:.3f}"
                            )

            return FillSimulationResult(
                inside_indices=inside_indices,
                outside_indices=outside_indices,
                final_transforms=final_transforms,
                settled_final_transforms=settled_final_transforms,
                settled_fell_out_indices=settled_fell_out_indices,
            )

        finally:
            Path(directive_path).unlink(missing_ok=True)
            Path(catch_floor_path).unlink(missing_ok=True)
            if meshcat is not None:
                del meshcat

    except Exception as e:
        console_logger.error(f"Fill simulation failed: {e}")
        return FillSimulationResult(
            inside_indices=[],
            outside_indices=list(range(len(new_fill_objects))),
            final_transforms=new_fill_transforms,
            error_message=str(e),
        )


def compute_bbox_corners(
    bbox_min: np.ndarray, bbox_max: np.ndarray
) -> list[np.ndarray]:
    """Compute the 8 corners of an axis-aligned bounding box."""
    return [
        np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
        np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
        np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),
        np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
        np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
        np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
        np.array([bbox_min[0], bbox_max[1], bbox_max[2]]),
        np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
    ]


def run_fill_simulation_loop(
    container_scene_obj: SceneObject,
    container_transform: RigidTransform,
    container_interior: ContainerInteriorBounds,
    fill_scene_objects: list[SceneObject],
    fill_collision_meshes: list[list[trimesh.Trimesh]],
    max_iterations: int,
    spawn_height_above_rim: float,
    height_stagger_fraction: float,
    min_height_stagger: float,
    nlp_influence_distance: float,
    nlp_solver_name: str,
    catch_floor_z: float,
    inside_z_threshold: float,
    simulation_time: float,
    simulation_time_step: float,
    max_nan_retries: int = 3,
) -> tuple[list[int], list[RigidTransform]]:
    """Run iterative fill simulation loop with retry for objects that fall out.

    Runs physics simulation iteratively, respawning objects that fall outside
    the container until all objects are settled or max iterations reached.

    Args:
        container_scene_obj: Temporary SceneObject for the container.
        container_transform: Container's world transform.
        container_interior: Interior bounds from compute_container_interior_bounds.
        fill_scene_objects: List of temporary SceneObjects for fill items.
        fill_collision_meshes: Collision meshes for each fill item.
        max_iterations: Maximum retry iterations.
        spawn_height_above_rim: Height above rim to spawn fill objects.
        height_stagger_fraction: Fraction of bbox diagonal for Z spacing.
        min_height_stagger: Minimum stagger between objects (meters).
        nlp_influence_distance: Distance threshold for NLP collision influence.
        nlp_solver_name: NLP solver name ("snopt" or "ipopt").
        catch_floor_z: Z position of catch floor.
        inside_z_threshold: Z threshold for inside detection.
        simulation_time: Simulation duration in seconds.
        simulation_time_step: Simulation time step in seconds.
        max_nan_retries: Max retries on NaN simulation errors (different seeds).

    Returns:
        Tuple of (inside_indices, final_fill_transforms) where inside_indices
        are indices of objects that ended up inside the container.
    """
    console_logger.info(
        f"Running fill simulation loop with {max_iterations} iterations"
    )
    inside_indices: list[int] = []
    final_fill_transforms: list[RigidTransform] = [RigidTransform()] * len(
        fill_scene_objects
    )
    remaining_indices = list(range(len(fill_scene_objects)))

    # Track settled objects for subsequent iterations.
    settled_objects: list[SceneObject] = []
    settled_transforms: list[RigidTransform] = []
    settled_indices: list[int] = []  # Original indices for settled objects.

    rng = np.random.default_rng()

    for iteration in range(max_iterations):
        if not remaining_indices:
            break

        console_logger.info(
            f"Fill iteration {iteration + 1}/{max_iterations}: "
            f"{len(remaining_indices)} objects to place, "
            f"{len(settled_objects)} already settled"
        )

        # Get remaining fill objects and their meshes.
        remaining_objects = [fill_scene_objects[i] for i in remaining_indices]
        remaining_meshes = [fill_collision_meshes[i] for i in remaining_indices]

        # Retry loop for NaN simulation errors (different spawn positions).
        sim_result = None
        for nan_retry in range(max_nan_retries):
            # Compute spawn transforms for remaining objects.
            spawn_transforms = compute_fill_spawn_transforms(
                fill_collision_meshes=remaining_meshes,
                container_interior=container_interior,
                container_transform=container_transform,
                spawn_height_above_rim=spawn_height_above_rim,
                height_stagger_fraction=height_stagger_fraction,
                min_height_stagger=min_height_stagger,
                rng=rng,
            )

            # Run NLP projection to resolve overlaps.
            projected_transforms, _ = project_fill_objects_non_penetrating(
                fill_scene_objects=remaining_objects,
                fill_initial_transforms=spawn_transforms,
                influence_distance=nlp_influence_distance,
                solver_name=nlp_solver_name,
            )

            # Run physics simulation with settled objects from previous iterations.
            sim_result = simulate_fill_physics(
                container_scene_object=container_scene_obj,
                container_transform=container_transform,
                new_fill_objects=remaining_objects,
                new_fill_transforms=projected_transforms,
                settled_fill_objects=settled_objects if settled_objects else None,
                settled_fill_transforms=(
                    settled_transforms if settled_transforms else None
                ),
                catch_floor_z=catch_floor_z,
                inside_z_threshold=inside_z_threshold,
                simulation_time=simulation_time,
                simulation_time_step=simulation_time_step,
            )

            # Check for NaN error and retry with different spawn positions.
            if sim_result.error_message and "nan" in sim_result.error_message.lower():
                console_logger.warning(
                    f"Simulation NaN error (attempt {nan_retry + 1}/{max_nan_retries})"
                )
                if nan_retry < max_nan_retries - 1:
                    console_logger.info("Retrying with different spawn positions...")
                    continue
            break  # Success or non-NaN error.

        if sim_result.error_message:
            console_logger.error(f"Fill simulation error: {sim_result.error_message}")
            # Continue to next iteration anyway.
            continue

        # Update settled object transforms (they may have shifted due to new objects).
        if sim_result.settled_final_transforms:
            for i, updated_transform in enumerate(sim_result.settled_final_transforms):
                settled_transforms[i] = updated_transform
                # Also update the final transforms dict.
                original_idx = settled_indices[i]
                final_fill_transforms[original_idx] = updated_transform

        # Map simulation results back to original indices.
        new_inside = []
        new_outside = []

        # Handle settled objects that were pushed out of container.
        if sim_result.settled_fell_out_indices:
            console_logger.info(
                f"{len(sim_result.settled_fell_out_indices)} settled objects "
                "were pushed out by new objects"
            )
            # Remove from inside_indices and add back to remaining for retry.
            # Process in reverse to avoid index shifting issues.
            for settled_idx in sorted(
                sim_result.settled_fell_out_indices, reverse=True
            ):
                original_idx = settled_indices[settled_idx]
                if original_idx in inside_indices:
                    inside_indices.remove(original_idx)
                # Add to new_outside for retry in next iteration.
                new_outside.append(original_idx)
                # Remove from settled tracking lists.
                del settled_objects[settled_idx]
                del settled_transforms[settled_idx]
                del settled_indices[settled_idx]

        for local_idx, final_transform in enumerate(sim_result.final_transforms):
            original_idx = remaining_indices[local_idx]
            if local_idx in sim_result.inside_indices:
                inside_indices.append(original_idx)
                final_fill_transforms[original_idx] = final_transform
                new_inside.append(original_idx)
                # Add to settled objects for next iteration.
                settled_objects.append(fill_scene_objects[original_idx])
                settled_transforms.append(final_transform)
                settled_indices.append(original_idx)
            else:
                new_outside.append(original_idx)

        # Update remaining for next iteration.
        remaining_indices = new_outside

        console_logger.info(
            f"Iteration {iteration + 1}: {len(new_inside)} inside, "
            f"{len(new_outside)} outside"
        )

    return inside_indices, final_fill_transforms


def compute_composite_bbox_in_local_frame(
    container_asset: SceneObject,
    container_transform: RigidTransform,
    fill_assets: list[SceneObject],
    final_fill_transforms: list[RigidTransform],
    inside_indices: list[int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute composite bounding box from container and fill objects in local frame.

    Args:
        container_asset: Container SceneObject with bbox.
        container_transform: Container's world transform.
        fill_assets: List of all fill asset SceneObjects.
        final_fill_transforms: Final transforms for each fill asset.
        inside_indices: Indices of fill assets that are inside the container.

    Returns:
        Tuple of (bbox_min, bbox_max) in container's local frame, or (None, None)
        if no valid bounding boxes.
    """
    all_bbox_min = np.array([np.inf, np.inf, np.inf])
    all_bbox_max = np.array([-np.inf, -np.inf, -np.inf])
    bbox_count = 0

    # Add container bbox.
    if container_asset.bbox_min is not None and container_asset.bbox_max is not None:
        bbox_count += 1
        corners = compute_bbox_corners(
            bbox_min=container_asset.bbox_min, bbox_max=container_asset.bbox_max
        )
        for corner in corners:
            world_corner = container_transform.multiply(corner)
            all_bbox_min = np.minimum(all_bbox_min, world_corner)
            all_bbox_max = np.maximum(all_bbox_max, world_corner)

    # Add fill objects bboxes.
    for idx in inside_indices:
        fill_asset = fill_assets[idx]
        fill_transform = final_fill_transforms[idx]
        if fill_asset.bbox_min is not None and fill_asset.bbox_max is not None:
            bbox_count += 1
            corners = compute_bbox_corners(
                bbox_min=fill_asset.bbox_min, bbox_max=fill_asset.bbox_max
            )
            for corner in corners:
                world_corner = fill_transform.multiply(corner)
                all_bbox_min = np.minimum(all_bbox_min, world_corner)
                all_bbox_max = np.maximum(all_bbox_max, world_corner)

    # Convert to local frame relative to container transform.
    if bbox_count == 0:
        return None, None

    inverse_transform = container_transform.inverse()
    world_corners = compute_bbox_corners(bbox_min=all_bbox_min, bbox_max=all_bbox_max)
    local_bbox_min = np.array([np.inf, np.inf, np.inf])
    local_bbox_max = np.array([-np.inf, -np.inf, -np.inf])
    for corner in world_corners:
        local_corner = inverse_transform.multiply(corner)
        local_bbox_min = np.minimum(local_bbox_min, local_corner)
        local_bbox_max = np.maximum(local_bbox_max, local_corner)

    return local_bbox_min, local_bbox_max
