"""Pile tool implementation for creating random piles of objects on surfaces.

This module provides functionality for:
- Computing pile spawn transforms with random XY, staggered Z, and random SO(3) rotation.
- Simulating pile physics using Drake.
- Creating composite pile objects that move as a unit.
"""

import logging
import math
import random
import tempfile
import time

from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

from omegaconf import DictConfig
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectives,
    MeshcatVisualizer,
    ProcessModelDirectives,
    Quaternion,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
)

from scenecode.agent_utils.asset_manager import AssetManager
from scenecode.agent_utils.physical_feasibility import _effective_to_scene_transform
from scenecode.agent_utils.room import (
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    clone_scene_object,
    SupportSurface,
    UniqueID,
    serialize_composite_member_asset,
)
from scenecode.manipuland_agents.tools.fill_container import compute_bbox_corners
from scenecode.manipuland_agents.tools.response_dataclasses import (
    ManipulandErrorType,
    PileCreationResult,
)
from scenecode.utils.sdf_utils import extract_base_link_name_from_sdf

console_logger = logging.getLogger(__name__)


def _random_rotation_matrix() -> RotationMatrix:
    """Generate a uniform random rotation matrix using Shoemake's algorithm."""
    u1 = random.random()
    u2 = random.random()
    u3 = random.random()

    # Shoemake's algorithm for uniform random quaternion.
    sqrt1u1 = math.sqrt(1 - u1)
    sqrtu1 = math.sqrt(u1)
    q = Quaternion(
        w=sqrt1u1 * math.sin(2 * math.pi * u2),
        x=sqrt1u1 * math.cos(2 * math.pi * u2),
        y=sqrtu1 * math.sin(2 * math.pi * u3),
        z=sqrtu1 * math.cos(2 * math.pi * u3),
    )
    return RotationMatrix(q)


def compute_pile_spawn_transforms(
    bounding_boxes: list[tuple[np.ndarray, np.ndarray]],
    base_transform: RigidTransform,
    surface_z: float,
    cfg: DictConfig,
) -> list[RigidTransform]:
    """Compute spawn transforms for pile objects.

    Objects are placed with:
    - Random XY within spawn radius (uniform disk distribution).
    - Staggered Z heights by bbox diagonal (no Z overlap for any rotation).
    - Random SO(3) rotation for messy appearance.

    Args:
        bounding_boxes: List of (bbox_min, bbox_max) for each object.
        base_transform: Base transform at pile center on surface.
        surface_z: Z coordinate of surface top.
        cfg: Configuration with pile_simulation settings.

    Returns:
        List of world transforms for each pile object.
    """
    if not bounding_boxes:
        return []

    # Compute bbox diagonal per object (worst-case height after any SO(3) rotation).
    diagonals = [
        float(np.linalg.norm(bbox_max - bbox_min))
        for bbox_min, bbox_max in bounding_boxes
    ]
    avg_diagonal = sum(diagonals) / len(diagonals)

    # Compute spawn radius from average diagonal.
    spawn_radius = max(cfg.min_spawn_radius, avg_diagonal * cfg.spawn_radius_scale)

    transforms = []
    current_z = surface_z + cfg.spawn_height_base

    base_xy = base_transform.translation()[:2]

    for diagonal in diagonals:

        # Random XY (uniform disk distribution).
        angle = random.uniform(0, 2 * math.pi)
        r = spawn_radius * math.sqrt(random.uniform(0, 1))
        x = float(base_xy[0]) + r * math.cos(angle)
        y = float(base_xy[1]) + r * math.sin(angle)

        # Z: center object at current layer (half diagonal above current_z).
        z = current_z + diagonal / 2

        # Random rotation (uniform SO(3)).
        rotation = _random_rotation_matrix()

        transforms.append(RigidTransform(rotation, [x, y, z]))

        # Stagger by diagonal to prevent overlap regardless of rotation.
        stagger = max(cfg.min_height_stagger, diagonal * cfg.height_stagger_fraction)
        current_z += stagger

    return transforms


def simulate_pile_physics(
    scene_objects: list[SceneObject],
    initial_transforms: list[RigidTransform],
    ground_xyz: tuple[float, float, float],
    ground_size: tuple[float, float],
    surface_z: float,
    inside_z_threshold: float,
    simulation_time: float,
    simulation_time_step: float,
    catch_floor_z: float = -5.0,
    output_html_path: Path | None = None,
) -> tuple[list[int], list[int], list[RigidTransform], str | None]:
    """Simulate pile objects dropping onto a surface.

    Creates a Drake simulation with:
    - Ground plane sized to match surface dimensions.
    - Catch floor below to detect objects that fell off.
    - All pile objects as free bodies.

    Args:
        scene_objects: List of SceneObjects to simulate.
        initial_transforms: Initial transforms for each object.
        ground_xyz: Center position (x, y, z) of ground plane.
        ground_size: Size (width, depth) of ground plane.
        surface_z: Z coordinate of surface top for inside classification.
        inside_z_threshold: Z threshold offset for inside classification.
        simulation_time: Duration to simulate.
        simulation_time_step: Simulation time step.
        catch_floor_z: Z position of catch floor.
        output_html_path: If provided, record simulation as HTML.

    Returns:
        Tuple of (inside_indices, outside_indices, final_transforms, error_message).
    """
    if len(scene_objects) != len(initial_transforms):
        return (
            [],
            list(range(len(scene_objects))),
            initial_transforms,
            "Mismatch between objects and transforms count",
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

        # Create ground plane SDF (sized to surface).
        ground_sdf = f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="ground_plane">
    <static>true</static>
    <pose>{ground_xyz[0]} {ground_xyz[1]} {ground_xyz[2]} 0 0 0</pose>
    <link name="ground_link">
      <collision name="ground_collision">
        <geometry>
          <box><size>{ground_size[0]} {ground_size[1]} 0.1</size></box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>"""

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
        ) as ground_file:
            ground_file.write(ground_sdf)
            ground_path = ground_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sdf", delete=False
        ) as catch_file:
            catch_file.write(catch_floor_sdf)
            catch_floor_path = catch_file.name

        # Build directive.
        directive_parts = ["directives:"]

        # Add ground and catch floor.
        directive_parts.append(
            f"""
- add_model:
    name: ground_plane
    file: file://{ground_path}
- add_model:
    name: catch_floor
    file: file://{catch_floor_path}"""
        )

        # Add pile objects as free bodies.
        model_names = []
        for i, (obj, transform) in enumerate(zip(scene_objects, initial_transforms)):
            if not obj.sdf_path or not obj.sdf_path.exists():
                continue

            model_name = f"pile_obj_{i}"
            model_names.append((i, model_name))

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
                console_logger.info(f"Saved pile simulation HTML to {output_html_path}")

            # Get final positions and classify.
            plant_context = plant.GetMyContextFromRoot(context)
            final_transforms = []
            inside_indices = []
            outside_indices = []

            # Threshold: objects above (surface_z + threshold) are "on surface".
            z_threshold = surface_z + inside_z_threshold

            for i, (obj_idx, model_name) in enumerate(model_names):
                model_instance = plant.GetModelInstanceByName(model_name)
                body_indices = plant.GetBodyIndices(model_instance)

                if body_indices:
                    body = plant.get_body(body_indices[0])
                    final_effective_pose = plant.EvalBodyPoseInWorld(
                        plant_context, body
                    )
                    final_pose = _effective_to_scene_transform(
                        final_effective_pose, scene_objects[obj_idx].internal_model_pose
                    )
                    final_transforms.append(final_pose)

                    # Classify by Z position.
                    final_z = final_pose.translation()[2]
                    if final_z > z_threshold:
                        inside_indices.append(obj_idx)
                        console_logger.debug(
                            f"Pile object {obj_idx} ON SURFACE: z={final_z:.3f}"
                        )
                    else:
                        outside_indices.append(obj_idx)
                        console_logger.debug(
                            f"Pile object {obj_idx} FELL OFF: z={final_z:.3f}"
                        )
                else:
                    final_transforms.append(initial_transforms[i])
                    outside_indices.append(obj_idx)

            console_logger.info(
                f"Pile simulation: {len(inside_indices)} on surface, "
                f"{len(outside_indices)} fell off"
            )

            return inside_indices, outside_indices, final_transforms, None

        finally:
            Path(directive_path).unlink(missing_ok=True)
            Path(ground_path).unlink(missing_ok=True)
            Path(catch_floor_path).unlink(missing_ok=True)
            if meshcat is not None:
                del meshcat

    except Exception as e:
        console_logger.error(f"Pile simulation failed: {e}")
        return (
            [],
            list(range(len(scene_objects))),
            initial_transforms,
            str(e),
        )


def _compute_pile_composite_bbox_in_local_frame(
    assets: list[SceneObject],
    final_transforms: list[RigidTransform],
    inside_indices: list[int],
    pile_transform: RigidTransform,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute composite bounding box for pile members in pile's local frame.

    Args:
        assets: List of SceneObjects in the pile.
        final_transforms: Final world transforms for each asset after simulation.
        inside_indices: Indices of assets that stayed on surface.
        pile_transform: The pile's reference transform.

    Returns:
        Tuple of (bbox_min, bbox_max) in pile's local frame, or (None, None)
        if no valid bounding boxes.
    """
    all_bbox_min = np.array([np.inf, np.inf, np.inf])
    all_bbox_max = np.array([-np.inf, -np.inf, -np.inf])
    bbox_count = 0

    for i in inside_indices:
        asset = assets[i]
        transform = final_transforms[i]
        if asset.bbox_min is not None and asset.bbox_max is not None:
            bbox_count += 1
            corners = compute_bbox_corners(
                bbox_min=asset.bbox_min, bbox_max=asset.bbox_max
            )
            for corner in corners:
                world_corner = transform.multiply(corner)
                all_bbox_min = np.minimum(all_bbox_min, world_corner)
                all_bbox_max = np.maximum(all_bbox_max, world_corner)

    if bbox_count == 0:
        return None, None

    # Transform world bbox back to local frame relative to pile_transform.
    inverse_transform = pile_transform.inverse()
    world_corners = compute_bbox_corners(bbox_min=all_bbox_min, bbox_max=all_bbox_max)
    local_bbox_min = np.array([np.inf, np.inf, np.inf])
    local_bbox_max = np.array([-np.inf, -np.inf, -np.inf])
    for corner in world_corners:
        local_corner = inverse_transform.multiply(corner)
        local_bbox_min = np.minimum(local_bbox_min, local_corner)
        local_bbox_max = np.maximum(local_bbox_max, local_corner)

    return local_bbox_min, local_bbox_max


def _load_bounding_box(
    sdf_path: Path, scale_factor: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Load bounding box from SDF's visual geometry.

    Args:
        sdf_path: Path to SDF file.
        scale_factor: Scale factor to apply to mesh vertices (default 1.0).

    Returns:
        Tuple of (bbox_min, bbox_max) arrays.

    Raises:
        ValueError: If geometry cannot be loaded.
    """
    # Find geometry file next to SDF.
    sdf_dir = sdf_path.parent
    geometry_path = None
    for ext in [".glb", ".obj", ".stl"]:
        candidate = sdf_dir / f"{sdf_path.stem}{ext}"
        if candidate.exists():
            geometry_path = candidate
            break

    if geometry_path is None:
        # Try common patterns.
        for pattern in ["*.glb", "*.obj", "*.stl"]:
            matches = list(sdf_dir.glob(pattern))
            if matches:
                geometry_path = matches[0]
                break

    if geometry_path is None:
        raise ValueError(f"No geometry file found for {sdf_path}")

    mesh = trimesh.load(geometry_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        if hasattr(mesh, "dump"):
            meshes = list(mesh.dump())
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
            else:
                raise ValueError(f"No valid mesh in {geometry_path}")
        else:
            raise ValueError(f"Cannot load mesh from {geometry_path}")

    # Apply scale factor to mesh vertices.
    if scale_factor != 1.0:
        mesh.vertices *= scale_factor

    return mesh.bounds[0], mesh.bounds[1]


def create_pile_tool_impl(
    asset_ids: list[str],
    surface_id: str,
    position_x: float,
    position_z: float,
    scene: RoomScene,
    cfg: DictConfig,
    asset_manager: AssetManager,
    support_surfaces: dict[str, SupportSurface],
    generate_unique_id: Callable[[str], UniqueID],
) -> str:
    """Create a pile of objects on a support surface.

    This is the core implementation logic for pile creation, taking all
    dependencies as explicit parameters.

    Args:
        asset_ids: List of asset IDs to pile (at least 2 objects).
        surface_id: ID of the support surface to place the pile on.
        position_x: X coordinate on the surface (in surface's local frame).
        position_z: Z coordinate on the surface (in surface's local frame).
        scene: The RoomScene to add the pile to.
        cfg: Configuration with pile_simulation settings.
        asset_manager: AssetManager for retrieving asset objects.
        support_surfaces: Dictionary of available support surfaces.
        generate_unique_id: Function to generate unique IDs for new objects.

    Returns:
        JSON string with PileCreationResult.
    """
    console_logger.info(
        f"Tool called: create_pile(asset_ids={asset_ids}, surface_id={surface_id}, "
        f"position_x={position_x}, position_z={position_z})"
    )
    start_time = time.time()

    # Validate at least 2 assets for a pile.
    if len(asset_ids) < 2:
        console_logger.warning(
            f"Pile creation failed: requires at least 2 assets, got {len(asset_ids)}"
        )
        return PileCreationResult(
            success=False,
            message=(
                "Pile requires at least 2 assets. "
                "Use place_manipuland_on_surface for single items."
            ),
            pile_object_id=None,
            parent_surface_id=surface_id,
            num_items=len(asset_ids),
            pile_count=0,
            removed_count=0,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # Validate surface exists.
    if surface_id not in support_surfaces:
        available_ids = list(support_surfaces.keys())
        console_logger.warning(
            f"Pile creation failed: invalid surface_id '{surface_id}'"
        )
        return PileCreationResult(
            success=False,
            message=(
                f"Invalid surface_id: {surface_id}. "
                f"Available surfaces: {available_ids}"
            ),
            pile_object_id=None,
            parent_surface_id=surface_id,
            num_items=len(asset_ids),
            pile_count=0,
            removed_count=0,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_SURFACE,
        ).to_json()

    target_surface = support_surfaces[surface_id]

    # Validate all assets exist and build asset list.
    assets: list[SceneObject] = []
    for asset_id in asset_ids:
        try:
            unique_id = UniqueID(asset_id)
        except Exception:
            console_logger.warning(
                f"Pile creation failed: invalid asset ID format '{asset_id}'"
            )
            return PileCreationResult(
                success=False,
                message=f"Invalid asset ID format: {asset_id}",
                pile_object_id=None,
                parent_surface_id=surface_id,
                num_items=len(asset_ids),
                pile_count=0,
                removed_count=0,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.ASSET_NOT_FOUND,
            ).to_json()

        asset = asset_manager.get_asset_by_id(unique_id)
        if not asset:
            available = asset_manager.list_available_assets()
            manipuland_ids = [
                str(a.object_id)
                for a in available
                if a.object_type == ObjectType.MANIPULAND
            ]
            console_logger.warning(
                f"Pile creation failed: asset '{asset_id}' not found"
            )
            return PileCreationResult(
                success=False,
                message=(
                    f"Asset {asset_id} not found. "
                    f"Available manipulands: {manipuland_ids}"
                ),
                pile_object_id=None,
                parent_surface_id=surface_id,
                num_items=len(asset_ids),
                pile_count=0,
                removed_count=0,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.ASSET_NOT_FOUND,
            ).to_json()

        # Validate SDF path exists for physics simulation.
        if not asset.sdf_path or not asset.sdf_path.exists():
            console_logger.warning(
                f"Pile creation failed: asset '{asset_id}' has no SDF file"
            )
            return PileCreationResult(
                success=False,
                message=(
                    f"Asset {asset_id} has no SDF file. "
                    "Cannot simulate pile physics."
                ),
                pile_object_id=None,
                parent_surface_id=surface_id,
                num_items=len(asset_ids),
                pile_count=0,
                removed_count=0,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.INVALID_OPERATION,
            ).to_json()

        assets.append(asset)

    # Load bounding boxes for each asset.
    bounding_boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for asset in assets:
        if asset.bbox_min is not None and asset.bbox_max is not None:
            bounding_boxes.append((asset.bbox_min.copy(), asset.bbox_max.copy()))
        else:
            # Try to load from geometry.
            try:
                bbox = _load_bounding_box(
                    asset.sdf_path, scale_factor=asset.scale_factor
                )
                bounding_boxes.append(bbox)
            except ValueError as e:
                console_logger.warning(
                    f"Pile creation failed: could not load bounding box for "
                    f"'{asset.name}': {e}"
                )
                return PileCreationResult(
                    success=False,
                    message=f"Failed to load bounding box for {asset.name}: {e}",
                    pile_object_id=None,
                    parent_surface_id=surface_id,
                    num_items=len(asset_ids),
                    pile_count=0,
                    removed_count=0,
                    inside_assets=[],
                    removed_assets=[],
                    error_type=ManipulandErrorType.INVALID_OPERATION,
                ).to_json()

    # Compute base transform on surface.
    position_2d = np.array([position_x, position_z])
    base_transform = target_surface.to_world_pose(
        position_2d=position_2d, rotation_2d=0.0
    )

    # Get surface dimensions for ground plane sizing.
    surface_min = target_surface.bounding_box_min[:2]
    surface_max = target_surface.bounding_box_max[:2]
    surface_width = float(surface_max[0] - surface_min[0])
    surface_depth = float(surface_max[1] - surface_min[1])

    # Get pile simulation config.
    pile_cfg = cfg.pile_simulation

    # Compute spawn transforms.
    surface_z = float(target_surface.transform.translation()[2])
    initial_transforms = compute_pile_spawn_transforms(
        bounding_boxes=bounding_boxes,
        base_transform=base_transform,
        surface_z=surface_z,
        cfg=pile_cfg,
    )

    # Create temporary SceneObjects for simulation.
    temp_scene_objects = []
    for i, asset in enumerate(assets):
        temp_obj = clone_scene_object(
            asset,
            object_id=UniqueID(f"pile_temp_{i}"),
            object_type=ObjectType.MANIPULAND,
            transform=initial_transforms[i],
            placement_info=None,
        )
        temp_scene_objects.append(temp_obj)

    # Ground plane position: centered at pile spawn position on surface.
    ground_xyz = (
        float(base_transform.translation()[0]),
        float(base_transform.translation()[1]),
        surface_z,
    )

    # Run physics simulation.
    sim_start_time = time.time()
    inside_indices, outside_indices, final_transforms, error_msg = (
        simulate_pile_physics(
            scene_objects=temp_scene_objects,
            initial_transforms=initial_transforms,
            ground_xyz=ground_xyz,
            ground_size=(surface_width, surface_depth),
            surface_z=surface_z,
            inside_z_threshold=pile_cfg.inside_z_threshold,
            simulation_time=pile_cfg.simulation_time,
            simulation_time_step=pile_cfg.simulation_time_step,
        )
    )
    sim_elapsed = time.time() - sim_start_time
    console_logger.info(f"Pile simulation completed in {sim_elapsed:.2f}s")

    if error_msg:
        console_logger.error(f"Pile creation failed: simulation error: {error_msg}")
        return PileCreationResult(
            success=False,
            message=f"Simulation failed: {error_msg}",
            pile_object_id=None,
            parent_surface_id=surface_id,
            num_items=len(asset_ids),
            pile_count=0,
            removed_count=len(asset_ids),
            inside_assets=[],
            removed_assets=[a.name for a in assets],
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # Check if we have at least 2 objects on surface.
    pile_count = len(inside_indices)
    removed_count = len(outside_indices)

    inside_asset_names = [assets[i].name for i in inside_indices]
    removed_asset_names = [assets[i].name for i in outside_indices]

    if pile_count < 2:
        console_logger.warning(
            f"Pile creation failed: only {pile_count} objects stayed on surface "
            f"(need at least 2)"
        )
        return PileCreationResult(
            success=False,
            message=(
                f"Only {pile_count} object(s) stayed on surface (need at least 2). "
                f"{removed_count} fell off. Try placing pile further from edges "
                f"or using fewer objects."
            ),
            pile_object_id=None,
            parent_surface_id=surface_id,
            num_items=len(asset_ids),
            pile_count=pile_count,
            removed_count=removed_count,
            inside_assets=inside_asset_names,
            removed_assets=removed_asset_names,
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # Pile is valid - create composite SceneObject.
    pile_id = generate_unique_id("pile")

    # Use first inside object's final transform as the pile's reference transform.
    first_inside_idx = inside_indices[0]
    pile_transform = final_transforms[first_inside_idx]

    # Build member_assets metadata with sdf_path for each member that stayed.
    member_assets = []
    for i in inside_indices:
        asset = assets[i]
        final_transform = final_transforms[i]
        member_assets.append(
            serialize_composite_member_asset(asset, final_transform)
        )

    # Compute composite bounding box.
    all_bbox_min, all_bbox_max = _compute_pile_composite_bbox_in_local_frame(
        assets=assets,
        final_transforms=final_transforms,
        inside_indices=inside_indices,
        pile_transform=pile_transform,
    )

    composite_object = SceneObject(
        object_id=pile_id,
        object_type=ObjectType.MANIPULAND,
        name=f"pile_{pile_count}",
        description=f"Pile of {pile_count} objects: " + ", ".join(inside_asset_names),
        transform=pile_transform,
        geometry_path=None,
        sdf_path=None,
        metadata={
            "composite_type": "pile",
            "member_assets": member_assets,
            "num_members": pile_count,
        },
        bbox_min=all_bbox_min,
        bbox_max=all_bbox_max,
        placement_info=PlacementInfo(
            parent_surface_id=target_surface.surface_id,
            position_2d=position_2d.copy(),
            rotation_2d=0.0,
            placement_method="pile_placement",
        ),
    )

    scene.add_object(composite_object)

    elapsed_time = time.time() - start_time

    # Build result message.
    if removed_count > 0:
        message = (
            f"Created pile '{pile_id}' with {pile_count} objects. "
            f"{removed_count} object(s) fell off: {removed_asset_names}"
        )
    else:
        message = f"Created pile '{pile_id}' with {pile_count} objects"

    console_logger.info(
        f"Successfully created pile {pile_id} with {pile_count} items "
        f"on surface {surface_id} in {elapsed_time:.2f}s"
    )

    return PileCreationResult(
        success=True,
        message=message,
        pile_object_id=str(pile_id),
        parent_surface_id=surface_id,
        num_items=len(asset_ids),
        pile_count=pile_count,
        removed_count=removed_count,
        inside_assets=inside_asset_names,
        removed_assets=removed_asset_names,
    ).to_json()
