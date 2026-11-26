"""Fill container tool implementation for filling containers with objects."""

import logging
import math
import time

from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

from omegaconf import DictConfig
from pydrake.all import RigidTransform

from scenecode.agent_utils.asset_manager import AssetManager
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
from scenecode.manipuland_agents.tools.fill_container import (
    compute_composite_bbox_in_local_frame,
    compute_container_interior_bounds,
    run_fill_simulation_loop,
)
from scenecode.manipuland_agents.tools.response_dataclasses import (
    FillContainerResult,
    ManipulandErrorType,
)
from scenecode.utils.mesh_loading import load_collision_meshes_from_sdf

console_logger = logging.getLogger(__name__)


def fill_container_tool_impl(
    container_asset_id: str,
    fill_asset_ids: list[str],
    surface_id: str,
    position_x: float,
    position_z: float,
    rotation_degrees: float,
    scene: RoomScene,
    cfg: DictConfig,
    asset_manager: AssetManager,
    support_surfaces: dict[str, SupportSurface],
    generate_unique_id: Callable[[str], UniqueID],
    top_surface_overlap_tolerance: float,
    is_top_surface_fn: Callable[[str], bool],
    validate_footprint_fn: Callable[
        [SupportSurface, Path | None, np.ndarray, float, float, float],
        tuple[bool, str | None],
    ],
) -> str:
    """Fill a container with objects using physics simulation.

    This is the core implementation logic for fill container creation, taking all
    dependencies as explicit parameters.

    Args:
        container_asset_id: ID of the container asset.
        fill_asset_ids: List of asset IDs to fill the container with.
        surface_id: ID of the support surface to place the filled container on.
        position_x: X coordinate on the surface (in surface's local frame).
        position_z: Z coordinate on the surface (in surface's local frame).
        rotation_degrees: Rotation of the container on the surface (degrees).
        scene: The RoomScene to add the filled container to.
        cfg: Configuration with fill_simulation settings.
        asset_manager: AssetManager for retrieving asset objects.
        support_surfaces: Dictionary of available support surfaces.
        generate_unique_id: Function to generate unique IDs for new objects.
        top_surface_overlap_tolerance: Tolerance for top surface boundary overlap.
        is_top_surface_fn: Function to check if a surface is a top surface.
        validate_footprint_fn: Function to validate footprint within surface boundary.

    Returns:
        JSON string with FillContainerResult.
    """
    console_logger.info(
        f"Tool called: fill_container(container={container_asset_id}, "
        f"fill_ids={fill_asset_ids}, surface={surface_id})"
    )
    start_time = time.time()

    # Early validation of placeholder/invalid IDs.
    # Agent sometimes passes these when confused about when to use fill_container.
    invalid_placeholders = {"__none__", "", "dummy", "none", "null"}

    if (
        container_asset_id.lower() in invalid_placeholders
        or container_asset_id.startswith("__")
    ):
        console_logger.warning(
            f"Fill container called with invalid container_asset_id: "
            f"'{container_asset_id}'"
        )
        return FillContainerResult(
            success=False,
            message=(
                f"Invalid container_asset_id '{container_asset_id}'. "
                "fill_container is only for placing items INSIDE containers "
                "(bowls, baskets, vases, bins). You must first generate a container "
                "with generate_manipuland(), then pass its ID here. "
                "For flat items on surfaces, use place_manipuland_on_surface. "
                "For messy random arrangements, use create_pile."
            ),
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_asset_ids),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # Validate fill_asset_ids don't contain placeholders.
    invalid_fills = [
        fid
        for fid in fill_asset_ids
        if fid.lower() in invalid_placeholders or fid.startswith("__")
    ]
    if invalid_fills:
        console_logger.warning(
            f"Fill container called with invalid fill_asset_ids: {invalid_fills}"
        )
        return FillContainerResult(
            success=False,
            message=(
                f"Invalid fill_asset_ids: {invalid_fills}. "
                "You must first generate fill items with generate_manipuland(), "
                "then pass their IDs here."
            ),
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_asset_ids),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # Validate fill_asset_ids is not empty.
    if not fill_asset_ids:
        console_logger.warning("Fill container failed: fill_asset_ids is empty")
        return FillContainerResult(
            success=False,
            message=(
                "fill_asset_ids cannot be empty. "
                "Use place_manipuland_on_surface for empty containers."
            ),
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=0,
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # Validate surface exists.
    if surface_id not in support_surfaces:
        available_ids = list(support_surfaces.keys())
        console_logger.warning(
            f"Fill container failed: invalid surface_id '{surface_id}'"
        )
        return FillContainerResult(
            success=False,
            message=(
                f"Invalid surface_id: {surface_id}. "
                f"Available surfaces: {available_ids}"
            ),
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_asset_ids),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_SURFACE,
        ).to_json()

    target_surface = support_surfaces[surface_id]

    # Get container asset.
    try:
        container_unique_id = UniqueID(container_asset_id)
    except Exception:
        console_logger.warning(
            f"Fill container failed: invalid container asset ID format "
            f"'{container_asset_id}'"
        )
        return FillContainerResult(
            success=False,
            message=f"Invalid container asset ID format: {container_asset_id}",
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_asset_ids),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.ASSET_NOT_FOUND,
        ).to_json()

    container_asset = asset_manager.get_asset_by_id(container_unique_id)
    if not container_asset:
        console_logger.warning(
            f"Fill container failed: container asset '{container_asset_id}' not found"
        )
        return FillContainerResult(
            success=False,
            message=f"Container asset '{container_asset_id}' not found",
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_asset_ids),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.ASSET_NOT_FOUND,
        ).to_json()

    # Validate container has SDF path.
    if not container_asset.sdf_path or not container_asset.sdf_path.exists():
        console_logger.warning(
            f"Fill container failed: container asset '{container_asset_id}' has no "
            f"SDF file"
        )
        return FillContainerResult(
            success=False,
            message=(
                f"Container asset '{container_asset_id}' has no SDF file. "
                "Cannot simulate fill physics."
            ),
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_asset_ids),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # Get all fill assets.
    fill_assets: list[SceneObject] = []
    fill_names: list[str] = []
    for asset_id in fill_asset_ids:
        try:
            fill_unique_id = UniqueID(asset_id)
        except Exception:
            console_logger.warning(
                f"Fill container failed: invalid fill asset ID format '{asset_id}'"
            )
            return FillContainerResult(
                success=False,
                message=f"Invalid fill asset ID format: {asset_id}",
                filled_container_id=None,
                container_asset_id=container_asset_id,
                fill_count=0,
                total_fill_attempted=len(fill_asset_ids),
                removed_count=0,
                parent_surface_id=surface_id,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.ASSET_NOT_FOUND,
            ).to_json()

        fill_asset = asset_manager.get_asset_by_id(fill_unique_id)
        if not fill_asset:
            console_logger.warning(
                f"Fill container failed: fill asset '{asset_id}' not found"
            )
            return FillContainerResult(
                success=False,
                message=f"Fill asset '{asset_id}' not found",
                filled_container_id=None,
                container_asset_id=container_asset_id,
                fill_count=0,
                total_fill_attempted=len(fill_asset_ids),
                removed_count=0,
                parent_surface_id=surface_id,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.ASSET_NOT_FOUND,
            ).to_json()

        if not fill_asset.sdf_path or not fill_asset.sdf_path.exists():
            console_logger.warning(
                f"Fill container failed: fill asset '{asset_id}' has no SDF file"
            )
            return FillContainerResult(
                success=False,
                message=(
                    f"Fill asset '{asset_id}' has no SDF file. "
                    "Cannot simulate fill physics."
                ),
                filled_container_id=None,
                container_asset_id=container_asset_id,
                fill_count=0,
                total_fill_attempted=len(fill_asset_ids),
                removed_count=0,
                parent_surface_id=surface_id,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.INVALID_OPERATION,
            ).to_json()

        fill_assets.append(fill_asset)
        fill_names.append(fill_asset.name)

    # Validate position is within surface bounds (convex hull).
    position_2d = np.array([position_x, position_z])
    try:
        if not target_surface.contains_point_2d(position_2d):
            console_logger.warning(
                f"Fill container failed: position ({position_x:.3f}, {position_z:.3f}) "
                f"outside surface {surface_id}"
            )
            return FillContainerResult(
                success=False,
                message=(
                    f"Position ({position_x:.3f}, {position_z:.3f}) is outside "
                    f"the convex hull of surface {surface_id}. "
                    f"Use list_support_surfaces() to see available surfaces."
                ),
                filled_container_id=None,
                container_asset_id=container_asset_id,
                fill_count=0,
                total_fill_attempted=len(fill_asset_ids),
                removed_count=0,
                parent_surface_id=surface_id,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
            ).to_json()
    except ValueError as e:
        console_logger.error(f"Surface {surface_id} has no mesh: {e}")
        return FillContainerResult(
            success=False,
            message=(
                f"Surface {surface_id} has no mesh geometry for "
                f"placement validation."
            ),
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_asset_ids),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_SURFACE,
        ).to_json()

    # Validate container footprint fits within surface boundary.
    overlap_ratio = (
        top_surface_overlap_tolerance if is_top_surface_fn(surface_id) else 0.0
    )
    is_valid, error_msg = validate_footprint_fn(
        target_surface=target_surface,
        geometry_path=container_asset.geometry_path,
        position_2d=position_2d,
        rotation_degrees=rotation_degrees,
        allow_overlap_ratio=overlap_ratio,
        scale_factor=container_asset.scale_factor,
    )
    if not is_valid:
        console_logger.warning(
            f"Fill container failed: footprint validation failed - {error_msg}"
        )
        return FillContainerResult(
            success=False,
            message=error_msg,
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_asset_ids),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
        ).to_json()

    # Validate container height fits within surface clearance.
    if container_asset.bbox_min is not None and container_asset.bbox_max is not None:
        container_height = float(
            container_asset.bbox_max[2] - container_asset.bbox_min[2]
        )
        surface_clearance = float(
            target_surface.bounding_box_max[2] - target_surface.bounding_box_min[2]
        )

        console_logger.info(
            f"Clearance check: container_height={container_height:.3f}m, "
            f"surface_clearance={surface_clearance:.3f}m"
        )

        if container_height > surface_clearance:
            console_logger.warning(
                f"Fill container failed: container height {container_height:.3f}m "
                f"exceeds clearance {surface_clearance:.3f}m"
            )
            return FillContainerResult(
                success=False,
                message=(
                    f"Container height {container_height:.3f}m exceeds surface "
                    f"clearance {surface_clearance:.3f}m. Make sure you are "
                    f"placing on the correct surface that you planned to use. "
                    f"If this is the intended surface, choose a shorter container "
                    f"or find a surface with more clearance."
                ),
                filled_container_id=None,
                container_asset_id=container_asset_id,
                fill_count=0,
                total_fill_attempted=len(fill_asset_ids),
                removed_count=0,
                parent_surface_id=surface_id,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
            ).to_json()

    # Load container collision geometry.
    try:
        container_meshes = load_collision_meshes_from_sdf(container_asset.sdf_path)
        if not container_meshes:
            console_logger.warning(
                "Fill container failed: no collision geometry found in container"
            )
            return FillContainerResult(
                success=False,
                message="No collision geometry found in container",
                filled_container_id=None,
                container_asset_id=container_asset_id,
                fill_count=0,
                total_fill_attempted=len(fill_asset_ids),
                removed_count=0,
                parent_surface_id=surface_id,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.INVALID_OPERATION,
            ).to_json()

        # Apply container's scale_factor to collision meshes.
        if container_asset.scale_factor != 1.0:
            for mesh in container_meshes:
                mesh.vertices *= container_asset.scale_factor

        container_interior = compute_container_interior_bounds(
            collision_meshes=container_meshes,
            top_rim_height_fraction=cfg.fill_simulation.top_rim_height_fraction,
            interior_scale=cfg.fill_simulation.interior_scale,
        )
    except Exception as e:
        console_logger.warning(
            f"Fill container failed: could not compute container interior: {e}"
        )
        return FillContainerResult(
            success=False,
            message=f"Failed to compute container interior: {e}",
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_asset_ids),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # Load fill collision meshes.
    fill_collision_meshes: list[list[trimesh.Trimesh]] = []
    for fill_asset in fill_assets:
        try:
            fill_meshes = load_collision_meshes_from_sdf(fill_asset.sdf_path)
            if fill_meshes:
                # Apply fill item's scale_factor to collision meshes.
                if fill_asset.scale_factor != 1.0:
                    for mesh in fill_meshes:
                        mesh.vertices *= fill_asset.scale_factor
                fill_collision_meshes.append(fill_meshes)
            else:
                fill_collision_meshes.append([])
        except Exception:
            fill_collision_meshes.append([])

    # Count number of empty fill meshes.
    empty_fill_meshes = sum(len(meshes) == 0 for meshes in fill_collision_meshes)
    if empty_fill_meshes > 0:
        console_logger.warning(
            f"Fill container failed: {empty_fill_meshes} fill assets have no collision "
            "meshes"
        )

    # Compute container transform on surface (position_2d already validated above).
    rotation_radians = math.radians(rotation_degrees)
    container_transform = target_surface.to_world_pose(
        position_2d=position_2d, rotation_2d=rotation_radians
    )

    # Create temporary SceneObject for container.
    container_scene_obj = clone_scene_object(
        container_asset,
        object_id=UniqueID("container_temp"),
        object_type=ObjectType.MANIPULAND,
        transform=container_transform,
        placement_info=None,
    )

    # Create temporary SceneObjects for fill items.
    fill_scene_objects = []
    for i, fill_asset in enumerate(fill_assets):
        fill_obj = clone_scene_object(
            fill_asset,
            object_id=UniqueID(f"fill_temp_{i}"),
            object_type=ObjectType.MANIPULAND,
            transform=RigidTransform(),
            placement_info=None,
        )
        fill_scene_objects.append(fill_obj)

    # Get fill simulation config.
    fill_cfg = cfg.fill_simulation

    # Run iterative fill simulation.
    inside_indices, final_fill_transforms = run_fill_simulation_loop(
        container_scene_obj=container_scene_obj,
        container_transform=container_transform,
        container_interior=container_interior,
        fill_scene_objects=fill_scene_objects,
        fill_collision_meshes=fill_collision_meshes,
        max_iterations=fill_cfg.max_iterations,
        spawn_height_above_rim=fill_cfg.spawn_height_above_rim,
        height_stagger_fraction=fill_cfg.height_stagger_fraction,
        min_height_stagger=fill_cfg.min_height_stagger,
        nlp_influence_distance=fill_cfg.nlp_influence_distance,
        nlp_solver_name=fill_cfg.nlp_solver_name,
        catch_floor_z=fill_cfg.catch_floor_z,
        inside_z_threshold=fill_cfg.inside_z_threshold,
        simulation_time=fill_cfg.simulation_time,
        simulation_time_step=fill_cfg.simulation_time_step,
        max_nan_retries=fill_cfg.max_nan_retries,
    )

    # Check if we have any objects inside.
    if not inside_indices:
        removed_names = [fill_names[i] for i in range(len(fill_assets))]
        console_logger.warning(
            f"Fill container failed: no fill objects remained inside after "
            f"{fill_cfg.max_iterations} iterations"
        )
        return FillContainerResult(
            success=False,
            message=(
                f"No fill objects remained inside container after "
                f"{fill_cfg.max_iterations} iterations. Removed: {removed_names}"
            ),
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_asset_ids),
            removed_count=len(fill_assets),
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=removed_names,
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # Create filled container composite object.
    filled_container_id = generate_unique_id("filled_container")

    # Build container_asset metadata.
    container_asset_data = serialize_composite_member_asset(
        container_asset, container_transform
    )

    # Build fill_assets metadata for objects that stayed inside.
    fill_assets_data = []
    inside_names = []
    for idx in inside_indices:
        fill_asset = fill_assets[idx]
        fill_transform = final_fill_transforms[idx]
        fill_assets_data.append(
            serialize_composite_member_asset(fill_asset, fill_transform)
        )
        inside_names.append(fill_asset.name)

    removed_indices = [i for i in range(len(fill_assets)) if i not in inside_indices]
    removed_names = [fill_names[i] for i in removed_indices]

    # Compute composite bounding box from container and fill objects.
    all_bbox_min, all_bbox_max = compute_composite_bbox_in_local_frame(
        container_asset=container_asset,
        container_transform=container_transform,
        fill_assets=fill_assets,
        final_fill_transforms=final_fill_transforms,
        inside_indices=inside_indices,
    )

    # Create composite SceneObject.
    composite_object = SceneObject(
        object_id=filled_container_id,
        object_type=ObjectType.MANIPULAND,
        name=f"filled_{container_asset.name}",
        description=(f"{container_asset.name} filled with " + ", ".join(inside_names)),
        transform=container_transform,
        geometry_path=None,
        sdf_path=None,
        metadata={
            "composite_type": "filled_container",
            "container_asset": container_asset_data,
            "fill_assets": fill_assets_data,
            "num_fill_objects": len(inside_indices),
        },
        bbox_min=all_bbox_min,
        bbox_max=all_bbox_max,
        placement_info=PlacementInfo(
            parent_surface_id=target_surface.surface_id,
            position_2d=position_2d.copy(),
            rotation_2d=rotation_radians,
            placement_method="fill_container",
        ),
    )

    scene.add_object(composite_object)

    elapsed_time = time.time() - start_time
    console_logger.info(
        f"Successfully created filled container {filled_container_id} with "
        f"{len(inside_indices)} items on surface {surface_id} in {elapsed_time:.2f}s"
    )

    return FillContainerResult(
        success=True,
        message=(
            f"Created '{filled_container_id}' with {len(inside_indices)} of "
            f"{len(fill_asset_ids)} objects inside"
        ),
        filled_container_id=str(filled_container_id),
        container_asset_id=container_asset_id,
        fill_count=len(inside_indices),
        total_fill_attempted=len(fill_asset_ids),
        removed_count=len(removed_indices),
        parent_surface_id=surface_id,
        inside_assets=inside_names,
        removed_assets=removed_names,
    ).to_json()
