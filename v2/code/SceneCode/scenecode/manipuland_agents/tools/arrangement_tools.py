"""Controlled arrangement tool for placing items at specific positions on flat containers.

This module provides the create_arrangement tool which places objects at user-specified
SE(2) poses (x, y, rotation) ON TOP of flat containers like trays, platters, and boards.

Unlike fill_container (which drops items INTO cavities at random positions and retries),
create_arrangement uses exact positions with all-or-nothing semantics.
"""

import logging
import math
import time

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from scenecode.manipuland_agents.tools.manipuland_tools import FillAssetItem

import numpy as np

from omegaconf import DictConfig
from pydrake.math import RigidTransform, RotationMatrix

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
    FillSimulationResult,
    compute_composite_bbox_in_local_frame,
    simulate_fill_physics,
)
from scenecode.manipuland_agents.tools.response_dataclasses import (
    FillContainerResult,
    ManipulandErrorType,
)
from scenecode.utils.collision_utils import compute_pairwise_collisions
from scenecode.utils.shape_analysis import is_circular_object

console_logger = logging.getLogger(__name__)


def _get_container_bounds_info(container_asset: SceneObject, cfg: DictConfig) -> dict:
    """Compute shape-aware container bounds (circular or rectangular).

    Uses is_circular_object from shape_analysis.py for robust mesh-based detection.
    Returns dict with shape info for validation and scene state display.

    Args:
        container_asset: Container SceneObject with bbox.
        cfg: Configuration with snap_to_object settings for circular detection.

    Returns:
        Dict with shape info: {"shape": "circular", "radius": ...} or
        {"shape": "rectangular", "x": [...], "y": [...]}.

    Raises:
        ValueError: If container has no bounding box (bug in asset generation).
    """
    if container_asset.bbox_min is None or container_asset.bbox_max is None:
        raise ValueError(
            f"Container '{container_asset.name}' has no bounding box. "
            f"This indicates a bug in asset generation."
        )

    width = float(container_asset.bbox_max[0] - container_asset.bbox_min[0])
    depth = float(container_asset.bbox_max[1] - container_asset.bbox_min[1])

    # Use existing mesh-based circular detection (volume ratio, cached).
    is_circular = is_circular_object(obj=container_asset, cfg=cfg)

    if is_circular:
        radius = min(width, depth) / 2
        return {"shape": "circular", "radius": radius}
    else:
        return {
            "shape": "rectangular",
            "x": [-width / 2, width / 2],
            "y": [-depth / 2, depth / 2],
        }


def _validate_item_within_container_bounds(
    item_asset: SceneObject, x_offset: float, y_offset: float, bounds_info: dict
) -> tuple[bool, str | None]:
    """Check if item CENTER is within container bounds.

    Only validates that the specified position is inside the container.
    Does not account for item size - physics simulation will catch items
    that overhang and fall off. This allows elongated items (forks, knives)
    to be placed near edges when oriented parallel to the edge.

    Args:
        item_asset: The item SceneObject (used for error messages).
        x_offset: X offset from container center in meters.
        y_offset: Y offset from container center in meters.
        bounds_info: Container bounds from _get_container_bounds_info().

    Returns:
        Tuple of (True, None) if valid, (False, error_message) if not.
    """
    if bounds_info["shape"] == "circular":
        radius = bounds_info["radius"]
        distance = math.sqrt(x_offset**2 + y_offset**2)

        if distance > radius:
            return False, (
                f"Item '{item_asset.name}' position ({x_offset:.2f}, {y_offset:.2f}) "
                f"is {distance:.2f}m from center, outside container radius {radius:.2f}m."
            )
    else:  # rectangular
        half_x = bounds_info["x"][1]
        half_y = bounds_info["y"][1]

        if abs(x_offset) > half_x or abs(y_offset) > half_y:
            return False, (
                f"Item '{item_asset.name}' position ({x_offset:.2f}, {y_offset:.2f}) "
                f"is outside container bounds: x in [{-half_x:.2f}, {half_x:.2f}], "
                f"y in [{-half_y:.2f}, {half_y:.2f}]."
            )

    return True, None


def create_arrangement_impl(
    container_asset_id: str,
    fill_assets: "list[FillAssetItem]",
    surface_id: str,
    position_x: float,
    position_z: float,
    rotation_degrees: float,
    scene: RoomScene,
    cfg: DictConfig,
    asset_manager: AssetManager,
    support_surfaces: dict[str, SupportSurface],
    generate_unique_id: Callable[[str], UniqueID],
    validate_footprint_fn: Callable[
        [SupportSurface, Path | None, np.ndarray, float, float, float],
        tuple[bool, str | None],
    ],
    top_surface_overlap_tolerance: float,
    is_top_surface_fn: Callable[[str], bool],
) -> str:
    """Create a controlled arrangement of items on a flat container.

    Like fill_container but with user-specified XY positions instead of random.
    Uses physics simulation for Z settling and stability check.

    Args:
        container_asset_id: ID of the container asset (tray, platter, board).
        fill_assets: List of dicts with keys: id (required), x, y, rotation (optional).
        surface_id: ID of the support surface to place arrangement on.
        position_x: X position of container on surface (meters).
        position_z: Z position of container on surface (meters).
        rotation_degrees: Rotation of entire arrangement on surface (degrees).
        scene: The RoomScene to add the arrangement to.
        cfg: Configuration with fill_simulation and arrangement settings.
        asset_manager: AssetManager for retrieving asset objects.
        support_surfaces: Dictionary of available support surfaces.
        generate_unique_id: Function to generate unique IDs for new objects.
        validate_footprint_fn: Function to validate footprint within surface boundary.
        top_surface_overlap_tolerance: Tolerance for top surface boundary overlap.
        is_top_surface_fn: Function to check if a surface is a top surface.

    Returns:
        JSON string with FillContainerResult.
    """
    console_logger.info(
        f"Tool called: create_arrangement(container={container_asset_id}, "
        f"fills={len(fill_assets)}, surface={surface_id})"
    )
    start_time = time.time()

    # 1. Validate fill_assets is not empty and has required keys.
    if not fill_assets:
        console_logger.warning("Create arrangement failed: fill_assets is empty")
        return FillContainerResult(
            success=False,
            message="fill_assets cannot be empty",
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

    for i, item in enumerate(fill_assets):
        missing_keys = [k for k in ["id", "x", "y"] if k not in item]
        if missing_keys:
            console_logger.warning(
                f"Create arrangement failed: fill_assets[{i}] missing {missing_keys}"
            )
            return FillContainerResult(
                success=False,
                message=(
                    f"fill_assets[{i}] missing required keys: {missing_keys}. "
                    f"Each item needs id, x, y (rotation is optional)."
                ),
                filled_container_id=None,
                container_asset_id=container_asset_id,
                fill_count=0,
                total_fill_attempted=len(fill_assets),
                removed_count=0,
                parent_surface_id=surface_id,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.INVALID_OPERATION,
            ).to_json()

    # 2. Validate surface exists.
    if surface_id not in support_surfaces:
        available_ids = list(support_surfaces.keys())
        console_logger.warning(
            f"Create arrangement failed: invalid surface_id '{surface_id}'"
        )
        return FillContainerResult(
            success=False,
            message=(
                f"Invalid surface_id: {surface_id}. Available surfaces: {available_ids}"
            ),
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_assets),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_SURFACE,
        ).to_json()

    target_surface = support_surfaces[surface_id]

    # 3. Get container asset.
    try:
        container_unique_id = UniqueID(container_asset_id)
    except Exception:
        console_logger.warning(
            f"Create arrangement failed: invalid container asset ID format "
            f"'{container_asset_id}'"
        )
        return FillContainerResult(
            success=False,
            message=f"Invalid container asset ID format: {container_asset_id}",
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_assets),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.ASSET_NOT_FOUND,
        ).to_json()

    container_asset = asset_manager.get_asset_by_id(container_unique_id)
    if not container_asset:
        console_logger.warning(
            f"Create arrangement failed: container asset '{container_asset_id}' not found"
        )
        return FillContainerResult(
            success=False,
            message=f"Container asset '{container_asset_id}' not found",
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_assets),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.ASSET_NOT_FOUND,
        ).to_json()

    # 4. Validate container has SDF path.
    if not container_asset.sdf_path or not container_asset.sdf_path.exists():
        console_logger.warning(
            f"Create arrangement failed: container asset '{container_asset_id}' "
            f"has no SDF file"
        )
        return FillContainerResult(
            success=False,
            message=(
                f"Container asset '{container_asset_id}' has no SDF file. "
                "Cannot simulate arrangement physics."
            ),
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_assets),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # 5. Reject thin_coverings (placemats, tablecloths).
    if container_asset.metadata.get("object_type") == "thin_covering":
        console_logger.warning(
            f"Create arrangement failed: '{container_asset_id}' is a thin_covering"
        )
        return FillContainerResult(
            success=False,
            message=(
                f"'{container_asset_id}' is a thin_covering (e.g., placemat). "
                f"Use place_manipuland to position items on thin_coverings. "
                f"create_arrangement is for containers with depth "
                f"(trays, platters, boards)."
            ),
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_assets),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # 6. Validate position is within surface bounds.
    position_2d = np.array([position_x, position_z])
    try:
        if not target_surface.contains_point_2d(position_2d):
            console_logger.warning(
                f"Create arrangement failed: position "
                f"({position_x:.3f}, {position_z:.3f}) outside surface {surface_id}"
            )
            return FillContainerResult(
                success=False,
                message=(
                    f"Position ({position_x:.3f}, {position_z:.3f}) is outside "
                    f"the convex hull of surface {surface_id}."
                ),
                filled_container_id=None,
                container_asset_id=container_asset_id,
                fill_count=0,
                total_fill_attempted=len(fill_assets),
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
            message=f"Surface {surface_id} has no mesh geometry for validation.",
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_assets),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_SURFACE,
        ).to_json()

    # 7. Validate container footprint fits within surface boundary.
    overlap_ratio = (
        top_surface_overlap_tolerance if is_top_surface_fn(surface_id) else 0.0
    )
    is_valid, error_msg = validate_footprint_fn(
        target_surface,
        container_asset.geometry_path,
        position_2d,
        rotation_degrees,
        overlap_ratio,
        container_asset.scale_factor,
    )
    if not is_valid:
        console_logger.warning(
            f"Create arrangement failed: footprint validation failed - {error_msg}"
        )
        return FillContainerResult(
            success=False,
            message=error_msg,
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_assets),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
        ).to_json()

    # 8. Compute shape-aware bounds.
    try:
        bounds_info = _get_container_bounds_info(
            container_asset=container_asset, cfg=cfg
        )
    except ValueError as e:
        console_logger.warning(f"Create arrangement failed: {e}")
        return FillContainerResult(
            success=False,
            message=str(e),
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_assets),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # 9. Get fill assets and validate positions within container bounds.
    fill_scene_assets: list[SceneObject] = []
    fill_names: list[str] = []

    for i, item in enumerate(fill_assets):
        asset_id = item["id"]
        x_offset = item.get("x", 0.0)
        y_offset = item.get("y", 0.0)

        try:
            fill_unique_id = UniqueID(asset_id)
        except Exception:
            console_logger.warning(
                f"Create arrangement failed: invalid fill asset ID format '{asset_id}'"
            )
            return FillContainerResult(
                success=False,
                message=f"Invalid fill asset ID format: {asset_id}",
                filled_container_id=None,
                container_asset_id=container_asset_id,
                fill_count=0,
                total_fill_attempted=len(fill_assets),
                removed_count=0,
                parent_surface_id=surface_id,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.ASSET_NOT_FOUND,
            ).to_json()

        fill_asset = asset_manager.get_asset_by_id(fill_unique_id)
        if not fill_asset:
            console_logger.warning(
                f"Create arrangement failed: fill asset '{asset_id}' not found"
            )
            return FillContainerResult(
                success=False,
                message=f"Fill asset '{asset_id}' not found in asset registry",
                filled_container_id=None,
                container_asset_id=container_asset_id,
                fill_count=0,
                total_fill_attempted=len(fill_assets),
                removed_count=0,
                parent_surface_id=surface_id,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.ASSET_NOT_FOUND,
            ).to_json()

        if not fill_asset.sdf_path or not fill_asset.sdf_path.exists():
            console_logger.warning(
                f"Create arrangement failed: fill asset '{asset_id}' has no SDF file"
            )
            return FillContainerResult(
                success=False,
                message=f"Fill asset '{asset_id}' has no SDF file.",
                filled_container_id=None,
                container_asset_id=container_asset_id,
                fill_count=0,
                total_fill_attempted=len(fill_assets),
                removed_count=0,
                parent_surface_id=surface_id,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.INVALID_OPERATION,
            ).to_json()

        # Validate item fits within container bounds (shape-aware).
        is_valid, error_msg = _validate_item_within_container_bounds(
            item_asset=fill_asset,
            x_offset=x_offset,
            y_offset=y_offset,
            bounds_info=bounds_info,
        )
        if not is_valid:
            console_logger.warning(f"Create arrangement failed: {error_msg}")
            return FillContainerResult(
                success=False,
                message=error_msg,
                filled_container_id=None,
                container_asset_id=container_asset_id,
                fill_count=0,
                total_fill_attempted=len(fill_assets),
                removed_count=0,
                parent_surface_id=surface_id,
                inside_assets=[],
                removed_assets=[],
                error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
            ).to_json()

        fill_scene_assets.append(fill_asset)
        fill_names.append(fill_asset.name)

    # 10. Compute container transform on surface.
    rotation_radians = math.radians(rotation_degrees)
    container_transform = target_surface.to_world_pose(
        position_2d=position_2d, rotation_2d=rotation_radians
    )

    # 11. Compute spawn transforms for fill objects.
    fill_cfg = cfg.fill_simulation
    container_height = 0.0
    if container_asset.bbox_max is not None:
        container_height = float(container_asset.bbox_max[2])

    spawn_transforms: list[RigidTransform] = []
    for i, item in enumerate(fill_assets):
        x_offset = item.get("x", 0.0)
        y_offset = item.get("y", 0.0)
        rotation_deg = item.get("rotation", 0.0)

        # Spawn above container surface with user-specified yaw.
        spawn_z = container_height + fill_cfg.spawn_height_above_rim
        local_translation = np.array([x_offset, y_offset, spawn_z])
        yaw_rotation = RotationMatrix.MakeZRotation(math.radians(rotation_deg))
        local_transform = RigidTransform(yaw_rotation, local_translation)
        spawn_transform = container_transform.multiply(local_transform)
        spawn_transforms.append(spawn_transform)

    # 12. Check for item-item collisions using SDF.
    try:
        collisions = compute_pairwise_collisions(
            objects=fill_scene_assets, transforms=spawn_transforms
        )
    except RuntimeError as e:
        console_logger.warning(f"Collision detection failed: {e}")
        collisions = []

    if collisions:
        # Fail with feedback about overlapping items.
        max_collision = max(collisions, key=lambda c: c.penetration_m)
        asset_a = fill_scene_assets[max_collision.obj_a_idx]
        asset_b = fill_scene_assets[max_collision.obj_b_idx]
        console_logger.warning(
            f"Create arrangement failed: {asset_a.name} and {asset_b.name} "
            f"overlap by {max_collision.penetration_m * 1000:.1f}mm"
        )
        return FillContainerResult(
            success=False,
            message=(
                f"Items '{asset_a.name}' and '{asset_b.name}' overlap by "
                f"{max_collision.penetration_m * 1000:.1f}mm. "
                f"Increase spacing between items to avoid collision."
            ),
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_assets),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
        ).to_json()

    # 13. Create temporary SceneObjects for physics simulation.
    container_scene_obj = clone_scene_object(
        container_asset,
        object_id=UniqueID("container_temp"),
        object_type=ObjectType.MANIPULAND,
        transform=container_transform,
        placement_info=None,
    )

    fill_scene_objects = [
        clone_scene_object(
            fill_asset,
            object_id=UniqueID(f"fill_temp_{i}"),
            object_type=ObjectType.MANIPULAND,
            transform=spawn_transforms[i],
            placement_info=None,
        )
        for i, fill_asset in enumerate(fill_scene_assets)
    ]

    # 14. Run physics simulation.
    sim_result: FillSimulationResult = simulate_fill_physics(
        container_scene_object=container_scene_obj,
        container_transform=container_transform,
        new_fill_objects=fill_scene_objects,
        new_fill_transforms=spawn_transforms,
        catch_floor_z=fill_cfg.catch_floor_z,
        inside_z_threshold=fill_cfg.inside_z_threshold,
        simulation_time=fill_cfg.simulation_time,
        simulation_time_step=fill_cfg.simulation_time_step,
    )

    if sim_result.error_message:
        console_logger.warning(
            f"Create arrangement failed: physics simulation error - "
            f"{sim_result.error_message}"
        )
        return FillContainerResult(
            success=False,
            message=f"Physics simulation failed: {sim_result.error_message}",
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_assets),
            removed_count=0,
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=[],
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # 15. Check for fall-off - FAIL if any items fell (no retry for arrangements).
    if sim_result.outside_indices:
        fallen_names = [fill_names[i] for i in sim_result.outside_indices]
        console_logger.warning(
            f"Create arrangement failed: {len(fallen_names)} items fell off container"
        )
        # Format container bounds for error message.
        if bounds_info["shape"] == "circular":
            bounds_str = (
                f"Container is circular with radius {bounds_info['radius']:.2f}m."
            )
        else:
            half_x = bounds_info["x"][1]
            half_y = bounds_info["y"][1]
            bounds_str = (
                f"Container bounds: x in [{-half_x:.2f}, {half_x:.2f}], "
                f"y in [{-half_y:.2f}, {half_y:.2f}]."
            )
        return FillContainerResult(
            success=False,
            message=(
                f"Items fell off container: {fallen_names}. "
                f"{bounds_str} Adjust positions closer to center."
            ),
            filled_container_id=None,
            container_asset_id=container_asset_id,
            fill_count=0,
            total_fill_attempted=len(fill_assets),
            removed_count=len(sim_result.outside_indices),
            parent_surface_id=surface_id,
            inside_assets=[],
            removed_assets=fallen_names,
            error_type=ManipulandErrorType.POSITION_OUT_OF_BOUNDS,
        ).to_json()

    # 16. Create composite SceneObject.
    filled_container_id = generate_unique_id("filled_container")

    # Build container_asset metadata.
    container_asset_data = serialize_composite_member_asset(
        container_asset, container_transform
    )

    # Build fill_assets metadata (extended for arrangements to include user poses).
    fill_assets_data = []
    inside_names = []
    for i, fill_asset in enumerate(fill_scene_assets):
        fill_transform = sim_result.final_transforms[i]
        item = fill_assets[i]
        fill_assets_data.append(
            serialize_composite_member_asset(
                fill_asset,
                fill_transform,
                extra_fields={
                    "user_pose": {
                        "x": item.get("x", 0.0),
                        "y": item.get("y", 0.0),
                        "rotation": item.get("rotation", 0.0),
                    },
                },
            )
        )
        inside_names.append(fill_asset.name)

    # Compute composite bounding box.
    all_bbox_min, all_bbox_max = compute_composite_bbox_in_local_frame(
        container_asset=container_asset,
        container_transform=container_transform,
        fill_assets=fill_scene_assets,
        final_fill_transforms=sim_result.final_transforms,
        inside_indices=list(range(len(fill_scene_assets))),
    )

    # Create composite SceneObject.
    composite_object = SceneObject(
        object_id=filled_container_id,
        object_type=ObjectType.MANIPULAND,
        name=f"filled_{container_asset.name}",
        description=f"{container_asset.name} with {', '.join(inside_names)}",
        transform=container_transform,
        geometry_path=None,
        sdf_path=None,
        metadata={
            "composite_type": "filled_container",
            "container_asset": container_asset_data,
            "fill_assets": fill_assets_data,
            "num_fill_objects": len(fill_scene_assets),
            "placement_method": "controlled",
            "container_bounds": bounds_info,
        },
        bbox_min=all_bbox_min,
        bbox_max=all_bbox_max,
        placement_info=PlacementInfo(
            parent_surface_id=target_surface.surface_id,
            position_2d=position_2d.copy(),
            rotation_2d=rotation_radians,
            placement_method="create_arrangement",
        ),
    )

    scene.add_object(composite_object)

    elapsed_time = time.time() - start_time
    console_logger.info(
        f"Created arrangement {filled_container_id} with "
        f"{len(fill_scene_assets)} items on {surface_id} in {elapsed_time:.2f}s"
    )

    return FillContainerResult(
        success=True,
        message=f"Created '{filled_container_id}' with {len(fill_scene_assets)} items",
        filled_container_id=str(filled_container_id),
        container_asset_id=container_asset_id,
        fill_count=len(fill_scene_assets),
        total_fill_attempted=len(fill_assets),
        removed_count=0,
        parent_surface_id=surface_id,
        inside_assets=inside_names,
        removed_assets=[],
    ).to_json()
