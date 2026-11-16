"""Stack tool implementation for creating stacks of objects."""

import logging
import math
import time

from typing import Callable

import numpy as np

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
from scenecode.manipuland_agents.tools.fill_container import compute_bbox_corners
from scenecode.manipuland_agents.tools.physics_utils import (
    load_collision_bounds_for_scene_object,
)
from scenecode.manipuland_agents.tools.response_dataclasses import (
    ManipulandErrorType,
    StackCreationResult,
)
from scenecode.manipuland_agents.tools.stacking import (
    compute_actual_stack_height,
    compute_initial_stack_transforms,
    simulate_stack_stability,
)

console_logger = logging.getLogger(__name__)


def _compute_stack_composite_bbox_in_local_frame(
    assets: list[SceneObject],
    final_transforms: list[RigidTransform],
    stack_transform: RigidTransform,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute composite bounding box for stack members in stack's local frame.

    Args:
        assets: List of SceneObjects in the stack.
        final_transforms: Final world transforms for each asset after simulation.
        stack_transform: The stack's reference transform (bottom member's transform).

    Returns:
        Tuple of (bbox_min, bbox_max) in stack's local frame, or (None, None)
        if no valid bounding boxes.
    """
    all_bbox_min = np.array([np.inf, np.inf, np.inf])
    all_bbox_max = np.array([-np.inf, -np.inf, -np.inf])
    bbox_count = 0

    for i, asset in enumerate(assets):
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

    # Transform world bbox back to local frame relative to stack_transform.
    inverse_transform = stack_transform.inverse()
    world_corners = compute_bbox_corners(bbox_min=all_bbox_min, bbox_max=all_bbox_max)
    local_bbox_min = np.array([np.inf, np.inf, np.inf])
    local_bbox_max = np.array([-np.inf, -np.inf, -np.inf])
    for corner in world_corners:
        local_corner = inverse_transform.multiply(corner)
        local_bbox_min = np.minimum(local_bbox_min, local_corner)
        local_bbox_max = np.maximum(local_bbox_max, local_corner)

    return local_bbox_min, local_bbox_max


def create_stack_tool_impl(
    asset_ids: list[str],
    surface_id: str,
    position_x: float,
    position_z: float,
    rotation_degrees: float,
    scene: RoomScene,
    cfg: DictConfig,
    asset_manager: AssetManager,
    support_surfaces: dict[str, SupportSurface],
    generate_unique_id: Callable[[str], UniqueID],
) -> str:
    """Create a stack of objects on a support surface.

    This is the core implementation logic for stack creation, taking all
    dependencies as explicit parameters.

    Args:
        asset_ids: List of asset IDs to stack (bottom to top order).
        surface_id: ID of the support surface to place the stack on.
        position_x: X coordinate on the surface (in surface's local frame).
        position_z: Z coordinate on the surface (in surface's local frame).
        rotation_degrees: Rotation of the stack on the surface (degrees).
        scene: The RoomScene to add the stack to.
        cfg: Configuration with stack_simulation settings.
        asset_manager: AssetManager for retrieving asset objects.
        support_surfaces: Dictionary of available support surfaces.
        generate_unique_id: Function to generate unique IDs for new objects.

    Returns:
        JSON string with StackCreationResult.
    """
    console_logger.info(
        f"Tool called: create_stack(asset_ids={asset_ids}, surface_id={surface_id}, "
        f"position_x={position_x}, position_z={position_z}, "
        f"rotation_degrees={rotation_degrees})"
    )
    start_time = time.time()

    # Validate at least 2 assets for a stack.
    if len(asset_ids) < 2:
        console_logger.warning(
            f"Stack creation failed: requires at least 2 assets, got {len(asset_ids)}"
        )
        return StackCreationResult(
            success=False,
            message=(
                "Stack requires at least 2 assets. "
                "Use place_manipuland_on_surface for single items."
            ),
            stack_object_id=None,
            stack_height=None,
            parent_surface_id=surface_id,
            num_items=len(asset_ids),
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # Validate surface exists.
    if surface_id not in support_surfaces:
        available_ids = list(support_surfaces.keys())
        console_logger.warning(
            f"Stack creation failed: invalid surface_id '{surface_id}'"
        )
        return StackCreationResult(
            success=False,
            message=(
                f"Invalid surface_id: {surface_id}. "
                f"Available surfaces: {available_ids}"
            ),
            stack_object_id=None,
            stack_height=None,
            parent_surface_id=surface_id,
            num_items=len(asset_ids),
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
                f"Stack creation failed: invalid asset ID format '{asset_id}'"
            )
            return StackCreationResult(
                success=False,
                message=f"Invalid asset ID format: {asset_id}",
                stack_object_id=None,
                stack_height=None,
                parent_surface_id=surface_id,
                num_items=len(asset_ids),
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
                f"Stack creation failed: asset '{asset_id}' not found"
            )
            return StackCreationResult(
                success=False,
                message=(
                    f"Asset {asset_id} not found. "
                    f"Available manipulands: {manipuland_ids}"
                ),
                stack_object_id=None,
                stack_height=None,
                parent_surface_id=surface_id,
                num_items=len(asset_ids),
                error_type=ManipulandErrorType.ASSET_NOT_FOUND,
            ).to_json()

        # Validate SDF path exists for physics simulation.
        if not asset.sdf_path or not asset.sdf_path.exists():
            console_logger.warning(
                f"Stack creation failed: asset '{asset_id}' has no SDF file"
            )
            return StackCreationResult(
                success=False,
                message=(
                    f"Asset {asset_id} has no SDF file. "
                    "Cannot simulate stack stability."
                ),
                stack_object_id=None,
                stack_height=None,
                parent_surface_id=surface_id,
                num_items=len(asset_ids),
                error_type=ManipulandErrorType.INVALID_OPERATION,
            ).to_json()

        assets.append(asset)

    # Load collision bounds for each asset, tracking which ones fail.
    collision_bounds_list = []
    unstackable_assets = []
    for asset in assets:
        temp_obj = clone_scene_object(
            asset,
            object_id=UniqueID("temp"),
            object_type=ObjectType.MANIPULAND,
            transform=RigidTransform(),
            placement_info=None,
        )
        try:
            bounds = load_collision_bounds_for_scene_object(temp_obj)
            collision_bounds_list.append(bounds)
        except ValueError:
            unstackable_assets.append(asset.name)

    if unstackable_assets:
        names = ", ".join(unstackable_assets)
        console_logger.warning(
            f"Stack creation failed: objects without collision geometry: {names}"
        )
        return StackCreationResult(
            success=False,
            message=(
                f"Cannot stack: {names} cannot be stacked because they are flat "
                f"decorative objects without physics geometry. To stack these items, "
                f"regenerate them as 3D objects with visible thickness "
                f"(e.g., 'folded towel with realistic thickness')."
            ),
            stack_object_id=None,
            stack_height=None,
            parent_surface_id=surface_id,
            num_items=len(asset_ids),
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    # Compute base transform on surface.
    position_2d = np.array([position_x, position_z])
    rotation_radians = math.radians(rotation_degrees)
    base_transform = target_surface.to_world_pose(
        position_2d=position_2d, rotation_2d=rotation_radians
    )

    # Compute initial stack transforms using collision geometry.
    initial_transforms = compute_initial_stack_transforms(
        collision_bounds_list=collision_bounds_list,
        base_transform=base_transform,
    )

    # Create temporary SceneObjects for simulation.
    temp_scene_objects = []
    for i, asset in enumerate(assets):
        temp_obj = clone_scene_object(
            asset,
            object_id=UniqueID(f"stack_temp_{i}"),
            object_type=ObjectType.MANIPULAND,
            transform=initial_transforms[i],
            placement_info=None,
        )
        temp_scene_objects.append(temp_obj)

    # Get simulation config.
    stack_cfg = cfg.stack_simulation

    # Ground plane position: XY from stack placement, Z from surface top.
    ground_xyz = (
        base_transform.translation()[0],
        base_transform.translation()[1],
        target_surface.transform.translation()[2],
    )

    # Run physics simulation.
    sim_start_time = time.time()
    sim_result = simulate_stack_stability(
        scene_objects=temp_scene_objects,
        initial_transforms=initial_transforms,
        ground_xyz=ground_xyz,
        simulation_time=stack_cfg.simulation_time,
        simulation_time_step=stack_cfg.simulation_time_step,
        position_threshold=stack_cfg.position_threshold,
    )
    sim_elapsed = time.time() - sim_start_time
    console_logger.info(f"Stack simulation completed in {sim_elapsed:.2f}s")

    if sim_result.error_message:
        console_logger.error(
            f"Stack creation failed: simulation error: {sim_result.error_message}"
        )
        return StackCreationResult(
            success=False,
            message=f"Simulation failed: {sim_result.error_message}",
            stack_object_id=None,
            stack_height=None,
            parent_surface_id=surface_id,
            num_items=len(asset_ids),
            error_type=ManipulandErrorType.INVALID_OPERATION,
        ).to_json()

    if not sim_result.is_stable:
        num_stable = len(sim_result.stable_indices)
        num_total = len(asset_ids)

        if num_stable > 0:
            stable_str = f"items 0-{max(sim_result.stable_indices)}"
        else:
            stable_str = "no items"

        console_logger.warning(
            f"Stack creation failed: unstable. {num_stable} of {num_total} "
            f"items stable ({stable_str})"
        )
        return StackCreationResult(
            success=False,
            message=(
                f"Stack unstable. {num_stable} of {num_total} items stable "
                f"({stable_str}). Retry with {num_stable} items."
            ),
            stack_object_id=None,
            stack_height=None,
            parent_surface_id=surface_id,
            num_items=len(asset_ids),
            error_type=ManipulandErrorType.STACK_UNSTABLE,
        ).to_json()

    # Check clearance using actual (settled) height from simulation.
    surface_z = target_surface.transform.translation()[2]
    actual_height = (
        compute_actual_stack_height(
            transforms=sim_result.final_transforms,
            collision_bounds_list=collision_bounds_list,
        )
        - surface_z
    )
    surface_clearance = float(
        target_surface.bounding_box_max[2] - target_surface.bounding_box_min[2]
    )

    if actual_height > surface_clearance:
        height_per_item = actual_height / len(asset_ids)
        max_items = int(surface_clearance / height_per_item)
        console_logger.warning(
            f"Stack creation failed: height {actual_height:.3f}m exceeds "
            f"clearance {surface_clearance:.3f}m (could fit ~{max_items} items)"
        )
        return StackCreationResult(
            success=False,
            message=(
                f"Stack height {actual_height:.2f}m exceeds clearance "
                f"{surface_clearance:.2f}m. "
                f"Height per item: ~{height_per_item:.2f}m. "
                f"Could fit ~{max_items} items (have {len(asset_ids)})."
            ),
            stack_object_id=None,
            stack_height=None,
            parent_surface_id=surface_id,
            num_items=len(asset_ids),
            error_type=ManipulandErrorType.STACK_EXCEEDS_CLEARANCE,
        ).to_json()

    # Stack is stable - create composite SceneObject.
    stack_id = generate_unique_id("stack")

    # Use bottom member's final transform as the stack's reference transform.
    stack_transform = sim_result.final_transforms[0]

    # Build member_assets metadata with sdf_path for each member.
    member_assets = []
    for i, (asset, final_transform) in enumerate(
        zip(assets, sim_result.final_transforms)
    ):
        member_assets.append(
            serialize_composite_member_asset(asset, final_transform)
        )

    # Compute composite bounding box.
    all_bbox_min, all_bbox_max = _compute_stack_composite_bbox_in_local_frame(
        assets=assets,
        final_transforms=sim_result.final_transforms,
        stack_transform=stack_transform,
    )

    composite_object = SceneObject(
        object_id=stack_id,
        object_type=ObjectType.MANIPULAND,
        name=f"stack_{len(assets)}",
        description=f"Stack of {len(assets)} objects: "
        + ", ".join(a.name for a in assets),
        transform=stack_transform,
        geometry_path=None,
        sdf_path=None,
        metadata={
            "composite_type": "stack",
            "member_assets": member_assets,
            "num_members": len(assets),
        },
        bbox_min=all_bbox_min,
        bbox_max=all_bbox_max,
        placement_info=PlacementInfo(
            parent_surface_id=target_surface.surface_id,
            position_2d=position_2d.copy(),
            rotation_2d=rotation_radians,
            placement_method="stack_placement",
        ),
    )

    scene.add_object(composite_object)

    elapsed_time = time.time() - start_time
    console_logger.info(
        f"Successfully created stack {stack_id} with {len(assets)} items, "
        f"height {actual_height:.3f}m on surface {surface_id} in {elapsed_time:.2f}s"
    )

    return StackCreationResult(
        success=True,
        message=(
            f"Created stack '{stack_id}' with {len(assets)} objects, "
            f"height {actual_height:.3f}m"
        ),
        stack_object_id=str(stack_id),
        stack_height=actual_height,
        parent_surface_id=surface_id,
        num_items=len(asset_ids),
    ).to_json()
