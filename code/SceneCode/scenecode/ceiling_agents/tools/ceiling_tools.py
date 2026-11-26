"""Tools for ceiling-mounted object generation and placement.

This module provides tools for generating and placing ceiling-mounted objects
(lights, fans, chandeliers, etc.) on the ceiling plane.
"""

import logging
import math

from typing import Any

from agents import function_tool
from omegaconf import DictConfig
from pydrake.all import RigidTransform, RollPitchYaw

from scenecode.agent_utils.action_logger import log_scene_action
from scenecode.agent_utils.asset_manager import AssetGenerationRequest, AssetManager
from scenecode.agent_utils.loop_detector import LoopDetector
from scenecode.agent_utils.placement_noise import (
    PlacementNoiseMode,
    apply_ceiling_placement_noise,
)
from scenecode.agent_utils.rescale_helpers import rescale_object_common
from scenecode.agent_utils.response_datatypes import (
    AssetGenerationResult as AssetGenerationResultDTO,
    AssetInfo,
    BoundingBox3D,
    GeneratedAsset,
)
from scenecode.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    UniqueID,
    clone_scene_object,
)
from scenecode.ceiling_agents.tools.response_dataclasses import (
    AvailableAssetsResult,
    CeilingErrorType,
    CeilingObjectInfo,
    CeilingOperationResult,
    CeilingSceneStateResult,
    PlaceCeilingObjectResult,
    RoomBoundsInfo,
)

console_logger = logging.getLogger(__name__)


def compute_ceiling_transform(
    x: float, y: float, rotation_deg: float, ceiling_height: float
) -> RigidTransform:
    """Place object at ceiling with top at ceiling_height.

    Due to CEILING_MOUNTED canonicalization (top at z=0), placing at
    z=ceiling_height puts the object's top flush with the ceiling.

    Args:
        x: Position along room X-axis (meters).
        y: Position along room Y-axis (meters).
        rotation_deg: Rotation around Z-axis (degrees).
        ceiling_height: Height of ceiling (meters).

    Returns:
        RigidTransform for object pose in world frame.
    """
    return RigidTransform(
        rpy=RollPitchYaw(roll=0, pitch=0, yaw=math.radians(rotation_deg)),
        p=[x, y, ceiling_height],
    )


class CeilingTools:
    """Tools for ceiling-mounted object generation and placement.

    Provides tools for generating 3D ceiling-mounted objects and placing them
    using SE(2) coordinates on the ceiling plane (x, y, rotation around Z).
    """

    def __init__(
        self,
        scene: RoomScene,
        room_bounds: tuple[float, float, float, float],
        ceiling_height: float,
        asset_manager: AssetManager,
        cfg: DictConfig,
    ):
        """Initialize ceiling tools.

        Args:
            scene: RoomScene instance to manipulate.
            room_bounds: Room XY bounds (min_x, min_y, max_x, max_y).
            ceiling_height: Height of ceiling above floor (meters).
            asset_manager: Asset manager for generating 3D assets.
            cfg: Configuration object containing loop detection and noise settings.
        """
        self.scene = scene
        self.room_bounds = room_bounds
        self.ceiling_height = ceiling_height
        self.asset_manager = asset_manager
        self.cfg = cfg

        # Initialize placement noise configuration.
        # Start with natural profile as default until planner sets it.
        self.placement_noise_config = cfg.placement_noise
        self.active_noise_profile = self.placement_noise_config.natural_profile

        # Initialize loop detector from config.
        loop_config = cfg.loop_detection
        loop_detector = LoopDetector(
            max_attempts=loop_config.max_repeated_attempts,
            window_size=loop_config.tracking_window,
            enabled=loop_config.enabled,
            default_error_factory=self._create_loop_error_response,
        )

        # Apply loop detection to implementation methods.
        self._place_ceiling_object_impl = loop_detector(self._place_ceiling_object_impl)
        self._move_ceiling_object_impl = loop_detector(self._move_ceiling_object_impl)
        self._remove_ceiling_object_impl = loop_detector(
            self._remove_ceiling_object_impl
        )

        # Create tool closures.
        self.tools = self._create_tool_closures()

    def set_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Update the active noise profile based on placement style.

        Args:
            mode: Placement noise mode (NATURAL or PERFECT).
        """
        if mode == PlacementNoiseMode.NATURAL:
            self.active_noise_profile = self.placement_noise_config.natural_profile
            console_logger.info("Placement noise set to NATURAL profile")
        elif mode == PlacementNoiseMode.PERFECT:
            self.active_noise_profile = self.placement_noise_config.perfect_profile
            console_logger.info("Placement noise set to PERFECT profile")
        else:
            console_logger.warning(
                f"Unsupported noise mode {mode}, keeping current profile"
            )

    def _create_loop_error_response(
        self, method_name: str, attempt_count: int, _args: tuple, kwargs: dict
    ) -> str:
        """Create ceiling-specific error response for loop detection."""
        # Extract identifiers from kwargs if available.
        object_id = kwargs.get("object_id", kwargs.get("asset_id", ""))

        result = CeilingOperationResult(
            success=False,
            message=(
                f"Loop detected: {attempt_count} similar attempts on '{method_name}'. "
                f"Try a different approach."
            ),
            object_id=object_id,
            error_type=CeilingErrorType.LOOP_DETECTED,
        )
        return result.to_json()

    def _create_tool_closures(self) -> dict[str, Any]:
        """Create tool closures that capture ceiling context."""

        @function_tool
        def generate_ceiling_assets(
            object_descriptions: list[str],
            short_names: list[str],
            desired_dimensions: list[list[float]],
            style_context: str | None = None,
        ) -> str:
            """Generate 3D ceiling-mounted assets from text descriptions.

            Creates ceiling objects like lights, fans, chandeliers, etc.

            Dimensions for ceiling objects:
            - width: extent along room X-axis
            - depth: extent along room Y-axis
            - height: how far object hangs down from ceiling

            Args:
                object_descriptions: List of object descriptions
                    (e.g., "Modern pendant light", "Ceiling fan with light").
                short_names: List of filesystem-safe names
                    (e.g., "pendant_1", "ceiling_fan").
                desired_dimensions: List of [width, depth, height] in meters.
                style_context: Optional style context for visual consistency
                    (e.g., "modern minimalist", "rustic farmhouse").

            Returns:
                JSON with IDs and details of created ceiling object assets.
            """
            console_logger.info(
                f"Tool called: generate_ceiling_assets("
                f"object_descriptions={object_descriptions}, "
                f"short_names={short_names})"
            )
            request = AssetGenerationRequest(
                object_descriptions=object_descriptions,
                short_names=short_names,
                object_type=ObjectType.CEILING_MOUNTED,
                desired_dimensions=desired_dimensions,
                style_context=style_context,
                scene_id=self.scene.scene_dir.name,
            )
            return self._generate_assets_impl(request)

        @function_tool
        def place_ceiling_object(
            asset_id: str,
            position_x: float,
            position_y: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Place a ceiling-mounted object on the ceiling.

            Position is in room coordinates (origin at room corner).
            Object's top will be flush with ceiling.

            Each placement gets a unique ID so you can move or remove it later.
            The same ceiling object can be placed multiple times.

            Coordinate system:
            - X: position along room X-axis (meters from room origin)
            - Y: position along room Y-axis (meters from room origin)
            - Rotation: degrees around Z-axis (looking from above)

            IMPORTANT: Check room bounds before placing. Use
            get_current_scene_state() to see room dimensions.
            Use the 'observe_scene' tool to see ceiling surface coordinates.

            Args:
                asset_id: ID of the ceiling asset to place.
                position_x: X position in room (meters from room origin).
                position_y: Y position in room (meters from room origin).
                rotation_degrees: Rotation around Z-axis (degrees).
                    Typically 0 for symmetric objects like lights.
                    Positive = counterclockwise when viewed from above.

            Returns:
                Placement result with world pose.
            """
            return self._place_ceiling_object_impl(
                asset_id=asset_id,
                position_x=position_x,
                position_y=position_y,
                rotation_degrees=rotation_degrees,
            )

        @function_tool
        def move_ceiling_object(
            object_id: str,
            position_x: float,
            position_y: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Move an existing ceiling object to a new position.

            Use this to reposition ceiling objects.
            You need the object ID from when you placed it or from
            'get_current_scene_state' tool.

            Args:
                object_id: ID of the ceiling object to move.
                position_x: New X position in room (meters).
                position_y: New Y position in room (meters).
                rotation_degrees: New rotation around Z-axis (degrees).

            Returns:
                Result of the move operation with new world pose.
            """
            return self._move_ceiling_object_impl(
                object_id=object_id,
                position_x=position_x,
                position_y=position_y,
                rotation_degrees=rotation_degrees,
            )

        @function_tool
        def remove_ceiling_object(object_id: str) -> str:
            """Remove a ceiling object from the scene.

            Args:
                object_id: ID of the ceiling object to remove.

            Returns:
                Result of the removal operation.
            """
            return self._remove_ceiling_object_impl(object_id=object_id)

        @function_tool
        def get_current_scene_state() -> str:
            """Get current scene state for ceiling objects.

            Shows:
            - Room bounds (min_x, min_y, max_x, max_y) and ceiling height
            - Ceiling objects already placed with their positions

            Returns:
                Scene state with room bounds and placed ceiling objects.
            """
            console_logger.info("Tool called: get_current_scene_state")
            return self._get_current_scene_state_impl()

        @function_tool
        def list_available_assets() -> str:
            """List all available ceiling assets.

            Returns list of generated ceiling assets that can be placed.

            Returns:
                List of available ceiling assets with IDs and descriptions.
            """
            console_logger.info("Tool called: list_available_assets")
            return self._list_available_assets_impl()

        @function_tool
        def rescale_ceiling_object(object_id: str, scale_factor: float) -> str:
            """Resize a ceiling object by a uniform scale factor.

            Use when the object's shape/proportions are correct but size is wrong.
            For shape or proportion issues, remove and regenerate instead.

            IMPORTANT: This rescales the underlying ASSET (SDF file). All instances
            of the same asset will be affected. This is usually what you want -
            if one light is too small, all lights of that type are too small.

            Args:
                object_id: ID of the ceiling object to rescale.
                scale_factor: Scale multiplier (e.g., 1.5 = 50% larger,
                    0.8 = 20% smaller). Must be positive and not 1.0.

            Returns:
                JSON with rescale result including new dimensions.
            """
            return self._rescale_ceiling_object_impl(
                object_id=object_id,
                scale_factor=scale_factor,
            )

        return {
            "generate_ceiling_assets": generate_ceiling_assets,
            "place_ceiling_object": place_ceiling_object,
            "move_ceiling_object": move_ceiling_object,
            "remove_ceiling_object": remove_ceiling_object,
            "rescale_ceiling_object": rescale_ceiling_object,
            "get_current_scene_state": get_current_scene_state,
            "list_available_assets": list_available_assets,
        }

    def _generate_assets_impl(self, request: AssetGenerationRequest) -> str:
        """Implementation for generating ceiling assets."""
        try:
            result = self.asset_manager.generate_assets(request=request)

            # Convert successful assets to DTOs.
            generated_assets = []
            for obj in result.successful_assets:
                if obj.bbox_min is None or obj.bbox_max is None:
                    raise RuntimeError(
                        f"Successful asset '{obj.name}' ({obj.object_id}) has no bbox. "
                        f"All successful assets must have bounding boxes."
                    )
                generated_assets.append(
                    GeneratedAsset(
                        name=obj.name,
                        object_id=str(obj.object_id),
                        description=obj.description,
                        width=float(obj.bbox_max[0] - obj.bbox_min[0]),
                        depth=float(obj.bbox_max[1] - obj.bbox_min[1]),
                        height=float(obj.bbox_max[2] - obj.bbox_min[2]),
                    )
                )

            # Handle partial success.
            if result.has_failures:
                failure_details = "; ".join(
                    f"{fa.description}: {fa.error_message}"
                    for fa in result.failed_assets
                )
                return AssetGenerationResultDTO(
                    success=False,
                    assets=generated_assets,
                    message=f"Partial success: {len(generated_assets)} generated, "
                    f"{len(result.failed_assets)} failed. Failures: {failure_details}",
                    successful_count=len(generated_assets),
                    failed_count=len(result.failed_assets),
                    failures=failure_details,
                ).to_json()

            # All succeeded.
            return AssetGenerationResultDTO(
                success=True,
                assets=generated_assets,
                message=f"Successfully generated {len(generated_assets)} ceiling "
                f"assets.",
            ).to_json()

        except Exception as e:
            console_logger.error(f"Error generating ceiling assets: {e}", exc_info=True)
            return AssetGenerationResultDTO(
                success=False,
                assets=[],
                message=f"Asset generation failed: {str(e)}",
            ).to_json()

    @log_scene_action
    def _place_ceiling_object_impl(
        self,
        asset_id: str,
        position_x: float,
        position_y: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for placing ceiling object."""
        console_logger.info(
            f"Tool called: place_ceiling_object("
            f"asset_id={asset_id}, position_x={position_x}, "
            f"position_y={position_y}, rotation_degrees={rotation_degrees})"
        )

        try:
            # Convert string ID to UniqueID.
            try:
                unique_id = UniqueID(asset_id)
            except Exception:
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    position_x=position_x,
                    position_y=position_y,
                    rotation_deg=rotation_degrees,
                    message=f"Invalid asset ID format: {asset_id}",
                    error_type=CeilingErrorType.ASSET_NOT_FOUND,
                )

            # Get asset from registry.
            original_asset = self.asset_manager.get_asset_by_id(unique_id)
            if not original_asset:
                all_assets = self.asset_manager.list_available_assets()
                available_assets = [
                    asset
                    for asset in all_assets
                    if asset.object_type == ObjectType.CEILING_MOUNTED
                ]
                available_ids = [str(a.object_id) for a in available_assets]
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    position_x=position_x,
                    position_y=position_y,
                    rotation_deg=rotation_degrees,
                    message=(
                        f"Asset {asset_id} not found. Available ceiling assets: "
                        f"{available_ids}"
                    ),
                    error_type=CeilingErrorType.ASSET_NOT_FOUND,
                )

            # Apply placement noise for realistic variation.
            noisy_x, noisy_y, noisy_rotation = apply_ceiling_placement_noise(
                position_x=position_x,
                position_y=position_y,
                rotation_deg=rotation_degrees,
                position_xy_std_meters=self.active_noise_profile.position_xy_std_meters,
                rotation_yaw_std_degrees=(
                    self.active_noise_profile.rotation_yaw_std_degrees
                ),
            )

            # Validate noisy position is within room bounds.
            min_x, min_y, max_x, max_y = self.room_bounds
            if not (min_x <= noisy_x <= max_x and min_y <= noisy_y <= max_y):
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    position_x=position_x,
                    position_y=position_y,
                    rotation_deg=rotation_degrees,
                    message=(
                        f"Position ({noisy_x:.2f}, {noisy_y:.2f}) is outside room "
                        f"bounds ({min_x:.2f}, {min_y:.2f}) to "
                        f"({max_x:.2f}, {max_y:.2f})"
                    ),
                    error_type=CeilingErrorType.POSITION_OUT_OF_BOUNDS,
                )

            console_logger.info(
                f"Placing ceiling object {asset_id} ({original_asset.name}) at "
                f"position ({noisy_x:.3f}, {noisy_y:.3f}), "
                f"rotation {noisy_rotation:.1f}°"
            )

            # Convert ceiling SE(2) to world SE(3).
            world_transform = compute_ceiling_transform(
                x=noisy_x,
                y=noisy_y,
                rotation_deg=noisy_rotation,
                ceiling_height=self.ceiling_height,
            )

            # Create new scene object with unique ID.
            object_id = self.scene.generate_unique_id(original_asset.name)
            scene_object = clone_scene_object(
                original_asset,
                object_id=object_id,
                object_type=ObjectType.CEILING_MOUNTED,
                transform=world_transform,
            )

            # Add to scene.
            self.scene.add_object(scene_object)

            console_logger.info(
                f"Successfully placed ceiling object '{original_asset.name}' as "
                f"{object_id}"
            )

            # Create success result.
            result = PlaceCeilingObjectResult(
                success=True,
                message=(
                    f"Successfully placed '{original_asset.name}' at "
                    f"({noisy_x:.3f}m, {noisy_y:.3f}m)"
                ),
                asset_id=asset_id,
                object_id=str(object_id),
                position_x=noisy_x,
                position_y=noisy_y,
                rotation_degrees=noisy_rotation,
            )

            return result.to_json()

        except Exception as e:
            console_logger.error(f"Error placing ceiling object: {e}", exc_info=True)
            return self._create_placement_failure_result(
                asset_id=asset_id,
                position_x=position_x,
                position_y=position_y,
                rotation_deg=rotation_degrees,
                message=f"Unexpected error: {str(e)}",
                error_type=None,
            )

    @log_scene_action
    def _move_ceiling_object_impl(
        self,
        object_id: str,
        position_x: float,
        position_y: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for moving ceiling object to new position."""
        console_logger.info(
            f"Tool called: move_ceiling_object("
            f"object_id={object_id}, position_x={position_x}, "
            f"position_y={position_y}, rotation_degrees={rotation_degrees})"
        )

        try:
            # Get the existing object.
            unique_id = UniqueID(object_id)
            scene_object = self.scene.get_object(unique_id)
            if scene_object is None:
                return CeilingOperationResult(
                    success=False,
                    message=f"Object {object_id} not found in scene.",
                    object_id=object_id,
                    error_type=CeilingErrorType.OBJECT_NOT_FOUND,
                ).to_json()

            # Verify it's a ceiling-mounted object.
            if scene_object.object_type != ObjectType.CEILING_MOUNTED:
                return CeilingOperationResult(
                    success=False,
                    message=(
                        f"Object {object_id} is not a ceiling-mounted object "
                        f"(type: {scene_object.object_type.value})."
                    ),
                    object_id=object_id,
                    error_type=CeilingErrorType.INVALID_OPERATION,
                ).to_json()

            # Validate position is within room bounds.
            min_x, min_y, max_x, max_y = self.room_bounds
            if not (min_x <= position_x <= max_x and min_y <= position_y <= max_y):
                return CeilingOperationResult(
                    success=False,
                    message=(
                        f"Position ({position_x:.2f}, {position_y:.2f}) is outside "
                        f"room bounds ({min_x:.2f}, {min_y:.2f}) to "
                        f"({max_x:.2f}, {max_y:.2f})"
                    ),
                    object_id=object_id,
                    error_type=CeilingErrorType.POSITION_OUT_OF_BOUNDS,
                ).to_json()

            # Convert ceiling SE(2) to world SE(3).
            world_transform = compute_ceiling_transform(
                x=position_x,
                y=position_y,
                rotation_deg=rotation_degrees,
                ceiling_height=self.ceiling_height,
            )

            # Update the object's transform.
            scene_object.transform = world_transform

            console_logger.info(
                f"Moved ceiling object '{scene_object.name}' to "
                f"({position_x:.3f}, {position_y:.3f})"
            )

            return CeilingOperationResult(
                success=True,
                message=(
                    f"Successfully moved '{scene_object.name}' to "
                    f"({position_x:.3f}m, {position_y:.3f}m)"
                ),
                object_id=object_id,
            ).to_json()

        except Exception as e:
            console_logger.error(f"Error moving ceiling object: {e}", exc_info=True)
            return CeilingOperationResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                object_id=object_id,
                error_type=None,
            ).to_json()

    @log_scene_action
    def _remove_ceiling_object_impl(self, object_id: str, **kwargs) -> str:
        """Implementation for removing ceiling object from scene."""
        console_logger.info(
            f"Tool called: remove_ceiling_object(object_id={object_id})"
        )

        try:
            unique_id = UniqueID(object_id)
            scene_object = self.scene.get_object(unique_id)
            if scene_object is None:
                return CeilingOperationResult(
                    success=False,
                    message=f"Object {object_id} not found in scene.",
                    object_id=object_id,
                    error_type=CeilingErrorType.OBJECT_NOT_FOUND,
                ).to_json()

            # Verify it's a ceiling-mounted object.
            if scene_object.object_type != ObjectType.CEILING_MOUNTED:
                return CeilingOperationResult(
                    success=False,
                    message=(
                        f"Object {object_id} is not a ceiling-mounted object "
                        f"(type: {scene_object.object_type.value})."
                    ),
                    object_id=object_id,
                    error_type=CeilingErrorType.INVALID_OPERATION,
                ).to_json()

            # Remove from scene.
            self.scene.remove_object(unique_id)

            console_logger.info(f"Removed ceiling object '{scene_object.name}'")

            return CeilingOperationResult(
                success=True,
                message=f"Successfully removed '{scene_object.name}'.",
                object_id=object_id,
            ).to_json()

        except Exception as e:
            console_logger.error(f"Error removing ceiling object: {e}", exc_info=True)
            return CeilingOperationResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                object_id=object_id,
                error_type=None,
            ).to_json()

    @log_scene_action
    def _rescale_ceiling_object_impl(
        self, object_id: str, scale_factor: float, **kwargs
    ) -> str:
        """Implementation for rescaling a ceiling object."""
        console_logger.info(
            f"Tool called: rescale_ceiling_object("
            f"object_id={object_id}, scale_factor={scale_factor})"
        )
        result = rescale_object_common(
            scene=self.scene,
            object_id=object_id,
            scale_factor=scale_factor,
            object_type_name="ceiling object",
            asset_registry=self.asset_manager.registry,
        )
        return result.to_json()

    def _get_current_scene_state_impl(self) -> str:
        """Implementation for getting current scene state."""
        # Build room bounds info.
        min_x, min_y, max_x, max_y = self.room_bounds
        room_info = RoomBoundsInfo(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            ceiling_height=self.ceiling_height,
        )

        # Build ceiling objects info.
        ceiling_objects_info = []
        for obj in self.scene.get_objects_by_type(ObjectType.CEILING_MOUNTED):
            # Get ceiling-local position from world transform.
            # Ceiling is at fixed Z height, so X, Y come from translation.
            translation = obj.transform.translation()
            pos_x = float(translation[0])
            pos_y = float(translation[1])

            # Extract yaw rotation from transform.
            rpy = RollPitchYaw(obj.transform.rotation())
            rot_deg = math.degrees(rpy.yaw_angle())

            # Get dimensions from bounding box.
            width = float(obj.bbox_max[0] - obj.bbox_min[0])
            depth = float(obj.bbox_max[1] - obj.bbox_min[1])
            height = float(obj.bbox_max[2] - obj.bbox_min[2])

            ceiling_objects_info.append(
                CeilingObjectInfo(
                    object_id=str(obj.object_id),
                    description=obj.description,
                    position_x=pos_x,
                    position_y=pos_y,
                    rotation_degrees=rot_deg,
                    dimensions=BoundingBox3D(width=width, depth=depth, height=height),
                )
            )

        result = CeilingSceneStateResult(
            room_bounds=room_info,
            ceiling_objects=ceiling_objects_info,
            object_count=len(ceiling_objects_info),
        )

        return result.to_json()

    def _list_available_assets_impl(self) -> str:
        """Implementation for listing available ceiling assets."""
        all_assets = self.asset_manager.list_available_assets()
        ceiling_assets = [
            asset
            for asset in all_assets
            if asset.object_type == ObjectType.CEILING_MOUNTED
        ]

        assets_info = []
        for asset in ceiling_assets:
            assets_info.append(
                AssetInfo(
                    asset_id=str(asset.object_id),
                    name=asset.name,
                    description=asset.description,
                    object_type=asset.object_type.value,
                    dimensions=BoundingBox3D(
                        width=float(asset.bbox_max[0] - asset.bbox_min[0]),
                        depth=float(asset.bbox_max[1] - asset.bbox_min[1]),
                        height=float(asset.bbox_max[2] - asset.bbox_min[2]),
                    ),
                )
            )

        result = AvailableAssetsResult(
            assets=assets_info,
            count=len(assets_info),
        )

        return result.to_json()

    def _create_placement_failure_result(
        self,
        asset_id: str,
        position_x: float,
        position_y: float,
        rotation_deg: float,
        message: str,
        error_type: CeilingErrorType | None,
    ) -> str:
        """Create a placement failure result."""
        result = PlaceCeilingObjectResult(
            success=False,
            asset_id=asset_id,
            object_id="",
            message=message,
            position_x=position_x,
            position_y=position_y,
            rotation_degrees=rotation_deg,
            error_type=error_type,
        )
        return result.to_json()
