"""Tools for wall-mounted object generation and placement.

This module provides tools for generating and placing wall-mounted objects
(mirrors, artwork, shelves, clocks, etc.) on wall surfaces.
"""

import json
import logging
import math

from typing import Any

import numpy as np

from agents import function_tool
from omegaconf import DictConfig
from pydrake.math import RollPitchYaw

from scenecode.agent_utils.action_logger import log_scene_action
from scenecode.agent_utils.asset_manager import AssetGenerationRequest, AssetManager
from scenecode.agent_utils.loop_detector import LoopDetector
from scenecode.agent_utils.placement_noise import (
    PlacementNoiseMode,
    apply_wall_placement_noise,
)
from scenecode.agent_utils.rescale_helpers import rescale_object_common
from scenecode.agent_utils.response_datatypes import (
    AssetGenerationResult as AssetGenerationResultDTO,
    AssetInfo,
    BoundingBox3D,
    GeneratedAsset,
    Position3D,
    Rotation3D,
)
from scenecode.agent_utils.room import (
    ObjectType,
    PlacementInfo,
    RoomScene,
    SceneObject,
    clone_scene_object,
    UniqueID,
)
from scenecode.wall_agents.tools.response_dataclasses import (
    AvailableAssetsResult,
    ExcludedRegionInfo,
    PlaceWallObjectResult,
    WallErrorType,
    WallObjectInfo,
    WallOperationResult,
    WallSceneStateResult,
    WallSurfaceInfo,
)
from scenecode.wall_agents.tools.wall_surface import WallSurface

console_logger = logging.getLogger(__name__)


class WallTools:
    """Tools for wall-mounted object generation and placement.

    Provides tools for generating 3D wall-mounted objects and placing them on
    wall surfaces using SE(2) coordinates (position along wall, height, rotation).
    """

    def __init__(
        self,
        scene: RoomScene,
        wall_surfaces: list[WallSurface],
        asset_manager: AssetManager,
        cfg: DictConfig,
    ):
        """Initialize wall tools.

        Args:
            scene: RoomScene instance to manipulate.
            wall_surfaces: List of wall surfaces for placement.
            asset_manager: Asset manager for generating 3D assets.
            cfg: Configuration object containing loop detection and noise settings.
        """
        self.scene = scene
        self.wall_surfaces = wall_surfaces
        self.asset_manager = asset_manager
        self.cfg = cfg

        # Index surfaces by ID for O(1) lookup when placing objects.
        self.surfaces_by_id: dict[str, WallSurface] = {
            str(s.surface_id): s for s in wall_surfaces
        }

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
        self._place_wall_object_impl = loop_detector(self._place_wall_object_impl)
        self._move_wall_object_impl = loop_detector(self._move_wall_object_impl)
        self._remove_wall_object_impl = loop_detector(self._remove_wall_object_impl)

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
        """Create wall-specific error response for loop detection."""
        # Extract identifiers from kwargs if available.
        object_id = kwargs.get("object_id", kwargs.get("asset_id", ""))

        result = WallOperationResult(
            success=False,
            message=(
                f"Loop detected: {attempt_count} similar attempts on '{method_name}'. "
                f"Try a different approach."
            ),
            object_id=object_id,
            error_type=WallErrorType.LOOP_DETECTED,
        )
        return result.to_json()

    def _create_tool_closures(self) -> dict[str, Any]:
        """Create tool closures that capture wall context."""

        @function_tool
        def generate_wall_assets(
            object_descriptions: list[str],
            short_names: list[str],
            desired_dimensions: list[list[float]],
            style_context: str | None = None,
        ) -> str:
            """Generate 3D wall-mounted assets from text descriptions.

            Creates wall objects like mirrors, artwork, shelves, clocks, etc.

            Dimensions for wall objects:
            - width: extent along wall (horizontal)
            - depth: how far object protrudes from wall (typically thin)
            - height: vertical extent

            Args:
                object_descriptions: List of object descriptions
                    (e.g., "Rectangular framed mirror", "Abstract canvas painting").
                short_names: List of filesystem-safe names
                    (e.g., "mirror_1", "painting_abstract").
                desired_dimensions: List of [width, depth, height] in meters.
                    Wall objects are typically thin.
                style_context: Optional style context for visual consistency
                    (e.g., "modern minimalist", "rustic farmhouse").

            Returns:
                JSON with IDs and details of created wall object assets.
            """
            console_logger.info(
                f"Tool called: generate_wall_assets("
                f"object_descriptions={object_descriptions}, "
                f"short_names={short_names})"
            )
            request = AssetGenerationRequest(
                object_descriptions=object_descriptions,
                short_names=short_names,
                object_type=ObjectType.WALL_MOUNTED,
                desired_dimensions=desired_dimensions,
                style_context=style_context,
                scene_id=self.scene.scene_dir.name,
            )
            return self._generate_assets_impl(request)

        @function_tool
        def list_wall_surfaces() -> str:
            """List all wall surfaces available for placement.

            Returns info about each wall including dimensions and excluded
            regions (doors/windows where objects cannot be placed).

            Returns:
                JSON with list of wall surfaces and their properties.
            """
            console_logger.info("Tool called: list_wall_surfaces")
            surfaces_info = []
            for surface in self.wall_surfaces:
                excluded = [
                    ExcludedRegionInfo(x_min=r[0], z_min=r[1], x_max=r[2], z_max=r[3])
                    for r in surface.excluded_regions
                ]
                surfaces_info.append(
                    WallSurfaceInfo(
                        surface_id=str(surface.surface_id),
                        wall_id=surface.wall_id,
                        wall_direction=surface.wall_direction.value,
                        length=surface.length,
                        height=surface.height,
                        excluded_regions=excluded,
                    )
                )

            result = {
                "num_surfaces": len(surfaces_info),
                "surfaces": [s.__dict__ for s in surfaces_info],
            }
            return json.dumps(result, indent=2, default=str)

        @function_tool
        def place_wall_object(
            asset_id: str,
            wall_surface_id: str,
            position_x: float,
            position_z: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Place a wall-mounted object on a wall surface.

            The object is placed using 2D coordinates on the wall plane.
            The system automatically converts to 3D world coordinates.

            Each placement gets a unique ID so you can move or remove it later.
            The same wall object can be placed multiple times.

            Coordinate system:
            - X: position along wall (meters from wall start, left to right)
            - Z: height on wall (meters from floor)
            - Rotation: degrees around wall normal (tilting the object)

            IMPORTANT: Check wall bounds and excluded regions (doors/windows)
            before placing. Use list_wall_surfaces() to see wall dimensions
            and excluded regions.
            Use the 'observe_scene' tool to see wall surface coordinates.

            Args:
                asset_id: ID of the wall asset to place.
                wall_surface_id: ID of the wall surface to place on.
                    Use list_wall_surfaces() to see available surfaces.
                position_x: Position along wall (meters from wall start).
                position_z: Height on wall (meters from floor).
                rotation_degrees: Rotation around wall normal (degrees).
                    Typically 0 for wall objects. Positive = counterclockwise
                    when looking at wall from inside room.

            Returns:
                Placement result with world pose and wall-relative pose.
            """
            return self._place_wall_object_impl(
                asset_id=asset_id,
                wall_surface_id=wall_surface_id,
                position_x=position_x,
                position_z=position_z,
                rotation_degrees=rotation_degrees,
            )

        @function_tool
        def move_wall_object(
            object_id: str,
            wall_surface_id: str,
            position_x: float,
            position_z: float,
            rotation_degrees: float = 0.0,
        ) -> str:
            """Move an existing wall object to a new position.

            Use this to reposition wall objects or move them between walls.
            You need the object ID from when you placed it or from
            'get_current_scene_state' tool.

            Args:
                object_id: ID of the wall object to move.
                wall_surface_id: ID of the target wall surface.
                    Can be same or different from current wall.
                position_x: New position along wall (meters from wall start).
                position_z: New height on wall (meters from floor).
                rotation_degrees: New rotation around wall normal (degrees).

            Returns:
                Result of the move operation with new world pose.
            """
            return self._move_wall_object_impl(
                object_id=object_id,
                wall_surface_id=wall_surface_id,
                position_x=position_x,
                position_z=position_z,
                rotation_degrees=rotation_degrees,
            )

        @function_tool
        def remove_wall_object(object_id: str) -> str:
            """Remove a wall object from the scene.

            Args:
                object_id: ID of the wall object to remove.

            Returns:
                Result of the removal operation.
            """
            return self._remove_wall_object_impl(object_id=object_id)

        @function_tool
        def get_current_scene_state() -> str:
            """Get current scene state for wall objects.

            Shows:
            - All wall surfaces with dimensions and excluded regions
            - Wall objects already placed with their positions

            Returns:
                Scene state with walls and placed wall objects.
            """
            console_logger.info("Tool called: get_current_scene_state")
            return self._get_current_scene_state_impl()

        @function_tool
        def list_available_assets() -> str:
            """List all available wall assets.

            Returns list of generated wall assets that can be placed.

            Returns:
                List of available wall assets with IDs and descriptions.
            """
            console_logger.info("Tool called: list_available_assets")
            return self._list_available_assets_impl()

        @function_tool
        def rescale_wall_object(object_id: str, scale_factor: float) -> str:
            """Resize a wall object by a uniform scale factor.

            Use when the object's shape/proportions are correct but size is wrong.
            For shape or proportion issues, remove and regenerate instead.

            IMPORTANT: This rescales the underlying ASSET (SDF file). All instances
            of the same asset will be affected. This is usually what you want -
            if one mirror is too small, all mirrors of that type are too small.

            Args:
                object_id: ID of the wall object to rescale.
                scale_factor: Scale multiplier (e.g., 1.5 = 50% larger,
                    0.8 = 20% smaller). Must be positive and not 1.0.

            Returns:
                JSON with rescale result including new dimensions.
            """
            return self._rescale_wall_object_impl(
                object_id=object_id,
                scale_factor=scale_factor,
            )

        return {
            "generate_wall_assets": generate_wall_assets,
            "list_wall_surfaces": list_wall_surfaces,
            "place_wall_object": place_wall_object,
            "move_wall_object": move_wall_object,
            "remove_wall_object": remove_wall_object,
            "rescale_wall_object": rescale_wall_object,
            "get_current_scene_state": get_current_scene_state,
            "list_available_assets": list_available_assets,
        }

    def _generate_assets_impl(self, request: AssetGenerationRequest) -> str:
        """Implementation for generating wall assets."""
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
                message=f"Successfully generated {len(generated_assets)} wall assets.",
            ).to_json()

        except Exception as e:
            console_logger.error(f"Error generating wall assets: {e}", exc_info=True)
            return AssetGenerationResultDTO(
                success=False,
                assets=[],
                message=f"Asset generation failed: {str(e)}",
            ).to_json()

    @log_scene_action
    def _place_wall_object_impl(
        self,
        asset_id: str,
        wall_surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for placing wall object on wall surface."""
        console_logger.info(
            f"Tool called: place_wall_object("
            f"asset_id={asset_id}, wall_surface_id={wall_surface_id}, "
            f"position_x={position_x}, position_z={position_z}, "
            f"rotation_degrees={rotation_degrees})"
        )

        try:
            # Validate wall_surface_id exists.
            surface = self.surfaces_by_id.get(wall_surface_id)
            if surface is None:
                available_ids = list(self.surfaces_by_id.keys())
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    wall_surface_id=wall_surface_id,
                    position_x=position_x,
                    position_z=position_z,
                    rotation_deg=rotation_degrees,
                    message=(
                        f"Invalid wall_surface_id: {wall_surface_id}. "
                        f"Available surfaces: {available_ids}"
                    ),
                    error_type=WallErrorType.SURFACE_NOT_FOUND,
                )

            # Convert string ID to UniqueID.
            try:
                unique_id = UniqueID(asset_id)
            except Exception:
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    wall_surface_id=wall_surface_id,
                    position_x=position_x,
                    position_z=position_z,
                    rotation_deg=rotation_degrees,
                    message=f"Invalid asset ID format: {asset_id}",
                    error_type=WallErrorType.ASSET_NOT_FOUND,
                )

            # Get asset from registry.
            original_asset = self.asset_manager.get_asset_by_id(unique_id)
            if not original_asset:
                all_assets = self.asset_manager.list_available_assets()
                available_assets = [
                    asset
                    for asset in all_assets
                    if asset.object_type == ObjectType.WALL_MOUNTED
                ]
                available_ids = [str(a.object_id) for a in available_assets]
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    wall_surface_id=wall_surface_id,
                    position_x=position_x,
                    position_z=position_z,
                    rotation_deg=rotation_degrees,
                    message=(
                        f"Asset {asset_id} not found. Available wall assets: "
                        f"{available_ids}"
                    ),
                    error_type=WallErrorType.ASSET_NOT_FOUND,
                )

            # Get object dimensions.
            object_width = float(
                original_asset.bbox_max[0] - original_asset.bbox_min[0]
            )
            object_height = float(
                original_asset.bbox_max[2] - original_asset.bbox_min[2]
            )

            # Apply placement noise for realistic variation.
            noisy_x, noisy_z, noisy_rotation = apply_wall_placement_noise(
                position_x=position_x,
                position_z=position_z,
                rotation_deg=rotation_degrees,
                position_along_wall_std_meters=(
                    self.active_noise_profile.position_along_wall_std_meters
                ),
                position_height_std_meters=(
                    self.active_noise_profile.position_height_std_meters
                ),
                rotation_std_degrees=self.active_noise_profile.rotation_std_degrees,
            )

            # Validate noisy position and dimensions fit on wall.
            # Must check AFTER noise to catch positions pushed beyond bounds.
            is_valid, error_msg = surface.check_object_bounds(
                position_x=noisy_x,
                position_z=noisy_z,
                object_width=object_width,
                object_height=object_height,
            )
            if not is_valid:
                return self._create_placement_failure_result(
                    asset_id=asset_id,
                    wall_surface_id=wall_surface_id,
                    position_x=position_x,
                    position_z=position_z,
                    rotation_deg=rotation_degrees,
                    message=error_msg,
                    error_type=WallErrorType.OVERLAPS_OPENING,
                )

            console_logger.info(
                f"Placing wall object {asset_id} ({original_asset.name}) at wall "
                f"position ({noisy_x:.3f}, {noisy_z:.3f}), rotation {noisy_rotation:.1f}°"
            )

            # Convert wall SE(2) to world SE(3).
            world_transform = surface.to_world_pose(
                position_x=noisy_x,
                position_z=noisy_z,
                rotation_deg=noisy_rotation,
            )

            # Create new scene object with unique ID.
            object_id = self.scene.generate_unique_id(original_asset.name)
            scene_object = clone_scene_object(
                original_asset,
                object_id=object_id,
                object_type=ObjectType.WALL_MOUNTED,
                transform=world_transform,
                placement_info=PlacementInfo(
                    parent_surface_id=surface.surface_id,
                    position_2d=np.array([noisy_x, noisy_z]),
                    rotation_2d=math.radians(noisy_rotation),
                    placement_method="wall_placement",
                ),
            )

            # Add to scene.
            self.scene.add_object(scene_object)

            # Extract world pose for response.
            world_position = world_transform.translation()
            world_rpy = RollPitchYaw(world_transform.rotation())

            console_logger.info(
                f"Successfully placed wall object '{original_asset.name}' as "
                f"{object_id} on wall {wall_surface_id}"
            )

            # Create success result.
            result = PlaceWallObjectResult(
                success=True,
                message=(
                    f"Successfully placed '{original_asset.name}' on wall at "
                    f"({noisy_x:.3f}m along wall, {noisy_z:.3f}m height)"
                ),
                asset_id=asset_id,
                object_id=str(object_id),
                wall_surface_id=wall_surface_id,
                position_x=noisy_x,
                position_z=noisy_z,
                rotation_deg=noisy_rotation,
                world_position=Position3D(
                    x=float(world_position[0]),
                    y=float(world_position[1]),
                    z=float(world_position[2]),
                ),
                world_rotation=Rotation3D(
                    roll=math.degrees(world_rpy.roll_angle()),
                    pitch=math.degrees(world_rpy.pitch_angle()),
                    yaw=math.degrees(world_rpy.yaw_angle()),
                ),
            )

            return result.to_json()

        except Exception as e:
            console_logger.error(f"Error placing wall object: {e}", exc_info=True)
            return self._create_placement_failure_result(
                asset_id=asset_id,
                wall_surface_id=wall_surface_id,
                position_x=position_x,
                position_z=position_z,
                rotation_deg=rotation_degrees,
                message=f"Unexpected error: {str(e)}",
                error_type=None,
            )

    @log_scene_action
    def _move_wall_object_impl(
        self,
        object_id: str,
        wall_surface_id: str,
        position_x: float,
        position_z: float,
        rotation_degrees: float = 0.0,
        **kwargs,
    ) -> str:
        """Implementation for moving wall object to new position."""
        console_logger.info(
            f"Tool called: move_wall_object("
            f"object_id={object_id}, wall_surface_id={wall_surface_id}, "
            f"position_x={position_x}, position_z={position_z}, "
            f"rotation_degrees={rotation_degrees})"
        )

        try:
            # Get the existing object.
            unique_id = UniqueID(object_id)
            scene_object = self.scene.get_object(unique_id)
            if scene_object is None:
                return WallOperationResult(
                    success=False,
                    message=f"Object {object_id} not found in scene.",
                    object_id=object_id,
                    error_type=WallErrorType.OBJECT_NOT_FOUND,
                ).to_json()

            # Verify it's a wall-mounted object.
            if scene_object.object_type != ObjectType.WALL_MOUNTED:
                return WallOperationResult(
                    success=False,
                    message=(
                        f"Object {object_id} is not a wall-mounted object "
                        f"(type: {scene_object.object_type.value})."
                    ),
                    object_id=object_id,
                    error_type=WallErrorType.INVALID_OPERATION,
                ).to_json()

            # Validate wall_surface_id exists.
            surface = self.surfaces_by_id.get(wall_surface_id)
            if surface is None:
                available_ids = list(self.surfaces_by_id.keys())
                return WallOperationResult(
                    success=False,
                    message=(
                        f"Invalid wall_surface_id: {wall_surface_id}. "
                        f"Available surfaces: {available_ids}"
                    ),
                    object_id=object_id,
                    error_type=WallErrorType.SURFACE_NOT_FOUND,
                ).to_json()

            # Get object dimensions.
            object_width = float(scene_object.bbox_max[0] - scene_object.bbox_min[0])
            object_height = float(scene_object.bbox_max[2] - scene_object.bbox_min[2])

            # Validate position and dimensions fit on wall.
            is_valid, error_msg = surface.check_object_bounds(
                position_x=position_x,
                position_z=position_z,
                object_width=object_width,
                object_height=object_height,
            )
            if not is_valid:
                return WallOperationResult(
                    success=False,
                    message=error_msg,
                    object_id=object_id,
                    error_type=WallErrorType.OVERLAPS_OPENING,
                ).to_json()

            # Convert wall SE(2) to world SE(3).
            world_transform = surface.to_world_pose(
                position_x=position_x,
                position_z=position_z,
                rotation_deg=rotation_degrees,
            )

            # Update the object's transform and placement info.
            scene_object.transform = world_transform
            scene_object.placement_info = PlacementInfo(
                parent_surface_id=surface.surface_id,
                position_2d=np.array([position_x, position_z]),
                rotation_2d=math.radians(rotation_degrees),
                placement_method="wall_placement",
            )

            console_logger.info(
                f"Moved wall object '{scene_object.name}' to "
                f"({position_x:.3f}, {position_z:.3f}) on wall {wall_surface_id}"
            )

            return WallOperationResult(
                success=True,
                message=(
                    f"Successfully moved '{scene_object.name}' to "
                    f"({position_x:.3f}m along wall, {position_z:.3f}m height)"
                ),
                object_id=object_id,
            ).to_json()

        except Exception as e:
            console_logger.error(f"Error moving wall object: {e}", exc_info=True)
            return WallOperationResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                object_id=object_id,
                error_type=None,
            ).to_json()

    @log_scene_action
    def _remove_wall_object_impl(self, object_id: str, **kwargs) -> str:
        """Implementation for removing wall object from scene."""
        console_logger.info(f"Tool called: remove_wall_object(object_id={object_id})")

        try:
            unique_id = UniqueID(object_id)
            scene_object = self.scene.get_object(unique_id)
            if scene_object is None:
                return WallOperationResult(
                    success=False,
                    message=f"Object {object_id} not found in scene.",
                    object_id=object_id,
                    error_type=WallErrorType.OBJECT_NOT_FOUND,
                ).to_json()

            # Verify it's a wall-mounted object.
            if scene_object.object_type != ObjectType.WALL_MOUNTED:
                return WallOperationResult(
                    success=False,
                    message=(
                        f"Object {object_id} is not a wall-mounted object "
                        f"(type: {scene_object.object_type.value})."
                    ),
                    object_id=object_id,
                    error_type=WallErrorType.INVALID_OPERATION,
                ).to_json()

            # Remove from scene.
            self.scene.remove_object(unique_id)

            console_logger.info(f"Removed wall object '{scene_object.name}'")

            return WallOperationResult(
                success=True,
                message=f"Successfully removed '{scene_object.name}'.",
                object_id=object_id,
            ).to_json()

        except Exception as e:
            console_logger.error(f"Error removing wall object: {e}", exc_info=True)
            return WallOperationResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                object_id=object_id,
                error_type=None,
            ).to_json()

    @log_scene_action
    def _rescale_wall_object_impl(
        self, object_id: str, scale_factor: float, **kwargs
    ) -> str:
        """Implementation for rescaling a wall object."""
        console_logger.info(
            f"Tool called: rescale_wall_object("
            f"object_id={object_id}, scale_factor={scale_factor})"
        )
        result = rescale_object_common(
            scene=self.scene,
            object_id=object_id,
            scale_factor=scale_factor,
            object_type_name="wall object",
            asset_registry=self.asset_manager.registry,
        )
        return result.to_json()

    def _get_current_scene_state_impl(self) -> str:
        """Implementation for getting current scene state."""
        # Build wall surfaces info.
        surfaces_info = []
        for surface in self.wall_surfaces:
            excluded = [
                ExcludedRegionInfo(x_min=r[0], z_min=r[1], x_max=r[2], z_max=r[3])
                for r in surface.excluded_regions
            ]
            surfaces_info.append(
                WallSurfaceInfo(
                    surface_id=str(surface.surface_id),
                    wall_id=surface.wall_id,
                    wall_direction=surface.wall_direction.value,
                    length=surface.length,
                    height=surface.height,
                    excluded_regions=excluded,
                )
            )

        # Build wall objects info.
        wall_objects_info = []
        for obj in self.scene.get_objects_by_type(ObjectType.WALL_MOUNTED):
            # Get wall-local position from placement_info.
            if obj.placement_info is None:
                raise RuntimeError(
                    f"Wall object '{obj.name}' ({obj.object_id}) has no placement_info. "
                    f"All wall objects must be placed via place_wall_object."
                )
            pos_x = float(obj.placement_info.position_2d[0])
            pos_z = float(obj.placement_info.position_2d[1])
            rot_deg = math.degrees(obj.placement_info.rotation_2d)
            surface_id = str(obj.placement_info.parent_surface_id)

            # Get dimensions from bounding box.
            width = float(obj.bbox_max[0] - obj.bbox_min[0])
            depth = float(obj.bbox_max[1] - obj.bbox_min[1])
            height = float(obj.bbox_max[2] - obj.bbox_min[2])

            wall_objects_info.append(
                WallObjectInfo(
                    object_id=str(obj.object_id),
                    description=obj.description,
                    wall_surface_id=surface_id,
                    position_x=pos_x,
                    position_z=pos_z,
                    rotation_deg=rot_deg,
                    dimensions=BoundingBox3D(width=width, depth=depth, height=height),
                )
            )

        result = WallSceneStateResult(
            wall_surfaces=surfaces_info,
            wall_objects=wall_objects_info,
            object_count=len(wall_objects_info),
        )

        return result.to_json()

    def _list_available_assets_impl(self) -> str:
        """Implementation for listing available wall assets."""
        all_assets = self.asset_manager.list_available_assets()
        wall_assets = [
            asset
            for asset in all_assets
            if asset.object_type == ObjectType.WALL_MOUNTED
        ]

        assets_info = []
        for asset in wall_assets:
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
        wall_surface_id: str,
        position_x: float,
        position_z: float,
        rotation_deg: float,
        message: str,
        error_type: WallErrorType | None,
    ) -> str:
        """Create a placement failure result."""
        result = PlaceWallObjectResult(
            success=False,
            asset_id=asset_id,
            object_id="",
            message=message,
            wall_surface_id=wall_surface_id,
            position_x=position_x,
            position_z=position_z,
            rotation_deg=rotation_deg,
            error_type=error_type,
        )
        return result.to_json()
