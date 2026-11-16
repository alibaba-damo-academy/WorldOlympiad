import logging
import math

from typing import Any

import numpy as np

from agents import function_tool
from omegaconf import DictConfig
from pydrake.all import RigidTransform, RollPitchYaw, RotationMatrix

from scenecode.agent_utils.action_logger import log_scene_action
from scenecode.agent_utils.room import RoomScene, SceneObject, UniqueID
from scenecode.furniture_agents.tools.response_dataclasses import (
    FacingCheckResult,
    FurnitureErrorType,
    Position3D,
    SceneStateResult,
    SimplifiedFurnitureInfo,
    SnapToObjectResult,
)
from scenecode.furniture_agents.tools.snapping_helpers import (
    resolve_collision_if_penetrating,
    select_and_execute_snap_algorithm,
)
from scenecode.utils.geometry_utils import (
    closest_point_on_aabb,
    compute_optimal_facing_yaw,
    ray_rectangle_intersection_2d,
)
from scenecode.utils.shape_analysis import is_circular_object

console_logger = logging.getLogger(__name__)

# Distance thresholds for snapping operations.
ALREADY_TOUCHING_THRESHOLD_M = 0.001  # 1mm threshold for contact detection.


class SceneTools:
    """Tools for scene state management."""

    def __init__(self, scene: RoomScene, cfg: DictConfig) -> None:
        """Initialize scene tools.

        Args:
            scene: RoomScene instance to manage.
            cfg: Configuration object with snap_to_object settings.
        """
        self.scene = scene
        self.cfg = cfg
        self.tools = self._create_tool_closures()

    def _create_tool_closures(self) -> dict[str, Any]:
        """Create closure-based tools that capture self."""

        @function_tool
        def get_current_scene_state() -> str:
            """See all furniture currently in the room with precise spatial data.

            Shows existing furniture with positions, rotations, and bounding boxes.
            Each object includes:
            - dimensions: Width, depth, height in meters (object's local frame)
            - world_bounds: Min/max corners showing exact space occupied in world

            Use this to check clearances, understand scale, and plan placements.

            Returns:
                List of furniture with IDs, positions, rotations, and bounding boxes.
            """
            return self._get_current_scene_impl()

        @function_tool
        def check_facing_tool(
            object_a_id: str, object_b_id: str, direction: str = "toward"
        ) -> str:
            """Verify furniture orientation relationships and get exact rotation angles.

            IMPORTANT: Use this tool to check facing relationships - it's MORE
            RELIABLE than visual assessment. Always verify orientation-critical
            furniture with this tool after placement or rotation.

            Checks whether object A is correctly oriented relative to object B,
            and provides the exact rotation needed for perfect alignment.

            Args:
                object_a_id: ID of the first object (the one being checked).
                object_b_id: ID of the second object (the target).
                direction: Direction to check. Valid values:
                    - "toward": Check if object A faces toward object B (default).
                      Use for: chairs→tables, sofas→TVs, seating→focal points.
                    - "away": Check if object A faces away from object B.
                      Use for: furniture against walls (drawers, shelves, washing
                      machines, desks with backs to walls).

            Returns:
                JSON with:
                - is_facing (bool): Whether the orientation is correct for the
                  specified direction. If false, rotation adjustment is required.
                - optimal_rotation_degrees (float): Absolute yaw rotation in degrees
                  for perfect alignment. Use this value directly with
                  move_furniture_tool() when is_facing=false.
                  Positive = counter-clockwise in top-down view.
                - current_rotation_degrees (float): Current absolute yaw of object A
            """
            return self._check_facing_impl(
                object_a_id=object_a_id,
                object_b_id=object_b_id,
                direction=direction,
            )

        @function_tool
        def snap_to_object_tool(
            object_id: str, target_id: str, orientation: str = "none"
        ) -> str:
            """CRITICAL: Eliminate gaps and align orientation between furniture.

            **PROBLEM**: Approximate placement creates visible gaps. Guessing exact
            positions AND rotations for chairs around tables requires 3-4 tool calls
            per chair (place → check_facing → rotate → snap).

            **SOLUTION**: One-step orient-and-snap. Automatically faces object toward/
            away from target, then snaps along facing direction. Reduces 3-4 calls to
            1 call per chair.

            **WHEN TO USE orientation PARAMETER**:
            - **orientation="toward"**: Chairs → tables, sofas → TVs, seating → focal points
              - Rotates chair to face table/target, then pushes forward until touching
              - Works for both round tables (face center) and rectangular (face edge)
              - Example: `snap_to_object_tool(chair_id, table_id, orientation="toward")`

            - **orientation="away"**: Furniture with backs/doors → walls
              - Rotates to face AWAY from wall, then pushes back until touching
              - For: wardrobes, dressers, cabinets, appliances, shelving
              - Example: `snap_to_object_tool(wardrobe_id, wall_id, orientation="away")`

            - **orientation="none"** (default): No orientation change
              - Only translation, rotation preserved

            Args:
                object_id: ID of object to move (must not be immutable).
                target_id: ID of wall or object to snap to (stays in place).
                orientation: Orientation mode. Options:
                    - "toward": Face target, then snap forward (chairs → tables)
                    - "away": Face away from target, snap back (furniture → walls)
                    - "none": No orientation change (default, backward compatible)

            Returns:
                JSON with success status, positions, distance moved, and rotation info.
            """
            return self._snap_to_object_impl(
                object_id=object_id,
                target_id=target_id,
                orientation=orientation,
            )

        return {
            "get_current_scene_state": get_current_scene_state,
            "check_facing_tool": check_facing_tool,
            "snap_to_object_tool": snap_to_object_tool,
        }

    def _get_current_scene_impl(self) -> str:
        """Implementation for get_current_scene_state."""
        console_logger.info("Tool called: get_current_scene_state")
        objects = [
            SimplifiedFurnitureInfo.from_scene_object(obj)
            for obj in self.scene.objects.values()
        ]

        result = SceneStateResult(
            success=True, furniture_count=len(objects), objects=objects
        )
        return result.to_json()

    def _check_facing_impl(
        self, object_a_id: str, object_b_id: str, direction: str = "toward"
    ) -> str:
        """Implementation for check_facing_tool.

        Args:
            object_a_id: ID of the first object (the one being checked).
            object_b_id: ID of the second object (the target).
            direction: Direction to check ("toward" or "away").

        Returns:
            JSON string with FacingCheckResult.
        """
        console_logger.info(
            f"Tool called: check_facing_tool({object_a_id}, {object_b_id}, "
            f"direction={direction})"
        )

        # Validate direction parameter.
        if direction not in ["toward", "away"]:
            console_logger.error(f"Invalid direction parameter: {direction}")
            return FacingCheckResult(
                success=False,
                object_a_id=object_a_id,
                object_b_id=object_b_id,
                is_facing=False,
                optimal_rotation_degrees=0.0,
                current_rotation_degrees=0.0,
                message=(
                    f"Invalid direction parameter: '{direction}'. Valid options "
                    f"are 'toward' or 'away'. Use 'toward' to check if object "
                    f"faces another object, or 'away' to check if object faces "
                    f"away from another object (e.g., furniture against walls)."
                ),
            ).to_json()

        # Convert string IDs to UniqueID (UniqueID is a subclass of str).
        obj_a_uid = UniqueID(object_a_id)
        obj_b_uid = UniqueID(object_b_id)

        # Get objects from scene.
        obj_a = self.scene.get_object(obj_a_uid)
        if obj_a is None:
            console_logger.error(f"Object A not found: {object_a_id}")
            return FacingCheckResult(
                success=False,
                object_a_id=object_a_id,
                object_b_id=object_b_id,
                is_facing=False,
                optimal_rotation_degrees=0.0,
                current_rotation_degrees=0.0,
                message=f"Object A with ID {object_a_id} not found in scene",
            ).to_json()

        obj_b = self.scene.get_object(obj_b_uid)
        if obj_b is None:
            console_logger.error(f"Object B not found: {object_b_id}")
            return FacingCheckResult(
                success=False,
                object_a_id=object_a_id,
                object_b_id=object_b_id,
                is_facing=False,
                optimal_rotation_degrees=0.0,
                current_rotation_degrees=0.0,
                message=f"Object B with ID {object_b_id} not found in scene",
            ).to_json()

        # Check that both objects have bounding boxes.
        if obj_a.bbox_min is None or obj_a.bbox_max is None:
            console_logger.error(f"Object A lacks bounding box: {object_a_id}")
            return FacingCheckResult(
                success=False,
                object_a_id=object_a_id,
                object_b_id=object_b_id,
                is_facing=False,
                optimal_rotation_degrees=0.0,
                current_rotation_degrees=0.0,
                message=f"Object A ({obj_a.name}) lacks bounding box data",
            ).to_json()

        if obj_b.bbox_min is None or obj_b.bbox_max is None:
            console_logger.error(f"Object B lacks bounding box: {object_b_id}")
            return FacingCheckResult(
                success=False,
                object_a_id=object_a_id,
                object_b_id=object_b_id,
                is_facing=False,
                optimal_rotation_degrees=0.0,
                current_rotation_degrees=0.0,
                message=f"Object B ({obj_b.name}) lacks bounding box data",
            ).to_json()

        # Get world-space bounding box for object B.
        world_bounds_b = obj_b.compute_world_bounds()
        if world_bounds_b is None:
            console_logger.error(
                f"Failed to compute world bounds for object B: {object_b_id}"
            )
            return FacingCheckResult(
                success=False,
                object_a_id=object_a_id,
                object_b_id=object_b_id,
                is_facing=False,
                optimal_rotation_degrees=0.0,
                current_rotation_degrees=0.0,
                message=f"Failed to compute world bounds for object B ({obj_b.name})",
            ).to_json()

        world_bbox_min_b, world_bbox_max_b = world_bounds_b

        # Get object A's position and rotation.
        origin_a = obj_a.transform.translation()
        rotation_a = obj_a.transform.rotation()
        rpy_a = rotation_a.ToRollPitchYaw()
        yaw_a_rad = rpy_a.yaw_angle()
        yaw_a_deg = math.degrees(yaw_a_rad)

        # Transform forward direction from object A's local frame to world frame.
        # For "away", use -y direction (back of object faces target).
        local_forward = (
            np.array([0.0, -1.0, 0.0])
            if direction == "away"
            else np.array([0.0, 1.0, 0.0])
        )

        world_forward = rotation_a @ local_forward

        # Normalize direction vector.
        world_forward_normalized = world_forward / np.linalg.norm(world_forward)

        # Perform 2D ray-rectangle intersection test in XY plane.
        # This ignores Z-height differences, which is correct for horizontal
        # facing relationships (e.g., chair at different height than table).
        is_facing = ray_rectangle_intersection_2d(
            ray_origin_2d=origin_a[:2],
            ray_direction_2d=world_forward_normalized[:2],
            rect_min_2d=world_bbox_min_b[:2],
            rect_max_2d=world_bbox_max_b[:2],
        )

        # Determine target point based on object B's shape.
        # Circular objects (round tables): face CENTER for uniform inward facing.
        # Rectangular objects: face CLOSEST AABB POINT for nearest-edge alignment.
        target_point = self._compute_orientation_target_point(obj=obj_a, target=obj_b)

        # Compute optimal yaw rotation.
        # For "toward": face the target point.
        # For "away": face away from the target point (add 180°).
        optimal_rotation_base = compute_optimal_facing_yaw(
            origin_a=origin_a,
            target_point=target_point,
        )

        if direction == "away":
            # Add 180° to point away from target.
            optimal_rotation_delta = optimal_rotation_base + 180.0
            # Normalize to [-180, 180] range.
            if optimal_rotation_delta > 180.0:
                optimal_rotation_delta -= 360.0
        else:
            optimal_rotation_delta = optimal_rotation_base

        console_logger.info(
            f"Facing check result: is_facing={is_facing}, "
            f"optimal_rotation={optimal_rotation_delta:.1f}°, direction={direction}"
        )

        # Create informative message based on facing status and direction.
        direction_desc = "toward" if direction == "toward" else "away from"
        if is_facing:
            action_message = (
                f"✓ {obj_a.name} is facing {direction_desc} {obj_b.name}. "
                "Current alignment is acceptable."
            )
        else:
            action_message = (
                f"⚠️ {obj_a.name} is NOT facing {direction_desc} {obj_b.name}. "
                f"To align perfectly, use move_furniture_tool('"
                f"{object_a_id}', rotation_yaw={optimal_rotation_delta:.1f}). "
                f"Consider if this alignment is functionally "
                f"desirable for this furniture type."
            )

        return FacingCheckResult(
            success=True,
            object_a_id=object_a_id,
            object_b_id=object_b_id,
            is_facing=bool(is_facing),
            optimal_rotation_degrees=float(optimal_rotation_delta),
            current_rotation_degrees=float(yaw_a_deg),
            message=action_message,
        ).to_json()

    def _compute_orientation_target_point(
        self,
        obj: SceneObject,
        target: SceneObject,
    ) -> np.ndarray:
        """Compute target point for orientation based on target shape.

        For circular objects (tables, chairs), uses the center point.
        For rectangular objects (desks, wardrobes), uses the closest AABB point.

        Args:
            obj: Object that will be oriented.
            target: Target object to orient toward/away from.

        Returns:
            3D point in world coordinates to use for orientation computation.
        """
        is_circular = is_circular_object(target, self.cfg)

        if is_circular:
            console_logger.info(
                f"{target.name} is circular, using center as target point"
            )
            return target.transform.translation()

        # Rectangular: use closest AABB point.
        target_bounds = target.compute_world_bounds()
        if target_bounds is None:
            console_logger.warning(
                f"Target {target.name} has no bounds for orientation"
            )
            return target.transform.translation()

        target_bbox_min, target_bbox_max = target_bounds
        console_logger.info(
            f"{target.name} is rectangular, using closest AABB point as target"
        )
        return closest_point_on_aabb(
            point=obj.transform.translation(),
            bbox_min=target_bbox_min,
            bbox_max=target_bbox_max,
        )

    def _compute_and_apply_orientation(
        self,
        obj: SceneObject,
        target: SceneObject,
        orientation: str,
        original_rpy: RollPitchYaw,
    ) -> tuple[bool, float]:
        """Compute and apply orientation to object based on target.

        Args:
            obj: Object to rotate (modified in-place).
            target: Target to face toward or away from.
            orientation: "toward", "away", or "none".
            original_rpy: Original roll-pitch-yaw of object.

        Returns:
            Tuple of (orientation_applied, new_yaw_rad).
        """
        if orientation == "none":
            return (False, original_rpy.yaw_angle())

        # Determine target point based on target's shape.
        target_point = self._compute_orientation_target_point(obj=obj, target=target)

        # Compute optimal yaw.
        optimal_yaw_deg = compute_optimal_facing_yaw(
            origin_a=obj.transform.translation(),
            target_point=target_point,
        )

        # For "away", add 180° to face away from target.
        if orientation == "away":
            optimal_yaw_deg += 180.0
            if optimal_yaw_deg > 180.0:
                optimal_yaw_deg -= 360.0

        new_yaw_rad = math.radians(optimal_yaw_deg)

        # Apply rotation.
        new_rotation = RotationMatrix(
            RollPitchYaw(
                roll=original_rpy.roll_angle(),
                pitch=original_rpy.pitch_angle(),
                yaw=new_yaw_rad,
            )
        )
        obj.transform = RigidTransform(R=new_rotation, p=obj.transform.translation())

        console_logger.info(
            f"Applied orientation '{orientation}': "
            f"{math.degrees(original_rpy.yaw_angle()):.1f}° → "
            f"{math.degrees(new_yaw_rad):.1f}°"
        )

        return (True, new_yaw_rad)

    def _create_snap_error(
        self,
        object_id: str,
        target_id: str,
        error_type: FurnitureErrorType,
        message: str,
        suggested_action: str,
    ) -> str:
        """Create standardized snap error result.

        Args:
            object_id: ID of the object being snapped.
            target_id: ID of the target object.
            error_type: Type of error that occurred.
            message: Human-readable error message.
            suggested_action: Suggested action for the agent.

        Returns:
            JSON string with error result.
        """
        console_logger.error(message)
        return SnapToObjectResult(
            success=False,
            message=message,
            object_id=object_id,
            target_id=target_id,
            error_type=error_type,
            suggested_action=suggested_action,
        ).to_json()

    def _validate_snap_inputs(
        self,
        object_id: str,
        target_id: str,
        orientation: str,
    ) -> tuple[SceneObject, SceneObject] | str:
        """Validate inputs for snap_to_object operation.

        Args:
            object_id: ID of object to move.
            target_id: ID of target object.
            orientation: Orientation mode ("toward", "away", or "none").

        Returns:
            Tuple of (obj, target) if validation passes, or error JSON string if it fails.
        """
        # Convert string IDs to UniqueID.
        obj_uid = UniqueID(object_id)
        target_uid = UniqueID(target_id)

        # Get objects from scene.
        obj = self.scene.get_object(obj_uid)
        if obj is None:
            return self._create_snap_error(
                object_id=object_id,
                target_id=target_id,
                error_type=FurnitureErrorType.OBJECT_NOT_FOUND,
                message=f"Object '{object_id}' not found in scene",
                suggested_action=(
                    "Call get_current_scene_state() to see all available objects and "
                    "their IDs, then retry with the correct object_id"
                ),
            )

        target = self.scene.get_object(target_uid)
        if target is None:
            return self._create_snap_error(
                object_id=object_id,
                target_id=target_id,
                error_type=FurnitureErrorType.OBJECT_NOT_FOUND,
                message=f"Target '{target_id}' not found in scene",
                suggested_action=(
                    "Call get_current_scene_state() to see all available objects "
                    "(including walls) and their IDs, then retry with the correct "
                    "target_id"
                ),
            )

        # Check if object is immutable.
        if obj.immutable:
            return self._create_snap_error(
                object_id=object_id,
                target_id=target_id,
                error_type=FurnitureErrorType.IMMUTABLE_OBJECT,
                message=f"Cannot snap {obj.name}: architectural element is immutable",
                suggested_action=(
                    "Only movable furniture can be snapped. Retry with a different "
                    "object_id"
                ),
            )

        # Validate orientation parameter.
        if orientation not in ["toward", "away", "none"]:
            return self._create_snap_error(
                object_id=object_id,
                target_id=target_id,
                error_type=FurnitureErrorType.INVALID_POSITION,
                message=(
                    f"Invalid orientation parameter: '{orientation}'. "
                    f"Valid options are 'toward', 'away', or 'none'."
                ),
                suggested_action=(
                    "Use orientation='toward' for chairs facing tables, "
                    "orientation='away' for furniture facing away from walls, "
                    "or orientation='none' for no orientation change"
                ),
            )

        return (obj, target)

    @log_scene_action
    def _snap_to_object_impl(
        self, object_id: str, target_id: str, orientation: str = "none"
    ) -> str:
        """Implementation for snap_to_object_tool.

        Args:
            object_id: ID of object to move.
            target_id: ID of target object (stays in place).
            orientation: Orientation mode for snapping. Valid values:
                - "toward": Orient object to face target, then snap along facing axis.
                - "away": Orient object to face away from target, then snap along axis.
                - "none": No orientation applied (default, backward compatible).

        Returns:
            JSON string with SnapToObjectResult.
        """
        console_logger.info(
            f"Tool called: snap_to_object_tool({object_id}, {target_id}, "
            f"orientation={orientation})"
        )

        # Validate inputs.
        validation_result = self._validate_snap_inputs(
            object_id=object_id, target_id=target_id, orientation=orientation
        )
        if isinstance(validation_result, str):
            return validation_result
        obj, target = validation_result

        # Get original position and rotation.
        original_pos = obj.transform.translation()
        original_rotation = obj.transform.rotation()
        original_rpy = original_rotation.ToRollPitchYaw()

        # ===== Resolve Collisions (BEFORE orientation). =====
        # Conservative AABB push-out to separate objects if penetrating.
        # This ensures rotation won't reintroduce collision.
        resolve_collision_if_penetrating(
            obj=obj,
            target=target,
            cfg=self.cfg,
            wall_normals=self.scene.room_geometry.wall_normals,
        )

        # ===== Apply Orientation Parameter. =====
        # Orient object toward/away from target if orientation is specified.
        orientation_applied, new_yaw_rad = self._compute_and_apply_orientation(
            obj=obj, target=target, orientation=orientation, original_rpy=original_rpy
        )

        # Track if any rotation was applied (for result reporting).
        rotation_applied = orientation_applied

        # ===== Snap with Algorithm Selection. =====
        snap_result = select_and_execute_snap_algorithm(
            obj=obj,
            target=target,
            orientation=orientation,
            orientation_applied=orientation_applied,
            object_id=object_id,
            target_id=target_id,
            cfg=self.cfg,
        )
        if isinstance(snap_result, str):
            return snap_result
        movement_vector, distance = snap_result

        # Check if already touching (distance < 1mm threshold).
        if distance < ALREADY_TOUCHING_THRESHOLD_M:
            rotation_msg = ""
            if rotation_applied:
                rotation_msg = f", rotation aligned to {math.degrees(new_yaw_rad):.1f}°"
            console_logger.info(
                f"{obj.name} already touching {target.name} "
                f"(distance={distance*1000:.2f}mm){rotation_msg}"
            )
            return SnapToObjectResult(
                success=True,
                message=(
                    f"{obj.name} already in contact with {target.name} "
                    f"(distance={distance*1000:.2f}mm) - no movement needed"
                    f"{rotation_msg}"
                ),
                object_id=object_id,
                target_id=target_id,
                original_position=Position3D(
                    x=float(original_pos[0]),
                    y=float(original_pos[1]),
                    z=float(original_pos[2]),
                ),
                new_position=Position3D(
                    x=float(original_pos[0]),
                    y=float(original_pos[1]),
                    z=float(original_pos[2]),
                ),
                distance_moved=0.0,
                rotation_applied=rotation_applied,
                rotation_angle_degrees=(
                    float(math.degrees(new_yaw_rad)) if rotation_applied else None
                ),
            ).to_json()

        # Create new transform (rotation may have been updated above).
        # Use current position (after collision resolution and orientation).
        new_position = obj.transform.translation() + movement_vector
        new_transform = RigidTransform(R=obj.transform.rotation(), p=new_position)

        # Move object.
        self.scene.move_object(object_id=obj.object_id, new_transform=new_transform)

        rotation_msg = ""
        if rotation_applied:
            rotation_msg = f", rotation aligned to {math.degrees(new_yaw_rad):.1f}°"

        console_logger.info(
            f"Snapped {obj.name} to {target.name}: "
            f"moved {distance:.3f}m from {original_pos} to {new_position}"
            f"{rotation_msg}"
        )

        success_msg = f"Successfully snapped {obj.name} to {target.name}"
        if rotation_applied:
            success_msg += f" (rotation aligned to {math.degrees(new_yaw_rad):.1f}°)"

        return SnapToObjectResult(
            success=True,
            message=success_msg,
            object_id=object_id,
            target_id=target_id,
            original_position=Position3D(
                x=float(original_pos[0]),
                y=float(original_pos[1]),
                z=float(original_pos[2]),
            ),
            new_position=Position3D(
                x=float(new_position[0]),
                y=float(new_position[1]),
                z=float(new_position[2]),
            ),
            distance_moved=float(distance),
            rotation_applied=rotation_applied,
            rotation_angle_degrees=(
                float(math.degrees(new_yaw_rad)) if rotation_applied else None
            ),
        ).to_json()
