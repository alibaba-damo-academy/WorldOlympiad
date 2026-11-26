"""Data Transfer Objects (DTOs) for furniture tool responses.

This module contains serializable datatypes for structured tool responses with
JSON serialization support. Classes use primitive types (str, float, int, bool)
rather than domain-specific types to enable clean JSON serialization.

## Architectural Pattern: Domain Models vs Response DTOs

This codebase follows a clear separation between:

1. **Domain Models** (in `agent_utils/`):
   - Use rich domain types (UniqueID, RigidTransform, ObjectType enum, Path)
   - Represent the core business logic and data structures
   - Example: SceneObject in `agent_utils/scene.py`

2. **Response DTOs** (this file):
   - Use primitive types for JSON serialization
   - Represent data for tool responses and API contracts
   - Example: SceneObjectInfo (serializable version of SceneObject)

## Conversion Pattern

Tools convert domain models to DTOs when returning responses:
- Domain model: SceneObject with RigidTransform → SceneObjectInfo with
  Position3D
- Domain model: ObjectType enum → str representation
- Domain model: Path objects → str paths

This separation enables:
- Clean JSON serialization without custom serializers
- Independent evolution of domain models and API contracts
- Clear boundaries between business logic and presentation
"""

import math

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydrake.all import RollPitchYaw

from scenecode.agent_utils.response_datatypes import (
    AssetInfo,
    BoundingBox3D,
    JSONSerializable,
    Position3D,
    Rotation3D,
)
from scenecode.agent_utils.room import SceneObject

# Re-export shared types for backwards compatibility.
__all__ = ["AssetInfo", "BoundingBox3D"]


class FurnitureErrorType(str, Enum):
    """Types of errors that can occur in furniture operations."""

    NO_MOVEMENT = "no_movement"
    LOOP_DETECTED = "loop_detected"
    OBJECT_NOT_FOUND = "object_not_found"
    INVALID_POSITION = "invalid_position"
    ASSET_NOT_FOUND = "asset_not_found"
    IMMUTABLE_OBJECT = "immutable_object"
    POSITION_OUT_OF_BOUNDS = "position_out_of_bounds"


@dataclass
class Coordinate2D(JSONSerializable):
    """2D coordinate point."""

    x: float
    """X coordinate in meters."""
    y: float
    """Y coordinate in meters."""


@dataclass
class WorldBounds3D(JSONSerializable):
    """World-frame AABB."""

    min_corner: Position3D
    """Minimum corner of the bounding box in world coordinates."""
    max_corner: Position3D
    """Maximum corner of the bounding box in world coordinates."""


@dataclass
class SimplifiedFurnitureInfo(JSONSerializable):
    """Furniture info showing controllable SE(2) pose on floor.

    This shows only the parameters the agent can control (x, y, yaw)
    rather than full SE(3) pose. Matches the move_furniture_tool interface.
    """

    object_id: str
    """Unique identifier for tool calls (e.g., move_furniture)."""
    description: str
    """What the furniture is."""
    position_x: float
    """X position in room coordinates (meters)."""
    position_y: float
    """Y position in room coordinates (meters)."""
    rotation_degrees: float
    """Rotation around Z-axis (degrees). This is the yaw angle."""
    dimensions: BoundingBox3D | None
    """Object dimensions (width, depth, height) in meters."""

    @classmethod
    def from_scene_object(cls, obj: SceneObject) -> "SimplifiedFurnitureInfo":
        """Create SimplifiedFurnitureInfo from SceneObject.

        Extracts SE(2) controllable pose from world transform.
        """
        # Extract position from transform.
        translation = obj.transform.translation()

        # Extract yaw from rotation.
        rpy = RollPitchYaw(obj.transform.rotation())
        yaw_degrees = math.degrees(rpy.yaw_angle())

        # Compute dimensions from bounding box.
        dimensions = None
        if obj.bbox_min is not None and obj.bbox_max is not None:
            bbox_size = obj.bbox_max - obj.bbox_min
            dimensions = BoundingBox3D(
                width=float(bbox_size[0]),
                depth=float(bbox_size[1]),
                height=float(bbox_size[2]),
            )

        return cls(
            object_id=str(obj.object_id),
            description=obj.description,
            position_x=float(translation[0]),
            position_y=float(translation[1]),
            rotation_degrees=yaw_degrees,
            dimensions=dimensions,
        )


@dataclass
class FurniturePlacementResult(JSONSerializable):
    """Result of adding furniture to a scene."""

    success: bool
    """Whether the operation succeeded."""
    message: str
    """Human-readable description of the result."""
    asset_id: str
    """ID of the asset that was requested for placement."""
    object_id: str
    """Unique identifier of the placed object (empty on failure)."""
    position: Position3D
    """3D position coordinates."""
    rotation: Rotation3D
    """3D rotation angles."""
    has_geometry: bool
    """Whether 3D geometry was successfully loaded."""
    error_type: FurnitureErrorType | None = None
    """Type of error if operation failed."""
    suggested_action: str | None = None
    """Suggested next action for the agent."""


@dataclass
class SceneObjectInfo(JSONSerializable):
    """Serializable representation of a scene object for tool responses.

    This is the DTO version of SceneObject from agent_utils/scene.py.
    Converts rich domain types to primitive types for JSON serialization.
    """

    object_id: str
    """Unique identifier of the object."""
    description: str
    """Text description of the object (e.g., 'A wooden dining table')."""
    position: Position3D
    """Current 3D position in the scene."""
    rotation: Rotation3D
    """Current 3D rotation in the scene."""
    object_type: str
    """Type category (FURNITURE, MANIPULAND, etc.)."""
    dimensions: BoundingBox3D | None
    """Object dimensions in local frame."""
    world_bounds: WorldBounds3D | None
    """World-frame bounding box."""
    immutable: bool
    """Whether this object is immutable (cannot be moved or removed)."""

    @classmethod
    def from_scene_object(cls, obj: SceneObject) -> "SceneObjectInfo":
        """Create SceneObjectInfo from a SceneObject domain model.

        Converts rich domain types (UniqueID, RigidTransform, ObjectType enum)
        to primitive types suitable for JSON serialization.

        Args:
            obj: SceneObject domain model to convert.

        Returns:
            SceneObjectInfo DTO with primitive types.
        """
        # Extract rotation from RigidTransform.
        rpy = obj.transform.rotation().ToRollPitchYaw()

        # Compute bounding box information.
        dimensions = None
        world_bounds = None
        if obj.bbox_min is not None and obj.bbox_max is not None:
            # Object-frame dimensions.
            bbox_size = obj.bbox_max - obj.bbox_min
            dimensions = BoundingBox3D(
                width=float(bbox_size[0]),
                depth=float(bbox_size[1]),
                height=float(bbox_size[2]),
            )

            # World-frame bounds.
            world_bounds_result = obj.compute_world_bounds()
            if world_bounds_result is not None:
                world_bbox_min, world_bbox_max = world_bounds_result
                world_bounds = WorldBounds3D(
                    min_corner=Position3D(
                        x=float(world_bbox_min[0]),
                        y=float(world_bbox_min[1]),
                        z=float(world_bbox_min[2]),
                    ),
                    max_corner=Position3D(
                        x=float(world_bbox_max[0]),
                        y=float(world_bbox_max[1]),
                        z=float(world_bbox_max[2]),
                    ),
                )

        return cls(
            object_id=str(obj.object_id),
            description=obj.description,
            position=Position3D(
                x=obj.transform.translation()[0],
                y=obj.transform.translation()[1],
                z=obj.transform.translation()[2],
            ),
            rotation=Rotation3D(
                roll=math.degrees(rpy.roll_angle()),
                pitch=math.degrees(rpy.pitch_angle()),
                yaw=math.degrees(rpy.yaw_angle()),
            ),
            object_type=obj.object_type.value,
            dimensions=dimensions,
            world_bounds=world_bounds,
            immutable=obj.immutable,
        )


@dataclass
class SceneStateResult(JSONSerializable):
    """Result of querying current scene state."""

    success: bool
    """Whether the query succeeded."""
    furniture_count: int
    """Total number of furniture objects in scene."""
    objects: list[SimplifiedFurnitureInfo]
    """List of all scene objects with SE(2) controllable pose."""
    message: str | None = None
    """Optional status or error message."""


@dataclass
class FurnitureOperationResult(JSONSerializable):
    """Generic result for furniture operations (move, remove)."""

    success: bool
    """Whether the operation succeeded."""
    message: str
    """Human-readable description of the result."""
    object_id: str
    """ID of the object that was operated on."""
    error_type: FurnitureErrorType | None = None
    """Type of error if operation failed."""
    current_position: Position3D | None = None
    """Current position of object if relevant."""
    attempted_position: Position3D | None = None
    """Position that was attempted if relevant."""
    current_rotation: Rotation3D | None = None
    """Current rotation of object if relevant."""
    attempted_rotation: Rotation3D | None = None
    """Rotation that was attempted if relevant."""
    suggested_action: str | None = None
    """Suggested next action for the agent."""


# GeneratedAsset and AssetGenerationResult are now imported from
# scenecode.agent_utils.response_datatypes (shared with manipuland agent)

# TodoItem, TodoSummary, and TodoOperationResult have been moved to
# scenecode.agent_utils.workflow_tools (shared with manipuland agent)


@dataclass
class CritiqueItem(JSONSerializable):
    """Represents a critique item."""

    critique_id: str
    """Unique identifier for the critique."""
    critique: str
    """The critique text describing issues."""
    proposed_solution: str
    """Suggested solution to address the critique."""
    previous_attempts: int
    """Number of previous attempts to fix this critique."""


@dataclass
class CritiquesListResult(JSONSerializable):
    """Result of getting unaddressed critiques."""

    success: bool
    """Whether the query succeeded."""
    critiques: list[CritiqueItem]
    """List of unaddressed critique items."""
    count: int
    """Total number of critiques returned."""


@dataclass
class IterationStatusResult(JSONSerializable):
    """Result of getting iteration status."""

    critique_count: int
    """Current number of critique iterations performed."""
    max_critiques: int
    """Maximum allowed critique iterations."""
    total_critiques: int
    """Total number of critiques generated."""
    addressed_critiques: int
    """Number of critiques that have been addressed."""
    unaddressed_critiques: int
    """Number of critiques that remain unaddressed."""
    should_continue: bool
    """Whether the iteration process should continue."""
    unaddressed_details: list[CritiqueItem]
    """Details of all unaddressed critiques."""


@dataclass
class CritiqueResult(JSONSerializable):
    """Result of critiquing a scene."""

    success: bool
    """Whether the critique was generated successfully."""
    critique_text: str
    """The critique feedback text."""
    message: str | None = None
    """Optional status or error message."""


@dataclass
class ValidationResult(JSONSerializable):
    """Result of validating critique resolution."""

    success: bool
    """Whether the validation succeeded."""
    validation_results: dict[str, Any]
    """The validation analysis results."""
    message: str | None = None
    """Optional status or error message."""


@dataclass
class DesignerExecutionResult(JSONSerializable):
    """Result of designer agent execution."""

    success: bool
    """Whether the designer execution succeeded."""
    summary: str
    """Summary of the designer's work completed."""
    task_type: str
    """Type of design task performed (INITIAL_DESIGN or ADDRESS_CRITIQUES)."""
    message: str | None = None
    """Optional status or error message."""


@dataclass
class AvailableAssetsResult(JSONSerializable):
    """Result of listing available assets for reuse."""

    success: bool
    assets: list[AssetInfo]
    count: int
    message: str


@dataclass
class FacingCheckResult(JSONSerializable):
    """Result of checking if object A is facing toward or away from object B.

    Helps verify spatial relationships between furniture objects. Use "toward"
    for furniture that should face something (chairs→tables, sofas→TVs) and
    "away" for furniture against walls (desks, shelves, appliances).
    """

    success: bool
    """Whether the check operation succeeded."""
    object_a_id: str
    """ID of the first object (the one being checked)."""
    object_b_id: str
    """ID of the second object (the target)."""
    is_facing: bool
    """Whether the orientation is correct for the specified direction."""
    optimal_rotation_degrees: float
    """Absolute yaw rotation in degrees for perfect facing alignment. Use directly
    with move_furniture_tool(). Positive = counter-clockwise."""
    current_rotation_degrees: float
    """Current absolute yaw rotation (in degrees) of object A."""
    message: str | None = None
    """Optional status or error message."""


@dataclass
class SnapToObjectResult(JSONSerializable):
    """Result of snapping object A to touch object B.

    Moves object_id to just touch target_id along shortest path.
    Use the orientation parameter to control rotation behavior.
    """

    success: bool
    """Whether the snap operation succeeded."""
    message: str
    """Human-readable description of the result."""
    object_id: str
    """ID of the object that was moved."""
    target_id: str
    """ID of the target object (stays in place)."""
    original_position: Position3D | None = None
    """Position before snapping."""
    new_position: Position3D | None = None
    """Position after snapping."""
    distance_moved: float | None = None
    """Distance moved in meters."""
    rotation_applied: bool = False
    """Whether orientation was aligned to wall (only for wall snapping)."""
    rotation_angle_degrees: float | None = None
    """New absolute yaw rotation in degrees if rotation was applied."""
    error_type: FurnitureErrorType | None = None
    """Type of error if operation failed."""
    suggested_action: str | None = None
    """Suggested next action for the agent."""
