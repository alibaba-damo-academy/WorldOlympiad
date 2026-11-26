"""Data Transfer Objects (DTOs) for ceiling tool responses.

This module contains serializable datatypes for structured tool responses with
JSON serialization support. Classes use primitive types (str, float, int, bool)
rather than domain-specific types to enable clean JSON serialization.

## Architectural Pattern: Domain Models vs Response DTOs

This follows the same separation as furniture, manipuland, and wall agents:

1. **Domain Models** (in `agent_utils/`):
   - Use rich domain types (UniqueID, RigidTransform)
   - Example: SceneObject with RigidTransform pose

2. **Response DTOs** (this file):
   - Use primitive types for JSON serialization
   - Example: CeilingObjectInfo with position as floats

## Conversion Pattern

Tools convert domain models to DTOs when returning responses:
- Domain model: SceneObject -> CeilingObjectInfo
"""

from dataclasses import dataclass
from enum import Enum

from scenecode.agent_utils.response_datatypes import (
    AssetInfo,
    BoundingBox3D,
    JSONSerializable,
)


class CeilingErrorType(str, Enum):
    """Types of errors that can occur in ceiling operations."""

    LOOP_DETECTED = "loop_detected"
    OBJECT_NOT_FOUND = "object_not_found"
    POSITION_OUT_OF_BOUNDS = "position_out_of_bounds"
    ASSET_NOT_FOUND = "asset_not_found"
    IMMUTABLE_OBJECT = "immutable_object"
    OVERLAPS_CEILING_OBJECT = "overlaps_ceiling_object"
    INVALID_OPERATION = "invalid_operation"


@dataclass
class Position2D(JSONSerializable):
    """2D position on ceiling plane in room coordinates."""

    x: float
    """X position in room coordinates (meters)."""
    y: float
    """Y position in room coordinates (meters)."""


@dataclass
class RoomBoundsInfo(JSONSerializable):
    """Room bounds information for ceiling placement."""

    min_x: float
    """Minimum X coordinate of room (meters)."""
    min_y: float
    """Minimum Y coordinate of room (meters)."""
    max_x: float
    """Maximum X coordinate of room (meters)."""
    max_y: float
    """Maximum Y coordinate of room (meters)."""
    ceiling_height: float
    """Height of ceiling above floor (meters)."""


@dataclass
class CeilingObjectInfo(JSONSerializable):
    """Information about a placed ceiling object."""

    object_id: str
    """Unique identifier for the object."""
    description: str
    """Text description of the object."""
    position_x: float
    """X position in room coordinates (meters)."""
    position_y: float
    """Y position in room coordinates (meters)."""
    rotation_degrees: float
    """Rotation around Z-axis (degrees)."""
    dimensions: BoundingBox3D
    """Object dimensions (width, depth, height)."""


@dataclass
class CeilingSceneStateResult(JSONSerializable):
    """Current state of ceiling objects in the scene."""

    room_bounds: RoomBoundsInfo
    """Room bounds and ceiling height."""
    ceiling_objects: list[CeilingObjectInfo]
    """List of placed ceiling objects."""
    object_count: int
    """Total number of ceiling objects."""


@dataclass
class PlaceCeilingObjectResult(JSONSerializable):
    """Result of placing a ceiling-mounted object."""

    success: bool
    """Whether placement succeeded."""
    asset_id: str
    """ID of the asset that was placed."""
    object_id: str
    """ID of placed object (empty string if failed)."""
    message: str
    """Human-readable result description."""
    position_x: float
    """Final X position after noise (meters)."""
    position_y: float
    """Final Y position after noise (meters)."""
    rotation_degrees: float
    """Final rotation after noise (degrees)."""
    error_type: CeilingErrorType | None = None
    """Error type if placement failed."""


@dataclass
class CeilingOperationResult(JSONSerializable):
    """Result of a ceiling object operation (move/remove)."""

    success: bool
    """Whether operation succeeded."""
    message: str
    """Human-readable result description."""
    object_id: str
    """ID of the affected object."""
    error_type: CeilingErrorType | None = None
    """Error type if operation failed."""


@dataclass
class AvailableAssetsResult(JSONSerializable):
    """Result of batch asset generation for ceiling placement."""

    assets: list[AssetInfo]
    """List of generated assets."""
    count: int
    """Number of generated assets."""
