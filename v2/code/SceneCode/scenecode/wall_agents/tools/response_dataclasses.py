"""Data Transfer Objects (DTOs) for wall tool responses.

This module contains serializable datatypes for structured tool responses with
JSON serialization support. Classes use primitive types (str, float, int, bool)
rather than domain-specific types to enable clean JSON serialization.

## Architectural Pattern: Domain Models vs Response DTOs

This follows the same separation as furniture and manipuland agents:

1. **Domain Models** (in `agent_utils/` and `wall_surface.py`):
   - Use rich domain types (UniqueID, RigidTransform, WallSurface)
   - Example: WallSurface with transform and excluded_regions

2. **Response DTOs** (this file):
   - Use primitive types for JSON serialization
   - Example: WallSurfaceInfo with position/dimensions as floats

## Conversion Pattern

Tools convert domain models to DTOs when returning responses:
- Domain model: WallSurface → WallSurfaceInfo
- Domain model: SceneObject → WallObjectInfo
"""

from dataclasses import dataclass
from enum import Enum

from scenecode.agent_utils.response_datatypes import (
    AssetInfo,
    BoundingBox3D,
    JSONSerializable,
    Position3D,
    Rotation3D,
)


class WallErrorType(str, Enum):
    """Types of errors that can occur in wall operations."""

    LOOP_DETECTED = "loop_detected"
    OBJECT_NOT_FOUND = "object_not_found"
    SURFACE_NOT_FOUND = "surface_not_found"
    POSITION_OUT_OF_BOUNDS = "position_out_of_bounds"
    ASSET_NOT_FOUND = "asset_not_found"
    IMMUTABLE_OBJECT = "immutable_object"
    OVERLAPS_OPENING = "overlaps_opening"
    OVERLAPS_FURNITURE = "overlaps_furniture"
    OVERLAPS_WALL_OBJECT = "overlaps_wall_object"
    INVALID_OPERATION = "invalid_operation"


@dataclass
class Position2D(JSONSerializable):
    """2D position in wall-local coordinates."""

    x: float
    """Position along wall (meters from wall start)."""
    z: float
    """Height on wall (meters from floor)."""


@dataclass
class ExcludedRegionInfo(JSONSerializable):
    """Information about an excluded region (door/window) on a wall."""

    x_min: float
    """Minimum X position along wall (meters)."""
    z_min: float
    """Minimum Z height (meters from floor)."""
    x_max: float
    """Maximum X position along wall (meters)."""
    z_max: float
    """Maximum Z height (meters from floor)."""


@dataclass
class WallSurfaceInfo(JSONSerializable):
    """Information about a wall surface for placement."""

    surface_id: str
    """Unique identifier for the surface."""
    wall_id: str
    """Wall identifier (e.g., 'living_room_north')."""
    wall_direction: str
    """Cardinal direction (north/south/east/west)."""
    length: float
    """Wall length in meters."""
    height: float
    """Wall height in meters."""
    excluded_regions: list[ExcludedRegionInfo]
    """Doors/windows where objects cannot be placed."""


@dataclass
class WallObjectInfo(JSONSerializable):
    """Simplified wall object information for scene state."""

    object_id: str
    """Unique identifier for the object."""
    description: str
    """Object description."""
    wall_surface_id: str
    """ID of the wall surface it's mounted on."""
    position_x: float
    """Position along wall (meters from wall start)."""
    position_z: float
    """Height on wall (meters from floor)."""
    rotation_deg: float
    """Rotation around wall normal (degrees)."""
    dimensions: BoundingBox3D
    """Object dimensions (width, depth, height)."""


@dataclass
class WallSceneStateResult(JSONSerializable):
    """Current scene state for wall object placement."""

    wall_surfaces: list[WallSurfaceInfo]
    """All wall surfaces in the room."""
    wall_objects: list[WallObjectInfo]
    """Wall objects already placed."""
    object_count: int
    """Total number of wall objects placed."""


@dataclass
class PlaceWallObjectResult(JSONSerializable):
    """Result of placing a wall-mounted object."""

    success: bool
    """Whether the placement succeeded."""
    asset_id: str
    """ID of the asset that was requested for placement."""
    object_id: str
    """ID of placed object (empty string if failed)."""
    message: str
    """Human-readable result description."""
    wall_surface_id: str
    """ID of the wall surface."""
    position_x: float
    """Final position along wall (meters)."""
    position_z: float
    """Final height on wall (meters)."""
    rotation_deg: float
    """Final rotation (degrees)."""
    world_position: Position3D | None = None
    """World [x, y, z] position of placed object."""
    world_rotation: Rotation3D | None = None
    """World rotation of placed object."""
    error_type: WallErrorType | None = None
    """Type of error if operation failed."""


@dataclass
class WallOperationResult(JSONSerializable):
    """Result of a wall object operation (move/remove)."""

    success: bool
    """Whether the operation succeeded."""
    message: str
    """Human-readable result description."""
    object_id: str
    """ID of the affected object."""
    error_type: WallErrorType | None = None
    """Type of error if operation failed."""


@dataclass
class AvailableAssetsResult(JSONSerializable):
    """Result of listing available assets for wall placement."""

    assets: list[AssetInfo]
    """List of available assets."""
    count: int
    """Number of available assets."""
