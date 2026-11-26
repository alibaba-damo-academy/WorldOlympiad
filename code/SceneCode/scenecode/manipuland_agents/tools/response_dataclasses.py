"""Data Transfer Objects (DTOs) for manipuland tool responses.

This module contains serializable datatypes for structured tool responses with
JSON serialization support. Classes use primitive types (str, float, int, bool)
rather than domain-specific types to enable clean JSON serialization.

## Architectural Pattern: Domain Models vs Response DTOs

This follows the same separation as furniture agents:

1. **Domain Models** (in `agent_utils/`):
   - Use rich domain types (UniqueID, RigidTransform, PlacementInfo, Path)
   - Example: SceneObject with placement_info

2. **Response DTOs** (this file):
   - Use primitive types for JSON serialization
   - Example: ManipulandInfo with separate position/rotation fields

## Conversion Pattern

Tools convert domain models to DTOs when returning responses:
- Domain model: PlacementInfo → Position2D + rotation_2d fields
- Domain model: SupportSurface → SupportSurfaceInfo
- Domain model: SceneObject → ManipulandInfo
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


class ManipulandErrorType(str, Enum):
    """Types of errors that can occur in manipuland operations."""

    LOOP_DETECTED = "loop_detected"
    OBJECT_NOT_FOUND = "object_not_found"
    SURFACE_NOT_FOUND = "surface_not_found"
    POSITION_OUT_OF_BOUNDS = "position_out_of_bounds"
    ASSET_NOT_FOUND = "asset_not_found"
    IMMUTABLE_OBJECT = "immutable_object"
    INVALID_SURFACE = "invalid_surface"
    NO_MOVEMENT = "no_movement"
    INVALID_OPERATION = "invalid_operation"
    STACK_UNSTABLE = "stack_unstable"
    STACK_EXCEEDS_CLEARANCE = "stack_exceeds_clearance"
    PHYSICS_RESOLUTION_FAILED = "physics_resolution_failed"
    OBJECTS_ON_DIFFERENT_SURFACES = "objects_on_different_surfaces"
    UNSUPPORTED_COMPOSITE_TYPE = "unsupported_composite_type"


@dataclass
class Position2D(JSONSerializable):
    """Represents a 2D position on a support surface."""

    x: float
    """X coordinate in surface frame (meters, left-right)."""
    y: float
    """Y coordinate in surface frame (meters, front-back)."""


@dataclass
class SupportSurfaceInfo(JSONSerializable):
    """Information about a support surface."""

    surface_id: str
    """Unique identifier for the surface."""
    bounds_min: Position2D
    """Minimum corner of surface in surface frame."""
    bounds_max: Position2D
    """Maximum corner of surface in surface frame."""
    world_x: float
    """Surface center X position in world frame (meters)."""
    world_y: float
    """Surface center Y position in world frame (meters)."""
    world_z: float
    """Surface center Z position in world frame (meters)."""
    clearance_height: float
    """Vertical clearance above surface in meters (maximum object height)."""


@dataclass
class SupportSurfaceWithManipulands(JSONSerializable):
    """Support surface with its manipulands grouped together.

    Used in get_current_scene_state to show which manipulands are on each surface.
    Manipuland positions are in surface-relative coordinates.
    """

    surface_id: str
    """Unique identifier for the surface."""
    bounds_min: Position2D
    """Minimum corner of surface in surface frame."""
    bounds_max: Position2D
    """Maximum corner of surface in surface frame."""
    world_x: float
    """Surface center X position in world frame (meters)."""
    world_y: float
    """Surface center Y position in world frame (meters)."""
    world_z: float
    """Surface center Z position in world frame (meters)."""
    clearance_height: float
    """Vertical clearance above surface in meters (maximum object height)."""
    manipulands: list["SimplifiedManipulandInfo"]
    """Manipulands placed on this surface with surface-relative positions."""


@dataclass
class ManipulandInfo(JSONSerializable):
    """Information about a manipuland object in the scene."""

    object_id: str
    """Unique identifier."""
    description: str
    """Text description."""
    object_type: str
    """Type of object ('manipuland')."""
    position: Position3D
    """World-frame position."""
    rotation: Rotation3D
    """World-frame rotation."""
    surface_position: Position2D | None
    """Position on support surface (if placed on surface)."""
    surface_rotation_deg: float | None
    """Rotation on support surface in degrees (if placed on surface)."""
    parent_surface_id: str | None
    """ID of support surface this is placed on (if applicable)."""
    has_geometry: bool
    """Whether 3D geometry exists."""


@dataclass
class SimplifiedManipulandInfo(JSONSerializable):
    """Simplified info about manipuland on current surface (no redundant fields)."""

    object_id: str
    """Unique identifier for tool calls (e.g., remove_manipuland)."""
    description: str
    """What the object is."""
    surface_position: Position2D | None
    """Position on surface (x, z in meters)."""
    surface_rotation_deg: float | None
    """Rotation on surface in degrees."""
    dimensions: BoundingBox3D | None
    """Object dimensions (width, depth, height) in meters."""
    composite_metadata: dict | None = None
    """Metadata for composite objects (stacks, filled containers, piles). None if not composite.

    For stacks: {"type": "stack", "members": ["asset_id_0", "asset_id_1", ...]}
    For filled containers: {"type": "filled_container", "container_id": "bowl_abc",
                            "fill_object_ids": ["apple_0", "apple_1"], "fill_count": 2}
    For piles: {"type": "pile", "members": ["asset_id_0", "asset_id_1", ...], "pile_count": 3}
    """


@dataclass
class SimplifiedFurnitureInfo(JSONSerializable):
    """Minimal furniture info for manipuland placement context."""

    object_id: str
    """Unique identifier for reference."""
    description: str
    """What the furniture is."""
    dimensions: BoundingBox3D | None
    """Object dimensions (width, depth, height) in meters."""


@dataclass
class CurrentSceneStateResult(JSONSerializable):
    """Current scene state for manipuland placement."""

    current_furniture: SimplifiedFurnitureInfo
    """Minimal furniture info (id, description, dimensions)."""
    surface: SupportSurfaceInfo
    """Surface bounds and dimensions for placement."""
    manipulands_on_surface: list[SimplifiedManipulandInfo]
    """Manipulands already placed (simplified, surface-relative only)."""
    manipuland_count: int
    """Number of manipulands on surface."""


@dataclass
class ManipulandPlacementResult(JSONSerializable):
    """Result of placing a manipuland on a support surface."""

    success: bool
    """Whether the operation succeeded."""
    message: str
    """Human-readable description of the result."""
    asset_id: str
    """ID of the asset that was requested for placement."""
    object_id: str
    """Unique identifier of the placed object (empty on failure)."""
    world_position: Position3D
    """Final 3D position in world coordinates."""
    world_rotation: Rotation3D
    """Final 3D rotation in world coordinates."""
    surface_position: Position2D
    """2D position on the support surface."""
    surface_rotation_deg: float
    """Rotation on surface in degrees."""
    parent_surface_id: str
    """ID of the support surface."""
    has_geometry: bool
    """Whether 3D geometry was successfully loaded."""
    error_type: ManipulandErrorType | None = None
    """Type of error if operation failed."""


@dataclass
class ManipulandOperationResult(JSONSerializable):
    """Generic result for manipuland operations (move, remove)."""

    success: bool
    """Whether the operation succeeded."""
    message: str
    """Human-readable description of the result."""
    object_id: str | None = None
    """ID of the affected object (if applicable)."""
    error_type: ManipulandErrorType | None = None
    """Type of error if operation failed."""


@dataclass
class AvailableAssetsResult(JSONSerializable):
    """Result containing list of available manipuland assets."""

    assets: list[AssetInfo]
    """List of available assets."""
    total_count: int
    """Total number of available assets."""
    message: str | None = None
    """Optional status message."""


@dataclass
class StackCreationResult(JSONSerializable):
    """Result of creating a stack of objects on a support surface.

    For failures, actionable feedback is included in the message field:
    - Unstable: "Stack unstable. Items 0-2 stable, items 3-4 fell. Try removing top items."
    - Clearance: "Stack height 0.45m exceeds clearance 0.30m. Try with first 3 items."
    """

    success: bool
    """Whether the stack was created successfully."""
    message: str
    """Human-readable description including actionable feedback on failure."""
    stack_object_id: str | None
    """Composite object ID (e.g., 'stack_a1'), None on failure."""
    stack_height: float | None
    """Total height of stack in meters, None on failure."""
    parent_surface_id: str
    """ID of the support surface the stack is placed on."""
    num_items: int
    """Number of items requested (always set)."""
    error_type: ManipulandErrorType | None = None
    """Type of error if operation failed."""


@dataclass
class FillContainerResult(JSONSerializable):
    """Result of filling a container with objects.

    For failures, actionable feedback is included in the message field:
    - Overflow: Shows which objects fit and which were removed.
    - Empty: No objects fit - try smaller fill objects or larger container.
    """

    success: bool
    """Whether the fill operation succeeded (at least one object inside)."""
    message: str
    """Human-readable description including actionable feedback on failure."""
    filled_container_id: str | None
    """Composite object ID (e.g., 'filled_container_a1'), None on failure."""
    container_asset_id: str
    """ID of the container asset used."""
    fill_count: int
    """Number of objects that stayed inside the container."""
    total_fill_attempted: int
    """Total number of fill objects attempted."""
    removed_count: int
    """Number of objects that fell outside and were deleted."""
    parent_surface_id: str
    """ID of the support surface the container is placed on."""
    inside_assets: list[str]
    """Names of assets that stayed inside (e.g., ["apple", "apple", "orange"])."""
    removed_assets: list[str]
    """Names of assets that fell outside (e.g., ["banana"])."""
    error_type: ManipulandErrorType | None = None
    """Type of error if operation failed."""


@dataclass
class PileCreationResult(JSONSerializable):
    """Result of creating a pile of objects on a support surface.

    For partial success, some objects may fall off the surface during physics
    simulation. The response indicates which objects stayed in the pile.
    """

    success: bool
    """Whether at least 2 objects stayed in the pile."""
    message: str
    """Human-readable description including actionable feedback."""
    pile_object_id: str | None
    """Composite object ID (e.g., 'pile_a1'), None on failure."""
    parent_surface_id: str
    """ID of the support surface the pile is placed on."""
    num_items: int
    """Number of items requested."""
    pile_count: int
    """Number of items that stayed in the pile."""
    removed_count: int
    """Number of items that fell off the surface."""
    inside_assets: list[str]
    """Names of assets that stayed in pile (e.g., ["block", "block", "toy"])."""
    removed_assets: list[str]
    """Names of assets that fell off (e.g., ["ball"])."""
    error_type: ManipulandErrorType | None = None
    """Type of error if operation failed."""


@dataclass
class PenetrationResolutionResult(JSONSerializable):
    """Result of resolving penetrations between objects on a surface."""

    success: bool
    """Whether the resolution succeeded."""
    message: str
    """Human-readable description of the result."""
    num_objects_considered: int
    """Number of objects that were considered for resolution."""
    num_objects_moved: int
    """Number of objects that were actually moved."""
    moved_object_ids: list[str]
    """IDs of objects that were moved."""
    max_displacement_m: float
    """Largest movement applied (meters)."""
    error_type: ManipulandErrorType | None = None
    """Type of error if operation failed."""
