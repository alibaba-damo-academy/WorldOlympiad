"""Shared Data Transfer Objects (DTOs) for structured tool responses.

This module provides common serializable datatypes for tool responses with
JSON serialization support. Classes use primitive types (str, float, int, bool)
rather than domain-specific types to enable clean JSON serialization without
custom serializers.

## Provided Types

- JSONSerializable: Mixin for automatic JSON serialization
- Position3D: 3D world coordinates (x, y, z)
- Rotation3D: Euler angle rotations (roll, pitch, yaw)
- BoundingBox3D: 3D bounding box dimensions (width, depth, height)
- AssetInfo: Available asset information for placement tools
- GeneratedAsset: 3D asset generation results with metadata
- AssetGenerationResult: Batch asset generation with partial success support
"""

from __future__ import annotations

import json

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scenecode.agent_utils.room import SceneObject


class JSONSerializable:
    """Mixin class that provides JSON serialization for dataclasses.

    Any dataclass that inherits from this mixin automatically gets a to_json()
    method that converts the dataclass to a JSON string using
    dataclasses.asdict().

    Example:
        @dataclass
        class MyData(JSONSerializable):
            value: int

        data = MyData(value=42)
        json_str = data.to_json()  # '{"value": 42}'
    """

    def to_json(self) -> str:
        """Convert dataclass to JSON string for tool response."""
        return json.dumps(asdict(self))


@dataclass
class Position3D(JSONSerializable):
    """Represents a 3D position in world frame."""

    x: float
    """X coordinate."""
    y: float
    """Y coordinate."""
    z: float
    """Z coordinate."""


@dataclass
class Rotation3D(JSONSerializable):
    """Represents a 3D rotation in Euler angles."""

    roll: float
    """Roll angle in degrees."""
    pitch: float
    """Pitch angle in degrees."""
    yaw: float
    """Yaw angle in degrees."""


@dataclass
class BoundingBox3D(JSONSerializable):
    """3D bounding box dimensions.

    Shared by all placement agents for representing object dimensions.
    """

    width: float
    """Width in meters (x-axis extent)."""
    depth: float
    """Depth in meters (y-axis extent)."""
    height: float
    """Height in meters (z-axis extent)."""


@dataclass
class AssetInfo(JSONSerializable):
    """Info about available asset for reuse.

    Shared by all placement agents (furniture, wall, ceiling, manipuland) for
    listing available assets via list_available_assets tool.
    """

    id: str
    """Unique identifier for the asset."""
    name: str
    """Display name of the furniture piece."""
    description: str
    """Text description used for generation."""
    type: str
    """Type category (FURNITURE, MANIPULAND, etc.)."""
    dimensions: BoundingBox3D | None = None
    """Object dimensions (width, depth, height) in meters."""
    scale_factor: float = 1.0
    """Current scale factor applied to this asset (1.0 = original size)."""

    @classmethod
    def from_scene_object(cls, obj: SceneObject) -> AssetInfo:
        """Convert SceneObject to AssetInfo DTO.

        Args:
            obj: SceneObject from the domain model

        Returns:
            AssetInfo DTO ready for JSON serialization
        """
        # Extract dimensions if bounding box data is available.
        dimensions = None
        if obj.bbox_min is not None and obj.bbox_max is not None:
            bbox_size = obj.bbox_max - obj.bbox_min
            dimensions = BoundingBox3D(
                width=float(bbox_size[0]),
                depth=float(bbox_size[1]),
                height=float(bbox_size[2]),
            )

        return cls(
            id=str(obj.object_id),
            name=obj.name,
            description=obj.description,
            type=obj.object_type.value,
            dimensions=dimensions,
            scale_factor=obj.scale_factor,
        )


@dataclass
class GeneratedAsset(JSONSerializable):
    """Represents a generated asset (furniture or manipuland).

    Only includes information useful for the agent (object_id for placement, dimensions
    for planning).
    """

    name: str
    """Display name of the object."""
    object_id: str
    """Unique identifier for the asset (use this for placement)."""
    description: str
    """Text description used for generation."""
    width: float | None = None
    """Object width in meters (x-axis)."""
    depth: float | None = None
    """Object depth in meters (y-axis)."""
    height: float | None = None
    """Object height in meters (z-axis)."""
    scale_factor: float = 1.0
    """Current scale factor applied to this asset (1.0 = original size)."""


@dataclass
class AssetGenerationResult(JSONSerializable):
    """Result of generating assets with partial success support.

    This dataclass is shared by both furniture and manipuland agents
    to represent the result of batch asset generation operations.
    """

    success: bool
    """Whether all assets were generated successfully."""
    assets: list[GeneratedAsset]
    """List of successfully generated assets."""
    message: str | None = None
    """Status or error message."""
    successful_count: int | None = None
    """Number of successfully generated assets (for partial success)."""
    failed_count: int | None = None
    """Number of failed assets (for partial success)."""
    failures: str | None = None
    """Details about failed assets (for partial success)."""
