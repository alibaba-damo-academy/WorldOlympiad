"""Response dataclasses for validation and policy interface tools.

These DTOs are returned by state and vision tools to provide structured
information to the validator and policy interface agents.
"""

from dataclasses import dataclass


@dataclass
class ObjectInfo:
    """Basic information about a scene object."""

    id: str
    """Unique object identifier."""

    description: str
    """Human-readable description of the object."""

    object_type: str
    """Object type (furniture, manipuland, etc.)."""

    position: list[float]
    """World position [x, y, z] in meters."""

    bbox_min: list[float] | None = None
    """Minimum corner of axis-aligned bounding box."""

    bbox_max: list[float] | None = None
    """Maximum corner of axis-aligned bounding box."""


@dataclass
class DistanceResult:
    """Result of distance query between two objects."""

    object_a: str
    """ID of first object."""

    object_b: str
    """ID of second object."""

    distance: float
    """Surface-to-surface distance in meters. Negative if penetrating."""

    in_contact: bool
    """True if objects are touching (distance < 1mm)."""

    a_dimensions: list[float] | None = None
    """Dimensions of object A [width, depth, height] in meters."""

    b_dimensions: list[float] | None = None
    """Dimensions of object B [width, depth, height] in meters."""


@dataclass
class SpatialRelationResult:
    """Spatial relationship between two objects."""

    object_a: str
    """ID of first object."""

    object_b: str
    """ID of second object."""

    vertical_surface_gap: float
    """Vertical gap from bottom of A to top of B (negative = A below B)."""

    horizontal_surface_gap: float
    """Horizontal gap between bbox edges (0 = touching/overlapping)."""

    a_footprint_overlaps_b: bool
    """True if A's XY footprint overlaps B's XY footprint."""

    a_dimensions: list[float]
    """Dimensions of object A [width, depth, height] in meters."""

    b_dimensions: list[float]
    """Dimensions of object B [width, depth, height] in meters."""


@dataclass
class SupportResult:
    """Result of support/contact analysis."""

    target: str
    """ID of object being supported."""

    surface: str
    """ID of supporting surface/object."""

    vertical_gap: float
    """Vertical gap from bottom of target to top of surface in meters."""

    in_contact: bool
    """True if objects are in contact (Drake signed distance check)."""

    footprint_on_surface_pct: float
    """Percentage of target's XY footprint that overlaps the surface (0.0-1.0)."""

    target_dimensions: list[float]
    """Dimensions of target object [width, depth, height] in meters."""


@dataclass
class ObjectDetailInfo:
    """Detailed information about an object including geometry."""

    id: str
    """Unique object identifier."""

    description: str
    """Human-readable description of the object."""

    object_type: str
    """Object type (furniture, manipuland, etc.)."""

    position: list[float]
    """World position [x, y, z] in meters."""

    orientation_euler_deg: dict[str, float]
    """Orientation as Euler angles {roll, pitch, yaw} in degrees."""

    tilt_from_upright_deg: float
    """Tilt angle from vertical (0 = upright, 90 = horizontal)."""

    dimensions: list[float]
    """Bounding box dimensions [width, depth, height] in meters."""
