"""Data Transfer Object for rescale operation results.

This module provides RescaleResult, a shared dataclass for all placement agents
to report the results of rescale operations.
"""

from dataclasses import dataclass
from enum import Enum

from scenecode.agent_utils.response_datatypes import BoundingBox3D, JSONSerializable


class RescaleErrorType(str, Enum):
    """Types of errors that can occur in rescale operations."""

    INVALID_SCALE_FACTOR = "invalid_scale_factor"
    NO_OP = "no_op"
    OBJECT_NOT_FOUND = "object_not_found"
    IMMUTABLE_OBJECT = "immutable_object"
    NO_SDF = "no_sdf"
    RESCALE_FAILED = "rescale_failed"


@dataclass
class RescaleResult(JSONSerializable):
    """Result of a rescale operation.

    This dataclass is shared by all placement agents (furniture, manipuland,
    wall, ceiling) to provide consistent rescale result reporting.

    IMPORTANT: Rescaling operates on the ASSET, not the instance. All objects
    sharing the same sdf_path are affected by a rescale operation.
    """

    success: bool
    """Whether the rescale operation succeeded."""

    message: str
    """Status or error message describing the result."""

    asset_id: str | None = None
    """The asset that was rescaled (from the object's sdf_path)."""

    object_id: str | None = None
    """The object that triggered the rescale operation."""

    affected_object_ids: list[str] | None = None
    """All object IDs affected by this rescale (may include multiple instances)."""

    scale_factor: float | None = None
    """The scale factor that was applied."""

    new_asset_scale: float | None = None
    """Cumulative scale on the asset after this operation (e.g., 1.5 = 150%)."""

    previous_dimensions: BoundingBox3D | None = None
    """Dimensions before rescaling."""

    new_dimensions: BoundingBox3D | None = None
    """Dimensions after rescaling."""

    error_type: RescaleErrorType | None = None
    """Error type if operation failed."""
