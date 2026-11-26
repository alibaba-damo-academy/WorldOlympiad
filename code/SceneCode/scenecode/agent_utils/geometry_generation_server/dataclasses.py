"""Dataclasses for geometry generation server API contracts.

This module contains serializable Data Transfer Objects (DTOs) used for
communication with the geometry generation server. These classes define the
HTTP API contract and use primitive types for JSON serialization.
"""

import json

from dataclasses import asdict, dataclass


@dataclass
class GeometryGenerationServerRequest:
    """Request payload for geometry generation server.

    This DTO defines the contract for geometry generation requests sent
    to the geometry generation server via HTTP. Contains all information
    needed to generate 3D geometry from a 2D image.
    """

    image_path: str
    """Absolute path to the input image file (PNG/JPG format)."""

    output_dir: str
    """Absolute path to directory where generated assets will be saved."""

    prompt: str
    """Text description of the asset to generate (e.g., 'Modern wooden chair')."""

    debug_folder: str | None = None
    """Optional absolute path to directory where debug images will be saved."""

    output_filename: str | None = None
    """Optional filename for the generated geometry file. If not provided, will be
    generated from prompt."""

    backend: str = "hunyuan3d"
    """3D generation backend to use. Either "hunyuan3d" or "sam3d"."""

    sam3d_config: dict | None = None
    """Configuration for SAM3D backend. Required if backend="sam3d". Should contain:
    - sam3_checkpoint (str): Path to SAM3 checkpoint
    - sam3d_checkpoint (str): Path to SAM 3D Objects checkpoint
    - mode (str): Segmentation mode ("foreground" or "object_description")
    - object_description (str | None): Object description (if mode="object_description")
    - threshold (float): Confidence threshold for mask generation
    """

    scene_id: str | None = None
    """Optional scene identifier for fair round-robin scheduling.

    When multiple scenes submit requests concurrently, the server uses this ID to
    group requests from the same scene together for fair GPU time allocation.
    All requests with the same scene_id are treated as a single "client" in the
    round-robin scheduler. If not provided, each HTTP request is treated as a
    separate client.
    """

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation suitable for HTTP requests.
        """
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string for HTTP request body.

        Returns:
            JSON string representation of the request.
        """
        return json.dumps(self.to_dict())


@dataclass
class GeometryGenerationServerResponse:
    """Response payload from geometry generation server.

    This DTO defines the contract for responses from the geometry
    generation server after successful 3D geometry generation.
    """

    geometry_path: str
    """Absolute path to the generated 3D geometry file (GLB format)."""


@dataclass
class GeometryGenerationError:
    """Error information for a failed geometry generation request.

    This DTO represents a geometry generation failure that occurred on the
    server. Used to communicate errors back to the client without stopping
    the entire batch.
    """

    index: int
    """Index of the failed request within the original batch."""

    error_message: str
    """Description of what went wrong during geometry generation."""


@dataclass
class StreamedResult:
    """Single result in a streaming batch response.

    This DTO represents one completed request result in a streaming
    NDJSON response from the server.
    """

    index: int
    """Index of this result within the original batch request."""

    status: str
    """Result status: either "success" or "error"."""

    data: dict | None = None
    """Response data for successful requests (contains geometry_path, etc.)."""

    error: str | None = None
    """Error message for failed requests."""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string for streaming response."""
        return json.dumps(self.to_dict())
