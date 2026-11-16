"""Dataclasses for articulated retrieval server API contracts.

This module contains serializable Data Transfer Objects (DTOs) used for
communication with the articulated retrieval server. These classes define the
HTTP API contract and use primitive types for JSON serialization.
"""

import json

from dataclasses import asdict, dataclass


@dataclass
class ArticulatedRetrievalServerRequest:
    """Request payload for articulated retrieval server.

    This DTO defines the contract for articulated asset retrieval requests sent
    to the articulated retrieval server via HTTP. Contains all information needed
    to perform semantic search over articulated object datasets (PartNet-Mobility,
    ArtVIP) using CLIP.
    """

    object_description: str
    """Text description of the object to retrieve (e.g., 'Modern wooden cabinet')."""

    object_type: str
    """Type of object to retrieve (e.g., 'FURNITURE')."""

    output_dir: str
    """Client-specified output directory where server will export combined mesh file."""

    desired_dimensions: tuple[float, float, float] | None = None
    """Optional desired dimensions (width, depth, height) in meters for size-based
    ranking."""

    scene_id: str | None = None
    """Optional scene identifier for fair round-robin scheduling.

    When multiple scenes submit requests concurrently, the server uses this ID to
    group requests from the same scene together for fair processing time allocation.
    All requests with the same scene_id are treated as a single "client" in the
    round-robin scheduler. If not provided, each HTTP request is treated as a
    separate client.
    """

    num_candidates: int = 1
    """Number of candidates to return, sorted by combined score (best first).

    For router validation retries, set to max_retries + 1 to get enough candidates.
    For non-router single retrieval, use default of 1.
    """

    def to_dict(self) -> dict:
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
class ArticulatedRetrievalResult:
    """Single articulated object retrieval result.

    Represents one matched object from articulated datasets with its
    similarity scores and metadata.
    """

    mesh_path: str
    """Absolute path to the exported combined mesh file (GLTF format)."""

    sdf_path: str
    """Absolute path to the original articulated SDF file."""

    object_id: str
    """Object identifier within the dataset (e.g., PartNet-Mobility ID)."""

    source: str
    """Source dataset: 'partnet_mobility' or 'artvip'."""

    description: str
    """Human-readable object description."""

    clip_score: float
    """CLIP semantic similarity score in range [0, 1], higher is better match."""

    bbox_score: float
    """Bounding box size similarity score in range [0, 1], higher is better match."""

    bounding_box_min: list[float]
    """Minimum corner of object bounding box [x, y, z] in meters."""

    bounding_box_max: list[float]
    """Maximum corner of object bounding box [x, y, z] in meters."""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ArticulatedRetrievalResult":
        """Create result from dictionary (e.g., from JSON deserialization).

        Args:
            data: Dictionary representation of result.

        Returns:
            ArticulatedRetrievalResult instance.
        """
        return cls(
            mesh_path=data["mesh_path"],
            sdf_path=data["sdf_path"],
            object_id=data["object_id"],
            source=data["source"],
            description=data["description"],
            clip_score=data["clip_score"],
            bbox_score=data["bbox_score"],
            bounding_box_min=data["bounding_box_min"],
            bounding_box_max=data["bounding_box_max"],
        )


@dataclass
class ArticulatedRetrievalServerResponse:
    """Response payload from articulated retrieval server.

    This DTO defines the contract for responses from the articulated retrieval
    server after successful semantic search over articulated object datasets.
    """

    results: list[ArticulatedRetrievalResult]
    """List of matching articulated objects, ordered by descending combined score."""

    query_description: str
    """Echo of the original query description for debugging."""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "results": [r.to_dict() for r in self.results],
            "query_description": self.query_description,
        }

    def to_json(self) -> str:
        """Convert to JSON string for HTTP response body."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "ArticulatedRetrievalServerResponse":
        """Create response from dictionary (e.g., from JSON deserialization).

        Args:
            data: Dictionary representation of response.

        Returns:
            ArticulatedRetrievalServerResponse instance.
        """
        results = [ArticulatedRetrievalResult.from_dict(r) for r in data["results"]]
        return cls(
            results=results,
            query_description=data["query_description"],
        )


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
    """Response data for successful requests (contains results list, etc.)."""

    error: str | None = None
    """Error message for failed requests."""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string for streaming response."""
        return json.dumps(self.to_dict())
