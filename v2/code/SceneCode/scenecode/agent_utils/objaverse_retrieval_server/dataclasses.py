"""Dataclasses for Objaverse retrieval server API contracts.

This module contains serializable Data Transfer Objects (DTOs) used for
communication with the Objaverse retrieval server. These classes define the
HTTP API contract and use primitive types for JSON serialization.
"""

import json

from dataclasses import asdict, dataclass


@dataclass
class ObjaverseRetrievalServerRequest:
    """Request payload for Objaverse retrieval server.

    This DTO defines the contract for Objaverse asset retrieval requests sent
    to the Objaverse retrieval server via HTTP. Contains all information needed
    to perform semantic search over the ObjectThor library using CLIP.
    """

    object_description: str
    """Text description of the object to retrieve (e.g., 'Modern wooden desk')."""

    object_type: str
    """Type of object to retrieve (e.g., 'FURNITURE', 'MANIPULAND')."""

    output_dir: str
    """Client-specified output directory where server will export mesh file."""

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
    """Number of candidates to return, sorted by bbox_score (best first).

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
class ObjaverseRetrievalResult:
    """Single Objaverse object retrieval result.

    Represents one matched object from the ObjectThor library with its
    similarity score and metadata.
    """

    mesh_path: str
    """Absolute path to the exported mesh file (GLB format)."""

    objaverse_uid: str
    """Objaverse/ObjectThor unique identifier."""

    object_name: str
    """Human-readable object name from description."""

    similarity_score: float
    """CLIP similarity score in range [0, 1], higher is better match."""

    size: tuple[float, float, float]
    """Object size (width, depth, height) in meters."""

    category: str
    """Object category (e.g., 'large_objects', 'small_objects')."""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ObjaverseRetrievalResult":
        """Create result from dictionary (e.g., from JSON deserialization).

        Args:
            data: Dictionary representation of result.

        Returns:
            ObjaverseRetrievalResult instance.
        """
        # JSON deserializes tuples as lists, so convert back to tuple.
        return cls(
            mesh_path=data["mesh_path"],
            objaverse_uid=data["objaverse_uid"],
            object_name=data["object_name"],
            similarity_score=data["similarity_score"],
            size=tuple(data["size"]),
            category=data["category"],
        )


@dataclass
class ObjaverseRetrievalServerResponse:
    """Response payload from Objaverse retrieval server.

    This DTO defines the contract for responses from the Objaverse retrieval
    server after successful semantic search over the ObjectThor library.
    """

    results: list[ObjaverseRetrievalResult]
    """List of matching Objaverse objects, ordered by descending similarity score."""

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
    def from_dict(cls, data: dict) -> "ObjaverseRetrievalServerResponse":
        """Create response from dictionary (e.g., from JSON deserialization).

        Args:
            data: Dictionary representation of response.

        Returns:
            ObjaverseRetrievalServerResponse instance.
        """
        results = [ObjaverseRetrievalResult.from_dict(r) for r in data["results"]]
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
