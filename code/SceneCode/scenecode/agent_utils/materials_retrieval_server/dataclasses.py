"""Dataclasses for materials retrieval server API contracts.

This module contains serializable Data Transfer Objects (DTOs) used for
communication with the materials retrieval server. These classes define the
HTTP API contract and use primitive types for JSON serialization.
"""

import json

from dataclasses import asdict, dataclass


@dataclass
class MaterialsRetrievalServerRequest:
    """Request payload for materials retrieval server.

    This DTO defines the contract for material retrieval requests sent
    to the materials retrieval server via HTTP.
    """

    material_description: str
    """Text description of the material to retrieve (e.g., 'warm hardwood floor')."""

    output_dir: str
    """Client-specified output directory where server will copy material files."""

    scene_id: str | None = None
    """Optional scene identifier for fair round-robin scheduling.

    When multiple scenes submit requests concurrently, the server uses this ID to
    group requests from the same scene together for fair processing time allocation.
    """

    num_candidates: int = 1
    """Number of candidates to return, sorted by CLIP score (best first)."""

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
class MaterialRetrievalResult:
    """Single material retrieval result.

    Represents one matched material with its similarity score and metadata.
    """

    material_path: str
    """Absolute path to the copied material folder in output_dir."""

    material_id: str
    """Material identifier (e.g., 'Wood094', 'Bricks001')."""

    similarity_score: float
    """CLIP similarity score in range [0, 1], higher is better match."""

    category: str
    """Material category (e.g., 'Wood Floor', 'Bricks', 'Carpet')."""

    color_texture: str
    """Absolute path to the color/albedo texture file."""

    normal_texture: str
    """Absolute path to the normal map texture file."""

    roughness_texture: str
    """Absolute path to the roughness texture file."""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MaterialRetrievalResult":
        """Create result from dictionary (e.g., from JSON deserialization).

        Args:
            data: Dictionary representation of result.

        Returns:
            MaterialRetrievalResult instance.
        """
        return cls(
            material_path=data["material_path"],
            material_id=data["material_id"],
            similarity_score=data["similarity_score"],
            category=data["category"],
            color_texture=data["color_texture"],
            normal_texture=data["normal_texture"],
            roughness_texture=data["roughness_texture"],
        )


@dataclass
class MaterialsRetrievalServerResponse:
    """Response payload from materials retrieval server.

    This DTO defines the contract for responses from the materials retrieval
    server after successful semantic search.
    """

    results: list[MaterialRetrievalResult]
    """List of matching materials, ordered by descending CLIP score."""

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
    def from_dict(cls, data: dict) -> "MaterialsRetrievalServerResponse":
        """Create response from dictionary (e.g., from JSON deserialization).

        Args:
            data: Dictionary representation of response.

        Returns:
            MaterialsRetrievalServerResponse instance.
        """
        results = [MaterialRetrievalResult.from_dict(r) for r in data["results"]]
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
