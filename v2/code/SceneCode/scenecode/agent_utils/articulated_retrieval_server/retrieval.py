"""Articulated object retrieval using CLIP similarity and bounding box ranking."""

import logging

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scenecode.agent_utils.articulated_retrieval_server.config import ArticulatedConfig
from scenecode.agent_utils.articulated_retrieval_server.data_loader import (
    ArticulatedMeshMetadata,
    ArticulatedPreprocessedData,
    load_preprocessed_data_multi_source,
)
from scenecode.agent_utils.clip_embeddings import (
    compute_clip_similarities,
    get_text_embedding,
)

console_logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    """A candidate articulated object from retrieval."""

    object_id: str
    """Object ID within the source dataset."""

    source: str
    """Data source name."""

    sdf_path: Path
    """Path to the SDF file."""

    description: str
    """Object description."""

    clip_score: float
    """CLIP similarity score (higher is better)."""

    bbox_score: float
    """Bounding box L1 distance score (lower is better)."""

    bounding_box_min: list[float]
    """Bounding box minimum [x, y, z] in canonical local frame (joints=0)."""

    bounding_box_max: list[float]
    """Bounding box maximum [x, y, z] in canonical local frame (joints=0)."""


class ArticulatedRetriever:
    """Retriever for articulated objects using CLIP + bounding box ranking.

    Two-stage retrieval:
    1. CLIP semantic filtering: Get top-k candidates by text similarity
    2. Bounding box ranking: Sort by L1 distance to desired dimensions
    """

    def __init__(
        self, config: ArticulatedConfig, clip_device: str | None = None
    ) -> None:
        """Initialize retriever with configuration.

        Args:
            config: Articulated retrieval configuration.
            clip_device: Target device for CLIP model (e.g., "cuda:0"). If None,
                uses default (cuda if available, else cpu).
        """
        self.config = config
        self.clip_device = clip_device
        self.preprocessed_data: ArticulatedPreprocessedData | None = None
        self._initialized = False

    def initialize(self) -> bool:
        """Load preprocessed data from all enabled sources.

        Returns:
            True if initialization succeeded and data is available.
        """
        if self._initialized:
            return self.preprocessed_data is not None

        self.preprocessed_data = load_preprocessed_data_multi_source(self.config)
        self._initialized = True

        if self.preprocessed_data is None:
            console_logger.warning("Failed to load articulated preprocessed data")
            return False

        console_logger.info(
            f"Articulated retriever initialized with "
            f"{len(self.preprocessed_data.metadata_by_id)} objects"
        )
        return True

    def retrieve(
        self,
        description: str,
        object_type: str,
        desired_dimensions: list[float] | None = None,
        top_k: int = 5,
    ) -> list[RetrievalCandidate]:
        """Retrieve articulated objects matching description.

        Args:
            description: Text description of desired object.
            object_type: Object type (FURNITURE, MANIPULAND, etc.).
            desired_dimensions: Optional [width, depth, height] in meters.
            top_k: Number of candidates to return.

        Returns:
            List of candidates sorted by relevance (best first).
        """
        if not self.initialize():
            console_logger.error("Retriever not initialized, cannot retrieve")
            return []

        assert self.preprocessed_data is not None

        console_logger.info(
            f"Retrieving articulated objects: description='{description}', "
            f"type={object_type}, dimensions={desired_dimensions}, top_k={top_k}"
        )

        # Stage 1: Filter by object type.
        filtered_indices = self.preprocessed_data.filter_by_object_type(object_type)

        if not filtered_indices:
            console_logger.warning(f"No articulated objects match type '{object_type}'")
            return []

        console_logger.info(
            f"Stage 1: {len(filtered_indices)} objects match type '{object_type}'"
        )

        # Stage 2: CLIP semantic ranking.
        text_embedding = get_text_embedding(description, device=self.clip_device)
        similarities = compute_clip_similarities(
            query_embedding=text_embedding,
            embeddings=self.preprocessed_data.clip_embeddings,
            indices=filtered_indices,
        )

        # Sort by CLIP similarity (descending) and take top candidates.
        sorted_by_clip = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Use configured top_k for CLIP pool.
        clip_pool_size = min(len(sorted_by_clip), self.config.use_top_k)
        clip_candidates = sorted_by_clip[:clip_pool_size]

        best_score = f"{clip_candidates[0][1]:.3f}" if clip_candidates else "N/A"
        console_logger.info(
            f"Stage 2: Top {len(clip_candidates)} CLIP candidates, "
            f"best score={best_score}"
        )

        # Build candidates with metadata.
        candidates: list[RetrievalCandidate] = []
        for emb_idx, clip_score in clip_candidates:
            object_id = self.preprocessed_data.embedding_index[emb_idx]
            metadata = self.preprocessed_data.get_metadata(object_id)

            if metadata is None:
                console_logger.warning(f"Missing metadata for object {object_id}")
                continue

            # Compute bounding box score if dimensions provided.
            if desired_dimensions is not None:
                bbox_score = self._compute_bbox_score(
                    metadata=metadata, desired_dimensions=desired_dimensions
                )
            else:
                bbox_score = 0.0

            candidates.append(
                RetrievalCandidate(
                    object_id=object_id,
                    source=metadata.source,
                    sdf_path=metadata.sdf_path,
                    description=metadata.description,
                    clip_score=clip_score,
                    bbox_score=bbox_score,
                    bounding_box_min=metadata.bounding_box_min,
                    bounding_box_max=metadata.bounding_box_max,
                )
            )

        # Stage 3: Sort by bounding box score if dimensions provided.
        if desired_dimensions is not None:
            # Lower bbox_score is better (L1 distance).
            candidates.sort(key=lambda c: c.bbox_score)
            console_logger.info(
                f"Stage 3: Re-ranked by bbox, best score={candidates[0].bbox_score:.3f}"
                if candidates
                else "Stage 3: No candidates to rank"
            )
        else:
            # Keep CLIP order if no dimensions.
            candidates.sort(key=lambda c: c.clip_score, reverse=True)

        # Return top_k results.
        results = candidates[:top_k]

        if results:
            # Build summary list for logging.
            summary = [
                (
                    c.object_id[:20],
                    f"clip={c.clip_score:.3f}",
                    f"bbox={c.bbox_score:.3f}",
                )
                for c in results
            ]
            console_logger.info(f"Returning {len(results)} candidates: {summary}")

        return results

    def _compute_bbox_score(
        self, metadata: ArticulatedMeshMetadata, desired_dimensions: list[float]
    ) -> float:
        """Compute L1 distance between object and desired dimensions.

        Args:
            metadata: Object metadata with bounding box.
            desired_dimensions: Desired [width, depth, height] in meters.

        Returns:
            L1 distance score (lower is better).
        """
        bbox_min = np.array(metadata.bounding_box_min)
        bbox_max = np.array(metadata.bounding_box_max)
        actual_dims = bbox_max - bbox_min

        desired = np.array(desired_dimensions)

        # L1 distance.
        return float(np.sum(np.abs(actual_dims - desired)))
