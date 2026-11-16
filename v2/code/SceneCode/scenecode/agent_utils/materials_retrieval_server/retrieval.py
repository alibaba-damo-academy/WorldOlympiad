"""Materials retrieval using CLIP similarity."""

import logging

from dataclasses import dataclass

from scenecode.agent_utils.clip_embeddings import (
    compute_clip_similarities,
    get_text_embedding,
)
from scenecode.agent_utils.materials_retrieval_server.config import MaterialsConfig
from scenecode.agent_utils.materials_retrieval_server.data_loader import (
    MaterialsPreprocessedData,
    load_preprocessed_data,
)

console_logger = logging.getLogger(__name__)


@dataclass
class MaterialRetrievalCandidate:
    """A candidate material from retrieval."""

    material_id: str
    """Material ID (e.g., 'Bricks001', 'Wood094')."""

    clip_score: float
    """CLIP similarity score (higher is better)."""

    category: str
    """Material category (e.g., 'Bricks', 'Wood Floor')."""

    tags: list[str]
    """Descriptive tags."""


class MaterialsRetriever:
    """Retriever for materials using CLIP semantic search.

    Single-stage retrieval:
    - CLIP semantic filtering: Get top-k candidates by text similarity

    Unlike articulated objects, materials don't have size constraints,
    so no bounding box ranking is needed.
    """

    def __init__(self, config: MaterialsConfig, clip_device: str | None = None) -> None:
        """Initialize retriever with configuration.

        Args:
            config: Materials retrieval configuration.
            clip_device: Target device for CLIP model (e.g., "cuda:0"). If None,
                uses default (cuda if available, else cpu).
        """
        self.config = config
        self.clip_device = clip_device
        self.preprocessed_data: MaterialsPreprocessedData | None = None
        self._initialized = False

    def initialize(self) -> bool:
        """Load preprocessed data.

        Returns:
            True if initialization succeeded and data is available.
        """
        if self._initialized:
            return self.preprocessed_data is not None

        self.preprocessed_data = load_preprocessed_data(self.config)
        self._initialized = True

        if self.preprocessed_data is None:
            console_logger.warning("Failed to load materials preprocessed data")
            return False

        console_logger.info(
            f"Materials retriever initialized with "
            f"{len(self.preprocessed_data.metadata_by_id)} materials"
        )
        return True

    def retrieve(
        self,
        description: str,
        top_k: int = 5,
    ) -> list[MaterialRetrievalCandidate]:
        """Retrieve materials matching description.

        Args:
            description: Text description of desired material (e.g., "warm hardwood
                floor", "red brick wall").
            top_k: Number of candidates to return.

        Returns:
            List of candidates sorted by CLIP similarity (best first).
        """
        if not self.initialize():
            console_logger.error("Retriever not initialized, cannot retrieve")
            return []

        assert self.preprocessed_data is not None

        console_logger.info(
            f"Retrieving materials: description='{description}', top_k={top_k}"
        )

        # Get text embedding for the query.
        text_embedding = get_text_embedding(description, device=self.clip_device)

        # Compute similarities with all materials.
        all_indices = list(range(len(self.preprocessed_data.embedding_index)))
        similarities = compute_clip_similarities(
            query_embedding=text_embedding,
            embeddings=self.preprocessed_data.clip_embeddings,
            indices=all_indices,
        )

        # Sort by CLIP similarity (descending).
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Take top-k candidates.
        top_results = sorted_results[: min(top_k, len(sorted_results))]

        best_score = f"{top_results[0][1]:.3f}" if top_results else "N/A"
        console_logger.info(
            f"Top {len(top_results)} CLIP candidates, best score={best_score}"
        )

        # Build candidate objects with metadata.
        candidates: list[MaterialRetrievalCandidate] = []
        for emb_idx, clip_score in top_results:
            material_id = self.preprocessed_data.embedding_index[emb_idx]
            metadata = self.preprocessed_data.get_metadata(material_id)

            if metadata is None:
                console_logger.warning(f"Missing metadata for material {material_id}")
                continue

            candidates.append(
                MaterialRetrievalCandidate(
                    material_id=material_id,
                    clip_score=clip_score,
                    category=metadata.category,
                    tags=metadata.tags,
                )
            )

        if candidates:
            summary = [
                (c.material_id, f"score={c.clip_score:.3f}", c.category)
                for c in candidates[:3]
            ]
            console_logger.info(f"Returning {len(candidates)} candidates: {summary}...")

        return candidates
