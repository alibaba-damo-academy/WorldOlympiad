"""Data loading utilities for materials preprocessed embeddings."""

import logging

from dataclasses import dataclass, field

import numpy as np
import yaml

from scenecode.agent_utils.materials_retrieval_server.config import MaterialsConfig

console_logger = logging.getLogger(__name__)


@dataclass
class MaterialMetadata:
    """Metadata for a single material."""

    material_id: str
    """Material ID (e.g., 'Bricks001', 'Wood094')."""

    category: str
    """Material category (e.g., 'Bricks', 'Wood Floor', 'Carpet')."""

    tags: list[str]
    """Descriptive tags (e.g., ['dark', 'red', 'smooth'])."""


@dataclass
class MaterialsPreprocessedData:
    """Container for preprocessed materials data."""

    metadata_by_id: dict[str, MaterialMetadata]
    """Maps material_id to metadata."""

    clip_embeddings: np.ndarray
    """CLIP embeddings array (N, 1024)."""

    embedding_index: list[str]
    """Maps array index to material_id."""

    _embedding_idx_lookup: dict[str, int] = field(
        init=False, default_factory=dict, repr=False
    )
    """Private O(1) lookup index from material_id to embedding index."""

    def __post_init__(self) -> None:
        """Build embedding index lookup."""
        self._embedding_idx_lookup = {
            mat_id: idx for idx, mat_id in enumerate(self.embedding_index)
        }

    def get_metadata(self, material_id: str) -> MaterialMetadata | None:
        """Get metadata for a specific material ID (O(1) lookup).

        Args:
            material_id: Material ID to look up.

        Returns:
            Material metadata if found, None otherwise.
        """
        return self.metadata_by_id.get(material_id)

    def get_embedding_index(self, material_id: str) -> int | None:
        """Get the embedding array index for a material ID (O(1) lookup).

        Args:
            material_id: Material ID to look up.

        Returns:
            Array index if found, None otherwise.
        """
        return self._embedding_idx_lookup.get(material_id)


def load_preprocessed_data(config: MaterialsConfig) -> MaterialsPreprocessedData | None:
    """Load preprocessed materials data.

    Args:
        config: Materials configuration.

    Returns:
        Loaded preprocessed data, or None if loading fails or disabled.
    """
    if not config.enabled:
        console_logger.info("Materials retrieval is disabled, skipping data load")
        return None

    embeddings_path = config.embeddings_path
    console_logger.info(f"Loading materials preprocessed data from {embeddings_path}")

    # Load CLIP embeddings.
    clip_embeddings_file = embeddings_path / "clip_embeddings.npy"
    clip_embeddings = np.load(clip_embeddings_file)
    console_logger.info(
        f"Loaded CLIP embeddings: shape={clip_embeddings.shape}, "
        f"dtype={clip_embeddings.dtype}"
    )

    # Load embedding index.
    embedding_index_file = embeddings_path / "embedding_index.yaml"
    with open(embedding_index_file, "r") as f:
        embedding_index = yaml.safe_load(f)

    # Load metadata index.
    metadata_index_file = embeddings_path / "metadata_index.yaml"
    with open(metadata_index_file, "r") as f:
        metadata_raw = yaml.safe_load(f)

    # Build metadata dict.
    metadata_by_id: dict[str, MaterialMetadata] = {}
    for mat_id, meta in metadata_raw.items():
        metadata_by_id[mat_id] = MaterialMetadata(
            material_id=mat_id,
            category=meta.get("category", "Unknown"),
            tags=meta.get("tags", []),
        )

    console_logger.info(f"Loaded {len(metadata_by_id)} materials")

    return MaterialsPreprocessedData(
        metadata_by_id=metadata_by_id,
        clip_embeddings=clip_embeddings,
        embedding_index=embedding_index,
    )
