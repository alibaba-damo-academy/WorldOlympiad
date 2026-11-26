"""Data loading utilities for articulated object preprocessed embeddings."""

import logging

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from scenecode.agent_utils.articulated_retrieval_server.config import (
    ArticulatedConfig,
    ArticulatedSourceConfig,
)

console_logger = logging.getLogger(__name__)


@dataclass
class ArticulatedMeshMetadata:
    """Metadata for a single articulated object."""

    object_id: str
    """Object ID within the source dataset."""

    source: str
    """Data source name ('partnet_mobility' or 'artvip')."""

    category: str
    """Object category (e.g., 'Cabinet', 'StorageFurniture')."""

    description: str
    """Text description for display and retrieval."""

    is_manipuland: bool
    """Whether this is a manipuland (can be picked up)."""

    placement_type: str
    """Placement type: 'floor', 'wall', 'ceiling', or 'on_object'."""

    sdf_path: Path
    """Path to the SDF file."""

    bounding_box_min: list[float]
    """Bounding box minimum [x, y, z] in canonical local frame (joints=0)."""

    bounding_box_max: list[float]
    """Bounding box maximum [x, y, z] in canonical local frame (joints=0)."""


@dataclass
class ArticulatedPreprocessedData:
    """Container for preprocessed articulated object data from one or more sources."""

    metadata_by_id: dict[str, ArticulatedMeshMetadata]
    """Maps object_id to metadata."""

    clip_embeddings: np.ndarray
    """CLIP embeddings array (N, 1024)."""

    embedding_index: list[str]
    """Maps array index to object_id."""

    _embedding_idx_lookup: dict[str, int] = field(
        init=False, default_factory=dict, repr=False
    )
    """Private O(1) lookup index from object_id to embedding index."""

    def __post_init__(self):
        """Build embedding index lookup."""
        self._embedding_idx_lookup = {
            obj_id: idx for idx, obj_id in enumerate(self.embedding_index)
        }

    def get_metadata(self, object_id: str) -> ArticulatedMeshMetadata | None:
        """Get metadata for a specific object ID (O(1) lookup).

        Args:
            object_id: Object ID to look up.

        Returns:
            Object metadata if found, None otherwise.
        """
        return self.metadata_by_id.get(object_id)

    def get_embedding_index(self, object_id: str) -> int | None:
        """Get the embedding array index for an object ID (O(1) lookup).

        Args:
            object_id: Object ID to look up.

        Returns:
            Array index if found, None otherwise.
        """
        return self._embedding_idx_lookup.get(object_id)

    def filter_by_object_type(self, object_type: str) -> list[int]:
        """Filter embedding indices by object type.

        Filtering logic:
        - FURNITURE: is_manipuland=False AND placement_type="floor"
        - MANIPULAND: is_manipuland=True
        - WALL_MOUNTED: placement_type="wall"
        - CEILING_MOUNTED: placement_type="ceiling"

        Args:
            object_type: Object type string (FURNITURE, MANIPULAND, etc.).

        Returns:
            List of embedding indices matching the filter.
        """
        matching_indices = []
        for object_id, metadata in self.metadata_by_id.items():
            matches = False

            if object_type == "FURNITURE":
                matches = (
                    not metadata.is_manipuland and metadata.placement_type == "floor"
                )
            elif object_type == "MANIPULAND":
                matches = metadata.is_manipuland
            elif object_type == "WALL_MOUNTED":
                matches = metadata.placement_type == "wall"
            elif object_type == "CEILING_MOUNTED":
                matches = metadata.placement_type == "ceiling"
            else:
                # Unknown type - warn and include all.
                console_logger.warning(
                    f"Unknown object_type '{object_type}', falling back to all"
                )
                matches = True

            if matches:
                idx = self.get_embedding_index(object_id)
                if idx is not None:
                    matching_indices.append(idx)

        console_logger.debug(
            f"Filtered {len(matching_indices)} objects for type '{object_type}'"
        )
        return matching_indices


def get_placement_type_from_options(placement_options: dict) -> str:
    """Convert placement options dict to canonical placement type string.

    Floor takes precedence if multiple options are true.

    Args:
        placement_options: Dict with on_floor, on_wall, on_ceiling, on_object keys.

    Returns:
        Placement type string.
    """
    if placement_options.get("on_floor", False):
        return "floor"
    elif placement_options.get("on_wall", False):
        return "wall"
    elif placement_options.get("on_ceiling", False):
        return "ceiling"
    elif placement_options.get("on_object", False):
        return "on_object"
    return "floor"  # Default.


def load_preprocessed_data(
    source_config: ArticulatedSourceConfig,
) -> ArticulatedPreprocessedData | None:
    """Load preprocessed data for a single articulated data source.

    Args:
        source_config: Configuration for the data source.

    Returns:
        Loaded preprocessed data, or None if loading fails.
    """
    if not source_config.enabled:
        console_logger.info(f"Source '{source_config.name}' is disabled, skipping")
        return None

    embeddings_path = source_config.embeddings_path
    console_logger.info(
        f"Loading articulated preprocessed data for '{source_config.name}' "
        f"from {embeddings_path}"
    )

    # Check required files exist.
    clip_embeddings_file = embeddings_path / "clip_embeddings.npy"
    embedding_index_file = embeddings_path / "embedding_index.yaml"
    metadata_index_file = embeddings_path / "metadata_index.yaml"
    for file_path in [clip_embeddings_file, embedding_index_file, metadata_index_file]:
        if not file_path.exists():
            console_logger.warning(
                f"Required file not found for source '{source_config.name}': {file_path}"
            )
            return None

    # Load CLIP embeddings.
    clip_embeddings = np.load(clip_embeddings_file)
    console_logger.info(
        f"Loaded CLIP embeddings for '{source_config.name}': "
        f"shape={clip_embeddings.shape}, dtype={clip_embeddings.dtype}"
    )

    # Load embedding index.
    with open(embedding_index_file, "r") as f:
        embedding_index = yaml.safe_load(f)

    # Load metadata index.
    # Use unsafe_load to handle numpy objects saved during embedding generation.
    # This is safe because we control the metadata files.
    with open(metadata_index_file, "r") as f:
        metadata_raw = yaml.unsafe_load(f)

    # Build metadata dict.
    metadata_by_id: dict[str, ArticulatedMeshMetadata] = {}
    for obj_id, meta in metadata_raw.items():
        # Resolve SDF path relative to data_path.
        sdf_rel_path = meta.get("sdf_path", "")
        sdf_path = source_config.data_path / sdf_rel_path

        # Get placement type from placement_options or directly.
        if "placement_options" in meta:
            placement_type = get_placement_type_from_options(
                placement_options=meta["placement_options"]
            )
        else:
            placement_type = meta.get("placement_type", "floor")

        # Bounding box is required - computed during embedding generation.
        if "bounding_box_min" not in meta or "bounding_box_max" not in meta:
            raise ValueError(
                f"Missing bounding box for object '{obj_id}' in source "
                f"'{source_config.name}'. Re-run compute_articulated_embeddings.py."
            )

        metadata_by_id[obj_id] = ArticulatedMeshMetadata(
            object_id=obj_id,
            source=source_config.name,
            category=meta.get("category", "Unknown"),
            description=meta.get("description", ""),
            is_manipuland=meta.get("is_manipuland", False),
            placement_type=placement_type,
            sdf_path=sdf_path,
            bounding_box_min=meta["bounding_box_min"],
            bounding_box_max=meta["bounding_box_max"],
        )

    console_logger.info(
        f"Loaded {len(metadata_by_id)} objects from source '{source_config.name}'"
    )

    return ArticulatedPreprocessedData(
        metadata_by_id=metadata_by_id,
        clip_embeddings=clip_embeddings,
        embedding_index=embedding_index,
    )


def load_preprocessed_data_multi_source(
    config: ArticulatedConfig,
) -> ArticulatedPreprocessedData | None:
    """Load and combine preprocessed data from multiple sources.

    Args:
        config: Articulated configuration with multiple sources.

    Returns:
        Combined preprocessed data from all enabled sources, or None if no data loaded.
    """
    if not config.has_enabled_sources:
        console_logger.warning("No enabled articulated sources configured")
        return None

    # Load data from each enabled source.
    source_data: list[ArticulatedPreprocessedData] = []
    for source_cfg in config.enabled_sources.values():
        data = load_preprocessed_data(source_cfg)
        if data is not None:
            source_data.append(data)

    if not source_data:
        console_logger.warning("No articulated data loaded from any source")
        return None

    # If only one source, return it directly.
    if len(source_data) == 1:
        console_logger.info("Single source loaded, returning directly")
        return source_data[0]

    # Combine multiple sources.
    console_logger.info(f"Combining {len(source_data)} articulated data sources")

    combined_metadata: dict[str, ArticulatedMeshMetadata] = {}
    combined_embeddings_list: list[np.ndarray] = []
    combined_embedding_index: list[str] = []
    for data in source_data:
        # Check for duplicate object_ids.
        for obj_id in data.metadata_by_id:
            if obj_id in combined_metadata:
                raise ValueError(
                    f"Duplicate object_id '{obj_id}' found: exists in source "
                    f"'{combined_metadata[obj_id].source}' and "
                    f"'{data.metadata_by_id[obj_id].source}'. "
                    f"Object IDs must be unique across all sources."
                )

        # Merge metadata.
        combined_metadata.update(data.metadata_by_id)

        # Track embeddings and indices.
        combined_embeddings_list.append(data.clip_embeddings)
        combined_embedding_index.extend(data.embedding_index)

    # Concatenate embeddings.
    combined_embeddings = np.concatenate(combined_embeddings_list, axis=0)

    console_logger.info(
        f"Combined articulated data: {len(combined_metadata)} objects, "
        f"embeddings shape {combined_embeddings.shape}"
    )

    return ArticulatedPreprocessedData(
        metadata_by_id=combined_metadata,
        clip_embeddings=combined_embeddings,
        embedding_index=combined_embedding_index,
    )
