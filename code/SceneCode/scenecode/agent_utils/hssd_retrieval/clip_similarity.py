"""CLIP-based similarity search for HSSD meshes."""

import logging

from scenecode.agent_utils.clip_embeddings import (
    compute_clip_similarities,
    get_text_embedding,
)
from scenecode.agent_utils.hssd_retrieval.data_loader import HssdPreprocessedData

console_logger = logging.getLogger(__name__)


def filter_meshes_by_category(
    preprocessed_data: HssdPreprocessedData, category: str
) -> list[int]:
    """Filter mesh indices by object category.

    Args:
        preprocessed_data: Loaded preprocessed data.
        category: Object category (e.g., "large_objects", "small_objects").

    Returns:
        List of mesh embedding indices for meshes in this category.
    """
    if category not in preprocessed_data.object_categories:
        console_logger.warning(
            f"Category {category} not found in object_categories. "
            f"Available: {list(preprocessed_data.object_categories.keys())}"
        )
        return []

    wordnet_keys = preprocessed_data.object_categories[category]

    mesh_indices = []
    for wordnet_key in wordnet_keys:
        if wordnet_key not in preprocessed_data.metadata_by_wordnet:
            continue

        for metadata in preprocessed_data.metadata_by_wordnet[wordnet_key]:
            idx = preprocessed_data.get_embedding_index(metadata.mesh_id)
            if idx is not None:
                mesh_indices.append(idx)

    console_logger.info(
        f"Filtered {len(mesh_indices)} meshes for category '{category}'"
    )

    return mesh_indices


def get_top_k_similar_meshes(
    text_description: str,
    preprocessed_data: HssdPreprocessedData,
    category: str | None = None,
    top_k: int = 5,
    device: str | None = None,
) -> list[tuple[str, float]]:
    """Get top-K most similar meshes to text description.

    Args:
        text_description: Object description text.
        preprocessed_data: Loaded preprocessed data.
        category: Optional object category to filter by.
        top_k: Number of top candidates to return.
        device: Target CLIP device (e.g., "cuda:0"). If None, uses default.

    Returns:
        List of (mesh_id, similarity_score) tuples, sorted by descending similarity.
    """
    console_logger.info(
        f"Computing CLIP similarities for '{text_description}' "
        f"(category={category}, top_k={top_k})"
    )

    text_embedding = get_text_embedding(text_description, device=device)

    if category:
        mesh_indices = filter_meshes_by_category(preprocessed_data, category)
    else:
        mesh_indices = list(range(len(preprocessed_data.embedding_index)))

    if not mesh_indices:
        console_logger.warning("No meshes to search")
        return []

    similarities = compute_clip_similarities(
        query_embedding=text_embedding,
        embeddings=preprocessed_data.clip_embeddings,
        indices=mesh_indices,
    )

    sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    top_k_items = sorted_items[:top_k]

    results = []
    for mesh_idx, similarity in top_k_items:
        mesh_id = preprocessed_data.embedding_index[mesh_idx]
        results.append((mesh_id, similarity))

    console_logger.info(
        f"Top-{len(results)} CLIP candidates: "
        f"{[(mesh_id[:8], f'{sim:.3f}') for mesh_id, sim in results]}"
    )

    return results
