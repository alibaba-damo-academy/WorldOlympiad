"""CLIP-based similarity search for Objaverse meshes.

Note: Objaverse uses ViT-L/14 embeddings (768-dim) which are different from
the default ViT-H-14 (1024-dim). We use a separate CLIP model loader that matches the
pre-computed embeddings.
"""

import logging

import numpy as np
import open_clip
import torch

from scenecode.agent_utils.objaverse_retrieval.data_loader import (
    ObjaversePreprocessedData,
)

console_logger = logging.getLogger(__name__)

# Cache OpenCLIP model for Objaverse (ViT-L/14, 768-dim).
_cached_model = None
_cached_tokenizer = None
_device = None


def _get_objaverse_clip_model(device: str | None = None):
    """Get cached OpenCLIP model for Objaverse (ViT-L/14) or load if not cached.

    ObjectThor embeddings were computed with ViT-L/14 using laion2b_s32b_b82k
    pretrained weights (768 dimensions). This is different from:
    - HSSD: ViT-H-14-378-quickgelu (1024 dimensions)
    - OpenAI's original CLIP: ViT-L/14 with openai weights

    Using the wrong pretrained weights will result in embedding space mismatch
    and poor retrieval results, even if dimensions match.

    Args:
        device: Target device (e.g., "cuda:0", "cuda:1", "cpu"). If None, uses
            "cuda" if available, else "cpu".

    Returns:
        Tuple of (model, tokenizer, device_str).
    """
    global _cached_model, _cached_tokenizer, _device

    # Determine target device.
    if device is not None:
        target_device = device
    elif _device is not None:
        target_device = _device
    else:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load or reload model if needed.
    if _cached_model is None or _device != target_device:
        # ObjectThor uses ViT-L/14 with laion2b_s32b_b82k weights (768-dim).
        # This MUST match the model used to compute the pre-computed embeddings.
        # See: https://github.com/allenai/Holodeck
        model_name = "ViT-L-14"
        pretrained = "laion2b_s32b_b82k"
        _device = target_device

        _cached_model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=_device
        )
        _cached_tokenizer = open_clip.get_tokenizer(model_name)

        console_logger.info(
            f"Loaded Objaverse CLIP model: {model_name} ({pretrained}) on {_device}"
        )

    return _cached_model, _cached_tokenizer, _device


def get_objaverse_text_embedding(text: str, device: str | None = None) -> np.ndarray:
    """Get CLIP text embedding for Objaverse matching.

    Uses ViT-L/14 with laion2b_s32b_b82k pretrained weights (768 dimensions)
    to match the pre-computed ObjectThor embeddings.

    Args:
        text: Text to embed.
        device: Target device (e.g., "cuda:0"). If None, uses default.

    Returns:
        Text embedding as NumPy array (768 dimensions), normalized.
    """
    model, tokenizer, device = _get_objaverse_clip_model(device=device)

    # Tokenize and encode text.
    text_tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # Normalize for cosine similarity.
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    embedding = text_features.cpu().numpy()[0]
    return embedding


def filter_meshes_by_category(
    preprocessed_data: ObjaversePreprocessedData, category: str
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

    uid_list = preprocessed_data.object_categories[category]

    mesh_indices = []
    for uid in uid_list:
        idx = preprocessed_data.get_embedding_index(uid)
        if idx is not None:
            mesh_indices.append(idx)

    console_logger.info(
        f"Filtered {len(mesh_indices)} meshes for category '{category}'"
    )

    return mesh_indices


def compute_clip_similarities(
    query_embedding: np.ndarray, embeddings: np.ndarray, indices: list[int]
) -> dict[int, float]:
    """Compute cosine similarities between query and candidate embeddings.

    Args:
        query_embedding: Query embedding (D,), should be normalized.
        embeddings: Candidate embeddings array (N, D).
        indices: List of indices in embeddings array to compare against.

    Returns:
        Dictionary mapping index to similarity score.
    """
    # Ensure query is normalized.
    query_norm = query_embedding / np.linalg.norm(query_embedding)

    # Extract selected embeddings and normalize all at once.
    selected_embeddings = embeddings[indices]
    norms = np.linalg.norm(selected_embeddings, axis=1, keepdims=True)
    # Avoid division by zero.
    norms = np.maximum(norms, 1e-8)
    selected_norms = selected_embeddings / norms

    # Vectorized dot product.
    similarities = selected_norms @ query_norm

    return dict(zip(indices, similarities))


def get_top_k_similar_meshes(
    text_description: str,
    preprocessed_data: ObjaversePreprocessedData,
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
        List of (uid, similarity_score) tuples, sorted by descending similarity.
    """
    console_logger.info(
        f"Computing CLIP similarities for '{text_description}' "
        f"(category={category}, top_k={top_k})"
    )

    text_embedding = get_objaverse_text_embedding(text_description, device=device)

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
        uid = preprocessed_data.embedding_index[mesh_idx]
        results.append((uid, similarity))

    console_logger.info(
        f"Top-{len(results)} CLIP candidates: "
        f"{[(uid[:8], f'{sim:.3f}') for uid, sim in results]}"
    )

    return results
