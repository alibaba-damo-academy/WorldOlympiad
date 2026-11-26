"""Core CLIP embedding functions using OpenCLIP.

This module provides text and image embedding functions using the OpenCLIP
ViT-H-14-378-quickgelu model with dfn5b pretrained weights (1024 dimensions).
"""

import gc
import logging

from pathlib import Path

import numpy as np
import open_clip
import torch

from PIL import Image

console_logger = logging.getLogger(__name__)

# Cache OpenCLIP model to avoid reloading on every embedding call.
_cached_model = None
_cached_tokenizer = None
_cached_preprocess = None
_device = None


def reset_clip_model_cache() -> None:
    """Clear cached OpenCLIP state and release CUDA memory if possible."""
    global _cached_model, _cached_tokenizer, _cached_preprocess, _device

    _cached_model = None
    _cached_tokenizer = None
    _cached_preprocess = None
    _device = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _get_clip_model(device: str | None = None):
    """Get cached OpenCLIP model or load if not cached.

    Args:
        device: Target device (e.g., "cuda:0", "cuda:1", "cpu"). If None, uses
            "cuda" if available, else "cpu".

    Returns:
        Tuple of (model, tokenizer, preprocess, device_str).
    """
    global _cached_model, _cached_tokenizer, _cached_preprocess, _device

    # Determine target device.
    if device is not None:
        target_device = device
    elif _device is not None:
        target_device = _device
    else:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load or reload model if needed.
    if _cached_model is None or _device != target_device:
        model_name = "ViT-H-14-378-quickgelu"
        pretrained = "dfn5b"
        _device = target_device

        _cached_model, _, _cached_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=_device
        )
        _cached_tokenizer = open_clip.get_tokenizer(model_name)

        console_logger.info(
            f"Loaded OpenCLIP model: {model_name} ({pretrained}) on {_device}"
        )

    return _cached_model, _cached_tokenizer, _cached_preprocess, _device


def get_text_embedding(text: str, device: str | None = None) -> np.ndarray:
    """Get CLIP text embedding using OpenCLIP.

    Uses ViT-H-14-378-quickgelu with dfn5b pretrained weights (1024 dimensions).

    Args:
        text: Text to embed.
        device: Target device (e.g., "cuda:0"). If None, uses default.

    Returns:
        Text embedding as NumPy array (1024 dimensions), normalized.
    """
    model, tokenizer, _, device = _get_clip_model(device=device)

    # Tokenize and encode text.
    text_tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # Normalize for cosine similarity.
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    embedding = text_features.cpu().numpy()[0]
    return embedding


def get_single_image_embedding(
    image_path: Path, device: str | None = None
) -> np.ndarray:
    """Get CLIP embedding for a single image.

    Args:
        image_path: Path to image file.
        device: Target device (e.g., "cuda:0"). If None, uses default.

    Returns:
        Image embedding as NumPy array (1024 dimensions), normalized.
    """
    model, _, preprocess, device = _get_clip_model(device=device)

    with Image.open(image_path) as image:
        image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    embedding = image_features.cpu().numpy()[0]
    return embedding


def get_multiview_image_embedding(
    image_paths: list[Path], device: str | None = None
) -> np.ndarray:
    """Get averaged CLIP image embedding from multiple views.

    Computes CLIP embeddings for each image and averages them,
    following the standard multi-view embedding approach.

    Args:
        image_paths: List of paths to image files (e.g., 8 rendered views).
        device: Target device (e.g., "cuda:0"). If None, uses default.

    Returns:
        Averaged image embedding as NumPy array (1024 dimensions), normalized.

    Raises:
        ValueError: If image_paths is empty.
    """
    if not image_paths:
        raise ValueError("image_paths cannot be empty")

    model, _, preprocess, device = _get_clip_model(device=device)

    # Batch process all images.
    image_tensors = []
    for path in image_paths:
        with Image.open(path) as image:
            image_tensors.append(preprocess(image.convert("RGB")))

    # Stack into batch tensor.
    batch_tensor = torch.stack(image_tensors).to(device)

    with torch.no_grad():
        image_features = model.encode_image(batch_tensor)
        # Normalize each embedding individually.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Average across views.
    averaged_features = image_features.mean(dim=0)
    # Re-normalize the averaged embedding.
    averaged_features = averaged_features / averaged_features.norm()

    embedding = averaged_features.cpu().numpy()
    return embedding


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
    selected_norms = selected_embeddings / np.linalg.norm(
        selected_embeddings, axis=1, keepdims=True
    )

    # Vectorized dot product.
    similarities = selected_norms @ query_norm

    return dict(zip(indices, similarities))
