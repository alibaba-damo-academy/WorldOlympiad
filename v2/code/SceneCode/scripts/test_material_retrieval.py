#!/usr/bin/env python3
"""Test AmbientCG material retrieval with CLIP embeddings.

This script tests CLIP-based retrieval by querying with text and displaying
retrieved material information.

Usage:
    python scripts/test_material_retrieval.py \
        --materials-dir data/materials \
        --query "red brick wall" \
        --top-k 5

    # Save preview images to output directory
    python scripts/test_material_retrieval.py \
        --materials-dir data/materials \
        --query "wooden floor" \
        --top-k 3 \
        --output-path material_test
"""

import argparse
import logging
import shutil

from pathlib import Path

import numpy as np
import yaml

from scenecode.agent_utils.clip_embeddings import (
    compute_clip_similarities,
    get_text_embedding,
)

console_logger = logging.getLogger(__name__)


def load_material_data(
    embeddings_path: Path,
) -> tuple[np.ndarray, list[str], dict]:
    """Load material embeddings and metadata.

    Args:
        embeddings_path: Path to embeddings directory.

    Returns:
        Tuple of (embeddings, material_ids, metadata_dict).
    """
    embeddings = np.load(embeddings_path / "clip_embeddings.npy")

    with open(embeddings_path / "embedding_index.yaml") as f:
        material_ids = yaml.safe_load(f)

    with open(embeddings_path / "metadata_index.yaml") as f:
        metadata = yaml.safe_load(f)

    return embeddings, material_ids, metadata


def retrieve_top_k(
    query: str,
    embeddings: np.ndarray,
    material_ids: list[str],
    top_k: int,
) -> list[tuple[str, float]]:
    """Retrieve top-k materials by CLIP similarity.

    Args:
        query: Text query.
        embeddings: Embedding matrix (N x D).
        material_ids: List of material IDs corresponding to embeddings.
        top_k: Number of results to return.

    Returns:
        List of (material_id, similarity_score) tuples.
    """
    text_embedding = get_text_embedding(query)

    all_indices = list(range(len(material_ids)))
    similarities = compute_clip_similarities(
        query_embedding=text_embedding,
        embeddings=embeddings,
        indices=all_indices,
    )

    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in sorted_results[:top_k]:
        results.append((material_ids[idx], score))

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test AmbientCG material retrieval with CLIP embeddings."
    )
    parser.add_argument(
        "--materials-dir",
        type=Path,
        required=True,
        help="Path to materials directory (with embeddings/ subdirectory).",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query for retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to save preview images.",
    )

    args = parser.parse_args()

    # Set up logging.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Validate paths.
    embeddings_path = args.materials_dir / "embeddings"
    if not embeddings_path.exists():
        console_logger.error(f"Embeddings not found: {embeddings_path}")
        console_logger.error("Run compute_ambientcg_embeddings.py first.")
        return

    # Load data.
    console_logger.info("Loading material embeddings...")
    embeddings, material_ids, metadata = load_material_data(embeddings_path)
    console_logger.info(
        f"Loaded {len(material_ids)} materials with {embeddings.shape[1]}D embeddings"
    )

    # Retrieve top-k.
    console_logger.info(f"\nQuerying: '{args.query}'")
    results = retrieve_top_k(
        query=args.query,
        embeddings=embeddings,
        material_ids=material_ids,
        top_k=args.top_k,
    )

    # Display results.
    console_logger.info(f"\nTop {len(results)} results:")
    console_logger.info("=" * 60)

    for i, (material_id, score) in enumerate(results, 1):
        meta = metadata.get(material_id, {})
        category = meta.get("category", "Unknown")
        tags = meta.get("tags", [])
        tags_str = ", ".join(tags[:8])  # Show first 8 tags.
        if len(tags) > 8:
            tags_str += f", ... (+{len(tags) - 8} more)"

        console_logger.info(f"\n{i}. {material_id} (score: {score:.4f})")
        console_logger.info(f"   Category: {category}")
        console_logger.info(f"   Tags: {tags_str}")

    # Copy preview images if output path specified.
    if args.output_path:
        args.output_path.mkdir(parents=True, exist_ok=True)
        console_logger.info(f"\nCopying preview images to: {args.output_path}")

        for i, (material_id, score) in enumerate(results, 1):
            preview_src = args.materials_dir / material_id / f"{material_id}.png"
            if preview_src.exists():
                preview_dst = args.output_path / f"rank{i:02d}_{material_id}.png"
                shutil.copy(preview_src, preview_dst)
                console_logger.info(f"  Copied: {preview_dst.name}")
            else:
                console_logger.warning(f"  Preview not found: {preview_src}")

        console_logger.info(f"\nDone! Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
