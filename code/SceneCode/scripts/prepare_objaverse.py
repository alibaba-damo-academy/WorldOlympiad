#!/usr/bin/env python3
"""
Prepare ObjectThor (Objaverse) data for scenecode retrieval.

This script processes ObjectThor data downloaded via download_objaverse_data.sh:
1. Loads pre-computed CLIP features (3 views, 768-dim, float16)
2. Averages multi-view embeddings to single embedding per object
3. Creates metadata index with categories and bounding boxes
4. Creates object category mapping for scenecode ObjectTypes
5. Outputs preprocessed/ directory ready for ObjaverseRetriever

Usage:
    python scripts/prepare_objaverse.py [--data-path DATA_PATH]
"""

import argparse
import json
import logging
import pickle

from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def map_placement_to_category(metadata: dict) -> str:
    """Map ObjectThor placement flags to scenecode categories.

    ObjectThor has explicit placement constraints that map well to scenecode types:
    - FURNITURE (large_objects): Floor-only items (on_floor=True, on_object=False)
    - MANIPULAND (small_objects): Can go on tables (on_object=True)
    - WALL_MOUNTED (wall_objects): Wall-only (on_wall=True, not on floor)
    - CEILING_MOUNTED (ceiling_objects): Ceiling-only (on_ceiling=True exclusively)

    Args:
        metadata: Object metadata with placement flags.

    Returns:
        Category string for scenecode.
    """
    on_ceiling = metadata.get("onCeiling", False)
    on_wall = metadata.get("onWall", False)
    on_floor = metadata.get("onFloor", False)
    on_object = metadata.get("onObject", False)

    # Priority: ceiling > wall-only > floor-only > can-go-on-objects
    if on_ceiling and not on_floor and not on_wall:
        return "ceiling_objects"
    if on_wall and not on_floor:
        return "wall_objects"
    if on_floor and not on_object:
        return "large_objects"  # Floor-only = furniture
    return "small_objects"  # Can go on tables = manipulands


def load_objathor_features(data_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load pre-computed CLIP features from ObjectThor.

    ObjectThor ships with pre-computed CLIP embeddings from 3 canonical views.
    Format: (N, 3, 768) float16 array + list of UIDs.

    Args:
        data_path: Path to objathor-assets directory.

    Returns:
        Tuple of (embeddings array (N, 3, 768), list of UIDs).
    """
    # Features are in features/ subdirectory.
    features_path = data_path / "features" / "clip_features.pkl"
    if not features_path.exists():
        raise FileNotFoundError(
            f"CLIP features not found at {features_path}. "
            "Run download_objaverse_data.sh first."
        )

    logger.info(f"Loading CLIP features from {features_path}")
    with open(features_path, "rb") as f:
        data = pickle.load(f)

    # ObjectThor clip_features.pkl contains dict with "uids" and "img_features".
    if isinstance(data, dict):
        features = data.get("img_features")
        if features is None:
            features = data.get("features")
        uids = data.get("uids")
        if features is None:
            raise ValueError(
                f"No features found in {features_path}. Keys: {data.keys()}"
            )
    else:
        # Fallback for alternative formats.
        features = data
        uids = None

    logger.info(f"Loaded features shape: {features.shape}")
    return features, uids


def load_objathor_annotations(data_path: Path) -> dict:
    """Load ObjectThor annotations with metadata.

    Args:
        data_path: Path to objathor-assets directory.

    Returns:
        Dict mapping UID -> metadata dict.
    """
    # Try gzipped first, then uncompressed.
    annotations_path = data_path / "annotations.json.gz"
    if not annotations_path.exists():
        annotations_path = data_path / "annotations.json"
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Annotations not found at {data_path}. "
            "Run download_objaverse_data.sh first."
        )

    logger.info(f"Loading annotations from {annotations_path}")
    if annotations_path.suffix == ".gz":
        import gzip

        with gzip.open(annotations_path, "rt") as f:
            annotations = json.load(f)
    else:
        with open(annotations_path, "r") as f:
            annotations = json.load(f)

    logger.info(f"Loaded {len(annotations)} object annotations")
    return annotations


def average_multiview_embeddings(features: np.ndarray) -> np.ndarray:
    """Average multi-view CLIP embeddings to single embedding per object.

    Args:
        features: Array of shape (N, 3, 768) with 3-view embeddings.

    Returns:
        Array of shape (N, 768) with averaged embeddings.
    """
    # Average across views (axis 1).
    averaged = features.mean(axis=1)
    # Normalize to unit vectors.
    norms = np.linalg.norm(averaged, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero.
    normalized = averaged / norms
    return normalized.astype(np.float32)


def create_metadata_index(annotations: dict, uids: list[str]) -> dict:
    """Create metadata index for retrieval.

    Args:
        annotations: ObjectThor annotations dict.
        uids: List of UIDs in embedding order.

    Returns:
        Dict mapping UID -> simplified metadata.
    """
    metadata_index = {}

    for uid in uids:
        if uid not in annotations:
            logger.warning(f"UID {uid} not in annotations, skipping")
            continue

        ann = annotations[uid]

        # Extract relevant fields.
        # ObjectThor uses "thor_metadata.assetMetadata.boundingBox" for actual size.
        bounding_box = None
        if "thor_metadata" in ann and "assetMetadata" in ann["thor_metadata"]:
            asset_meta = ann["thor_metadata"]["assetMetadata"]
            if "boundingBox" in asset_meta:
                bb = asset_meta["boundingBox"]
                # Format: {"x": float, "y": float, "z": float} in meters.
                bounding_box = [bb.get("x", 0), bb.get("y", 0), bb.get("z", 0)]

        # Get category based on placement flags.
        category = map_placement_to_category(ann)

        # Get description (ObjectType is often descriptive).
        object_type = ann.get("objectType", "Unknown")
        description = ann.get("description", object_type)

        metadata_index[uid] = {
            "name": object_type,
            "category": category,
            "description": description,
            "bounding_box": bounding_box,
            "on_floor": ann.get("onFloor", False),
            "on_object": ann.get("onObject", False),
            "on_wall": ann.get("onWall", False),
            "on_ceiling": ann.get("onCeiling", False),
        }

    return metadata_index


def create_category_mapping(metadata_index: dict) -> dict[str, list[str]]:
    """Create category -> UIDs mapping for fast category filtering.

    Args:
        metadata_index: Metadata index from create_metadata_index.

    Returns:
        Dict mapping category -> list of UIDs.
    """
    category_mapping: dict[str, list[str]] = {
        "large_objects": [],
        "small_objects": [],
        "wall_objects": [],
        "ceiling_objects": [],
    }

    for uid, meta in metadata_index.items():
        category = meta.get("category", "small_objects")
        if category in category_mapping:
            category_mapping[category].append(uid)

    # Log statistics.
    for cat, uids in category_mapping.items():
        logger.info(f"Category {cat}: {len(uids)} objects")

    return category_mapping


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ObjectThor data for scenecode retrieval"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/objathor-assets",
        help="Path to ObjectThor data directory",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    preprocessed_path = data_path / "preprocessed"
    preprocessed_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing ObjectThor data from {data_path}")
    logger.info(f"Output directory: {preprocessed_path}")

    # Load features and annotations.
    features, uids = load_objathor_features(data_path)
    annotations = load_objathor_annotations(data_path)

    # If UIDs not in features file, derive from annotations.
    if uids is None:
        uids = list(annotations.keys())
        logger.warning("UIDs not in features file, using annotation keys")

    # Average multi-view embeddings.
    logger.info("Averaging multi-view embeddings...")
    averaged_embeddings = average_multiview_embeddings(features)
    logger.info(f"Averaged embeddings shape: {averaged_embeddings.shape}")

    # Create metadata index.
    logger.info("Creating metadata index...")
    metadata_index = create_metadata_index(annotations, uids)
    logger.info(f"Metadata index has {len(metadata_index)} entries")

    # Create category mapping.
    logger.info("Creating category mapping...")
    category_mapping = create_category_mapping(metadata_index)

    # Save outputs.
    embeddings_path = preprocessed_path / "clip_embeddings.npy"
    logger.info(f"Saving embeddings to {embeddings_path}")
    np.save(embeddings_path, averaged_embeddings)

    index_path = preprocessed_path / "embedding_index.yaml"
    logger.info(f"Saving embedding index to {index_path}")
    with open(index_path, "w") as f:
        yaml.dump({"uids": uids}, f, default_flow_style=False)

    metadata_path = preprocessed_path / "metadata_index.json"
    logger.info(f"Saving metadata index to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata_index, f, indent=2)

    categories_path = preprocessed_path / "object_categories.json"
    logger.info(f"Saving category mapping to {categories_path}")
    with open(categories_path, "w") as f:
        json.dump(category_mapping, f, indent=2)

    logger.info("=" * 50)
    logger.info("Preprocessing complete!")
    logger.info(f"Total objects: {len(uids)}")
    logger.info(f"Embeddings: {embeddings_path}")
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"Categories: {categories_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
