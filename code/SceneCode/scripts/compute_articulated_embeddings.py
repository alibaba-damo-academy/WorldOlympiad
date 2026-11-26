#!/usr/bin/env python3
"""Compute CLIP image embeddings for articulated objects.

This script computes CLIP image embeddings for articulated SDF assets by
rendering multi-view images and averaging their embeddings. This enables
text-to-image semantic retrieval.

Must be run separately for each data source.

Usage:
    # For PartNet-Mobility processed assets
    python scripts/compute_articulated_embeddings.py \
        --source partnet_mobility \
        --data-path data/partnet_mobility_sdf \
        --output-path data/partnet_mobility_sdf/embeddings

    # For ArtVIP assets
    python scripts/compute_articulated_embeddings.py \
        --source artvip \
        --data-path data/artvip_sdf \
        --output-path data/artvip_sdf/embeddings

    # Keep rendered images for inspection
    python scripts/compute_articulated_embeddings.py \
        --source partnet_mobility \
        --data-path data/partnet_mobility_sdf \
        --output-path data/partnet_mobility_sdf/embeddings \
        --keep-renders
"""

import argparse
import json
import logging
import shutil
import tempfile

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from tqdm import tqdm

from scenecode.agent_utils.articulated_retrieval_server.data_loader import (
    get_placement_type_from_options,
)
from scenecode.agent_utils.blender.renderer import (
    ARTICULATED_LIGHT_ENERGY,
    BlenderRenderer,
)
from scenecode.agent_utils.clip_embeddings import get_multiview_image_embedding
from scenecode.agent_utils.sdf_mesh_utils import combine_sdf_meshes_at_joint_angles

console_logger = logging.getLogger(__name__)


def find_partnet_assets(data_path: Path) -> list[tuple[str, Path]]:
    """Find PartNet-Mobility processed assets.

    Args:
        data_path: Path to processed PartNet-Mobility directory.

    Returns:
        List of (object_id, asset_dir) tuples.
    """
    assets = []
    for child in sorted(data_path.iterdir()):
        if child.is_dir():
            analysis_path = child / "analysis.json"
            if analysis_path.exists():
                assets.append((child.name, child))
    return assets


def find_artvip_assets(data_path: Path) -> list[tuple[str, Path]]:
    """Find ArtVIP assets.

    ArtVIP has category subdirectories (large_furniture, small_furniture, etc.)
    with model directories inside.

    Args:
        data_path: Path to ArtVIP SDF directory.

    Returns:
        List of (object_id, asset_dir) tuples.
    """
    assets = []
    for category_dir in sorted(data_path.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        # Skip the embeddings directory.
        if category_dir.name == "embeddings":
            continue

        for model_dir in sorted(category_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            # Look for properties file.
            properties_files = list(model_dir.glob("*_properties.json"))
            if properties_files:
                # Object ID includes category for uniqueness.
                object_id = f"{category_dir.name}/{model_dir.name}"
                assets.append((object_id, model_dir))
    return assets


def load_partnet_metadata(asset_dir: Path) -> dict | None:
    """Load metadata from PartNet-Mobility analysis.json.

    Args:
        asset_dir: Path to asset directory.

    Returns:
        Metadata dict or None if not found or invalid.
    """
    analysis_path = asset_dir / "analysis.json"
    if not analysis_path.exists():
        return None

    # Load JSON with error handling.
    try:
        with open(analysis_path) as f:
            analysis = json.load(f)
    except json.JSONDecodeError as e:
        console_logger.warning(f"Malformed JSON in {analysis_path}: {e}")
        return None
    except Exception as e:
        console_logger.warning(f"Failed to read {analysis_path}: {e}")
        return None

    # Find SDF path - return None if not found.
    sdf_path = asset_dir / "mobility.sdf"
    if not sdf_path.exists():
        console_logger.warning(f"No mobility.sdf found in {asset_dir}")
        return None

    # Get placement type from placement_options.
    placement_options = analysis.get("placement_options", {})
    placement_type = get_placement_type_from_options(
        placement_options=placement_options
    )

    return {
        "category": analysis.get("category", "Unknown"),
        "description": analysis.get("object_description", ""),
        "is_manipuland": analysis.get("is_manipuland", False),
        "placement_type": placement_type,
        "placement_options": placement_options,
        "sdf_path": str(sdf_path.relative_to(asset_dir.parent)),
    }


def load_artvip_metadata(asset_dir: Path) -> dict | None:
    """Load metadata from ArtVIP *_properties.json.

    Args:
        asset_dir: Path to asset directory.

    Returns:
        Metadata dict or None if not found or invalid.
    """
    # Find properties file.
    properties_files = list(asset_dir.glob("*_properties.json"))
    if not properties_files:
        return None

    # Load JSON with error handling.
    try:
        with open(properties_files[0]) as f:
            props = json.load(f)
    except json.JSONDecodeError as e:
        console_logger.warning(f"Malformed JSON in {properties_files[0]}: {e}")
        return None
    except Exception as e:
        console_logger.warning(f"Failed to read {properties_files[0]}: {e}")
        return None

    # Find SDF path.
    sdf_files = list(asset_dir.glob("*.sdf"))
    if not sdf_files:
        console_logger.warning(f"No SDF file found in {asset_dir}")
        return None

    # Get placement type from placement_options.
    placement_options = props.get("placement_options", {})
    placement_type = get_placement_type_from_options(
        placement_options=placement_options
    )

    # Get category from parent directory name.
    category = asset_dir.parent.name

    # Relative SDF path from data root.
    sdf_rel = f"{category}/{asset_dir.name}/{sdf_files[0].name}"

    return {
        "category": category,
        "description": props.get("description", ""),
        "is_manipuland": props.get("is_manipuland", False),
        "placement_type": placement_type,
        "placement_options": placement_options,
        "sdf_path": sdf_rel,
    }


@dataclass
class EmbeddingResult:
    """Result of embedding computation for a single asset."""

    embedding: np.ndarray
    """CLIP image embedding (1024,)."""

    bounding_box_min: list[float]
    """Bounding box minimum [x, y, z] computed from mesh."""

    bounding_box_max: list[float]
    """Bounding box maximum [x, y, z] computed from mesh."""


def render_and_embed_asset(
    sdf_path: Path,
    object_id: str,
    renderer: BlenderRenderer,
    render_output_dir: Path | None,
) -> EmbeddingResult | None:
    """Render multi-view images and compute averaged CLIP embedding.

    Also computes the bounding box from the combined mesh at joint positions=0,
    in the model's local frame. More reliable than trusting metadata.

    Args:
        sdf_path: Path to the SDF file.
        object_id: Object ID for naming.
        renderer: BlenderRenderer instance.
        render_output_dir: Directory to save renders (None for temp dir).

    Returns:
        EmbeddingResult with embedding and bounding box, or None on failure.
    """
    # Create temporary or persistent render directory.
    if render_output_dir is not None:
        # Create subdirectory for this asset.
        # Replace / with _ for nested object IDs (e.g., artvip category/model).
        safe_id = object_id.replace("/", "_")
        asset_render_dir = render_output_dir / safe_id
        asset_render_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = None
    else:
        temp_dir = tempfile.mkdtemp(prefix="clip_render_")
        asset_render_dir = Path(temp_dir)

    try:
        # Combine SDF meshes at default pose (joints=0).
        combined_mesh = combine_sdf_meshes_at_joint_angles(
            sdf_path, use_max_angles=False
        )

        # Export combined mesh as GLB for rendering.
        glb_path = asset_render_dir / "combined.glb"
        combined_mesh.export(glb_path)

        # Render 8 views (4 upper + 4 lower at 30° elevation).
        # Use lower light energy for articulated objects (more reflective materials).
        image_paths = renderer.render_multiview_for_clip_embedding(
            mesh_path=glb_path,
            output_dir=asset_render_dir,
            width=224,
            height=224,
            elevation_degrees=30.0,
            light_energy=ARTICULATED_LIGHT_ENERGY,
        )

        if not image_paths:
            console_logger.warning(f"No images rendered for {object_id}")
            return None

        # Compute averaged CLIP embedding from all views.
        embedding = get_multiview_image_embedding(image_paths)

        # Compute bounding box from mesh and convert Y-up (GLTF) to Z-up (Drake).
        # GLTF uses Y-up, Drake/Blender use Z-up.
        # Conversion: X_zup = X_yup, Y_zup = -Z_yup, Z_zup = Y_yup
        bbox_min_yup = combined_mesh.bounds[0]
        bbox_max_yup = combined_mesh.bounds[1]
        bbox_min = [bbox_min_yup[0], -bbox_max_yup[2], bbox_min_yup[1]]
        bbox_max = [bbox_max_yup[0], -bbox_min_yup[2], bbox_max_yup[1]]

        return EmbeddingResult(
            embedding=embedding, bounding_box_min=bbox_min, bounding_box_max=bbox_max
        )

    except Exception as e:
        console_logger.warning(f"Failed to render/embed {object_id}: {e}")
        return None

    finally:
        # Clean up temporary directory if used.
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def compute_embeddings(
    source: str, data_path: Path, output_path: Path, keep_renders: bool = False
) -> None:
    """Compute and save CLIP image embeddings for articulated assets.

    Args:
        source: Source name ('partnet_mobility' or 'artvip').
        data_path: Path to asset data directory.
        output_path: Path to save embeddings.
        keep_renders: If True, save rendered images to output_path/renders.
    """
    console_logger.info(
        f"Computing image embeddings for source '{source}'. "
        f"Data path: {data_path}. Output path: {output_path}."
    )

    # Find assets based on source type.
    if source == "partnet_mobility":
        assets = find_partnet_assets(data_path)
        load_metadata = load_partnet_metadata
    elif source == "artvip":
        assets = find_artvip_assets(data_path)
        load_metadata = load_artvip_metadata
    else:
        raise ValueError(f"Unknown source: {source}")

    console_logger.info(f"Found {len(assets)} assets")

    if not assets:
        console_logger.error("No assets found, exiting")
        return

    # Initialize renderer.
    renderer = BlenderRenderer()

    # Setup render output directory if keeping renders.
    render_output_dir = output_path / "renders" if keep_renders else None
    if render_output_dir:
        render_output_dir.mkdir(parents=True, exist_ok=True)
        console_logger.info(f"Saving renders to: {render_output_dir}")

    # Compute embeddings.
    embeddings_list: list[np.ndarray] = []
    embedding_index: list[str] = []
    metadata_index: dict[str, dict] = {}
    skipped_metadata = 0
    skipped_render = 0
    for object_id, asset_dir in tqdm(assets, desc="Computing embeddings"):
        # Load metadata.
        metadata = load_metadata(asset_dir)
        if metadata is None:
            skipped_metadata += 1
            continue

        # Resolve SDF path from relative path.
        sdf_path = data_path / metadata["sdf_path"]

        # Render and compute embedding and bounding box.
        result = render_and_embed_asset(
            sdf_path=sdf_path,
            object_id=object_id,
            renderer=renderer,
            render_output_dir=render_output_dir,
        )

        if result is None:
            skipped_render += 1
            continue

        embeddings_list.append(result.embedding)
        embedding_index.append(object_id)

        # Store metadata with bounding box computed from mesh.
        metadata["bounding_box_min"] = result.bounding_box_min
        metadata["bounding_box_max"] = result.bounding_box_max
        metadata_index[object_id] = metadata

    console_logger.info(
        f"Computed {len(embeddings_list)} embeddings. "
        f"Skipped: {skipped_metadata} (metadata errors), "
        f"{skipped_render} (render/embedding errors)"
    )
    if not embeddings_list:
        console_logger.error("No embeddings computed, exiting")
        return

    # Save outputs.
    output_path.mkdir(parents=True, exist_ok=True)

    # Save embeddings as numpy array.
    embeddings_array = np.stack(embeddings_list, axis=0).astype(np.float32)
    embeddings_file = output_path / "clip_embeddings.npy"
    np.save(embeddings_file, embeddings_array)
    console_logger.info(f"Saved embeddings: shape={embeddings_array.shape}")

    # Save embedding index.
    embedding_index_file = output_path / "embedding_index.yaml"
    with open(embedding_index_file, "w") as f:
        yaml.dump(embedding_index, f, default_flow_style=False)
    console_logger.info(f"Saved embedding index: {len(embedding_index)} entries")

    # Save metadata index.
    metadata_index_file = output_path / "metadata_index.yaml"
    with open(metadata_index_file, "w") as f:
        yaml.dump(metadata_index, f, default_flow_style=False, allow_unicode=True)
    console_logger.info(f"Saved metadata index: {len(metadata_index)} entries")

    console_logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Compute CLIP image embeddings for articulated objects."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["partnet_mobility", "artvip"],
        help="Data source type.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to asset data directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save embeddings.",
    )
    parser.add_argument(
        "--keep-renders",
        action="store_true",
        help="Keep rendered images in output_path/renders.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    # Configure logging.
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Validate data path exists.
    if not args.data_path.exists():
        console_logger.error(f"Data path does not exist: {args.data_path}")
        return

    compute_embeddings(
        source=args.source,
        data_path=args.data_path,
        output_path=args.output_path,
        keep_renders=args.keep_renders,
    )


if __name__ == "__main__":
    main()
