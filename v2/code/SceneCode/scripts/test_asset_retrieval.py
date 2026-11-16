#!/usr/bin/env python3
"""Test asset retrieval (HSSD, Objaverse, articulated) with rendered visualization.

This script tests CLIP-based retrieval by querying with text and rendering
multi-view images of retrieved meshes.

Usage:
    # Test Objaverse retrieval (uses default paths)
    python scripts/test_asset_retrieval.py \
        --source objaverse \
        --object-type FURNITURE \
        --query "wooden bookshelf" \
        --top-k 5 \
        --output-path output/retrieval_test

    # Test HSSD retrieval
    python scripts/test_asset_retrieval.py \
        --source hssd \
        --query "wooden dining chair" \
        --top-k 3 \
        --output-path output/retrieval_test

    # Test PartNet-Mobility retrieval (uses default paths)
    python scripts/test_asset_retrieval.py \
        --source partnet_mobility \
        --query "wooden cabinet with drawers" \
        --top-k 3 \
        --output-path output/retrieval_test

    # Test ArtVIP retrieval (uses default paths)
    python scripts/test_asset_retrieval.py \
        --source artvip \
        --query "office stapler" \
        --top-k 5 \
        --output-path output/retrieval_test

    # Test combined PartNet-Mobility + ArtVIP retrieval
    python scripts/test_asset_retrieval.py \
        --source combined \
        --query "ceiling fan" \
        --top-k 5 \
        --output-path output/retrieval_test

    # Filter articulated by object type (like main pipeline)
    python scripts/test_asset_retrieval.py \
        --source combined \
        --object-type CEILING_MOUNTED \
        --query "ceiling fan" \
        --top-k 5 \
        --output-path output/retrieval_test
"""

import argparse
import logging
import tempfile

from pathlib import Path

import numpy as np
import trimesh

from scenecode.agent_utils.articulated_retrieval_server.config import (
    ArticulatedConfig,
    ArticulatedSourceConfig,
)
from scenecode.agent_utils.articulated_retrieval_server.data_loader import (
    load_preprocessed_data_multi_source,
)
from scenecode.agent_utils.blender.renderer import (
    ARTICULATED_LIGHT_ENERGY,
    VLM_ANALYSIS_LIGHT_ENERGY,
    BlenderRenderer,
)
from scenecode.agent_utils.clip_embeddings import (
    compute_clip_similarities,
    get_text_embedding,
)
from scenecode.agent_utils.hssd_retrieval.data_loader import load_preprocessed_data
from scenecode.agent_utils.objaverse_retrieval.clip_similarity import (
    compute_clip_similarities as compute_objaverse_clip_similarities,
    get_objaverse_text_embedding,
)
from scenecode.agent_utils.objaverse_retrieval.config import ObjaverseConfig
from scenecode.agent_utils.objaverse_retrieval.data_loader import (
    construct_objaverse_mesh_path,
    load_preprocessed_data as load_objaverse_preprocessed_data,
)
from scenecode.agent_utils.sdf_mesh_utils import combine_sdf_meshes_at_joint_angles

console_logger = logging.getLogger(__name__)


def load_articulated_data(
    source: str,
    data_path: Path | None = None,
    embeddings_path: Path | None = None,
    object_type: str | None = None,
) -> tuple[np.ndarray, list[str], dict, dict]:
    """Load articulated embeddings using main pipeline's loader.

    Args:
        source: Source name - "partnet_mobility", "artvip", or "combined".
        data_path: Path to data directory (optional for combined).
        embeddings_path: Path to embeddings directory (optional for combined).
        object_type: Filter by object type (FURNITURE, MANIPULAND, etc.).

    Returns:
        Tuple of (embeddings, object_ids, metadata_dict, source_map).
        source_map maps object_id to (source_name, data_path).
    """
    partnet_data_path = Path("data/partnet_mobility_sdf")
    artvip_data_path = Path("data/artvip_sdf")

    # Build config based on source.
    if source == "combined":
        sources = {
            "partnet_mobility": ArticulatedSourceConfig(
                name="partnet_mobility",
                enabled=(partnet_data_path / "embeddings").exists(),
                data_path=partnet_data_path,
                embeddings_path=partnet_data_path / "embeddings",
            ),
            "artvip": ArticulatedSourceConfig(
                name="artvip",
                enabled=(artvip_data_path / "embeddings").exists(),
                data_path=artvip_data_path,
                embeddings_path=artvip_data_path / "embeddings",
            ),
        }
    elif source == "partnet_mobility":
        sources = {
            "partnet_mobility": ArticulatedSourceConfig(
                name="partnet_mobility",
                enabled=True,
                data_path=data_path or partnet_data_path,
                embeddings_path=embeddings_path or (partnet_data_path / "embeddings"),
            ),
        }
    else:  # artvip
        sources = {
            "artvip": ArticulatedSourceConfig(
                name="artvip",
                enabled=True,
                data_path=data_path or artvip_data_path,
                embeddings_path=embeddings_path or (artvip_data_path / "embeddings"),
            ),
        }

    config = ArticulatedConfig(sources=sources, use_top_k=5)

    # Load using main pipeline function.
    data = load_preprocessed_data_multi_source(config)
    if data is None:
        raise ValueError(f"No articulated data found for source '{source}'")

    # Apply object type filter if specified.
    if object_type:
        matching_indices = data.filter_by_object_type(object_type)
        if not matching_indices:
            raise ValueError(
                f"No objects found for object_type '{object_type}' in source '{source}'"
            )
        console_logger.info(
            f"Filtered to {len(matching_indices)} objects for type '{object_type}'"
        )
        # Get filtered object IDs.
        filtered_ids = [data.embedding_index[idx] for idx in matching_indices]
    else:
        filtered_ids = data.embedding_index
        matching_indices = list(range(len(data.embedding_index)))

    # Build source_map from metadata (only for filtered objects).
    source_map = {}
    for obj_id in filtered_ids:
        meta = data.metadata_by_id[obj_id]
        if meta.source == "partnet_mobility":
            source_map[obj_id] = ("partnet_mobility", partnet_data_path)
        else:
            source_map[obj_id] = ("artvip", artvip_data_path)

    # Convert metadata to dict format expected by rest of script.
    # Note: meta.sdf_path is already resolved (full path), so we store it directly.
    metadata_dict = {
        obj_id: {
            "description": data.metadata_by_id[obj_id].description,
            "sdf_path": str(data.metadata_by_id[obj_id].sdf_path),
            "category": data.metadata_by_id[obj_id].category,
        }
        for obj_id in filtered_ids
    }

    # Filter embeddings to matching indices.
    filtered_embeddings = data.clip_embeddings[matching_indices]

    return filtered_embeddings, filtered_ids, metadata_dict, source_map


def load_hssd_data(embeddings_path: Path) -> tuple[np.ndarray, list[str], dict]:
    """Load HSSD embeddings and metadata.

    Args:
        embeddings_path: Path to preprocessed data directory.

    Returns:
        Tuple of (embeddings, mesh_ids, metadata_dict).
    """
    preprocessed = load_preprocessed_data(preprocessed_path=embeddings_path)

    # Build metadata dict from preprocessed data.
    metadata = {}
    for mesh_id in preprocessed.embedding_index:
        meta = preprocessed.get_metadata(mesh_id)
        if meta:
            metadata[mesh_id] = {
                "description": f"{meta.name} ({meta.wordnet_key})",
                "wordnet_key": meta.wordnet_key,
                "name": meta.name,
            }

    return preprocessed.clip_embeddings, preprocessed.embedding_index, metadata


def load_objaverse_data(
    data_path: Path | None = None,
    embeddings_path: Path | None = None,
    object_type: str | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    """Load Objaverse embeddings and metadata.

    Args:
        data_path: Path to Objaverse data directory.
        embeddings_path: Path to preprocessed data directory.
        object_type: Filter by object type (FURNITURE, MANIPULAND, etc.).

    Returns:
        Tuple of (embeddings, mesh_ids, metadata_dict).
    """
    default_data_path = Path("data/objathor-assets")
    default_preprocessed_path = default_data_path / "preprocessed"

    actual_data_path = data_path or default_data_path
    actual_preprocessed_path = embeddings_path or default_preprocessed_path

    # Validate config (raises if paths don't exist).
    config = ObjaverseConfig(
        data_path=actual_data_path, preprocessed_path=actual_preprocessed_path
    )

    preprocessed = load_objaverse_preprocessed_data(
        preprocessed_path=config.preprocessed_path
    )

    # Apply object type filter if specified.
    if object_type:
        # Map object type to category.
        category = config.object_type_mapping.get(object_type.upper())
        if category is None:
            raise ValueError(
                f"Unknown object_type '{object_type}'. "
                f"Available: {list(config.object_type_mapping.keys())}"
            )

        # Get UIDs for this category.
        category_uids = set(preprocessed.object_categories.get(category, []))
        if not category_uids:
            raise ValueError(f"No objects found for category '{category}'")

        # Filter embedding index to matching UIDs.
        filtered_ids = [
            uid for uid in preprocessed.embedding_index if uid in category_uids
        ]
        matching_indices = [
            preprocessed.embedding_index.index(uid) for uid in filtered_ids
        ]

        console_logger.info(
            f"Filtered to {len(filtered_ids)} objects for type '{object_type}' "
            f"(category='{category}')"
        )
    else:
        filtered_ids = preprocessed.embedding_index
        matching_indices = list(range(len(preprocessed.embedding_index)))

    # Build metadata dict from preprocessed data.
    metadata = {}
    for mesh_id in filtered_ids:
        meta = preprocessed.get_metadata(mesh_id)
        if meta:
            metadata[mesh_id] = {
                "description": meta.description or meta.name,
                "name": meta.name,
                "category": meta.category,
                "bounding_box": meta.bounding_box,
            }

    # Filter embeddings to matching indices.
    filtered_embeddings = preprocessed.clip_embeddings[matching_indices]

    return filtered_embeddings, filtered_ids, metadata


def retrieve_top_k(
    query: str,
    embeddings: np.ndarray,
    object_ids: list[str],
    top_k: int,
    source: str = "articulated",
) -> list[tuple[str, float]]:
    """Retrieve top-k objects by CLIP similarity.

    Args:
        query: Text query.
        embeddings: Embedding matrix (N x D).
        object_ids: List of object IDs corresponding to embeddings.
        top_k: Number of results to return.
        source: Data source - affects which CLIP model to use.
            "objaverse" uses ViT-L/14 (768-dim), others use ViT-H-14 (1024-dim).

    Returns:
        List of (object_id, similarity_score) tuples.
    """
    # Use source-specific CLIP model to match pre-computed embeddings.
    if source == "objaverse":
        # Objaverse uses ViT-L/14 (768-dim).
        text_embedding = get_objaverse_text_embedding(query)
        compute_fn = compute_objaverse_clip_similarities
    else:
        # HSSD and articulated use ViT-H-14 (1024-dim).
        text_embedding = get_text_embedding(query)
        compute_fn = compute_clip_similarities

    # Compute similarities for all indices.
    all_indices = list(range(len(object_ids)))
    similarities = compute_fn(
        query_embedding=text_embedding,
        embeddings=embeddings,
        indices=all_indices,
    )

    # Sort by similarity (descending).
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Map indices to object IDs.
    results = []
    for idx, score in sorted_results[:top_k]:
        results.append((object_ids[idx], score))

    return results


def load_mesh(
    source: str, object_id: str, data_path: Path | None, metadata: dict
) -> trimesh.Trimesh | None:
    """Load mesh for an object.

    Args:
        source: Source name.
        object_id: Object ID.
        data_path: Path to data directory.
        metadata: Metadata dictionary.

    Returns:
        Trimesh object, or None on failure.
    """
    try:
        if source == "hssd":
            from scenecode.agent_utils.hssd_retrieval.data_loader import (
                construct_hssd_mesh_path,
            )

            if data_path is None:
                console_logger.warning("HSSD requires data_path")
                return None
            mesh_path = construct_hssd_mesh_path(data_path, object_id)
            if not mesh_path.exists():
                console_logger.warning(f"HSSD mesh not found: {mesh_path}")
                return None
            return trimesh.load(mesh_path, force="mesh")

        if source == "objaverse":
            objaverse_data_path = data_path or Path("data/objathor-assets")
            mesh_path = construct_objaverse_mesh_path(objaverse_data_path, object_id)
            return trimesh.load(mesh_path, force="mesh")

        # For articulated sources, combine SDF meshes.
        meta = metadata.get(object_id, {})
        sdf_path_str = meta.get("sdf_path")
        if not sdf_path_str:
            console_logger.warning(f"No SDF path for {object_id}")
            return None

        # sdf_path from main pipeline is already resolved (relative to cwd).
        sdf_path = Path(sdf_path_str)
        if not sdf_path.exists():
            console_logger.warning(f"SDF not found: {sdf_path}")
            return None

        return combine_sdf_meshes_at_joint_angles(
            sdf_path=sdf_path, use_max_angles=False
        )

    except Exception as e:
        console_logger.error(f"Failed to load mesh for {object_id}: {e}")
        return None


def render_mesh_multiview(
    mesh: trimesh.Trimesh,
    output_dir: Path,
    object_id: str,
    rank: int,
    score: float,
    source: str,
) -> list[Path]:
    """Render multi-view images of a mesh.

    Args:
        mesh: Trimesh object to render.
        output_dir: Directory to save renders.
        object_id: Object ID for filename.
        rank: Result rank (1-indexed).
        score: Similarity score.
        source: Data source name.

    Returns:
        List of paths to rendered images.
    """
    # Create temporary GLB file for rendering.
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
        glb_path = Path(f.name)

    try:
        mesh.export(glb_path)

        renderer = BlenderRenderer()

        # Create output subdirectory for this result.
        # Sanitize object_id for filesystem (replace / with _).
        safe_id = object_id.replace("/", "_")
        result_dir = output_dir / f"rank{rank:02d}_{safe_id}"
        result_dir.mkdir(parents=True, exist_ok=True)

        # Choose light energy based on source.
        if source in ("partnet_mobility", "artvip"):
            light_energy = ARTICULATED_LIGHT_ENERGY
        else:
            light_energy = VLM_ANALYSIS_LIGHT_ENERGY

        # Render multiview for analysis (top, bottom, 4 sides).
        image_paths = renderer.render_multiview_for_analysis(
            mesh_path=glb_path,
            output_dir=result_dir,
            elevation_degrees=20.0,
            num_side_views=4,
            include_vertical_views=True,
            width=512,
            height=512,
            light_energy=light_energy,
        )

        # Write metadata file.
        metadata_path = result_dir / "info.txt"
        with open(metadata_path, "w") as f:
            f.write(f"Rank: {rank}\n")
            f.write(f"Object ID: {object_id}\n")
            f.write(f"Score: {score:.4f}\n")
            f.write(f"Source: {source}\n")

        console_logger.info(f"Rendered {len(image_paths)} views to {result_dir}")
        return image_paths

    finally:
        # Clean up temporary file.
        if glb_path.exists():
            glb_path.unlink()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test asset retrieval (HSSD, Objaverse, articulated) with "
        "rendered visualization."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["objaverse", "hssd", "partnet_mobility", "artvip", "combined"],
        help="Data source to query. Use 'combined' for merged PartNet + ArtVIP.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to asset data directory (not needed for 'combined').",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=None,
        help="Path to embeddings directory (not needed for 'combined').",
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
        required=True,
        help="Path to save rendered images.",
    )
    parser.add_argument(
        "--object-type",
        type=str,
        default=None,
        choices=["FURNITURE", "MANIPULAND", "WALL_MOUNTED", "CEILING_MOUNTED"],
        help="Filter by object type (like main pipeline agent filtering).",
    )

    args = parser.parse_args()

    # Set up logging.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Validate paths if explicitly provided.
    if args.data_path and not args.data_path.exists():
        console_logger.error(f"Data path not found: {args.data_path}")
        return
    if args.embeddings_path and not args.embeddings_path.exists():
        console_logger.error(f"Embeddings path not found: {args.embeddings_path}")
        return

    # Validate object-type is only used with sources that support it.
    sources_with_object_type = {"objaverse", "partnet_mobility", "artvip", "combined"}
    if args.object_type and args.source not in sources_with_object_type:
        console_logger.error(
            f"--object-type is only supported for: {sources_with_object_type}"
        )
        return

    # Create output directory.
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Load data.
    source_map = None  # Only used for articulated sources.
    console_logger.info(f"\nLoading {args.source} data...")
    if args.source == "hssd":
        # HSSD requires embeddings path (use default if not provided).
        embeddings_path = args.embeddings_path or Path("data/hssd-preprocessed")
        if not embeddings_path.exists():
            console_logger.error(
                f"HSSD preprocessed path not found: {embeddings_path}. "
                "Provide --embeddings-path or ensure data/hssd-preprocessed exists."
            )
            return
        embeddings, object_ids, metadata = load_hssd_data(embeddings_path)
    elif args.source == "objaverse":
        embeddings, object_ids, metadata = load_objaverse_data(
            data_path=args.data_path,
            embeddings_path=args.embeddings_path,
            object_type=args.object_type,
        )
    else:
        # All articulated sources (partnet_mobility, artvip, combined).
        embeddings, object_ids, metadata, source_map = load_articulated_data(
            source=args.source,
            data_path=args.data_path,
            embeddings_path=args.embeddings_path,
            object_type=args.object_type,
        )

    console_logger.info(
        f"  Loaded {len(object_ids)} objects with {embeddings.shape[1]}D embeddings"
    )

    # Retrieve top-k.
    console_logger.info(f"\nQuerying: '{args.query}'")
    results = retrieve_top_k(
        query=args.query,
        embeddings=embeddings,
        object_ids=object_ids,
        top_k=args.top_k,
        source=args.source,
    )

    console_logger.info(f"\nTop {len(results)} results:")
    for i, (object_id, score) in enumerate(results, 1):
        meta = metadata.get(object_id, {})
        description = meta.get("description", "N/A")
        console_logger.info(f"  {i}. {object_id} (score: {score:.4f})")
        console_logger.info(f"     Description: {description}")

    # Render each result.
    console_logger.info("\n" + "=" * 60)
    console_logger.info(f"Rendering to: {args.output_path}")
    console_logger.info("=" * 60)

    for i, (object_id, score) in enumerate(results, 1):
        console_logger.info(f"\n[{i}/{len(results)}] Rendering {object_id}...")

        # For articulated sources, look up actual source and data path from source_map.
        if source_map is not None and object_id in source_map:
            actual_source, data_path = source_map[object_id]
        else:
            actual_source = args.source
            data_path = args.data_path

        mesh = load_mesh(
            source=actual_source,
            object_id=object_id,
            data_path=data_path,
            metadata=metadata,
        )
        if mesh is None:
            console_logger.warning(f"  Skipping: failed to load mesh")
            continue

        render_mesh_multiview(
            mesh=mesh,
            output_dir=args.output_path,
            object_id=object_id,
            rank=i,
            score=score,
            source=actual_source,
        )

    console_logger.info(f"\nDone! Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
