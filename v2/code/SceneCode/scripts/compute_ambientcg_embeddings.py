"""Compute CLIP embeddings for AmbientCG PBR materials.

This script computes CLIP image embeddings for downloaded AmbientCG materials
using their preview images. The embeddings enable text-based semantic retrieval.

Usage:
    # Compute embeddings for all downloaded materials
    python scripts/compute_ambientcg_embeddings.py --materials-dir data/materials

    # Specify output directory and preview size
    python scripts/compute_ambientcg_embeddings.py \
        --materials-dir data/materials \
        --output data/materials/embeddings \
        --preview-size 1024

    # Dry run to see what would be processed
    python scripts/compute_ambientcg_embeddings.py --materials-dir data/materials --dry-run
"""

import argparse
import logging
import shutil
import sys
import tempfile
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import yaml

from tqdm import tqdm

from scenecode.agent_utils.clip_embeddings import get_single_image_embedding

console_logger = logging.getLogger(__name__)

# API configuration.
API_BASE_URL = "https://ambientcg.com/api/v2/full_json"
PAGE_SIZE = 250
MAX_RETRIES = 3
RETRY_DELAY = 1.0


def fetch_materials_page(
    offset: int,
    session: requests.Session,
) -> tuple[list[dict], int]:
    """Fetch a single page of materials from the API.

    Args:
        offset: Pagination offset.
        session: Requests session for connection pooling.

    Returns:
        Tuple of (list of asset dicts, total number of results).
    """
    params = {
        "type": "Material",
        "include": "previewData",
        "limit": PAGE_SIZE,
        "offset": offset,
    }

    response = session.get(url=API_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    return data.get("foundAssets", []), data.get("numberOfResults", 0)


def fetch_all_materials(session: requests.Session) -> list[dict]:
    """Fetch all materials from the API with pagination.

    Args:
        session: Requests session for connection pooling.

    Returns:
        List of all material asset dicts.
    """
    all_assets = []
    offset = 0
    total = None

    console_logger.info("Fetching material metadata from AmbientCG API...")

    while True:
        assets, total_results = fetch_materials_page(offset=offset, session=session)

        if total is None:
            total = total_results
            console_logger.info(f"Found {total} materials in API")

        all_assets.extend(assets)

        if len(assets) < PAGE_SIZE or len(all_assets) >= total:
            break

        offset += PAGE_SIZE
        time.sleep(0.1)

    return all_assets


def get_preview_url(asset: dict, preview_size: int) -> str | None:
    """Extract preview image URL for the specified size.

    Args:
        asset: Asset dict from API.
        preview_size: Desired preview size (e.g., 1024).

    Returns:
        Preview URL or None if not found.
    """
    preview_image = asset.get("previewImage", {})

    # Prefer JPG with white background (better for CLIP), fall back to PNG.
    for key in [f"{preview_size}-JPG-FFFFFF", f"{preview_size}-PNG"]:
        if key in preview_image:
            return preview_image[key]

    return None


def download_preview(url: str, output_path: Path, session: requests.Session) -> bool:
    """Download a preview image with retries.

    Args:
        url: Preview image URL.
        output_path: Path to save the image.
        session: Requests session.

    Returns:
        True if successful, False otherwise.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url=url, timeout=30)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            return True

        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                console_logger.warning(f"Failed to download {url}: {e}")
                return False

    return False


def compute_embeddings(
    materials_dir: Path,
    output_dir: Path,
    preview_size: int,
    concurrent: int,
    dry_run: bool = False,
) -> None:
    """Compute CLIP embeddings for downloaded AmbientCG materials.

    Args:
        materials_dir: Directory containing downloaded materials.
        output_dir: Directory to save embeddings.
        preview_size: Preview image size to use.
        concurrent: Number of concurrent preview downloads.
        dry_run: If True, just list materials without processing.
    """
    session = requests.Session()

    # Fetch all material metadata from API.
    all_assets = fetch_all_materials(session=session)

    # Build mapping of asset_id -> asset data.
    asset_map = {asset.get("assetId"): asset for asset in all_assets}

    # Find downloaded materials.
    downloaded_ids = []
    for child in sorted(materials_dir.iterdir()):
        if child.is_dir() and child.name in asset_map:
            downloaded_ids.append(child.name)

    console_logger.info(
        f"Found {len(downloaded_ids)} downloaded materials in {materials_dir}"
    )

    if not downloaded_ids:
        console_logger.error("No downloaded materials found matching API data")
        return

    if dry_run:
        console_logger.info("Dry run - materials that would be processed:")
        for asset_id in downloaded_ids[:20]:
            asset = asset_map[asset_id]
            category = asset.get("displayCategory", "Unknown")
            console_logger.info(f"  {asset_id} ({category})")
        if len(downloaded_ids) > 20:
            console_logger.info(f"  ... and {len(downloaded_ids) - 20} more")
        return

    # Create temp directory for preview images.
    temp_dir = Path(tempfile.mkdtemp(prefix="ambientcg_previews_"))
    console_logger.info(f"Downloading preview images to {temp_dir}")

    try:
        # Download preview images concurrently.
        preview_paths: dict[str, Path] = {}
        download_tasks = []

        for asset_id in downloaded_ids:
            asset = asset_map[asset_id]
            preview_url = get_preview_url(asset=asset, preview_size=preview_size)

            if preview_url is None:
                console_logger.warning(f"No preview URL for {asset_id}")
                continue

            preview_path = temp_dir / f"{asset_id}.png"
            download_tasks.append((asset_id, preview_url, preview_path))

        # Download previews with progress bar.
        console_logger.info(f"Downloading {len(download_tasks)} preview images...")

        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = {
                executor.submit(
                    download_preview,
                    url=url,
                    output_path=path,
                    session=session,
                ): (asset_id, path)
                for asset_id, url, path in download_tasks
            }

            with tqdm(total=len(futures), desc="Downloading previews") as pbar:
                for future in as_completed(futures):
                    asset_id, path = futures[future]
                    try:
                        if future.result():
                            preview_paths[asset_id] = path
                    except Exception as e:
                        console_logger.warning(f"Error downloading {asset_id}: {e}")
                    pbar.update(1)

        console_logger.info(f"Downloaded {len(preview_paths)} preview images")

        if not preview_paths:
            console_logger.error("No preview images downloaded, exiting")
            return

        # Compute CLIP embeddings.
        console_logger.info("Computing CLIP embeddings...")
        embeddings_list: list[np.ndarray] = []
        embedding_index: list[str] = []
        metadata_index: dict[str, dict] = {}

        # Process in order of asset_id for determinism.
        sorted_asset_ids = sorted(preview_paths.keys())

        for asset_id in tqdm(sorted_asset_ids, desc="Computing embeddings"):
            preview_path = preview_paths[asset_id]
            asset = asset_map[asset_id]

            try:
                embedding = get_single_image_embedding(image_path=preview_path)
                embeddings_list.append(embedding)
                embedding_index.append(asset_id)

                # Extract metadata.
                metadata_index[asset_id] = {
                    "category": asset.get("displayCategory", "Unknown"),
                    "tags": asset.get("tags", []),
                    "dimensions": {
                        "x": asset.get("dimensionX"),
                        "y": asset.get("dimensionY"),
                        "z": asset.get("dimensionZ"),
                    },
                }

            except Exception as e:
                console_logger.warning(f"Failed to embed {asset_id}: {e}")

        console_logger.info(f"Computed {len(embeddings_list)} embeddings")

        if not embeddings_list:
            console_logger.error("No embeddings computed, exiting")
            return

        # Save outputs.
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save embeddings as numpy array.
        embeddings_array = np.stack(embeddings_list, axis=0).astype(np.float32)
        embeddings_file = output_dir / "clip_embeddings.npy"
        np.save(embeddings_file, embeddings_array)
        console_logger.info(
            f"Saved embeddings to {embeddings_file}: shape={embeddings_array.shape}"
        )

        # Save embedding index.
        embedding_index_file = output_dir / "embedding_index.yaml"
        with open(embedding_index_file, "w") as f:
            yaml.dump(embedding_index, f, default_flow_style=False)
        console_logger.info(
            f"Saved embedding index to {embedding_index_file}: "
            f"{len(embedding_index)} entries"
        )

        # Save metadata index.
        metadata_index_file = output_dir / "metadata_index.yaml"
        with open(metadata_index_file, "w") as f:
            yaml.dump(metadata_index, f, default_flow_style=False, allow_unicode=True)
        console_logger.info(
            f"Saved metadata index to {metadata_index_file}: "
            f"{len(metadata_index)} entries"
        )

        console_logger.info("Done!")

    finally:
        # Clean up temp directory.
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute CLIP embeddings for AmbientCG materials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--materials-dir",
        type=Path,
        default=Path("./materials"),
        help="Directory containing downloaded materials (default: ./materials)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for embeddings (default: materials_dir/embeddings)",
    )
    parser.add_argument(
        "--preview-size",
        type=int,
        default=1024,
        choices=[64, 128, 256, 512, 1024, 2048],
        help="Preview image size to use (default: 1024)",
    )
    parser.add_argument(
        "--concurrent",
        "-c",
        type=int,
        default=8,
        help="Concurrent preview downloads (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List materials without processing",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Set up logging.
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate materials directory.
    if not args.materials_dir.exists():
        console_logger.error(f"Materials directory not found: {args.materials_dir}")
        return 1

    # Set default output directory.
    if args.output is None:
        args.output = args.materials_dir / "embeddings"

    # Compute embeddings.
    try:
        compute_embeddings(
            materials_dir=args.materials_dir,
            output_dir=args.output,
            preview_size=args.preview_size,
            concurrent=args.concurrent,
            dry_run=args.dry_run,
        )
        return 0
    except Exception as e:
        console_logger.error(f"Failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
