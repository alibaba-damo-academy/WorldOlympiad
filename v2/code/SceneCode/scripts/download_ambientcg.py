"""Download PBR materials from AmbientCG.

AmbientCG provides free CC0 PBR materials for 3D rendering. This script
downloads materials in bulk with configurable resolution and format.

Usage:
    python scripts/download_ambientcg.py --resolution 2K --format PNG
    python scripts/download_ambientcg.py -r 4K -f JPG -o data/materials -c 8
    python scripts/download_ambientcg.py --dry-run  # List without downloading

API Documentation: https://docs.ambientcg.com/
"""

import argparse
import logging
import sys
import time
import zipfile

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests

from tqdm import tqdm

console_logger = logging.getLogger(__name__)

# API configuration.
API_BASE_URL = "https://ambientcg.com/api/v2/full_json"
DOWNLOAD_BASE_URL = "https://ambientcg.com/get"
PAGE_SIZE = 250
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds


@dataclass
class MaterialDownload:
    """Represents a material to download."""

    asset_id: str
    download_url: str
    file_name: str
    size_bytes: int


def fetch_materials_page(
    offset: int, session: requests.Session
) -> tuple[list[dict], int]:
    """Fetch a single page of materials from the API.

    Args:
        offset: Pagination offset.
        session: Requests session for connection pooling.

    Returns:
        Tuple of (list of asset dicts, total number of results).

    Raises:
        requests.RequestException: If the API request fails.
    """
    params = {
        "type": "Material",
        "include": "downloadData",
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

    console_logger.info("Fetching material list from AmbientCG API...")

    while True:
        assets, total_results = fetch_materials_page(offset=offset, session=session)

        if total is None:
            total = total_results
            console_logger.info(f"Found {total} materials total")

        all_assets.extend(assets)
        console_logger.info(f"Fetched {len(all_assets)}/{total} materials")

        if len(assets) < PAGE_SIZE or len(all_assets) >= total:
            break

        offset += PAGE_SIZE
        time.sleep(0.1)  # Small delay to be respectful.

    return all_assets


def filter_downloads(
    assets: list[dict], resolution: str, file_format: str
) -> list[MaterialDownload]:
    """Filter assets to get download URLs for the specified resolution/format.

    Args:
        assets: List of asset dicts from the API.
        resolution: Desired resolution (e.g., "2K").
        file_format: Desired format (e.g., "PNG").

    Returns:
        List of MaterialDownload objects.
    """
    target_attribute = f"{resolution}-{file_format}"
    downloads = []

    for asset in assets:
        asset_id = asset.get("assetId", "")

        # Navigate the nested download structure.
        download_folders = asset.get("downloadFolders", {})
        default_folder = download_folders.get("default", {})
        filetype_categories = default_folder.get("downloadFiletypeCategories", {})
        zip_category = filetype_categories.get("zip", {})
        download_list = zip_category.get("downloads", [])

        # Find matching download.
        for download in download_list:
            if download.get("attribute") == target_attribute:
                downloads.append(
                    MaterialDownload(
                        asset_id=asset_id,
                        download_url=download.get("downloadLink", ""),
                        file_name=download.get("fileName", ""),
                        size_bytes=download.get("size", 0),
                    )
                )
                break

    return downloads


def download_material(
    material: MaterialDownload,
    output_dir: Path,
    session: requests.Session,
    extract: bool = True,
) -> bool:
    """Download and optionally extract a single material.

    Args:
        material: MaterialDownload object with download info.
        output_dir: Base output directory.
        session: Requests session for connection pooling.
        extract: Whether to extract the ZIP file.

    Returns:
        True if successful, False otherwise.
    """
    material_dir = output_dir / material.asset_id

    # Skip if already downloaded.
    if material_dir.exists() and any(material_dir.iterdir()):
        return True

    zip_path = output_dir / material.file_name

    # Download with retries.
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                url=material.download_url,
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            # Write to file.
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            break

        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                console_logger.warning(
                    f"Retry {attempt + 1}/{MAX_RETRIES} for {material.asset_id}: {e}"
                )
                time.sleep(RETRY_DELAY)
            else:
                console_logger.error(f"Failed to download {material.asset_id}: {e}")
                return False

    # Extract if requested.
    if extract:
        try:
            material_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(material_dir)
            zip_path.unlink()  # Remove ZIP after extraction.
        except zipfile.BadZipFile as e:
            console_logger.error(f"Failed to extract {material.asset_id}: {e}")
            zip_path.unlink(missing_ok=True)
            return False

    return True


def download_all_materials(
    downloads: list[MaterialDownload],
    output_dir: Path,
    concurrent: int,
    extract: bool,
) -> tuple[int, int]:
    """Download all materials with concurrent workers.

    Args:
        downloads: List of MaterialDownload objects.
        output_dir: Base output directory.
        concurrent: Number of concurrent download workers.
        extract: Whether to extract ZIP files.

    Returns:
        Tuple of (successful count, failed count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    # Use session for connection pooling.
    session = requests.Session()

    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = {
            executor.submit(
                download_material,
                material=material,
                output_dir=output_dir,
                session=session,
                extract=extract,
            ): material
            for material in downloads
        }

        with tqdm(total=len(downloads), desc="Downloading materials") as pbar:
            for future in as_completed(futures):
                material = futures[future]
                try:
                    if future.result():
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    console_logger.error(f"Error downloading {material.asset_id}: {e}")
                    failed += 1
                pbar.update(1)

    return successful, failed


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download PBR materials from AmbientCG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=str,
        default="2K",
        choices=["1K", "2K", "4K", "8K", "12K", "16K"],
        help="Resolution to download (default: 2K)",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="JPG",
        choices=["PNG", "JPG"],
        help="File format (default: JPG)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/materials"),
        help="Output directory (default: data/materials)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Max materials to download, 0 for all (default: 0)",
    )
    parser.add_argument(
        "--concurrent",
        "-c",
        type=int,
        default=4,
        help="Concurrent downloads (default: 4)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep ZIP files without extracting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List materials without downloading",
    )

    args = parser.parse_args()

    # Set up logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Fetch all materials.
    session = requests.Session()
    assets = fetch_all_materials(session=session)

    # Filter to matching downloads.
    downloads = filter_downloads(
        assets=assets, resolution=args.resolution, file_format=args.format
    )

    if not downloads:
        console_logger.error(
            f"No materials found with {args.resolution}-{args.format} format"
        )
        return 1

    # Apply limit.
    if args.limit > 0:
        downloads = downloads[: args.limit]

    # Calculate total size.
    total_size = sum(d.size_bytes for d in downloads)
    console_logger.info(
        f"Found {len(downloads)} materials "
        f"({format_size(total_size)} total) matching {args.resolution}-{args.format}"
    )

    # Dry run - just list materials.
    if args.dry_run:
        console_logger.info("Dry run - listing materials:")
        for d in downloads[:20]:  # Show first 20.
            console_logger.info(f"  {d.asset_id}: {format_size(d.size_bytes)}")
        if len(downloads) > 20:
            console_logger.info(f"  ... and {len(downloads) - 20} more")
        return 0

    # Download materials.
    console_logger.info(f"Downloading to {args.output.resolve()}")
    successful, failed = download_all_materials(
        downloads=downloads,
        output_dir=args.output,
        concurrent=args.concurrent,
        extract=not args.no_extract,
    )

    console_logger.info(f"Download complete: {successful} successful, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
