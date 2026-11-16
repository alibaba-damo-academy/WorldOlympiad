"""Geometry caching for floor plan GLTF files.

Provides caching for wall, floor, and window GLTFs to avoid expensive
regeneration when geometry hasn't changed. Cache keys are SHA-256 hashes
of all properties that affect the GLTF output.
"""

import hashlib
import json
import logging
import shutil

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from scenecode.utils.material import Material

console_logger = logging.getLogger(__name__)


def floor_cache_key(
    width: float,
    depth: float,
    thickness: float,
    material: Material | None,
    texture_scale: float = 0.5,
) -> str:
    """Generate cache key for floor GLTF.

    Args:
        width: Floor width (X dimension) in meters.
        depth: Floor depth (Y dimension) in meters.
        thickness: Floor thickness in meters.
        material: Floor material.
        texture_scale: UV scale for texturing.

    Returns:
        SHA-256 hash string (first 16 chars).
    """
    state = {
        "width": width,
        "depth": depth,
        "thickness": thickness,
        "material": str(material.path) if material else None,
        "texture_scale": texture_scale,
    }
    content_json = json.dumps(state, sort_keys=True)
    return hashlib.sha256(content_json.encode()).hexdigest()[:16]


def wall_cache_key(
    width: float,
    height: float,
    thickness: float,
    material: Material | None,
    openings: list[dict] | None = None,
) -> str:
    """Generate cache key for wall GLTF.

    Args:
        width: Wall width in meters.
        height: Wall height in meters.
        thickness: Wall thickness in meters.
        material: Wall material.
        openings: List of opening dicts (from WallOpening.to_dict()).

    Returns:
        SHA-256 hash string (first 16 chars).
    """
    state = {
        "width": width,
        "height": height,
        "thickness": thickness,
        "material": str(material.path) if material else None,
        "openings": openings or [],
    }
    content_json = json.dumps(state, sort_keys=True)
    return hashlib.sha256(content_json.encode()).hexdigest()[:16]


def window_cache_key(
    width: float, height: float, depth: float, is_horizontal: bool
) -> str:
    """Generate cache key for window frame GLTF.

    Rotation for wall orientation is baked into the mesh during generation,
    but we still need is_horizontal in the cache key since the mesh differs.

    Args:
        width: Window width in meters.
        height: Window height in meters.
        depth: Window depth (wall thickness) in meters.
        is_horizontal: Whether window is on N/S wall (rotation baked into mesh).

    Returns:
        SHA-256 hash string (first 16 chars).
    """
    state = {
        "width": width,
        "height": height,
        "depth": depth,
        "is_horizontal": is_horizontal,
    }
    content_json = json.dumps(state, sort_keys=True)
    return hashlib.sha256(content_json.encode()).hexdigest()[:16]


@dataclass
class GeometryCache:
    """Cache for floor plan geometry GLTFs.

    Stores wall, floor, and window GLTFs in a cache directory, keyed by
    content hashes. When geometry is needed, checks cache first and copies
    to output location if found, otherwise generates and caches.

    Attributes:
        cache_dir: Root directory for cached GLTFs.
    """

    cache_dir: Path

    def __post_init__(self) -> None:
        """Create cache directory structure."""
        self.walls_dir = self.cache_dir / "walls"
        self.floors_dir = self.cache_dir / "floors"
        self.windows_dir = self.cache_dir / "windows"

        self.walls_dir.mkdir(parents=True, exist_ok=True)
        self.floors_dir.mkdir(parents=True, exist_ok=True)
        self.windows_dir.mkdir(parents=True, exist_ok=True)

        # Track cache statistics for logging.
        self._hits = 0
        self._misses = 0

    def _get_cached_gltf_dir(self, subdir: Path, cache_key: str) -> Path:
        """Get path to cached GLTF directory for a cache key."""
        return subdir / cache_key

    def _cache_exists(self, subdir: Path, cache_key: str) -> bool:
        """Check if cached GLTF exists."""
        cached_dir = self._get_cached_gltf_dir(subdir, cache_key)
        # Check for any .gltf file in the directory.
        if not cached_dir.exists():
            return False
        return any(cached_dir.glob("*.gltf"))

    def _copy_from_cache(
        self, subdir: Path, cache_key: str, output_dir: Path
    ) -> Path | None:
        """Copy cached GLTF to output location.

        Args:
            subdir: Cache subdirectory (walls/floors/windows).
            cache_key: Hash key for the cached item.
            output_dir: Directory to copy files to.

        Returns:
            Path to copied GLTF file, or None if cache miss.
        """
        cached_dir = self._get_cached_gltf_dir(subdir, cache_key)
        if not cached_dir.exists():
            return None

        # Find the GLTF file in cache.
        gltf_files = list(cached_dir.glob("*.gltf"))
        if not gltf_files:
            return None

        # Copy all files from cache to output (includes .gltf and .bin files).
        output_dir.mkdir(parents=True, exist_ok=True)
        for cached_file in cached_dir.iterdir():
            if cached_file.is_file():
                shutil.copy2(cached_file, output_dir / cached_file.name)

        return output_dir / gltf_files[0].name

    def _store_in_cache(self, subdir: Path, cache_key: str, source_dir: Path) -> None:
        """Store generated GLTF in cache.

        Args:
            subdir: Cache subdirectory (walls/floors/windows).
            cache_key: Hash key for caching.
            source_dir: Directory containing generated GLTF files.
        """
        cached_dir = self._get_cached_gltf_dir(subdir, cache_key)
        cached_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files to cache.
        for source_file in source_dir.iterdir():
            if source_file.is_file():
                shutil.copy2(source_file, cached_dir / source_file.name)

    def get_or_create_wall(
        self,
        cache_key: str,
        output_dir: Path,
        create_fn: Callable[[Path], None],
    ) -> Path:
        """Get wall GLTF from cache or create it.

        Args:
            cache_key: Wall cache key from Wall.cache_key().
            output_dir: Directory for output GLTF.
            create_fn: Function to create GLTF, called with output path.

        Returns:
            Path to wall GLTF file.
        """
        # Try cache first.
        cached_path = self._copy_from_cache(self.walls_dir, cache_key, output_dir)
        if cached_path is not None:
            self._hits += 1
            console_logger.debug(f"Wall cache HIT: {cache_key[:8]}...")
            return cached_path

        # Cache miss - generate.
        self._misses += 1
        console_logger.debug(f"Wall cache MISS: {cache_key[:8]}...")

        output_dir.mkdir(parents=True, exist_ok=True)
        gltf_path = output_dir / "wall.gltf"
        create_fn(gltf_path)

        # Store in cache.
        self._store_in_cache(self.walls_dir, cache_key, output_dir)

        return gltf_path

    def get_or_create_floor(
        self,
        cache_key: str,
        output_dir: Path,
        create_fn: Callable[[Path], None],
    ) -> Path:
        """Get floor GLTF from cache or create it.

        Args:
            cache_key: Floor cache key from floor_cache_key().
            output_dir: Directory for output GLTF.
            create_fn: Function to create GLTF, called with output path.

        Returns:
            Path to floor GLTF file.
        """
        cached_path = self._copy_from_cache(self.floors_dir, cache_key, output_dir)
        if cached_path is not None:
            self._hits += 1
            console_logger.debug(f"Floor cache HIT: {cache_key[:8]}...")
            return cached_path

        self._misses += 1
        console_logger.debug(f"Floor cache MISS: {cache_key[:8]}...")

        output_dir.mkdir(parents=True, exist_ok=True)
        gltf_path = output_dir / "floor.gltf"
        create_fn(gltf_path)

        self._store_in_cache(self.floors_dir, cache_key, output_dir)

        return gltf_path

    def get_or_create_window(
        self,
        cache_key: str,
        output_dir: Path,
        create_fn: Callable[[Path], None],
    ) -> Path:
        """Get window GLTF from cache or create it.

        Args:
            cache_key: Window cache key from window_cache_key().
            output_dir: Directory for output GLTF.
            create_fn: Function to create GLTF, called with output path.

        Returns:
            Path to window GLTF file.
        """
        cached_path = self._copy_from_cache(self.windows_dir, cache_key, output_dir)
        if cached_path is not None:
            self._hits += 1
            console_logger.debug(f"Window cache HIT: {cache_key[:8]}...")
            return cached_path

        self._misses += 1
        console_logger.debug(f"Window cache MISS: {cache_key[:8]}...")

        output_dir.mkdir(parents=True, exist_ok=True)
        gltf_path = output_dir / "window.gltf"
        create_fn(gltf_path)

        self._store_in_cache(self.windows_dir, cache_key, output_dir)

        return gltf_path

    def get_stats(self) -> dict[str, int]:
        """Get cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def log_stats(self) -> None:
        """Log cache statistics."""
        stats = self.get_stats()
        if stats["total"] > 0:
            console_logger.info(
                f"Geometry cache: {stats['hits']}/{stats['total']} hits "
                f"({stats['hit_rate']:.1%})"
            )
