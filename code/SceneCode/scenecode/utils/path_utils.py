"""Path manipulation utilities for scene serialization."""

from pathlib import Path


def safe_relative_path(path: Path, base_dir: Path | None) -> str:
    """Get path relative to base_dir, or absolute path if outside base_dir.

    Used for serialization to ensure paths are portable across different
    scene directories.

    Args:
        path: The path to make relative.
        base_dir: The base directory to compute relative path from.

    Returns:
        Relative path string if path is under base_dir, otherwise absolute path.
    """
    if base_dir is None:
        return str(path)
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        # Path is outside base_dir, use absolute path.
        return str(path.absolute())
