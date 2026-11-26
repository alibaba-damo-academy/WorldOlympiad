"""3D shape analysis utilities for geometric object classification.

This module provides capabilities for analyzing 3D object geometry to detect
shape characteristics like circular/cylindrical objects. Uses volume ratio
analysis with caching for efficient repeated queries.
"""

import logging

import numpy as np
import trimesh

from omegaconf import DictConfig

from scenecode.agent_utils.room import SceneObject
from scenecode.utils.geometry_utils import convert_bbox_zup_to_yup

console_logger = logging.getLogger(__name__)


# Module-level cache for circular object detection.
# Maps geometry_path -> is_circular boolean for efficient lookup.
_CIRCULAR_DETECTION_CACHE: dict[str, bool] = {}


def is_circular_object(obj: SceneObject, cfg: DictConfig) -> bool:
    """Detect if object is circular/cylindrical using volume ratio.

    Circular objects (cylinders, round tables, round ottomans) have convex hull
    volume significantly less than AABB volume due to corner space. Rectangular
    objects have ratio closer to 1.0.

    This uses a two-level cache:
    1. Module-level cache keyed by geometry_path (shared across all objects)
    2. Computed from mesh volume ratio (expensive, cached after first call)

    Args:
        obj: Scene object to check.
        cfg: Configuration with snap_to_object.circular_detection_volume_ratio_threshold.

    Returns:
        True if object appears circular (below threshold), False otherwise.

    Examples:
        - Cylinder: volume = πr²h, AABB = 4r²h → ratio = π/4 ≈ 0.785
        - Octagon: ratio ≈ 0.828
        - Rectangle/Square: ratio ≈ 1.0
    """
    if not obj.geometry_path:
        return False

    geometry_path_str = str(obj.geometry_path)

    # Check cache first (fast path for identical tables/chairs).
    if geometry_path_str in _CIRCULAR_DETECTION_CACHE:
        return _CIRCULAR_DETECTION_CACHE[geometry_path_str]

    # Cache miss - compute from mesh geometry.
    if obj.bbox_min is None or obj.bbox_max is None:
        _CIRCULAR_DETECTION_CACHE[geometry_path_str] = False
        return False

    # Convert bbox from Z-up (Drake storage) to Y-up (GLTF/trimesh).
    bbox_min_yup, bbox_max_yup = convert_bbox_zup_to_yup(obj.bbox_min, obj.bbox_max)

    try:
        # Load mesh and compute volumes (mesh is in Y-up from GLTF).
        mesh = trimesh.load(obj.geometry_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        if not isinstance(mesh, trimesh.Trimesh):
            _CIRCULAR_DETECTION_CACHE[geometry_path_str] = False
            return False

        # Apply scale_factor to mesh vertices so mesh volume matches bbox scale.
        # bbox_min/bbox_max are already scaled (updated by apply_scale).
        if obj.scale_factor != 1.0:
            mesh.vertices *= obj.scale_factor

        # Use convex hull volume for robustness (handles non-manifold meshes).
        mesh_volume = mesh.convex_hull.volume

        # AABB volume in object frame (Y-up to match mesh).
        bbox_size = bbox_max_yup - bbox_min_yup
        aabb_volume = float(np.prod(bbox_size))

        # Check for degenerate volume (cubic meters scale requires smaller epsilon).
        if aabb_volume < 1e-9:
            _CIRCULAR_DETECTION_CACHE[geometry_path_str] = False
            return False

        volume_ratio = mesh_volume / aabb_volume

        # Get threshold from config with validation.
        if not hasattr(cfg.snap_to_object, "circular_detection_volume_ratio_threshold"):
            raise ValueError(
                "Missing required config: "
                "snap_to_object.circular_detection_volume_ratio_threshold"
            )

        threshold = cfg.snap_to_object.circular_detection_volume_ratio_threshold

        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"circular_detection_volume_ratio_threshold must be in [0.0, 1.0], "
                f"got {threshold}"
            )

        is_circular = volume_ratio < threshold

        # Cache result for future lookups (benefits identical objects).
        _CIRCULAR_DETECTION_CACHE[geometry_path_str] = is_circular

        console_logger.info(
            f"Circular detection for {obj.name}: volume_ratio={volume_ratio:.3f}, "
            f"threshold={threshold:.3f}, is_circular={is_circular}"
        )

        return is_circular

    except Exception as e:
        console_logger.warning(
            f"Failed to compute volume ratio for {obj.name}: {e}. "
            f"Assuming non-circular."
        )
        # Cache negative result to avoid repeated failures.
        _CIRCULAR_DETECTION_CACHE[geometry_path_str] = False
        return False
