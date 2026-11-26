"""Shared filtering logic for support surfaces."""

import logging

import numpy as np

from scenecode.agent_utils.room import SupportSurface
from scenecode.utils.geometry_utils import safe_convex_hull_2d

console_logger = logging.getLogger(__name__)


def filter_surface_by_inscribed_radius(
    surface: SupportSurface,
    min_inscribed_radius_m: float,
) -> tuple[bool, str | None]:
    """Filter surface by inscribed radius using convex hull analysis.

    Computes the true inscribed radius of the surface by:
    1. Computing 2D convex hull of surface mesh vertices
    2. Finding minimum distance from centroid to hull edges
    3. Capping at half the smaller bounding box dimension (prevents inflation
       for thin rectangles)

    This is more robust than simple width/depth thresholds as it:
    - Handles degenerate geometry (collinear/duplicate vertices)
    - Accounts for actual surface shape, not just bounding box
    - Prevents false positives on thin rectangles

    Args:
        surface: SupportSurface object with mesh geometry.
        min_inscribed_radius_m: Minimum inscribed radius threshold in meters.

    Returns:
        Tuple of (should_keep, rejection_reason):
        - should_keep: True if surface passes filter, False otherwise
        - rejection_reason: Human-readable reason if rejected, None if kept
    """
    # Surface must have mesh geometry for convex hull analysis.
    if surface.mesh is None:
        return False, "missing mesh geometry"

    # Extract 2D vertices (XY plane) from mesh.
    xy_vertices = surface.mesh.vertices[:, :2]

    # Handle degenerate geometry gracefully using safe wrapper.
    hull, processed_vertices = safe_convex_hull_2d(xy_vertices)
    if hull is None:
        return (
            False,
            "degenerate geometry (ConvexHull failed - collinear/duplicate vertices)",
        )

    hull_vertices = processed_vertices[hull.vertices]

    # Compute 2D bounding box for capping inscribed radius.
    x_min, y_min = xy_vertices.min(axis=0)
    x_max, y_max = xy_vertices.max(axis=0)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # Compute inscribed radius (improved approximation).
    # Use centroid-to-hull distance, but cap at half the bbox dimensions.
    centroid_2d = np.mean(hull_vertices, axis=0)
    distances = np.linalg.norm(hull_vertices - centroid_2d, axis=1)
    centroid_inscribed_radius = np.min(distances)

    # Better approximation: cap at half the smaller bbox dimension.
    # This prevents long thin rectangles from having inflated inscribed radius.
    inscribed_radius = min(
        centroid_inscribed_radius, bbox_width / 2.0, bbox_height / 2.0
    )

    if inscribed_radius < min_inscribed_radius_m:
        reason = (
            f"inscribed radius {inscribed_radius:.3f}m "
            f"< {min_inscribed_radius_m:.3f}m threshold"
        )
        return False, reason

    return True, None


def filter_surface_by_area(
    surface: SupportSurface,
    min_area_m2: float,
) -> tuple[bool, str | None]:
    """Filter surface by minimum area threshold.

    Args:
        surface: SupportSurface object.
        min_area_m2: Minimum area threshold in square meters.

    Returns:
        Tuple of (should_keep, rejection_reason).
    """
    area = surface.area

    if area < min_area_m2:
        reason = f"area {area:.3f}m² < {min_area_m2:.3f}m² threshold"
        return False, reason

    return True, None
