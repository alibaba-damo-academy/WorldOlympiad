"""Load pre-validated support surfaces from HSM for HSSD assets.

This module provides functionality to load support surfaces that were pre-computed
by HSM authors for HSSD objects.

The support surfaces are stored in compressed JSON format (.json.gz) in the HSSD
data directory under support-surfaces/{mesh_id}/{mesh_id}.supportSurface.json.gz.
"""

import gzip
import json
import logging

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from pydrake.math import RigidTransform, RollPitchYaw

from scenecode.agent_utils.room import SupportSurface
from scenecode.agent_utils.support_surface_filters import (
    filter_surface_by_area,
    filter_surface_by_inscribed_radius,
)

console_logger = logging.getLogger(__name__)


@dataclass
class HsmSupportSurfaceData:
    """HSM support surface data from JSON."""

    index: int
    """Surface index."""

    area: float
    """Surface area."""

    normal: np.ndarray
    """Surface normal vector in HSM coordinates (Y-up, Z-forward)."""

    is_horizontal: bool
    """Whether surface is horizontal."""

    corners: np.ndarray
    """8 corner points defining the bounding box in HSM coordinates."""

    clearance: float
    """Minimum clearance height above the surface in meters."""


def _compute_obb_corners(obb_data: dict) -> np.ndarray:
    """Compute 8 corner points from HSM OBB (oriented bounding box) data.

    Ported from HSM's BoundingBox.corners property
    (hsm_core/scene_motif/core/bounding_box.py).

    HSM stores OBBs as (centroid, axesLengths, normalizedAxes) rather than
    explicit corners. This function computes corners following HSM's algorithm.

    Args:
        obb_data: Dictionary with keys:
            - centroid: [x, y, z] center point
            - axesLengths: [width, height, depth] full dimensions (not half-extents!)
            - normalizedAxes: 9 floats representing 3x3 rotation matrix (row-major)

    Returns:
        Array of shape (8, 3) containing the 8 corner points in HSM's order.
    """
    centroid = np.array(obb_data["centroid"])
    axes_lengths = np.array(obb_data["axesLengths"])
    normalized_axes = np.array(obb_data["normalizedAxes"]).reshape(3, 3)

    # Half sizes (axes_lengths are FULL dimensions in HSM).
    half_size = axes_lengths / 2.0

    # Compute min point in OBB local frame.
    min_local = (
        centroid
        - half_size[0] * normalized_axes[0]
        - half_size[1] * normalized_axes[1]
        - half_size[2] * normalized_axes[2]
    )

    # Generate 8 corners in HSM's order (matching BoundingBox.corners property).
    corners = np.zeros((8, 3))
    corners[0] = min_local
    corners[1] = min_local + axes_lengths[0] * normalized_axes[0]
    corners[2] = min_local + axes_lengths[2] * normalized_axes[2]
    corners[3] = (
        min_local
        + axes_lengths[0] * normalized_axes[0]
        + axes_lengths[2] * normalized_axes[2]
    )
    corners[4] = min_local + axes_lengths[1] * normalized_axes[1]
    corners[5] = (
        min_local
        + axes_lengths[0] * normalized_axes[0]
        + axes_lengths[1] * normalized_axes[1]
    )
    corners[6] = (
        min_local
        + axes_lengths[1] * normalized_axes[1]
        + axes_lengths[2] * normalized_axes[2]
    )
    corners[7] = (
        centroid
        + half_size[0] * normalized_axes[0]
        + half_size[1] * normalized_axes[1]
        + half_size[2] * normalized_axes[2]
    )

    return corners


def _filter_surfaces_by_quality(
    surfaces: list[SupportSurface],
    min_area_m2: float,
    min_inscribed_radius_m: float,
) -> list[SupportSurface]:
    """Filter support surfaces by area and inscribed radius.

    Uses robust convex hull-based inscribed radius computation instead of simple
    dimension thresholds. This provides consistent filtering across HSSD and
    generative assets.

    Args:
        surfaces: List of SupportSurface objects.
        min_area_m2: Minimum area in m².
        min_inscribed_radius_m: Minimum inscribed radius in meters.

    Returns:
        Filtered list of surfaces that meet quality criteria.
    """
    filtered = []

    for surface in surfaces:
        # Filter by area.
        keep, reason = filter_surface_by_area(surface=surface, min_area_m2=min_area_m2)
        if not keep:
            console_logger.debug(f"Filtering surface {surface.surface_id}: {reason}")
            continue

        # Filter by inscribed radius (convex hull-based).
        keep, reason = filter_surface_by_inscribed_radius(
            surface=surface, min_inscribed_radius_m=min_inscribed_radius_m
        )
        if not keep:
            console_logger.debug(f"Filtering surface {surface.surface_id}: {reason}")
            continue

        filtered.append(surface)

    console_logger.info(
        f"Quality filtering: {len(surfaces)} → {len(filtered)} surfaces"
    )
    return filtered


def _filter_surfaces_by_layer_spacing(
    surfaces: list[SupportSurface],
    min_spacing: float,
    top_clearance: float,
) -> list[SupportSurface]:
    """Filter surfaces by layer spacing to match HSM's behavior.

    Ported from HSM's _calculate_layer_heights_and_filter() function.

    Groups surfaces by Z-height and filters out surfaces that are too close
    to the surface above them. This prevents multiple overlapping surfaces
    at slightly different heights from all being kept.

    Args:
        surfaces: List of SupportSurface objects.
        min_spacing: Minimum spacing to surface above in meters (5cm default).
        top_clearance: Default clearance for top surfaces without obstacles above.

    Returns:
        Filtered list with surfaces too close to layer above removed.
    """
    if len(surfaces) == 0:
        return surfaces

    # Sort surfaces by Z-height (ascending).
    # Use transform.translation()[2] to get Z coordinate of surface origin.
    surfaces_with_heights = [
        (surface, surface.transform.translation()[2]) for surface in surfaces
    ]
    surfaces_with_heights.sort(key=lambda x: x[1])

    # Compute spacing to next layer above for each surface.
    # Build dict: height -> spacing_to_next_layer.
    layer_heights = {}
    for i in range(len(surfaces_with_heights) - 1):
        current_height = surfaces_with_heights[i][1]
        next_height = surfaces_with_heights[i + 1][1]
        spacing = next_height - current_height
        layer_heights[current_height] = spacing

    # Filter surfaces by spacing.
    filtered = []
    for surface, height in surfaces_with_heights:
        # Get spacing to layer above (use top_clearance if no layer above).
        space_above = layer_heights.get(height, top_clearance)

        if space_above >= min_spacing:
            filtered.append(surface)
        else:
            console_logger.debug(
                f"Filtering surface {surface.surface_id}: spacing to layer above "
                f"{space_above:.3f}m < minimum {min_spacing:.3f}m"
            )

    console_logger.info(
        f"Layer spacing filtering: {len(surfaces)} → {len(filtered)} surfaces"
    )

    # Sort by area (largest first) to match HSM's original ordering.
    # This ensures support_surfaces[0] is the largest surface for placement tests.
    filtered.sort(key=lambda s: s.area, reverse=True)

    return filtered


def _convert_hsm_to_scenecode_coords(points: np.ndarray) -> np.ndarray:
    """Convert coordinates from HSM (Y-up, Z-forward) to scenecode (Z-up, Y-forward).

    HSM surfaces are stored in Y-up, Z-forward format.
    GLTF meshes are also stored in Y-up, but Drake converts them to Z-up when loading.
    Support surfaces must be converted to match Drake's Z-up representation.

    Transformation: [X, Y, Z]_HSM → [X, Z, -Y]_scenecode

    Args:
        points: Array of shape (N, 3) in HSM coordinates (Y-up, Z-forward).

    Returns:
        Array of shape (N, 3) in scenecode coordinates (Z-up, Y-forward).
    """
    # HSM: X-right, Y-up, Z-forward
    # scenecode/Drake: X-right, Y-forward, Z-up
    # Transform: [X, Y, Z]_HSM → [X, -Z, Y]_scenecode
    # - X stays X (right direction in both)
    # - Y (up in HSM) becomes Z (up in scenecode)
    # - Z (forward in HSM) becomes Y (forward in scenecode), but negated because
    #   HSM Z-forward points in opposite direction to scenecode Y-forward
    return np.array([points[:, 0], -points[:, 2], points[:, 1]]).T


def _load_hsm_support_surfaces(
    json_path: Path, top_surface_clearance_m: float
) -> list[HsmSupportSurfaceData]:
    """Load HSM support surface data from compressed JSON file.

    Args:
        json_path: Path to .json.gz file containing support surface data.
        top_surface_clearance_m: Default clearance for top surfaces without samples.

    Returns:
        List of HsmSupportSurfaceData objects in HSM coordinates.

    Raises:
        FileNotFoundError: If JSON file does not exist.
        ValueError: If JSON data is malformed.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Support surface file not found: {json_path}")

    try:
        with gzip.open(json_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from {json_path}: {e}") from e

    if "supportSurfaces" not in data:
        raise ValueError(f"Missing 'supportSurfaces' key in {json_path}")

    surfaces: list[HsmSupportSurfaceData] = []

    for surface_data in data["supportSurfaces"]:
        # Extract relevant fields.
        index = surface_data["index"]
        area = surface_data["area"]
        normal = np.array(
            [
                surface_data["modelNormal"]["x"],
                surface_data["modelNormal"]["y"],
                surface_data["modelNormal"]["z"],
            ]
        )
        is_horizontal = surface_data["isHorizontal"]

        # Parse OBB to corner points.
        # Use "obb" (world space after modelToWorld) not "modelObb" (model space).
        # HSSD meshes have modelToWorld baked into the GLTF, so Drake loads them
        # in HSM world space. Support surfaces must match this frame.
        obb_data = surface_data["obb"]
        corners_hsm = _compute_obb_corners(obb_data)

        # Extract clearance from samples (use minimum clearance for safety).
        samples = surface_data.get("samples", [])
        if samples:
            clearances = [s["clearance"] for s in samples if "clearance" in s]
            # Use minimum clearance (most conservative estimate).
            clearance = min(clearances) if clearances else top_surface_clearance_m
        else:
            # No samples - this is a top surface, use configured clearance.
            clearance = top_surface_clearance_m

        surface = HsmSupportSurfaceData(
            index=index,
            area=area,
            normal=normal,
            is_horizontal=is_horizontal,
            corners=corners_hsm,
            clearance=clearance,
        )
        surfaces.append(surface)

    console_logger.debug(
        f"Loaded {len(surfaces)} support surfaces from {json_path.name}"
    )

    return surfaces


def _corners_to_bbox_and_transform(
    corners: np.ndarray, clearance: float, surface_offset_m: float = 0.01
) -> tuple[np.ndarray, np.ndarray, RigidTransform]:
    """Convert 8 corner points to bounding box and transform with clearance.

    Args:
        corners: Array of shape (8, 3) defining bounding box corners.
        clearance: Clearance height above the surface in meters.
        surface_offset_m: Gravity settling offset (default 0.01m).

    Returns:
        Tuple of (bbox_min, bbox_max, transform) where:
        - bbox_min/bbox_max are in surface-local frame with clearance
        - transform places the bbox center at the centroid
    """
    # Compute centroid (will be the origin of surface frame).
    centroid = corners.mean(axis=0)

    # Compute bounding box extents around centroid.
    bbox_min = corners.min(axis=0) - centroid
    bbox_max = corners.max(axis=0) - centroid

    # Apply clearance to Z dimension (vertical in scenecode coords).
    # bbox_min[2] = surface_offset_m (gravity settling offset).
    # bbox_max[2] = surface_offset_m + clearance (available space).
    bbox_min[2] = surface_offset_m
    bbox_max[2] = surface_offset_m + clearance

    # Create transform with offset centroid as position, identity rotation.
    # HSM surfaces are already axis-aligned (OBB with identity axes).
    # Offset the centroid upward by surface_offset_m to match extracted surfaces.
    offset_centroid = centroid.copy()
    offset_centroid[2] += surface_offset_m
    transform = RigidTransform(p=offset_centroid, rpy=RollPitchYaw([0.0, 0.0, 0.0]))

    return bbox_min, bbox_max, transform


def load_hssd_support_surfaces(
    mesh_id: str, config, scene: "RoomScene", data_dir: Path | None = None
) -> list[SupportSurface] | None:
    """Load pre-validated support surfaces for HSSD asset.

    Note: Returns surfaces in mesh-local frame (identity transform). Caller must
    transform to world frame using furniture object's transform.

    Args:
        mesh_id: HSSD mesh ID (SHA-1 hash).
        config: Configuration object with clearance settings.
        scene: RoomScene object for generating unique surface IDs.
        data_dir: Optional data directory. If None, uses default from project root.

    Returns:
        List of SupportSurface objects in scenecode coordinates (Z-up, Y-forward,
        mesh-local frame), or None if surfaces not found or loading fails.
    """
    if data_dir is None:
        # Default to project root data directory.
        data_dir = Path(__file__).parent.parent.parent.parent / "data"

    support_surfaces_dir = data_dir / "hssd-models" / "support-surfaces"
    json_path = support_surfaces_dir / mesh_id / f"{mesh_id}.supportSurface.json.gz"

    if not json_path.exists():
        console_logger.debug(
            f"Pre-validated support surfaces not found for mesh {mesh_id[:8]}. "
            "Will recompute using HSM algorithm."
        )
        return None

    try:
        # Load HSM support surfaces in HSM coordinates.
        hsm_surfaces = _load_hsm_support_surfaces(
            json_path=json_path,
            top_surface_clearance_m=config.top_surface_clearance_m,
        )

        # Convert to scenecode coordinates and create SupportSurface objects.
        support_surfaces: list[SupportSurface] = []

        for hsm_surface in hsm_surfaces:
            # Skip non-horizontal surfaces (HSM provides all surfaces, we only want
            # horizontal ones for placement).
            if not hsm_surface.is_horizontal:
                continue

            # Convert corners from HSM to scenecode coordinates.
            corners_scenecode = _convert_hsm_to_scenecode_coords(hsm_surface.corners)

            # Adjust clearance for surface offset (match HSM-computed behavior).
            # This prevents fake collisions by accounting for the gravity settling offset.
            clearance_adjusted = hsm_surface.clearance - config.surface_offset_m

            # Filter out surfaces with insufficient clearance (use adjusted value).
            if clearance_adjusted < config.min_clearance_m:
                console_logger.debug(
                    f"Filtering surface {hsm_surface.index}: adjusted clearance "
                    f"{clearance_adjusted:.3f}m < min {config.min_clearance_m}m "
                    f"(raw clearance: {hsm_surface.clearance:.3f}m, offset: "
                    f"{config.surface_offset_m:.3f}m)"
                )
                continue

            # Convert to bbox + transform representation with clearance.
            bbox_min, bbox_max, transform = _corners_to_bbox_and_transform(
                corners=corners_scenecode,
                clearance=clearance_adjusted,
                surface_offset_m=config.surface_offset_m,
            )

            # Create simple rectangular mesh from bounding box for visualization.
            # The surface plane is at Z=0 in surface-local coords. The transform
            # origin is already offset by surface_offset_m above the physical surface,
            # so mesh vertices at Z=0 represent the offset placement surface.
            x_min, y_min, _ = bbox_min
            x_max, y_max, _ = bbox_max
            # Create 4 corners of the rectangle at the placement surface (Z=0).
            vertices = np.array(
                [
                    [x_min, y_min, 0.0],  # Bottom-left.
                    [x_max, y_min, 0.0],  # Bottom-right.
                    [x_max, y_max, 0.0],  # Top-right.
                    [x_min, y_max, 0.0],  # Top-left.
                ]
            )
            # Create 2 triangles to form the rectangle.
            # Counter-clockwise winding for upward-facing normal.
            faces = np.array(
                [
                    [0, 1, 2],  # First triangle.
                    [0, 2, 3],  # Second triangle.
                ]
            )
            surface_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            # Create SupportSurface object (mesh-local frame).
            # Use scene's generate_surface_id() for consistent S_0, S_1, S_2, ... format.
            surface = SupportSurface(
                surface_id=scene.generate_surface_id(),
                bounding_box_min=bbox_min,
                bounding_box_max=bbox_max,
                transform=transform,
                mesh=surface_mesh,
            )
            support_surfaces.append(surface)

        console_logger.info(
            f"Loaded {len(support_surfaces)} raw surfaces for HSSD mesh {mesh_id[:8]}"
        )

        # Filter by area and inscribed radius.
        support_surfaces = _filter_surfaces_by_quality(
            support_surfaces,
            min_area_m2=config.min_surface_area_m2,
            min_inscribed_radius_m=config.min_inscribed_radius_m,
        )

        # Filter by layer spacing.
        support_surfaces = _filter_surfaces_by_layer_spacing(
            support_surfaces,
            min_spacing=config.min_clearance_m,
            top_clearance=config.top_surface_clearance_m,
        )

        console_logger.info(
            f"After HSM filtering: {len(support_surfaces)} usable surfaces "
            f"for HSSD mesh {mesh_id[:8]}"
        )

        return support_surfaces

    except Exception as e:
        console_logger.warning(
            f"Failed to load pre-validated support surfaces for mesh {mesh_id[:8]}: {e}. "
            "Will recompute using HSM algorithm.",
            exc_info=True,
        )
        return None
