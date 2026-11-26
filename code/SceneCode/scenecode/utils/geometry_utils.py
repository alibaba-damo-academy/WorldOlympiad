"""Pure geometric utilities for coordinate transformations and calculations.

This module provides coordinate system conversions (Y-up/Z-up), rigid transform
operations, and geometric calculations. Depends only on numpy, trimesh, and
PyDrake for maximum reusability.
"""

import logging
import math

import numpy as np
import trimesh

from pydrake.math import RigidTransform
from scipy.spatial import ConvexHull
from scipy.spatial._qhull import QhullError

console_logger = logging.getLogger(__name__)


def convert_mesh_yup_to_zup(mesh: trimesh.Trimesh) -> None:
    """Convert mesh vertices from Y-up (GLTF) to Z-up (Drake) coordinates in-place.

    GLTF/Blender use Y-up coordinate system, while Drake uses Z-up.
    Drake automatically converts Y-up GLTF to Z-up when loading meshes via SDF,
    but trimesh.load() keeps meshes in their native Y-up coordinates.

    This function must be called before applying Drake RigidTransform to meshes
    loaded with trimesh.

    Transformation: (x, y, z)_yup → (x, -z, y)_zup
    - X stays the same (right direction in both)
    - Y (up in GLTF) becomes Z (up in Drake)
    - Z (depth in GLTF) becomes -Y (forward/back in Drake, inverted)

    Args:
        mesh: Mesh with vertices in Y-up coordinates (modified in-place).
    """
    vertices_zup = np.column_stack(
        [
            mesh.vertices[:, 0],  # X unchanged
            -mesh.vertices[:, 2],  # Y = -Z_yup
            mesh.vertices[:, 1],  # Z = Y_yup
        ]
    )
    mesh.vertices = vertices_zup


def convert_bbox_zup_to_yup(
    bbox_min: np.ndarray, bbox_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert bounding box from Z-up (Drake) to Y-up (GLTF/trimesh) coordinates.

    Drake stores bounding boxes in Z-up coordinates, but trimesh meshes loaded
    with trimesh.load() are in Y-up coordinates. This function converts bbox
    coordinates to match the mesh coordinate system for operations like volume
    comparison.

    Transformation: (x, y, z)_zup → (x, -z, y)_yup
    - X stays the same
    - Y (forward in Drake) becomes -Z (depth in GLTF, inverted)
    - Z (up in Drake) becomes Y (up in GLTF)

    Note: This transformation is involutive (its own inverse).

    Args:
        bbox_min: Minimum corner in Z-up coordinates.
        bbox_max: Maximum corner in Z-up coordinates.

    Returns:
        Tuple of (bbox_min_yup, bbox_max_yup) with ensured min < max ordering.
    """
    bbox_min_yup = np.array([bbox_min[0], -bbox_min[2], bbox_min[1]])
    bbox_max_yup = np.array([bbox_max[0], -bbox_max[2], bbox_max[1]])

    # Ensure min < max after transformation (negation can swap order).
    return (
        np.minimum(bbox_min_yup, bbox_max_yup),
        np.maximum(bbox_min_yup, bbox_max_yup),
    )


def convert_bbox_yup_to_zup(
    bbox_min: np.ndarray, bbox_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert bounding box from Y-up (GLTF/trimesh) to Z-up (Drake) coordinates.

    When computing bounding boxes from trimesh meshes (which are in Y-up), use
    this function to convert them to Drake's Z-up coordinate system for storage
    in SceneObject.

    Transformation: (x, y, z)_yup → (x, -z, y)_zup
    - X stays the same
    - Y (up in GLTF) becomes Z (up in Drake)
    - Z (depth in GLTF) becomes -Y (forward in Drake, inverted)

    Note: This transformation is involutive (its own inverse).

    Args:
        bbox_min: Minimum corner in Y-up coordinates.
        bbox_max: Maximum corner in Y-up coordinates.

    Returns:
        Tuple of (bbox_min_zup, bbox_max_zup) with ensured min < max ordering.
    """
    bbox_min_zup = np.array([bbox_min[0], -bbox_min[2], bbox_min[1]])
    bbox_max_zup = np.array([bbox_max[0], -bbox_max[2], bbox_max[1]])

    # Ensure min < max after transformation (negation can swap order).
    return (
        np.minimum(bbox_min_zup, bbox_max_zup),
        np.maximum(bbox_min_zup, bbox_max_zup),
    )


def ray_rectangle_intersection_2d(
    ray_origin_2d: np.ndarray,
    ray_direction_2d: np.ndarray,
    rect_min_2d: np.ndarray,
    rect_max_2d: np.ndarray,
) -> bool:
    """Check if a 2D ray intersects an axis-aligned rectangle.

    Uses the slab method for 2D ray-rectangle intersection testing.
    This is useful for horizontal facing checks where Z-height differences
    should not affect the result (e.g., chair at different height than table).

    Args:
        ray_origin_2d: Ray origin point [x, y].
        ray_direction_2d: Ray direction vector [x, y] (should be normalized).
        rect_min_2d: Rectangle minimum corner [x, y].
        rect_max_2d: Rectangle maximum corner [x, y].

    Returns:
        True if the ray intersects the rectangle, False otherwise.
        Returns True when ray origin is inside the rectangle, which is correct
        for facing checks (e.g., chair tucked under table).
    """
    # Initialize t_min and t_max for the intersection interval.
    t_min = -np.inf
    t_max = np.inf

    # Test intersection with each slab (x, y).
    for i in range(2):
        if abs(ray_direction_2d[i]) < 1e-8:
            # Ray is parallel to slab. Check if origin is within slab bounds.
            if ray_origin_2d[i] < rect_min_2d[i] or ray_origin_2d[i] > rect_max_2d[i]:
                return False
        else:
            # Compute intersection t value with slab lines.
            t1 = (rect_min_2d[i] - ray_origin_2d[i]) / ray_direction_2d[i]
            t2 = (rect_max_2d[i] - ray_origin_2d[i]) / ray_direction_2d[i]

            # Ensure t1 is entry and t2 is exit.
            if t1 > t2:
                t1, t2 = t2, t1

            # Update the intersection interval.
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            # Early exit if interval becomes invalid.
            if t_min > t_max:
                return False

    # Intersection occurs if the interval is valid and includes t >= 0.
    return t_max >= 0.0


def ray_aabb_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> bool:
    """Check if a ray intersects an axis-aligned bounding box.

    Uses the slab method for ray-AABB intersection testing.

    Args:
        ray_origin: Ray origin point [x, y, z].
        ray_direction: Ray direction vector [x, y, z] (should be normalized).
        bbox_min: AABB minimum corner [x, y, z].
        bbox_max: AABB maximum corner [x, y, z].

    Returns:
        True if the ray intersects the AABB, False otherwise.
    """
    # Initialize t_min and t_max for the intersection interval.
    t_min = -np.inf
    t_max = np.inf

    # Test intersection with each slab (x, y, z).
    for i in range(3):
        if abs(ray_direction[i]) < 1e-8:
            # Ray is parallel to slab. Check if origin is within slab bounds.
            if ray_origin[i] < bbox_min[i] or ray_origin[i] > bbox_max[i]:
                return False
        else:
            # Compute intersection t value with slab planes.
            t1 = (bbox_min[i] - ray_origin[i]) / ray_direction[i]
            t2 = (bbox_max[i] - ray_origin[i]) / ray_direction[i]

            # Ensure t1 is entry and t2 is exit.
            if t1 > t2:
                t1, t2 = t2, t1

            # Update the intersection interval.
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            # Early exit if interval becomes invalid.
            if t_min > t_max:
                return False

    # Intersection occurs if the interval is valid and includes t >= 0.
    return t_max >= 0.0


def closest_point_on_aabb(
    point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray
) -> np.ndarray:
    """Find the closest point on an AABB to a given point.

    Args:
        point: Query point [x, y, z].
        bbox_min: AABB minimum corner [x, y, z].
        bbox_max: AABB maximum corner [x, y, z].

    Returns:
        Closest point on the AABB surface [x, y, z].
    """
    # Clamp each coordinate to the AABB bounds.
    return np.clip(point, bbox_min, bbox_max)


def compute_optimal_facing_yaw(
    origin_a: np.ndarray,
    target_point: np.ndarray,
) -> float:
    """Compute optimal yaw rotation to face a target point.

    Args:
        origin_a: Origin position of object A [x, y, z].
        target_point: Target point to face [x, y, z].

    Returns:
        Optimal absolute yaw rotation in degrees. Use directly with
        move_furniture_tool().
    """
    # Compute vector from A to target (only x-y plane matters for yaw).
    direction_to_target = target_point - origin_a
    dx = direction_to_target[0]
    dy = direction_to_target[1]

    # Check for zero vector (objects at same position).
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        console_logger.warning(
            "Objects at same position, cannot compute facing direction"
        )
        return 0.0

    # Compute desired world yaw to point toward target.
    desired_world_yaw = math.atan2(-dx, dy)

    # Convert to degrees and return absolute rotation.
    return math.degrees(desired_world_yaw)


def rigid_transform_to_matrix(transform: RigidTransform) -> np.ndarray:
    """Convert Drake RigidTransform to 4x4 homogeneous matrix for trimesh.

    Args:
        transform: Drake RigidTransform object.

    Returns:
        4x4 numpy array representing the transformation.
    """
    matrix = np.eye(4)
    matrix[:3, :3] = transform.rotation().matrix()  # 3x3 rotation matrix
    matrix[:3, 3] = transform.translation()  # 3x1 translation vector
    return matrix


def compute_aabb_corners(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    """Compute 8 corners of an axis-aligned bounding box.

    Args:
        bbox_min: Minimum corner [x, y, z].
        bbox_max: Maximum corner [x, y, z].

    Returns:
        Array of shape (8, 3) with corner positions in consistent ordering.
    """
    return np.array(
        [
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
        ]
    )


def safe_convex_hull_2d(
    vertices_2d: np.ndarray,
) -> tuple[ConvexHull | None, np.ndarray | None]:
    """Safely compute 2D ConvexHull with degenerate input handling.

    Prevents Qhull issues by:
    1. Ensuring C-contiguous array layout
    2. Removing duplicate vertices
    3. Checking for minimum vertex count

    Args:
        vertices_2d: Array of shape (N, 2) with 2D vertices.

    Returns:
        Tuple of (hull, processed_vertices):
        - hull: ConvexHull object if successful, None if input is degenerate.
        - processed_vertices: The preprocessed (contiguous, deduplicated) vertex
          array that hull.vertices indices refer to. None if hull is None.
    """
    # Ensure C-contiguous array (Qhull expects this).
    vertices = np.ascontiguousarray(vertices_2d)

    # Remove duplicate vertices.
    vertices = np.unique(vertices, axis=0)

    # Need at least 3 points for a 2D hull.
    if len(vertices) < 3:
        return None, None

    try:
        return ConvexHull(vertices), vertices
    except (QhullError, ValueError):
        return None, None


def compute_ordered_convex_hull_vertices_2d(
    vertices_3d: list[np.ndarray],
) -> np.ndarray | None:
    """Compute ordered 2D vertices from 3D convex hull vertices.

    Uses ConvexHull to determine proper vertex ordering for polygon creation.
    Unordered vertices can create self-intersecting polygons.

    Args:
        vertices_3d: List of 3D vertices [x, y, z].

    Returns:
        Array of shape (N, 2) with ordered 2D vertices [x, y] in sequential
        order around the convex hull perimeter. Returns None if hull
        computation fails (degenerate geometry).
    """
    # Project to 2D (use XY plane).
    hull_vertices_2d = np.array([(v[0], v[1]) for v in vertices_3d])

    # Compute convex hull safely to avoid Qhull segfaults.
    hull, processed_vertices = safe_convex_hull_2d(hull_vertices_2d)
    if hull is None:
        return None

    # Get vertices in sequential order around perimeter.
    return processed_vertices[hull.vertices]
