"""Shared physics utilities for manipuland simulation."""

import logging

import trimesh

from scenecode.agent_utils.room import SceneObject
from scenecode.utils.mesh_loading import load_collision_meshes_from_sdf

console_logger = logging.getLogger(__name__)


def compute_collision_bounds(
    collision_meshes: list[trimesh.Trimesh],
) -> tuple[float, float]:
    """Get z_min and z_max from collision geometry.

    Args:
        collision_meshes: List of collision mesh pieces (convex hulls).

    Returns:
        Tuple of (z_min, z_max) representing collision bounds along Z-axis.

    Raises:
        ValueError: If collision_meshes is empty.
    """
    if not collision_meshes:
        raise ValueError("No collision meshes provided")

    z_min = min(m.vertices[:, 2].min() for m in collision_meshes)
    z_max = max(m.vertices[:, 2].max() for m in collision_meshes)
    return float(z_min), float(z_max)


def load_collision_bounds_for_scene_object(
    obj: SceneObject,
) -> tuple[float, float]:
    """Load collision geometry and compute z-bounds for a SceneObject.

    Applies both SDF scale (from mesh loading) and object's scale_factor
    (from runtime rescale operations).

    Args:
        obj: SceneObject with sdf_path.

    Returns:
        Tuple of (z_min, z_max) from collision geometry.

    Raises:
        ValueError: If object has no SDF path or no collision geometry.
    """
    if not obj.sdf_path:
        raise ValueError(f"Object {obj.name} has no SDF path")

    collision_meshes = load_collision_meshes_from_sdf(obj.sdf_path)
    if not collision_meshes:
        raise ValueError(f"No collision geometry found for {obj.name}")

    z_min, z_max = compute_collision_bounds(collision_meshes)

    # Apply object's scale_factor (set by rescale operations).
    z_min *= obj.scale_factor
    z_max *= obj.scale_factor

    return z_min, z_max
