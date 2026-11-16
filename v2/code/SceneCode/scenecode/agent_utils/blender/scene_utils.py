"""Utilities for scene analysis and bounds computation in Blender."""

import logging

import bpy

from mathutils import Vector

logger = logging.getLogger(__name__)

# Floor level detection tolerance in meters.
FLOOR_LEVEL_TOLERANCE_M = 0.01


def get_floor_bounds(client_objects: bpy.types.Collection | None) -> list[float]:
    """Get floor bounds from the lowest geometry.

    Args:
        client_objects: Collection containing scene objects.

    Returns:
        list: Floor bounds as [min_x, min_y, floor_z, max_x, max_y].

    Raises:
        ValueError: If no client objects or mesh objects are found.
    """
    if not client_objects:
        raise ValueError("No client objects available for floor bounds computation")

    mesh_objs = [obj for obj in client_objects.objects if obj.type == "MESH"]
    if not mesh_objs:
        raise ValueError("No mesh objects available for floor bounds computation")

    # Find the lowest geometry (floor level).
    floor_z = float("inf")
    floor_objects = []
    for obj in mesh_objs:
        obj_min_z = float("inf")
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            obj_min_z = min(obj_min_z, world_corner.z)

        if obj_min_z < floor_z:
            floor_z = obj_min_z
            floor_objects = [obj]  # Start new list
        elif abs(obj_min_z - floor_z) < FLOOR_LEVEL_TOLERANCE_M:
            floor_objects.append(obj)

    # Get bounds of all floor-level objects.
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    for obj in floor_objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_x = min(min_x, world_corner.x)
            max_x = max(max_x, world_corner.x)
            min_y = min(min_y, world_corner.y)
            max_y = max(max_y, world_corner.y)

    return [min_x, min_y, floor_z, max_x, max_y]


def compute_scene_bounds(client_objects: bpy.types.Collection) -> tuple[Vector, float]:
    """Compute axis-aligned 3D square bounding box of scene objects.

    Args:
        client_objects: Collection containing scene mesh objects.

    Returns:
        tuple: A tuple of (bbox_center, max_dim) where:
            - bbox_center (Vector): Center point of the scene bounding box.
            - max_dim (float): Maximum dimension of the bounding box in meters.

    Raises:
        ValueError: If no mesh objects are found in the scene.
    """
    mesh_objs = [obj for obj in client_objects.objects if obj.type == "MESH"]
    if not mesh_objs:
        raise ValueError("No mesh objects found in scene")

    bbox_min = Vector((float("inf"),) * 3)
    bbox_max = Vector((float("-inf"),) * 3)
    for obj in mesh_objs:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            bbox_min = Vector(map(min, bbox_min, world_corner))
            bbox_max = Vector(map(max, bbox_max, world_corner))

    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min
    max_dim = max(bbox_size)
    return bbox_center, max_dim


def disable_backface_culling(objects: list[bpy.types.Object]) -> None:
    """Disable backface culling for all materials on the given objects.

    This ensures meshes render correctly from both sides, fixing issues
    with single-sided meshes (common in PartNet-Mobility models).

    Args:
        objects: List of Blender objects to process.
    """
    for obj in objects:
        if obj.type == "MESH" and obj.data:
            for mat in obj.data.materials:
                if mat is not None:
                    mat.use_backface_culling = False
