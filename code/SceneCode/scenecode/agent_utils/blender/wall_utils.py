"""Utilities for wall detection and visibility management in Blender scenes."""

import logging

import bpy
import numpy as np

from mathutils import Vector

console_logger = logging.getLogger(__name__)


def looks_like_wall(obj: bpy.types.Object) -> bool:
    """Detect if a Blender object looks like a wall using geometric heuristics.

    Temporary workaround until Drake exports geometry names in glTF (Issue #100).
    Walls are tall (vertical extent), thin in one horizontal dimension, and have
    high aspect ratio. Uses world-space bounding box to handle parent transforms.

    Args:
        obj: Blender object to check.

    Returns:
        True if object appears to be a wall based on dimensions and orientation.
    """
    if obj.type != "MESH":
        return False

    # Exclude metric overlay objects (coordinate frames, annotations, etc.).
    if "MetricOverlay" in obj.name or "Annotation" in obj.name:
        return False

    # Get world-space bounding box by transforming all 8 corners.
    bbox_local = np.array(obj.bound_box)
    bbox_world = np.array([obj.matrix_world @ Vector(corner) for corner in bbox_local])

    # Find world-space dimensions.
    min_coords = bbox_world.min(axis=0)
    max_coords = bbox_world.max(axis=0)
    world_dimensions = max_coords - min_coords

    width_x = world_dimensions[0]
    depth_y = world_dimensions[1]
    height_z = world_dimensions[2]

    # Walls should be tall (significant vertical extent).
    MIN_WALL_HEIGHT = 1.5  # At least 1.5m tall.
    if height_z < MIN_WALL_HEIGHT:
        return False

    # Walls should be thin in one horizontal dimension.
    MIN_THICKNESS = 0.2  # Max 20cm thick.
    horizontal_dims = [width_x, depth_y]
    min_horizontal = min(horizontal_dims)

    if min_horizontal > MIN_THICKNESS:
        return False

    # Check aspect ratio (height vs thin dimension).
    aspect_ratio = height_z / min_horizontal
    MIN_ASPECT_RATIO = 5  # Height should be at least 5x the thickness.

    if aspect_ratio < MIN_ASPECT_RATIO:
        return False

    return True


def get_wall_normal_from_metadata(
    obj: bpy.types.Object, wall_normals: dict[str, list[float]] | None
) -> Vector | None:
    """Get pre-computed room-facing normal for a wall object.

    Matches Blender wall object to pre-computed wall normal by position.
    Since Drake glTF export loses object names, we match by comparing
    wall center position in world coordinates.

    Args:
        obj: Blender wall object.
        wall_normals: Dictionary mapping wall names to normal vectors (2D lists).

    Returns:
        Room-facing normal vector (2D, in XY plane), or None if not found.
    """
    if not wall_normals:
        return None

    # Get bounding box center in world space.
    # For GLTF meshes, object origin is at (0,0,0) so we use bbox center.
    bbox_local = np.array(obj.bound_box)
    bbox_world = np.array([obj.matrix_world @ Vector(corner) for corner in bbox_local])
    center = (bbox_world.min(axis=0) + bbox_world.max(axis=0)) / 2
    wall_x = center[0]
    wall_y = center[1]

    # For rectangular rooms centered at origin, match walls by position.
    # Walls are positioned at room boundaries (e.g., x = ±half_length).
    for wall_name, normal_list in wall_normals.items():
        # Match wall by position with generous threshold.
        # Support both old names (left/right/back/front) and new names (west/east/south/north).
        matched = False
        if ("left" in wall_name or "west" in wall_name) and wall_x < -0.5:
            matched = True
        elif ("right" in wall_name or "east" in wall_name) and wall_x > 0.5:
            matched = True
        elif ("back" in wall_name or "south" in wall_name) and wall_y < -0.5:
            matched = True
        elif ("front" in wall_name or "north" in wall_name) and wall_y > 0.5:
            matched = True

        if matched:
            return Vector((normal_list[0], normal_list[1], 0.0))

    return None


def should_hide_wall(
    obj: bpy.types.Object,
    camera_direction: Vector,
    is_top_view: bool,
    wall_normals: dict[str, list[float]] | None,
) -> bool:
    """Determine if wall should be hidden based on camera viewpoint.

    Uses pre-computed room-facing normals to determine if wall blocks the
    camera's view into the room (hide for dollhouse view) or is on the far
    side (show).

    Args:
        obj: Wall object to check.
        camera_direction: Direction camera is pointing (normalized).
        is_top_view: True if this is a top-down view.
        wall_normals: Dictionary mapping wall names to normal vectors.

    Returns:
        True if wall should be hidden.
    """
    # Top view: show all walls (doors/windows visible as openings in wall geometry).
    if is_top_view:
        return False

    # Side view: hide walls that block camera view into room.
    # Get pre-computed room-facing normal.
    wall_normal = get_wall_normal_from_metadata(obj=obj, wall_normals=wall_normals)
    if wall_normal is None:
        return False

    # Check if wall is between camera and room center.
    # Wall normal points from wall toward room center (inward).
    # Camera direction points from camera toward scene center.
    # If they point in SAME direction (positive dot): both point toward center
    # from same side → wall is BETWEEN camera and room → hide for dollhouse view.
    # If they point in OPPOSITE directions (negative dot): wall is on FAR side
    # from camera → show it.
    dot_product = camera_direction.dot(wall_normal)

    # Hide if wall is between camera and room (same direction, positive dot).
    # Use threshold to handle numerical precision.
    return dot_product > 0.1


def restore_hidden_walls() -> None:
    """Restore all hidden walls to visible state.

    This should be called before rendering each new view to ensure walls
    hidden in the previous view are made visible again.
    """
    all_meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    wall_objects = [obj for obj in all_meshes if looks_like_wall(obj)]

    for obj in wall_objects:
        obj.hide_render = False
        obj.hide_viewport = False

    bpy.context.view_layer.update()
