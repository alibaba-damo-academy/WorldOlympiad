"""Utilities for creating and managing coordinate frames in Blender scenes."""

import logging

import bpy
import numpy as np

from mathutils import Vector

logger = logging.getLogger(__name__)


def create_coordinate_frame(
    position: Vector,
    max_dim: float,
    scale_factor: float = 0.01,
    add_labels: bool = True,
    name_prefix: str = "",
    shaft_len_override: float | None = None,
    tip_len_override: float | None = None,
    tip_radius_override: float | None = None,
    shaft_radius_override: float | None = None,
    axis_x: Vector | None = None,
    axis_y: Vector | None = None,
    axis_z: Vector | None = None,
) -> None:
    """Create RGB coordinate frame with arrow axes.

    Creates three arrows (+X=red, +Y=green, +Z=blue) representing a
    coordinate frame.

    Args:
        position: Center position for the coordinate frame.
        max_dim: Maximum dimension for scaling arrow sizes.
        scale_factor: Scale factor for base_scale calculation (default: 0.01).
        add_labels: Whether to add text labels (+X, +Y, +Z) (default: True).
        name_prefix: Prefix for Blender object names (default: "").
        shaft_len_override: Optional explicit shaft length (overrides calculation).
        tip_len_override: Optional explicit tip length (overrides calculation).
        tip_radius_override: Optional explicit tip radius (overrides calculation).
        shaft_radius_override: Optional explicit shaft radius (overrides calculation).
        axis_x: Optional custom X axis direction (default: (1, 0, 0)).
        axis_y: Optional custom Y axis direction (default: (0, 1, 0)).
        axis_z: Optional custom Z axis direction (default: (0, 0, 1)).
    """
    base_scale = max_dim * scale_factor
    shaft_len = shaft_len_override if shaft_len_override is not None else max_dim * 0.6
    tip_len = tip_len_override if tip_len_override is not None else max_dim * 0.15
    tip_radius = (
        tip_radius_override if tip_radius_override is not None else base_scale * 2.5
    )
    shaft_radius = (
        shaft_radius_override if shaft_radius_override is not None else base_scale * 1.0
    )

    def make_material(
        name: str, rgba: tuple[float, float, float, float]
    ) -> bpy.types.Material:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = rgba
        # Disable backface culling to ensure visibility from all camera angles.
        mat.use_backface_culling = False
        return mat

    def create_arrow_axis(
        name: str, direction: Vector, color: tuple, label_text: str | None = None
    ) -> None:
        mat = make_material(f"{name_prefix}{name}_mat", color)

        # Shaft.
        bpy.ops.mesh.primitive_cylinder_add(
            radius=shaft_radius,
            depth=shaft_len,
            location=position,
        )
        shaft = bpy.context.active_object
        shaft.name = f"{name_prefix}{name}_shaft"
        shaft.data.materials.append(mat)
        shaft.location = position + (direction.normalized() * shaft_len / 2)
        shaft.rotation_mode = "QUATERNION"
        shaft.rotation_quaternion = direction.to_track_quat("Z", "Y")

        # Tip.
        bpy.ops.mesh.primitive_cone_add(radius1=tip_radius, depth=tip_len)
        tip = bpy.context.active_object
        tip.name = f"{name_prefix}{name}_tip"
        tip.data.materials.append(mat)
        tip.location = position + (direction.normalized() * (shaft_len + tip_len / 2))
        tip.rotation_mode = "QUATERNION"
        tip.rotation_quaternion = direction.to_track_quat("Z", "Y")

        # Optional label.
        if label_text and add_labels:
            bpy.ops.object.text_add(
                location=tip.location + direction.normalized() * (tip_len * 0.8)
            )
            text = bpy.context.active_object
            text.data.body = label_text
            text.scale = (base_scale * 5,) * 3
            text.data.extrude = base_scale * 0.5
            text.data.align_x = "CENTER"
            text.data.align_y = "CENTER"
            text.rotation_mode = "QUATERNION"
            text.rotation_quaternion = direction.to_track_quat("Z", "Y")
            text_mat = make_material(f"{name_prefix}{name}_label", (1, 1, 1, 1))
            text.data.materials.append(text_mat)

    # Create the three coordinate axes.
    # Use provided axes if available, otherwise default to world axes.
    x_dir = axis_x if axis_x is not None else Vector((1, 0, 0))
    y_dir = axis_y if axis_y is not None else Vector((0, 1, 0))
    z_dir = axis_z if axis_z is not None else Vector((0, 0, 1))

    create_arrow_axis("X", x_dir, (1, 0, 0, 1), "+X")
    create_arrow_axis("Y", y_dir, (0, 1, 0, 1), "+Y")
    create_arrow_axis("Z", z_dir, (0, 0, 1, 1), "+Z")


def _add_coordinate_frame_for_view(
    bbox_center: Vector,
    max_dim: float,
    floor_bounds: list[float],
    frame_scale_factor: float,
    rendering_mode: str,
    apply_inset: bool,
    frame_origin: np.ndarray | None = None,
    bbox_axis_x: Vector | None = None,
    bbox_axis_y: Vector | None = None,
    bbox_axis_z: Vector | None = None,
) -> None:
    """Helper to add coordinate frame for different view types.

    Args:
        bbox_center: Center point of scene bounding box.
        max_dim: Maximum dimension of scene bounding box.
        floor_bounds: Floor bounds [min_x, min_y, floor_z, max_x, max_y].
        frame_scale_factor: Scale factor for the frame.
        rendering_mode: Rendering mode - "furniture" or "manipuland".
        apply_inset: Whether to apply wall inset (side views only).
        frame_origin: Optional frame origin for manipuland mode.
        bbox_axis_x: Optional custom X axis for manipuland mode.
        bbox_axis_y: Optional custom Y axis for manipuland mode.
        bbox_axis_z: Optional custom Z axis for manipuland mode.
    """
    # Determine frame position based on rendering mode.
    if rendering_mode == "manipuland" and frame_origin is not None:
        # Manipuland mode: position at bottom-left corner in camera-aligned view.
        # This corner is computed to be at the bottom-left after camera rotation.
        frame_position = Vector(frame_origin.tolist())
    else:
        # Furniture mode: position at floor corner.
        min_x, min_y, floor_z = floor_bounds[0], floor_bounds[1], floor_bounds[2]

        if apply_inset:
            # Offset inward from corner to avoid being hidden by walls.
            # Wall thickness is 0.1m, positioned with center at boundary,
            # so each wall extends 0.05m (half thickness) inward from corner.
            # Add 0.02m safety margin.
            inset = 0.07
            frame_position = Vector(
                (
                    min_x + inset,
                    min_y + inset,
                    floor_z + max(0.15, max_dim * 0.02),  # Min 15cm clearance.
                )
            )
        else:
            frame_position = Vector((min_x, min_y, floor_z + max(0.15, max_dim * 0.02)))

    # Calculate camera distance.
    try:
        camera_location = Vector(bpy.context.scene.camera.location)
        camera_distance = (camera_location - bbox_center).length
    except (ValueError, AttributeError):
        # Fallback for test environments where camera location might be None/empty.
        logger.warning("No camera location found, using fallback camera distance.")
        camera_distance = max_dim * 2

    # Calculate frame dimensions based on camera distance.
    # This ensures consistent screen-space appearance regardless of scene size.
    frame_scale = camera_distance * frame_scale_factor
    shaft_len = frame_scale * 0.6
    tip_len = frame_scale * 0.15
    tip_radius = frame_scale * 0.04
    shaft_radius = frame_scale * 0.015

    # Create coordinate frame without labels.
    # Use same axes as green bounding box for perfect alignment.
    if rendering_mode == "manipuland" and bbox_axis_x is not None:
        create_coordinate_frame(
            position=frame_position,
            max_dim=max_dim,
            scale_factor=0.01,  # Not used when overrides are provided.
            add_labels=False,
            name_prefix="MetricOverlay_",
            shaft_len_override=shaft_len,
            tip_len_override=tip_len,
            tip_radius_override=tip_radius,
            shaft_radius_override=shaft_radius,
            axis_x=bbox_axis_x,
            axis_y=bbox_axis_y,
            axis_z=bbox_axis_z,
        )
    else:
        # Furniture mode: use default world axes.
        create_coordinate_frame(
            position=frame_position,
            max_dim=max_dim,
            scale_factor=0.01,  # Not used when overrides are provided.
            add_labels=False,
            name_prefix="MetricOverlay_",
            shaft_len_override=shaft_len,
            tip_len_override=tip_len,
            tip_radius_override=tip_radius,
            shaft_radius_override=shaft_radius,
        )


def add_coordinate_frame_side_view(
    bbox_center: Vector,
    max_dim: float,
    floor_bounds: list[float],
    frame_scale_factor: float = 0.15,
    rendering_mode: str = "furniture",
    frame_origin: np.ndarray | None = None,
    bbox_axis_x: Vector | None = None,
    bbox_axis_y: Vector | None = None,
    bbox_axis_z: Vector | None = None,
) -> None:
    """Add coordinate frame markers for side views.

    Args:
        bbox_center: Center point of scene bounding box.
        max_dim: Maximum dimension of scene bounding box.
        floor_bounds: Floor bounds [min_x, min_y, floor_z, max_x, max_y].
        frame_scale_factor: Scale factor for the frame.
        rendering_mode: Rendering mode - "furniture" or "manipuland".
        frame_origin: Optional frame origin for manipuland mode.
        bbox_axis_x: Optional custom X axis for manipuland mode.
        bbox_axis_y: Optional custom Y axis for manipuland mode.
        bbox_axis_z: Optional custom Z axis for manipuland mode.
    """
    _add_coordinate_frame_for_view(
        bbox_center=bbox_center,
        max_dim=max_dim,
        floor_bounds=floor_bounds,
        frame_scale_factor=frame_scale_factor,
        rendering_mode=rendering_mode,
        apply_inset=True,  # Side views apply wall inset.
        frame_origin=frame_origin,
        bbox_axis_x=bbox_axis_x,
        bbox_axis_y=bbox_axis_y,
        bbox_axis_z=bbox_axis_z,
    )


def add_coordinate_frame_top_view(
    bbox_center: Vector,
    max_dim: float,
    floor_bounds: list[float],
    rendering_mode: str = "furniture",
    frame_origin: np.ndarray | None = None,
    bbox_axis_x: Vector | None = None,
    bbox_axis_y: Vector | None = None,
    bbox_axis_z: Vector | None = None,
) -> None:
    """Add coordinate frame markers for top-down views.

    Args:
        bbox_center: Center point of scene bounding box.
        max_dim: Maximum dimension of scene bounding box.
        floor_bounds: Floor bounds [min_x, min_y, floor_z, max_x, max_y].
        rendering_mode: Rendering mode - "furniture" or "manipuland".
        frame_origin: Optional frame origin for manipuland mode.
        bbox_axis_x: Optional custom X axis for manipuland mode.
        bbox_axis_y: Optional custom Y axis for manipuland mode.
        bbox_axis_z: Optional custom Z axis for manipuland mode.
    """
    _add_coordinate_frame_for_view(
        bbox_center=bbox_center,
        max_dim=max_dim,
        floor_bounds=floor_bounds,
        frame_scale_factor=0.15,  # Fixed scale factor for top views.
        rendering_mode=rendering_mode,
        apply_inset=True,  # Apply wall inset so frame is visible when walls are shown.
        frame_origin=frame_origin,
        bbox_axis_x=bbox_axis_x,
        bbox_axis_y=bbox_axis_y,
        bbox_axis_z=bbox_axis_z,
    )


def add_coordinate_frame_wall_view(
    wall_center: np.ndarray,
    wall_length: float,
    wall_height: float,
    wall_direction: str,
) -> None:
    """Add coordinate frame at (0,0) corner for wall orthographic view.

    Creates a 2D coordinate frame (red X, blue Z) at the bottom-left corner
    of the wall as viewed from inside the room. The frame shows wall-local
    coordinates where X is along the wall and Z is height.

    Args:
        wall_center: Wall center position in world coordinates [x, y, z].
        wall_length: Wall length in meters.
        wall_height: Wall height in meters.
        wall_direction: Wall direction ("north", "south", "east", "west").
    """
    wall_dir = wall_direction.lower()

    # Calculate (0,0) position in world coordinates.
    # Wall coordinates: x=0 is left edge (in camera view), z=0 is floor.
    # Offset from center: x_offset = -length/2, z_offset = -height/2.
    x_offset = -wall_length / 2
    z_offset = -wall_height / 2

    if wall_dir == "north":
        # North wall at +Y, viewed from -Y. Left is -X world (wall X=0 at world -X).
        frame_pos = np.array(
            [
                wall_center[0] + x_offset,  # = wall_center[0] - length/2 (-X is left).
                wall_center[1],
                wall_center[2] + z_offset,  # = wall_center[2] - height/2 (floor).
            ]
        )
        # Wall +X direction = toward right in view = world +X.
        axis_x = Vector((1, 0, 0))
    elif wall_dir == "south":
        # South wall at -Y, viewed from +Y. Left is +X world (wall X=0 at world +X).
        frame_pos = np.array(
            [
                wall_center[0] - x_offset,  # = wall_center[0] + length/2 (+X is left).
                wall_center[1],
                wall_center[2] + z_offset,
            ]
        )
        # Wall +X direction = toward right in view = world -X.
        axis_x = Vector((-1, 0, 0))
    elif wall_dir == "east":
        # East wall at +X, viewed from -X. Left is +Y world.
        frame_pos = np.array(
            [
                wall_center[0],
                wall_center[1] - x_offset,  # = wall_center[1] + length/2 (+Y is left).
                wall_center[2] + z_offset,
            ]
        )
        # Wall +X direction = toward right in view = world -Y.
        axis_x = Vector((0, -1, 0))
    elif wall_dir == "west":
        # West wall at -X, viewed from +X. Left is -Y world.
        frame_pos = np.array(
            [
                wall_center[0],
                wall_center[1] + x_offset,  # = wall_center[1] - length/2 (-Y is left).
                wall_center[2] + z_offset,
            ]
        )
        # Wall +X direction = toward right in view = world +Y.
        axis_x = Vector((0, 1, 0))
    else:
        logger.warning(f"Unknown wall direction: {wall_direction}")
        return

    # Z axis is always world +Z (up).
    axis_z = Vector((0, 0, 1))

    # Calculate frame dimensions based on wall size.
    # Scale proportionally to wall dimensions (similar to top-view frame).
    max_dim = max(wall_length, wall_height)
    shaft_len = max_dim * 0.08  # 8% of max dimension.
    tip_len = shaft_len * 0.25
    tip_radius = shaft_len * 0.08
    shaft_radius = shaft_len * 0.04

    # Offset frame into the room so it's clearly visible in front of wall.
    # Move toward camera (into room) based on wall direction.
    # North wall at +Y: camera at -Y, so move frame toward -Y (decrease Y).
    # South wall at -Y: camera at +Y, so move frame toward +Y (increase Y).
    # East wall at +X: camera at -X, so move frame toward -X (decrease X).
    # West wall at -X: camera at +X, so move frame toward +X (increase X).
    offset_dist = 0.3
    if wall_dir == "north":
        frame_pos[1] -= offset_dist  # Move toward camera (-Y direction).
    elif wall_dir == "south":
        frame_pos[1] += offset_dist  # Move toward camera (+Y direction).
    elif wall_dir == "east":
        frame_pos[0] -= offset_dist  # Move toward camera (-X direction).
    elif wall_dir == "west":
        frame_pos[0] += offset_dist  # Move toward camera (+X direction).

    # Small Z offset just to avoid z-fighting with floor (not to reposition frame).
    frame_pos[2] += 0.01

    frame_position = Vector(frame_pos.tolist())

    logger.info(
        f"Adding wall coordinate frame for {wall_dir} wall:\n"
        f"  wall_center={wall_center}\n"
        f"  frame_position={frame_position}\n"
        f"  axis_x={axis_x}, axis_z={axis_z}\n"
        f"  shaft_len={shaft_len:.3f}, tip_len={tip_len:.3f}"
    )

    # Use the main create_coordinate_frame with custom axes.
    # Set axis_y to a very small vector to effectively hide it.
    try:
        create_coordinate_frame(
            position=frame_position,
            max_dim=max(wall_length, wall_height),
            scale_factor=0.02,
            add_labels=False,
            name_prefix="WallFrame_",
            shaft_len_override=shaft_len,
            tip_len_override=tip_len,
            tip_radius_override=tip_radius,
            shaft_radius_override=shaft_radius,
            axis_x=axis_x,
            axis_y=Vector((0, 0, 0.001)),  # Tiny Y axis (effectively hidden).
            axis_z=axis_z,
        )
        logger.info(f"Successfully created coordinate frame for {wall_dir} wall")
    except Exception as e:
        logger.error(f"Failed to create coordinate frame for {wall_dir} wall: {e}")


def remove_wall_coordinate_frame() -> None:
    """Remove wall coordinate frame objects from the scene."""
    objects_to_remove: list[bpy.types.Object] = []
    for obj in bpy.data.objects:
        # Match both old custom arrows and new create_coordinate_frame objects.
        if "WallFrame_" in obj.name:
            objects_to_remove.append(obj)

    for obj in objects_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)


def remove_coordinate_frame() -> None:
    """Remove all metric overlay objects from the scene.

    Finds and removes all objects with 'MetricOverlay' in their name.
    """
    # Find and remove all objects with "MetricOverlay" in their name.
    objects_to_remove: list[bpy.types.Object] = []
    for obj in bpy.data.objects:
        if "MetricOverlay" in obj.name:
            objects_to_remove.append(obj)

    # Remove objects.
    for obj in objects_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
