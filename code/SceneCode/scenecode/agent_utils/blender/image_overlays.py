"""Utilities for adding image overlays to rendered images."""

import logging
import math

from pathlib import Path

import bpy
import numpy as np

from mathutils import Matrix, Vector
from PIL import Image, ImageDraw

from scenecode.agent_utils.blender.annotations import (
    create_annotation_collection,
    load_annotation_font,
)

logger = logging.getLogger(__name__)

# Grid overlay constants.
GRID_SPACING_M = 0.1  # 10cm grid spacing.
GRID_LINE_THICKNESS_M = 0.005  # 5mm thick lines.
GRID_LINE_EMISSION_STRENGTH = 5.0
GRID_ITERATION_EPSILON = 0.001  # Small epsilon for floating point comparison.

# Bounding box visualization constants.
BBOX_WIREFRAME_THICKNESS_M = 0.05  # 5cm thick wireframe.


def add_number_overlay(image_path: Path, number: int) -> None:
    """Add a white square with black number to top-left of image.

    Args:
        image_path: Path to the image file to modify.
        number: The number to overlay.
    """
    # Open image.
    img = Image.open(image_path)

    # Create drawing context.
    draw = ImageDraw.Draw(img)

    # Define overlay size and position.
    square_size = 50
    margin = 10

    # Draw white square.
    draw.rectangle(
        [(margin, margin), (margin + square_size, margin + square_size)],
        fill="white",
    )

    # Draw number using cross-platform font loading.
    # Use fixed 32px font size by calculating appropriate divisor.
    font_size = 32
    image_width = img.size[0]
    base_font_size_divisor = image_width / font_size
    font = load_annotation_font(image_width, base_font_size_divisor)

    # Center text in square.
    text = str(number)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = margin + (square_size - text_width) // 2
    text_y = margin + (square_size - text_height) // 2

    draw.text((text_x, text_y), text, fill="black", font=font)

    # Save image.
    img.save(image_path)


def add_support_surface_grid(
    surface_min: list[float], surface_max: list[float]
) -> None:
    """Add coordinate grid spanning support surface for manipuland mode.

    Creates a grid of red lines on the support surface to help visualize
    placement coordinates. Grid spacing is 0.1m (10cm).

    Args:
        surface_min: Minimum bounds [x, y, z] in Z-up coordinates.
        surface_max: Maximum bounds [x, y, z] in Z-up coordinates.
    """
    grid_spacing = GRID_SPACING_M
    line_thickness = GRID_LINE_THICKNESS_M

    # Create material for grid lines.
    grid_mat = bpy.data.materials.new(name="MetricOverlay_SurfaceGrid")
    grid_mat.use_nodes = True
    bsdf = grid_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)  # Red.
        bsdf.inputs["Emission Color"].default_value = (1.0, 0.0, 0.0, 1.0)
        bsdf.inputs["Emission Strength"].default_value = GRID_LINE_EMISSION_STRENGTH
    grid_mat.use_backface_culling = False

    # Create horizontal grid lines on XY plane at surface top (max Z in Z-up).
    x_min, y_min, z_min = surface_min
    x_max, y_max, z_max = surface_max

    # Grid should be on horizontal plane (XY) at surface top height (max Z).
    grid_height = z_max

    # Lines parallel to Y axis (depth), varying X (width).
    x = x_min
    while x <= x_max + GRID_ITERATION_EPSILON:
        bpy.ops.mesh.primitive_cylinder_add(
            radius=line_thickness,
            depth=y_max - y_min,
            location=(x, (y_min + y_max) / 2, grid_height),
        )
        line = bpy.context.active_object
        line.name = f"MetricOverlay_GridLineY_{x:.2f}"
        line.rotation_euler = (0, math.pi / 2, 0)  # Rotate to align with Y.
        line.data.materials.append(grid_mat)
        x += grid_spacing

    # Lines parallel to X axis (width), varying Y (depth).
    y = y_min
    while y <= y_max + GRID_ITERATION_EPSILON:
        bpy.ops.mesh.primitive_cylinder_add(
            radius=line_thickness,
            depth=x_max - x_min,
            location=((x_min + x_max) / 2, y, grid_height),
        )
        line = bpy.context.active_object
        line.name = f"MetricOverlay_GridLineX_{y:.2f}"
        line.rotation_euler = (0, 0, math.pi / 2)  # Rotate to align with X.
        line.data.materials.append(grid_mat)
        y += grid_spacing


def add_support_surface_debug_volume(
    corners: np.ndarray, color: tuple[float, float, float, float] | None = None
) -> None:
    """Add semi-transparent volume mesh showing placeable region above surface.

    Creates a 3D volume visualization from 8 corner points with 50% transparency.
    This matches the volume mode from visualize_support_surfaces.py.

    Args:
        corners: Array of shape (8, 3) with corner points in Drake Z-up world coords.
        color: Optional RGBA color tuple (0-1 range). Defaults to green if not provided.
    """
    # Our corners array has min/max ordering from local space:
    # 0: (min,min,min), 1: (max,min,min), 2: (min,max,min), 3: (max,max,min)
    # 4: (min,min,max), 5: (max,min,max), 6: (min,max,max), 7: (max,max,max)

    # Compute center of the oriented box.
    center = corners.mean(axis=0)

    # Compute oriented extents from edge vectors.
    # Edge along X: from corner 0 to corner 1.
    edge_x = corners[1] - corners[0]
    # Edge along Y: from corner 0 to corner 2.
    edge_y = corners[2] - corners[0]
    # Edge along Z: from corner 0 to corner 4.
    edge_z = corners[4] - corners[0]

    # Compute lengths (extents).
    extent_x = np.linalg.norm(edge_x)
    extent_y = np.linalg.norm(edge_y)
    extent_z = np.linalg.norm(edge_z)

    # Compute rotation matrix from edge directions.
    # Normalize edges to get basis vectors.
    axis_x = edge_x / extent_x if extent_x > 0 else np.array([1, 0, 0])
    axis_y = edge_y / extent_y if extent_y > 0 else np.array([0, 1, 0])
    axis_z = edge_z / extent_z if extent_z > 0 else np.array([0, 0, 1])

    # Note: Bbox axes (_bbox_axis_x/y/z) are now pre-computed during
    # initialization in render_agent_observation_views() and don't need
    # to be recomputed here. This ensures they're always available
    # regardless of whether the bbox is drawn.

    # Create rotation matrix from basis vectors (column-major for Blender).
    # Each column is a basis vector defining the orientation.
    rot_matrix = Matrix(
        (
            (axis_x[0], axis_y[0], axis_z[0]),
            (axis_x[1], axis_y[1], axis_z[1]),
            (axis_x[2], axis_y[2], axis_z[2]),
        )
    )

    # Create cube primitive at center (solid volume, not wireframe).
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=Vector(center.tolist()))
    bbox_obj = bpy.context.active_object
    bbox_obj.name = "MetricOverlay_SupportSurfaceBounds"

    # Apply rotation.
    bbox_obj.rotation_mode = "QUATERNION"
    bbox_obj.rotation_quaternion = rot_matrix.to_quaternion()

    # Scale to match extents.
    bbox_obj.scale = Vector((extent_x, extent_y, extent_z))

    # Create opaque material (matching overlay surfaces).
    mat = bpy.data.materials.new(name="mat_support_surface_debug")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Use provided color or default to green.
    if color is None:
        color = (0.0, 1.0, 0.0, 1.0)

    # Add Principled BSDF shader.
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Alpha"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.5

    # Add Material Output.
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Use opaque blend method to match overlay surfaces.
    mat.blend_method = "OPAQUE"

    bbox_obj.data.materials.append(mat)

    # Ensure bbox is not hidden from render.
    bbox_obj.hide_render = False

    # Link to annotations collection.
    collection = create_annotation_collection()
    if bbox_obj.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(bbox_obj)
    collection.objects.link(bbox_obj)

    logger.debug(
        f"Created opaque support surface volume at {center} "
        f"with extents ({extent_x}, {extent_y}, {extent_z})"
    )
