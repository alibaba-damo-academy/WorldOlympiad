"""Thin covering generator for procedural flat textured surfaces.

This module provides utilities for generating thin textured meshes
with PBR materials. Supports floor coverings (rugs, mats), wall coverings
(posters, prints), and surface coverings (tablecloths, placemats).
These objects are purely decorative (no collision geometry) and are
static (welded to world).
"""

import logging
import math
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from scenecode.utils.gltf_generation import (
    create_glb_from_mesh_data,
    zup_to_yup_transform,
)
from scenecode.utils.material import Material

console_logger = logging.getLogger(__name__)


def infer_thin_covering_shape(description: str) -> str:
    """Infer thin covering shape from description using keyword matching.

    Args:
        description: Thin covering description text.

    Returns:
        "circular" if description contains round/circular keywords,
        "rectangular" otherwise.
    """
    desc_lower = description.lower()
    if "round" in desc_lower or "circular" in desc_lower:
        return "circular"
    return "rectangular"


def create_rectangular_thin_covering_glb(
    width: float,
    second_dim: float,
    thickness: float,
    material_folder: Path,
    output_path: Path,
    texture_scale: float | None,
    is_wall: bool = False,
) -> Path:
    """Create a thin rectangular covering mesh with PBR material as GLB.

    For floor coverings (is_wall=False):
        Centered in XY at origin with bottom at z=0. Width is X, second_dim is Y (depth).
        Primary texture face is +Z (top, visible from above).

    For wall coverings (is_wall=True):
        Centered in X, with z from 0 to second_dim (height). Back at y=0.
        Primary texture face is +Y (front, visible from inside room).

    Args:
        width: X dimension in meters.
        second_dim: Y dimension (depth) for floor, Z dimension (height) for wall.
        thickness: Thin dimension in meters (Z for floor, Y for wall).
        material_folder: Path to folder containing PBR textures.
        output_path: Where to save the GLB file. Must have .glb extension.
        texture_scale: Meters per texture tile. None = cover mode (no tiling).
        is_wall: If True, create vertical wall covering; if False, floor covering.

    Returns:
        Path to the created GLB file.
    """
    textures = Material.from_path(material_folder).get_all_textures()

    half_w = width / 2.0

    if is_wall:
        # Wall covering: XZ plane, back at y=0, front at y=thickness.
        # Centered in X, z from 0 to height (second_dim).
        height = second_dim

        vertices_zup = np.array(
            [
                # Front face (+Y, faces into room): vertices 0-3 - PRIMARY TEXTURE FACE
                [-half_w, thickness, 0],
                [+half_w, thickness, 0],
                [+half_w, thickness, height],
                [-half_w, thickness, height],
                # Back face (-Y, against wall): vertices 4-7
                [+half_w, 0, 0],
                [-half_w, 0, 0],
                [-half_w, 0, height],
                [+half_w, 0, height],
                # Right face (+X): vertices 8-11
                [+half_w, thickness, 0],
                [+half_w, 0, 0],
                [+half_w, 0, height],
                [+half_w, thickness, height],
                # Left face (-X): vertices 12-15
                [-half_w, 0, 0],
                [-half_w, thickness, 0],
                [-half_w, thickness, height],
                [-half_w, 0, height],
                # Top face (+Z): vertices 16-19
                [-half_w, thickness, height],
                [+half_w, thickness, height],
                [+half_w, 0, height],
                [-half_w, 0, height],
                # Bottom face (-Z): vertices 20-23
                [-half_w, 0, 0],
                [+half_w, 0, 0],
                [+half_w, thickness, 0],
                [-half_w, thickness, 0],
            ],
            dtype=np.float32,
        )

        # UV coordinates for wall covering.
        if texture_scale is None:
            # Cover mode: texture spans entire surface (no tiling).
            uv_front = (1.0, 1.0)
            uv_side_lr = (1.0, 1.0)
            uv_top_bottom = (1.0, 1.0)
        else:
            uv_front = (width / texture_scale, height / texture_scale)
            uv_side_lr = (thickness / texture_scale, height / texture_scale)
            uv_top_bottom = (width / texture_scale, thickness / texture_scale)

        uvs = np.array(
            [
                # Front face (main artwork) - full texture.
                # Flip both U and V for correct orientation after coordinate transform.
                [uv_front[0], uv_front[1]],
                [0, uv_front[1]],
                [0, 0],
                [uv_front[0], 0],
                # Back face (against wall) - same UV flip.
                [uv_front[0], uv_front[1]],
                [0, uv_front[1]],
                [0, 0],
                [uv_front[0], 0],
                # Left/right faces (edges)
                [0, 0],
                [uv_side_lr[0], 0],
                [uv_side_lr[0], uv_side_lr[1]],
                [0, uv_side_lr[1]],
                [0, 0],
                [uv_side_lr[0], 0],
                [uv_side_lr[0], uv_side_lr[1]],
                [0, uv_side_lr[1]],
                # Top/bottom faces (edges)
                [0, 0],
                [uv_top_bottom[0], 0],
                [uv_top_bottom[0], uv_top_bottom[1]],
                [0, uv_top_bottom[1]],
                [0, 0],
                [uv_top_bottom[0], 0],
                [uv_top_bottom[0], uv_top_bottom[1]],
                [0, uv_top_bottom[1]],
            ],
            dtype=np.float32,
        )

        log_msg = (
            f"Created wall covering GLB: {output_path.name} "
            f"({width:.2f}m x {height:.2f}m x {thickness:.3f}m)"
        )

    else:
        # Floor covering: XY plane, bottom at z=0, top at z=thickness.
        # Centered in XY.
        depth = second_dim
        half_d = depth / 2.0

        vertices_zup = np.array(
            [
                # Front face (+Y): vertices 0-3
                [+half_w, +half_d, 0],
                [-half_w, +half_d, 0],
                [-half_w, +half_d, thickness],
                [+half_w, +half_d, thickness],
                # Back face (-Y): vertices 4-7
                [-half_w, -half_d, 0],
                [+half_w, -half_d, 0],
                [+half_w, -half_d, thickness],
                [-half_w, -half_d, thickness],
                # Right face (+X): vertices 8-11
                [+half_w, -half_d, 0],
                [+half_w, +half_d, 0],
                [+half_w, +half_d, thickness],
                [+half_w, -half_d, thickness],
                # Left face (-X): vertices 12-15
                [-half_w, +half_d, 0],
                [-half_w, -half_d, 0],
                [-half_w, -half_d, thickness],
                [-half_w, +half_d, thickness],
                # Top face (+Z): vertices 16-19 - PRIMARY TEXTURE FACE
                [-half_w, -half_d, thickness],
                [+half_w, -half_d, thickness],
                [+half_w, +half_d, thickness],
                [-half_w, +half_d, thickness],
                # Bottom face (-Z): vertices 20-23
                [-half_w, +half_d, 0],
                [+half_w, +half_d, 0],
                [+half_w, -half_d, 0],
                [-half_w, -half_d, 0],
            ],
            dtype=np.float32,
        )

        # UV coordinates for floor covering.
        if texture_scale is None:
            # Cover mode: texture spans entire surface (no tiling).
            uv_front_back = (1.0, 1.0)
            uv_left_right = (1.0, 1.0)
            uv_top_bottom = (1.0, 1.0)
        else:
            uv_front_back = (width / texture_scale, thickness / texture_scale)
            uv_left_right = (depth / texture_scale, thickness / texture_scale)
            uv_top_bottom = (width / texture_scale, depth / texture_scale)

        uvs = np.array(
            [
                # Front/back faces
                [0, 0],
                [uv_front_back[0], 0],
                [uv_front_back[0], uv_front_back[1]],
                [0, uv_front_back[1]],
                [0, 0],
                [uv_front_back[0], 0],
                [uv_front_back[0], uv_front_back[1]],
                [0, uv_front_back[1]],
                # Left/right faces
                [0, 0],
                [uv_left_right[0], 0],
                [uv_left_right[0], uv_left_right[1]],
                [0, uv_left_right[1]],
                [0, 0],
                [uv_left_right[0], 0],
                [uv_left_right[0], uv_left_right[1]],
                [0, uv_left_right[1]],
                # Top face (+Z): vertices 16-19 go back-left, back-right, front-right, front-left.
                # Flip V so texture top aligns with mesh back (top of screen in top-down view).
                [0, uv_top_bottom[1]],
                [uv_top_bottom[0], uv_top_bottom[1]],
                [uv_top_bottom[0], 0],
                [0, 0],
                # Bottom face (-Z): vertices 20-23 go front-left, front-right, back-right, back-left.
                [0, 0],
                [uv_top_bottom[0], 0],
                [uv_top_bottom[0], uv_top_bottom[1]],
                [0, uv_top_bottom[1]],
            ],
            dtype=np.float32,
        )

        log_msg = (
            f"Created floor covering GLB: {output_path.name} "
            f"({width:.2f}m x {depth:.2f}m x {thickness:.3f}m)"
        )

    # Transform to Y-up for GLTF.
    vertices = zup_to_yup_transform(vertices_zup)

    # Triangle indices differ for floor vs wall due to vertex ordering.
    # Wall vertices are ordered for viewing from +Y in Z-up, which requires
    # reversed winding compared to floor vertices.
    # fmt: off
    if is_wall:
        # Wall: reverse winding for correct face normals.
        indices = np.array([
            0, 2, 1, 0, 3, 2,        # Front (reversed)
            4, 6, 5, 4, 7, 6,        # Back (reversed)
            8, 10, 9, 8, 11, 10,     # Right (reversed)
            12, 14, 13, 12, 15, 14,  # Left (reversed)
            16, 18, 17, 16, 19, 18,  # Top (reversed)
            20, 22, 21, 20, 23, 22,  # Bottom (reversed)
        ], dtype=np.uint32)
    else:
        indices = np.array([
            0, 1, 2, 0, 2, 3,        # Front
            4, 5, 6, 4, 6, 7,        # Back
            8, 9, 10, 8, 10, 11,     # Right
            12, 13, 14, 12, 14, 15,  # Left
            16, 17, 18, 16, 18, 19,  # Top
            20, 21, 22, 20, 22, 23,  # Bottom
        ], dtype=np.uint32)
    # fmt: on

    # Normals for each face in Drake Z-up.
    normals_zup_per_face = np.array(
        [
            [0, 1, 0],  # Front (+Y)
            [0, -1, 0],  # Back (-Y)
            [1, 0, 0],  # Right (+X)
            [-1, 0, 0],  # Left (-X)
            [0, 0, 1],  # Top (+Z)
            [0, 0, -1],  # Bottom (-Z)
        ],
        dtype=np.float32,
    )
    normals_zup = np.repeat(normals_zup_per_face, 4, axis=0)
    normals = zup_to_yup_transform(normals_zup)

    create_glb_from_mesh_data(
        vertices=vertices,
        normals=normals,
        uvs=uvs,
        indices=indices,
        color_texture_path=textures["color"],
        normal_texture_path=textures["normal"],
        roughness_texture_path=textures["roughness"],
        output_path=output_path,
    )

    console_logger.info(log_msg)

    return output_path


def create_circular_thin_covering_glb(
    radius: float,
    thickness: float,
    material_folder: Path,
    output_path: Path,
    texture_scale: float | None,
    num_segments: int = 64,
    is_wall: bool = False,
) -> Path:
    """Create a thin circular covering mesh with PBR material as GLB.

    For floor coverings (is_wall=False):
        XY plane disc centered at origin, bottom at z=0, top at z=thickness.
        Primary texture face is +Z (top, visible from above).

    For wall coverings (is_wall=True):
        XZ plane disc centered in X, with center at z=radius. Back at y=0.
        Primary texture face is +Y (front, visible from inside room).
        Suitable for circular mirrors, clocks, decorative medallions.

    Args:
        radius: Radius in meters.
        thickness: Thin dimension in meters (Z for floor, Y for wall).
        material_folder: Path to folder containing PBR textures.
        output_path: Where to save the GLB file. Must have .glb extension.
        texture_scale: Meters per texture tile. None = cover mode (no tiling).
        num_segments: Number of segments around the circle (default: 64).
        is_wall: If True, create vertical wall covering; if False, floor covering.

    Returns:
        Path to the created GLB file.
    """
    textures = Material.from_path(material_folder).get_all_textures()

    # Generate circle vertices.
    angles = np.linspace(0, 2 * math.pi, num_segments, endpoint=False)

    if is_wall:
        # Wall covering: XZ plane disc, back at y=0, front at y=thickness.
        # Center at (0, thickness/2, radius) so bottom of disc is at z=0.
        center_z = radius

        # Front disc (+Y face, faces into room): center + perimeter.
        front_center = np.array([[0, thickness, center_z]], dtype=np.float32)
        front_perimeter = np.column_stack(
            [
                radius * np.cos(angles),
                np.full(num_segments, thickness),
                center_z + radius * np.sin(angles),
            ]
        ).astype(np.float32)

        # Back disc (-Y face, against wall): center + perimeter.
        back_center = np.array([[0, 0, center_z]], dtype=np.float32)
        back_perimeter = np.column_stack(
            [
                radius * np.cos(angles),
                np.full(num_segments, 0),
                center_z + radius * np.sin(angles),
            ]
        ).astype(np.float32)

        # Side vertices (duplicated for sharp edges).
        side_front = front_perimeter.copy()
        side_back = back_perimeter.copy()

        # Combine all vertices.
        vertices_zup = np.vstack(
            [
                front_center,
                front_perimeter,
                back_center,
                back_perimeter,
                side_front,
                side_back,
            ]
        )

        # Index offsets.
        front_center_idx = 0
        front_perim_start = 1
        back_center_idx = 1 + num_segments
        back_perim_start = 2 + num_segments
        side_front_start = 2 + 2 * num_segments
        side_back_start = 2 + 3 * num_segments

        # Build indices.
        # Wall mode requires reversed winding for correct face normals after
        # Z-up to Y-up coordinate transform.
        indices_list = []

        # Front disc triangles (fan from center) - PRIMARY TEXTURE FACE.
        for i in range(num_segments):
            next_i = (i + 1) % num_segments
            # Reversed winding for wall: center, next_i, i
            indices_list.extend(
                [front_center_idx, front_perim_start + next_i, front_perim_start + i]
            )

        # Back disc triangles (normal winding - reversed from front).
        for i in range(num_segments):
            next_i = (i + 1) % num_segments
            # Normal winding for back face (opposite of front)
            indices_list.extend(
                [back_center_idx, back_perim_start + i, back_perim_start + next_i]
            )

        # Side triangles (quad strip) - reversed winding.
        for i in range(num_segments):
            next_i = (i + 1) % num_segments
            indices_list.extend(
                [
                    side_front_start + i,
                    side_back_start + next_i,
                    side_back_start + i,
                    side_front_start + i,
                    side_front_start + next_i,
                    side_back_start + next_i,
                ]
            )

        # Build normals.
        normals_list = []

        # Front disc normals (all pointing +Y into room).
        front_normal = np.array([0, 1, 0], dtype=np.float32)
        normals_list.append(front_normal)
        for _ in range(num_segments):
            normals_list.append(front_normal)

        # Back disc normals (all pointing -Y toward wall).
        back_normal = np.array([0, -1, 0], dtype=np.float32)
        normals_list.append(back_normal)
        for _ in range(num_segments):
            normals_list.append(back_normal)

        # Side normals (pointing outward radially in XZ plane).
        for angle in angles:
            side_normal = np.array([np.cos(angle), 0, np.sin(angle)], dtype=np.float32)
            normals_list.append(side_normal)
        for angle in angles:
            side_normal = np.array([np.cos(angle), 0, np.sin(angle)], dtype=np.float32)
            normals_list.append(side_normal)

        log_msg = (
            f"Created circular wall covering GLB: {output_path.name} "
            f"(radius={radius:.2f}m, thickness={thickness:.3f}m)"
        )

    else:
        # Floor covering: XY plane disc, bottom at z=0, top at z=thickness.

        # Top disc: center + perimeter vertices.
        top_center = np.array([[0, 0, thickness]], dtype=np.float32)
        top_perimeter = np.column_stack(
            [
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.full(num_segments, thickness),
            ]
        ).astype(np.float32)

        # Bottom disc: center + perimeter vertices.
        bottom_center = np.array([[0, 0, 0]], dtype=np.float32)
        bottom_perimeter = np.column_stack(
            [
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.full(num_segments, 0),
            ]
        ).astype(np.float32)

        # Side vertices (duplicated for sharp edges).
        side_top = top_perimeter.copy()
        side_bottom = bottom_perimeter.copy()

        # Combine all vertices.
        vertices_zup = np.vstack(
            [
                top_center,
                top_perimeter,
                bottom_center,
                bottom_perimeter,
                side_top,
                side_bottom,
            ]
        )

        # Index offsets.
        top_center_idx = 0
        top_perim_start = 1
        bottom_center_idx = 1 + num_segments
        bottom_perim_start = 2 + num_segments
        side_top_start = 2 + 2 * num_segments
        side_bottom_start = 2 + 3 * num_segments

        # Build indices.
        indices_list = []

        # Top disc triangles (fan from center) - PRIMARY TEXTURE FACE.
        for i in range(num_segments):
            next_i = (i + 1) % num_segments
            indices_list.extend(
                [top_center_idx, top_perim_start + i, top_perim_start + next_i]
            )

        # Bottom disc triangles (reversed winding).
        for i in range(num_segments):
            next_i = (i + 1) % num_segments
            indices_list.extend(
                [bottom_center_idx, bottom_perim_start + next_i, bottom_perim_start + i]
            )

        # Side triangles (quad strip).
        for i in range(num_segments):
            next_i = (i + 1) % num_segments
            indices_list.extend(
                [
                    side_top_start + i,
                    side_bottom_start + i,
                    side_bottom_start + next_i,
                    side_top_start + i,
                    side_bottom_start + next_i,
                    side_top_start + next_i,
                ]
            )

        # Build normals.
        normals_list = []

        # Top disc normals (all pointing up).
        top_normal = np.array([0, 0, 1], dtype=np.float32)
        normals_list.append(top_normal)
        for _ in range(num_segments):
            normals_list.append(top_normal)

        # Bottom disc normals (all pointing down).
        bottom_normal = np.array([0, 0, -1], dtype=np.float32)
        normals_list.append(bottom_normal)
        for _ in range(num_segments):
            normals_list.append(bottom_normal)

        # Side normals (pointing outward radially in XY plane).
        for angle in angles:
            side_normal = np.array([np.cos(angle), np.sin(angle), 0], dtype=np.float32)
            normals_list.append(side_normal)
        for angle in angles:
            side_normal = np.array([np.cos(angle), np.sin(angle), 0], dtype=np.float32)
            normals_list.append(side_normal)

        log_msg = (
            f"Created circular floor covering GLB: {output_path.name} "
            f"(radius={radius:.2f}m, thickness={thickness:.3f}m)"
        )

    # Transform to Y-up for GLTF.
    vertices = zup_to_yup_transform(vertices_zup)

    indices = np.array(indices_list, dtype=np.uint32)

    normals_zup = np.array(normals_list, dtype=np.float32)
    normals = zup_to_yup_transform(normals_zup)

    # Build UVs - different mapping for floor vs wall after coordinate transform.
    uvs_list = []

    if texture_scale is None:
        # Cover mode: texture spans entire surface (no tiling).
        if is_wall:
            # Wall: flip both U and V for correct orientation after transform.
            uvs_list.append([0.5, 0.5])  # Center
            for angle in angles:
                u = 0.5 - 0.5 * np.cos(angle)  # Flip U
                v = 0.5 - 0.5 * np.sin(angle)  # Flip V
                uvs_list.append([u, v])

            uvs_list.append([0.5, 0.5])  # Center
            for angle in angles:
                u = 0.5 - 0.5 * np.cos(angle)  # Flip U
                v = 0.5 - 0.5 * np.sin(angle)  # Flip V
                uvs_list.append([u, v])
        else:
            # Floor: flip V for correct orientation in top-down view.
            uvs_list.append([0.5, 0.5])  # Center
            for angle in angles:
                u = 0.5 + 0.5 * np.cos(angle)
                v = 0.5 - 0.5 * np.sin(angle)  # Flip V
                uvs_list.append([u, v])

            uvs_list.append([0.5, 0.5])  # Center
            for angle in angles:
                u = 0.5 + 0.5 * np.cos(angle)
                v = 0.5 - 0.5 * np.sin(angle)  # Flip V
                uvs_list.append([u, v])

        # Side UVs (cylindrical mapping, no tiling).
        for i in range(num_segments):
            u = i / num_segments
            uvs_list.append([u, 1.0])  # Front/top
        for i in range(num_segments):
            u = i / num_segments
            uvs_list.append([u, 0])  # Back/bottom
    else:
        uv_scale = radius / texture_scale

        if is_wall:
            # Wall: flip both U and V for correct orientation after transform.
            uvs_list.append([0.5, 0.5])  # Center
            for angle in angles:
                u = 0.5 - 0.5 * uv_scale * np.cos(angle)  # Flip U
                v = 0.5 - 0.5 * uv_scale * np.sin(angle)  # Flip V
                uvs_list.append([u, v])

            uvs_list.append([0.5, 0.5])  # Center
            for angle in angles:
                u = 0.5 - 0.5 * uv_scale * np.cos(angle)  # Flip U
                v = 0.5 - 0.5 * uv_scale * np.sin(angle)  # Flip V
                uvs_list.append([u, v])
        else:
            # Floor: flip V for correct orientation in top-down view.
            uvs_list.append([0.5, 0.5])  # Center
            for angle in angles:
                u = 0.5 + 0.5 * uv_scale * np.cos(angle)
                v = 0.5 - 0.5 * uv_scale * np.sin(angle)  # Flip V
                uvs_list.append([u, v])

            uvs_list.append([0.5, 0.5])  # Center
            for angle in angles:
                u = 0.5 + 0.5 * uv_scale * np.cos(angle)
                v = 0.5 - 0.5 * uv_scale * np.sin(angle)  # Flip V
                uvs_list.append([u, v])

        # Side UVs (cylindrical mapping).
        circumference = 2 * math.pi * radius
        uv_circ = circumference / texture_scale
        uv_height = thickness / texture_scale
        for i in range(num_segments):
            u = (i / num_segments) * uv_circ
            uvs_list.append([u, uv_height])  # Front/top
        for i in range(num_segments):
            u = (i / num_segments) * uv_circ
            uvs_list.append([u, 0])  # Back/bottom

    uvs = np.array(uvs_list, dtype=np.float32)

    create_glb_from_mesh_data(
        vertices=vertices,
        normals=normals,
        uvs=uvs,
        indices=indices,
        color_texture_path=textures["color"],
        normal_texture_path=textures["normal"],
        roughness_texture_path=textures["roughness"],
        output_path=output_path,
    )

    console_logger.info(log_msg)

    return output_path


def generate_thin_covering_sdf(
    visual_mesh_path: Path,
    output_path: Path,
    model_name: str | None = None,
    collision_dims: tuple[float, float, float] | None = None,
    collision_shape: str = "rectangular",
) -> Path:
    """Generate Drake SDF file for a static thin covering.

    For floor/manipuland thin coverings (rugs, tablecloths), no collision
    geometry is included - they are purely decorative. For wall thin coverings
    (paintings, posters), collision geometry is added so Drake can detect
    furniture collisions.

    Args:
        visual_mesh_path: Path to the thin covering GLTF mesh file.
        output_path: Path where SDF file will be saved.
        model_name: Optional name for the model (defaults to mesh stem).
        collision_dims: Optional (width, depth, height) for collision primitive.
            If provided, adds collision geometry. Used for wall coverings.
        collision_shape: Shape of collision primitive ("rectangular" or "circular").
            Only used when collision_dims is provided.

    Returns:
        Path to the generated SDF file.

    Raises:
        FileNotFoundError: If visual mesh file doesn't exist.
    """
    if not visual_mesh_path.exists():
        raise FileNotFoundError(f"Visual mesh not found: {visual_mesh_path}")

    model_name = model_name or visual_mesh_path.stem

    console_logger.debug(f"Generating thin covering SDF for '{model_name}'")

    # Create SDF XML structure.
    sdf = ET.Element("sdf", version="1.7")
    model = ET.SubElement(sdf, "model", name=model_name)

    # Add single link.
    link = ET.SubElement(model, "link", name="base_link")

    # Visual geometry (external mesh reference).
    visual = ET.SubElement(link, "visual", name="visual")
    visual_geom = ET.SubElement(visual, "geometry")
    visual_mesh_elem = ET.SubElement(visual_geom, "mesh")

    # Use relative URI for mesh (assumes mesh is in same directory as SDF).
    mesh_filename = visual_mesh_path.name
    visual_uri = ET.SubElement(visual_mesh_elem, "uri")
    visual_uri.text = mesh_filename

    # Add collision geometry if dimensions provided (for wall thin coverings).
    # Wall covering visual mesh bounds (from create_rectangular_thin_covering_glb):
    #   X: -width/2 to +width/2 (centered)
    #   Y: 0 to thickness (back at Y=0 against wall)
    #   Z: 0 to height (bottom at Z=0)
    # Collision primitive must be offset to match these bounds.
    if collision_dims is not None:
        width, depth, height = collision_dims
        collision = ET.SubElement(link, "collision", name="collision")

        # Offset collision to align with visual mesh.
        # Visual mesh: Y from 0 to depth, Z from 0 to height.
        # Collision primitive is centered, so offset by (depth/2, height/2).
        collision_pose = ET.SubElement(collision, "pose")

        collision_geom = ET.SubElement(collision, "geometry")

        if collision_shape == "circular":
            # Cylinder for circular wall coverings (round paintings/mirrors).
            # Default SDF cylinder axis is Z. For wall covering, axis should be Y
            # (thin direction perpendicular to wall). Rotate 90° around X.
            roll = math.pi / 2  # Rotate Z-axis to Y-axis.
            collision_pose.text = f"0 {depth / 2:.6f} {height / 2:.6f} {roll:.6f} 0 0"

            cylinder = ET.SubElement(collision_geom, "cylinder")
            radius_elem = ET.SubElement(cylinder, "radius")
            radius_elem.text = f"{width / 2:.6f}"
            length_elem = ET.SubElement(cylinder, "length")
            length_elem.text = f"{depth:.6f}"  # Thickness along rotated axis.
            console_logger.debug(
                f"Added cylinder collision: r={width / 2:.3f}m, len={depth:.3f}m"
            )
        else:
            # Box for rectangular wall coverings.
            collision_pose.text = f"0 {depth / 2:.6f} {height / 2:.6f} 0 0 0"

            box = ET.SubElement(collision_geom, "box")
            size_elem = ET.SubElement(box, "size")
            size_elem.text = f"{width:.6f} {depth:.6f} {height:.6f}"
            console_logger.debug(
                f"Added box collision: {width:.3f}m x {depth:.3f}m x {height:.3f}m"
            )

    # Format XML with indentation.
    ET.indent(sdf, space="  ", level=0)

    # Create ElementTree and write to file.
    tree = ET.ElementTree(sdf)

    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with XML declaration.
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    console_logger.info(f"Generated thin covering SDF: {output_path}")

    return output_path
