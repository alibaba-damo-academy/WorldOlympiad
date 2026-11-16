"""Wall geometry generation with door/window cutouts using trimesh + manifold3d.

This module creates wall meshes with openings for doors and windows.
Uses manifold3d backend for parallel-safe boolean operations.
"""

import logging

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from PIL import Image
from trimesh.visual.material import PBRMaterial as TrimeshPBRMaterial
from trimesh.visual.texture import TextureVisuals

from scenecode.agent_utils.house import OpeningType
from scenecode.utils.gltf_generation import get_zup_to_yup_matrix
from scenecode.utils.material import Material

console_logger = logging.getLogger(__name__)


def apply_pbr_material(mesh: trimesh.Trimesh, material: Material) -> trimesh.Trimesh:
    """Apply PBR material textures to a mesh with existing UV coordinates.

    Args:
        mesh: Mesh with UV coordinates already set.
        material: PBR material with textures.

    Returns:
        Mesh with PBR material applied.
    """
    # Find texture files.
    color_path = material.get_texture("Color")
    normal_path = material.get_texture("Normal")
    roughness_path = material.get_texture("Roughness")

    if not color_path:
        console_logger.warning(f"No color texture found in {material.path}")
        return mesh

    # Load images.
    color_img = Image.open(color_path)
    normal_img = Image.open(normal_path) if normal_path else None
    roughness_img = Image.open(roughness_path) if roughness_path else None

    # Create PBR material.
    pbr_material = TrimeshPBRMaterial(
        baseColorTexture=color_img,
        normalTexture=normal_img,
        metallicRoughnessTexture=roughness_img,
    )

    # Get existing UV coordinates.
    if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        uvs = mesh.visual.uv
    else:
        console_logger.warning("Mesh has no UV coordinates, skipping PBR material")
        return mesh

    # Apply texture visual with PBR material.
    mesh.visual = TextureVisuals(uv=uvs, material=pbr_material)

    return mesh


@dataclass
class WallSpec:
    """Specification for a wall's position and bounding box.

    Used by floor plan generators to define wall geometry for SDF generation.
    """

    name: str
    """Wall identifier (e.g., "left_wall", "front_wall")."""

    center_x: float
    """X position of wall center."""

    center_y: float
    """Y position of wall center."""

    bbox_width: float
    """Bounding box width (X dimension)."""

    bbox_depth: float
    """Bounding box depth (Y dimension)."""

    thickness: float
    """Wall thickness (same as bbox dimension in one axis)."""


@dataclass
class WallDimensions:
    """Dimensions for a wall segment."""

    width: float
    """Length along the wall direction (meters)."""

    height: float
    """Wall height (meters)."""

    thickness: float = 0.05
    """Wall thickness (meters), default 5cm."""


@dataclass
class WallOpening:
    """An opening (door or window) in a wall.

    Position is measured from the left edge of the wall (looking at it from outside).
    """

    position_along_wall: float
    """Distance from wall start to opening LEFT EDGE (meters)."""

    width: float
    """Opening width (meters)."""

    height: float
    """Opening height (meters)."""

    sill_height: float = 0.0
    """Height from floor to bottom of opening (meters)."""

    opening_type: OpeningType = OpeningType.DOOR
    """Type of opening."""

    def to_dict(self) -> dict:
        """Serialize opening to dictionary for cache key generation."""
        return {
            "position_along_wall": self.position_along_wall,
            "width": self.width,
            "height": self.height,
            "sill_height": self.sill_height,
            "opening_type": self.opening_type.value,
        }


def create_wall_mesh(
    width: float, height: float, thickness: float = 0.05
) -> trimesh.Trimesh:
    """Create a plain wall mesh without openings.

    Args:
        width: Wall length (meters).
        height: Wall height (meters).
        thickness: Wall thickness (meters).

    Returns:
        A trimesh box representing the wall.
    """
    wall = trimesh.creation.box(extents=[width, thickness, height])
    # Center the wall at origin, with bottom at z=0.
    wall.apply_translation([0, 0, height / 2])
    return wall


def create_wall_with_openings(
    dimensions: WallDimensions, openings: list[WallOpening]
) -> trimesh.Trimesh:
    """Create wall mesh with door/window cutouts.

    Uses manifold3d backend for parallel-safe boolean operations.

    Args:
        dimensions: Wall dimensions (width, height, thickness).
        openings: List of openings to cut from the wall.

    Returns:
        A trimesh with openings cut out.
    """
    # Create base wall mesh.
    wall = trimesh.creation.box(
        extents=[dimensions.width, dimensions.thickness, dimensions.height]
    )
    # Position wall with bottom at z=0, centered on x and y.
    wall.apply_translation([0, 0, dimensions.height / 2])

    if not openings:
        return wall

    # Cut each opening.
    for opening in openings:
        # Create cutter box slightly larger in Y to ensure clean cut.
        cutter = trimesh.creation.box(
            extents=[
                opening.width,
                dimensions.thickness * 2,  # Ensure full penetration.
                opening.height,
            ]
        )

        # Position cutter at opening location.
        # X: offset from center of wall. position_along_wall is LEFT EDGE.
        opening_center_x = (
            opening.position_along_wall + opening.width / 2 - dimensions.width / 2
        )
        # Z: bottom of opening + half height.
        opening_center_z = opening.sill_height + opening.height / 2

        cutter.apply_translation([opening_center_x, 0, opening_center_z])

        # Perform boolean difference using manifold3d backend.
        wall = wall.difference(cutter, engine="manifold")

    return wall


def apply_box_uv_projection(
    mesh: trimesh.Trimesh, scale: float = 0.5
) -> trimesh.Trimesh:
    """Apply box UV projection for consistent material tiling.

    Projects UVs onto the mesh using a box projection method, which works well
    for architectural geometry like walls.

    Note: This function calls unmerge_vertices() to ensure each face has its own
    vertices with correct UV coordinates. This is required because different faces
    may project onto different planes (XY, XZ, or YZ), and shared vertices would
    otherwise get incorrect averaged UVs.

    Args:
        mesh: Input mesh.
        scale: Meters per UV unit (default 0.5 = 2 tiles per meter).

    Returns:
        Mesh with UV coordinates applied.
    """
    if mesh.vertices.shape[0] == 0:
        return mesh

    # Unmerge vertices so each face has its own vertices. This is required for
    # correct box UV projection since different faces project onto different
    # planes (XY, XZ, or YZ) based on their normal direction.
    mesh.unmerge_vertices()

    # Compute face normals if not present.
    if mesh.face_normals is None or len(mesh.face_normals) == 0:
        mesh.fix_normals()

    # For each face, determine dominant axis and project UVs.
    vertices = mesh.vertices
    faces = mesh.faces
    face_normals = mesh.face_normals

    # Create UV array. Since vertices are unmerged, each vertex belongs to
    # exactly one face, so we can assign UVs directly without accumulation.
    uvs = np.zeros((len(vertices), 2))

    for face_idx, face in enumerate(faces):
        normal = face_normals[face_idx]
        abs_normal = np.abs(normal)

        # Determine projection plane based on dominant normal component.
        if abs_normal[0] >= abs_normal[1] and abs_normal[0] >= abs_normal[2]:
            # X-dominant: project onto YZ plane.
            u_axis, v_axis = 1, 2
        elif abs_normal[1] >= abs_normal[0] and abs_normal[1] >= abs_normal[2]:
            # Y-dominant: project onto XZ plane.
            u_axis, v_axis = 0, 2
        else:
            # Z-dominant: project onto XY plane.
            u_axis, v_axis = 0, 1

        # Direct UV assignment for each vertex in this face.
        for vertex_idx in face:
            uvs[vertex_idx, 0] = vertices[vertex_idx, u_axis] / scale
            uvs[vertex_idx, 1] = vertices[vertex_idx, v_axis] / scale

    # Create visual with UVs.
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)

    return mesh


def create_wall_gltf(
    dimensions: WallDimensions,
    openings: list[WallOpening] | None = None,
    output_path: Path | None = None,
    uv_scale: float = 0.5,
    material: Material | None = None,
) -> trimesh.Trimesh:
    """Create a wall mesh with openings and export to GLTF.

    Args:
        dimensions: Wall dimensions.
        openings: List of openings (doors/windows) to cut.
        output_path: If provided, export GLTF to this path.
        uv_scale: Meters per UV unit for texture tiling.
        material: If provided, apply PBR material with textures.

    Returns:
        The generated wall mesh.
    """
    if openings:
        wall = create_wall_with_openings(dimensions=dimensions, openings=openings)
    else:
        wall = create_wall_mesh(
            width=dimensions.width,
            height=dimensions.height,
            thickness=dimensions.thickness,
        )

    # Apply UV projection for materials.
    wall = apply_box_uv_projection(wall, scale=uv_scale)

    # Apply PBR material if provided.
    if material:
        wall = apply_pbr_material(mesh=wall, material=material)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Transform to Y-up for GLTF export (GLTF spec requires Y-up).
        wall_yup = wall.copy()
        wall_yup.apply_transform(get_zup_to_yup_matrix())
        wall_yup.export(str(output_path), file_type="gltf")
        console_logger.info(f"Exported wall GLTF to {output_path}")

    return wall


def validate_openings(
    dimensions: WallDimensions, openings: list[WallOpening], margin: float = 0.1
) -> list[str]:
    """Validate that openings fit within wall bounds.

    Args:
        dimensions: Wall dimensions.
        openings: List of openings to validate.
        margin: Minimum distance from opening edge to wall edge (meters).

    Returns:
        List of validation error messages (empty if all valid).
    """
    errors = []
    for i, opening in enumerate(openings):
        # Check horizontal bounds. position_along_wall is LEFT EDGE.
        left_edge = opening.position_along_wall
        right_edge = opening.position_along_wall + opening.width

        if left_edge < margin:
            errors.append(
                f"Opening {i}: left edge ({left_edge:.2f}m) too close to wall start "
                f"(minimum margin: {margin}m)"
            )

        if right_edge > dimensions.width - margin:
            errors.append(
                f"Opening {i}: right edge ({right_edge:.2f}m) too close to wall end "
                f"(wall width: {dimensions.width:.2f}m, minimum margin: {margin}m)"
            )

        # Check vertical bounds.
        top_edge = opening.sill_height + opening.height

        if opening.sill_height < 0:
            errors.append(
                f"Opening {i}: sill height ({opening.sill_height:.2f}m) cannot be negative"
            )

        if top_edge > dimensions.height - margin:
            errors.append(
                f"Opening {i}: top edge ({top_edge:.2f}m) too close to wall top "
                f"(wall height: {dimensions.height:.2f}m, minimum margin: {margin}m)"
            )

    # Check for overlapping openings.
    for i, op1 in enumerate(openings):
        for j, op2 in enumerate(openings):
            if i >= j:
                continue

            # position_along_wall is LEFT EDGE.
            left1 = op1.position_along_wall
            right1 = op1.position_along_wall + op1.width
            left2 = op2.position_along_wall
            right2 = op2.position_along_wall + op2.width

            # Check horizontal overlap.
            if not (right1 <= left2 or right2 <= left1):
                bottom1, top1 = op1.sill_height, op1.sill_height + op1.height
                bottom2, top2 = op2.sill_height, op2.sill_height + op2.height

                # Check vertical overlap.
                if not (top1 <= bottom2 or top2 <= bottom1):
                    errors.append(f"Openings {i} and {j} overlap")

    return errors
