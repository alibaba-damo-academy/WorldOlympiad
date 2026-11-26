"""Mesh loading utilities for SDF-based geometry.

This module provides capabilities for loading collision and visual geometry
meshes from SDF files. Handles both single meshes and Scene objects containing
multiple convex decomposition pieces with coordinate system conversions.
"""

import logging
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np
import trimesh

from scenecode.agent_utils.room import SceneObject
from scenecode.utils.geometry_utils import (
    convert_mesh_yup_to_zup,
    rigid_transform_to_matrix,
)

console_logger = logging.getLogger(__name__)


def load_collision_meshes_from_sdf(sdf_path: Path) -> list[trimesh.Trimesh]:
    """Load all collision meshes from an SDF file.

    Parses the SDF file to find collision geometry mesh references and loads
    each collision mesh. Handles both single meshes and Scene objects containing
    multiple convex decomposition pieces.

    Applies the SDF's <scale> element to each mesh if present, so returned
    geometry matches what Drake uses in physics simulation.

    Args:
        sdf_path: Path to the SDF file.

    Returns:
        List of collision mesh pieces (individual convex hulls from decomposition).
    """
    if not sdf_path.exists():
        console_logger.warning(f"SDF file not found: {sdf_path}")
        return []

    try:
        tree = ET.parse(sdf_path)
        root = tree.getroot()
    except ET.ParseError as e:
        console_logger.warning(f"Failed to parse SDF {sdf_path}: {e}")
        return []

    collision_meshes = []
    sdf_dir = sdf_path.parent

    # Iterate over collision elements to access both uri and scale children.
    for collision_elem in root.findall(".//collision"):
        mesh_elem = collision_elem.find("geometry/mesh")
        if mesh_elem is None:
            continue

        uri_elem = mesh_elem.find("uri")
        if uri_elem is None or not uri_elem.text:
            continue

        # URI is relative path from SDF directory.
        mesh_path = sdf_dir / uri_elem.text

        if not mesh_path.exists():
            console_logger.warning(f"Collision mesh not found: {mesh_path}")
            continue

        # Parse scale from sibling element (uri and scale are siblings under mesh).
        scale = np.array([1.0, 1.0, 1.0])
        scale_elem = mesh_elem.find("scale")
        if scale_elem is not None and scale_elem.text:
            try:
                scale = np.array([float(s) for s in scale_elem.text.strip().split()])
            except (ValueError, IndexError):
                pass  # Keep default scale.

        try:
            # Load collision mesh.
            coll_mesh = trimesh.load(str(mesh_path), force="mesh")

            # Extract individual convex pieces if Scene, applying scale.
            if isinstance(coll_mesh, trimesh.Scene):
                for g in coll_mesh.geometry.values():
                    if isinstance(g, trimesh.Trimesh):
                        scaled = g.copy()
                        scaled.vertices *= scale
                        collision_meshes.append(scaled)
            elif isinstance(coll_mesh, trimesh.Trimesh):
                scaled = coll_mesh.copy()
                scaled.vertices *= scale
                collision_meshes.append(scaled)

        except Exception as e:
            console_logger.warning(f"Failed to load collision mesh {mesh_path}: {e}")
            continue

    return collision_meshes


def load_and_convert_visual_mesh(geometry_path: Path) -> list[trimesh.Trimesh]:
    """Load visual mesh and convert from Y-up (GLTF) to Z-up (Drake).

    Handles both single Trimesh and Scene objects with multiple geometries.

    Args:
        geometry_path: Path to the visual geometry file.

    Returns:
        List containing a single concatenated Trimesh in Z-up coordinates.

    Raises:
        ValueError: If mesh cannot be loaded or contains no Trimesh objects.
    """
    visual_mesh = trimesh.load(str(geometry_path), force="mesh")

    if isinstance(visual_mesh, trimesh.Scene):
        meshes = [
            g for g in visual_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"No Trimesh objects in Scene from {geometry_path}")
        visual_mesh = trimesh.util.concatenate(meshes)

    if not isinstance(visual_mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh from {geometry_path}")

    # Convert from Y-up to Z-up.
    convert_mesh_yup_to_zup(visual_mesh)
    return [visual_mesh]


def load_object_collision_geometry(obj: SceneObject) -> list[trimesh.Trimesh]:
    """Load collision geometry with automatic fallback to visual geometry.

    First attempts to load collision meshes from the object's SDF file.
    If no collision geometry is available, falls back to loading and converting
    the visual geometry from Y-up (GLTF) to Z-up (Drake) coordinates.

    Applies both SDF scale (from mesh loading) and object's runtime scale_factor.

    Args:
        obj: Scene object with sdf_path and/or geometry_path.

    Returns:
        List of collision mesh pieces (Z-up Drake coordinates, scaled).

    Raises:
        ValueError: If both SDF and geometry paths are missing.
    """
    # Try to load collision geometry from SDF.
    collision_meshes = []
    if obj.sdf_path:
        collision_meshes = load_collision_meshes_from_sdf(obj.sdf_path)

    # Fallback to visual geometry if no collision geometry available.
    if not collision_meshes:
        if not obj.geometry_path:
            raise ValueError(f"Object {obj.name} missing both SDF and geometry paths")
        console_logger.warning(
            f"No collision geometry for {obj.name}, using visual geometry"
        )
        collision_meshes = load_and_convert_visual_mesh(obj.geometry_path)

    # Canonicalize articulated raw SDF geometry into the scene's visual frame.
    if obj.has_internal_model_pose():
        internal_pose_matrix = rigid_transform_to_matrix(obj.internal_model_pose)
        for mesh in collision_meshes:
            mesh.apply_transform(internal_pose_matrix)

    # Apply object's runtime scale_factor (set by rescale operations).
    if obj.scale_factor != 1.0:
        for mesh in collision_meshes:
            mesh.vertices *= obj.scale_factor

    return collision_meshes


def get_collision_vertices_world(obj: SceneObject) -> np.ndarray:
    """Get collision geometry vertices transformed to world coordinates.

    Loads all collision meshes from the object's SDF file, concatenates their
    vertices, and transforms them to world coordinates using the object's pose.

    Args:
        obj: Scene object with sdf_path and transform.

    Returns:
        Nx3 array of vertices in world coordinates (Z-up Drake frame).

    Raises:
        ValueError: If SDF path missing or no collision geometry found.
    """
    # Load collision geometry with automatic fallback to visual geometry.
    collision_meshes = load_object_collision_geometry(obj)

    # Collect all vertices from all collision pieces (already in Z-up from OBJ files).
    all_vertices = []
    for mesh in collision_meshes:
        all_vertices.append(mesh.vertices)

    # Concatenate all vertices.
    vertices_local = np.vstack(all_vertices)

    console_logger.info(
        f"Loaded {len(collision_meshes)} collision pieces, "
        f"{len(vertices_local)} total vertices for {obj.name}"
    )

    # Transform to world coordinates.
    transform_matrix = rigid_transform_to_matrix(obj.transform)
    vertices_homogeneous = np.hstack(
        [vertices_local, np.ones((len(vertices_local), 1))]
    )
    vertices_world = (transform_matrix @ vertices_homogeneous.T).T[:, :3]

    return vertices_world
