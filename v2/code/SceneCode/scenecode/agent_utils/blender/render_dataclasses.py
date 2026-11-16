"""Dataclasses for blender rendering operations."""

from dataclasses import dataclass
from pathlib import Path

from mathutils import Vector


@dataclass
class OverlayRenderingSetup:
    """Data required for metric rendering operations."""

    camera_obj: "bpy.types.Object"
    bbox_center: Vector
    max_dim: float
    camera_distance: float


@dataclass
class LinkMeshInfo:
    """Mesh information for a single articulated link."""

    link_name: str
    """Name of the link in the URDF."""

    mesh_paths: list[Path]
    """List of mesh file paths (OBJ, GLTF, GLB) for this link's visual geometry."""

    origins: list[tuple[float, float, float]]
    """Origin offsets for each mesh file (xyz in meters)."""

    world_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """World position of the link from joint chain (xyz in meters)."""

    world_rotation: tuple[tuple[float, ...], ...] | None = None
    """World rotation matrix of the link from joint chain (3x3)."""


@dataclass
class ArticulatedRenderResult:
    """Result from articulated object multi-view rendering."""

    combined_image_paths: list[Path]
    """Paths to combined (all links) view images."""

    link_image_paths: dict[str, list[Path]]
    """Mapping of link name to per-link view images."""

    link_dimensions: dict[str, tuple[float, float, float]]
    """Mapping of link name to bounding box dimensions (width, depth, height)."""

    combined_dimensions: tuple[float, float, float]
    """Bounding box dimensions of the entire object (width, depth, height)."""
