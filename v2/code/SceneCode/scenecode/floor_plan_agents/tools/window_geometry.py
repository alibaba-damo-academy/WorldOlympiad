"""Window geometry generation using trimesh.

Creates window meshes with frame and glass components for placement in wall openings.
"""

import logging

from dataclasses import dataclass
from pathlib import Path

import trimesh

from trimesh.visual.material import PBRMaterial

from scenecode.utils.gltf_generation import get_zup_to_yup_matrix

console_logger = logging.getLogger(__name__)


@dataclass
class WindowDimensions:
    """Dimensions for a window."""

    width: float
    """Window width (meters)."""

    height: float
    """Window height (meters)."""

    depth: float = 0.1
    """Window depth/thickness (meters)."""

    frame_width: float = 0.05
    """Frame profile width (meters)."""

    glass_thickness: float = 0.01
    """Glass pane thickness (meters)."""


@dataclass
class WindowStyle:
    """Style parameters for window appearance."""

    frame_color: tuple[float, float, float, float] = (0.3, 0.25, 0.2, 1.0)
    """Frame color (RGBA)."""

    frame_roughness: float = 0.6
    """Frame roughness for PBR material."""

    frame_metallic: float = 0.0
    """Frame metallic factor for PBR material."""

    glass_color: tuple[float, float, float, float] = (0.9, 0.95, 0.95, 0.12)
    """Glass color (RGBA). Nearly transparent with subtle cool tint."""

    glass_roughness: float = 0.02
    """Glass roughness for PBR material. Very low for clear glass."""

    glass_metallic: float = 0.0
    """Glass metallic factor for PBR material."""


def create_window_frame(dimensions: WindowDimensions) -> trimesh.Trimesh:
    """Create window frame mesh (outer frame with inner cutout).

    Args:
        dimensions: Window dimensions.

    Returns:
        Frame mesh with hole for glass.
    """
    # Create outer frame box.
    outer = trimesh.creation.box(
        extents=[dimensions.width, dimensions.depth, dimensions.height]
    )

    # Create inner cutout (slightly smaller than outer, full depth for clean cut).
    inner_width = dimensions.width - 2 * dimensions.frame_width
    inner_height = dimensions.height - 2 * dimensions.frame_width

    if inner_width <= 0 or inner_height <= 0:
        # Frame is too thick, return solid frame.
        console_logger.warning(
            f"Window frame width ({dimensions.frame_width}m) too large for "
            f"window size ({dimensions.width}x{dimensions.height}m). "
            "Returning solid frame."
        )
        return outer

    inner = trimesh.creation.box(
        extents=[inner_width, dimensions.depth * 2, inner_height]
    )

    # Perform boolean difference.
    frame = outer.difference(inner, engine="manifold")

    return frame


def create_window_glass(dimensions: WindowDimensions) -> trimesh.Trimesh:
    """Create glass pane mesh.

    Args:
        dimensions: Window dimensions.

    Returns:
        Glass pane mesh.
    """
    glass_width = dimensions.width - 2 * dimensions.frame_width
    glass_height = dimensions.height - 2 * dimensions.frame_width

    if glass_width <= 0 or glass_height <= 0:
        # No room for glass.
        return trimesh.Trimesh()

    glass = trimesh.creation.box(
        extents=[glass_width, dimensions.glass_thickness, glass_height]
    )

    return glass


def create_window_mesh(
    width: float,
    height: float,
    depth: float = 0.1,
    frame_width: float = 0.05,
    style: WindowStyle | None = None,
) -> trimesh.Scene:
    """Create complete window with frame and glass as a scene.

    Args:
        width: Window width (meters).
        height: Window height (meters).
        depth: Window depth (meters).
        frame_width: Frame profile width (meters).
        style: Optional style parameters.

    Returns:
        A trimesh Scene containing frame and glass meshes.
    """
    if style is None:
        style = WindowStyle()

    dimensions = WindowDimensions(
        width=width, height=height, depth=depth, frame_width=frame_width
    )

    # Create frame with PBR material.
    frame = create_window_frame(dimensions)
    frame_material = PBRMaterial(
        name="window_frame",
        baseColorFactor=list(style.frame_color),
        metallicFactor=style.frame_metallic,
        roughnessFactor=style.frame_roughness,
    )
    frame.visual = trimesh.visual.TextureVisuals(material=frame_material)

    # Create glass with transparent PBR material.
    glass = create_window_glass(dimensions)
    if len(glass.vertices) > 0:
        glass_material = PBRMaterial(
            name="window_glass",
            baseColorFactor=list(style.glass_color),
            metallicFactor=style.glass_metallic,
            roughnessFactor=style.glass_roughness,
            alphaMode="BLEND",  # Required for transparency in GLTF/GLB.
            doubleSided=True,  # Glass should be visible from both sides.
        )
        glass.visual = trimesh.visual.TextureVisuals(material=glass_material)

    # Create scene with both components.
    scene = trimesh.Scene()
    scene.add_geometry(frame, geom_name="frame")
    if len(glass.vertices) > 0:
        scene.add_geometry(glass, geom_name="glass")

    return scene


def create_window_gltf(
    width: float,
    height: float,
    depth: float = 0.1,
    frame_width: float = 0.05,
    output_path: Path | None = None,
    style: WindowStyle | None = None,
) -> trimesh.Scene:
    """Create window and optionally export to GLTF.

    Args:
        width: Window width (meters).
        height: Window height (meters).
        depth: Window depth (meters).
        frame_width: Frame profile width (meters).
        output_path: If provided, export GLTF to this path.
        style: Optional style parameters.

    Returns:
        The generated window scene.
    """
    window = create_window_mesh(
        width=width, height=height, depth=depth, frame_width=frame_width, style=style
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Transform to Y-up for GLTF export (GLTF spec requires Y-up).
        window_yup = window.copy()
        window_yup.apply_transform(get_zup_to_yup_matrix())
        window_yup.export(str(output_path), file_type="gltf")
        console_logger.info(f"Exported window GLTF to {output_path}")

    return window


def create_simple_window_mesh(
    width: float, height: float, depth: float = 0.1
) -> trimesh.Trimesh:
    """Create a simple single-mesh window (combined frame and glass).

    For cases where a single mesh is preferred over a scene.

    Args:
        width: Window width (meters).
        height: Window height (meters).
        depth: Window depth (meters).

    Returns:
        A single trimesh representing the window.
    """
    dimensions = WindowDimensions(width=width, height=height, depth=depth)

    frame = create_window_frame(dimensions)
    glass = create_window_glass(dimensions)

    if len(glass.vertices) > 0:
        # Combine frame and glass into single mesh.
        combined = trimesh.util.concatenate([frame, glass])
        return combined
    else:
        return frame
