"""Mesh canonicalization via BlenderServer.

This module provides mesh canonicalization that routes through BlenderServer
to avoid fork+bpy crashes in parallel workers.
"""

import logging

from pathlib import Path
from typing import TYPE_CHECKING

from scenecode.agent_utils.room import ObjectType

if TYPE_CHECKING:
    from scenecode.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)


def canonicalize_mesh(
    gltf_path: Path,
    output_path: Path,
    up_axis: str,
    front_axis: str,
    blender_server: "BlenderServer",
    object_type: ObjectType = ObjectType.FURNITURE,
) -> Path:
    """Canonicalize GLTF mesh to standard orientation using BlenderServer.

    This function delegates to BlenderServer to avoid fork+bpy crashes.
    Forked workers cannot safely use embedded bpy due to GPU/OpenGL state
    corruption from fork.

    Pipeline:
    - Input: Y-up GLTF (from convert_glb_to_gltf with export_yup=True)
    - Blender imports and converts Y-up GLTF to Z-up internally
    - VLM axes (analyzed in Blender's Z-up space) can be used directly
    - Rotate mesh to canonical orientation in Blender's Z-up space
    - Export as Y-up GLTF (Drake will convert Y-up to Z-up on load)

    Target orientation in Blender (Z-up):
    - Up direction → +Z axis
    - Front direction → +Y axis
    - Right direction → +X axis

    Object-type-specific placement in Blender (Z-up):
    - FURNITURE: Bottom at z=0, centered in XY
    - MANIPULAND: Same as furniture
    - WALL_MOUNTED: Min Y at y=0, centered in XZ
    - CEILING_MOUNTED: Top at z=0, centered in XY

    Args:
        gltf_path: Path to input Y-up GLTF file.
        output_path: Path where canonicalized GLTF will be saved.
        up_axis: Up axis in Blender coordinates (e.g., "+Z", "-Y").
        front_axis: Front axis in Blender coordinates (e.g., "+Y", "+X").
        blender_server: BlenderServer instance for canonicalization. REQUIRED -
            forked workers cannot safely use embedded bpy due to GPU/OpenGL
            state corruption from fork.
        object_type: Type of object (determines placement strategy).

    Returns:
        Path to the canonicalized GLTF file.

    Raises:
        FileNotFoundError: If input GLTF file doesn't exist.
        RuntimeError: If BlenderServer canonicalization fails.
    """
    # Map ObjectType enum to string for server API.
    object_type_str = object_type.value

    console_logger.info(
        f"Canonicalizing GLTF via BlenderServer: {gltf_path} "
        f"(up={up_axis}, front={front_axis}, type={object_type_str})"
    )

    return blender_server.canonicalize_mesh(
        input_path=gltf_path,
        output_path=output_path,
        up_axis=up_axis,
        front_axis=front_axis,
        object_type=object_type_str,
    )
