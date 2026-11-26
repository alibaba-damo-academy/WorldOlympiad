"""Blender-based mesh conversion utilities.

This module provides mesh conversion functions that use bpy (Blender Python API).
These functions should only be called from the BlenderServer subprocess to ensure
bpy crashes don't kill the main scene worker process.
"""

import logging

from pathlib import Path

console_logger = logging.getLogger(__name__)


def convert_glb_to_gltf_impl(
    input_path: Path, output_path: Path, export_yup: bool = True
) -> Path:
    """Convert GLB file to GLTF with separate texture files using Blender.

    Drake requires GLTF files with separate textures rather than GLB files
    with embedded textures. This function uses Blender to import a GLB file
    and export it as GLTF_SEPARATE format, which creates separate files for
    textures and binary data.

    This function should only be called from BlenderServer to isolate bpy crashes.

    Coordinate System Handling:
    - export_yup=True: Converts Blender's Z-up to GLTF's Y-up standard
      (used for initial conversion before canonicalization)
    - export_yup=False: Preserves Blender's Z-up orientation
      (used after canonicalization for Drake)

    Pipeline workflow:
    1. Initial GLB→GLTF conversion uses export_yup=True (creates Y-up GLTF)
    2. VLM analyzes the Y-up GLTF (Blender imports and converts to Z-up)
    3. Canonicalization processes mesh in Blender's Z-up space
    4. Final export uses export_yup=False to preserve Z-up for Drake

    Args:
        input_path: Path to input GLB or GLTF file. Must exist.
        output_path: Path for output GLTF file. Textures and .bin files will
            be saved in the same directory with related names.
        export_yup: If True, converts to Y-up GLTF standard. If False, keeps
            Blender's Z-up orientation. Default True for initial conversion.

    Returns:
        Path to the converted GLTF file.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        RuntimeError: If Blender conversion fails.
    """
    import bpy

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    console_logger.info(f"Converting {input_path.suffix} to GLTF: {output_path}")

    # Clear existing scene.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Import GLB/GLTF file.
    bpy.ops.import_scene.gltf(filepath=str(input_path))

    # Select all imported objects.
    bpy.ops.object.select_all(action="SELECT")

    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export as GLTF with separate files (textures, bin data).
    # export_yup parameter controls coordinate system:
    # - True: Convert Z-up (Blender) to Y-up (GLTF standard) for pre-canonicalization
    # - False: Keep Z-up (Blender) for post-canonicalization Drake assets
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLTF_SEPARATE",  # Separate .gltf, .bin, textures.
        use_selection=True,
        export_yup=export_yup,
    )

    console_logger.info(
        f"Converted to GLTF with separate textures (Drake compatible): {output_path}"
    )

    return output_path
