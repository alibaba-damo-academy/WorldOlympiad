"""Coordinate system conversion utilities for mesh processing.

This module handles conversions between Blender's coordinate space (where VLM
analyzes meshes) and GLB's native coordinate space (where trimesh operates).
"""

import logging

import numpy as np

console_logger = logging.getLogger(__name__)


def blender_axis_to_glb_axis(blender_axis: str) -> str:
    """Convert axis from Blender's view to GLB's native coordinate space.

    When Blender imports a GLB file, it assumes Y-up GLTF standard and applies
    a transformation to convert to Blender's Z-up system. VLM analyzes the mesh
    in Blender's post-import space, so we need to convert those axes back to
    GLB's native space for trimesh canonicalization.

    Transformation: Blender imports GLB with -90° rotation around X
    - GLB (X, Y, Z) → Blender (X, Z, -Y)

    So the inverse mapping is:
    - Blender X → GLB X
    - Blender Y → GLB -Z
    - Blender Z → GLB Y

    Args:
        blender_axis: Axis in Blender's coordinate space (e.g., "+Z", "-Y").

    Returns:
        Axis in GLB's native coordinate space.
    """
    if len(blender_axis) != 2:
        raise ValueError(f"Invalid axis: {blender_axis}. Expected format: '+X', '-Y'")

    sign = blender_axis[0]
    axis = blender_axis[1]

    if sign not in ("+", "-"):
        raise ValueError(f"Invalid sign: {sign} in {blender_axis}")
    if axis not in ("X", "Y", "Z"):
        raise ValueError(f"Invalid axis: {axis} in {blender_axis}")

    # Inverse of Blender's import transformation.
    axis_mapping = {
        "X": "X",  # Blender X → GLB X
        "Y": "Z",  # Blender Y → GLB Z (note: sign will flip below for -Y)
        "Z": "Y",  # Blender Z → GLB Y
    }

    # Special handling for Y axis which maps with sign flip.
    if axis == "Y":
        new_sign = "-" if sign == "+" else "+"
        return new_sign + axis_mapping[axis]

    return sign + axis_mapping[axis]


def parse_axis_string(axis: str) -> np.ndarray:
    """Parse axis string to unit vector.

    Args:
        axis: Axis string (e.g., "+X", "-Y", "+Z").

    Returns:
        Unit vector as numpy array [x, y, z].

    Raises:
        ValueError: If axis string is invalid.
    """
    if len(axis) != 2:
        raise ValueError(
            f"Invalid axis string: {axis}. Expected format: '+X', '-Y', etc."
        )

    sign = axis[0]
    axis_letter = axis[1]

    if sign not in ("+", "-"):
        raise ValueError(f"Invalid axis sign: {sign}. Expected '+' or '-' in {axis}")

    if axis_letter not in ("X", "Y", "Z"):
        raise ValueError(
            f"Invalid axis letter: {axis_letter}. Expected 'X', 'Y', or 'Z' in {axis}"
        )

    # Create unit vector.
    direction = 1.0 if sign == "+" else -1.0

    if axis_letter == "X":
        return np.array([direction, 0.0, 0.0])
    elif axis_letter == "Y":
        return np.array([0.0, direction, 0.0])
    else:  # axis_letter == "Z"
        return np.array([0.0, 0.0, direction])
