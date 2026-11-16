"""SDF file utilities for parsing, validation, and serialization.

This module provides utilities for working with SDF (Simulation Description Format)
files used by Drake for physics simulation.
"""

import xml.etree.ElementTree as ET

from pathlib import Path

from pydrake.all import Quaternion, RigidTransform, RollPitchYaw, RotationMatrix


def parse_pose(pose_text: str) -> tuple[list[float], list[float]]:
    """Parse SDF pose string into position and orientation.

    Args:
        pose_text: SDF pose string in format 'x y z roll pitch yaw'.

    Returns:
        Tuple of ([x, y, z], [roll, pitch, yaw]) where angles are in radians.

    Example:
        >>> xyz, rpy = parse_pose("1.0 2.0 3.0 0.1 0.2 0.3")
        >>> xyz
        [1.0, 2.0, 3.0]
        >>> rpy
        [0.1, 0.2, 0.3]
    """
    values = [float(v) for v in pose_text.strip().split()]
    if len(values) != 6:
        raise ValueError(f"Expected 6 values in pose, got {len(values)}: {pose_text}")
    return values[:3], values[3:]


def pose_to_string(xyz: list[float], rpy: list[float]) -> str:
    """Format position and orientation as SDF pose string.

    Args:
        xyz: Position [x, y, z].
        rpy: Orientation [roll, pitch, yaw] in radians.

    Returns:
        SDF pose string 'x y z roll pitch yaw'.
    """
    return (
        f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
        f"{rpy[0]:.8f} {rpy[1]:.8f} {rpy[2]:.8f}"
    )


def parse_scale(scale_text: str) -> list[float]:
    """Parse SDF scale string into scale factors.

    Args:
        scale_text: SDF scale string in format 'sx sy sz'.

    Returns:
        List [sx, sy, sz] of scale factors.

    Example:
        >>> parse_scale("1.5 1.5 1.5")
        [1.5, 1.5, 1.5]
    """
    values = [float(v) for v in scale_text.strip().split()]
    if len(values) != 3:
        raise ValueError(f"Expected 3 values in scale, got {len(values)}: {scale_text}")
    return values


def scale_to_string(scale: list[float]) -> str:
    """Format scale factors as SDF scale string.

    Args:
        scale: Scale factors [sx, sy, sz].

    Returns:
        SDF scale string 'sx sy sz'.
    """
    return f"{scale[0]:.8f} {scale[1]:.8f} {scale[2]:.8f}"


def scale_inertia(
    ixx: float,
    iyy: float,
    izz: float,
    ixy: float,
    ixz: float,
    iyz: float,
    scale_factor: float,
) -> tuple[float, float, float, float, float, float]:
    """Scale inertia tensor components for uniform scaling.

    For uniform scaling by factor s with constant mass, inertia scales as s^2
    because inertia is proportional to distance^2 from rotation axis.

    Args:
        ixx, iyy, izz: Diagonal components of inertia tensor.
        ixy, ixz, iyz: Off-diagonal components of inertia tensor.
        scale_factor: Uniform scale factor.

    Returns:
        Tuple of scaled (ixx, iyy, izz, ixy, ixz, iyz).
    """
    s2 = scale_factor * scale_factor
    return (
        ixx * s2,
        iyy * s2,
        izz * s2,
        ixy * s2,
        ixz * s2,
        iyz * s2,
    )


def scale_pose_translation(xyz: list[float], scale_factor: float) -> list[float]:
    """Scale the translation component of a pose.

    Args:
        xyz: Position [x, y, z].
        scale_factor: Uniform scale factor.

    Returns:
        Scaled position [x*s, y*s, z*s].
    """
    return [v * scale_factor for v in xyz]


def is_static_sdf_model(sdf_path: Path) -> bool:
    """Check if an SDF model is declared as static.

    Static models are automatically welded to the world by Drake, so we should
    NOT add an explicit add_weld directive for them (would cause duplicate joint).

    Args:
        sdf_path: Path to the SDF file.

    Returns:
        True if the model has <static>true</static>, False otherwise.
    """
    try:
        tree = ET.parse(sdf_path)
        root = tree.getroot()

        # Look for <static>true</static> in the model.
        for static_elem in root.findall(".//static"):
            if static_elem.text and static_elem.text.strip().lower() == "true":
                return True
        return False
    except Exception:
        # If we can't parse, assume not static (safer to weld).
        return False


def extract_base_link_name_from_sdf(sdf_path: Path) -> str:
    """
    Extract the root link name from an SDF file.

    For articulated objects with joints, finds the link that is not a child of
    any joint (the root of the kinematic tree). For simple single-link objects,
    returns the first link found.

    Args:
        sdf_path (Path): Path to the SDF file.

    Returns:
        str: The name of the root link in the SDF file.

    Raises:
        ValueError: If no links are found in the SDF file.
    """
    try:
        tree = ET.parse(sdf_path)
        root = tree.getroot()

        # Collect all link names.
        all_links: set[str] = set()
        for link in root.findall(".//link"):
            if "name" in link.attrib:
                all_links.add(link.attrib["name"])

        if not all_links:
            raise ValueError(f"No link elements found in SDF file: {sdf_path}")

        # Collect all child links from joints.
        child_links: set[str] = set()
        for joint in root.findall(".//joint"):
            child_elem = joint.find("child")
            if child_elem is not None and child_elem.text:
                child_links.add(child_elem.text.strip())

        # Root link is one that is not a child of any joint.
        root_links = all_links - child_links

        if root_links:
            # If multiple roots exist, prefer one that contains "body" or "base".
            for link_name in root_links:
                if "body" in link_name.lower() or "base" in link_name.lower():
                    return link_name
            # Otherwise return any root link.
            return next(iter(root_links))

        # Fallback: if all links are children (circular or no joints), return first.
        return next(iter(all_links))

    except ET.ParseError as e:
        raise ValueError(f"Failed to parse SDF file {sdf_path}: {e}")
    except FileNotFoundError:
        raise ValueError(f"SDF file not found: {sdf_path}")


def extract_model_pose_from_sdf(sdf_path: Path) -> RigidTransform:
    """Extract the top-level model pose from an SDF file.

    Returns identity when the model omits a top-level pose.

    Args:
        sdf_path: Path to the SDF file.

    Returns:
        Model pose as a RigidTransform.

    Raises:
        ValueError: If the SDF cannot be parsed.
    """
    try:
        tree = ET.parse(sdf_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse SDF file {sdf_path}: {e}")
    except FileNotFoundError:
        raise ValueError(f"SDF file not found: {sdf_path}")

    model_elem = root.find('./model')
    if model_elem is None:
        model_elem = root.find('.//model')
    if model_elem is None:
        return RigidTransform()

    pose_elem = model_elem.find('pose')
    if pose_elem is None or not pose_elem.text:
        return RigidTransform()

    xyz, rpy = parse_pose(pose_elem.text)
    return RigidTransform(RollPitchYaw(rpy), xyz)


def serialize_rigid_transform(transform: RigidTransform) -> dict[str, list[float]]:
    """Convert RigidTransform to serializable dict for hashing."""
    translation = transform.translation()
    quaternion = transform.rotation().ToQuaternion().wxyz()

    return {
        "translation": [
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        ],
        "rotation_wxyz": [
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        ],
    }


def deserialize_rigid_transform(data: dict) -> RigidTransform:
    """Convert serialized dict back to RigidTransform.

    Args:
        data: Dict with "translation" and "rotation_wxyz" keys as produced by
            serialize_rigid_transform().

    Returns:
        RigidTransform reconstructed from the serialized data.
    """
    translation = data.get("translation", [0, 0, 0])
    rotation_wxyz = data.get("rotation_wxyz", [1, 0, 0, 0])
    rotation = RotationMatrix(Quaternion(wxyz=rotation_wxyz))
    return RigidTransform(rotation, translation)
