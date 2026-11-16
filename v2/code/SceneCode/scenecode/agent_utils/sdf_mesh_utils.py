"""Utilities for working with SDF files and their meshes.

Provides functions to extract, combine, and process meshes from SDF files,
particularly for articulated objects with multiple links and joints.
"""

import logging
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np
import trimesh

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import BodyIndex, JointIndex
from scipy.spatial.transform import Rotation

console_logger = logging.getLogger(__name__)


class SDFParseError(ValueError):
    """Raised when SDF parsing fails due to malformed content."""


def _parse_pose_string(pose_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse an SDF pose string into translation and rotation.

    SDF pose format: "x y z roll pitch yaw" (meters and radians).

    Args:
        pose_str: Space-separated pose string "x y z roll pitch yaw".

    Returns:
        Tuple of (translation [3], rotation_matrix [3,3]).

    Raises:
        SDFParseError: If pose string is malformed.
    """
    try:
        values = [float(v) for v in pose_str.strip().split()]
    except ValueError as e:
        raise SDFParseError(f"Invalid pose string '{pose_str}': {e}") from e

    if len(values) != 6:
        raise SDFParseError(
            f"Invalid pose string '{pose_str}', expected 6 values, got {len(values)}"
        )

    # Validate values are finite.
    if not all(np.isfinite(v) for v in values):
        raise SDFParseError(f"Pose string contains non-finite values: '{pose_str}'")

    translation = np.array(values[:3])
    roll, pitch, yaw = values[3:6]

    # Convert Euler angles (roll, pitch, yaw) to rotation matrix.
    # SDF uses extrinsic XYZ convention (roll about X, pitch about Y, yaw about Z).
    rotation = Rotation.from_euler("xyz", [roll, pitch, yaw])
    rotation_matrix = rotation.as_matrix()

    return translation, rotation_matrix


def _pose_to_transform_matrix(
    translation: np.ndarray, rotation_matrix: np.ndarray
) -> np.ndarray:
    """Create a 4x4 homogeneous transformation matrix.

    Args:
        translation: 3D translation vector.
        rotation_matrix: 3x3 rotation matrix.

    Returns:
        4x4 homogeneous transformation matrix.
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


def _find_child(
    element: ET.Element, tag: str, ns: dict[str, str]
) -> ET.Element | None:
    """Find a direct child with or without an SDF XML namespace."""
    child = element.find(tag, ns)
    if child is not None:
        return child
    return element.find(tag)


def _parse_mesh_scale(mesh_elem: ET.Element, ns: dict[str, str]) -> np.ndarray:
    """Parse an SDF mesh scale element, defaulting to unit scale."""
    scale_elem = _find_child(mesh_elem, "scale", ns)
    if scale_elem is None or not scale_elem.text:
        return np.ones(3)

    scale_str = scale_elem.text.strip()
    try:
        values = [float(v) for v in scale_str.split()]
    except ValueError as e:
        raise SDFParseError(f"Invalid mesh scale '{scale_str}': {e}") from e

    if len(values) != 3:
        raise SDFParseError(
            f"Invalid mesh scale '{scale_str}', expected 3 values, got {len(values)}"
        )

    if not all(np.isfinite(v) for v in values):
        raise SDFParseError(f"Mesh scale contains non-finite values: '{scale_str}'")

    return np.array(values)


def _scale_to_transform_matrix(scale: np.ndarray) -> np.ndarray:
    """Create a 4x4 scale matrix."""
    transform = np.eye(4)
    transform[0, 0] = scale[0]
    transform[1, 1] = scale[1]
    transform[2, 2] = scale[2]
    return transform


def _yup_to_zup_transform_matrix() -> np.ndarray:
    """Create a 4x4 matrix converting GLTF/Blender Y-up vertices to Drake Z-up."""
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    return transform


def _zup_to_yup_transform_matrix() -> np.ndarray:
    """Create a 4x4 matrix converting Drake Z-up vertices to GLTF/Blender Y-up."""
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    return transform


def get_sdf_bounding_box(
    sdf_path: Path, use_max_angles: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Get the axis-aligned bounding box of an SDF at default pose (joints=0) or open
    state (use_max_angles=True).

    Args:
        sdf_path: Path to the SDF file.
        use_max_angles: If True, set all joints to upper limits (open state).

    Returns:
        Tuple of (bbox_min, bbox_max) as numpy arrays.
    """
    combined_mesh = combine_sdf_meshes_at_joint_angles(
        sdf_path=sdf_path, use_max_angles=use_max_angles
    )
    return combined_mesh.bounds[0], combined_mesh.bounds[1]


def get_sdf_dimensions(sdf_path: Path) -> np.ndarray:
    """Get the dimensions (width, depth, height) of an SDF at default pose.

    Args:
        sdf_path: Path to the SDF file.

    Returns:
        Numpy array of [width, depth, height] in meters.
    """
    bbox_min, bbox_max = get_sdf_bounding_box(sdf_path)
    return bbox_max - bbox_min


def _get_link_transforms_via_drake(
    sdf_path: Path, use_max_angles: bool = False
) -> dict[str, np.ndarray]:
    """Use Drake to get link transforms at specified joint configuration using forward
    kinematics.

    Args:
        sdf_path: Path to the SDF file.
        use_max_angles: If True, set all joints to upper limits (open state).

    Returns:
        Dict mapping link/body name to its 4x4 world transform.
    """
    # Create plant.
    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)

    # Add model from SDF.
    _ = parser.AddModels(str(sdf_path))
    plant.Finalize()

    # Create context for FK computation.
    context = plant.CreateDefaultContext()

    # Set joint positions.
    if use_max_angles:
        # Get current positions and modify them.
        positions = plant.GetPositions(context).copy()

        # Set each joint to the limit with larger absolute value.
        for joint_idx in range(plant.num_joints()):
            joint = plant.get_joint(JointIndex(joint_idx))
            # Skip fixed joints (welded).
            if joint.num_positions() == 0:
                continue

            # Get both limits.
            lower = joint.position_lower_limits()
            upper = joint.position_upper_limits()
            if len(upper) > 0:
                q_start = joint.position_start()
                for i in range(len(upper)):
                    lo = lower[i] if i < len(lower) else 0.0
                    hi = upper[i]
                    # Use whichever limit has larger absolute value.
                    if np.isfinite(lo) and np.isfinite(hi):
                        limit = hi if abs(hi) >= abs(lo) else lo
                    elif np.isfinite(hi):
                        limit = hi
                    elif np.isfinite(lo):
                        limit = lo
                    else:
                        continue
                    positions[q_start + i] = limit

        plant.SetPositions(context, positions)

    # Get world frame transforms for all bodies.
    transforms: dict[str, np.ndarray] = {}
    world_frame = plant.world_frame()
    for body_idx in range(plant.num_bodies()):
        body = plant.get_body(BodyIndex(body_idx))
        body_name = body.name()

        # Get body frame transform in world.
        body_frame = body.body_frame()
        transform = plant.CalcRelativeTransform(
            context=context, frame_A=world_frame, frame_B=body_frame
        )

        # Convert RigidTransform to 4x4 matrix.
        matrix = np.eye(4)
        matrix[:3, :3] = transform.rotation().matrix()
        matrix[:3, 3] = transform.translation()
        transforms[body_name] = matrix

    return transforms


def extract_visual_meshes_with_joint_angles(
    sdf_path: Path, use_max_angles: bool = False
) -> list[tuple[Path, np.ndarray]]:
    """Extract visual meshes with transforms at specified joint configuration. Uses
    forward kinematics to get the link transforms.

    Args:
        sdf_path: Path to the SDF file.
        use_max_angles: If True, use upper joint limits (open/extended state).

    Returns:
        List of (mesh_path, transform_matrix) tuples.
    """
    if not sdf_path.exists():
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    sdf_dir = sdf_path.parent

    # Get link transforms via Drake.
    link_transforms = _get_link_transforms_via_drake(sdf_path, use_max_angles)

    # Parse SDF XML to get visual mesh URIs.
    try:
        tree = ET.parse(sdf_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise SDFParseError(f"Failed to parse SDF XML at {sdf_path}: {e}") from e

    # Extract visual meshes with link transforms from Drake.
    visual_meshes: list[tuple[Path, np.ndarray]] = []

    ns = {"sdf": "http://www.gazebosim.org/schemas/sdf/v1.11"}
    links = root.findall(".//link", ns) or root.findall(".//link")

    for link in links:
        link_name = link.get("name", "")

        # Get link transform from Drake (defaults to identity if not found).
        link_transform = link_transforms.get(link_name, np.eye(4))

        visuals = link.findall("visual", ns) or link.findall("visual")

        for visual in visuals:
            # Get visual pose relative to link.
            visual_pose_elem = _find_child(visual, "pose", ns)
            if visual_pose_elem is not None and visual_pose_elem.text:
                trans, rot = _parse_pose_string(visual_pose_elem.text)
                visual_transform = _pose_to_transform_matrix(trans, rot)
            else:
                visual_transform = np.eye(4)

            geometry = _find_child(visual, "geometry", ns)
            if geometry is None:
                continue

            mesh_elem = _find_child(geometry, "mesh", ns)
            if mesh_elem is None:
                continue

            uri_elem = _find_child(mesh_elem, "uri", ns)
            if uri_elem is None or not uri_elem.text:
                continue

            mesh_uri = uri_elem.text.strip()
            mesh_path = sdf_dir / mesh_uri

            if not mesh_path.exists():
                console_logger.warning(f"Visual mesh not found: {mesh_path}")
                continue

            # trimesh keeps GLTF vertices in their native Y-up frame, while Drake FK
            # and SDF visual poses are Z-up. Apply FK in Z-up, then convert back so
            # the combined GLTF remains in the same Y-up convention as its inputs.
            scale_transform = _scale_to_transform_matrix(
                _parse_mesh_scale(mesh_elem, ns)
            )
            combined_transform = (
                _zup_to_yup_transform_matrix()
                @ link_transform
                @ visual_transform
                @ _yup_to_zup_transform_matrix()
                @ scale_transform
            )
            visual_meshes.append((mesh_path, combined_transform))

    return visual_meshes


def combine_sdf_meshes_at_joint_angles(
    sdf_path: Path, use_max_angles: bool = False
) -> trimesh.Trimesh:
    """Combine SDF meshes at specified joint configuration using Drake FK.

    Args:
        sdf_path: Path to the SDF file.
        use_max_angles: If True, set all joints to upper limits (open state).

    Returns:
        Combined trimesh at specified joint configuration.
    """
    visual_meshes = extract_visual_meshes_with_joint_angles(
        sdf_path=sdf_path, use_max_angles=use_max_angles
    )

    if not visual_meshes:
        raise ValueError(f"No visual meshes found in SDF: {sdf_path}")

    transformed_meshes: list[trimesh.Trimesh] = []
    for mesh_path, transform in visual_meshes:
        try:
            loaded = trimesh.load(mesh_path, force="mesh")

            if isinstance(loaded, trimesh.Scene):
                scene_meshes = []
                for node_name in loaded.graph.nodes_geometry:
                    transform_matrix, geometry_name = loaded.graph[node_name]
                    geometry = loaded.geometry[geometry_name]
                    if isinstance(geometry, trimesh.Trimesh):
                        mesh_copy = geometry.copy()
                        mesh_copy.apply_transform(transform_matrix)
                        scene_meshes.append(mesh_copy)

                if scene_meshes:
                    loaded = trimesh.util.concatenate(scene_meshes)
                else:
                    continue

            if not isinstance(loaded, trimesh.Trimesh):
                continue

            loaded.apply_transform(transform)
            transformed_meshes.append(loaded)

        except Exception as e:
            console_logger.warning(f"Failed to load mesh {mesh_path}: {e}")
            continue

    if not transformed_meshes:
        raise ValueError(f"Failed to load any meshes from SDF: {sdf_path}")

    combined = trimesh.util.concatenate(transformed_meshes)

    joint_state = "max" if use_max_angles else "zero"
    console_logger.info(
        f"Combined {len(transformed_meshes)} meshes from {sdf_path.name} "
        f"at {joint_state} joint angles: "
        f"{len(combined.vertices)} vertices, {len(combined.faces)} faces"
    )

    return combined
