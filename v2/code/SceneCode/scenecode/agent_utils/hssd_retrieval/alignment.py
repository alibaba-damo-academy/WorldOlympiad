"""HSSD mesh alignment and coordinate system transforms.

Handles rotation of HSSD meshes to canonical HSM orientation (Y-up, Z-forward).
Meshes remain in Y-up coordinates throughout for GLTF export compatibility.
"""

import logging

import numpy as np
import trimesh

from scenecode.agent_utils.hssd_retrieval.data_loader import HssdMeshMetadata

console_logger = logging.getLogger(__name__)

# HSM canonical orientation vectors.
_CANONICAL_UP = np.array([0.0, 1.0, 0.0])  # Y-axis
_CANONICAL_FRONT = np.array([0.0, 0.0, 1.0])  # Z-axis


def parse_vector(vector_str: str) -> np.ndarray:
    """Parse a vector from string format "x,y,z".

    Args:
        vector_str: Vector as string "x,y,z".

    Returns:
        NumPy array of shape (3,).
    """
    parts = vector_str.split(",")
    return np.array([float(p) for p in parts])


def compute_rotation_matrix(
    source_up: np.ndarray, source_front: np.ndarray
) -> np.ndarray:
    """Compute rotation matrix to align source orientation to canonical.

    HSM canonical orientation:
    - Up: [0, 1, 0] (Y-axis)
    - Front: [0, 0, 1] (Z-axis)

    Args:
        source_up: Source up vector (3,), will be normalized.
        source_front: Source front vector (3,), will be normalized.

    Returns:
        Rotation matrix (3, 3) that rotates source → canonical.
    """
    # Normalize input vectors.
    source_up_norm = source_up / np.linalg.norm(source_up)
    source_front_norm = source_front / np.linalg.norm(source_front)

    # Compute right vectors (perpendicular to up and front).
    target_right = np.cross(_CANONICAL_FRONT, _CANONICAL_UP)
    target_right = target_right / np.linalg.norm(target_right)

    source_right = np.cross(source_front_norm, source_up_norm)
    source_right = source_right / np.linalg.norm(source_right)

    # Build rotation matrix from basis vectors.
    source_basis = np.column_stack([source_right, source_up_norm, source_front_norm])
    target_basis = np.column_stack([target_right, _CANONICAL_UP, _CANONICAL_FRONT])

    rotation = target_basis @ np.linalg.inv(source_basis)

    return rotation


def apply_hssd_alignment_transform(
    mesh: trimesh.Trimesh, metadata: HssdMeshMetadata
) -> trimesh.Trimesh:
    """Apply HSSD alignment transform to rotate mesh to canonical orientation.

    Args:
        mesh: Input mesh in HSSD's original orientation.
        metadata: HSSD metadata containing up/front vectors.

    Returns:
        Mesh rotated to HSM canonical orientation (Y-up, Z-forward).
    """
    # Skip alignment if orientation vectors are missing or empty.
    if not metadata.up or not metadata.front:
        console_logger.debug(
            f"Mesh {metadata.mesh_id[:8]} ({metadata.name}) lacks orientation data, "
            "skipping HSSD alignment (will use downstream canonicalization)"
        )
        return mesh

    source_up = parse_vector(metadata.up)
    source_front = parse_vector(metadata.front)

    # Normalize for canonical check.
    source_up_norm = source_up / np.linalg.norm(source_up)
    source_front_norm = source_front / np.linalg.norm(source_front)

    is_already_canonical = np.allclose(
        source_up_norm, _CANONICAL_UP, atol=1e-6
    ) and np.allclose(source_front_norm, _CANONICAL_FRONT, atol=1e-6)

    if is_already_canonical:
        console_logger.debug(
            f"Mesh {metadata.mesh_id} already in canonical orientation"
        )
        return mesh

    rotation_3x3 = compute_rotation_matrix(source_up, source_front)

    transform_4x4 = np.eye(4)
    transform_4x4[:3, :3] = rotation_3x3

    mesh_copy = mesh.copy()
    mesh_copy.apply_transform(transform_4x4)

    console_logger.debug(
        f"Applied HSSD alignment to mesh {metadata.mesh_id}: "
        f"up={metadata.up} → [0,1,0], front={metadata.front} → [0,0,1]"
    )

    return mesh_copy
