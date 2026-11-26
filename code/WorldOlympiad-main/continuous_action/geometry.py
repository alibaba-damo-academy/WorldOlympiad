"""Pose conversion helpers for model-specific action layouts."""

from __future__ import annotations

import numpy as np


def _project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def rot6d_to_matrix(rotation: np.ndarray) -> np.ndarray:
    """Decode Cosmos' column-based 6D rotation representation."""

    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (6,):
        raise ValueError(f"rot6d value must have shape (6,), got {value.shape}")
    approximate = np.stack((value[:3], value[3:], np.cross(value[:3], value[3:])), axis=-1)
    return _project_to_rotation(approximate)


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {value.shape}")
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = value / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def ee_pose_xyzw_to_initial_frame(ee_trajectory: np.ndarray) -> np.ndarray:
    """Convert absolute ``xyz + quaternion_xyzw`` poses to initial-EE XYZ."""

    poses = np.asarray(ee_trajectory, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] < 7:
        raise ValueError(f"ee trajectory must have shape (T, >=7), got {poses.shape}")
    initial_rotation = quaternion_xyzw_to_matrix(poses[0, 3:7])
    return (initial_rotation.T @ (poses[:, :3] - poses[0, :3]).T).T


def integrate_framewise_rot6d(relative_actions: np.ndarray, translation_scale: float = 1.0) -> np.ndarray:
    """Integrate ``[delta_xyz, delta_rot6d]`` actions from an identity pose.

    This matches Cosmos' ``backward_framewise`` convention:
    ``T[i+1] = T[i] @ delta_T[i]``.
    """

    actions = np.asarray(relative_actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 9:
        raise ValueError(f"relative actions must have shape (T, >=9), got {actions.shape}")
    if translation_scale <= 0:
        raise ValueError("translation_scale must be positive")

    current = np.eye(4, dtype=np.float64)
    positions = [current[:3, 3].copy()]
    for action in actions:
        delta = np.eye(4, dtype=np.float64)
        delta[:3, 3] = action[:3] / float(translation_scale)
        delta[:3, :3] = rot6d_to_matrix(action[3:9])
        current = current @ delta
        positions.append(current[:3, 3].copy())
    return np.stack(positions)


def integrate_translation_deltas(
    deltas: np.ndarray,
    *,
    scale: float = 1.0,
    include_origin: bool = True,
) -> np.ndarray:
    """Integrate Cartesian translation deltas expressed in a fixed frame."""

    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"translation deltas must have shape (T, >=3), got {values.shape}")
    if scale <= 0:
        raise ValueError("scale must be positive")
    path = np.cumsum(values[:, :3] / float(scale), axis=0)
    if include_origin:
        path = np.concatenate([np.zeros((1, 3), dtype=np.float64), path], axis=0)
    return path
