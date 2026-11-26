"""Physics object-motion inputs for the current shape/speed formula.

Each role independently estimates a rigid transform from the initially tracked
object particles. The resulting path follows the initial visible-surface centroid.
This is an object-motion profile, separate from the robotics EE optimizer profile.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from .schema import ActionTrajectory
from .metrics import score_action_trajectories
from .current_metric import CURRENT_METRIC_VERSION, current_metric_config_sha256

INPUT_VERSION = "physics_object_motion_rigid_fit.v1"

def object_trajectory(flow_path, *, fps, expected_frames, video_sha256, role):
    path = Path(flow_path)
    if role not in {"reference", "imagination"}:
        raise ValueError("explicit reference/imagination role required")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("positive native FPS required")
    with np.load(path, allow_pickle=False) as data:
        positions = np.asarray(data["position"], dtype=np.float64)
        validity = np.asarray(data["valid_mask"])
    if positions.ndim != 3 or positions.shape[-1] != 3 or positions.shape[0] != expected_frames:
        raise ValueError("object-flow shape does not match native video")
    if validity.shape != positions.shape[:2] or not np.isin(validity, [0,1]).all():
        raise ValueError("invalid object-flow validity tensor")
    valid = validity.astype(bool) & np.isfinite(positions).all(axis=2)
    initial_valid = valid[0]
    if initial_valid.sum() < 3:
        raise ValueError("initial object has fewer than three observable particles")
    anchor = positions[0, initial_valid].mean(axis=0)
    trajectory = []
    counts = []
    residuals = []
    for frame in range(expected_frames):
        common = initial_valid & valid[frame]
        counts.append(int(common.sum()))
        if common.sum() < 3:
            raise ValueError(f"incomplete observed trajectory coverage at frame {frame}")
        source = positions[0, common]
        target = positions[frame, common]
        if np.array_equal(source, target):
            trajectory.append(anchor.copy())
            residuals.append(0.0)
            continue
        source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
        x, y = source-source_mean, target-target_mean
        if np.linalg.matrix_rank(x) < 2:
            raise ValueError(f"degenerate particle geometry at frame {frame}")
        u, _, vt = np.linalg.svd(x.T @ y)
        correction = np.eye(3)
        correction[2,2] = np.linalg.det(vt.T @ u.T)
        rotation = vt.T @ correction @ u.T
        translation = target_mean - rotation @ source_mean
        trajectory.append(rotation @ anchor + translation)
        residuals.append(float(np.sqrt(np.mean(np.sum((source @ rotation.T + translation-target)**2, axis=1)))))
    metadata = {
        "input_version": INPUT_VERSION, "role": role,
        "source_artifact": str(path.resolve()),
        "source_flow_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_video_sha256": video_sha256,
        "source_video_fps": float(fps), "source_frame_indices": list(range(expected_frames)),
        "observed_frame_mask": [True]*expected_frames,
        "visible_particle_counts": counts, "rigid_fit_rmse_m": residuals,
        "trajectory_point": "centroid of initially visible object particles transported by the estimated object transform",
        "per_role_pose_estimation": "proper-rotation least-squares fit of matching initial/current particles",
        "cross_role_alignment": "shared calibrated axes; no fitted rotation or scale between reference and prediction",
        "trajectory_smoothing": False, "temporal_subsampling": False,
        "rotation_only_and_other_object_motion": "not separate CAE components in this Cartesian-point profile",
    }
    return ActionTrajectory(np.asarray(trajectory), np.arange(expected_frames)/float(fps),
                            coordinate_frame="registered_world", metadata=metadata)

def score_object_pair(reference, predicted):
    for role, trajectory in [("reference", reference), ("imagination", predicted)]:
        if trajectory.metadata.get("input_version") != INPUT_VERSION or trajectory.metadata.get("role") != role:
            raise ValueError("physics pair has incorrect input provenance")
    if reference.coordinate_frame != predicted.coordinate_frame:
        raise ValueError("physics trajectories must share registered axes")
    result = score_action_trajectories(reference, predicted).to_dict()
    result.update(input_version=INPUT_VERSION, tracked_entity="primary task object",
                  metric_config_sha256=current_metric_config_sha256())
    return result
