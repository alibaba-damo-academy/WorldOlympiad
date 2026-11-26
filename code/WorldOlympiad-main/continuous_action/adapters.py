"""Adapters from Dream2Flow and WAM outputs to canonical action trajectories."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import (
    ee_pose_xyzw_to_initial_frame,
    integrate_framewise_rot6d,
    integrate_translation_deltas,
)
from .schema import ActionTrajectory, timestamps_for_samples


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _load_torch(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(f"loading {path.suffix} artifacts requires PyTorch") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_action_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], (list, tuple)):
            return payload["data"]
        for key in ("action", "actions", "pred_action", "action_trajectory"):
            if key in payload:
                return _extract_action_payload(payload[key])
        outputs = payload.get("outputs")
        if isinstance(outputs, list) and outputs:
            return _extract_action_payload(outputs[0])
        content = payload.get("content")
        if content is not None:
            return _extract_action_payload(content)
    return payload


def load_action_array(path: str | Path, *, key: str | None = None) -> np.ndarray:
    """Load a raw action array from JSON, NumPy, or PyTorch output."""

    artifact_path = Path(path)
    suffix = artifact_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    elif suffix == ".npy":
        payload = np.load(artifact_path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(artifact_path, allow_pickle=False) as data:
            selected = key or next((name for name in ("action", "actions", "pred_action") if name in data), None)
            if selected is None:
                if len(data.files) != 1:
                    raise ValueError(f"choose a key from {data.files} for {artifact_path}")
                selected = data.files[0]
            payload = np.asarray(data[selected])
    elif suffix in (".pt", ".pth"):
        payload = _load_torch(artifact_path)
    else:
        raise ValueError(f"unsupported action artifact suffix: {suffix}")
    if key is not None and isinstance(payload, dict):
        payload = payload[key]
    array = _as_numpy(_extract_action_payload(payload)).astype(np.float64, copy=False)
    if array.size == 0:
        raise ValueError(f"empty action artifact: {artifact_path}")
    return array


def load_dream2flow_plan(
    path: str | Path,
    *,
    duration: float | None = None,
    sample_rate: float | None = None,
) -> ActionTrajectory:
    """Load Dream2Flow's final robot plan, never its intermediate object flow."""

    artifact_path = Path(path)
    suffix = artifact_path.suffix.lower()
    if suffix in (".pt", ".pth"):
        payload = _load_torch(artifact_path)
        if not isinstance(payload, dict) or "ee_trajectory" not in payload:
            raise ValueError(
                "Dream2Flow reference must be trajectory_optimization_plan.pt "
                "containing 'ee_trajectory', not object_flow_result.pt"
            )
        ee_trajectory = _as_numpy(payload["ee_trajectory"])
        embedded_timestamps = _as_numpy(payload["timestamps"]) if "timestamps" in payload else None
        source_frame_indices = (
            _as_numpy(payload["source_frame_indices"]).astype(np.int64, copy=False)
            if "source_frame_indices" in payload
            else None
        )
        source_video_fps = (
            float(_as_numpy(payload["source_video_fps"]).item())
            if "source_video_fps" in payload
            else None
        )
        observed_frame_mask = (
            _as_numpy(payload["observed_frame_mask"]).astype(bool, copy=False)
            if "observed_frame_mask" in payload
            else None
        )
    elif suffix == ".npz":
        with np.load(artifact_path, allow_pickle=False) as data:
            if "ee_trajectory" not in data:
                raise ValueError("Dream2Flow NPZ must contain 'ee_trajectory'")
            ee_trajectory = np.asarray(data["ee_trajectory"])
            embedded_timestamps = np.asarray(data["timestamps"]) if "timestamps" in data else None
            source_frame_indices = (
                np.asarray(data["source_frame_indices"], dtype=np.int64)
                if "source_frame_indices" in data
                else None
            )
            source_video_fps = (
                float(np.asarray(data["source_video_fps"]).item())
                if "source_video_fps" in data
                else None
            )
            observed_frame_mask = (
                np.asarray(data["observed_frame_mask"], dtype=bool)
                if "observed_frame_mask" in data
                else None
            )
    else:
        raise ValueError("Dream2Flow plan must be a .pt/.pth or .npz artifact")

    positions = ee_pose_xyzw_to_initial_frame(ee_trajectory)
    if observed_frame_mask is not None and observed_frame_mask.shape != (len(positions),):
        raise ValueError(
            "Dream2Flow plan observed_frame_mask must align with ee_trajectory; "
            f"got {observed_frame_mask.shape} for {len(positions)} poses"
        )
    timestamps = (
        np.asarray(embedded_timestamps, dtype=np.float64)
        if embedded_timestamps is not None
        else timestamps_for_samples(len(positions), duration=duration, sample_rate=sample_rate)
    )
    return ActionTrajectory(
        positions=positions,
        timestamps=timestamps,
        metadata={
            "provider": "dream2flow",
            "source_artifact": str(artifact_path),
            "source_key": "ee_trajectory",
            "reference_kind": "planned_action_trajectory",
            "timing_source": (
                "embedded_source_video_timestamps"
                if embedded_timestamps is not None
                else "manifest_duration_or_sample_rate"
            ),
            "source_frame_indices": (
                source_frame_indices.tolist() if source_frame_indices is not None else None
            ),
            "source_video_fps": source_video_fps,
            "observed_frame_mask": (
                observed_frame_mask.tolist() if observed_frame_mask is not None else None
            ),
        },
    )


def load_dream2flow_video_plan(
    path: str | Path, *, video_path: str | Path, video_role: str,
    source_model: str | None = None, object_name: str | None = None,
    duration: float | None = None, sample_rate: float | None = None,
) -> ActionTrajectory:
    """Load a verified official plan in the pair's common registered world frame."""
    from .official_plan import validate_plan
    meta = validate_plan(path, video_path, video_role)
    if object_name is not None and meta['object_name'] != object_name:
        raise ValueError('official plan tracks a different task object')
    with np.load(path, allow_pickle=False) as data:
        poses = np.asarray(data['ee_trajectory'], dtype=np.float64)
        timestamps = np.asarray(data['timestamps'], dtype=np.float64)
        indices = np.asarray(data['source_frame_indices'], dtype=np.int64)
        observed = np.asarray(data['observed_frame_mask'], dtype=bool)
    if poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all():
        raise ValueError('official plan requires finite T x 7 end-effector poses')
    expected_indices = np.arange(0, meta['source_video_frame_count'], max(1, meta['source_video_frame_count'] // meta['optimizer_config']['max_num_timesteps_for_optimization']))
    if not np.array_equal(indices, expected_indices) or not np.allclose(timestamps, indices / meta['source_video_fps'], rtol=0, atol=1e-10):
        raise ValueError('official plan native timeline mismatch')
    if observed.shape != (len(poses),) or len(indices) != len(poses):
        raise ValueError('official plan observation/timeline shape mismatch')
    if sample_rate is not None and abs(sample_rate - meta['source_video_fps']) > 1e-8:
        raise ValueError('manifest FPS differs from official plan native video')
    return ActionTrajectory(
        positions=poses[:, :3], timestamps=timestamps,
        coordinate_frame='registered_world', units='m',
        metadata={**meta, 'tracked_object': meta['object_name'],
                  'source_artifact': str(path), 'source_video': str(video_path),
                  'source_model': source_model, 'observed_frame_mask': observed.tolist(),
                  'source_frame_indices':indices.tolist(),
                  'source_frame_step_max':int(np.diff(indices).max()),
                  'trajectory_sample_interval_max_sec':float(np.diff(timestamps).max()),
                  'timing_source': 'official_subsampled_native_video_timestamps'},
    )


def _contiguous_index_ranges(indices: np.ndarray) -> list[list[int]]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return []
    ranges: list[list[int]] = []
    start = previous = int(values[0])
    for raw_index in values[1:]:
        index = int(raw_index)
        if index != previous + 1:
            ranges.append([start, previous])
            start = index
        previous = index
    ranges.append([start, previous])
    return ranges


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_dream2flow_object_flow_arrays(
    artifact_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load the released object-flow tensors without changing their values."""

    suffix = artifact_path.suffix.lower()
    if suffix == ".npz":
        with np.load(artifact_path, allow_pickle=False) as payload:
            if "position" not in payload or "valid_mask" not in payload:
                raise ValueError(
                    "Dream2Flow object-flow NPZ must contain position and valid_mask"
                )
            position = np.asarray(payload["position"])
            valid_mask = np.asarray(payload["valid_mask"])
            initial_pixel_coords = (
                np.asarray(payload["initial_pixel_coords"])
                if "initial_pixel_coords" in payload
                else None
            )
            region_mask = (
                np.asarray(payload["region_mask"])
                if "region_mask" in payload
                else None
            )
        return position, valid_mask, initial_pixel_coords, region_mask
    if suffix not in {".pt", ".pth"}:
        raise ValueError("Dream2Flow object flow must be a .pt/.pth or .npz artifact")

    try:
        import torch
    except ImportError as exc:
        raise ImportError("loading Dream2Flow object flow requires PyTorch") from exc
    try:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(artifact_path, map_location="cpu")
    if not isinstance(payload, dict) or "flow" not in payload:
        raise ValueError("Dream2Flow object-flow artifact must contain a 'flow' object")
    flow = payload["flow"]
    position_value = getattr(flow, "position", None)
    valid_value = getattr(flow, "valid_mask", None)
    if position_value is None or valid_value is None:
        raise ValueError("Dream2Flow flow object must contain position and valid_mask")
    return (
        _as_numpy(position_value),
        _as_numpy(valid_value),
        _as_numpy(payload["initial_pixel_coords"])
        if payload.get("initial_pixel_coords") is not None
        else None,
        _as_numpy(payload["region_mask"])
        if payload.get("region_mask") is not None
        else None,
    )


def load_dream2flow_video_object_flow(*args, **kwargs):
    """Retired source. Raw flow is an intermediate, never an action trajectory."""
    raise ValueError("median object-flow adaptation retired; run official Dream2Flow optimizer and load dream2flow_video_plan")


def cosmos_rot6d_actions_to_trajectory(
    actions: np.ndarray,
    *,
    duration: float | None = None,
    sample_rate: float | None = None,
    translation_scale: float = 1.0,
    provider: str = "cosmos3",
    conditioning_mode: str = "inverse_dynamics",
) -> ActionTrajectory:
    """Convert Cosmos ``delta_xyz + rot6d`` actions to an EE path."""

    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 9:
        raise ValueError(f"Cosmos actions must have shape (T, >=9), got {values.shape}")
    positions = integrate_framewise_rot6d(values[:, :9], translation_scale=translation_scale)
    timestamps = timestamps_for_samples(len(positions), duration=duration, sample_rate=sample_rate)
    return ActionTrajectory(
        positions=positions,
        timestamps=timestamps,
        metadata={
            "provider": provider,
            "conditioning_mode": conditioning_mode,
            "raw_action_dim": int(values.shape[1]),
            "rotation_format": "rot6d",
            "pose_convention": "backward_framewise",
            "translation_scale": float(translation_scale),
        },
    )


def load_cosmos_actions(
    path: str | Path,
    *,
    duration: float | None = None,
    sample_rate: float | None = None,
    translation_scale: float = 1.0,
    conditioning_mode: str = "inverse_dynamics",
) -> ActionTrajectory:
    trajectory = cosmos_rot6d_actions_to_trajectory(
        load_action_array(path),
        duration=duration,
        sample_rate=sample_rate,
        translation_scale=translation_scale,
        conditioning_mode=conditioning_mode,
    )
    metadata = dict(trajectory.metadata)
    metadata["source_artifact"] = str(Path(path))
    return ActionTrajectory(
        positions=trajectory.positions,
        timestamps=trajectory.timestamps,
        coordinate_frame=trajectory.coordinate_frame,
        units=trajectory.units,
        metadata=metadata,
    )


def load_cosmos_absolute_poses(
    path: str | Path,
    *,
    duration: float | None = None,
    sample_rate: float | None = None,
    conditioning_mode: str = "first_frame_policy",
) -> ActionTrajectory:
    """Load Cosmos RoboLab ``xyz + quaternion_xyzw`` policy output.

    The official RoboLab service converts its internal frame-wise rot6d action
    deltas back to absolute end-effector poses.  We move those poses into the
    initial-EE coordinate frame before scoring.
    """

    artifact_path = Path(path)
    values = load_action_array(artifact_path)
    if values.ndim != 2 or values.shape[1] < 7:
        raise ValueError(f"Cosmos absolute poses must have shape (T, >=7), got {values.shape}")
    positions = ee_pose_xyzw_to_initial_frame(values[:, :7])
    timestamps = timestamps_for_samples(len(positions), duration=duration, sample_rate=sample_rate)
    return ActionTrajectory(
        positions=positions,
        timestamps=timestamps,
        metadata={
            "provider": "cosmos3",
            "conditioning_mode": conditioning_mode,
            "source_artifact": str(artifact_path),
            "raw_action_dim": int(values.shape[1]),
            "source_representation": "absolute_xyz_quaternion_xyzw",
        },
    )


def load_cosmos_joint_positions(
    path: str | Path,
    *,
    fk_ee_trajectory_path: str | Path | None,
    duration: float | None = None,
    sample_rate: float | None = None,
    conditioning_mode: str = "first_frame_policy",
) -> ActionTrajectory:
    """Load Policy-DROID joint targets through a verified FK artifact.

    The released checkpoint returns seven absolute Franka joint positions and
    a gripper value, not Cartesian EE poses.  A robot-specific forward-
    kinematics result is therefore mandatory.  Its JSON sidecar must bind the
    EE poses to the exact raw action file by SHA-256 so trajectories from a
    different run cannot be paired accidentally.
    """

    artifact_path = Path(path)
    values = load_action_array(artifact_path)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"Cosmos Policy-DROID joint actions must have shape (T,8), got {values.shape}")
    if fk_ee_trajectory_path is None:
        raise ValueError(
            "cosmos_joint_pos requires fk_ee_trajectory_path generated with the matching "
            "robot URDF, joint order, and end-effector link"
        )

    fk_path = Path(fk_ee_trajectory_path)
    if fk_path.suffix.lower() != ".npz":
        raise ValueError("Cosmos FK artifact must be an .npz file containing ee_trajectory")
    with np.load(fk_path, allow_pickle=False) as payload:
        if "ee_trajectory" not in payload:
            raise ValueError("Cosmos FK artifact must contain ee_trajectory")
        ee_trajectory = np.asarray(payload["ee_trajectory"], dtype=np.float64)
    if len(ee_trajectory) != len(values):
        raise ValueError(
            f"Cosmos FK trajectory length {len(ee_trajectory)} does not match actions {len(values)}"
        )

    sidecar_path = fk_path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise ValueError(f"Cosmos FK artifact is missing provenance sidecar: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if sidecar.get("source_actions_sha256") != expected_sha256:
        raise ValueError("Cosmos FK artifact does not match the raw action file SHA-256")
    for required in ("urdf_path", "target_link_name", "joint_order"):
        if not sidecar.get(required):
            raise ValueError(f"Cosmos FK provenance is missing {required!r}")

    positions = ee_pose_xyzw_to_initial_frame(ee_trajectory)
    timestamps = timestamps_for_samples(len(positions), duration=duration, sample_rate=sample_rate)
    return ActionTrajectory(
        positions=positions,
        timestamps=timestamps,
        metadata={
            "provider": "cosmos3-policy-droid",
            "conditioning_mode": conditioning_mode,
            "source_artifact": str(artifact_path),
            "source_representation": "absolute_joint_position7_gripper",
            "fk_ee_trajectory_path": str(fk_path),
            "fk_urdf_path": sidecar["urdf_path"],
            "fk_target_link_name": sidecar["target_link_name"],
            "fk_joint_order": sidecar["joint_order"],
        },
    )


def _flatten_lingbot_actions(actions: np.ndarray, *, channel_first: bool) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    while values.ndim > 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim == 2:
        return values.T if channel_first else values
    if values.ndim == 3:
        if not channel_first:
            raise ValueError("3D LingBot arrays require channel_first=True")
        channels, frames, actions_per_frame = values.shape
        return values.transpose(1, 2, 0).reshape(frames * actions_per_frame, channels)
    raise ValueError(f"unsupported LingBot action shape: {values.shape}")


def load_lingbot_actions(
    path: str | Path,
    *,
    duration: float | None = None,
    sample_rate: float | None = None,
    translation_scale: float | None = None,
    channel_first: bool = False,
    conditioning_mode: str = "first_frame_policy",
    rotation_to_initial_ee: np.ndarray | None = None,
) -> ActionTrajectory:
    """Convert LingBot-VA EEF deltas to the common Cartesian path."""

    values = _flatten_lingbot_actions(load_action_array(path), channel_first=channel_first)
    if values.shape[1] < 3:
        raise ValueError(f"LingBot actions need at least xyz channels, got {values.shape}")
    if translation_scale is None:
        raise ValueError(
            "LingBot translation_scale (raw action units per metre) must be set explicitly; "
            "the released controller commands must not be silently labelled as metres"
        )
    translation = values[:, :3]
    coordinate_frame = "lingbot_controller"
    if rotation_to_initial_ee is not None:
        rotation = np.asarray(rotation_to_initial_ee, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation_to_initial_ee must have shape (3,3), got {rotation.shape}")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-5
        ):
            raise ValueError("rotation_to_initial_ee must be a proper orthonormal rotation")
        translation = (rotation @ translation.T).T
        coordinate_frame = "initial_ee"
    positions = integrate_translation_deltas(translation, scale=translation_scale)
    timestamps = timestamps_for_samples(len(positions), duration=duration, sample_rate=sample_rate)
    return ActionTrajectory(
        positions=positions,
        timestamps=timestamps,
        coordinate_frame=coordinate_frame,
        metadata={
            "provider": "lingbot-va",
            "conditioning_mode": conditioning_mode,
            "source_artifact": str(Path(path)),
            "raw_action_dim": int(values.shape[1]),
            "translation_semantics": "fixed-frame delta",
            "translation_scale": float(translation_scale),
            "frame_mapping": (
                "explicit_controller_to_initial_ee_rotation"
                if rotation_to_initial_ee is not None
                else "unmapped_controller_frame"
            ),
        },
    )
