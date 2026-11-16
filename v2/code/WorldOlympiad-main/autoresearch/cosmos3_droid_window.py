"""Frozen Cosmos3-Nano-Policy-DROID single-window adapter.

The adapter has no simulator dependency.  Isaac, PyBullet or another native
backend may provide the conditioning image and current robot state, while this
module owns only the official 33-frame / 32-action Cosmos invocation contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def recorded_prompt_format(windows: list[dict[str, Any]]) -> bool | None:
    """Read the actual common format; historical records may be unknown."""
    values = [row.get('model_metadata', {}).get('format_prompt_as_json') for row in windows]
    if not values or all(value is None for value in values):
        return None
    if any(not isinstance(value, bool) for value in values) or len(set(values)) != 1:
        raise RuntimeError('Cosmos prompt format changed or lost provenance between windows')
    return values[0]


def restore_decoded_droid_panels(
    video: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Restore the released 640x360 + 2x(320x180) DROID panel geometry.

    Cosmos decodes the 640x540 conditioning canvas at either 540 or 528 rows.
    The 528-row output is a bottom crop: the 360-row wrist panel is unchanged
    and the exterior row retains its top 168 of 180 rows.  Splitting at 2/3 of
    528 incorrectly includes eight wrist rows and vertically shifts both
    exterior views.  Preserve all decoded pixels and restore only the missing
    exterior bottom rows with black padding.
    """

    values = np.asarray(video)
    if (
        values.ndim != 4
        or values.shape[-1] != 3
        or values.shape[2] != 640
        or values.shape[1] not in {528, 540}
    ):
        raise ValueError(f"unexpected decoded DROID video shape: {values.shape}")
    wrist = values[:, :360]
    exterior = values[:, 360:]
    missing_bottom = 180 - exterior.shape[1]
    if missing_bottom < 0:
        raise ValueError("decoded DROID exterior panel is taller than 180 rows")
    if missing_bottom:
        exterior = np.pad(
            exterior,
            ((0, 0), (0, missing_bottom), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    left = exterior[:, :, :320]
    right = exterior[:, :, 320:]
    return wrist, left, right, {
        "decoded_height": int(values.shape[1]),
        "wrist_height": 360,
        "decoded_exterior_height": int(values.shape[1] - 360),
        "restored_exterior_height": 180,
        "exterior_bottom_padding": int(missing_bottom),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".partial")
    staging.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    staging.replace(path)


def validate_resume_identity(
    progress: dict[str, Any],
    *,
    official_runtime_task: str,
    official_task_file_sha256: str,
    worldir_hash: str,
    prompt: str,
    seed_schedule_seed: int,
    runtime_identity_sha256: str,
) -> None:
    """Fail closed before replaying an interrupted native rollout."""

    expected = {
        "schema_version": "cosmos_worldir_isaac_progress.v2",
        "official_runtime_task": official_runtime_task,
        "official_task_file_sha256": official_task_file_sha256,
        "worldir_hash": worldir_hash,
        "prompt": prompt,
        "seed_schedule_seed": seed_schedule_seed,
        "runtime_identity_sha256": runtime_identity_sha256,
    }
    changed = {
        key: {"expected": value, "actual": progress.get(key)}
        for key, value in expected.items()
        if progress.get(key) != value
    }
    if changed:
        raise RuntimeError(f"Cosmos resume identity differs: {changed}")


def checkpoint_frames_sha256(frames: list[dict[str, Any]]) -> str:
    """Return the content id for a committed trajectory prefix."""

    payload = json.dumps(
        frames, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_resume_ledger(
    progress: dict[str, Any],
    trajectory: dict[str, Any],
    *,
    official_max_steps: int,
) -> list[dict[str, Any]]:
    """Validate one commit ledger and return its trusted trajectory prefix.

    A crash can occur after the next window's trajectory is written but before
    its progress commit. Only that explicitly sourced, uncommitted suffix may
    be discarded; missing or altered committed frames fail closed.
    """

    frames = trajectory.get("frames")
    windows = progress.get("windows")
    if not isinstance(frames, list) or not isinstance(windows, list):
        raise RuntimeError("Cosmos resume ledger is missing frames or windows")
    completed = int(progress.get("completed_windows", -1))
    current_count = int(progress.get("current_frame_count", -1))
    if completed < 0 or current_count < 1 or len(windows) != completed:
        raise RuntimeError("Cosmos resume ledger counts are inconsistent")
    if official_max_steps <= 0:
        raise RuntimeError("Cosmos official task step budget is invalid")
    expected_count = 1
    seed_schedule = np.random.default_rng(int(progress.get("seed_schedule_seed", 0))).integers(
        0, 2**31, size=completed
    )
    short_windows: list[int] = []
    for expected_index, record in enumerate(windows):
        if not isinstance(record, dict) or record.get("window_index") != expected_index:
            raise RuntimeError("Cosmos resume window indices are not contiguous")
        executed = int(record.get("executed_action_count", -1))
        remaining = official_max_steps - (expected_count - 1)
        planned = min(32, remaining)
        if (
            remaining <= 0
            or record.get("conditioning_frame") != expected_count - 1
            or record.get("planned_action_count") != planned
            or executed < 1
            or executed > planned
        ):
            raise RuntimeError("Cosmos resume window action count is invalid")
        if record.get("request_seed") != int(seed_schedule[expected_index]):
            raise RuntimeError("Cosmos resume window request seed changed")
        if executed < planned:
            short_windows.append(expected_index)
        for field in ("executed_actions_sha256", "metadata_sha256"):
            value = record.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise RuntimeError("Cosmos resume window artifact digest is missing")
        expected_count += executed
    terminated = bool(progress.get("terminated_early"))
    completed_task = bool(progress.get("task_completed"))
    if terminated and completed_task:
        raise RuntimeError("Cosmos resume ledger has conflicting terminal states")
    terminal = terminated or completed_task
    if short_windows and (
        short_windows != [completed - 1] or not terminal
    ):
        raise RuntimeError("Cosmos resume ledger has a non-terminal short window")
    action_count = expected_count - 1
    if action_count > official_max_steps:
        raise RuntimeError("Cosmos resume ledger exceeds the official timeout")
    if terminal:
        terminal_field = "termination_frame" if terminated else "completion_frame"
        if progress.get(terminal_field) != action_count:
            raise RuntimeError("Cosmos resume terminal frame differs from its action count")
    if current_count != expected_count or len(frames) < current_count:
        raise RuntimeError("Cosmos resume trajectory/window counts differ")
    if len(frames) > current_count and (
        terminal or action_count >= official_max_steps
    ):
        raise RuntimeError("Cosmos terminal checkpoint has an uncommitted suffix")
    committed = frames[:current_count]
    if checkpoint_frames_sha256(committed) != progress.get("checkpoint_frames_sha256"):
        raise RuntimeError("Cosmos committed trajectory prefix digest changed")
    for expected_frame, frame in enumerate(frames):
        if (
            frame.get("frame") != expected_frame
            or frame.get("sim_step") != expected_frame * 8
            or abs(float(frame.get("time_s", -1.0)) - expected_frame / 15.0) > 1e-9
        ):
            raise RuntimeError("Cosmos trajectory frame clock is not contiguous")
    for frame in frames[current_count:]:
        source = str(frame.get("policy_source", ""))
        if not source.startswith("cosmos_window_"):
            raise RuntimeError("Cosmos uncommitted trajectory suffix has no window source")
        try:
            source_index = int(source.removeprefix("cosmos_window_"))
        except ValueError as exc:
            raise RuntimeError("Cosmos uncommitted trajectory suffix is malformed") from exc
        if source_index != completed:
            raise RuntimeError("Cosmos uncommitted suffix is not the next window")
    return committed


def validate_replayed_state(
    expected: dict[str, Any], actual: dict[str, Any], *, atol: float = 2e-5
) -> None:
    """Compare every recorded dynamic state at a resume boundary."""

    if set(expected) != set(actual):
        raise RuntimeError("Cosmos replay state fields differ")
    quaternion_fields = {
        "robot_base_orientation_xyzw",
        "cup_orientation_xyzw",
        "tray_orientation_xyzw",
    }
    changed: list[str] = []
    for name in sorted(expected):
        left = np.asarray(expected[name], dtype=np.float64)
        right = np.asarray(actual[name], dtype=np.float64)
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            changed.append(name)
            continue
        if name in quaternion_fields:
            matches = np.allclose(left, right, atol=atol, rtol=0.0) or np.allclose(
                left, -right, atol=atol, rtol=0.0
            )
        else:
            matches = np.allclose(left, right, atol=atol, rtol=0.0)
        if not matches:
            changed.append(name)
    if changed:
        raise RuntimeError(f"Isaac deterministic resume state diverged: {changed}")


def _validate_executable_action_timeline(actions: np.ndarray) -> None:
    values = np.asarray(actions)
    if values.shape != (33, 8) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid Cosmos executable action timeline: {values.shape}")
    if not 0.0 <= float(values[0, -1]) <= 1.0:
        raise RuntimeError("initial simulator gripper state lies outside [0,1]")
    if not np.isin(values[1:, -1], (0.0, 1.0)).all():
        raise RuntimeError("Cosmos future executable gripper commands are not binary")


def run_server_window(
    *,
    window_root: Path,
    input_image: Path,
    joints: np.ndarray,
    gripper: float,
    prompt: str,
    host: str,
    port: int,
    request_index: int,
    seed_schedule_seed: int,
    expected_request_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Query one already-loaded official RoboLab policy server window.

    The released server returns 32 future actions.  This adapter applies the
    same gripper threshold as RoboLab's Cosmos3 client and prepends the live
    simulator state so both serving paths expose the same 33x8 timeline.
    """

    from PIL import Image
    from openpi_client import websocket_client_policy

    image = np.asarray(Image.open(input_image).convert("RGB"), dtype=np.uint8)
    if image.shape != (540, 640, 3):
        raise RuntimeError(f"Cosmos DROID input must be 540x640 RGB, got {image.shape}")
    joint_values = np.asarray(joints, dtype=np.float32)
    if joint_values.shape != (7,) or not np.isfinite(joint_values).all():
        raise RuntimeError(f"invalid DROID joint state: {joint_values.shape}")
    if not np.isfinite(gripper) or not 0.0 <= float(gripper) <= 1.0:
        raise RuntimeError("DROID measured gripper state must lie within [0,1]")

    class _LongInferencePolicy(websocket_client_policy.WebsocketClientPolicy):
        def _wait_for_server(self):
            # The official policy handler is synchronous. It cannot service
            # websocket pings while generating; disable client keepalives too.
            import websockets.sync.client
            from openpi_client import msgpack_numpy
            headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
            connection = websockets.sync.client.connect(
                self._uri, compression=None, max_size=None,
                additional_headers=headers, ping_interval=None, ping_timeout=None,
            )
            return connection, msgpack_numpy.unpackb(connection.recv())

    client = _LongInferencePolicy(host, int(port))
    try:
        response = client.infer(
            {
                "prompt": prompt,
                "observation/image": image,
                "observation/joint_position": joint_values,
                "observation/gripper_position": np.asarray([gripper], dtype=np.float32),
            }
        )
    finally:
        connection = getattr(client, "_ws", None)
        if connection is not None:
            connection.close()
    actual_request_index = response.get("worldolympiad_request_index")
    if response.get('worldolympiad_diagnostic_only'):
        raise RuntimeError('fixed-seed diagnostic server cannot supply a formal closed-loop rollout')
    actual_request_seed = response.get("worldolympiad_request_seed")
    if actual_request_index != int(request_index) or actual_request_seed != int(
        expected_request_seed
    ):
        raise RuntimeError(
            "Cosmos server request schedule differs: "
            f"expected index/seed {request_index}/{expected_request_seed}, "
            f"got {actual_request_index}/{actual_request_seed}"
        )
    effective_prompt_format = response.get('worldolympiad_format_prompt_as_json')
    if not isinstance(effective_prompt_format, bool):
        raise RuntimeError('Cosmos server omitted its effective prompt format; restart with the current launcher')
    future_raw = np.asarray(response["action"], dtype=np.float32)
    if future_raw.shape != (32, 8) or not np.isfinite(future_raw).all():
        raise RuntimeError(f"unexpected Cosmos server action shape: {future_raw.shape}")
    future = future_raw.copy()
    future[:, -1] = (future[:, -1] > 0.5).astype(future.dtype)
    initial = np.concatenate(
        [joint_values, np.asarray([gripper], dtype=np.float32)]
    )[None]
    actions = np.concatenate([initial, future], axis=0)
    raw_actions = np.concatenate([initial, future_raw], axis=0)
    _validate_executable_action_timeline(actions)

    model_root = window_root / "model"
    model_root.mkdir(parents=True)
    np.save(model_root / "actions.npy", actions, allow_pickle=False)
    np.save(model_root / "actions_raw.npy", raw_actions, allow_pickle=False)
    decoded_video = response.get("video")
    video_shape = None
    if decoded_video is not None:
        video = np.asarray(decoded_video, dtype=np.uint8)
        if (
            video.ndim != 4
            or video.shape[0] != 33
            or video.shape[-1] != 3
            or video.shape[2] != 640
            or video.shape[1] not in {528, 540}
        ):
            raise RuntimeError(f"unexpected Cosmos server video shape: {video.shape}")
        np.save(model_root / "droid_concat_rollout.npy", video, allow_pickle=False)
        video_shape = list(video.shape)
    metadata = {
        "schema_version": "cosmos3_droid_server_window.v1",
        "action_source": "official_robolab_websocket_server",
        "server": f"{host}:{int(port)}",
        "request_index": int(request_index),
        "request_seed": int(expected_request_seed),
        "seed_schedule_seed": int(seed_schedule_seed),
        "server_seed_contract": "numpy.default_rng(seed_schedule_seed), one draw per request",
        "conditioning_image": os.path.relpath(
            input_image.resolve(), window_root.resolve()
        ),
        "decoded_video_shape": video_shape,
        "format_prompt_as_json": effective_prompt_format,
        "prompt_format_evidence": "server_transform_attestation",
    }
    write_json_atomic(model_root / "actions.json", metadata)
    return actions, raw_actions, metadata
