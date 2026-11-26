"""CPU-side LingBot-VA RoboTwin transport and Aloha EE-action decoder.

The policy API returns absolute end-effector targets.  This module deliberately
contains no checkpoint/model loading; the native adapter owns the simulator
clock and consumes decoded actions through RoboTwin's EE-action protocol.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


def _unit_quaternion(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("EE quaternion must contain four finite xyzw values")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("EE quaternion must be nonzero")
    return q / norm


def _quat_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def decode_action(raw, initial):
    """Decode one 16-value dual-arm relative EE action into absolute xyzw targets.

    The wire vector is [left xyz, left qxyzw, left gripper, right ...].
    Translation is relative to the observed EE pose, orientation is composed
    with the observed quaternion, and gripper values are absolute.
    """
    raw = np.asarray(raw, dtype=float)
    initial = np.asarray(initial, dtype=float)
    if raw.shape != (16,) or initial.shape != (16,) or not np.isfinite(raw).all() or not np.isfinite(initial).all():
        raise ValueError("LingBot-VA EE actions and initial state must be finite 16-vectors")
    decoded = raw.copy()
    for start in (0, 8):
        decoded[start:start + 3] = initial[start:start + 3] + raw[start:start + 3]
        q0 = _unit_quaternion(initial[start + 3:start + 7])
        dq = _unit_quaternion(raw[start + 3:start + 7])
        decoded[start + 3:start + 7] = _unit_quaternion(_quat_multiply(q0, dq))
    return decoded


def executable_actions(result):
    """Validate model response and return its EE-action sequence."""
    if not isinstance(result, dict) or result.get("action_type") != "ee":
        raise RuntimeError("LingBot-VA response must use action_type='ee'")
    actions = np.asarray(result.get("actions"), dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 16 or not len(actions) or not np.isfinite(actions).all():
        raise RuntimeError("LingBot-VA response must contain finite 16-D dual-arm EE actions")
    count = result.get("replan_steps", len(actions))
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(actions):
        raise RuntimeError("LingBot-VA response has an invalid replanning prefix")
    return actions[:count].copy()


class LingBotVAClient:
    """Small HTTP client for the isolated LingBot inference service."""
    def __init__(self, port=28000, timeout=300):
        self.url = f"http://127.0.0.1:{int(port)}/infer"
        self.timeout = float(timeout)

    def infer(self, observation, instruction):
        if not isinstance(observation, dict) or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("inference requires an observation mapping and nonempty instruction")
        payload = {"observation": observation, "instruction": instruction}
        request = Request(self.url, data=json.dumps(payload, allow_nan=False).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        return executable_actions(result)


# Native RoboTwin helpers shared by the configured bridge.
CAPTURE_CAMERAS = {
    "head": "head_camera",
    "left_wrist": "left_camera",
    "right_wrist": "right_camera",
}


def setup_native_task(task, config):
    """Apply the configured renderer denoiser during RoboTwin task setup.

    RoboTwin task setup may overwrite SAPIEN's global denoiser setting. Wrap
    that setter for the duration of setup so the deployment profile wins while
    preserving the original function object afterward.
    """
    try:
        import sapien
    except ImportError:
        task.setup_demo(**config)
        return
    render = sapien.render
    original = render.set_ray_tracing_denoiser
    configured = os.environ.get("ROBOTWIN_RT_DENOISER")
    if configured is None:
        task.setup_demo(**config)
        return

    def set_configured_denoiser(_mode):
        return original(configured)

    render.set_ray_tracing_denoiser = set_configured_denoiser
    try:
        task.setup_demo(**config)
    finally:
        render.set_ray_tracing_denoiser = original


def restore_initial_object_poses(task, plan):
    """Restore dynamic native objects to the plan's authoritative initial poses."""
    import sapien

    handles = getattr(task, "wo_handles", {})
    for item in plan.get("objects", []):
        if item.get("fixed", False):
            continue
        body = handles.get(item.get("object_id"))
        pose = item.get("pose_rt")
        if body is None or not isinstance(pose, dict):
            continue
        body.set_root_pose(sapien.Pose(pose["p"], pose["q_wxyz"]))
        if hasattr(body, "set_root_velocity"):
            body.set_root_velocity(np.zeros(6))


def add_camera_frames(videos, observation):
    """Append the real head and both wrist RGB views from one observation."""
    camera_data = observation["observation"]
    for name, camera in CAPTURE_CAMERAS.items():
        videos[name].add(np.asarray(camera_data[camera]["rgb"]))


def write_camera_videos(videos, out, primary_arm):
    """Write all camera streams and preserve a legacy primary-wrist alias."""
    if primary_arm not in {"left", "right"}:
        raise ValueError("primary_arm must be left or right")
    out = Path(out)
    for name in CAPTURE_CAMERAS:
        videos[name].write()
    source_name = f"{primary_arm}_wrist"
    source = videos[source_name].path
    alias_path = out / "videos" / "wrist.mp4"
    shutil.copy2(source, alias_path)
    shutil.copy2(out / f"initial_{source_name}.png", out / "initial_wrist.png")
    mapping = {name: {"camera": camera, "source_video": f"videos/{name}.mp4"}
               for name, camera in CAPTURE_CAMERAS.items()}
    mapping["wrist_alias"] = {"camera": f"{primary_arm}_camera",
                              "source_video": f"videos/{source_name}.mp4"}
    return mapping


# Short alias used by runtime integrations.
Client = LingBotVAClient
