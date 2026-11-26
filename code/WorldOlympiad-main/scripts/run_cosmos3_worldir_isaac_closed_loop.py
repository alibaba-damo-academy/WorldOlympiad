#!/usr/bin/env python3
"""Chunk-closed-loop Cosmos3 DROID rollout in native Isaac Sim 5.1.

WorldIR/USD is the scene authority. Isaac PhysX executes Cosmos actions and
native Isaac RTX cameras return fresh DROID RGB observations every policy
window; no SceneCode/Blender observation bridge is used.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback
from types import SimpleNamespace

import cv2  # RoboLab requires OpenCV before Isaac Lab.
from isaaclab.app import AppLauncher


PHYSICS_HZ = 120
ACTION_HZ = 15
PHYSICS_STEPS_PER_ACTION = PHYSICS_HZ // ACTION_HZ
POLICY_ACTION_CHUNK = 32
if PHYSICS_HZ % ACTION_HZ:
    raise RuntimeError("Cosmos control rate must divide the native physics rate")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--world-repo", type=Path, required=True)
parser.add_argument("--world-usd", type=Path, required=True)
parser.add_argument("--worldir-plan", type=Path, required=True)
parser.add_argument("--reference-trajectory", type=Path, required=True)
parser.add_argument("--robot-usd", type=Path, required=True)
parser.add_argument("--robot-usd-sha256", required=True)
parser.add_argument("--runtime-identity", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument(
    "--official-robolab-task",
    default=None,
    help="Official RoboLab task binding as robolab/tasks/.../file.py:TaskClass.",
)
parser.add_argument("--official-robolab-task-sha256", default=None)
parser.add_argument("--task-graph", type=Path)
parser.add_argument("--evaluation-input-manifest", type=Path)
parser.add_argument("--prompt", required=True)
parser.add_argument(
    "--seed-schedule-seed",
    type=int,
    default=0,
    help="Seed for the deterministic per-window Cosmos request-seed schedule.",
)
parser.add_argument(
    "--policy-server-host",
    required=True,
    help="Already-loaded official RoboLab WebSocket policy server.",
)
parser.add_argument("--policy-server-port", type=int, default=28000)
parser.add_argument('--policy-model', choices=['cosmos3', 'gr00t'], default='cosmos3')
parser.add_argument("--resume", action="store_true")
parser.add_argument('--robolab-sim-config',type=Path,default=Path(__file__).resolve().parents[1]/'deployment/seetacloud/robolab_simulation_profile.json')
parser.add_argument('--legacy-cpu-physics',action='store_true',help='Reproduce historical CPU physics artifacts only')
parser.add_argument(
    "--observation-only",
    action="store_true",
    help="Render and certify frame-zero DROID views, then exit before loading Cosmos.",
)
parser.add_argument(
    "--replay-actions-from",
    type=Path,
    help="Replay certified per-window executed_actions.npy files instead of invoking Cosmos.",
)
AppLauncher.add_app_launcher_args(parser)
ARGS, _ = parser.parse_known_args()
POLICY_ACTION_CHUNK = 8 if ARGS.policy_model == 'gr00t' else 32
if ARGS.policy_model == 'gr00t' and ARGS.resume:
    raise ValueError('GR00T resume is not yet certified; use a fresh output directory')
ARGS.headless = True
ARGS.enable_cameras = True
if ARGS.legacy_cpu_physics: ARGS.robolab_sim_config=None
ARGS.device = json.loads(ARGS.robolab_sim_config.read_text())['sim']['device'] if ARGS.robolab_sim_config else 'cpu'
APP = AppLauncher(ARGS).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402
from robolab.robots.droid import DroidCfg  # noqa: E402
from robolab.variations.camera import (  # noqa: E402
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
)
from robolab.core.task.task_utils import load_task_from_file  # noqa: E402
from robolab.core.world.world_state import WorldState  # noqa: E402


WORLD_REPO = ARGS.world_repo.resolve()
if str(WORLD_REPO) not in sys.path:
    sys.path.insert(0, str(WORLD_REPO))
if str(WORLD_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(WORLD_REPO / "scripts"))

from autoresearch.robotics_views import (  # noqa: E402
    compose_cosmos_droid_view,
)
from autoresearch.robotics_droid_control import (  # noqa: E402
    executable_robolab_gripper_target,
    normalize_robolab_gripper_position,
)
from autoresearch.cosmos3_droid_window import (  # noqa: E402
    checkpoint_frames_sha256,
    run_server_window,
    recorded_prompt_format,
    validate_resume_identity,
    validate_resume_ledger,
    validate_replayed_state,
    write_json_atomic,
)
from autoresearch.wam_cosmos import (  # noqa: E402
    FROZEN_CHECKPOINT_REVISION,
    FROZEN_FRAMEWORK_REVISION,
    FROZEN_MODEL_CONFIG_SHA256,
    FROZEN_WAN_VAE_SHA256,
    SOURCE_MODEL,
)
from autoresearch.wam_adapter_contracts import sha256_file  # noqa: E402
from autoresearch.cosmos_simulation_profile import build_simulation_config
from autoresearch.cosmos_scene_state import live_object_records, dynamic_state_checkpoint
from autoresearch.cosmos_task_graph_predicates import NativeTaskGraphPredicates, protocol

from autoresearch.cosmos_droid_native import (
    _camera_bgr, _collision_filter_paths, _contact_active, _exterior_camera, _quat_multiply_wxyz, _render_observation, _rotate_wxyz, _wrist_camera
)

_run_server_window = run_server_window
_write_json = write_json_atomic


def _load_official_runtime_task(
    binding: str | None,
    expected_sha256: str | None,
    *,
    task_graph: Path | None = None,
    trajectory: dict | None = None,
) -> dict:
    if binding == "worldolympiad/task_graph_native_episode":
        if task_graph is None or not expected_sha256:
            raise ValueError("native task-graph runtime requires a bound graph")
        if sha256_file(task_graph) != expected_sha256:
            raise ValueError("task graph changed after input preparation")
        native = protocol()
        return {"binding": binding, "task_file": task_graph,
            "task_file_sha256": expected_sha256, "task_class": "TaskGraphNativeEpisode",
            "episode_length_s": native["episode_length_s"],
            "max_steps": int(native["episode_length_s"] * ACTION_HZ),
            "success_func": None, "success_params": {},
            "success_function": "worldolympiad.native_task_graph_predicates",
            "success_function_name": "native_task_graph_predicates",
            "termination_policy": "native_predicates_or_independent_episode_timeout",
            "native_protocol_sha256": native["sha256"]}
    if not binding or not expected_sha256:
        raise ValueError("official RoboLab task binding and digest are required")
    try:
        relative_file, class_name = binding.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError("official RoboLab task must be file.py:TaskClass") from exc
    import robolab

    robolab_root = Path(robolab.__file__).resolve().parent.parent
    task_file = (robolab_root / relative_file).resolve()
    try:
        task_file.relative_to(robolab_root)
    except ValueError as exc:
        raise ValueError("official RoboLab task escapes the installed repository") from exc
    if not task_file.is_file():
        raise FileNotFoundError(task_file)
    actual_sha256 = sha256_file(task_file)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "official RoboLab task digest changed: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    classes = load_task_from_file(str(task_file), allow_multiple=True)
    matches = [item for item in classes if item.__name__ == class_name]
    if len(matches) != 1:
        raise ValueError(f"RoboLab task class not found: {binding}")
    task_class = matches[0]
    terminations = task_class.terminations()
    success = getattr(terminations, "success", None)
    function_name = getattr(getattr(success, "func", None), "__name__", "")
    if success is None or function_name not in {
        "object_in_container",
        "stacked",
        "object_on_center",
        "object_on_top",
    }:
        raise ValueError(
            f"WorldIR adapter does not support RoboLab success function {function_name!r}"
        )
    params = dict(success.params or {})
    if function_name == "object_in_container":
        params["object"] = "cup"
        params["container"] = "tray"
        if params.get("require_gripper_detached"):
            params["gripper_name"] = "gripper"
    elif function_name == "stacked":
        params["objects"] = ["tray", "cup"]
        params["order"] = "bottom_to_top"
    else:
        params["object"] = "cup"
        params["reference_object"] = "tray"
        if params.get("require_contact_with") is True:
            params["require_contact_with"] = "tray"
        if params.get("require_gripper_detached"):
            params["gripper_name"] = "gripper"
    episode_length_s = float(task_class.episode_length_s)
    if episode_length_s <= 0:
        raise ValueError("official RoboLab task has no positive episode length")
    control_period_s = 1.0 / ACTION_HZ
    return {
        "binding": binding,
        "task_file": task_file,
        "task_file_sha256": actual_sha256,
        "task_class": class_name,
        "episode_length_s": episode_length_s,
        "max_steps": int(math.ceil(episode_length_s / control_period_s)),
        "success_func": success.func,
        "success_params": params,
        "success_function": f"{success.func.__module__}.{success.func.__name__}",
        "success_function_name": function_name,
        "termination_policy": "official_task_success_or_episode_horizon",
    }


class _WorldIRTaskWorld(WorldState):
    """Expose the live WorldIR bodies to RoboLab's official task predicate."""

    def __init__(self, *, stage, cup, tray, static_tray_pose):
        # Constructed before sim.reset(), when RigidObject.device is not initialized.
        self.env = SimpleNamespace(device=torch.device(ARGS.device))
        # This adapter bypasses WorldState.set_world because it does not own a
        # ManagerBasedRLEnv.  Initialize the base predicate caches that its
        # official geometry and stateful predicates read.
        self._local_geometry_cache = {}
        self._predicate_state = {}
        self._stage = stage
        self._cup = cup
        self._tray = tray
        self._static_tray_pose = static_tray_pose
        self._left_contact = False
        self._right_contact = False
        self._container_contact = False

    def set_contacts(self, left: bool, right: bool, container: bool) -> None:
        self._left_contact = left
        self._right_contact = right
        self._container_contact = container

    def _get_prim(self, body_name: str, env_id: int = 0):
        paths = {
            "cup": "/World/WorldIR/Scene/cup",
            "tray": "/World/WorldIR/Scene/tray",
        }
        if body_name not in paths:
            raise ValueError(f"unsupported WorldIR predicate body: {body_name}")
        prim = self._stage.GetPrimAtPath(paths[body_name])
        if not prim.IsValid():
            raise ValueError(f"missing WorldIR predicate prim: {paths[body_name]}")
        return prim

    def get_pose(self, body_name: str, is_relative: bool = True, as_matrix: bool = False, env_id: int | None = None):
        if as_matrix or env_id != 0:
            raise ValueError("WorldIR task predicate supports one live environment")
        if body_name == "cup":
            return (
                self._cup.data.root_link_pos_w[0].detach().clone(),
                self._cup.data.root_link_quat_w[0].detach().clone(),
            )
        if body_name != "tray":
            raise ValueError(f"unsupported WorldIR predicate body: {body_name}")
        if self._tray is not None:
            return (
                self._tray.data.root_link_pos_w[0].detach().clone(),
                self._tray.data.root_link_quat_w[0].detach().clone(),
            )
        pose = self._static_tray_pose
        xyzw = pose["quat_xyzw"]
        return (
            torch.tensor(pose["xyz"], dtype=torch.float32, device=self._cup.device),
            torch.tensor([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=torch.float32, device=self._cup.device),
        )

    def in_contact(self, body1: str, body2: str, force_threshold: float = 0.1, env_id: int | None = None):
        names = {body1, body2}
        if "cup" in names and "gripper" in names:
            return self._left_contact or self._right_contact
        if names == {"cup", "tray"}:
            return self._container_contact
        return False

    def is_supported_on_surface(
        self, body_name: str, surface_name: str, env_id: int | None = None
    ):
        if env_id != 0:
            raise ValueError("WorldIR task predicate supports one live environment")
        if {body_name, surface_name} != {"cup", "tray"}:
            return False
        return self._container_contact


def _wxyz_to_xyzw(values) -> list[float]:
    w, x, y, z = (float(value) for value in values)
    return [x, y, z, w]


def _tensor_list(value) -> list[float]:
    return [float(item) for item in value.detach().cpu().tolist()]


def _body_position(robot: Articulation, name: str) -> list[float]:
    index = robot.data.body_names.index(name)
    return _tensor_list(robot.data.body_pos_w[0, index])


def _rigid_state(asset):
    from autoresearch.floating_runtime import runtime_object_state
    return runtime_object_state(asset, _single_rigid_state)


def _single_rigid_state(asset):
    position = _tensor_list(asset.data.root_link_pos_w[0])
    orientation = _wxyz_to_xyzw(asset.data.root_link_quat_w[0])
    return {"position": position, "link_position": position,
            "orientation_xyzw": orientation, "link_orientation_xyzw": orientation,
            "linear_velocity": _tensor_list(asset.data.root_link_lin_vel_w[0]),
            "angular_velocity": _tensor_list(asset.data.root_link_ang_vel_w[0])}


def _body_states(cup, tray, static_tray_pose, extra_objects):
    target = _rigid_state(tray) if tray is not None else {
        "position": list(static_tray_pose["xyz"]), "link_position": list(static_tray_pose["xyz"]),
        "orientation_xyzw": list(static_tray_pose["quat_xyzw"]),
        "link_orientation_xyzw": list(static_tray_pose["quat_xyzw"]),
        "linear_velocity": [0., 0., 0.], "angular_velocity": [0., 0., 0.]}
    return {"cup": _rigid_state(cup), "tray": target,
            **{name: _rigid_state(asset) for name, asset in (extra_objects or {}).items()}}


def _frame_record(
    *,
    frame_index: int,
    template: dict,
    robot: Articulation,
    joint_target: torch.Tensor,
    cup: RigidObject,
    tray: RigidObject | None,
    static_tray_pose: dict | None,
    left_contact: bool,
    right_contact: bool,
    container_contact: bool,
    source_window: int | None,
    scene_bindings=(),
    extra_objects=None,
) -> dict:
    result = copy.deepcopy(template)
    result["frame"] = frame_index
    result["sim_step"] = frame_index * PHYSICS_STEPS_PER_ACTION
    result["time_s"] = frame_index / ACTION_HZ
    result["phase"] = "initial" if frame_index == 0 else "cosmos_isaac_closed_loop"
    result["policy_source"] = (
        "initial_worldir_state" if source_window is None else f"cosmos_window_{source_window:06d}"
    )

    robot_record = result["robot"]
    robot_record["base_position"] = _tensor_list(robot.data.root_link_pos_w[0])
    robot_record["base_orientation_xyzw"] = _wxyz_to_xyzw(robot.data.root_link_quat_w[0])
    robot_record["joint_names"] = list(robot.data.joint_names)
    robot_record["joint_positions"] = _tensor_list(robot.data.joint_pos[0])
    robot_record["joint_velocities"] = _tensor_list(robot.data.joint_vel[0])
    robot_record["joint_position_target"] = _tensor_list(joint_target[0])
    robot_record["base_linear_velocity"] = _tensor_list(
        robot.data.root_link_lin_vel_w[0]
    )
    robot_record["base_angular_velocity"] = _tensor_list(
        robot.data.root_link_ang_vel_w[0]
    )
    robot_record["end_effector_position"] = _body_position(robot, "base_link")
    base_index = robot.data.body_names.index("base_link")
    robot_record["end_effector_orientation_xyzw"] = _wxyz_to_xyzw(
        robot.data.body_quat_w[0, base_index]
    )
    robot_record["finger_link_positions"] = {
        "left_inner_finger": _body_position(robot, "left_inner_finger"),
        "right_inner_finger": _body_position(robot, "right_inner_finger"),
    }
    finger_id = robot.data.joint_names.index("finger_joint")
    # RoboLab's DROID observation contract reports the measured continuous
    # finger position normalized to [0, 1].  Only executable future commands
    # are thresholded to binary open/close values; quantizing feedback here
    # discards partial closure at a policy-window boundary.
    robot_record["droid_gripper_position"] = normalize_robolab_gripper_position(
        robot.data.joint_pos[0, finger_id].item()
    )

    for name, asset in (("cup", cup),):
        position = _tensor_list(asset.data.root_link_pos_w[0])
        orientation = _wxyz_to_xyzw(asset.data.root_link_quat_w[0])
        result[name]["position"] = position
        result[name]["orientation_xyzw"] = orientation
        result[name]["link_position"] = position
        result[name]["link_orientation_xyzw"] = orientation
        result[name]["linear_velocity"] = _tensor_list(
            asset.data.root_link_lin_vel_w[0]
        )
        result[name]["angular_velocity"] = _tensor_list(
            asset.data.root_link_ang_vel_w[0]
        )
    if tray is not None:
        tray_position = _tensor_list(tray.data.root_link_pos_w[0])
        tray_orientation = _wxyz_to_xyzw(tray.data.root_link_quat_w[0])
    else:
        if static_tray_pose is None:
            raise RuntimeError("static tray pose is missing")
        tray_position = [float(value) for value in static_tray_pose["xyz"]]
        tray_orientation = [
            float(value) for value in static_tray_pose["quat_xyzw"]
        ]
        tray_linear_velocity = [0.0, 0.0, 0.0]
        tray_angular_velocity = [0.0, 0.0, 0.0]
    if tray is not None:
        tray_linear_velocity = _tensor_list(tray.data.root_link_lin_vel_w[0])
        tray_angular_velocity = _tensor_list(tray.data.root_link_ang_vel_w[0])
    # Legacy policy/checkpoint slot aliases the measured target; no extra
    # simulator body is spawned when the declared target is the support table.
    result.setdefault("tray", {})
    result["tray"]["position"] = tray_position
    result["tray"]["orientation_xyzw"] = tray_orientation
    result["tray"]["link_position"] = tray_position
    result["tray"]["link_orientation_xyzw"] = tray_orientation
    result["tray"]["linear_velocity"] = tray_linear_velocity
    result["tray"]["angular_velocity"] = tray_angular_velocity
    contacts = []
    if left_contact:
        contacts.append("left_inner_finger")
    if right_contact:
        contacts.append("right_inner_finger")
    result["robot_cup_contact_links"] = contacts
    result["robot_cup_contact_evidence"] = {
        "left_inner_finger": left_contact,
        "right_inner_finger": right_contact,
        "simultaneous_dual_contact": left_contact and right_contact,
    }
    result["object_container_contact"] = container_contact
    states = _body_states(cup, tray, static_tray_pose, extra_objects)
    result["objects"] = live_object_records(result.get("objects", {}), scene_bindings, states)
    result["scene_state_checkpoint"] = dynamic_state_checkpoint(scene_bindings, states)
    result["scene_state_source"] = "live_native_physx"
    return result


def _frame_replay_state(frame: dict) -> dict[str, list[float]]:
    robot = frame["robot"]
    cup = frame["cup"]
    tray = frame["tray"]
    return {
        "robot_joint_positions": robot["joint_positions"],
        "robot_joint_velocities": robot["joint_velocities"],
        "robot_joint_position_target": robot["joint_position_target"],
        "robot_base_position": robot["base_position"],
        "robot_base_orientation_xyzw": robot["base_orientation_xyzw"],
        "robot_base_linear_velocity": robot["base_linear_velocity"],
        "robot_base_angular_velocity": robot["base_angular_velocity"],
        "cup_position": cup["link_position"],
        "cup_orientation_xyzw": cup["link_orientation_xyzw"],
        "cup_linear_velocity": cup["linear_velocity"],
        "cup_angular_velocity": cup["angular_velocity"],
        "tray_position": tray["link_position"],
        "tray_orientation_xyzw": tray["link_orientation_xyzw"],
        "tray_linear_velocity": tray["linear_velocity"],
        "tray_angular_velocity": tray["angular_velocity"],
        "frame_index": [frame["frame"]],
        "sim_step": [frame["sim_step"]],
        "left_contact": [
            bool(frame["robot_cup_contact_evidence"]["left_inner_finger"])
        ],
        "right_contact": [
            bool(frame["robot_cup_contact_evidence"]["right_inner_finger"])
        ],
        "container_contact": [bool(frame["object_container_contact"])],
        **frame.get("scene_state_checkpoint", {}),
    }


def _live_replay_state(
    robot: Articulation,
    joint_target: torch.Tensor,
    cup: RigidObject,
    tray: RigidObject | None,
    static_tray_pose: dict | None,
    frame_index: int,
    left_contact: bool,
    right_contact: bool,
    container_contact: bool,
    scene_bindings=(),
    extra_objects=None,
) -> dict[str, list[float]]:
    if tray is None:
        if static_tray_pose is None:
            raise RuntimeError("static tray pose is missing")
        tray_position = [float(value) for value in static_tray_pose["xyz"]]
        tray_orientation = [float(value) for value in static_tray_pose["quat_xyzw"]]
        tray_linear_velocity = tray_angular_velocity = [0.0, 0.0, 0.0]
    else:
        tray_position = _tensor_list(tray.data.root_link_pos_w[0])
        tray_orientation = _wxyz_to_xyzw(tray.data.root_link_quat_w[0])
        tray_linear_velocity = _tensor_list(tray.data.root_link_lin_vel_w[0])
        tray_angular_velocity = _tensor_list(tray.data.root_link_ang_vel_w[0])
    return {
        "robot_joint_positions": _tensor_list(robot.data.joint_pos[0]),
        "robot_joint_velocities": _tensor_list(robot.data.joint_vel[0]),
        "robot_joint_position_target": _tensor_list(joint_target[0]),
        "robot_base_position": _tensor_list(robot.data.root_link_pos_w[0]),
        "robot_base_orientation_xyzw": _wxyz_to_xyzw(robot.data.root_link_quat_w[0]),
        "robot_base_linear_velocity": _tensor_list(robot.data.root_link_lin_vel_w[0]),
        "robot_base_angular_velocity": _tensor_list(robot.data.root_link_ang_vel_w[0]),
        "cup_position": _tensor_list(cup.data.root_link_pos_w[0]),
        "cup_orientation_xyzw": _wxyz_to_xyzw(cup.data.root_link_quat_w[0]),
        "cup_linear_velocity": _tensor_list(cup.data.root_link_lin_vel_w[0]),
        "cup_angular_velocity": _tensor_list(cup.data.root_link_ang_vel_w[0]),
        "tray_position": tray_position,
        "tray_orientation_xyzw": tray_orientation,
        "tray_linear_velocity": tray_linear_velocity,
        "tray_angular_velocity": tray_angular_velocity,
        "frame_index": [frame_index],
        "sim_step": [frame_index * PHYSICS_STEPS_PER_ACTION],
        "left_contact": [left_contact],
        "right_contact": [right_contact],
        "container_contact": [container_contact],
        **dynamic_state_checkpoint(scene_bindings,
                                  _body_states(cup, tray, static_tray_pose, extra_objects)),
    }






def _manipuland_motion_diagnostic(
    *, initial_position, previous_position, current_position
) -> dict[str, float]:
    """Measure realized object motion without imposing a rollout terminal."""

    initial = np.asarray(initial_position, dtype=np.float64)
    previous = np.asarray(previous_position, dtype=np.float64)
    current = np.asarray(current_position, dtype=np.float64)
    if any(value.shape != (3,) for value in (initial, previous, current)):
        raise RuntimeError("Isaac manipuland position must contain three values")
    if not all(np.isfinite(value).all() for value in (initial, previous, current)):
        raise RuntimeError("Isaac produced a non-finite manipuland pose")
    return {
        "step_displacement_m": float(np.linalg.norm(current - previous)),
        "drop_below_initial_m": float(initial[2] - current[2]),
        "xy_displacement_from_initial_m": float(
            np.linalg.norm(current[:2] - initial[:2])
        ),
    }














def main() -> None:
    if ARGS.output_dir.exists() and not ARGS.resume:
        raise FileExistsError(f"refusing to overwrite {ARGS.output_dir}")
    if not ARGS.output_dir.exists() and ARGS.resume:
        raise FileNotFoundError(f"cannot resume missing run {ARGS.output_dir}")
    for path in (
        ARGS.world_usd, ARGS.worldir_plan, ARGS.reference_trajectory,
        ARGS.robot_usd, ARGS.runtime_identity,
    ):
        if not path.resolve().exists():
            raise FileNotFoundError(path)
    if ARGS.replay_actions_from is not None and not ARGS.replay_actions_from.resolve().is_dir():
        raise FileNotFoundError(ARGS.replay_actions_from)
    if sha256_file(ARGS.robot_usd.resolve()) != ARGS.robot_usd_sha256:
        raise ValueError("native RoboLab robot USD changed after recipe review")
    runtime_identity_path = ARGS.runtime_identity.resolve()
    runtime_identity = json.loads(runtime_identity_path.read_text(encoding="utf-8"))
    identity = runtime_identity.get("identity")
    if (
        runtime_identity.get("schema_version") != "wam_runtime_identity.v1"
        or not isinstance(identity, dict)
        or identity.get("profile") != ('gr00t_n1.6_droid' if ARGS.policy_model == 'gr00t' else 'cosmos3_nano_droid')
    ):
        raise ValueError("Cosmos runtime identity certificate is invalid")
    runtime_identity_sha256 = str(runtime_identity.get("identity_sha256") or "")
    if len(runtime_identity_sha256) != 64:
        raise ValueError("Cosmos runtime identity digest is invalid")

    output = ARGS.output_dir.resolve()
    simulation_root = output / "simulation"
    capture_root = output / "capture"
    windows_root = capture_root / "windows"
    observations_root = capture_root / "observations"
    simulation_root.mkdir(parents=True, exist_ok=ARGS.resume)
    windows_root.mkdir(parents=True, exist_ok=ARGS.resume)
    reference = json.loads(ARGS.reference_trajectory.read_text(encoding="utf-8"))
    official_task = _load_official_runtime_task(
        ARGS.official_robolab_task,
        ARGS.official_robolab_task_sha256,
        task_graph=ARGS.task_graph,
        trajectory=reference,
    )
    official_max_steps = int(official_task["max_steps"])
    evaluation_window = None
    if ARGS.evaluation_input_manifest:
        from autoresearch.cosmos_evaluation_window import validate, digest
        evaluation_window = json.loads(ARGS.evaluation_input_manifest.read_text())["simulator_state"].get("evaluation_window")
    if evaluation_window is not None:
        validate(evaluation_window)
        official_task["original_episode_max_steps"] = official_max_steps
        official_max_steps = evaluation_window["max_steps"]
        official_task["max_steps"] = official_max_steps
        official_task["episode_length_s"] = evaluation_window["physical_duration_s"]
        official_task["termination_policy"] = evaluation_window["policy"]
    frame_template = reference["frames"][0]
    plan = json.loads(ARGS.worldir_plan.read_text(encoding="utf-8"))
    robot_plan = next(body for body in plan["bodies"] if body["name"] == "robot")
    from autoresearch.cosmos_runtime_bodies import contact_target_body
    tray_plan = contact_target_body(plan, json.loads(ARGS.task_graph.read_text()))
    target_prim_path = "/World/WorldIR/Scene/" + tray_plan["name"]
    graph_verifier = (NativeTaskGraphPredicates(json.loads(ARGS.task_graph.read_text()),
        plan, ARGS.worldir_plan.parent) if official_task["task_class"] == "TaskGraphNativeEpisode" else None)
    robot_xyzw = robot_plan["pose"]["quat_xyzw"]

    sim_cfg = sim_utils.SimulationCfg(
        device="cpu", dt=1.0 / PHYSICS_HZ, render_interval=PHYSICS_STEPS_PER_ACTION,
        gravity=(0.0, 0.0, -9.81),
        use_fabric=True,
        physx=sim_utils.PhysxCfg(
            solver_type=1, min_position_iteration_count=1, max_position_iteration_count=32,
            min_velocity_iteration_count=0, max_velocity_iteration_count=1, enable_ccd=True,
        ),
    )
    if ARGS.robolab_sim_config:
        sim_cfg=build_simulation_config(sim_utils,ARGS.robolab_sim_config)
    sim = SimulationContext(sim_cfg)
    print("ISAAC_STAGE simulation_context_ready", flush=True)
    add_reference_to_stage(str(ARGS.world_usd.resolve()), "/World/WorldIR")
    print("ISAAC_STAGE worldir_reference_ready", flush=True)
    stage = get_current_stage()
    from wo.models.reference_environment import add_reference_ground
    _write_json(simulation_root / "reference_environment.json", add_reference_ground(stage))
    nested = stage.GetPrimAtPath("/World/WorldIR/PhysicsScene")
    if nested.IsValid():
        nested.SetActive(False)
    sim_utils.DomeLightCfg(intensity=350.0, color=(0.85, 0.9, 1.0)).func(
        "/World/DomeLight", sim_utils.DomeLightCfg(intensity=350.0, color=(0.85, 0.9, 1.0))
    )
    sim_utils.DistantLightCfg(intensity=750.0, color=(1.0, 0.93, 0.82)).func(
        "/World/KeyLight", sim_utils.DistantLightCfg(intensity=750.0, color=(1.0, 0.93, 0.82))
    )
    robot_cfg = DroidCfg().robot.replace(prim_path="/World/Robot")
    robot_cfg.spawn.usd_path = str(ARGS.robot_usd.resolve())
    robot_cfg.init_state.pos = tuple(robot_plan["pose"]["xyz"])
    robot_cfg.init_state.rot = (robot_xyzw[3], robot_xyzw[0], robot_xyzw[1], robot_xyzw[2])
    robot = Articulation(robot_cfg)
    print("ISAAC_STAGE robot_articulation_ready", flush=True)
    robot_position = tuple(float(value) for value in robot_plan["pose"]["xyz"])
    robot_rotation_wxyz = (
        float(robot_xyzw[3]),
        float(robot_xyzw[0]),
        float(robot_xyzw[1]),
        float(robot_xyzw[2]),
    )
    left_camera = _exterior_camera(
        OverShoulderLeftCameraCfg().over_shoulder_left_camera,
        "left_camera",
        robot_position,
        robot_rotation_wxyz,
    )
    right_camera = _exterior_camera(
        OverShoulderRightCameraCfg().over_shoulder_right_camera,
        "right_camera",
        robot_position,
        robot_rotation_wxyz,
    )
    wrist_camera = _wrist_camera()
    print("ISAAC_STAGE cameras_ready", flush=True)
    from autoresearch.floating_runtime import create_scene_object, runtime_rigid_paths, FloatingObject
    from autoresearch.cosmos_contact_support import create_link_contact_sensors
    dynamic_bodies = {body['name']: body for body in plan['bodies']
                      if body['name'] != 'robot' and not body['static']}
    runtime_objects = {name: create_scene_object(body) for name, body in dynamic_bodies.items()}
    cup = runtime_objects['cup']
    tray = runtime_objects.get(tray_plan['name'])
    rigid_paths = {name: runtime_rigid_paths(dynamic_bodies[name], asset)
                   for name, asset in runtime_objects.items()}
    static_tray_pose = tray_plan["pose"] if tray is None else None
    scene_bindings = [SimpleNamespace(body_name=body["name"],
        trajectory_object_name=(body.get("metadata") or {}).get("trajectory_object_name", body["name"]),
        dynamic=not bool(body.get("static"))) for body in plan["bodies"]
        if body["name"] != "robot"]
    extra_objects = {name: asset for name, asset in runtime_objects.items()
                     if name not in {'cup', 'tray'}}
    task_world = _WorldIRTaskWorld(
        stage=stage, cup=cup, tray=tray, static_tray_pose=static_tray_pose
    )
    print("ISAAC_STAGE rigid_objects_ready", flush=True)
    container_sensor = create_link_contact_sensors(stage, rigid_paths['cup'],
        rigid_paths.get(tray_plan['name']) or _collision_filter_paths(stage, target_prim_path))
    left_sensor = ContactSensor(ContactSensorCfg(
        prim_path="/World/Robot/Gripper/Robotiq_2F_85/left_inner_finger",
        filter_prim_paths_expr=rigid_paths['cup'],
    ))
    right_sensor = ContactSensor(ContactSensorCfg(
        prim_path="/World/Robot/Gripper/Robotiq_2F_85/right_inner_finger",
        filter_prim_paths_expr=rigid_paths['cup'],
    ))
    # Extra placement subjects (e.g. all rigid contents of a transfer)
    # require their own measured gripper release, not the cup's contact bit.
    release_sensors = {}
    if graph_verifier is not None:
        release_names = {name for identity in graph_verifier.release_subjects
                         for name in graph_verifier.names(identity)} - {"cup"}
        for name in sorted(release_names):
            release_sensors[name] = create_link_contact_sensors(stage, rigid_paths[name], [
                "/World/Robot/Gripper/Robotiq_2F_85/left_inner_finger",
                "/World/Robot/Gripper/Robotiq_2F_85/right_inner_finger"])
    # Covering requires actual load-bearing contact against ALL possible
    # supports, not the legacy target-only aggregate contact bit.
    from autoresearch.cosmos_contact_support import read_contact_support
    support_sensors = {}
    if graph_verifier is not None:
        support_names = {name for goal in graph_verifier.goals
                         if goal['predicate'] == 'covering'
                         for name in graph_verifier.names(goal['subject'])}
        for name in sorted(support_names):
            filters, identities = [], []
            for body in plan['bodies']:
                if body['name'] in {name, 'robot'}:
                    continue
                paths = rigid_paths.get(body['name']) or _collision_filter_paths(
                    stage, '/World/WorldIR/Scene/' + body['prim_name'])
                filters.extend(paths)
                identities.extend([body['name']] * len(paths))
            robot_paths = _collision_filter_paths(stage, '/World/Robot')
            filters.extend(robot_paths)
            identities.extend(['robot'] * len(robot_paths))
            support_sensors[name] = (create_link_contact_sensors(
                stage, rigid_paths[name], filters, track_contact_points=True), identities)
    print("ISAAC_STAGE contact_sensors_ready", flush=True)
    sim.reset()
    for asset in runtime_objects.values():
        if isinstance(asset, FloatingObject):
            asset.initialize()
    print("ISAAC_STAGE simulation_reset", flush=True)
    arm_ids, arm_names = robot.find_joints([f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)
    gripper_ids, _ = robot.find_joints(["finger_joint"], preserve_order=True)
    if arm_names != [f"panda_joint{i}" for i in range(1, 8)]:
        raise RuntimeError(f"unexpected arm joint order: {arm_names}")
    target = robot.data.joint_pos.clone()
    initial_arm = torch.tensor(frame_template["robot"]["joint_positions"][:7], dtype=torch.float32,device=target.device)
    target[0, arm_ids] = initial_arm
    target[0, gripper_ids] = 0.0
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    robot.set_joint_position_target(target)
    for _ in range(16):
        robot.write_data_to_sim(); sim.step(render=_ >= 8); robot.update(sim_cfg.dt)
        cup.update(sim_cfg.dt)
        for asset in extra_objects.values():
            asset.update(sim_cfg.dt)
        if tray is not None:
            tray.update(sim_cfg.dt)
        left_sensor.update(sim_cfg.dt, force_recompute=True)
        right_sensor.update(sim_cfg.dt, force_recompute=True)
        container_sensor.update(sim_cfg.dt, force_recompute=True)

    initial_left = _contact_active(left_sensor)
    initial_right = _contact_active(right_sensor)
    initial_container = _contact_active(container_sensor)
    initial_frame = _frame_record(
        frame_index=0, template=frame_template, robot=robot, joint_target=target,
        cup=cup, tray=tray,
        static_tray_pose=static_tray_pose,
        left_contact=initial_left, right_contact=initial_right,
        container_contact=initial_container, source_window=None,
        scene_bindings=scene_bindings, extra_objects=extra_objects,
    )
    initial_frame["robot_subject_contacts"] = {}
    for name, sensor in release_sensors.items():
        sensor.update(sim_cfg.dt, force_recompute=True)
        initial_frame["robot_subject_contacts"][name] = _contact_active(sensor)
    if graph_verifier is not None:
        graph_verifier.update(initial_frame)
    if ARGS.observation_only:
        preflight_root = capture_root / "preflight"
        preflight_root.mkdir(parents=True, exist_ok=True)
        observation = _render_observation(
            frame_index=0,
            window_root=preflight_root,
            observation_root=observations_root,
            sim=sim,
            wrist=wrist_camera,
            left=left_camera,
            right=right_camera,
        )
        result = {
            "schema_version": "cosmos_worldir_isaac_observation_preflight.v1",
            "status": "complete",
            "camera_contract": "released RoboLab WRIST_LEFT_RIGHT_HEAD DROID rig transformed by the WorldIR robot-root pose at 640x360",
            "worldir_hash": plan["world_hash"],
            "conditioning_image": str(observation),
        }
        _write_json(capture_root / "observation_preflight.json", result)
        print(json.dumps(result, indent=2), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    completed_windows = 0
    window_records = []
    terminated_early = False
    task_completed = False
    completion_frame = None
    completion_evidence = None
    termination_frame = None
    termination_reason = None
    termination_evidence = None
    if ARGS.resume:
        progress = json.loads((capture_root / "capture_progress.json").read_text(encoding="utf-8"))
        from autoresearch.cosmos_evaluation_window import digest
        if progress.get("evaluation_window_sha256") != (digest(evaluation_window) if evaluation_window else None):
            raise RuntimeError("evaluation window changed across resume")
        expected_sim_hash=sha256_file(ARGS.robolab_sim_config) if ARGS.robolab_sim_config else None
        if progress.get('simulation_config_source_sha256') != expected_sim_hash:
            raise RuntimeError('simulation profile changed across resume; historical CPU runs require --legacy-cpu-physics')
        validate_resume_identity(
            progress,
            official_runtime_task=ARGS.official_robolab_task,
            official_task_file_sha256=official_task["task_file_sha256"],
            worldir_hash=plan["world_hash"],
            prompt=ARGS.prompt,
            seed_schedule_seed=ARGS.seed_schedule_seed,
            runtime_identity_sha256=runtime_identity_sha256,
        )
        completed_windows = int(progress["completed_windows"])
        window_records = list(progress["windows"])
        saved_trajectory = json.loads((simulation_root / "trajectory.json").read_text(encoding="utf-8"))
        frames = validate_resume_ledger(
            progress, saved_trajectory, official_max_steps=official_max_steps
        )
        if len(saved_trajectory["frames"]) != len(frames):
            saved_trajectory["frames"] = frames
            _write_json(simulation_root / "trajectory.json", saved_trajectory)
        (simulation_root / "trajectory.json.partial").unlink(missing_ok=True)
        (capture_root / "capture_progress.json.partial").unlink(missing_ok=True)
        terminated_early = bool(progress["terminated_early"])
        task_completed = bool(progress["task_completed"])
        if terminated_early and task_completed:
            raise RuntimeError("resume checkpoint cannot be both failed and successful")
        if terminated_early:
            raise RuntimeError(
                "obsolete harness-terminated Cosmos checkpoints cannot be resumed"
            )
        completion_frame = progress.get("completion_frame")
        completion_evidence = progress.get("completion_evidence")
        termination_frame = progress.get("termination_frame")
        termination_reason = progress.get("termination_reason")
        termination_evidence = progress.get("termination_evidence")
        replay_frame_index = 0
        replay_left = bool(initial_frame["robot_cup_contact_evidence"]["left_inner_finger"])
        replay_right = bool(initial_frame["robot_cup_contact_evidence"]["right_inner_finger"])
        replay_container = bool(initial_frame["object_container_contact"])
        interrupted = capture_root / "interrupted_windows"
        for partial in sorted(windows_root.glob("window_*")):
            index = int(partial.name.removeprefix("window_"))
            if index < completed_windows:
                continue
            interrupted.mkdir(parents=True, exist_ok=True)
            destination = interrupted / partial.name
            attempt = 1
            while destination.exists():
                destination = interrupted / f"{partial.name}.attempt_{attempt:03d}"
                attempt += 1
            shutil.move(str(partial), str(destination))
        for window_index in range(completed_windows):
            committed_window = windows_root / f"window_{window_index:06d}"
            action_path = committed_window / "executed_actions.npy"
            metadata_path = committed_window / "metadata.json"
            ledger_record = window_records[window_index]
            if sha256_file(action_path) != ledger_record["executed_actions_sha256"]:
                raise RuntimeError(f"committed Cosmos actions changed: {action_path}")
            if sha256_file(metadata_path) != ledger_record["metadata_sha256"]:
                raise RuntimeError(f"committed Cosmos metadata changed: {metadata_path}")
            metadata_record = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_metadata = {
                key: value
                for key, value in ledger_record.items()
                if key not in {"executed_actions_sha256", "metadata_sha256"}
            }
            if metadata_record != expected_metadata:
                raise RuntimeError(f"Cosmos metadata/ledger differ: {metadata_path}")
            replay_actions = np.load(action_path, allow_pickle=False)
            expected_actions = int(ledger_record["executed_action_count"])
            if (
                replay_actions.shape != (expected_actions, 8)
                or not np.isfinite(replay_actions).all()
            ):
                raise RuntimeError(f"invalid committed replay actions: {action_path}")
            if len(ledger_record["execution"]) != expected_actions:
                raise RuntimeError(f"Cosmos execution ledger count changed: {metadata_path}")
            for action_index, action in enumerate(replay_actions):
                target[0, arm_ids] = torch.from_numpy(action[:7]).to(target.device)
                target[0, gripper_ids] = executable_robolab_gripper_target(action[7])
                robot.set_joint_position_target(target)
                replay_left = replay_right = replay_container = False
                for _ in range(PHYSICS_STEPS_PER_ACTION):
                    robot.write_data_to_sim(); sim.step(render=False); robot.update(sim_cfg.dt)
                    cup.update(sim_cfg.dt)
                    for asset in extra_objects.values():
                        asset.update(sim_cfg.dt)
                    if tray is not None:
                        tray.update(sim_cfg.dt)
                    left_sensor.update(sim_cfg.dt, force_recompute=True)
                    right_sensor.update(sim_cfg.dt, force_recompute=True)
                    container_sensor.update(sim_cfg.dt, force_recompute=True)
                    replay_left |= _contact_active(left_sensor)
                    replay_right |= _contact_active(right_sensor)
                    replay_container |= _contact_active(container_sensor)
                replay_frame_index += 1
                expected_frame = frames[replay_frame_index]
                expected_execution = ledger_record["execution"][action_index]
                expected_contacts = expected_frame["robot_cup_contact_evidence"]
                if (
                    replay_left != bool(expected_contacts["left_inner_finger"])
                    or replay_right != bool(expected_contacts["right_inner_finger"])
                    or replay_container != bool(expected_frame["object_container_contact"])
                    or replay_left != bool(expected_execution["left_contact"])
                    or replay_right != bool(expected_execution["right_contact"])
                    or replay_container != bool(expected_execution["container_contact"])
                ):
                    raise RuntimeError(
                        f"Isaac deterministic resume contacts diverged at frame {replay_frame_index}"
                    )
                task_world.set_contacts(replay_left, replay_right, replay_container)
                replay_success = (graph_verifier.update(_frame_record(
                    frame_index=replay_frame_index, template=frame_template, robot=robot,
                    joint_target=target, cup=cup, tray=tray, static_tray_pose=static_tray_pose,
                    left_contact=replay_left, right_contact=replay_right,
                    container_contact=replay_container, source_window=window_index,
                    scene_bindings=scene_bindings, extra_objects=extra_objects))
                    if graph_verifier is not None else bool(official_task["success_func"](
                        task_world, **official_task["success_params"], env_id=0)))
                if not evaluation_window and replay_success and not (
                    task_completed
                    and completion_frame == replay_frame_index
                    and replay_frame_index == len(frames) - 1
                ):
                    raise RuntimeError("Cosmos resume ledger crosses an official success frame")
                if (
                    completion_frame is not None
                    and completion_frame == replay_frame_index
                    and not replay_success
                ):
                    raise RuntimeError("Cosmos resume success no longer passes its predicate")
        if replay_frame_index != len(frames) - 1:
            raise RuntimeError("Cosmos replay did not reach the committed frame")
        if list(robot.data.joint_names) != frames[-1]["robot"]["joint_names"]:
            raise RuntimeError("Cosmos replay robot joint names changed")
        validate_replayed_state(
            _frame_replay_state(frames[-1]),
            _live_replay_state(
                robot,
                target,
                cup,
                tray,
                static_tray_pose,
                replay_frame_index,
                replay_left,
                replay_right,
                replay_container, scene_bindings=scene_bindings, extra_objects=extra_objects,
            ),
        )
    else:
        frames = [initial_frame]
    terminal_frame = frames[-1]
    terminal_contacts = terminal_frame["robot_cup_contact_evidence"]
    task_world.set_contacts(
        bool(terminal_contacts["left_inner_finger"]),
        bool(terminal_contacts["right_inner_finger"]),
        bool(terminal_frame["object_container_contact"]),
    )
    terminal_predicate = (graph_verifier.update(terminal_frame) if graph_verifier is not None else bool(
        official_task["success_func"](task_world, **official_task["success_params"], env_id=0)))
    if ARGS.resume and not terminated_early and terminal_predicate != task_completed:
        raise RuntimeError("resume checkpoint and official task predicate disagree")
    if not ARGS.resume and terminal_predicate:
        task_completed = True
        completion_frame = int(terminal_frame["frame"])
        completion_evidence = {
            "official_predicate_passed": True,
            "function": official_task["success_function"],
            "role_bindings": {
                "object": "cup",
                "container": "tray",
                "gripper": "gripper",
            },
            "object_container_contact": bool(
                terminal_frame["object_container_contact"]
            ),
            "gripper_detached_at_completion": not bool(
                terminal_frame["robot_cup_contact_links"]
            ),
            "gripper_open_at_completion": bool(
                terminal_frame["robot"]["droid_gripper_position"] > 0.5
            ),
            "measured_droid_gripper_position": float(
                terminal_frame["robot"]["droid_gripper_position"]
            ),
            "detachment_semantics": "official_instantaneous_no_contact",
        }
    if not ARGS.resume:
        trajectory_payload = {
            "schema_version": "worldir_isaac_cosmos_closed_loop_trajectory.v1",
            "authority": {"physics": f"Isaac Sim 5.1 PhysX on {ARGS.device}", "worldir_hash": plan["world_hash"]},
            "physics_hz": PHYSICS_HZ, "sample_hz": ACTION_HZ,
            "control_hz": ACTION_HZ, "frames": frames,
        }
        _write_json(simulation_root / "trajectory.json", trajectory_payload)
        _write_json(capture_root / "capture_progress.json", {
            "schema_version": "cosmos_worldir_isaac_progress.v2",
            "rollout_controller": "task_graph_native_episode" if graph_verifier is not None else "official_robolab_task",
            "official_runtime_task": ARGS.official_robolab_task,
            "official_task_file_sha256": official_task["task_file_sha256"],
            "worldir_hash": plan["world_hash"],
            "prompt": ARGS.prompt,
            "seed_schedule_seed": ARGS.seed_schedule_seed,
            "runtime_identity_sha256": runtime_identity_sha256,
            "simulation_config_source_sha256": sha256_file(ARGS.robolab_sim_config) if ARGS.robolab_sim_config else None,
            "current_frame_count": len(frames),
            "completed_windows": 0,
            "evaluation_window_sha256": digest(evaluation_window) if evaluation_window else None,
            "windows": [],
            "checkpoint_frames_sha256": checkpoint_frames_sha256(frames),
            "terminated_early": terminated_early,
            "termination_frame": termination_frame,
            "termination_reason": termination_reason,
            "termination_evidence": termination_evidence,
            "task_completed": task_completed,
            "completion_frame": completion_frame,
            "completion_evidence": completion_evidence,
        })
    initial_cup_position = np.asarray(frames[0]["cup"]["link_position"], dtype=np.float64)
    seeds = np.random.default_rng(ARGS.seed_schedule_seed).integers(
        0,
        2**31,
        size=math.ceil(official_max_steps / POLICY_ACTION_CHUNK),
    )
    pending_windows = (
        range(completed_windows, len(seeds))
        if not terminated_early and (evaluation_window or not task_completed)
        else ()
    )
    for window_index in pending_windows:
        seed = int(seeds[window_index])
        remaining = official_max_steps - (len(frames) - 1)
        if remaining <= 0:
            break
        valid_count = min(POLICY_ACTION_CHUNK, remaining)
        window_root = windows_root / f"window_{window_index:06d}"
        window_root.mkdir(parents=True)
        observation = _render_observation(
            frame_index=int(frames[-1]["frame"]), window_root=window_root,
            observation_root=observations_root, sim=sim,
            wrist=wrist_camera, left=left_camera, right=right_camera,
        )
        arm_before = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().copy()
        gripper_before = normalize_robolab_gripper_position(
            robot.data.joint_pos[0, gripper_ids[0]].item()
        )
        if ARGS.replay_actions_from is None and ARGS.policy_model == 'gr00t':
            from wo.models.gr00t_window import run_native_window
            actions, raw_actions, metadata = run_native_window(
                window_root=window_root, observation_root=observations_root,
                frame_index=int(frames[-1]['frame']), joints=arm_before,
                gripper=gripper_before, prompt=ARGS.prompt,
                host=ARGS.policy_server_host, port=ARGS.policy_server_port)
        elif ARGS.replay_actions_from is None:
            actions, raw_actions, metadata = _run_server_window(
                window_root=window_root,
                input_image=observation,
                joints=arm_before,
                gripper=gripper_before,
                prompt=ARGS.prompt,
                host=ARGS.policy_server_host,
                port=ARGS.policy_server_port,
                request_index=window_index,
                seed_schedule_seed=ARGS.seed_schedule_seed,
                expected_request_seed=seed,
            )
        else:
            replay_path = (
                ARGS.replay_actions_from.resolve()
                / "capture"
                / "windows"
                / f"window_{window_index:06d}"
                / "executed_actions.npy"
            )
            replay = np.load(replay_path, allow_pickle=False)
            if replay.ndim != 2 or replay.shape[1] != 8 or len(replay) < valid_count:
                raise RuntimeError(f"invalid replay action window: {replay_path} {replay.shape}")
            future = replay[:valid_count]
            if len(future) < POLICY_ACTION_CHUNK:
                future = np.concatenate(
                    [
                        future,
                        np.repeat(
                            future[-1:], POLICY_ACTION_CHUNK - len(future), axis=0
                        ),
                    ],
                    axis=0,
                )
            initial_action = np.concatenate(
                [arm_before.astype(np.float32), np.asarray([gripper_before], dtype=np.float32)]
            )[None]
            actions = np.concatenate([initial_action, future], axis=0)
            raw_actions = actions.copy()
            metadata = {
                "action_source": "certified_replay",
                "source_path": str(replay_path),
                "conditioning_image": str(observation),
            }
        if ARGS.replay_actions_from is None and ARGS.policy_model == 'cosmos3':
            recorded_prompt_format([*window_records, {'model_metadata': metadata}])
        execution = []
        executed_actions = []
        for action_index, action in enumerate(actions[1 : 1 + valid_count]):
            target[0, arm_ids] = torch.from_numpy(action[:7]).to(target.device)
            target[0, gripper_ids] = executable_robolab_gripper_target(action[7])
            robot.set_joint_position_target(target)
            any_left = any_right = any_container = False
            subject_contacts = {name: False for name in release_sensors}
            support_steps = []
            for _ in range(PHYSICS_STEPS_PER_ACTION):
                robot.write_data_to_sim(); sim.step(render=False); robot.update(sim_cfg.dt)
                cup.update(sim_cfg.dt)
                for asset in extra_objects.values():
                    asset.update(sim_cfg.dt)
                if tray is not None:
                    tray.update(sim_cfg.dt)
                left_sensor.update(sim_cfg.dt, force_recompute=True)
                right_sensor.update(sim_cfg.dt, force_recompute=True)
                container_sensor.update(sim_cfg.dt, force_recompute=True)
                for name, sensor in release_sensors.items():
                    sensor.update(sim_cfg.dt, force_recompute=True)
                    subject_contacts[name] |= _contact_active(sensor)
                for sensor, _ in support_sensors.values():
                    sensor.update(sim_cfg.dt, force_recompute=True)
                if support_sensors:
                    support_steps.append({name: read_contact_support(sensor, identities, sim_cfg.dt)
                                          for name, (sensor, identities) in support_sensors.items()})
                lf = left_sensor.data.force_matrix_w
                rf = right_sensor.data.force_matrix_w
                any_left |= bool(lf is not None and torch.linalg.vector_norm(lf).item() > 1e-4)
                any_right |= bool(rf is not None and torch.linalg.vector_norm(rf).item() > 1e-4)
                cf = container_sensor.data.force_matrix_w
                any_container |= bool(
                    cf is not None and torch.linalg.vector_norm(cf).item() > 1e-4
                )
            previous_cup_position = np.asarray(
                frames[-1]["cup"]["link_position"], dtype=np.float64
            )
            frame = _frame_record(
                frame_index=len(frames), template=frame_template, robot=robot,
                joint_target=target, cup=cup, tray=tray,
                static_tray_pose=static_tray_pose,
                left_contact=any_left, right_contact=any_right,
                container_contact=any_container, source_window=window_index,
                scene_bindings=scene_bindings, extra_objects=extra_objects,
            )
            frame["robot_subject_contacts"] = subject_contacts
            frame['contact_support_physics_steps'] = support_steps
            frame['contact_support'] = support_steps[-1] if support_steps else {}
            frames.append(frame)
            if ARGS.policy_model == 'gr00t':
                # WAM video evidence is the executed simulator state, not decoded
                # Cosmos predictions or repeated policy-conditioning snapshots.
                _render_observation(
                    frame_index=int(frame['frame']), window_root=window_root,
                    observation_root=observations_root, sim=sim,
                    wrist=wrist_camera, left=left_camera, right=right_camera)
            executed_actions.append(action)
            motion = _manipuland_motion_diagnostic(
                initial_position=initial_cup_position,
                previous_position=previous_cup_position,
                current_position=frame["cup"]["link_position"],
            )
            execution.append({
                "action_index": action_index,
                "left_contact": any_left,
                "right_contact": any_right,
                "container_contact": any_container,
                "manipuland_motion": motion,
            })
            task_world.set_contacts(any_left, any_right, any_container)
            task_completed = (graph_verifier.update(frame) if graph_verifier is not None else bool(
                official_task["success_func"](task_world, **official_task["success_params"], env_id=0)))
            if task_completed and completion_frame is None:
                completion_frame = int(frame["frame"])
                completion_evidence = {
                    "native_predicate_evidence": graph_verifier.last_evidence if graph_verifier is not None else None,
                    "official_predicate_passed": True,
                    "function": official_task["success_function"],
                    "role_bindings": {
                        "object": "cup",
                        "container": "tray",
                        "gripper": "gripper",
                    },
                    "object_container_contact": any_container,
                    "gripper_detached_at_completion": not (any_left or any_right),
                    "gripper_open_at_completion": bool(
                        frame["robot"]["droid_gripper_position"] > 0.5
                    ),
                    "measured_droid_gripper_position": float(
                        frame["robot"]["droid_gripper_position"]
                    ),
                    "detachment_semantics": "official_instantaneous_no_contact",
                }
                if not evaluation_window:
                    break
        executed_actions_array = np.asarray(executed_actions, dtype=np.float32)
        np.save(window_root / "executed_actions.npy", executed_actions_array, allow_pickle=False)
        executed_count = len(executed_actions)
        record = {
            "window_index": window_index, "request_seed": int(seed),
            "planned_action_count": valid_count, "executed_action_count": executed_count,
            "conditioning_frame": len(frames) - executed_count - 1,
            "joint_position_before": arm_before.tolist(), "gripper_before": gripper_before,
            "raw_gripper_min": float(raw_actions[1:, -1].min()),
            "raw_gripper_max": float(raw_actions[1:, -1].max()),
            "execution": execution, "model_metadata": metadata,
        }
        metadata_path = window_root / "metadata.json"
        _write_json(metadata_path, record)
        window_records.append({
            **record,
            "executed_actions_sha256": sha256_file(
                window_root / "executed_actions.npy"
            ),
            "metadata_sha256": sha256_file(metadata_path),
        })
        _write_json(simulation_root / "trajectory.json", {
            "schema_version": "worldir_isaac_cosmos_closed_loop_trajectory.v1",
            "authority": {"physics": f"Isaac Sim 5.1 PhysX on {ARGS.device}", "worldir_hash": plan["world_hash"]},
            "physics_hz": PHYSICS_HZ, "sample_hz": ACTION_HZ,
            "control_hz": ACTION_HZ, "frames": frames,
        })
        _write_json(capture_root / "capture_progress.json", {
            "schema_version": "cosmos_worldir_isaac_progress.v2",
            "rollout_controller": "task_graph_native_episode" if graph_verifier is not None else "official_robolab_task",
            "official_runtime_task": ARGS.official_robolab_task,
            "official_task_file_sha256": official_task["task_file_sha256"],
            "worldir_hash": plan["world_hash"],
            "prompt": ARGS.prompt,
            "seed_schedule_seed": ARGS.seed_schedule_seed,
            "runtime_identity_sha256": runtime_identity_sha256,
            "simulation_config_source_sha256": sha256_file(ARGS.robolab_sim_config) if ARGS.robolab_sim_config else None,
            "current_frame_count": len(frames), "completed_windows": len(window_records),
            "evaluation_window_sha256": digest(evaluation_window) if evaluation_window else None,
            "windows": window_records,
            "checkpoint_frames_sha256": checkpoint_frames_sha256(frames),
            "terminated_early": terminated_early,
            "termination_frame": termination_frame,
            "termination_reason": termination_reason,
            "termination_evidence": termination_evidence,
            "task_completed": task_completed,
            "completion_frame": completion_frame,
            "completion_evidence": completion_evidence,
        })
        if task_completed and not evaluation_window:
            break

    initial_cup = np.asarray(frames[0]["cup"]["link_position"])
    final_cup = np.asarray(frames[-1]["cup"]["link_position"])
    cup_positions = np.asarray(
        [frame["cup"]["link_position"] for frame in frames], dtype=np.float64
    )
    cup_step_displacements = np.linalg.norm(np.diff(cup_positions, axis=0), axis=1)
    max_cup_step_displacement = float(cup_step_displacements.max(initial=0.0))
    robot_joint_positions = np.asarray(
        [frame["robot"]["joint_positions"] for frame in frames], dtype=np.float64
    )
    physics_stability_pass = bool(
        np.isfinite(cup_positions).all()
        and np.isfinite(robot_joint_positions).all()
    )
    tray_xy = np.asarray(frames[-1]["tray"]["link_position"][:2])
    cup_xy = final_cup[:2]
    max_z = max(float(frame["cup"]["link_position"][2]) for frame in frames)
    dual_count = sum(
        bool(frame["robot_cup_contact_evidence"]["simultaneous_dual_contact"])
        for frame in frames
    )
    release_window = frames[-min(30, len(frames)):]
    final_robot_cup_contact_frame_count = sum(
        bool(frame["robot_cup_contact_links"]) for frame in release_window
    )
    sustained_final_no_gripper_contact = final_robot_cup_contact_frame_count == 0
    final_cup_positions = np.asarray(
        [frame["cup"]["link_position"] for frame in release_window], dtype=np.float64
    )
    final_cup_settle_displacement = float(
        np.linalg.norm(np.diff(final_cup_positions, axis=0), axis=1).max(initial=0.0)
    )
    tray_positions = np.asarray(
        [frame["tray"]["link_position"] for frame in frames], dtype=np.float64
    )
    tray_displacement = float(
        np.linalg.norm(tray_positions - tray_positions[0], axis=1).max(initial=0.0)
    )
    effective_prompt_format = recorded_prompt_format(window_records) if ARGS.policy_model == 'cosmos3' else None
    result = {
        "schema_version": "cosmos_worldir_isaac_closed_loop.v3",
        "status": "complete",
        "physics_authority": f"RoboLab Isaac Lab 2.3.2 / Isaac Sim 5.1 PhysX on {ARGS.device}",
        "simulation_config_source_sha256": sha256_file(ARGS.robolab_sim_config) if ARGS.robolab_sim_config else None,
        "simulation_profile": 'original_robolab' if ARGS.robolab_sim_config else 'legacy_cpu',
        "observation_bridge": "native Isaac RTX DROID RGB from the live PhysX state every 32 actions",
        "camera_contract": "released RoboLab WRIST_LEFT_RIGHT_HEAD DROID rig transformed by the WorldIR robot-root pose at 640x360",
        "native_isaac_rtx_renderer": "live Isaac RTX camera observations",
        "prompt": ARGS.prompt,
        "model": SOURCE_MODEL,
        "checkpoint_revision": FROZEN_CHECKPOINT_REVISION,
        "framework_revision": FROZEN_FRAMEWORK_REVISION,
        "model_config_sha256": FROZEN_MODEL_CONFIG_SHA256,
        "wan_vae_sha256": FROZEN_WAN_VAE_SHA256,
        "runtime_identity": os.path.relpath(runtime_identity_path, capture_root),
        "runtime_identity_sha256": runtime_identity_sha256,
        "robot_usd_sha256": ARGS.robot_usd_sha256,
        "action_space": "joint_pos",
        "action_postprocessing": "official_robolab_gripper_gt_0.5_binary_v1",
        "native_action_hz": 15.0,
        "observation_feedback": "simulator_closed_loop",
        "closed_loop_horizon_steps": POLICY_ACTION_CHUNK,
        "refresh_horizon_steps": POLICY_ACTION_CHUNK,
        "video_stitching_policy": "fresh_condition_once_per_32_step_window_v1",
        "benchmark_target": "imagination_video_only",
        "actions_scored": False,
        "format_prompt_as_json": effective_prompt_format,
        "prompt_format_evidence": "per_window_server_transform_attestation" if effective_prompt_format is not None else "unknown_legacy_or_replay",
        "conditioning_frame_included": True,
        "rollout_controller": (
            "task_graph_native_episode"
            if graph_verifier is not None
            else "official_robolab_task"
        ),
        "official_runtime_task": ARGS.official_robolab_task,
        "official_task_class": official_task["task_class"],
        "official_task_file_sha256": official_task["task_file_sha256"],
        "official_task_success_function": official_task["success_function"],
        "official_task_episode_length_s": official_task["episode_length_s"],
        "official_task_max_steps": official_max_steps,
        "termination_policy": official_task["termination_policy"],
        "native_protocol_sha256": official_task.get("native_protocol_sha256"),
        "native_predicate_evidence": graph_verifier.last_evidence if graph_verifier is not None else None,
        "elapsed_control_time_sec": (len(frames) - 1) / ACTION_HZ,
        "video_duration_sec": len(frames) / ACTION_HZ,
        "actual_duration_sec": len(frames) / ACTION_HZ,
        "actual_future_steps": len(frames) - 1,
        "worldir_plan": os.path.relpath(ARGS.worldir_plan.resolve(), capture_root),
        "seed_schedule_seed": ARGS.seed_schedule_seed,
        "action_source": (
            "cosmos3_nano_policy_droid"
            if ARGS.replay_actions_from is None
            else "certified_action_replay"
        ),
        "worldir_hash": plan["world_hash"], "frame_count": len(frames),
        "action_count": len(frames) - 1, "window_count": len(window_records),
        "terminated_early": terminated_early,
        "termination_frame": termination_frame,
        "termination_reason": (
            "evaluation_window_complete" if evaluation_window else "task_completed" if task_completed else (
                "native_task_timeout"
                if graph_verifier is not None
                else "official_task_timeout"
            )
        ),
        "termination_evidence": termination_evidence,
        "completion_frame": completion_frame,
        "completion_evidence": completion_evidence,
        "cup_displacement_m": float(np.linalg.norm(final_cup - initial_cup)),
        "cup_lift_m": float(max_z - initial_cup[2]),
        "max_cup_step_displacement_m": max_cup_step_displacement,
        "finite_physics_state_pass": physics_stability_pass,
        "physics_stability_pass": physics_stability_pass,
        "dual_contact_frame_count": dual_count,
        "sustained_final_no_gripper_contact": sustained_final_no_gripper_contact,
        "final_contact_window_frames": len(release_window),
        "final_robot_cup_contact_frame_count": final_robot_cup_contact_frame_count,
        "final_cup_settle_displacement_m": final_cup_settle_displacement,
        "tray_displacement_m": tray_displacement,
        "final_cup_tray_xy_distance_m": float(np.linalg.norm(cup_xy - tray_xy)),
        "task_success": task_completed,
        "evaluation_window": evaluation_window,
        "evaluation_window_sha256": digest(evaluation_window) if evaluation_window else None,
        "trajectory": os.path.relpath(
            simulation_root / "trajectory.json", capture_root
        ),
        "windows": window_records,
    }
    if ARGS.policy_model == 'gr00t':
        result.update(model='nvidia/GR00T-N1.6-DROID',
                      schema_version='gr00t_worldir_isaac_closed_loop.v1',
                      action_source='gr00t_n1.6_droid' if ARGS.replay_actions_from is None else 'certified_action_replay',
                      checkpoint_revision=identity.get('checkpoint_revision'),
                      framework_revision=identity.get('framework_revision'),
                      model_config_sha256=identity.get('model_config_sha256'),
                      wan_vae_sha256=None,
                      observation_bridge='native Isaac RTX separate left and wrist RGB every 8 actions',
                      video_stitching_policy='native_simulator_camera_frames',
                      benchmark_target='native_simulator_execution', actions_scored=True)
    _write_json(capture_root / "capture_metadata.json", result)
    print(
        json.dumps({key: value for key, value in result.items() if key != "windows"}, indent=2),
        flush=True,
    )
    # AppLauncher owns application teardown.  Calling SimulationContext.stop()
    # after a Cosmos torchrun subprocess can block in Isaac's render callback.
    sys.stdout.flush()
    sys.stderr.flush()
    # Isaac Sim 5.1 can also deadlock while tearing down Replicator after a
    # torchrun child exits.  At this point every result is atomically closed;
    # process exit is the reliable resource-release boundary on this host.
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Isaac Sim can deadlock in APP.close() after a model-side exception.
        # No formal capture is published on this path, so fail immediately and
        # let the launcher remove the incomplete scratch tree.
        os._exit(1)
