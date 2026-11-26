"""Source-owned native SAPIEN bridge for registered RoboTwin policies.

The wrapper uses only the configured RoboTwin checkout and the official pinned
LingBot websocket policy client. For reference-bound stack_blocks_two it replaces
RoboTwin's random load_actors hook with WOV2 SDF actors and executes the native
take_action path. Stack goals retain RoboTwin's native predicate; supported WOV2
on goals are checked against loaded collision geometry and native contacts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _task_setup_inputs(robotwin_repo: Path, task_name: str, output: Path, seed: int):
    import yaml
    config_root = robotwin_repo / "env_cfg" / "task_config"
    with (config_root / "demo_clean.yml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    with (config_root / "_camera_config.yml").open(encoding="utf-8") as stream:
        cameras = yaml.safe_load(stream)
    with (config_root / "_embodiment_config.yml").open(encoding="utf-8") as stream:
        embodiments = yaml.safe_load(stream)
    config["task_name"] = task_name
    config["seed"] = int(seed)
    config["now_ep_num"] = 0
    config["is_test"] = True
    config["eval_mode"] = True
    config["render_freq"] = 0
    config["save_data"] = False
    config["save_path"] = str(output / "data")
    config["eval_video_log"] = False
    config["eval_video_save_dir"] = None
    embodiment_types = config["embodiment"]
    if len(embodiment_types) != 1:
        raise ValueError("native LingBot bridge requires the configured dual-arm embodiment")
    embodiment = embodiments[embodiment_types[0]]["file_path"]
    embodiment_path = Path(embodiment)
    if not embodiment_path.is_absolute():
        embodiment_path = (robotwin_repo / embodiment_path).resolve()
    config["left_robot_file"] = str(embodiment_path)
    config["right_robot_file"] = str(embodiment_path)
    with (embodiment_path / "config.yml").open(encoding="utf-8") as stream:
        embodiment_config = yaml.safe_load(stream)
    config["left_embodiment_config"] = embodiment_config
    config["right_embodiment_config"] = embodiment_config
    config["dual_arm_embodied"] = True
    camera = cameras[config["camera"]["head_camera_type"]]
    config["head_camera_h"] = camera["h"]
    config["head_camera_w"] = camera["w"]
    config["left_arm_dim"] = len(embodiment_config["arm_joints_name"][0])
    config["right_arm_dim"] = len(embodiment_config["arm_joints_name"][1])
    return config


def _format_obs(observation: dict[str, Any], prompt: str) -> dict[str, Any]:
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
        "observation.state": observation["joint_action"]["vector"],
        "task": prompt,
    }


def _initial_eef(obs):
    endpose = obs["endpose"]
    return np.asarray(endpose["left_endpose"] + [endpose["left_gripper"]]
                      + endpose["right_endpose"] + [endpose["right_gripper"]], dtype=np.float64)


def _absolute_action(action, initial, euler_to_quat, add_init_pose):
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape == (14,):
        return np.concatenate((action[:3], euler_to_quat(*action[3:6]), action[6:10],
                               euler_to_quat(*action[10:13]), action[13:14]))
    if action.shape != (16,):
        raise ValueError(f"upstream LingBot action has unsupported width {action.shape}")
    value = add_init_pose(action, initial)
    value = np.asarray(value, dtype=np.float64)
    value[3:7] /= np.linalg.norm(value[3:7])
    value[11:15] /= np.linalg.norm(value[11:15])
    return value


def _pose_xyz_xyzw(pose):
    """Serialize a native SAPIEN pose in WOV2's xyz / xyzw convention."""
    position = np.asarray(pose.p, dtype=float).reshape(3)
    # SAPIEN poses store quaternions as wxyz; WOV2 trajectory files use xyzw.
    q_wxyz = np.asarray(pose.q, dtype=float).reshape(4)
    return position.tolist(), [float(q_wxyz[1]), float(q_wxyz[2]),
                               float(q_wxyz[3]), float(q_wxyz[0])]


def _native_trajectory_frame(task):
    """Sample actual RoboTwin state after setup or a completed native action."""
    robot = task.robot
    left, right = robot.left_entity, robot.right_entity
    base_position, base_quaternion = _pose_xyz_xyzw(left.get_root_pose())
    tcp = np.asarray(robot.get_left_tcp_pose(), dtype=float).reshape(7)
    joint_names = ([joint.get_name() for joint in left.get_active_joints()]
                   + [joint.get_name() for joint in right.get_active_joints()])
    joint_positions = (np.asarray(left.get_qpos(), dtype=float).reshape(-1).tolist()
                       + np.asarray(right.get_qpos(), dtype=float).reshape(-1).tolist())
    if len(joint_names) != len(joint_positions):
        raise ValueError("native LingBot joint names and positions differ in length")
    objects = {}
    for object_id, handle in sorted(task.wo_handles.items()):
        position, orientation = _pose_xyz_xyzw(handle.get_pose())
        objects[object_id] = {
            "position": position, "orientation_xyzw": orientation,
            "link_position": position, "link_orientation_xyzw": orientation,
            "role": getattr(task, "wo_object_roles", {}).get(object_id, "object"),
            "joint_positions": {},
        }
    frame_index = int(task.take_action_cnt)
    return {
        "frame": frame_index, "time_s": frame_index / 15.0,
        "phase": "native_policy", "operator_step_index": 0,
        "robot": {"base_position": base_position,
                  "base_orientation_xyzw": base_quaternion,
                  "joint_names": joint_names, "joint_positions": joint_positions,
                  "end_effector_position": tcp[:3].tolist(),
                  # RoboTwin's transforms3d TCP quaternion is wxyz.
                  "end_effector_orientation_xyzw": [float(tcp[4]), float(tcp[5]),
                                                      float(tcp[6]), float(tcp[3])]},
        "objects": objects,
    }


def _write_native_episode_evidence(out: Path, task_spec_path: Path,
                                   frames: list[dict], policy: str = "lingbot_va") -> tuple[Path, Path]:
    """Publish only state sampled from this rollout and its bound TaskSpec graph."""
    import hashlib
    # This file is launched by absolute path in the configured RoboTwin Python,
    # so it has no package context and relative imports fail in production.
    from wo.spec import TaskSpec
    from wo.models.cosmos import task_graph

    if len(frames) < 2:
        raise ValueError("native rollout needs initial and post-action simulator states")
    task_spec_path = Path(task_spec_path).resolve()
    spec = TaskSpec.load(task_spec_path)
    spec.validate()
    graph = task_graph(spec)
    if not graph.get("success_program"):
        raise ValueError("bound TaskSpec generated no success graph for native rollout")
    task_sha256 = hashlib.sha256(task_spec_path.read_bytes()).hexdigest()
    graph["source_binding"] = {"task_spec": str(task_spec_path),
                                "task_sha256": task_sha256,
                                "source": "bound_task_spec"}
    trajectory = {
        "schema_version": ("wo_lingbot_native_simulation.v1" if policy == "lingbot_va"
                           else "wo_native_simulation.v1"),
        "producer": f"{policy}_native_bridge.v1",
        "authority": {"physics": "RoboTwin SAPIEN", "state": "native actor and robot state"},
        "task_sha256": task_sha256, "frames": frames,
    }
    if policy == "lingbot_va":
        trajectory["control_hz"] = 15
    else:
        trajectory["sampling"] = "physics_15hz_plus_action_boundaries"
        trajectory["clock"] = "measured_sapien_physics_steps"
    simulation = out / "native_run" / "simulation"
    simulation.mkdir(parents=True, exist_ok=True)
    trajectory_path = simulation / "trajectory.json"
    graph_path = out / "task_graph.json"
    trajectory_path.write_text(json.dumps(trajectory, indent=1), encoding="utf-8")
    graph_path.write_text(json.dumps(graph, indent=1), encoding="utf-8")
    return trajectory_path, graph_path


def _update_policy_context(client, task, action_stop: int, key_frames, action) -> bool:
    """Encode post-action frames only when another policy inference can follow.

    The upstream streaming video encoder requires a complete temporal chunk.
    RoboTwin can end a rollout in the middle of such a chunk (success or native
    action limit); sending that partial terminal chunk is both unnecessary and
    can make the VAE fail while trying to produce an action that will never run.
    """
    if task.eval_success or task.take_action_cnt >= action_stop:
        return False
    client.infer({"obs": key_frames, "compute_kv_cache": True,
                  "imagine": False, "save_visualization": False, "state": action})
    return True


def _run_policy(task, client, prompt: str, capture_dir: Path, action_limit: int,
                *, model_helpers=None, record_native_trajectory=False):
    """Run the official LingBot websocket protocol over native RoboTwin actions."""
    if model_helpers is None:
        from evaluation.robotwin.geometry import euler2quat
        from scipy.spatial.transform import Rotation as Rotation

        def add_eef(new_pose, init_pose):
            new_r = Rotation.from_quat(new_pose[3:7])
            init_r = Rotation.from_quat(init_pose[3:7])
            return np.concatenate((new_pose[:3] + init_pose[:3], (init_r * new_r).as_quat(),
                                   new_pose[7:8]))

        def add_pose(new_pose, init_pose):
            return np.concatenate((add_eef(new_pose[:8], init_pose[:8]),
                                   add_eef(new_pose[8:], init_pose[8:])))
        euler_to_quat, add_init_pose = euler2quat, add_pose
    else:
        euler_to_quat, add_init_pose = model_helpers

    import imageio
    capture_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    camera_keys = {"head": "head_camera", "left_wrist": "left_camera",
                   "right_wrist": "right_camera"}
    frame_count = 0
    executed = 0
    try:
        initial_observation = task.get_obs()
        initial_pose = _initial_eef(initial_observation)
        trajectory_frames = ([_native_trajectory_frame(task)]
                             if record_native_trajectory else None)
        for camera_name, camera_key in camera_keys.items():
            path = capture_dir / f"{camera_name}.mp4"
            writers[camera_name] = imageio.get_writer(str(path), fps=10)
            writers[camera_name].append_data(np.asarray(
                initial_observation["observation"][camera_key]["rgb"], dtype=np.uint8))
        frame_count += 1
        first_obs = _format_obs(initial_observation, prompt)
        client.infer({"reset": True, "prompt": prompt, "save_visualization": False})
        first = True
        action_stop = min(int(task.step_lim), int(action_limit))
        while task.take_action_cnt < action_stop:
            if first:
                ret = client.infer({"obs": first_obs, "prompt": prompt,
                                    "save_visualization": False,
                                    "video_guidance_scale": 5.0,
                                    "action_guidance_scale": 5.0})
            else:
                ret = client.infer({"obs": first_obs, "prompt": prompt,
                                    "save_visualization": False,
                                    "video_guidance_scale": 5.0,
                                    "action_guidance_scale": 5.0})
            action = np.asarray(ret["action"])
            if action.ndim != 3 or action.shape[0] not in (14, 16) or action.shape[2] % 4:
                raise ValueError("upstream LingBot response violates native 14/16-D EE action contract")
            action_per_frame = action.shape[2] // 4
            key_frames = []
            start_idx = 1 if first else 0
            for i in range(start_idx, action.shape[1]):
                for j in range(action.shape[2]):
                    if task.take_action_cnt >= action_stop:
                        break
                    native_action = _absolute_action(action[:, i, j], initial_pose,
                                                     euler_to_quat, add_init_pose)
                    task.take_action(native_action, action_type="ee")
                    executed += 1
                    observation = task.get_obs()
                    if trajectory_frames is not None:
                        trajectory_frames.append(_native_trajectory_frame(task))
                    if (j + 1) % action_per_frame == 0:
                        formatted = _format_obs(observation, prompt)
                        key_frames.append(formatted)
                    for camera_name, camera_key in camera_keys.items():
                        writers[camera_name].append_data(np.asarray(
                            observation["observation"][camera_key]["rgb"], dtype=np.uint8))
                    frame_count += 1
                    if task.eval_success:
                        break
                if task.eval_success:
                    break
            first = False
            if not _update_policy_context(client, task, action_stop, key_frames, action):
                break
        if trajectory_frames is not None:
            if len(trajectory_frames) != executed + 1:
                raise RuntimeError("native simulator trajectory is not aligned with executed actions")
            task.wo_native_trajectory_frames = trajectory_frames
        return {"frames": frame_count, "native_actions": executed,
                "native_step_limit": int(task.step_lim)}
    finally:
        for writer in writers.values():
            writer.close()


def _run_qpos_policy(task, client, prompt: str, capture_dir: Path, action_limit: int,
                     *, record_native_trajectory=False, max_sim_time_s=None):
    """Execute FastWAM/Long-WAM's measured 14-D joint targets in RoboTwin."""
    import imageio

    camera_keys = {"head": "head_camera", "left_wrist": "left_camera",
                   "right_wrist": "right_camera"}
    capture_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    frames = 0
    executed = 0
    requests = 0
    elapsed_s = 0.0
    next_sample_s = 1.0 / 15.0
    if max_sim_time_s is not None and (not math.isfinite(max_sim_time_s)
                                        or max_sim_time_s <= 0):
        raise ValueError("window_s must be a finite positive number")
    timestep = float(task.scene.get_timestep())
    if not math.isfinite(timestep) or timestep <= 0:
        raise RuntimeError("native simulator exposes no valid physics timestep")
    try:
        observation = task.get_obs()
        trajectory = ([_native_trajectory_frame(task)] if record_native_trajectory else None)
        if trajectory is not None:
            trajectory[0]["time_s"] = 0.0
        for name, camera in camera_keys.items():
            writers[name] = imageio.get_writer(str(capture_dir / f"{name}.mp4"), fps=10)
            writers[name].append_data(np.asarray(
                observation["observation"][camera]["rgb"], dtype=np.uint8))
        frames += 1
        action_stop = min(int(task.step_lim), int(action_limit))
        while (task.take_action_cnt < action_stop and not task.eval_success
               and (max_sim_time_s is None or elapsed_s < max_sim_time_s)):
            requests += 1
            actions = np.asarray(client.infer(observation, prompt), dtype=float)
            if (actions.ndim != 2 or actions.shape[1] != 14 or not len(actions)
                    or not np.isfinite(actions).all()):
                raise RuntimeError("native qpos policy returned invalid joint targets")
            for action in actions:
                if (task.take_action_cnt >= action_stop or task.eval_success
                        or (max_sim_time_s is not None and elapsed_s >= max_sim_time_s)):
                    break
                before = int(task.take_action_cnt)
                original_step = task.scene.step
                physics_steps = 0

                def counted_step():
                    nonlocal physics_steps, next_sample_s
                    original_step()
                    physics_steps += 1
                    current_s = elapsed_s + physics_steps * timestep
                    if trajectory is not None and current_s + 1e-9 >= next_sample_s:
                        frame = _native_trajectory_frame(task)
                        frame["frame"] = len(trajectory)
                        frame["time_s"] = current_s
                        trajectory.append(frame)
                        next_sample_s = (math.floor(current_s * 15 + 1e-9) + 1) / 15

                task.scene.step = counted_step
                try:
                    task.take_action(action, action_type="qpos")
                finally:
                    task.scene.step = original_step
                if int(task.take_action_cnt) <= before:
                    raise RuntimeError("native qpos action did not advance the simulator")
                if physics_steps <= 0:
                    raise RuntimeError("native qpos action executed no simulator physics steps")
                elapsed_s += physics_steps * timestep
                executed += 1
                observation = task.get_obs()
                if trajectory is not None:
                    if trajectory[-1]["time_s"] + 1e-9 < elapsed_s:
                        frame = _native_trajectory_frame(task)
                        frame["frame"] = len(trajectory)
                        frame["time_s"] = elapsed_s
                        trajectory.append(frame)
                for name, camera in camera_keys.items():
                    writers[name].append_data(np.asarray(
                        observation["observation"][camera]["rgb"], dtype=np.uint8))
                frames += 1
        if trajectory is not None:
            if len(trajectory) < executed + 1:
                raise RuntimeError("native qpos trajectory missed an executed action")
            task.wo_native_trajectory_frames = trajectory
        return {"frames": frames, "native_actions": executed,
                "model_requests": requests,
                "simulation_elapsed_s": elapsed_s,
                "native_step_limit": int(task.step_lim)}
    finally:
        for writer in writers.values():
            writer.close()


def _native_scene_check(task, observation, out: Path, setup: dict) -> dict[str, Any]:
    """Save native input frames and measure authored actor visibility without policy inference."""
    from PIL import Image, ImageColor

    out.mkdir(parents=True, exist_ok=True)
    observation = observation["observation"]
    cameras = ("head_camera", "left_camera", "right_camera")
    palette_values = sorted(set(ImageColor.colormap.values()))
    palette = np.asarray([ImageColor.getrgb(value) for value in palette_values], dtype=np.uint8)
    visible_pixels: dict[str, dict[str, int]] = {}
    camera_paths: dict[str, str] = {}
    actor_ids: dict[str, int] = {}
    for object_id, actor in sorted(getattr(task, "wo_handles", {}).items()):
        # SAPIEN 2 entities expose the segmentation identity as per_scene_id;
        # get_id() is not part of that API. Keep generic aliases for compatible
        # actor wrappers, but prefer the native ID that camera actor labels use.
        value = None
        for getter_name, attribute_name in (("get_per_scene_id", "per_scene_id"),
                                            ("get_id", "id")):
            getter = getattr(actor, getter_name, None)
            if callable(getter):
                value = getter()
                break
            candidate = getattr(actor, attribute_name, None)
            if candidate is not None:
                value = candidate
                break
        if value is None:
            raise RuntimeError(f"native actor {object_id} exposes no renderer identity")
        actor_id = int(value) & 0xFF
        actor_ids[object_id] = actor_id
        visible_pixels[object_id] = {}
    for camera in cameras:
        camera_obs = observation.get(camera)
        if not isinstance(camera_obs, dict) or "rgb" not in camera_obs:
            continue
        rgb = np.asarray(camera_obs["rgb"], dtype=np.uint8)
        path = out / "cameras" / f"{camera}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(path)
        camera_paths[camera] = str(path)
        segmentation = camera_obs.get("actor_segmentation")
        if segmentation is None:
            continue
        segmentation = np.asarray(segmentation, dtype=np.uint8)
        for object_id, actor_id in actor_ids.items():
            count = int(np.all(segmentation == palette[actor_id], axis=-1).sum())
            visible_pixels[object_id][camera] = count
    missing = sorted(object_id for object_id, views in visible_pixels.items()
                     if not any(count > 0 for count in views.values()))
    result = {
        "schema_version": "wo_lingbot_native_scene_check.v1",
        "status": "ready" if visible_pixels and not missing else "failed",
        "task_class": type(task).__name__,
        "required_visible": sorted(visible_pixels),
        "visible_pixels": visible_pixels,
        "actor_ids": actor_ids,
        "missing_objects": missing,
        "camera_images": camera_paths,
        "camera_shapes": {camera: list(np.asarray(observation[camera]["rgb"]).shape)
                          for camera in camera_paths},
        "renderer": {"engine": "RoboTwin SAPIEN", "camera_config": setup.get("camera"),
                     "head_camera_h": setup.get("head_camera_h"),
                     "head_camera_w": setup.get("head_camera_w")},
        "native_objects": {object_id: {"position": np.asarray(actor.get_pose().p, dtype=float).tolist()}
                            for object_id, actor in sorted(getattr(task, "wo_handles", {}).items())},
    }
    (out / "scene_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _action_limit(step_lim: int, window_s: float | None) -> int:
    limit = int(step_lim)
    if window_s is None:
        return limit
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be a finite positive number")
    return min(limit, max(1, math.ceil(window_s * 15)))


def run_bridge(*, robotwin_repo: str | Path, wov2_source: str | Path, out: str | Path,
               port: int, seed: int, window_s: float | None = None,
               policy: str = "lingbot_va",
               stock_task: str | None = None,
               reference: str | Path | None = None, task_spec_path: str | Path | None = None,
               reference_state_path: str | Path | None = None, prompt: str | None = None,
               scene_check: bool = False, client_factory=None, task_factory=None,
               policy_runner=None, actor_loader=None, dynamics_binder=None,
               sapien_module=None) -> dict[str, Any]:
    """Invoke configured upstream task and return policy_check/native evidence."""
    robotwin_repo, wov2_source, out = (Path(robotwin_repo).resolve(),
                                        Path(wov2_source).resolve(), Path(out).resolve())
    if not robotwin_repo.is_dir() or not wov2_source.is_dir():
        raise FileNotFoundError("configured RoboTwin/WOV2 source checkout is missing")
    if policy not in {"lingbot_va", "fastwam", "longwam"}:
        raise ValueError(f"unsupported native RoboTwin policy: {policy}")
    task_name = stock_task or "stack_blocks_two"
    if task_name != "stack_blocks_two":
        raise ValueError("source-owned LingBot bridge currently admits stack_blocks_two only")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(wov2_source))
    sys.path.insert(0, str(robotwin_repo))
    lingbot_repo = os.environ.get("WO_LINGBOT_REPO")
    if lingbot_repo:
        sys.path.insert(0, str(Path(lingbot_repo).resolve()))
    os.chdir(robotwin_repo)
    import importlib
    if task_factory is None:
        task_module = importlib.import_module(f"envs.{task_name}")
        task_factory = getattr(task_module, task_name)
    task = task_factory()
    args = _task_setup_inputs(robotwin_repo, task_name, out, seed)
    task_spec = None
    if reference is not None:
        if task_spec_path is None or reference_state_path is None:
            raise ValueError("reference-bound bridge requires TaskSpec and reference-state archives")
        from wo.models.lingbot_va_bridge import initialize_reference_stack_blocks_two
        task_spec = json.loads(Path(task_spec_path).read_text(encoding="utf-8"))
        reference_state = json.loads(Path(reference_state_path).read_text(encoding="utf-8"))
        if sapien_module is None:
            import sapien.core as sapien_module
        initialize_reference_stack_blocks_two(
            task, reference=reference, task_spec=task_spec,
            reference_state=reference_state, setup_kwargs=args,
            actor_loader=actor_loader, dynamics_binder=dynamics_binder,
            sapien_module=sapien_module)
        task.wo_object_roles = {obj["object_id"]: obj.get("role", "object")
                                for obj in task_spec.get("objects", [])}
        args["task_name"] = task_name
    else:
        task.setup_demo(**args)
    task.set_instruction(prompt or (json.loads(Path(task_spec_path).read_text()).get("instruction")
                                    if task_spec_path else
                                    "Stack the green block on top of the red block."))
    instruction = task.get_instruction()
    if scene_check:
        try:
            if reference is None or not getattr(task, "wo_handles", None):
                raise ValueError("native scene check requires reference-bound task actors")
            task.data_type["actor_segmentation"] = True
            observation = task.get_obs()
            result = _native_scene_check(task, observation, out, args)
            if result["status"] != "ready":
                raise RuntimeError(f"native camera visibility check failed: {result['missing_objects']}")
            return result
        finally:
            task.close_env()
            if getattr(task, "render_freq", 0):
                task.viewer.close()
    if client_factory is None:
        if policy == "lingbot_va":
            from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
            client_factory = lambda: WebsocketClientPolicy(port=int(port))
        else:
            from wo.models.fastwam_client import FastWAMClient
            client_factory = lambda: FastWAMClient(port=int(port))
    client = client_factory()
    record_native_trajectory = reference is not None and policy_runner is None
    native_qpos_runner = policy_runner is None and policy in {"fastwam", "longwam"}
    if policy_runner is None:
        if policy == "lingbot_va":
            def policy_runner(bound_task, bound_client, text, destination, limit):
                return _run_policy(bound_task, bound_client, text, destination, limit,
                                   record_native_trajectory=record_native_trajectory)
        else:
            def policy_runner(bound_task, bound_client, text, destination, limit):
                return _run_qpos_policy(bound_task, bound_client, text, destination, limit,
                                        record_native_trajectory=record_native_trajectory,
                                        max_sim_time_s=window_s)
    capture_dir = out / "capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    action_limit = (int(task.step_lim) if native_qpos_runner else
                    _action_limit(getattr(task, "step_lim", 1000), window_s))
    try:
        capture = policy_runner(task, client, instruction, capture_dir, action_limit)
        if record_native_trajectory:
            frames = getattr(task, "wo_native_trajectory_frames", None)
            if (not isinstance(frames, list)
                    or (len(frames) < int(capture["native_actions"]) + 1
                        if native_qpos_runner else
                        len(frames) != int(capture["native_actions"]) + 1)):
                raise RuntimeError("native rollout produced no complete simulator trajectory")
            _write_native_episode_evidence(out, Path(task_spec_path), frames, policy)
        task_success = bool(task.check_success())
        result = {"schema_version": "wo_native_capture.v1", "task": task_name,
                  "task_success": task_success,
                  "completion_evidence": {"native_predicate_evidence": {
                      "predicate_source": getattr(task, "wo_goal_predicate_source",
                                                   "upstream RoboTwin check_success"),
                      "result": task_success, "task_class": type(task).__name__,
                      **getattr(task, "wo_goal_evidence", {})}},
                  "capture": {**capture, "videos": {
                      "head": str((capture_dir / "head.mp4").relative_to(out)),
                      "left_wrist": str((capture_dir / "left_wrist.mp4").relative_to(out)),
                      "right_wrist": str((capture_dir / "right_wrist.mp4").relative_to(out))}},
                  "termination": {"native_eval_success": bool(task.eval_success),
                                  "take_action_cnt": int(task.take_action_cnt),
                                  "step_limit": int(task.step_lim)}}
        capture_path = capture_dir / "capture_metadata.json"
        capture_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if reference is not None:
            episode = {"model": policy, "reference": str(Path(reference).resolve()),
                       "task": task_name, "seed": int(seed), "task_success": task_success,
                       "native_result": result, "capture": result["capture"]}
            (out / "episode.json").write_text(json.dumps(episode, indent=2), encoding="utf-8")
        return result
    finally:
        task.close_env()
        if getattr(task, "render_freq", 0):
            task.viewer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-repo", type=Path, required=True)
    parser.add_argument("--wov2-source", type=Path, required=True)
    parser.add_argument("--stock-task")
    parser.add_argument("--ref", type=Path)
    parser.add_argument("--task-spec", type=Path)
    parser.add_argument("--reference-state", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-s", type=float)
    parser.add_argument("--policy", default="lingbot_va")
    parser.add_argument("--prompt")
    parser.add_argument("--scene-check", action="store_true",
                        help="initialize bound native scene and audit cameras without policy inference")
    args = parser.parse_args()
    run_bridge(robotwin_repo=args.robotwin_repo, wov2_source=args.wov2_source,
               stock_task=args.stock_task, reference=args.ref,
               task_spec_path=args.task_spec, reference_state_path=args.reference_state,
               out=args.out, port=args.port, seed=args.seed, window_s=args.window_s,
               prompt=args.prompt, policy=args.policy,
               scene_check=args.scene_check)


if __name__ == "__main__":
    main()
