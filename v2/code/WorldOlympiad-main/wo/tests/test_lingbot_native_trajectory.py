import json
from types import SimpleNamespace

import numpy as np
import pytest

from wo.models.lingbot_va_native_bridge import (
    _native_trajectory_frame,
    _run_policy,
    _write_native_episode_evidence,
)
from wo.spec import Predicate, Step, TaskObject, TaskSpec


class Pose:
    def __init__(self, p, q):
        self.p = np.asarray(p, dtype=float)
        self.q = np.asarray(q, dtype=float)  # SAPIEN wxyz


class Actor:
    def __init__(self, p, q):
        self.pose = Pose(p, q)

    def get_pose(self):
        return self.pose


def _task():
    left = SimpleNamespace(
        get_root_pose=lambda: Pose([1, 2, 3], [0.5, 0.5, 0.5, 0.5]),
        get_active_joints=lambda: [SimpleNamespace(get_name=lambda: "aloha_left_joint")],
        get_qpos=lambda: np.asarray([0.25]),
    )
    right = SimpleNamespace(
        get_active_joints=lambda: [SimpleNamespace(get_name=lambda: "aloha_right_joint")],
        get_qpos=lambda: np.asarray([-0.5]),
    )
    robot = SimpleNamespace(
        left_entity=left, right_entity=right,
        get_left_tcp_pose=lambda: [0.1, 0.2, 0.3, 0.5, 0.1, 0.2, 0.3],
    )
    return SimpleNamespace(
        robot=robot, take_action_cnt=15,
        wo_handles={"apple": Actor([0.4, 0.5, 0.6], [1, 0, 0, 0]),
                    "coaster": Actor([-0.1, 0.2, 0.3], [1, 0, 0, 0])},
        wo_object_roles={"apple": "manipuland", "coaster": "destination"},
    )


def test_native_trajectory_frame_records_real_sapien_state_and_quaternion_conventions():
    frame = _native_trajectory_frame(_task())
    assert frame["time_s"] == pytest.approx(1.0)
    assert frame["robot"]["base_position"] == [1.0, 2.0, 3.0]
    assert frame["robot"]["base_orientation_xyzw"] == [0.5, 0.5, 0.5, 0.5]
    assert frame["robot"]["end_effector_position"] == [0.1, 0.2, 0.3]
    assert frame["robot"]["end_effector_orientation_xyzw"] == [0.1, 0.2, 0.3, 0.5]
    assert frame["robot"]["joint_names"] == ["aloha_left_joint", "aloha_right_joint"]
    assert frame["robot"]["joint_positions"] == [0.25, -0.5]
    assert frame["objects"]["apple"]["link_position"] == [0.4, 0.5, 0.6]
    assert frame["objects"]["apple"]["role"] == "manipuland"


def test_native_episode_evidence_binds_generated_task_graph_and_measured_frames(tmp_path):
    spec = TaskSpec(
        "apple_to_coaster_onboarding_01", "pick_place", "Put the apple on the coaster.",
        (TaskObject("apple", "apple", "manipuland"),
         TaskObject("coaster", "coaster", "destination"),
         TaskObject("worktable", "table", "support")),
        (Step("transport", "apple", Predicate("on", "apple", "coaster"), target="coaster"),),
    )
    task_spec = tmp_path / "task.json"
    task_spec.write_text(json.dumps(spec.to_dict()))
    task = _task()
    task.take_action_cnt = 0
    frame0 = _native_trajectory_frame(task)
    task.take_action_cnt = 1
    frame1 = _native_trajectory_frame(task)
    trajectory_path, graph_path = _write_native_episode_evidence(
        tmp_path / "episode", task_spec, [frame0, frame1])
    trajectory = json.loads(trajectory_path.read_text())
    graph = json.loads(graph_path.read_text())
    assert trajectory["producer"] == "lingbot_va_native_bridge.v1"
    assert trajectory["authority"]["physics"] == "RoboTwin SAPIEN"
    assert trajectory["frames"] == [frame0, frame1]
    assert trajectory["frames"][1]["time_s"] > trajectory["frames"][0]["time_s"]
    assert graph["success_program"][0]["subject"] == "apple"
    assert graph["success_program"][0]["object"] == "coaster"
    assert graph["source_binding"]["source"] == "bound_task_spec"
    assert graph["source_binding"]["task_sha256"] == trajectory["task_sha256"]


def test_native_episode_evidence_works_when_bridge_is_loaded_as_standalone_script(tmp_path):
    """The registered adapter invokes this bridge by file path, not package name."""
    import importlib.util
    from pathlib import Path

    bridge_path = Path(__file__).parents[1] / "models" / "lingbot_va_native_bridge.py"
    spec = importlib.util.spec_from_file_location("lingbot_native_bridge_script", bridge_path)
    bridge = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(bridge)

    task = TaskSpec(
        "apple_to_coaster_onboarding_01", "pick_place", "Put the apple on the coaster.",
        (TaskObject("apple", "apple", "manipuland"),
         TaskObject("coaster", "coaster", "destination")),
        (Step("transport", "apple", Predicate("on", "apple", "coaster"), target="coaster"),),
    )
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task.to_dict()))
    trajectory, graph = bridge._write_native_episode_evidence(
        tmp_path / "episode", task_path,
        [_native_trajectory_frame(_task()), _native_trajectory_frame(_task())],
    )
    assert trajectory.is_file() and graph.is_file()
    assert json.loads(trajectory.read_text())["producer"] == "lingbot_va_native_bridge.v1"
    assert json.loads(graph.read_text())["source_binding"]["source"] == "bound_task_spec"


def test_native_episode_evidence_refuses_to_publish_a_single_unmeasured_frame(tmp_path):
    spec = TaskSpec(
        "apple_to_coaster_onboarding_01", "pick_place", "Put the apple on the coaster.",
        (TaskObject("apple", "apple", "manipuland"), TaskObject("coaster", "coaster", "destination")),
        (Step("transport", "apple", Predicate("on", "apple", "coaster"), target="coaster"),),
    )
    task_spec = tmp_path / "task.json"
    task_spec.write_text(json.dumps(spec.to_dict()))
    with pytest.raises(ValueError, match="initial and post-action"):
        _write_native_episode_evidence(tmp_path / "episode", task_spec, [_native_trajectory_frame(_task())])


def test_native_policy_records_initial_and_each_completed_action_state(tmp_path, monkeypatch):
    class Writer:
        def append_data(self, _frame):
            pass

        def close(self):
            pass

    import sys
    monkeypatch.setitem(sys.modules, "imageio", SimpleNamespace(
        get_writer=lambda *_args, **_kwargs: Writer()))

    class Task:
        def __init__(self):
            self.robot = _task().robot
            self.take_action_cnt = 0
            self.step_lim = 2
            self.eval_success = False
            self.wo_handles = _task().wo_handles
            self.wo_object_roles = {"apple": "manipuland", "coaster": "destination"}

        def get_obs(self):
            rgb = np.zeros((2, 2, 3), dtype=np.uint8)
            return {
                "observation": {key: {"rgb": rgb} for key in
                                ("head_camera", "left_camera", "right_camera")},
                "joint_action": {"vector": np.zeros(2)},
                "endpose": {"left_endpose": [0.] * 6, "left_gripper": 0.,
                            "right_endpose": [0.] * 6, "right_gripper": 0.},
            }

        def take_action(self, _action, action_type):
            assert action_type == "ee"
            self.take_action_cnt += 1
            self.wo_handles["apple"].pose = Pose(
                [0.4 + self.take_action_cnt * 0.01, 0.5, 0.6], [1, 0, 0, 0])

    class Client:
        def infer(self, request):
            if "obs" in request and not request.get("compute_kv_cache"):
                return {"action": np.zeros((14, 1, 4), dtype=np.float32)}
            return {}

    task = Task()
    action = _run_policy(
        task, Client(), "put apple on coaster", tmp_path, action_limit=2,
        model_helpers=(lambda *_angles: [0., 0., 0., 1.], lambda pose, _initial: pose),
        record_native_trajectory=True,
    )
    frames = task.wo_native_trajectory_frames
    assert action["native_actions"] == 2
    assert len(frames) == action["native_actions"] + 1
    assert [frame["frame"] for frame in frames] == [0, 1, 2]
    assert [frame["time_s"] for frame in frames] == pytest.approx([0, 1 / 15, 2 / 15])
    assert [frame["objects"]["apple"]["link_position"][0] for frame in frames] == pytest.approx(
        [0.4, 0.41, 0.42])
