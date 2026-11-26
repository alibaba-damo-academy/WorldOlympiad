import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import ImageColor

from wo.models import lingbot_va_bridge, lingbot_va_native_bridge as bridge
from wo.models.lingbot_va_native_bridge import _native_scene_check


class _Pose:
    p = np.array([0.1, -0.2, 0.8])


class _Actor:
    def __init__(self, actor_id):
        self._actor_id = actor_id

    def get_id(self):
        return self._actor_id

    def get_pose(self):
        return _Pose()


def test_stack_blocks_binding_accepts_two_object_on_goal_with_physical_support():
    task_spec = {
        "schema_version": "wo_task_spec.v1",
        "objects": [
            {"object_id": "apple", "role": "manipuland"},
            {"object_id": "serving_tray", "role": "support"},
        ],
        "steps": [{"goal": {"kind": "on", "subject": "apple", "target": "serving_tray"}}],
    }

    assert lingbot_va_bridge.stack_blocks_two_object_ids(task_spec) == {
        "block1": "serving_tray", "block2": "apple",
    }


def test_stack_blocks_binding_allows_explicit_fixed_native_table_support():
    task_spec = {
        "schema_version": "wo_task_spec.v1",
        "objects": [
            {"object_id": "apple", "category": "apple", "role": "manipuland"},
            {"object_id": "tray", "category": "tray", "role": "destination"},
            {"object_id": "table", "category": "table", "role": "support",
             "geometry_constraints": {"fixed": True}},
        ],
        "steps": [{"goal": {"kind": "on", "subject": "apple", "target": "tray"}}],
    }

    assert lingbot_va_bridge.stack_blocks_two_object_ids(task_spec) == {
        "block1": "tray", "block2": "apple",
    }

    movable_table = {**task_spec, "objects": [
        *task_spec["objects"][:2],
        {**task_spec["objects"][2], "geometry_constraints": {"fixed": False}},
    ]}
    with pytest.raises(ValueError, match="fixed native table support"):
        lingbot_va_bridge.stack_blocks_two_object_ids(movable_table)


def test_stack_blocks_binding_still_rejects_distractors_and_support_subjects():
    base = {
        "schema_version": "wo_task_spec.v1",
        "objects": [
            {"object_id": "apple", "role": "manipuland"},
            {"object_id": "tray", "role": "support"},
        ],
        "steps": [{"goal": {"kind": "on", "subject": "apple", "target": "tray"}}],
    }
    with_extra = {**base, "objects": [*base["objects"], {"object_id": "cup", "role": "manipuland"}]}
    with pytest.raises(ValueError, match="fixed native table support"):
        lingbot_va_bridge.stack_blocks_two_object_ids(with_extra)

    support_subject = {
        **base,
        "steps": [{"goal": {"kind": "on", "subject": "tray", "target": "apple"}}],
    }
    with pytest.raises(ValueError, match="subject must be a movable"):
        lingbot_va_bridge.stack_blocks_two_object_ids(support_subject)


def test_native_scene_check_measures_reference_actor_pixels_and_saves_camera_image(tmp_path):
    palette_values = sorted(set(ImageColor.colormap.values()))
    palette = np.asarray([ImageColor.getrgb(value) for value in palette_values], dtype=np.uint8)
    rgb = np.full((4, 6, 3), 127, dtype=np.uint8)
    segmentation = np.zeros((4, 6, 3), dtype=np.uint8)
    segmentation[1:3, 2:5] = palette[7]
    task = type("NativeTask", (), {
        "wo_handles": {"apple": _Actor(7)},
    })()
    obs = {"observation": {"head_camera": {"rgb": rgb, "actor_segmentation": segmentation}}}

    result = _native_scene_check(task, obs, tmp_path, {"head_camera_h": 4, "head_camera_w": 6})

    assert result["status"] == "ready"
    assert result["required_visible"] == ["apple"]
    assert result["visible_pixels"]["apple"]["head_camera"] == 6
    assert result["camera_shapes"]["head_camera"] == [4, 6, 3]
    assert (tmp_path / "cameras/head_camera.png").is_file()
    assert result["native_objects"]["apple"]["position"] == [0.1, -0.2, 0.8]
    assert json.loads((tmp_path / "scene_check.json").read_text())["status"] == "ready"


def test_native_scene_check_uses_sapien_per_scene_id_for_actor_segmentation(tmp_path):
    palette_values = sorted(set(ImageColor.colormap.values()))
    palette = np.asarray([ImageColor.getrgb(value) for value in palette_values], dtype=np.uint8)

    class SapienActor:
        # SAPIEN 2 Entity API; it does not provide get_id().
        def get_per_scene_id(self):
            return 11

        def get_pose(self):
            return _Pose()

    task = type("NativeTask", (), {"wo_handles": {"apple": SapienActor()}})()
    segmentation = np.zeros((3, 5, 3), dtype=np.uint8)
    segmentation[1, 2:4] = palette[11]
    observation = {"observation": {"head_camera": {
        "rgb": np.zeros((3, 5, 3), dtype=np.uint8),
        "actor_segmentation": segmentation,
    }}}

    result = _native_scene_check(task, observation, tmp_path, {})

    assert result["status"] == "ready"
    assert result["actor_ids"] == {"apple": 11}
    assert result["visible_pixels"]["apple"]["head_camera"] == 2


def test_native_scene_check_fails_closed_when_task_object_is_not_in_any_camera(tmp_path):
    task = type("NativeTask", (), {"wo_handles": {"plate": _Actor(9)}})()
    obs = {"observation": {"head_camera": {
        "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
        "actor_segmentation": np.zeros((2, 3, 3), dtype=np.uint8),
    }}}

    result = _native_scene_check(task, obs, tmp_path, {})

    assert result["status"] == "failed"
    assert result["missing_objects"] == ["plate"]


@pytest.mark.parametrize("policy", ["lingbot_va", "fastwam", "longwam"])
def test_reference_scene_check_initializes_native_task_without_policy_client(tmp_path, monkeypatch, policy):
    repo, source = tmp_path / "robotwin", tmp_path / "wov2"
    repo.mkdir()
    source.mkdir()
    ref = tmp_path / "reference"
    ref.mkdir()
    task_spec = tmp_path / "task.json"
    task_spec.write_text(json.dumps({"instruction": "Place the apple on the plate."}))
    reference_state = tmp_path / "reference_state.json"
    reference_state.write_text("{}")
    palette_values = sorted(set(ImageColor.colormap.values()))
    palette = np.asarray([ImageColor.getrgb(value) for value in palette_values], dtype=np.uint8)

    class NativeTask:
        def __init__(self):
            self.data_type = {}
            self.wo_handles = {"apple": _Actor(7)}
            self.closed = False

        def set_instruction(self, instruction):
            self.instruction = instruction

        def get_instruction(self):
            return self.instruction

        def get_obs(self):
            seg = np.zeros((3, 4, 3), dtype=np.uint8)
            seg[1, 1] = palette[7]
            assert self.data_type["actor_segmentation"] is True
            return {"observation": {"head_camera": {
                "rgb": np.zeros((3, 4, 3), dtype=np.uint8),
                "actor_segmentation": seg,
            }}}

        def close_env(self):
            self.closed = True

    native_task = NativeTask()
    monkeypatch.setattr(bridge.os, "chdir", lambda _path: None)
    monkeypatch.setattr(bridge, "_task_setup_inputs", lambda *_args: {})
    monkeypatch.setattr(
        lingbot_va_bridge, "initialize_reference_stack_blocks_two",
        lambda task, **_kwargs: setattr(task, "wo_handles", {"apple": _Actor(7)}),
    )
    policy_calls = []

    result = bridge.run_bridge(
        robotwin_repo=repo, wov2_source=source, out=tmp_path / "native",
        port=28991, seed=0, reference=ref, task_spec_path=task_spec,
        reference_state_path=reference_state, scene_check=True,
        policy=policy,
        task_factory=lambda: native_task, sapien_module=object(),
        client_factory=lambda: policy_calls.append("must not be called"),
    )

    assert result["status"] == "ready"
    assert result["visible_pixels"]["apple"]["head_camera"] == 1
    assert policy_calls == []
    assert native_task.closed


def test_reference_object_binding_translates_source_tabletop_frame():
    class Handle:
        def set_root_pose(self, pose):
            self.pose = pose

    state = {"schema_version": "wo_reference_state.v1", "initial_objects": {
        "tray": {"link_position": [0.1, -0.2, -0.0015],
                  "link_orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
    }}
    handle = Handle()
    pose_factory = lambda p, q: SimpleNamespace(p=list(p), q=list(q))

    evidence = lingbot_va_bridge.bind_reference_objects(
        state, {"tray": handle}, pose_factory=pose_factory,
        position_offset=(0.02, 0.03, 0.74))

    assert evidence["tray"]["source_position"] == [0.1, -0.2, -0.0015]
    assert evidence["tray"]["position"] == pytest.approx([0.12, -0.17, 0.7385])
    assert evidence["tray"]["position_offset"] == [0.02, 0.03, 0.74]
    assert handle.pose.p == evidence["tray"]["position"]
    assert handle.pose.q == [1.0, 0.0, 0.0, 0.0]


def test_reference_initialization_uses_native_table_and_does_not_duplicate_fixed_table_support(monkeypatch):
    class Pose:
        def __init__(self, p, q):
            self.p = np.asarray(p, dtype=float)
            self.q = np.asarray(q, dtype=float)

    class Actor:
        def set_root_pose(self, pose):
            self.pose = pose

        def set_root_velocity(self, velocity):
            self.velocity = velocity

    class Table:
        def get_pose(self):
            return Pose([0.12, -0.08, 0.755], [1, 0, 0, 0])

    class Task:
        def __init__(self):
            self.scene = object()
            self.table = Table()
            self.table_z_bias = -0.25  # actual entity pose, not a nominal-height guess
            self.wo_handles = {}

        def load_actors(self):
            raise AssertionError("stock randomized actors must remain bypassed")

        def setup_demo(self, **_kwargs):
            self.load_actors()

        def check_success(self):
            return False

    task = Task()
    task_spec = {
        "schema_version": "wo_task_spec.v1",
        "objects": [
            {"object_id": "apple", "category": "apple", "role": "manipuland",
             "geometry_constraints": {}},
            {"object_id": "tray", "category": "tray", "role": "destination",
             "geometry_constraints": {"fixed": True}},
            {"object_id": "table", "category": "table", "role": "support",
             "geometry_constraints": {"fixed": True}},
        ],
        "steps": [{"goal": {"kind": "on", "subject": "apple", "target": "tray"}}],
    }
    reference_state = {
        "schema_version": "wo_reference_state.v1",
        "initial_objects": {
            "apple": {"link_position": [0.1, 0.2, 1.03],
                      "link_orientation_xyzw": [0, 0, 0, 1]},
            "tray": {"link_position": [0.0, 0.0, 1.0],
                     "link_orientation_xyzw": [0, 0, 0, 1]},
            "table": {"link_position": [0.0, 0.0, 0.0],
                      "link_orientation_xyzw": [0, 0, 0, 1]},
        },
        "measured_dynamics": {"apple": {"mass_kg": 0.1}, "tray": {"mass_kg": 1.0},
                              "table": {"mass_kg": 100.0}},
    }
    monkeypatch.setattr(lingbot_va_bridge, "_verify_bound_reference", lambda *_args: None)
    monkeypatch.setattr(lingbot_va_bridge, "prepared_sdf_for_object", lambda *_args: "asset.sdf")
    monkeypatch.setattr(lingbot_va_bridge, "sdf_collision_vertices_model",
                        lambda *_args: np.array([[0.0, 0.0, 1.0]]))

    result = lingbot_va_bridge.initialize_reference_stack_blocks_two(
        task, reference="ref", task_spec=task_spec, reference_state=reference_state,
        setup_kwargs={}, actor_loader=lambda *_args, **_kwargs: Actor(),
        dynamics_binder=lambda *_args: None, sapien_module=SimpleNamespace(Pose=Pose),
    )

    assert set(task.wo_handles) == {"apple", "tray"}  # table is native environment geometry
    assert task.wo_handles["apple"].pose.p == pytest.approx([0.22, 0.12, 0.785])
    assert task.wo_handles["tray"].pose.p == pytest.approx([0.12, -0.08, 0.755])
    assert result["initial_pose_evidence"]["apple"]["position_offset"] == pytest.approx(
        [0.12, -0.08, -0.245])
