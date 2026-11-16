"""LingBot-VA adapter: scene plan, layout, predicates world hook, CAE TCP, not_applicable."""

import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from wo.models import lingbot_va as L
from wo.models import lingbot_va_client as C
from wo.spec import Predicate, Step, TaskObject, TaskSpec

def test_runtime_python_uses_profile_venv_launcher_when_environment_was_resolved(tmp_path, monkeypatch):
    env_root = tmp_path / "runtime-env"
    launcher = env_root / "bin/python"
    site = env_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    launcher.parent.mkdir(parents=True)
    site.mkdir(parents=True)
    launcher.symlink_to(sys.executable)
    (env_root / "pyvenv.cfg").write_text(
        f"home = {Path(sys.executable).parent}\ninclude-system-site-packages = false\n"
    )
    (site / "wo_venv_probe.py").write_text("VALUE = 'venv-site-packages'\n")
    config = tmp_path / "two-interface.json"
    config.write_text(json.dumps({"tested_model": {"robotwin_python": str(launcher)}}))
    monkeypatch.setenv("WO_EVAL_CONFIG", str(config))
    # Simulate an outer launcher that already replaced the configured symlink
    # with its shared interpreter target.
    monkeypatch.setenv("WO_ROBOTWIN_PYTHON", os.path.realpath(launcher))

    resolved = L._python_launcher("WO_ROBOTWIN_PYTHON", "robotwin_python", tmp_path / "fallback")
    assert resolved == launcher
    probe = subprocess.run(
        [str(resolved), "-c", "import sys, wo_venv_probe; print(sys.prefix); print(wo_venv_probe.VALUE)"],
        check=True, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ""},
    )
    assert probe.stdout.splitlines() == [str(env_root), "venv-site-packages"]


CUBE_OBJ = "\n".join(
    [f"v {x} {y} {z}" for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    + ["f 1 2 4 3", "f 5 7 8 6", "f 1 5 6 2", "f 3 4 8 7", "f 1 3 7 5", "f 2 6 8 4"]) + "\n"


def test_server_startup_timeout_is_configurable(monkeypatch):
    monkeypatch.delenv(L.SERVER_STARTUP_TIMEOUT_ENV, raising=False)
    assert L._server_startup_timeout() == 3600.0

    monkeypatch.setenv(L.SERVER_STARTUP_TIMEOUT_ENV, "2400")
    assert L._server_startup_timeout() == 2400.0

    monkeypatch.setenv(L.SERVER_STARTUP_TIMEOUT_ENV, "0")
    with pytest.raises(ValueError, match="must be a positive number"):
        L._server_startup_timeout()



def test_nvidia_vulkan_icd_selects_loadable_standard_path(monkeypatch, tmp_path):
    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)
    etc_icd = tmp_path / "etc" / "vulkan" / "icd.d" / "nvidia_icd.json"
    share_icd = tmp_path / "usr" / "share" / "vulkan" / "icd.d" / "nvidia_icd.json"
    etc_icd.parent.mkdir(parents=True, exist_ok=True)
    etc_icd.write_text(json.dumps({"ICD": {"library_path": "/missing/libGLX_nvidia.so.0"}}))
    share_icd.parent.mkdir(parents=True, exist_ok=True)
    share_icd.write_text(json.dumps({"ICD": {"library_path": "libGLX_nvidia.so.0"}}))
    monkeypatch.setattr(L, "VULKAN_ICD_CANDIDATES", (etc_icd, share_icd))
    monkeypatch.setattr(L.ctypes.util, "find_library", lambda name: "libGLX_nvidia.so.0" if name == "GLX_nvidia" else None)

    assert L._nvidia_vulkan_icd_path() == str(share_icd)


def test_nvidia_vulkan_icd_fails_early_when_driver_library_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)
    icd = tmp_path / "nvidia_icd.json"
    icd.write_text(json.dumps({"ICD": {"library_path": "libGLX_nvidia.so.0"}}))
    monkeypatch.setattr(L, "VULKAN_ICD_CANDIDATES", (icd,))
    monkeypatch.setattr(L.ctypes.util, "find_library", lambda _name: None)

    with pytest.raises(RuntimeError, match="No loadable NVIDIA Vulkan ICD"):
        L._nvidia_vulkan_icd_path()



def _sdf(path: Path, name: str, size: tuple, mass: float = 0.1, visual: str = "m.gltf") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "c.obj").write_text(CUBE_OBJ)
    sc = " ".join(str(v) for v in size)
    path.write_text(f"""<?xml version='1.0'?><sdf version="1.7"><model name="{name}"><link name="base_link">
<inertial><mass>{mass}</mass><inertia><ixx>0.0001</ixx><iyy>0.0001</iyy><izz>0.0001</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
<visual name="v"><geometry><mesh><uri>{visual}</uri><scale>{sc}</scale></mesh></geometry></visual>
<collision name="c"><geometry><mesh><uri>{path.parent / 'c.obj'}</uri><scale>{sc}</scale></mesh></geometry></collision>
</link></model></sdf>""")
    return path


def make_ref(tmp: Path, *, spread: float = 0.3, articulated_support: bool = False) -> Path:
    ref = tmp / "ref"
    objs = {"block": (0.04, 0.04, 0.04), "pad": (0.2, 0.2, 0.01)}
    top = 0.95
    base = (0.0, -0.45, top)
    scene = {"case_id": "c0", "robot_base_position": list(base), "robot_base_yaw": math.pi / 2,
             "robot_urdf": "unused_objects_only.urdf",
             "support": {"object_id": "table", "role": "support", "fixed": True, "scale": 1.0,
                         "position": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1],
                         "sdf_path": str(_sdf(tmp / "assets/table/t.sdf", "t", (1.0, 1.0, 2 * top)))},
             "objects": {}, "articulations": {}}
    for i, (oid, size) in enumerate(objs.items()):
        src = _sdf(tmp / f"assets/{oid}/{oid}.sdf", oid, size)
        _sdf(ref / "physics_sdf" / f"{oid}_s1.000" / f"{oid}.sdf", oid, size)
        x = (-0.5 + i) * spread
        scene["objects"][oid] = {"object_id": oid, "role": "manipuland" if oid == "block" else "destination",
                                 "fixed": False, "scale": 1.0, "sdf_path": str(src),
                                 "position": [x, 0.08, top + size[2] / 2], "orientation_xyzw": [0, 0, 0, 1]}
    if articulated_support:
        scene["articulations"] = {"door": {"host": "table", "joint_name": "hinge", "child_link": "door"}}
    ref.mkdir(parents=True, exist_ok=True)
    # Canonical prepared physics and recorded link poses are required inputs.
    states = {}
    for row in [scene["support"], *scene["objects"].values()]:
        oid = row["object_id"]
        physics = ref / "physics_sdf" / f"{oid}_s1.000" / Path(row["sdf_path"]).name
        if not physics.exists():
            physics.parent.mkdir(parents=True, exist_ok=True)
            physics.write_text(Path(row["sdf_path"]).read_text())
        tree = ET.parse(physics)
        for link in tree.getroot().iter("link"):
            for visual in list(link.findall("visual")):
                link.remove(visual)
        tree.write(physics)
        (physics.parent / ".path").write_text(str(physics))
        states[oid] = {"link_position": row["position"], "link_orientation_xyzw": row["orientation_xyzw"],
                       "joint_positions": {}}
    (ref / "trajectory.json").write_text(json.dumps({"frames": [{"objects": states}]}))
    (ref / "scene_inputs.json").write_text(json.dumps(scene))
    spec = TaskSpec("c0", "pick_place", "put the block on the pad",
                    (TaskObject("block", "block", "manipuland"), TaskObject("pad", "pad", "destination"),
                     TaskObject("table", "table", "support")),
                    (Step("transport", "block", Predicate("on_region", "block", "pad"), target="pad"),))
    (tmp / "task.json").write_text(json.dumps(spec.to_dict()))
    (ref / "certificate.json").write_text(json.dumps({"task": str(tmp / "task.json"), "family": "pick_place"}))
    return ref


def test_plan_rebases_scene_onto_robotwin_table(tmp_path):
    ref = make_ref(tmp_path)
    plan, spec = L.build_plan(ref, tmp_path / "out", window_s=60, seed=3)
    objs = {o["object_id"]: o for o in plan["objects"]}
    assert set(objs) == {"block", "pad"} and plan["support_id"] == "table"
    for o in objs.values():  # bottom on the native table top, inside the dual-arm workspace
        bottom = o["pose_rt"]["p"][2] + o["local_bounds"][0][2]
        assert bottom >= L.TABLE_TOP_RT + 0.003 - 1e-6
        x, y = o["pose_rt"]["p"][:2]
        assert L.WORKSPACE_RT[0][0] <= x <= L.WORKSPACE_RT[0][1]
        assert L.WORKSPACE_RT[1][0] <= y <= L.WORKSPACE_RT[1][1]
    # lateral layout relative to the robot is preserved (no compression needed)
    assert plan["transform"]["xy_scale"] == 1.0
    dx = objs["pad"]["pose_rt"]["p"][0] - objs["block"]["pose_rt"]["p"][0]
    assert dx == pytest.approx(0.3)
    # glTF visuals: Y-up -> Z-up basis, scale permuted into the glTF frame
    vis = objs["pad"]["visuals"][0]
    assert vis["q_wxyz"] == pytest.approx([math.sqrt(0.5), math.sqrt(0.5), 0, 0])
    assert vis["scale"] == pytest.approx([0.2, 0.01, 0.2])
    assert objs["block"]["mass"] == pytest.approx(0.1)
    assert plan["goals"] == [{"kind": "on_region", "subject": "block", "target": "pad", "params": {},
                              "require_release": True, "primitive": "transport"}]
    assert plan["required_visible"] == ["block", "pad"]
    graph = json.loads((tmp_path / "out/task_graph.json").read_text())
    assert graph["success_program"][0]["predicate"] == "on_region"
    assert json.loads((tmp_path / "out/lingbot/scene_plan.json").read_text())["seed"] == 3


def test_reference_frame_roundtrip(tmp_path):
    ref = make_ref(tmp_path)
    plan, _ = L.build_plan(ref, tmp_path / "out", window_s=10, seed=0)
    frame = C.Frame(plan["transform"])
    scene = json.loads((ref / "scene_inputs.json").read_text())
    for o in plan["objects"]:
        src = scene["objects"][o["object_id"]]
        assert frame.to_ref(o["pose_rt"]["p"]) == pytest.approx(src["position"], abs=1e-9)
        assert np.abs(frame.rot_to_ref(o["pose_rt"]["q_wxyz"])) == pytest.approx(
            np.abs(src["orientation_xyzw"]), abs=1e-9)


def test_native_table_lift_uses_measured_collision_bounds():
    tf = {"yaw": 0.0, "xy_scale": 1.0, "xy_offset": [0.0, 0.0], "z_offset": 0.0}
    objects = [{"object_id": "box", "fixed": False, "role": "manipuland",
                "position": [0.0, 0.0, 0.741],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                "local_bounds": [[-0.1, -0.1, -0.01], [0.1, 0.1, 0.01]]}]
    delta = L.lift_native_table_objects(tf, objects)
    assert delta == pytest.approx(0.012)
    assert tf["z_offset"] == pytest.approx(delta)
    # Already-clear objects are not moved, avoiding a blanket/task-specific lift.
    tf2 = {"yaw": 0.0, "xy_scale": 1.0, "xy_offset": [0.0, 0.0], "z_offset": 0.0}
    objects[0]["position"][2] = 0.80
    assert L.lift_native_table_objects(tf2, objects) == pytest.approx(0.0)


def test_wide_layout_is_compressed_or_not_applicable(tmp_path):
    plan, _ = L.build_plan(make_ref(tmp_path / "a", spread=0.6), tmp_path / "a/out", window_s=10, seed=0)
    assert L.MIN_LAYOUT_SCALE <= plan["transform"]["xy_scale"] < 1.0
    with pytest.raises(L.Unsupported):
        L.build_plan(make_ref(tmp_path / "b", spread=2.0), tmp_path / "b/out", window_s=10, seed=0)


def test_unsupported_writes_not_applicable_episode(tmp_path, monkeypatch):
    from wo.eval import case, report

    ref = make_ref(tmp_path, articulated_support=True)
    (ref / "physics_sdf/table_s1.000/t.sdf").unlink()
    out = tmp_path / "ep"
    monkeypatch.setattr(sys, "argv", ["lingbot_va", "--ref", str(ref), "--out", str(out), "--seed", "2"])
    L.main()  # raises Unsupported before any server is started
    ep = json.loads((out / "episode.json").read_text())
    assert ep["model"] == "lingbot_va" and ep["status"] == "not_applicable" and ep["task_success"] is None
    # The articulated-support path is attempted; this minimal fixture lacks
    # the authoritative support physics artifact, so it fails closed rather
    # than silently substituting the RoboTwin table.
    assert ep["termination_reason"].startswith("not_applicable: table: missing expert physics asset")
    assert case.score(case.build_case(ref, out, out / "case"))["status"] == "pending"
    assert report.episode_result(out)["status"] == "pending"


def test_predicates_use_native_support_hook():
    from wo.predicates import support_below

    class World:
        def support_below(self, oid):
            return "pad"

    assert support_below(World(), "block") == "pad"


def test_decode_action_composes_initial_pose():
    init = np.array([0.1, 0.2, 0.9, 0, 0, 0, 1, 1.0, -0.1, 0.2, 0.9, 0, 0, 0, 1, 0.0])
    raw = np.array([0.01, 0, 0, 0, 0, 0, 1, 0.5, 0, 0.02, 0, 0, 0, 0, 1, 0.2])
    out = C.decode_action(raw, init)
    assert out[:3] == pytest.approx([0.11, 0.2, 0.9]) and out[7] == pytest.approx(0.5)
    assert out[8:11] == pytest.approx([-0.1, 0.22, 0.9]) and out[15] == pytest.approx(0.2)


def test_cae_uses_recorded_tcp_and_time_for_non_franka(tmp_path):
    from wo.eval.cae_episodes import entity_tracks

    frames = [{"time_s": i / 40.0, "robot": {"base_position": [0, -0.65, 0], "base_orientation_xyzw": [0, 0, 0, 1],
                                             "joint_names": ["fl_joint1", "fr_joint1"], "joint_positions": [0, 0],
                                             "end_effector_position": [0.01 * i, 0, 0.9]},
               "objects": {"block": {"link_position": [0, 0, 0.8], "link_orientation_xyzw": [0, 0, 0, 1]}}}
              for i in range(5)]
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"frames": frames}))
    tr = entity_tracks(path, ["block"])
    assert tr["end_effector"].positions[:, 0] == pytest.approx([0, 0.01, 0.02, 0.03, 0.04])
    assert tr["end_effector"].timestamps[-1] == pytest.approx(0.1)


def test_registered_in_pipeline():
    from wo.agent.stages import MODELS

    assert MODELS["lingbot_va"] == "wo.models.lingbot_va"


def test_primary_arm_is_the_one_moving_the_subject():
    def fr(i, block, left, right):
        return {"time_s": i, "objects": {"block": {"link_position": block}},
                "robot": {"left_end_effector_position": left, "right_end_effector_position": right}}

    frames = [fr(0, [0, 0, 0], [0, 0, 0.1], [1, 0, 0.1])]
    frames += [fr(i, [0.05 * i, 0, 0], [0.05 * i, 0, 0.05], [1, 0.2 * i, 0.5]) for i in range(1, 5)]
    assert C.set_primary_arm(frames, ["block"]) == "left"  # right travelled further but never near
    assert frames[-1]["robot"]["end_effector_position"] == [0.2, 0, 0.05]


def test_opt_in_reference_support_keeps_authored_geometry_and_material(tmp_path):
    ref = make_ref(tmp_path)
    task = tmp_path / 'task.json'
    data = json.loads(task.read_text())
    data['objects'][-1]['geometry_constraints'] = {'preserve_native_geometry': True}
    task.write_text(json.dumps(data))
    plan, _ = L.build_plan(ref, tmp_path / 'out', window_s=60, seed=0)
    table = next(o for o in plan['objects'] if o['object_id'] == 'table')
    assert plan['support_representation'] == 'scene_asset'
    assert table['fixed'] is True and table['role'] == 'support'
    assert table['local_bounds'] == [[-.5,-.5,-.95],[.5,.5,.95]]
    assert table['visuals'][0]['file'].endswith('assets/table/m.gltf')
    assert plan['transform']['xy_scale'] == 1.0
    assert plan['required_visible'] == ['block', 'pad']
