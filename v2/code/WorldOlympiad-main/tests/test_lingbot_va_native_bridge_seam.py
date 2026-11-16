import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from wo.models import lingbot_va, lingbot_va_native_bridge
from wo.models.lingbot_va_bridge import (
    RoboTwinRuntime,
    bind_reference_objects,
    bind_stack_blocks_two,
    build_bridge_command,
    policy_check_receipt,
    runtime_from_profile,
    validate_capture_metadata,
    initialize_reference_stack_blocks_two,
    _verify_bound_reference,
    native_on_predicate,
    sdf_collision_vertices_model,
    stack_blocks_two_object_ids,
)


def _manifest(root: Path):
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def _runtime(tmp_path: Path) -> RoboTwinRuntime:
    workspace = tmp_path / "workspace"
    repo = workspace / "runtime" / "RoboTwin"
    python = workspace / "envs" / "robotwin" / "bin" / "python"
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    python.write_text("#!/python\n")
    profile = workspace / "profile.json"
    profile.write_text(json.dumps({
        "workspace_root": str(workspace),
        "tested_model": {"robotwin_repo": "runtime/RoboTwin",
                         "robotwin_python": "envs/robotwin/bin/python"},
    }))
    return runtime_from_profile(profile)


def test_profile_paths_and_bridge_command_are_absolute_and_reference_bound(tmp_path):
    runtime = _runtime(tmp_path)
    assert runtime.repo == (runtime.workspace / "runtime/RoboTwin").resolve()
    assert runtime.python == (runtime.workspace / "envs/robotwin/bin/python").resolve()
    ref = tmp_path / "ref"
    output = tmp_path / "episode"
    script = tmp_path / "lingbot_va_native_bridge.py"
    command = build_bridge_command(runtime, script, stock_task="stack_blocks_two",
                                   reference=ref, output=output, port=28004,
                                   seed=7, window_s=60, prompt="stack the blocks")
    assert command[:4] == [str(runtime.python), str(script.resolve()),
                           "--robotwin-repo", str(runtime.repo)]
    assert command[command.index("--stock-task") + 1] == "stack_blocks_two"
    assert command[command.index("--ref") + 1] == str(ref.resolve())
    assert command[command.index("--out") + 1] == str(output.resolve())
    assert command[command.index("--port") + 1] == "28004"
    assert command[command.index("--prompt") + 1] == "stack the blocks"
    with pytest.raises(ValueError, match="admits only"):
        build_bridge_command(runtime, script, stock_task="another_task", reference=ref,
                             output=output, port=28004)


def test_terminal_partial_chunk_is_not_sent_to_streaming_policy_context():
    class Client:
        def __init__(self):
            self.calls = []
        def infer(self, payload):
            self.calls.append(payload)

    class Task:
        eval_success = False
        take_action_cnt = 9

    client = Client()
    frames, action = [{"frame": 1}], object()
    # Continuing rollouts still forward the exact native observations/action.
    assert lingbot_va_native_bridge._update_policy_context(
        client, Task(), 10, frames, action)
    assert client.calls == [{"obs": frames, "compute_kv_cache": True,
                             "imagine": False, "save_visualization": False,
                             "state": action}]

    # A success or an action-limit boundary can leave an incomplete frame chunk;
    # neither needs another policy context update.
    client.calls.clear()
    Task.eval_success = True
    assert not lingbot_va_native_bridge._update_policy_context(
        client, Task(), 10, frames[:0], action)
    Task.eval_success = False
    Task.take_action_cnt = 10
    assert not lingbot_va_native_bridge._update_policy_context(
        client, Task(), 10, frames[:0], action)
    assert client.calls == []


def test_episode_adapter_passes_archived_reference_objects_to_native_client(tmp_path, monkeypatch):
    ref, out = tmp_path / "bound-ref", tmp_path / "episode"
    ref.mkdir()
    spec, state = tmp_path / "task_spec.json", tmp_path / "reference_state.json"
    spec.write_text("{}")
    state.write_text("{}")
    monkeypatch.setattr(lingbot_va, "_reference_bridge_inputs", lambda *_: (spec, state))
    monkeypatch.setattr(lingbot_va, "start_server", lambda *args, **kwargs: SimpleNamespace(poll=lambda: 0))
    monkeypatch.setattr(lingbot_va, "_wait_port", lambda *args: None)
    captured = {}
    def run_client(_out, args, *, gpu):
        captured.update(args=args, gpu=gpu)
        receipt = out / "episode.json"
        out.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"model": "lingbot_va", "reference": str(ref.resolve())}))
        return 0
    monkeypatch.setattr(lingbot_va, "run_client", run_client)
    result = lingbot_va.run_episode(ref, out, gpu=3, port=28005, seed=4, window_s=9)
    assert result["reference"] == str(ref.resolve())
    args = captured["args"]
    for option, expected in (("--ref", ref.resolve()), ("--task-spec", spec.resolve()),
                             ("--reference-state", state.resolve())):
        assert Path(args[args.index(option) + 1]) == expected
    assert args[args.index("--window-s") + 1] == "9"
    assert captured["gpu"] == 3


def test_reference_binding_requires_exact_object_ids_and_uses_sapien_quaternion_order():
    state = {"schema_version": "wo_reference_state.v1", "initial_objects": {
        "red_ref": {"link_position": [0.1, -0.2, 0.8],
                    "link_orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "green_ref": {"link_position": [-0.1, 0.2, 0.85],
                      "link_orientation_xyzw": [0.0, 0.0, 0.7071067811865476,
                                                  0.7071067811865476]},
    }}
    calls = {}

    class Handle:
        def __init__(self, name):
            self.name = name

        def set_root_pose(self, pose):
            calls[self.name] = {"pose": pose}

        def set_root_velocity(self, velocity):
            calls[self.name]["velocity"] = velocity

    handles = {key: Handle(key) for key in state["initial_objects"]}
    evidence = bind_reference_objects(
        state, handles, pose_factory=lambda p, q: {"p": list(p), "q_wxyz": list(q)})
    assert calls["red_ref"]["pose"] == {"p": [0.1, -0.2, 0.8], "q_wxyz": [1, 0, 0, 0]}
    assert calls["green_ref"]["pose"]["q_wxyz"] == pytest.approx(
        [0.7071067811865476, 0, 0, 0.7071067811865476])
    assert calls["green_ref"]["velocity"] == [0.0] * 6
    assert evidence["green_ref"]["pose_source"] == "WOV2 reference frame0 link pose"
    with pytest.raises(ValueError, match="identity mismatch"):
        bind_reference_objects(state, {"red_ref": handles["red_ref"]}, pose_factory=lambda p, q: (p, q))


def test_stack_task_fields_bind_explicit_reference_handles_without_replacing_predicate():
    predicate = lambda: "upstream native predicate"
    task = SimpleNamespace(check_success=predicate)
    refs = {"r": object(), "g": object()}
    binding = bind_stack_blocks_two(task, refs, {"block1": "r", "block2": "g"})
    assert task.block1 is refs["r"] and task.block2 is refs["g"]
    assert task.check_success is predicate
    assert binding == {"block1": "r", "block2": "g"}
    with pytest.raises(ValueError, match="explicit"):
        bind_stack_blocks_two(task, refs, {"block1": "r"})


def test_native_capture_evidence_is_preserved_in_policy_check_receipt():
    metadata = {
        "task_success": False,
        "completion_evidence": {"native_predicate_evidence": {
            "predicate_source": "upstream RoboTwin check_success",
            "result": False, "task": "stack_blocks_two"}},
        "capture": {"video": "videos/head.mp4", "frames": 12},
    }
    assert validate_capture_metadata(metadata) == metadata
    receipt = policy_check_receipt("stack_blocks_two", "lingbot_va", metadata)
    assert receipt == {"control_only": True, "task": "stack_blocks_two",
                       "model": "lingbot_va", "native_result": metadata}
    with pytest.raises(ValueError, match="measured task result"):
        validate_capture_metadata({**metadata, "task_success": 0})
    bad = json.loads(json.dumps(metadata))
    bad["completion_evidence"]["native_predicate_evidence"]["predicate_source"] = "adapter estimate"
    with pytest.raises(ValueError, match="native task predicate"):
        validate_capture_metadata(bad)


def test_native_wov2_on_goal_mapping_and_geometry_contact_predicate():
    spec = {"schema_version": "wo_task_spec.v1", "objects": [
        {"object_id": "apple", "role": "manipuland"},
        {"object_id": "coaster", "role": "destination"},
        {"object_id": "worktable", "role": "support"}],
        "steps": [{"goal": {"kind": "on", "subject": "apple", "target": "coaster"}}]}
    assert stack_blocks_two_object_ids(spec) == {"block1": "coaster", "block2": "apple"}

    apple = ([-0.0275, -0.0275, 0.015], [0.0275, 0.0275, 0.070])
    coaster = ([-0.070, -0.070, 0.0], [0.070, 0.070, 0.010])
    passed, evidence = native_on_predicate(apple, coaster, touching=True)
    assert passed and evidence["gap_to_target_top_m"] == pytest.approx(0.005)
    assert evidence["xy_overlap"] == pytest.approx(1.0)
    # A side collision or a geometrically aligned but noncontacting object is
    # not task completion.
    side_contact = ([0.071, -0.01, 0.0], [0.126, 0.045, 0.055])
    assert not native_on_predicate(side_contact, coaster, touching=True)[0]
    assert not native_on_predicate(apple, coaster, touching=False)[0]


def test_collision_only_prepared_sdf_uses_bound_mesh_as_native_visual(tmp_path):
    from wo.models.lingbot_va_bridge import add_sdf_mesh_actor

    mesh = tmp_path / "collision.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    sdf = tmp_path / "asset.sdf"
    sdf.write_text(
        "<sdf><model name='asset'><link name='root'>"
        "<collision name='collision'><pose>0.1 0.2 0.3 0 0 0</pose><geometry><mesh>"
        f"<uri>{mesh}</uri><scale>0.5 0.6 0.7</scale>"
        "</mesh></geometry></collision></link></model></sdf>")

    class Pose:
        def __init__(self, p=None, q=None):
            self.p, self.q = p, q

    class Builder:
        def __init__(self):
            self.visuals, self.collisions = [], []
        def set_physx_body_type(self, value): self.body_type = value
        def add_visual_from_file(self, **kwargs): self.visuals.append(kwargs)
        def add_multiple_convex_collisions_from_file(self, **kwargs): self.collisions.append(kwargs)
        def build(self, name): return name

    class Scene:
        def __init__(self): self.builder = Builder()
        def create_actor_builder(self): return self.builder

    scene = Scene()
    actor = add_sdf_mesh_actor(scene, sdf, "apple", fixed=False,
                               sapien_module=SimpleNamespace(Pose=Pose))
    assert actor == "apple"
    assert scene.builder.body_type == "dynamic"
    assert len(scene.builder.visuals) == len(scene.builder.collisions) == 1
    visual, collision = scene.builder.visuals[0], scene.builder.collisions[0]
    assert visual["filename"] == collision["filename"] == str(mesh)
    assert visual["scale"] == collision["scale"] == [0.5, 0.6, 0.7]
    assert visual["pose"].p == collision["pose"].p == pytest.approx([0.1, 0.2, 0.3])


def test_native_collision_vertices_include_sdf_scale_and_pose(tmp_path):
    obj = tmp_path / "cube.obj"
    obj.write_text("v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nv -1 1 -1\n"
                   "v -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1\n"
                   "f 1 2 3 4\nf 5 8 7 6\nf 1 5 6 2\nf 2 6 7 3\n"
                   "f 3 7 8 4\nf 5 1 4 8\n")
    sdf = tmp_path / "cube.sdf"
    sdf.write_text("<sdf><model name='cube'><link name='root'>"
                   "<collision name='c'><pose>0.2 0 0 0 0 0</pose><geometry><mesh>"
                   f"<uri>{obj}</uri><scale>0.1 0.2 0.3</scale></mesh></geometry></collision>"
                   "</link></model></sdf>")
    vertices = sdf_collision_vertices_model(sdf)
    assert vertices.min(axis=0) == pytest.approx([0.1, -0.2, -0.3])
    assert vertices.max(axis=0) == pytest.approx([0.3, 0.2, 0.3])


def test_reference_setup_binds_on_goal_to_native_collision_predicate(tmp_path):
    ref = tmp_path / "ref"
    physics = ref / "physics_sdf"
    vertices = "v -0.5 -0.5 -0.5\nv 0.5 -0.5 -0.5\nv 0.5 0.5 -0.5\nv -0.5 0.5 -0.5\n" \
               "v -0.5 -0.5 0.5\nv 0.5 -0.5 0.5\nv 0.5 0.5 0.5\nv -0.5 0.5 0.5\n" \
               "f 1 2 3 4\nf 5 8 7 6\nf 1 5 6 2\nf 2 6 7 3\nf 3 7 8 4\nf 5 1 4 8\n"
    for oid in ("coaster", "apple"):
        folder = physics / f"{oid}_s1.000"
        folder.mkdir(parents=True)
        (folder / "mesh.obj").write_text(vertices)
        sdf = folder / f"{oid}.sdf"
        sdf.write_text("<sdf><model name='obj'><link name='root'><collision name='c'>"
                       "<geometry><mesh><uri>mesh.obj</uri><scale>0.04 0.04 0.04</scale>"
                       "</mesh></geometry></collision></link></model></sdf>")
        (folder / ".path").write_text(str(sdf.resolve()))
    task_spec = {"schema_version": "wo_task_spec.v1", "objects": [
        {"object_id": "coaster", "role": "destination"},
        {"object_id": "apple", "role": "manipuland"}],
        "steps": [{"goal": {"kind": "on", "subject": "apple", "target": "coaster"}}]}
    initial = {"coaster": {"link_position": [0, 0, 0],
                             "link_orientation_xyzw": [0, 0, 0, 1]},
               "apple": {"link_position": [0, 0, 0.05],
                         "link_orientation_xyzw": [0, 0, 0, 1]}}
    state = {"schema_version": "wo_reference_state.v1", "reference": str(ref.resolve()),
             "source_sha256": _manifest(ref), "initial_objects": initial,
             "measured_dynamics": {oid: {"root": {}} for oid in initial}}

    class Pose:
        def __init__(self, p=None, q=None):
            self.p, self.q = list(p or [0, 0, 0]), list(q or [1, 0, 0, 0])
    class Handle:
        def __init__(self, name): self.name, self.pose = name, Pose()
        def set_pose(self, pose): self.pose = pose
        def get_pose(self): return self.pose
        def set_velocity(self, value): pass
        def set_angular_velocity(self, value): pass
    class Task:
        def __init__(self):
            self.scene, self.table_z_bias = object(), 0.0
            self.eval_success = False
        def load_actors(self): raise AssertionError("stock random reset must not run")
        def setup_demo(self, **kwargs): self.load_actors()
        def check_success(self): return False
        def check_actors_contact(self, first, second):
            return {first, second} == {"apple", "coaster"}

    task = Task()
    evidence = initialize_reference_stack_blocks_two(
        task, reference=ref, task_spec=task_spec, reference_state=state, setup_kwargs={},
        actor_loader=lambda scene, sdf, oid, **kwargs: Handle(oid),
        dynamics_binder=lambda *args: None, sapien_module=SimpleNamespace(Pose=Pose))
    assert evidence["native_predicate"].startswith("WOV2 on predicate")
    assert task.check_success() is True
    assert task.wo_goal_evidence["touching"] is True
    assert task.wo_goal_evidence["xy_overlap"] == pytest.approx(1.0)


def test_actual_setup_hook_imports_reference_meshes_and_bypasses_stock_random_reset(tmp_path):
    ref = tmp_path / "reference"
    (ref / "physics_sdf").mkdir(parents=True)
    for oid, xyz in (("bottom", [0.1, 0.0, 0.8]), ("top", [0.1, 0.0, 0.85])):
        folder = ref / "physics_sdf" / f"{oid}_s0.050"
        folder.mkdir()
        mesh = folder / "mesh.obj"
        mesh.write_text("# fake mesh geometry\n")
        sdf = folder / "asset.sdf"
        sdf.write_text(
            "<sdf version='1.7'><model name='asset'><link name='root'>"
            "<visual name='visual'><geometry><mesh><uri>mesh.obj</uri>"
            "</mesh></geometry></visual>"
            "<collision name='collision'><geometry><mesh><uri>mesh.obj</uri>"
            "</mesh></geometry></collision>"
            "</link></model></sdf>")
        (folder / ".path").write_text(str(sdf))

    spec = {"schema_version": "wo_task_spec.v1", "objects": [
        {"object_id": "bottom", "role": "manipuland", "geometry_constraints": {}},
        {"object_id": "top", "role": "manipuland", "geometry_constraints": {}},
    ], "steps": [{"goal": {"kind": "stacked", "subject": "top", "target": "bottom"}}]}
    state = {"schema_version": "wo_reference_state.v1", "reference": str(ref.resolve()),
             "source_sha256": _manifest(ref), "initial_objects": {
        "bottom": {"link_position": [0.1, 0, 0.8], "link_orientation_xyzw": [0, 0, 0, 1]},
        "top": {"link_position": [0.1, 0, 0.85], "link_orientation_xyzw": [0, 0, 0, 1]},
    }, "measured_dynamics": {"bottom": {"root": {}}, "top": {"root": {}}}}

    class Pose:
        def __init__(self, p=None, q=None):
            self.p, self.q = list(p or [0, 0, 0]), list(q or [1, 0, 0, 0])

    class NativeHandle:
        def __init__(self, name):
            self.name, self.pose = name, Pose()
            self.linear = self.angular = None

        def set_pose(self, pose): self.pose = pose
        def get_pose(self): return self.pose
        def set_velocity(self, value): self.linear = list(value)
        def set_angular_velocity(self, value): self.angular = list(value)

    class Builder:
        def __init__(self): self.calls = []
        def set_physx_body_type(self, value): self.calls.append(("body", value))
        def add_visual_from_file(self, **kw): self.calls.append(("visual", kw))
        def add_multiple_convex_collisions_from_file(self, **kw): self.calls.append(("collision", kw))
        def build(self, name):
            assert sum(call[0] == "visual" for call in self.calls) == 1
            assert sum(call[0] == "collision" for call in self.calls) == 1
            return NativeHandle(name)

    class Scene:
        def create_actor_builder(self): return Builder()

    predicate = lambda self: "native check_success"

    class Task:
        check_success = predicate
        def __init__(self):
            self.scene, self.table_z_bias = Scene(), 0.0
            self.stock_load_calls = 0
            self.block1 = self.block2 = None
        def load_actors(self): self.stock_load_calls += 1
        def setup_demo(self, **kwargs):
            self.load_actors()
            # Model the native settle phase; the bridge must restore frame zero.
            for handle in self.wo_handles.values():
                handle.set_pose(Pose([9, 9, 9], [1, 0, 0, 0]))

    bound_dynamics = []
    task = Task()
    evidence = initialize_reference_stack_blocks_two(
        task, reference=ref, task_spec=spec, reference_state=state, setup_kwargs={"seed": 3},
        sapien_module=SimpleNamespace(Pose=Pose),
        dynamics_binder=lambda scene, handle, dynamics, pose_factory: bound_dynamics.append(
            (handle.name, dynamics, pose_factory)),
    )
    assert task.stock_load_calls == 0
    assert task.check_success.__func__ is predicate
    assert task.block1 is task.wo_handles["bottom"]
    assert task.block2 is task.wo_handles["top"]
    assert task.block1.get_pose().p == [0.1, 0, 0.8]
    assert task.block2.get_pose().p == [0.1, 0, 0.85]
    assert task.block1.linear == [0, 0, 0] and task.block2.angular == [0, 0, 0]
    assert len(bound_dynamics) == 2
    assert evidence["stock_random_actor_reset_bypassed"] is True
    assert evidence["native_predicate"] == "upstream RoboTwin check_success"


def test_reference_integrity_accepts_only_hash_bound_sdf_asset_closure(tmp_path):
    ref = tmp_path / "reference"
    sdf_dir = ref / "physics_sdf" / "apple_s1.000"
    sdf_dir.mkdir(parents=True)
    mesh = tmp_path / "generated" / "apple.obj"
    mesh.parent.mkdir()
    mesh.write_text("# generated mesh\\n")
    sdf = sdf_dir / "apple.sdf"
    sdf.write_text(
        "<sdf version='1.7'><model name='apple'><link name='root'>"
        "<visual name='visual'><geometry><mesh><uri>../../../generated/apple.obj</uri>"
        "</mesh></geometry></visual></link></model></sdf>")
    marker = sdf_dir / ".path"
    marker.write_text(str(sdf.resolve()))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    state = {"reference": str(ref.resolve()), "source_sha256": {
        str(marker.resolve()): sha(marker), str(sdf.resolve()): sha(sdf),
        str(mesh.resolve()): sha(mesh)}}

    assert _verify_bound_reference(ref, state) == ref.resolve()

    state["source_sha256"].pop(str(mesh.resolve()))
    with pytest.raises(ValueError, match="absent from its integrity manifest"):
        _verify_bound_reference(ref, state)
    state["source_sha256"][str(mesh.resolve())] = sha(mesh)

    unbound = tmp_path / "unbound.obj"
    unbound.write_text("# not referenced by the prepared SDF\\n")
    state["source_sha256"][str(unbound.resolve())] = sha(unbound)
    with pytest.raises(ValueError, match="not bound to its assets"):
        _verify_bound_reference(ref, state)

    state["source_sha256"].pop(str(unbound.resolve()))
    mesh.write_text("# changed generated mesh\\n")
    with pytest.raises(ValueError, match="not bound to its assets"):
        _verify_bound_reference(ref, state)


def test_reference_integrity_follows_only_marked_prepared_sdf_variant(tmp_path):
    ref = tmp_path / "reference"
    folder = ref / "physics_sdf" / "coaster_s1.000"
    folder.mkdir(parents=True)
    mesh = folder / "canonical.obj"
    mesh.write_text("# actually loaded\\n")
    canonical = folder / "coaster.sdf"
    canonical.write_text(
        "<sdf><model><link><visual><geometry><mesh><uri>canonical.obj</uri>"
        "</mesh></geometry></visual></link></model></sdf>")
    unused = folder / "wooden_coaster.sdf"
    unused.write_text(
        "<sdf><model><link><visual><geometry><mesh><uri>../../../missing/unused.obj</uri>"
        "</mesh></geometry></visual></link></model></sdf>")
    marker = folder / ".path"
    marker.write_text(str(canonical.resolve()))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    state = {"reference": str(ref.resolve()), "source_sha256": {
        str(marker.resolve()): sha(marker), str(canonical.resolve()): sha(canonical),
        str(mesh.resolve()): sha(mesh)}}

    # The reference adapter loads the marked canonical SDF; other cached
    # conversion variants are not part of its integrity closure.
    assert _verify_bound_reference(ref, state) == ref.resolve()

    marker.write_text(str(unused.resolve()))
    state["source_sha256"][str(marker.resolve())] = sha(marker)
    with pytest.raises(ValueError, match="bound reference asset is missing"):
        _verify_bound_reference(ref, state)


def test_reference_bound_run_bridge_exercises_native_hook_without_inference(tmp_path, monkeypatch):
    ref = tmp_path / "ref"
    ref.mkdir()
    physics = ref / "physics_sdf"
    physics.mkdir()
    for oid in ("bottom", "top"):
        folder = physics / f"{oid}_s0.050"
        folder.mkdir()
        (folder / "mesh.obj").write_text("# bridge fake mesh\n")
        sdf = folder / "asset.sdf"
        sdf.write_text(
            "<sdf version='1.7'><model name='asset'><link name='root'>"
            "<visual name='visual'><geometry><mesh><uri>mesh.obj</uri></mesh></geometry></visual>"
            "<collision name='collision'><geometry><mesh><uri>mesh.obj</uri></mesh></geometry></collision>"
            "</link></model></sdf>")
        (folder / ".path").write_text(str(sdf.resolve()))

    spec = {"schema_version": "wo_task_spec.v1", "instruction": "Stack top on bottom.",
            "objects": [{"object_id": oid, "role": "manipuland",
                         "geometry_constraints": {}} for oid in ("bottom", "top")],
            "steps": [{"goal": {"kind": "stacked", "subject": "top", "target": "bottom"}}]}
    spec_path = tmp_path / "task_spec.json"
    spec_path.write_text(json.dumps(spec))
    state = {"schema_version": "wo_reference_state.v1", "reference": str(ref.resolve()),
             "source_sha256": _manifest(ref), "initial_objects": {
                 "bottom": {"link_position": [0.2, 0.1, 0.8],
                            "link_orientation_xyzw": [0, 0, 0, 1]},
                 "top": {"link_position": [0.2, 0.1, 0.85],
                         "link_orientation_xyzw": [0, 0, 0, 1]},
             }, "measured_dynamics": {"bottom": {"root": {}}, "top": {"root": {}}}}
    state_path = tmp_path / "reference_state.json"
    state_path.write_text(json.dumps(state))

    class Pose:
        def __init__(self, p=None, q=None):
            self.p, self.q = list(p or [0, 0, 0]), list(q or [1, 0, 0, 0])

    class Handle:
        def __init__(self, name):
            self.name, self.pose = name, Pose()
        def set_pose(self, pose): self.pose = pose
        def get_pose(self): return self.pose
        def set_velocity(self, value): pass
        def set_angular_velocity(self, value): pass

    predicate_calls = []
    class Task:
        def __init__(self):
            self.scene, self.table_z_bias = object(), 0.0
            self.stock_load_calls = 0
            self.eval_success, self.take_action_cnt, self.step_lim = False, 0, 8
            self.render_freq = 0
        def load_actors(self): self.stock_load_calls += 1
        def setup_demo(self, **kwargs): self.load_actors()
        def check_success(self):
            predicate_calls.append("native")
            return True
        def set_instruction(self, value): self.instruction = value
        def get_instruction(self): return self.instruction
        def close_env(self): self.closed = True

    repo, source = tmp_path / "RoboTwin", tmp_path / "WorldOlympiad-main"
    repo.mkdir()
    source.mkdir()
    task = Task()
    def policy_runner(bound_task, client, prompt, capture_dir, action_limit):
        assert prompt == "Stack top on bottom."
        assert bound_task.block1 is bound_task.wo_handles["bottom"]
        assert bound_task.block2 is bound_task.wo_handles["top"]
        assert bound_task.block1.pose.p == [0.2, 0.1, 0.8]
        assert bound_task.block2.pose.p == [0.2, 0.1, 0.85]
        assert action_limit == 8
        for name in ("head", "left_wrist", "right_wrist"):
            (capture_dir / f"{name}.mp4").write_bytes(b"CPU fake capture")
        return {"frames": 1, "native_actions": 0}

    monkeypatch.setattr(lingbot_va_native_bridge, "_task_setup_inputs",
                        lambda *args: {"task_name": "stack_blocks_two"})
    old_cwd = Path.cwd()
    try:
        result = lingbot_va_native_bridge.run_bridge(
            robotwin_repo=repo, wov2_source=source, reference=ref,
            task_spec_path=spec_path, reference_state_path=state_path,
            out=tmp_path / "episode", port=28004, seed=5,
            client_factory=lambda: object(), task_factory=lambda: task,
            policy_runner=policy_runner,
            actor_loader=lambda scene, sdf, oid, **kwargs: Handle(oid),
            dynamics_binder=lambda *args: None,
            sapien_module=SimpleNamespace(Pose=Pose))
    finally:
        os.chdir(old_cwd)

    assert task.stock_load_calls == 0
    assert task.closed and predicate_calls == ["native"]
    assert result["task_success"] is True
    assert result["completion_evidence"]["native_predicate_evidence"] == {
        "predicate_source": "upstream RoboTwin check_success", "result": True,
        "task_class": "Task"}
    saved = json.loads((tmp_path / "episode/capture/capture_metadata.json").read_text())
    episode = json.loads((tmp_path / "episode/episode.json").read_text())
    assert saved == result and episode["native_result"] == result
    assert set(result["capture"]["videos"]) == {"head", "left_wrist", "right_wrist"}


def test_native_bridge_cli_forwards_window_to_rollout(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(lingbot_va_native_bridge, "run_bridge", lambda **kwargs: called.update(kwargs))
    monkeypatch.setattr("sys.argv", ["bridge", "--robotwin-repo", str(tmp_path),
                                 "--wov2-source", str(tmp_path), "--out", str(tmp_path),
                                 "--port", "28000", "--window-s", "5"])
    lingbot_va_native_bridge.main()
    assert called["window_s"] == 5.0


def test_native_bridge_window_caps_actions_at_15_hz():
    assert lingbot_va_native_bridge._action_limit(1000, 5) == 75
    assert lingbot_va_native_bridge._action_limit(8, 5) == 8
    assert lingbot_va_native_bridge._action_limit(1000, None) == 1000
    with pytest.raises(ValueError, match="finite positive"):
        lingbot_va_native_bridge._action_limit(1000, 0)
