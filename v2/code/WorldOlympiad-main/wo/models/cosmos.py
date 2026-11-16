"""Cosmos3-Nano-Policy-DROID native episode (Isaac/RoboLab) for a wo reference.

    python -m wo.models.cosmos --ref REF_DIR --out OUT_DIR [--gpu 0] [--port 28000]

REF_DIR is a wo.expert.run output (trajectory.json, scene_inputs.json,
certificate.json).  The adapter derives what the registered native runner needs:

  OUT/bundle/            asset copies + manifest {bundle_artifacts_sha256}
  OUT/bundle/trajectory.json   reference frames + prepared_assets
  OUT/task_graph.json    typed controller/success program from the TaskSpec
  OUT/model/runtime_identity.json, OUT/native_backend/..., OUT/native_run/...

The model is run in its own simulator; the reference only supplies the initial
scene, the task and the matched evaluation window (reference duration).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

from ..scene import ROBOT_URDF, WORKSPACE
from ..spec import TaskSpec

W = WORKSPACE
R = Path(os.environ.get("WO_REPO", W / "code/WorldOlympiad-main"))  # dev: run a worktree
PY = W / "envs/autoresearch/bin/python"
ISAAC = W / "envs/robolab-isaac51/bin/python"
TORCHRUN = W / "envs/cosmos/bin/torchrun"
BLENDER = W / "runtime/blender-4.4.3-linux-x64/blender"
ROBOT_USD = (W / "cache/official_assets/robolab/f72d39445ab699c9cfd9bf551e8d21b6075293b9"
             / "franka_robotiq_2f_85_flattened.usd")
ROBOT_USD_SHA = "f555695465687548a1bd31b5e3f30385182d476a67c17080b7820ad0ef747e41"
NATIVE_PREDICATES = {"inside": "inside", "on_region": "on_region", "hung": "hung",
                     "contact": "contact", "on": "on", "stacked": "stacked",
                     "inserted": "inserted", "covering": "covering"}
PLACE_OPERATOR = {"inside": "place", "on_region": "place", "on": "place", "stacked": "stack",
                  "inserted": "insert", "hung": "hang", "covering": "cover"}


class Unsupported(RuntimeError):
    """The registered native runner cannot represent this task (capability gap)."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sdf_scales(path: Path) -> set[str]:
    import re

    found = set(re.findall(r"<scale>([^<]*)</scale>", path.read_text()))
    return found or {"1 1 1"}


def task_graph(spec: TaskSpec) -> dict:
    controller, success = [], []
    for st in spec.steps:
        # The Cosmos WorldIR tabletop runner is deliberately strict: capabilities
        # that are not implemented by its native simulator must become a measured
        # not_applicable episode, rather than an infrastructure failure.
        # Articulate steps never reach this graph: run_episode dispatches them to
        # the joint-preserving articulated runner (cosmos_articulated.py).
        # Covering still requires a native load-bearing support measurement.
        # Insertion uses the shared depth predicate on live collision geometry.
        if st.primitive not in {"transport", "tool_contact", "push", "reorient"}:
            raise Unsupported(f"Cosmos native runner cannot represent primitive {st.primitive!r}")
        pred = ("on_region" if st.goal.kind in {"upright", "inverted"}
                else NATIVE_PREDICATES.get(st.goal.kind))
        if pred is None:
            raise Unsupported(f"Cosmos native runner has no measurement for goal {st.goal.kind!r}")
        target = st.goal.target or st.target
        if target is None and st.primitive == "reorient":
            target = spec.support.object_id
        if st.primitive != "push":
            controller.append({"operator": "pick", "subjects": [st.subject], "target": None,
                               "constraints": {}})
        op = ("push" if st.primitive == "push" else
              "place" if st.primitive == "reorient" else
              "strike" if st.primitive == "tool_contact" else PLACE_OPERATOR[st.goal.kind])
        controller.append({"operator": op, "subjects": [st.subject], "target": target,
                           "constraints": dict(st.constraints)})
        goal = {"predicate": pred, "subject": st.subject, "object": target}
        if st.goal.kind in {"upright", "inverted"} or st.constraints.get("keep_upright"):
            axis = st.goal.params.get("semantic_up_axis", [0., 0., 1.])
            # The terminal axis is expressed in the authored object frame;
            # native evaluators use the same semantic predicate metadata.
            goal.update(terminal_upright=True,
                        terminal_upright_axis_local=list(axis),
                        terminal_upright_axis_world=[0., 0., -1. if st.goal.kind == "inverted" else 1.])
        if st.primitive == "reorient":
            goal["reorientation_required"] = True
        if st.primitive == "push":
            goal["nonprehensile_required"] = True
        success.append(goal)
    return {
        "schema_version": "wo_cosmos_task_graph.v1",
        "case_id": spec.case_id,
        "instruction": spec.instruction,
        "scene_generation": {"objects": [
            {"object_id": o.object_id, "role": o.role, "category": o.category or "object",
             "visual_description": o.description} for o in spec.objects]},
        "controller_program": controller,
        "success_program": success,
        "articulation_components": [],
    }


def _archive_physics_sdf(physics: Path, output: Path, *, mesh_dir: str = 'expert_collision') -> None:
    """Archive actual SDF mesh targets with bundle-local URIs (no geometry changes)."""
    _archive_physics_xml(physics, output, mesh_dir=mesh_dir, urdf=False)


def _archive_physics_urdf(physics: Path, output: Path) -> None:
    """Archive authoritative expert URDF without changing joints/inertia/mesh scales."""
    _archive_physics_xml(physics, output, mesh_dir='expert_collision', urdf=True)


def _archive_physics_xml(physics: Path, output: Path, *, mesh_dir: str, urdf: bool) -> None:
    import xml.etree.ElementTree as ET

    tree = ET.parse(physics)
    if urdf and tree.getroot().tag != 'robot':
        raise ValueError('expert physics URDF requires a robot root')
    archived_paths = {}
    nodes = tree.findall('.//mesh') if urdf else tree.findall('.//mesh/uri')
    for index, node in enumerate(nodes):
        raw = (node.get('filename', '') if urdf else node.text or '').strip()
        if urdf and raw.startswith('package://'):
            raw = raw.removeprefix('package://')
        elif urdf and '://' in raw:
            raise ValueError('expert URDF requires a local mesh filename')
        if not raw:
            raise ValueError('expert physics SDF has an empty mesh URI')
        mesh = Path(raw)
        if not mesh.is_absolute():
            mesh = physics.parent / mesh
        if not mesh.is_file():
            raise FileNotFoundError(mesh)
        key = mesh.resolve()
        relative = archived_paths.get(key)
        if relative is None:
            relative = Path(mesh_dir) / str(index) / mesh.name
            archived = output.parent / relative
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mesh, archived)
            if mesh.suffix.lower() == '.gltf':
                # glTF is a document, not a self-contained mesh. Preserve every
                # external buffer/image by content, rebasing URIs inside this
                # immutable role-local archive (including ../ and escaped names).
                from urllib.parse import unquote, urlparse
                payload = json.loads(mesh.read_text())
                for collection in ('buffers', 'images'):
                    for resource in payload.get(collection, []):
                        uri = resource.get('uri')
                        if not uri or uri.startswith('data:'):
                            continue
                        if urlparse(uri).scheme or Path(uri).is_absolute():
                            raise ValueError('archived glTF requires local relative resources')
                        dependency = (mesh.parent / unquote(uri)).resolve()
                        name = _sha(dependency) + dependency.suffix
                        shutil.copy2(dependency, archived.parent / name)
                        resource['uri'] = name
                archived.write_text(json.dumps(payload))
            archived_paths[key] = relative
        if urdf:
            node.set('filename', relative.as_posix())
        else:
            node.text = relative.as_posix()
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding='unicode', xml_declaration=True)


def _archive_expert_physics(src: Path, phys: Path, dst_dir: Path,
                            object_id: str) -> dict[str, str]:
    """Publish exactly the expert's physics format; never substitute source physics.

    URDF mesh scales already contain the expert's cumulative object scale. Do
    not apply the source SDF scale again or flatten its movable joints.
    """
    prefix = f'assets/{object_id}'
    record = {'source_sdf': f'{prefix}/{src.name}'}
    if phys.is_file():
        if _sdf_scales(phys) != _sdf_scales(src):
            raise RuntimeError(f'{object_id}: physics SDF scale {_sdf_scales(phys)} '
                               f'differs from the asset {_sdf_scales(src)}')
        target = dst_dir / f'physics_{src.name}'
        _archive_physics_sdf(phys, target)
        record['physics_sdf'] = f'{prefix}/{target.name}'
    elif phys.with_suffix('.urdf').is_file():
        target = dst_dir / f'physics_{src.stem}.urdf'
        _archive_physics_urdf(phys.with_suffix('.urdf'), target)
        record['physics_urdf'] = f'{prefix}/{target.name}'
    else:
        raise FileNotFoundError(f'{object_id}: missing expert physics asset {phys}')
    return record


def _publish_initial_joint_measurements(traj, prepared):
    import math
    for oid, record in prepared.items():
        initial_object = traj['frames'][0].get('objects', {}).get(oid, {})
        if 'joint_positions' not in initial_object:
            continue  # Legacy references are missing measurements, not zero.
        values = initial_object['joint_positions']
        if not isinstance(values, dict) or any(
                not isinstance(name, str) or not name or
                not math.isfinite(float(value)) for name, value in values.items()):
            raise ValueError('invalid authoritative initial object joint measurements')
        record['initial_joint_positions'] = {name: float(value) for name, value in values.items()}
        record['initial_joint_state_source'] = 'trajectory.json:frames[0].objects.' + oid + '.joint_positions'


def build_bundle(ref: Path, out: Path, *, window_s: float | None = None) -> tuple[Path, Path, Path, TaskSpec]:
    cert = json.loads((ref / "certificate.json").read_text())
    scene = json.loads((ref / "scene_inputs.json").read_text())
    from ..scene import canonicalize_support_aliases
    spec = canonicalize_support_aliases(TaskSpec.load(Path(cert["task"])), cert["generation"])
    graph = task_graph(spec)
    bundle = out / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "assets").mkdir(parents=True)
    prepared = {}
    entries = [scene["support"], *scene["objects"].values()]
    for o in entries:
        src = Path(o["sdf_path"])
        dst_dir = bundle / "assets" / o["object_id"]
        shutil.copytree(src.parent, dst_dir)
        # Source collision and visual URIs may be absolute or traverse outside
        # their original directory. Bind their actual targets too; copying the
        # directory alone leaves such URIs outside the immutable bundle.
        _archive_physics_sdf(src, dst_dir / src.name, mesh_dir='source_meshes')
        # the physics SDF the expert actually simulated (visuals stripped)
        # exactly the physics SDF the expert simulated: keyed by the object's cumulative
        # scale (a glob could return another scale variant -> visual/collision mismatch)
        phys = ref / "physics_sdf" / f"{o['object_id']}_s{float(o.get('scale', 1.0)):.3f}" / src.name
        prepared[o["object_id"]] = _archive_expert_physics(
            src, phys, dst_dir, o['object_id'])
    traj = json.loads((ref / "trajectory.json").read_text())
    # New references archive measured passive object joints in every frame.
    # Preserve those measurements explicitly in the physics record. Legacy
    # references without measurements remain uninitialized (never invent zero).
    _publish_initial_joint_measurements(traj, prepared)
    # Isolated exact-input replay evidence supplements legacy missing joints;
    # never replace the canonical trajectory or its reference binding.
    supplements = sorted((ref.parent.parent / 'expert_checks' / spec.case_id).glob(
        '*/initialization.json'))
    if supplements and any('initial_joint_positions' not in r for r in prepared.values()):
        from wo.expert.initialization_check import load_bound_supplement
        if len(supplements) != 1:
            raise ValueError('ambiguous initialization supplements')
        evidence = supplements[0]
        measured = load_bound_supplement(ref, evidence)
        archived = bundle / 'initialization_supplement.json'
        archived.write_bytes(evidence.read_bytes())
        supplemental_records = {oid: {} for oid in prepared}
        _publish_initial_joint_measurements({'frames': [measured['measured_frame0']]}, supplemental_records)
        for oid, row in supplemental_records.items():
            if 'initial_joint_positions' not in prepared[oid]:
                prepared[oid].update(row)
                prepared[oid]['initial_joint_state_source'] = (
                    'initialization_supplement.json:measured_frame0.objects.' + oid + '.joint_positions')
                prepared[oid]['initial_joint_state_evidence_sha256'] = _sha(archived)
    traj["prepared_assets"] = prepared
    (bundle / "trajectory.json").write_text(json.dumps(traj))
    digests = {p.relative_to(bundle).as_posix(): _sha(p)
               for p in sorted(bundle.rglob("*")) if p.is_file()}
    manifest = bundle / "reference_manifest.json"
    manifest.write_text(json.dumps({"schema_version": "wo_cosmos_stub_bundle.v1",
                                    "case_id": spec.case_id,
                                    "bundle_artifacts_sha256": digests}, indent=1))
    graph_path = out / "task_graph.json"
    graph_path.write_text(json.dumps(graph, indent=1))
    frames = len(traj["frames"]) if window_s is None else int(round(window_s * 15)) + 1
    window = {"policy": "fixed_common_physical_window.v1", "control_hz": 15.0,
              "max_steps": frames - 1, "frame_count": frames,
              "physical_duration_s": (frames - 1) / 15.0,
              "reference_manifest_sha256": _sha(manifest), "reference_view_id": "exterior_left",
              "reference_video_sha256": None, "success_ends_episode": False,
              "model_time_rescaled": False, "reference_time_rescaled": False}
    eval_manifest = out / "evaluation_input.json"
    eval_manifest.write_text(json.dumps({"simulator_state": {"evaluation_window": window}}, indent=1))
    return manifest, graph_path, eval_manifest, spec


def _run(args: list[str], log: Path, env: dict | None = None, cwd: Path = R) -> None:
    with log.open("w") as fh:
        rc = subprocess.run([str(a) for a in args], cwd=cwd, env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    if rc != 0:
        tail = log.read_text(errors="replace").splitlines()[-15:]
        raise RuntimeError(f"{args[1] if len(args) > 1 else args[0]} failed (rc={rc}):\n" + "\n".join(tail))


def _wait_port(port: int, timeout: float, proc: subprocess.Popen) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("Cosmos policy server exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(3)
    raise RuntimeError("Cosmos policy server did not open its port")


def _runtime_identity(out: Path, logs: Path, env_r: dict) -> Path:
    (out / "model").mkdir(exist_ok=True)
    _run([PY, R / "scripts/verify_wam_runtime_identity.py", "--profile", "cosmos3_nano_droid",
          "--lock", R / "deployment/upstream_lock.json",
          "--checkpoint-root", W / "models/Cosmos3-Nano-Policy-DROID",
          "--source-root", W / "code/cosmos-framework",
          "--runtime-tree-root", W / "code/RoboLab-isaac51",
          "--external", f"Wan-AI/Wan2.2-TI2V-5B#Wan2.2_VAE.pth={W}/models/Wan2.2-TI2V-5B-vae",
          "--cache-root", W / "outputs/.runtime_identity_cache",
          "--output", out / "model/runtime_identity.json"], logs / "identity.log", env_r)
    return out / "model/runtime_identity.json"


def _client_pkgs(out: Path) -> Path:
    """openpi policy client package for the Isaac interpreter."""
    client = out / "client_pkgs"
    if not client.exists():
        import glob

        src = glob.glob(str(W / "envs/cosmos/lib/python3*/site-packages/openpi_client"))
        if not src:
            raise RuntimeError("openpi_client package not found in envs/cosmos")
        shutil.copytree(src[0], client / "openpi_client")
    return client


def _with_policy_server(out: Path, logs: Path, env: dict, port: int, seed: int, client_run) -> None:
    server_log = (logs / "policy_server.log").open("w")
    server = subprocess.Popen(
        [str(TORCHRUN), "--standalone", "--nproc_per_node=1",
         str(R / "scripts/launch_cosmos3_robolab_server.py"),
         "--cosmos-repo", str(W / "code/cosmos-framework"),
         "--checkpoint-path", str(W / "models/Cosmos3-Nano-Policy-DROID"),
         "--wan-vae-path", str(W / "models/Wan2.2-TI2V-5B-vae/Wan2.2_VAE.pth"),
         "--output-dir", str(out / "policy_server"), "--port", str(port),
         "--seed", str(seed), "--request-seed-offset", "0", "--decode-video"],
        cwd=R, env=dict(env, PYTHONPATH=str(W / "code/cosmos-framework")),
        stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        _wait_port(port, 900, server)
        client_run()
    finally:
        os.killpg(server.pid, 15)
        server_log.close()


def _episode_result(out: Path, ref: Path, spec: TaskSpec, prompt: str, frames: int, seed: int,
                    **extra) -> dict:
    meta = json.loads((out / "native_run/capture/capture_metadata.json").read_text())
    result = {"schema_version": "wo_model_episode.v1", "model": "cosmos3_nano_droid",
              "case_id": spec.case_id, "reference": str(ref), "prompt": prompt,
              "window_s": (frames - 1) / 15.0, "seed": seed,
              "task_success": meta.get("task_success"),
              "termination_reason": meta.get("termination_reason"),
              "videos": {v: str(p) for v in ("exterior_left", "exterior_right", "wrist")
                         for p in [next((out / "native_run/export" / v).glob("imagination_*.mp4"), None)]
                         if p is not None},
              "simulated_trajectory": str(out / "native_run/simulation/trajectory.json"), **extra}
    (out / "episode.json").write_text(json.dumps(result, indent=1))
    return result


def _export(out: Path, frames: int, logs: Path, env_r: dict) -> None:
    _run([PY, R / "scripts/export_cosmos_worldir_isaac_imagination.py",
          "--capture-root", out / "native_run/capture", "--output-dir", out / "native_run/export",
          "--frame-count", str(frames), "--fps", "15"], logs / "export.log", env_r)


def _envs(gpu: int) -> tuple[dict, dict, dict]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    env_r = dict(env, PYTHONPATH=str(R), BLENDER_BIN=str(BLENDER),
                 __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    env_i = dict(env, PYTHONPATH=f"{R}:{W / 'code/RoboLab-isaac51'}")
    return env, env_r, env_i


def is_articulated(spec: TaskSpec) -> bool:
    return any(st.primitive == "articulate" for st in spec.steps)


def run_articulated_episode(ref: Path, out: Path, *, gpu: int = 0, port: int = 28000,
                            prompt: str | None = None, window_s: float | None = None,
                            seed: int = 0, policy_model: str = "cosmos3") -> dict:
    """Joint-preserving native runner with an explicitly selected DROID policy."""
    if policy_model not in ("cosmos3", "gr00t"):
        raise ValueError("unsupported articulated policy")
    from . import cosmos_articulated as art

    cert = json.loads((ref / "certificate.json").read_text())
    spec = TaskSpec.load(Path(cert["task"]))
    try:
        b = art.binding(spec, cert)
    except art.ArticulationUnsupported as exc:
        raise Unsupported(str(exc)) from exc
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    for derived in (out / "articulated_task", out / "native_run", out / "policy_server"):
        shutil.rmtree(derived, ignore_errors=True)  # derived per run; never mix runs
    graph_path = out / "task_graph.json"
    graph_path.write_text(json.dumps(art.task_graph(spec, b), indent=1))
    n_ref = len(json.loads((ref / "trajectory.json").read_text())["frames"])
    frames = n_ref if window_s is None else int(round(window_s * 15)) + 1
    window = {"policy": "fixed_common_physical_window.v1", "control_hz": 15.0,
              "max_steps": frames - 1, "frame_count": frames,
              "physical_duration_s": (frames - 1) / 15.0, "reference_view_id": "exterior_left",
              "reference_video_sha256": None, "success_ends_episode": False,
              "model_time_rescaled": False, "reference_time_rescaled": False}
    prompt = prompt or spec.instruction
    env, env_r, env_i = _envs(gpu)
    if policy_model == "gr00t":
        from . import gr00t
        identity = gr00t.runtime_identity(out, dependencies=gr00t.runtime_dependencies())
    else:
        identity = _runtime_identity(out, logs, env_r)
    os.environ.setdefault("BLENDER_BIN", str(BLENDER))  # in-process material-preserving export
    try:
        task = art.prepare(ref, out, b=b, prompt=prompt, seed=seed, window=window,
                           robot_usd=ROBOT_USD, runtime_identity=identity)
    except art.ArticulationUnsupported as exc:
        raise Unsupported(str(exc)) from exc
    usd = task.parent / "scene/world.usd"
    _run([ISAAC, R / "scripts/materialize_worldir_isaac_usd.py", "--plan",
          task.parent / "isaac/isaac_compile_plan.json", "--output", usd,
          "--fixed-articulation-body", b["host"]], logs / "usd.log", env_i)
    client = _client_pkgs(out) if policy_model == "cosmos3" else ""
    (out / "native_run").mkdir()
    launch = (lambda callback: gr00t.with_policy_server(logs, env, port, callback)) if policy_model == "gr00t" else (
        lambda callback: _with_policy_server(out, logs, env, port, seed, callback))
    launch(lambda: _run(
        [ISAAC, R / "scripts/run_cosmos3_articulated_isaac.py", "--task", task, "--world-usd", usd,
         "--policy-model", policy_model, "--execute-actions-per-window", "8" if policy_model == "gr00t" else "32",
         "--output", out / "native_run/capture", "--policy-server-host", "127.0.0.1",
         "--policy-server-port", str(port)],
        logs / "episode.log", dict(env_i, PYTHONPATH=f"{R}:{W / 'code/RoboLab-isaac51'}:{client}")))
    meta = art.convert(ref, out, b, policy_model=policy_model)
    if policy_model == "gr00t":
        _run([PY, '-m', 'wo.models.native_isaac_export',
              '--capture', out / 'native_run/capture', '--output', out / 'native_run/export',
              '--frame-count', str(meta['frame_count'])], logs / 'export.log', env_r)
    else:
        _export(out, meta["frame_count"], logs, env_r)
    return _episode_result(out, ref, spec, prompt, frames, seed, runner="articulated",
                           model="gr00t_n1.6_droid" if policy_model == "gr00t" else "cosmos3_nano_droid",
                           completion_frame=meta["completion_frame"],
                           final_joint_fraction=meta["final_joint_fraction"],
                           max_joint_fraction=meta["max_joint_fraction"])


def run_episode(ref: Path, out: Path, *, gpu: int = 0, port: int = 28000,
                prompt: str | None = None, window_s: float | None = None, seed: int = 0) -> dict:
    cert = json.loads((ref / "certificate.json").read_text())
    if is_articulated(TaskSpec.load(Path(cert["task"]))):
        return run_articulated_episode(ref, out, gpu=gpu, port=port, prompt=prompt,
                                       window_s=window_s, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    manifest, graph_path, eval_manifest, spec = build_bundle(ref, out, window_s=window_s)
    prompt = prompt or spec.instruction
    env, env_r, env_i = _envs(gpu)
    _runtime_identity(out, logs, env_r)
    backend = out / "native_backend"
    for derived in (backend, out / "native_run", out / "policy_server"):
        shutil.rmtree(derived, ignore_errors=True)  # derived per run; never mix runs
    _run([PY, R / "scripts/build_task_graph_worldir.py", "--reference-manifest", manifest,
          "--task-graph", graph_path, "--trajectory", manifest.parent / "trajectory.json",
          "--robot-urdf", ROBOT_URDF, "--prompt", prompt, "--scene-id", spec.case_id,
          "--output-dir", backend], logs / "worldir.log", env_r)
    usd = backend / "scene/task_graph_worldir.usd"
    _run([ISAAC, R / "scripts/materialize_worldir_isaac_usd.py", "--plan",
          backend / "isaac/isaac_compile_plan.json", "--output", usd], logs / "usd.log", env_i)
    client = _client_pkgs(out)
    _with_policy_server(out, logs, env, port, seed, lambda: _run(
        [ISAAC, R / "scripts/run_cosmos3_worldir_isaac_closed_loop.py",
         "--world-repo", R, "--world-usd", usd,
         "--worldir-plan", backend / "isaac/isaac_compile_plan.json",
         "--reference-trajectory", manifest.parent / "trajectory.json",
         "--robot-usd", ROBOT_USD, "--robot-usd-sha256", ROBOT_USD_SHA,
         "--runtime-identity", out / "model/runtime_identity.json",
         "--output-dir", out / "native_run", "--prompt", prompt,
         "--seed-schedule-seed", str(seed), "--policy-server-host", "127.0.0.1",
         "--policy-server-port", str(port),
         "--official-robolab-task", "worldolympiad/task_graph_native_episode",
         "--task-graph", graph_path, "--official-robolab-task-sha256", _sha(graph_path),
         "--evaluation-input-manifest", eval_manifest],
        logs / "episode.log", dict(env, PYTHONPATH=f"{R}:{W / 'code/RoboLab-isaac51'}:{client}")))
    frames = json.loads(eval_manifest.read_text())["simulator_state"]["evaluation_window"]["frame_count"]
    _export(out, frames, logs, env_r)
    return _episode_result(out, ref, spec, prompt, frames, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=28000)
    ap.add_argument("--prompt", help="override the task instruction (prompt ablations)")
    ap.add_argument("--window-s", type=float, help="episode length; default = reference duration")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    try:
        result = run_episode(a.ref, a.out, gpu=a.gpu, port=a.port, prompt=a.prompt,
                             window_s=a.window_s, seed=a.seed)
    except Unsupported as exc:
        cert = json.loads((a.ref / 'certificate.json').read_text())
        spec = TaskSpec.load(Path(cert['task']))
        result = {'schema_version': 'wo_model_episode.v1', 'model': 'cosmos3_nano_droid',
                  'case_id': spec.case_id, 'task_family': spec.family, 'reference': str(a.ref),
                  'seed': a.seed, 'window_s': a.window_s, 'task_success': None,
                  'status': 'not_applicable', 'reason': str(exc),
                  'termination_reason': 'not_applicable', 'videos': {}}
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / 'episode.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
