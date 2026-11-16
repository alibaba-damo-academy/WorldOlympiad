"""Cosmos3-Nano-Policy-DROID native episode for articulate tasks (door/drawer/lid).

The rigid WorldIR tabletop runner cannot preserve joints, so articulate steps are
dispatched to the registered articulated runner
(scripts/run_cosmos3_articulated_isaac.py) with a task contract derived from the
wo reference:

  OUT/articulated_task/          WorldIR + Isaac plan + task.json (runner input)
  OUT/articulated_task/scene/world.usd
  OUT/native_run/capture/        runner output (windows/, progress.json, trajectory.json)
  OUT/native_run/simulation/trajectory.json   wo-format frames (robot, objects,
                                               articulation_positions)
  OUT/native_run/capture/capture_metadata.json  wo measurement of the goal

Success and completion come from the native JointTermination monitor, including
its release and stability checks. Joint-band progress remains a separate
diagnostic and cannot substitute for a missing native measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..predicates import THRESHOLDS
from ..spec import TaskSpec

INPUT_CONTRACT = "wo_articulated_inputs.v1"
GOALS = {"joint_open", "joint_closed"}


class ArticulationUnsupported(RuntimeError):
    pass


class ArticulationMeasurementError(ValueError):
    pass


def binding(spec: TaskSpec, cert: dict) -> dict:
    """The single joint goal of an articulate-only task (pure; no simulator)."""
    steps = list(spec.steps)
    if len(steps) != 1 or steps[0].primitive != "articulate":
        raise ArticulationUnsupported(
            "Cosmos articulated runner represents exactly one articulate step; "
            f"got {[s.primitive for s in steps]}")
    st = steps[0]
    if st.goal.kind not in GOALS:
        raise ArticulationUnsupported(f"articulated runner has no measurement for {st.goal.kind!r}")
    comp = st.goal.subject or st.subject
    a = (cert.get("articulations") or {}).get(comp)
    if a is None:
        raise ArticulationUnsupported(f"reference certificate lacks articulation {comp!r}")
    if a["joint_type"] not in {"revolute", "prismatic"}:
        raise ArticulationUnsupported(f"joint type {a['joint_type']!r}")
    default = THRESHOLDS["joint_open_fraction" if st.goal.kind == "joint_open" else "joint_closed_fraction"]
    return {"component": comp, "host": a["host"], "joint_name": a["joint_name"],
            "child_link": a["child_link"], "joint_type": a["joint_type"],
            "lower": float(a["lower"]), "upper": float(a["upper"]),
            "q_closed": float(a["q_closed"]), "q_open": float(a["q_open"]),
            "goal": st.goal.kind, "threshold": float(st.goal.params.get("threshold", default))}


def fraction(q: float, b: dict) -> float:
    span = b["q_open"] - b["q_closed"]
    return float((q - b["q_closed"]) / span) if abs(span) > 1e-9 else 0.0


def satisfied(frac: float, b: dict) -> bool:
    return frac >= b["threshold"] if b["goal"] == "joint_open" else frac <= b["threshold"]


def goal_band(b: dict) -> tuple[float, float]:
    """JointTask target/tolerance whose band equals the wo predicate region.

    JointTask accepts |q - target| <= tolerance; the wo predicate is one-sided
    (fraction >= threshold for open, <= threshold for closed) within the joint
    range, so the band is centred on that region of [0, 1].
    """
    lo, hi = (b["threshold"], 1.0) if b["goal"] == "joint_open" else (0.0, b["threshold"])
    span = b["q_open"] - b["q_closed"]
    return b["q_closed"] + 0.5 * (lo + hi) * span, 0.5 * (hi - lo) * abs(span) + 1e-4


def task_graph(spec: TaskSpec, b: dict) -> dict:
    return {
        "schema_version": "wo_cosmos_task_graph.v1",
        "case_id": spec.case_id,
        "instruction": spec.instruction,
        "scene_generation": {"objects": [
            {"object_id": o.object_id, "role": o.role, "category": o.category or "object",
             "visual_description": o.description} for o in spec.objects]},
        "controller_program": [{"operator": "articulate", "subjects": [b["component"]],
                                "target": None, "constraints": {}}],
        "success_program": [{"predicate": b["goal"], "subject": b["component"],
                             "object": None, "threshold": b["threshold"]}],
        "articulation_components": [b],
    }


def _host_entry(scene: dict, host: str) -> dict:
    if scene["support"]["object_id"] == host:
        return scene["support"]
    return scene["objects"][host]


def native_initial_position(reference_position: float) -> float:
    """The imported SDF spawns at zero, not Bullet's settled limit residual.

    Keep the existing reference-equivalence guard; never widen joint limits or
    transplant solver residuals into the native forward-kinematics contract.
    """
    import math
    q = float(reference_position)
    if not math.isfinite(q) or abs(q) > 1e-3:
        raise ArticulationUnsupported("articulated runner spawns the joint at its model "
                                      f"configuration (q=0); reference starts at q={q:.4f}")
    return 0.0


def fixed_background_bodies(ref: Path, out: Path, scene: dict, traj: dict, host: str):
    import shutil
    from ..scene import canonicalize_support_aliases, fixed_background_ids
    from .cosmos import (_archive_physics_sdf, _archive_expert_physics,
                         _publish_initial_joint_measurements, _sha)
    from scripts.build_task_graph_worldir import bind_task_physics, import_task_urdf, initial_joint_metadata
    from autoresearch.models import Pose
    from autoresearch.scenecode_importer import import_sdf_body
    cert = json.loads((ref / 'certificate.json').read_text())
    spec = TaskSpec.load(Path(cert['task']))
    if cert.get('generation'):
        spec = canonicalize_support_aliases(spec, cert['generation'])
    ids = fixed_background_ids(spec, scene['objects'], hosts=[host])
    unsupported = {o.object_id for o in spec.objects
                   if o.geometry_constraints.get('fixed') is True and o.object_id != host} - ids
    if unsupported:
        raise ValueError(f'articulated input cannot omit task-related fixed objects: {sorted(unsupported)}')
    if not ids:
        return [], None
    bundle = out / 'fixed_background_bundle'
    prepared, sources = {}, {}
    for oid in sorted(ids):
        row = scene['objects'][oid];src = Path(row['sdf_path'])
        if float(row.get('scale', 1.0)) != 1.0:
            raise ValueError('fixed background requires its authoritative unit-scale source')
        dst = bundle / 'assets' / oid
        shutil.copytree(src.parent, dst)
        _archive_physics_sdf(src, dst / src.name, mesh_dir='source_meshes')
        physics = ref / 'physics_sdf' / f'{oid}_s1.000' / src.name
        prepared[oid] = _archive_expert_physics(src, physics, dst, oid)
        sources[oid] = _sha(src)
    _publish_initial_joint_measurements(traj, prepared)
    digests = {p.relative_to(bundle).as_posix():_sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    manifest = bundle / 'reference_manifest.json'
    manifest.write_text(json.dumps({'bundle_artifacts_sha256':digests}, indent=2))
    runtime = bind_task_physics(manifest, prepared, out / 'fixed_background_runtime', digests,
                               allow_subject_joint_lock=False)
    bodies, poses = [], {}
    for oid in sorted(ids):
        row = scene['objects'][oid]
        state = traj['frames'][0]['objects'].get(oid)
        source = 'reference trajectory frame0 link pose'
        if state is None:
            state = {'link_position':row['position'], 'link_orientation_xyzw':row['orientation_xyzw']}
            source = 'fixed scene_inputs pose; stationary non-articulated root omitted by reference'
        importer = import_task_urdf if prepared[oid].get('physics_urdf') else import_sdf_body
        bodies.append(importer(runtime[oid], body_name=oid, semantic_category=spec.object(oid).category,
            pose=Pose(xyz=tuple(state['link_position']), quat_xyzw=tuple(state['link_orientation_xyzw'])),
            static=True, target=False, support=None,
            metadata={'task_object_id':oid, 'task_role':'distractor', 'source_sdf_sha256':sources[oid],
                      'pose_source':source, **initial_joint_metadata(prepared[oid])}))
        poses[oid] = source
    return bodies, {'manifest':str(manifest), 'sha256':_sha(manifest),
                    'source_sdf_sha256':sources, 'pose_sources':poses}


def prepare(ref: Path, out: Path, *, b: dict, prompt: str, seed: int, window: dict,
            robot_usd: Path, runtime_identity: Path) -> Path:
    """Write the runner's task contract + Isaac compile plan (CPU only)."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    from autoresearch.articulated_native_task import JointTask
    from autoresearch.io import sha256_file, write_json
    from autoresearch.isaac_compiler import prepare_isaac_bundle
    from autoresearch.math3d import compose
    from autoresearch.models import BodyIR, ContactMaterial, Geometry, LinkIR, Pose, WorldIR
    from autoresearch.scenecode_importer import import_sdf_body
    from autoresearch.worldir_fixed_joint_collapse import collapse_fixed_joint_tree
    from dataclasses import asdict

    scene = json.loads((ref / "scene_inputs.json").read_text())
    traj = json.loads((ref / "trajectory.json").read_text())
    frame0 = traj["frames"][0]
    host = _host_entry(scene, b["host"])
    if abs(float(host.get("scale", 1.0)) - 1.0) > 1e-9:
        raise ArticulationUnsupported("articulated host with non-unit scale")
    q0 = native_initial_position(
        (frame0.get("articulation_positions") or {}).get(b["component"], b["q_closed"]))
    src = Path(host["sdf_path"])
    body = import_sdf_body(src, body_name=b["host"], semantic_category="articulated_host",
                           static=True, target=True,
                           material=ContactMaterial(lateral_friction=.9, restitution=0))
    body.pose = Pose(xyz=tuple(map(float, host["position"])),
                     quat_xyzw=tuple(map(float, host["orientation_xyzw"])))
    collapse = collapse_fixed_joint_tree(body)
    links = {link.name: link for link in body.links}
    joint = next((j for j in body.joints if j.name == b["joint_name"]), None)
    if joint is None or joint.type != b["joint_type"]:
        raise ArticulationUnsupported(f"host SDF lacks {b['joint_type']} joint {b['joint_name']!r}")
    child = joint.child
    child_world = compose(body.pose, links[child].pose)
    body_r = Rotation.from_quat(body.pose.quat_xyzw)
    axis = body_r.apply(Rotation.from_quat(joint.pose.quat_xyzw).apply(joint.axis))
    axis = axis / np.linalg.norm(axis)
    pivot = compose(body.pose, joint.pose).xyz
    target, tol = goal_band(b)
    lower, upper = float(joint.lower), float(joint.upper)
    task = JointTask(joint.type, tuple(float(v) for v in axis), tuple(child_world.xyz),
                     tuple(child_world.quat_xyzw), tuple(pivot), q0, target, lower, upper,
                     tol, .02, 5, window["max_steps"])

    def box(name, size, xyz, color):
        g = Geometry(name=name, kind="box", size=tuple(size), rgba=tuple(color))
        return BodyIR(name=name, semantic_category="support", static=True, pose=Pose(xyz=tuple(xyz)),
                      links=[LinkIR(name="base", visuals=[g], collisions=[g])])

    rb = frame0["robot"]
    robot = BodyIR(name="robot", semantic_category="robot", static=True,
                   pose=Pose(xyz=tuple(rb["base_position"]), quat_xyzw=tuple(rb["base_orientation_xyzw"])),
                   links=[LinkIR(name="native")], metadata={"usd_sha256": sha256_file(robot_usd)})
    bodies = [body, robot, box("ground", (8, 8, .05), (0, 0, -.025), (.65, .65, .65, 1))]
    mount = traj.get("robot_mount")
    if mount and mount.get("height_m", 0) > 0:
        h = mount["height_m"]
        bodies.append(box("robot_mount", [*mount["size_xy_m"], h], [*rb["base_position"][:2], h / 2],
                          (.28, .3, .32, 1)))
    backgrounds, background_binding = fixed_background_bodies(ref, out, scene, traj, b['host'])
    bodies.extend(backgrounds)
    world = WorldIR(scene_id=f"{b['host']}_articulated", prompt=prompt, bodies=bodies,
                    provenance={"asset_origin": "wo_reference", "reference": str(ref),
                                "source_sdf_sha256": sha256_file(src)})
    if background_binding:
        world.provenance['fixed_background_binding'] = background_binding
    root = out / "articulated_task"
    root.mkdir(parents=True)
    write_json(root / "fixed_joint_collapse.json", collapse)
    world.write_json(root / "worldir.json")
    prepare_isaac_bundle(world, root / "isaac", native_body_names=["robot"], preserve_visual_materials=True)
    plan = json.loads((root / "isaac/isaac_compile_plan.json").read_text())
    host_plan = next(x for x in plan["bodies"] if x["name"] == body.name)
    child_plan = next(x for x in host_plan["links"] if x["name"] == child)
    write_json(root / "initial_state.json", frame0)
    cert_sha = sha256_file(ref / "certificate.json")
    contract = {
        "schema_version": "cosmos_articulated_project_task.v1",
        "contract_owner": "wo.models.cosmos_articulated",
        "input_contract_version": INPUT_CONTRACT,
        "instruction": prompt, "worldir_hash": world.content_hash(),
        "worldir_plan": "isaac/isaac_compile_plan.json",
        "initial_state": "initial_state.json",
        "initial_state_sha256": sha256_file(root / "initial_state.json"),
        "source_sdf": str(src), "source_sdf_sha256": sha256_file(src),
        "visual_sdf": str(src), "visual_sdf_sha256": sha256_file(src),
        "host_object": body.name, "fixed_articulation_body": body.name,
        "joint_name": joint.name, "child_link": child,
        "child_prim": f"/World/WorldIR/Scene/{host_plan['prim_name']}/{child_plan['prim_name']}",
        "robot_usd": str(robot_usd), "robot_usd_sha256": sha256_file(robot_usd),
        "runtime_identity": str(runtime_identity),
        "runtime_identity_sha256": sha256_file(runtime_identity),
        "reference_manifest_sha256": cert_sha,
        "evaluation_window": dict(window, reference_manifest_sha256=cert_sha),
        "episode_seconds": window["physical_duration_s"], "termination": asdict(task),
        "wo_goal": b, "seed_schedule_seed": int(seed), "formal_cae_certified": False,
    }
    write_json(root / "task.json", contract)
    return root / "task.json"


def _native_measurement(capture: Path, raw: list, progress: dict) -> dict:
    """Require the completed runner result to agree with its captured monitor."""
    try:
        result = json.loads((capture / "result.json").read_text())
        terminal = result["terminal"]["terminal"]
        valid = (
            bool(raw) and result["schema_version"] == "cosmos_articulated_policy_trial.v1"
            and result["status"] == "complete"
            and type(result["task_success"]) is bool
            and type(terminal["released"]) is bool
            and type(terminal["stable_steps"]) is int and terminal["stable_steps"] >= 0
            and terminal["done"] is True
            and result["frame_count"] == len(raw)
            and all(f["frame"] == i for i, f in enumerate(raw))
            and result["action_count"] == len(raw) - 1
            and result["window_count"] == len(progress["windows"])
            and progress["step"] == len(raw) - 1
            and result["terminal"] == raw[-1]
            and terminal == progress["terminal"]
            and result["task_success"] == terminal["task_success"]
            and result["completion_frame"] == terminal["completion_frame"]
            and result["termination_reason"] == terminal["reason"]
        )
        completion = result["completion_frame"]
        if result["task_success"]:
            valid = valid and terminal["released"] and terminal["stable_steps"] > 0 and completion is not None
        if completion is not None:
            valid = valid and type(completion) is int and 0 < completion < len(raw)
            if valid:
                first = raw[completion]["terminal"]
                valid = (first["task_success"] is True and first["released"] is True
                         and first["stable_steps"] > 0 and first["completion_frame"] == completion)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise ArticulationMeasurementError("missing or malformed native articulation measurement") from exc
    if not valid:
        raise ArticulationMeasurementError("native articulation result disagrees with captured monitor")
    return result


def convert(ref: Path, out: Path, b: dict, *, policy_model: str = "cosmos3") -> dict:
    """Runner output -> wo simulation trajectory + capture metadata (no simulator)."""
    capture = out / "native_run/capture"
    raw = json.loads((capture / "trajectory.json").read_text())["frames"]
    progress = json.loads((capture / "progress.json").read_text())
    result = _native_measurement(capture, raw, progress)
    ref_frame0 = json.loads((ref / "trajectory.json").read_text())["frames"][0]
    host_state = ref_frame0["objects"][b["host"]]
    comp = b["component"]
    frames, fracs = [], []
    for i, f in enumerate(raw):
        r = f["robot"]
        q = float(f["articulated"]["position"])
        fracs.append(fraction(q, b))
        frames.append({
            "frame": i, "time_s": i / 15.0,
            "robot": {"base_position": r["base_position"], "base_orientation_xyzw": r["base_xyzw"],
                      "joint_names": r["joint_names"], "joint_positions": r["joint_positions"],
                      "end_effector_position": r["ee_position"],
                      "droid_gripper_position": r["gripper"]},
            "objects": {b["host"]: host_state},
            "articulated": {"component_id": comp, "joint_name": b["joint_name"], "position": q},
            "articulation_positions": {comp: q},
            "articulated_links": {comp: {"link": b["child_link"], "position": f["child_link_position"],
                                         "orientation_xyzw": f["child_link_xyzw"]}},
            "finger_contacts": f.get("finger_contacts"),
        })
    sim = out / "native_run/simulation"
    sim.mkdir(parents=True, exist_ok=True)
    (sim / "trajectory.json").write_text(json.dumps(
        {"schema_version": "wo_cosmos_articulated_trajectory.v1", "fps": 15, "frames": frames}))
    band_success = satisfied(fracs[-1], b)
    band_completion = None
    if band_success:
        band_completion = len(fracs) - 1
        while band_completion > 0 and satisfied(fracs[band_completion - 1], b):
            band_completion -= 1
    completion = result["completion_frame"]
    meta = {
        "schema_version": "wo_cosmos_articulated_capture.v1", "status": "complete",
        "model": "gr00t_n1.6_droid" if policy_model == "gr00t" else "cosmos3_nano_droid",
        "action_source": "gr00t_n1.6_droid" if policy_model == "gr00t" else "cosmos3_nano_policy_droid",
        "rollout_controller": "articulated_native_episode",
        "frame_count": len(frames), "action_count": len(frames) - 1,
        "window_count": len(progress["windows"]), "windows": progress["windows"],
        "task_success": result["task_success"], "termination_reason": result["termination_reason"],
        "completion_frame": completion,
        "completion_evidence": None if completion is None else {"native_predicate_evidence": {
            "predicate_source": "autoresearch.articulated_native_task.JointTermination",
            "predicate": b["goal"], "subject": comp, "threshold": b["threshold"],
            "joint_fraction": fracs[completion], "measured_at_s": completion / 15.0,
            "native_monitor": raw[completion]["terminal"],
            "joint_state": raw[completion]["articulated"]}},
        "terminal_native_monitor": result["terminal"]["terminal"],
        "joint_band_diagnostic": {"final_satisfied": band_success,
                                  "continuous_to_end_from_frame": band_completion,
                                  "includes_release_and_stability": False},
        "goal": b, "initial_joint_fraction": fracs[0], "final_joint_fraction": fracs[-1],
        "max_joint_fraction": max(fracs) if b["goal"] == "joint_open" else None,
        "min_joint_fraction": min(fracs) if b["goal"] == "joint_closed" else None,
        "runner_result": {k: result.get(k) for k in ("task_success", "termination_reason",
                                                     "completion_frame", "maximum_progress",
                                                     "contact_frames", "frame_count")},
        "evaluation_window": result.get("evaluation_window"),
        "trajectory": "../simulation/trajectory.json",
    }
    (capture / "capture_metadata.json").write_text(json.dumps(meta, indent=1))
    return meta
