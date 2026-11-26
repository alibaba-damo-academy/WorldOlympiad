"""Reference frame-zero and effective scene physics, without rewriting a reference."""
import json
from pathlib import Path
import numpy as np
from wo.expert.initialization_check import asset_closure, load_scene, measured_dynamics, sha
from wo.sim.world import PreparedBody, World


def collect(ref):
    ref = Path(ref).resolve()
    names = ("certificate.json", "scene_inputs.json", "trajectory.json")
    manifest = {str(ref / name): sha(ref / name) for name in names}
    scene = load_scene(json.loads((ref / "scene_inputs.json").read_text()))
    trajectory = json.loads((ref / "trajectory.json").read_text())
    initial = trajectory["frames"][0]
    prepared = {}
    initial_objects = dict(initial["objects"])
    pose_sources = {}
    for obj in [scene.support, *scene.objects.values()]:
        marker = ref / "physics_sdf" / f"{obj.object_id}_s{obj.scale:.3f}" / ".path"
        artifact = Path(marker.read_text().strip()).resolve()
        if ref not in artifact.parents:
            raise ValueError("reference physics must be the canonical prepared artifact")
        manifest[str(marker)] = sha(marker)
        asset_closure(artifact, manifest)
        prepared[obj.object_id] = PreparedBody(obj.object_id, artifact, {})
        state = initial_objects.get(obj.object_id)
        if not state and obj.fixed and not any(a.get("host") == obj.object_id for a in scene.articulations.values()):
            state = {"link_position": obj.position, "link_orientation_xyzw": obj.orientation_xyzw,
                     "joint_positions": {}}
            initial_objects[obj.object_id] = state
            pose_sources[obj.object_id] = "fixed scene_inputs pose; reference trajectory omits stationary support"
        else:
            pose_sources[obj.object_id] = "reference trajectory frame0 link pose"
        if not state or "link_position" not in state or "link_orientation_xyzw" not in state:
            raise ValueError("missing canonical reference initial link pose: " + obj.object_id)
    # World performs exactly the expert's mass/material overrides. Physics
    # parameters do not depend on a new rollout or its task outcome.
    world = World(scene, prepared, mode="objects")
    try:
        dynamics = measured_dynamics(world)
    finally:
        world.close()
    if any(sha(p) != h for p, h in manifest.items()):
        raise ValueError("reference changed while measuring effective scene physics")
    return {"schema_version": "wo_reference_state.v1", "reference": str(ref),
            "source_sha256": manifest, "initial_objects": initial_objects, "initial_pose_sources": pose_sources,
            "initial_articulations": initial.get("articulation_positions", {}),
            "measured_dynamics": dynamics,
            "physics_source": "World construction with exact archived prepared physics; getDynamicsInfo after expert overrides",
            "world_code_sha256": sha(Path(__file__).parents[1] / "sim/world.py"),
            "task_success_claim": False}


def bind(ref, out, objects):
    data = collect(ref)
    path = Path(out) / "lingbot/reference_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != data:
        raise ValueError("reference-state archive changed; use a fresh episode")
    path.write_text(json.dumps(data, indent=1))
    for obj in objects:
        state = data["initial_objects"][obj["object_id"]]
        obj.update(position=state["link_position"], orientation_xyzw=state["link_orientation_xyzw"],
                   reference_dynamics=data["measured_dynamics"][obj["object_id"]])
        if obj.get("urdf"):
            obj["initial_joints"] = dict(state.get("joint_positions") or {})
        elif not obj["fixed"]:
            rows = obj["reference_dynamics"]
            if len(rows) != 1:
                raise ValueError("rigid builder has ambiguous reference link dynamics")
            obj["measured_dynamics"] = next(iter(rows.values()))
            obj["mass"] = obj["measured_dynamics"]["mass_kg"]
    return {"archive": str(path.resolve()), "sha256": sha(path), "schema_version": data["schema_version"]}


def verify(plan):
    binding = plan.get("reference_state")
    if not binding:
        return
    path = Path(binding["archive"])
    if sha(path) != binding["sha256"]:
        raise ValueError("reference-state archive changed")
    data = json.loads(path.read_text())
    if plan.get("support_dynamics") != data["measured_dynamics"][plan["support_id"]]:
        raise ValueError("native support dynamics differ from reference")
    for obj in plan["objects"]:
        oid = obj["object_id"]
        state = data["initial_objects"][oid]
        if (obj["position"] != state["link_position"] or
                obj["orientation_xyzw"] != state["link_orientation_xyzw"] or
                obj.get("reference_dynamics") != data["measured_dynamics"][oid]):
            raise ValueError("native plan differs from canonical initial state/dynamics")
    return data


def pose_delta(expected, actual):
    p, q = expected
    r, s = actual
    q, s = np.asarray(q, float), np.asarray(s, float)
    if not all(np.isfinite(v).all() for v in map(np.asarray, (p, q, r, s))):
        raise ValueError("nonfinite initial pose")
    if any(abs(np.linalg.norm(v)-1) > 1e-4 for v in (q, s)):
        raise ValueError("nonunit initial quaternion")
    return {"position_delta_m": float(np.linalg.norm(np.asarray(p)-np.asarray(r))),
            "orientation_delta_rad": float(2*np.arccos(np.clip(abs(np.dot(q,s)), 0, 1)))}


def initial_audit(plan, world, frame):
    rows = {}
    for obj in plan["objects"]:
        oid = obj["object_id"]
        p, q = world.object_pose(oid)
        # world.object_pose is the adapter boundary: SapienWorld converts
        # SAPIEN Pose.q (wxyz) to xyzw. Convert back before Frame, whose
        # quaternion helper explicitly consumes wxyz.
        q_wxyz = np.asarray(q, dtype=float)[[3, 0, 1, 2]]
        delta = pose_delta((obj["position"], obj["orientation_xyzw"]),
                           (frame.to_ref(p), frame.rot_to_ref(q_wxyz)))
        delta["passed"] = delta["position_delta_m"] <= .02 and delta["orientation_delta_rad"] <= np.deg2rad(5)
        delta["reference"] = {"position": obj["position"], "orientation_xyzw": obj["orientation_xyzw"]}
        delta["native_in_reference_frame"] = {"position": frame.to_ref(p), "orientation_xyzw": frame.rot_to_ref(q_wxyz)}
        rows[oid] = delta
    joints = {}
    by_id = {o["object_id"]:o for o in plan["objects"]}
    for comp, binding in plan.get("articulations",{}).items():
        expected = by_id[binding["host"]]["initial_joints"][binding["joint_name"]]
        actual = world.joint_q(comp)
        tolerance = .005 if binding.get("joint_type")=="prismatic" else float(np.deg2rad(5))
        joints[comp] = {"reference":expected,"native":actual,"absolute_delta":abs(actual-expected),
                       "tolerance":tolerance,"passed":abs(actual-expected)<=tolerance}
    return {"status": "ready" if all(r["passed"] for r in [*rows.values(),*joints.values()]) else "input_mismatch",
            "articulations":joints,
            "position_tolerance_m": .02, "orientation_tolerance_deg": 5,
            "objects": rows, "source": "native measured link poses before policy; inverse rigid layout transform"}
