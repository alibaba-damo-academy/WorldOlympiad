"""Run the general expert on one case and write the reference simulation bundle.

    python -m wo.expert.run --task TASK.json --generation GEN_DIR --out OUT_DIR

Outputs (OUT_DIR):
    trajectory.json      renderer-compatible frames (objects, robot, active pair)
    expert_actions.npy   (N, 8) DROID joint-position actions at 15 Hz
    certificate.json     per-step predicate evidence, planning stats, variants
    scene_inputs.json    resolved scene (asset paths, poses, robot base)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
import time
from pathlib import Path

import numpy as np

from ..scene import resolve_scenecode, canonicalize_support_aliases
from ..sim.world import CONTROL_HZ, DROID_HOME, World, prepare_bodies
from ..spec import TaskSpec
from .embodiment import fit_gripper, place_robot
from .kin import Kinematics
from .rollout import free_actions, run_steps, sync

SCHEMA = "wo_expert_reference.v1"
# Endpoint primitives whose predicates must be re-evaluated after the settled
# home/retreat phase.  Push is an endpoint action too: omitting it here can
# turn a failed upright push into a certificate with no final_goals evidence.
SETTLED_ENDPOINT_PRIMITIVES = frozenset({"transport", "push", "reorient", "tool_contact"})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _droid_destinations(spec, scene) -> list[str]:
    """Return ordinary transport destinations eligible for envelope scaling.

    Articulated hosts remain coupled assemblies; scaling one as a destination
    can shrink handles and joint geometry below the graspable envelope.
    """
    hosts = {a.get("host") for a in scene.articulations.values()}
    return list(dict.fromkeys(s.goal.target for s in spec.steps
                              if s.primitive == "transport" and s.goal.target in scene.objects
                              and s.goal.target not in hosts))


def _require_authored_scale(spec, object_ids, scale):
    locked = [oid for oid in object_ids
              if spec.object(oid).geometry_constraints.get("overall_dimensions_m") is not None]
    if scale != 1.0 and locked:
        raise RuntimeError(f"authored dimensions are incompatible with embodiment fit; "
                           f"cannot resize {locked} by {scale:g}; author a compatible physical source")


def audit_initial_goals(world, spec, initial: dict[str, Any]) -> list[dict]:
    """Return endpoint goals already true on the settled initial state."""
    from .. import predicates
    findings = []
    for step in spec.steps:
        if step.primitive == "tool_contact" and step.goal.kind == "contact":
            continue
        ok, evidence = predicates.evaluate(world, step.goal, initial=initial)
        findings.append({"goal": step.goal.kind, "subject": step.subject,
                         "passed": ok, "evidence": evidence})
    return findings


def initial_containment_evidence(world, spec) -> list[dict]:
    """Require declared source containment to hold after physical settling."""
    from .. import predicates
    from ..spec import Predicate

    evidence = []
    for obj in spec.objects:
        if obj.initial_container is None:
            continue
        passed, measured = predicates.evaluate(
            world, Predicate("inside", obj.object_id, obj.initial_container))
        evidence.append({"subject": obj.object_id, "container": obj.initial_container,
                         "passed": passed, "evidence": measured})
    failed = [x["subject"] for x in evidence if not x["passed"]]
    if failed:
        raise RuntimeError(f"declared contents are not initially inside their source container: {failed}")
    return evidence


def archive_previous_attempt(out: Path) -> Path | None:
    """Keep retry evidence and remove stale results before any new simulation.

    Geometry caches stay in place; this is an evidence archive, not a relocated
    self-contained scene. Both failed attempts and successful references survive.
    """
    names = ("certificate.json", "trajectory.json", "expert_actions.npy",
             "scene_inputs.json", "task_snapshot.json", "replay", "replay.log", "replay_plan.json")
    previous = [out / name for name in names if (out / name).exists()]
    if not previous:
        return None
    history = out / ".history" / f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
    history.mkdir(parents=True)
    for path in previous:
        shutil.move(str(path), str(history / path.name))
    print(f"Previous reference evidence: {history}", flush=True)
    return history


def run_case(task_path: Path, generation: Path, out: Path, *, layout: str = "scene") -> dict:
    # Re-measure the task-bound source at the real expert admission boundary.
    # Do not trust a copied physical_size_validation.json from an older export.
    from ..tasks.scene import validate_source
    payload = json.loads(task_path.read_text())
    if payload.get("generation"):
        validate_source(task_path)
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    archive_previous_attempt(out)
    (out / "task_snapshot.json").write_bytes(task_path.read_bytes())
    spec = canonicalize_support_aliases(TaskSpec.load(task_path), generation)
    scene = resolve_scenecode(spec, generation)
    variants = []
    subjects = list(dict.fromkeys(s.subject for s in spec.steps
                                  if s.primitive in {"transport", "reorient"}))
    if layout == "droid":
        # one common scale for the task objects keeps the designed relations
        # (the bottle still fits the basket) while the destination enters the
        # DROID size envelope; grasp fitting below may shrink subjects further
        from .embodiment import destination_scale, scaled_scene as _scale

        dests = _droid_destinations(spec, scene)
        # Articulated hosts are assemblies, not graspable destinations. Scaling
        # them to the DROID envelope also scales handles/joint offsets and can
        # make the moving part physically ungraspable. Keep the authored
        # assembly coupled and scale only ordinary transport destinations.
        k = min([destination_scale(scene, d, out) for d in dests] or [1.0])
        if k < 0.999:
            resized = [oid for oid in dict.fromkeys([*dests, *subjects]) if oid in scene.objects]
            _require_authored_scale(spec, resized, round(k, 3))
            for oid in resized:
                scene = _scale(scene, oid, round(k, 3), out)
        variants.append({"policy": "common_task_object_scale", "objects": [*dests, *subjects],
                         "scale": round(k, 3)})
    for subject in subjects:
        previous_scale = scene.objects[subject].scale
        scene, info = fit_gripper(scene, subject, out)
        _require_authored_scale(spec, [subject], scene.objects[subject].scale / previous_scale)
        variants.append(info)
    try:
        scene, robot_placement = place_robot(scene, out, spec, layout=layout)
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA, "case_id": spec.case_id, "family": spec.family,
            "task": str(task_path), "task_sha256": _sha(out / "task_snapshot.json"),
            "generation": str(generation), "success": False, "steps": [],
            "frame_count": 0, "duration_s": 0.0,
            "source_assets": {obj.object_id: {
                "sdf_path": str(obj.sdf_path),
                "sdf_sha256": _sha(obj.sdf_path) if obj.sdf_path.is_file() else None,
            } for obj in [scene.support, *scene.objects.values()]},
            "failure": {"stage": "robot_placement", "type": type(exc).__name__, "detail": str(exc)},
            "wall_time_s": round(time.time() - t0, 1),
        }
        (out / "certificate.json").write_text(json.dumps(failure, indent=1))
        raise
    prepared = prepare_bodies(scene, out)
    d = World(scene, prepared, mode="dynamic")
    k = World(scene, prepared, mode="kinematic")

    def objects_world(src: World) -> World:
        o = World(scene, prepared, mode="objects")
        sync(o, src)
        return o

    # settle the scene before frame zero, then record the conditioning frame
    d.step_physics(240)
    initial = {oid: [v.tolist() for v in d.object_pose(oid)] for oid in d.bodies}
    # the renderer needs an active subject/target in every frame, conditioning included
    first = spec.steps[0]
    movable = [oid for oid in scene.objects] or [scene.support.object_id]
    d.active_subject = first.subject if first.subject in scene.objects else movable[0]
    d.active_target = first.goal.target or first.target
    if d.active_target not in scene.objects:
        d.active_target = next((o for o in movable if o != d.active_subject), None)
    d.record(commanded=[*DROID_HOME, 0.0])
    for _ in range(int(0.5 * CONTROL_HZ)):
        d.act([*DROID_HOME, 0.0])
    initial_relations = initial_containment_evidence(d, spec)
    # Preserve the audit for every reference, but reject a pre-solved initial
    # state only for the scoped single-step curated-development contract.
    # Multi-step tasks may intentionally start with a later endpoint already
    # true (for example open-then-close); rejecting those changes legacy
    # benchmark/replay semantics. Contact remains an event and is omitted by
    # audit_initial_goals.
    initial_goal_audit = audit_initial_goals(d, spec, initial)
    enforce_unsolved_initial = len(spec.steps) == 1
    if enforce_unsolved_initial and any(item["passed"] for item in initial_goal_audit):
        raise RuntimeError("initial goal already satisfied: " + json.dumps(initial_goal_audit))
    reports = run_steps(d, k, objects_world, spec.steps)
    passed = bool(reports) and all(r.passed for r in reports) and len(reports) == len(spec.steps)
    # return home and hold
    kin = Kinematics(k)
    sync(k, d)
    home_path = kin.plan(d.arm_q(), np.asarray(DROID_HOME), gripper=0.0)
    if home_path is not None:
        for q in free_actions(home_path):
            d.act([*q.tolist(), 0.0])
    for _ in range(int(CONTROL_HZ)):
        d.act([*DROID_HOME, 0.0])
    # final goal check on the settled end state (every step goal must still hold)
    from .. import predicates

    final = []
    for idx, step in enumerate(spec.steps):
        if step.primitive not in SETTLED_ENDPOINT_PRIMITIVES:
            continue
        if step.primitive == "tool_contact" and step.goal.kind == "contact":
            # Contact is an event and is no longer true after recoil/put-down;
            # retain the certificate produced by run_steps rather than relabel
            # a legitimate transient contact as an endpoint failure.
            report = reports[idx] if idx < len(reports) else None
            ok = bool(report and report.passed)
            ev = report.evidence if report else {"missing_step_report": True}
        else:
            ok, ev = predicates.evaluate_step(d, step, initial=initial)
        final.append({"goal": step.goal.kind, "subject": step.subject, "passed": ok, "evidence": ev})
    passed = passed and all(f["passed"] for f in final)

    actions = np.asarray([f["policy_source"]["commanded_action"] for f in d.frames], dtype=np.float32)
    np.save(out / "expert_actions.npy", actions)
    trajectory = {
        "schema_version": "scenecode_pybullet_droid_policy_rollout.v1",
        "producer": SCHEMA,
        "authority": {"physics": "PyBullet", "visual_assets": "official SceneCode SDF/glTF",
                      "robot": "RoboLab DROID Franka+Robotiq 2F-85 portable URDF"},
        "control_hz": CONTROL_HZ,
        "robot_mount": robot_placement["mount"],
        "frames": d.frames,
    }
    (out / "trajectory.json").write_text(json.dumps(trajectory))
    (out / "scene_inputs.json").write_text(json.dumps(scene.to_dict(), indent=1))
    certificate = {
        "schema_version": SCHEMA,
        "case_id": spec.case_id,
        "family": spec.family,
        "task": str(task_path),
        "task_sha256": _sha(out / "task_snapshot.json"),
        "generation": str(generation),
        "success": passed,
        # Preserve the settled pre-action predicate audit as readiness evidence;
        # a reference is not admissible merely because its terminal rollout
        # succeeds when the authored task began solved.
        "initial_goal_audit": initial_goal_audit,
        "steps": [r.__dict__ for r in reports],
        "final_goals": final,
        "embodiment_variants": variants,
        "robot_placement": robot_placement,
        "articulations": {comp: {k: a[k] for k in ("host", "joint_name", "child_link", "lower",
                                                    "upper", "q_closed", "q_open", "joint_type")}
                          for comp, a in d.articulation.items()},
        "frame_count": len(d.frames),
        "duration_s": len(d.frames) / CONTROL_HZ,
        "initial_object_poses": initial,
        "initial_relations": initial_relations,
        "object_attachment_constraints": 0,
        "wall_time_s": round(time.time() - t0, 1),
    }
    (out / "certificate.json").write_text(json.dumps(certificate, indent=1, default=float))
    d.close()
    k.close()
    return certificate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--generation", type=Path,
                    help="SceneCode generation dir (default: the task file's 'generation')")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout", choices=("scene", "droid"), default="scene",
                    help="droid: move task objects into the DROID tabletop workspace")
    ap.add_argument("--require-success", action="store_true", help="exit nonzero for a failed reference certificate")
    args = ap.parse_args()
    generation = args.generation or Path(json.loads(args.task.read_text())["generation"])
    cert = run_case(args.task, generation, args.out, layout=args.layout)
    print(json.dumps({k: cert[k] for k in ("case_id", "success", "frame_count", "duration_s",
                                           "wall_time_s")}))
    for s in cert["steps"]:
        print(s["step_index"], s["primitive"], s["passed"], s["stage"], s["detail"])
    if args.require_success and not cert["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
