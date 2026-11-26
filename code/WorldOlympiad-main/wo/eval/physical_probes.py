"""Task-specific physical probes, measured from a trajectory (no VLM).

The generic video-physics question bank (gravity / buoyancy / melting / ...)
leaves a tabletop manipulation episode with exactly one relevant question, so
its score saturates near the judge's confidence ceiling and carries no signal.
These probes replace the "is this video physically possible at all" question
with "did the physical events this task is made of actually happen, and did
they happen the way contact mechanics require".

Every probe is derived from the TaskSpec primitives, so the set is task
specific without being case specific, and every probe reports the measured
quantity it scored, so a judge's claim can be cross-checked against geometry.

    python -m wo.eval.physical_probes EPISODE_DIR [EPISODE_DIR ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

METRIC_VERSION = "task_physics.v1"

# A probe is unmeasurable when the quantity it needs never exists in the
# episode (no lift segment, no joint record).  Unmeasurable probes are reported
# and excluded from the mean - they are not zeros.
UNMEASURABLE = None

LIFT_M = 0.02            # a rise this far above the resting height counts as lifted
NEAR_M = 0.08            # end effector this close to an object counts as engaged
FAR_M = 0.15             # beyond this the end effector cannot be acting on it
SETTLE_SPEED = 0.01      # m/s treated as at rest
MAX_SPEED_MPS = 6.0      # above a fall from any tabletop; a real jump is 100x this
DRIFT_M = 0.01           # unexplained displacement budget for an untouched object
ATTACH_STD_M = 0.02      # grasp is rigid when the object-TCP offset varies less


# Which question each probe answers: "could this have happened at all" versus
# "did the physical event this task is made of happen".
PLAUSIBILITY = {"no_teleport", "no_spontaneous_motion", "host_static", "carry_attachment",
                "joint_monotonic"}


def _soft(value: float, budget: float) -> float:
    """1.0 at value 0, decaying smoothly; 0.5 at one budget, ~0.1 at three."""
    return float(1.0 / (1.0 + max(value, 0.0) / max(budget, 1e-9)))


def _ramp(value: float, target: float) -> float:
    return float(np.clip(value / max(target, 1e-9), 0.0, 1.0))


def _speed(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    if len(t) < 2:
        return np.zeros(len(t))
    dt = np.gradient(t)
    return np.linalg.norm(np.gradient(pos, axis=0) / dt[:, None], axis=1)


def _runs(mask: np.ndarray, *, min_len: int = 3) -> list[tuple[int, int]]:
    """Half-open index ranges of consecutive True values at least min_len long."""
    out, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask)))
    return out


def _probe(name: str, score, measured: dict, note: str = "", *, not_applicable: bool = False) -> dict:
    return {"probe": name, "score": None if score is None else round(float(score), 4),
            "measured": {k: (None if v is None else round(float(v), 4)) for k, v in measured.items()},
            "note": note, "status": "not_applicable" if not_applicable else ("scored" if score is not None else "pending"),
            **({"reason": note} if not_applicable or score is None else {})}


# ------------------------------------------------------------------ probes
def lift_rise(obj: np.ndarray, t: np.ndarray) -> dict:
    """The transported object must leave its resting height."""
    rise = float(np.max(obj[:, 2]) - obj[0, 2])
    return _probe("lift_rise", _ramp(rise, LIFT_M), {"max_rise_m": rise, "required_m": LIFT_M},
                  "a transport whose object never rises never happened physically")


def carry_attachment(obj: np.ndarray, ee: np.ndarray, t: np.ndarray) -> dict:
    """While held, the object must move rigidly with the gripper."""
    held = (obj[:, 2] - obj[0, 2] > LIFT_M) & (np.linalg.norm(obj - ee, axis=1) < FAR_M)
    if held.sum() < 3:
        return _probe("carry_attachment", UNMEASURABLE, {"held_frames": float(held.sum())},
                      "no frame holds the object above its rest height within reach", not_applicable=True)
    offset = obj[held] - ee[held]
    std = float(np.mean(np.std(offset, axis=0)))
    return _probe("carry_attachment", _soft(std, ATTACH_STD_M),
                  {"offset_std_m": std, "budget_m": ATTACH_STD_M, "held_frames": float(held.sum())},
                  "a grasped object that slides or drifts inside the gripper is not in contact")


def release_settling(obj: np.ndarray, t: np.ndarray) -> dict:
    """After the last motion the object must come to rest, not keep moving."""
    v = _speed(obj, t)
    tail = max(1, int(round(1.0 / max(np.median(np.diff(t)) if len(t) > 1 else 1 / 15, 1e-6))))
    final = float(np.mean(v[-tail:]))
    return _probe("release_settling", _soft(final, SETTLE_SPEED),
                  {"final_speed_mps": final, "budget_mps": SETTLE_SPEED},
                  "an object still moving at the end of the window never settled onto its support")


def _free_bodies(tracks: dict, constrained: set[str]) -> dict:
    """Object tracks whose motion is unconstrained, so a displacement really
    does need an explanation.  Articulated parts swing on their hinge."""
    return {n: p for n, p in tracks.items()
            if n != "end_effector" and n not in constrained and len(p) and p.shape[-1] == 3}


def no_teleport(tracks: dict, t: np.ndarray, constrained: set[str] = frozenset()) -> dict:
    """No object may jump further in one frame than any real motion could carry
    it.  The budget is a speed limit, not a distance: a knocked-over block in
    free fall moves fast but continuously, and that is physically fine."""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 15
    budget = MAX_SPEED_MPS * dt
    worst, who = 0.0, ""
    for name, pos in _free_bodies(tracks, constrained).items():
        if len(pos) < 2:
            continue
        step = float(np.max(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        if step > worst:
            worst, who = step, name
    return _probe("no_teleport", _soft(max(worst - budget, 0.0), budget),
                  {"max_frame_step_m": worst, "budget_m": budget,
                   "implied_speed_mps": worst / dt},
                  f"largest single-frame jump: {who or 'none'}")


def no_spontaneous_motion(tracks: dict, t: np.ndarray,
                          constrained: set[str] = frozenset()) -> dict:
    """An object the gripper is nowhere near, and that nothing else is moving
    against, must stay where it is."""
    ee = tracks.get("end_effector")
    if ee is None:
        return _probe("no_spontaneous_motion", UNMEASURABLE, {}, "no end-effector track")
    free = _free_bodies(tracks, constrained)
    measured = False
    movers = [p for p in tracks.values()
              if len(p) and p.shape[-1] == 3 and float(np.max(np.linalg.norm(p - p[0], axis=1))) > LIFT_M]
    worst, who = 0.0, ""
    for name, pos in free.items():
        if len(pos) < 2:
            continue
        far = np.linalg.norm(pos - ee, axis=1) > FAR_M
        for other in movers:
            if other is pos or len(other) != len(pos):
                continue
            far &= np.linalg.norm(pos - other, axis=1) > FAR_M
        # within one uninterrupted stretch of being untouched: an object that was
        # moved earlier must not be charged for that motion again
        for lo, hi in _runs(far, min_len=3):
            measured = True
            seg = pos[lo:hi]
            # only horizontal drift is unexplained: falling and settling under
            # gravity needs no contact, and the vertical axis is where it shows
            drift = float(np.max(np.linalg.norm(seg[:, :2] - seg[0, :2], axis=1)))
            if drift > worst:
                worst, who = drift, name
    if not measured:
        return _probe("no_spontaneous_motion", UNMEASURABLE, {},
                      "every object is within reach of the gripper or another mover throughout", not_applicable=True)
    excess = max(worst - DRIFT_M, 0.0)
    return _probe("no_spontaneous_motion", _soft(excess, DRIFT_M),
                  {"max_unexplained_horizontal_drift_m": worst, "budget_m": DRIFT_M},
                  f"largest unexplained drift: {who}")


def contact_before_motion(obj: np.ndarray, ee: np.ndarray, t: np.ndarray, *, name: str) -> dict:
    """Whatever moves the object must have reached it first."""
    disp = np.linalg.norm(obj - obj[0], axis=1)
    moved = np.argmax(disp > LIFT_M) if np.any(disp > LIFT_M) else None
    if moved is None:
        return _probe(name, UNMEASURABLE, {"max_displacement_m": float(np.max(disp))},
                      "the object never moved, so nothing had to touch it", not_applicable=True)
    gap = float(np.min(np.linalg.norm(obj[:moved + 1] - ee[:moved + 1], axis=1)))
    excess = max(gap - NEAR_M, 0.0)
    return _probe(name, _soft(excess, NEAR_M),
                  {"gap_at_onset_m": gap, "budget_m": NEAR_M, "onset_s": float(t[moved])},
                  "an object that starts moving while the gripper is far away was moved by nothing")


def push_direction(obj: np.ndarray, goal: np.ndarray) -> dict:
    """A pushed object must travel toward its goal, not away from it."""
    moved = obj[-1] - obj[0]
    want = np.asarray(goal, float)[:len(moved)] - obj[0]
    n = np.linalg.norm(moved[:2]) * np.linalg.norm(want[:2])
    if n < 1e-6:
        return _probe("push_direction", UNMEASURABLE, {"displacement_m": float(np.linalg.norm(moved))},
                      "no planar motion to orient", not_applicable=True)
    cos = float(np.dot(moved[:2], want[:2]) / n)
    return _probe("push_direction", float(np.clip((cos + 1) / 2, 0, 1)),
                  {"cosine_to_goal": cos}, "work done against the goal direction is not a push")


def joint_monotonic(q: np.ndarray, t: np.ndarray) -> dict:
    """An articulated part must swing one way, not chatter."""
    dq = np.diff(q)
    total = float(np.sum(np.abs(dq)))
    net = float(abs(q[-1] - q[0]))
    if total < 1e-4:
        return _probe("joint_monotonic", UNMEASURABLE, {"total_travel_rad": total},
                      "the joint never moved", not_applicable=True)
    return _probe("joint_monotonic", net / total, {"net_rad": net, "total_travel_rad": total},
                  "a hinge driven back and forth is being fought, not opened")


def host_static(host: np.ndarray) -> dict:
    """Opening a door must not drag its cabinet across the room."""
    disp = float(np.max(np.linalg.norm(host - host[0], axis=1)))
    return _probe("host_static", _soft(max(disp - 0.02, 0.0), 0.02), {"host_displacement_m": disp},
                  "an articulated host that translates was pushed, not articulated")


# ------------------------------------------------------------------ assembly
def probes_for_spec(spec, tracks: dict[str, np.ndarray], t: np.ndarray,
                    articulations: list[dict] = (),
                    joints: dict[str, np.ndarray] | None = None) -> list[dict]:
    """Probe set derived from the task's primitives (never from its identity)."""
    joints = joints or {}
    constrained = {a["component"] for a in (articulations or [])}
    out = [no_teleport(tracks, t, constrained), no_spontaneous_motion(tracks, t, constrained)]
    ee = tracks.get("end_effector")
    arts = {a["component"]: a for a in (articulations or [])}
    for step in spec.steps:
        sub = step.subject
        pos = tracks.get(sub)
        if pos is None or ee is None or len(pos) < 3:
            out.append(_probe(f"{step.primitive}:{sub}", UNMEASURABLE, {},
                              "subject has no track in this episode"))
            continue
        if step.primitive in ("transport", "reorient"):
            out += [lift_rise(pos, t), carry_attachment(pos, ee, t), release_settling(pos, t),
                    contact_before_motion(pos, ee, t, name="contact_before_lift")]
        elif step.primitive == "push":
            out.append(contact_before_motion(pos, ee, t, name="contact_before_push"))
            out.append(release_settling(pos, t))
            dest = step.goal.target or step.target
            if dest in tracks and len(tracks[dest]):
                out.append(push_direction(pos, tracks[dest][0]))
        elif step.primitive == "tool_contact":
            out += [carry_attachment(pos, ee, t),
                    contact_before_motion(pos, ee, t, name="contact_before_strike")]
            tgt = step.goal.target or step.target
            if tgt in tracks and len(tracks[tgt]) == len(pos):
                out.append(contact_before_motion(tracks[tgt], pos, t, name="tool_reached_target"))
        elif step.primitive == "articulate":
            a = arts.get(sub)
            if a is None or sub not in joints:
                out.append(_probe(f"articulate:{sub}", UNMEASURABLE, {},
                                  "no recorded joint value for this component"))
            else:
                out.append(joint_monotonic(joints[sub], t))
            host = a and tracks.get(a.get("host"))
            if host is not None and len(host) > 1:
                out.append(host_static(host))
    return out


def aggregate(probes: list[dict]) -> dict:
    by_name: dict[str, list[float]] = {}
    for p in probes:
        if p["score"] is not None:
            by_name.setdefault(p["probe"], []).append(p["score"])
    per_probe = {k: float(np.mean(v)) for k, v in sorted(by_name.items())}
    groups = {}
    for label, members in (("plausibility", PLAUSIBILITY),
                           ("task_events", set(per_probe) - PLAUSIBILITY)):
        vals = [v for k, v in per_probe.items() if k in members]
        groups[label] = round(float(np.mean(vals)), 4) if vals else None
    present = [v for v in groups.values() if v is not None]
    applicability = {}
    for label in ("plausibility", "task_events"):
        declared = [p for p in probes if (p["probe"] in PLAUSIBILITY) == (label == "plausibility")]
        pending = [p for p in declared if p["score"] is None and p.get("status") != "not_applicable"]
        state = "pending" if pending else ("scored" if groups[label] is not None else "not_applicable")
        applicability[label] = {"status": state, "score": groups[label],
                                "declared_probes": [p["probe"] for p in declared]}
        if state != "scored":
            applicability[label]["reason"] = ("declared probe measurements missing" if pending else
                "no declared applicable event in this episode; see probe precondition evidence")
    unresolved = [p for p in probes if p["score"] is None and p.get("status") != "not_applicable"]
    state = "pending" if unresolved else ("scored" if present else "not_applicable")
    return {"status": state,
            "metric_version": METRIC_VERSION,
            # Original weights and numeric values are unchanged; only explicit
            # conditional applicability distinguishes absent events from missing tracks.
            "score": float(np.mean(present)) if present else None,
            "subscores": groups, "subscore_applicability": applicability,
            "measured_probe_count": len(per_probe), "probe_count": len(probes),
            "per_probe": {k: round(v, 4) for k, v in per_probe.items()},
            "probes": probes,
            **({"reason": "declared probe measurement missing"} if unresolved else
               ({"reason": "all declared probe preconditions absent"} if state == "not_applicable" else {}))}



def episode_probes(ep_dir: Path) -> dict:
    """Probe an episode from the trajectory its own simulator executed."""
    from ..spec import TaskSpec
    from .cae_episodes import entity_tracks, episode_articulations, require_entity_tracks

    ep = json.loads((ep_dir / "episode.json").read_text())
    ref = Path(ep["reference"])
    traj = ep_dir / "native_run/simulation/trajectory.json"
    if not traj.is_file():
        # a native importer limitation, or a run that never reached simulation:
        # reported as unscorable, never as a physics failure
        return {"status": ep.get("termination_reason") or "unmeasurable",
                "metric_version": METRIC_VERSION, "score": None, "subscores": {},
                "measured_probe_count": 0, "probe_count": 0, "per_probe": {}, "probes": [],
                "reason": ep.get("not_applicable_reason") or f"no trajectory at {traj}",
                "source": "simulator_execution", "episode": ep_dir.name,
                "task_success": ep.get("task_success")}
    case = json.loads((ref / "certificate.json").read_text())["case_id"]
    spec = TaskSpec.from_dict(json.loads((ep_dir.parents[1] / "tasks" / f"{case}.json").read_text()))
    graph = json.loads((ep_dir / "task_graph.json").read_text())
    arts = episode_articulations(graph, ref)
    objects = sorted({o.object_id for o in spec.objects} | {a["component"] for a in arts})
    tr = entity_tracks(traj, objects, arts, ref)
    require_entity_tracks(tr, ["end_effector", *dict.fromkeys(s.subject for s in spec.steps)], "task physics")
    tracks = {k: v.positions for k, v in tr.items()}
    frames = json.loads(traj.read_text())["frames"]
    t = np.array([f.get("time_s", i / 15.0) for i, f in enumerate(frames)], float)
    if np.any(np.diff(t) <= 0):
        t = np.arange(len(frames)) / 15.0
    joints = {}
    for a in arts:
        rec = [(f.get("articulation_positions") or {}).get(a["component"]) for f in frames]
        if all(x is not None for x in rec):
            joints[a["component"]] = np.asarray(rec, float)
    missing_joints = [s.subject for s in spec.steps if s.primitive == "articulate" and s.subject not in joints]
    if missing_joints:
        raise ValueError("task physics: missing recorded joint values: " + ", ".join(missing_joints))
    res = aggregate(probes_for_spec(spec, tracks, t, arts, joints))
    res.update(source="simulator_execution", episode=ep_dir.name,
               task_success=ep.get("task_success"))
    return res


def main() -> None:
    for d in sys.argv[1:]:
        r = episode_probes(Path(d))
        print(json.dumps({"episode": r["episode"], "success": r["task_success"],
                          "task_physics": None if r["score"] is None else round(r["score"], 3),
                          "subscores": r["subscores"], "per_probe": r["per_probe"]}))


if __name__ == "__main__":
    main()
