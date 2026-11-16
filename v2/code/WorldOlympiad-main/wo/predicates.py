"""One predicate library, one threshold table.

Used by goal sampling (drop tests), expert certification and native-model
evaluation adapters.  Every check returns (passed, evidence) so the evidence is
recorded instead of a bare boolean.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .spec import Predicate

THRESHOLDS = {
    "upright_cos": math.cos(math.radians(12.0)),  # keep_upright / upright
    "inverted_cos": math.cos(math.radians(25.0)),
    "rest_speed_mps": 0.02,
    "rest_angular_rps": 0.3,
    "xy_overlap_fraction": 0.80,  # inside / covering / on footprint overlap
    "below_rim_fraction": 0.25,  # inside: bottom below rim by this fraction of target height
    "inserted_depth_fraction": 0.30,
    "stack_gap_m": 0.012,
    "hung_clearance_m": 0.02,
    "on_region_height_m": 0.04,
    "joint_open_fraction": 0.60,
    "joint_closed_fraction": 0.10,
    "pose_delta_m": 0.05,
}


def _quat_axis(orn_xyzw, axis=(0, 0, 1)) -> np.ndarray:
    x, y, z, w = orn_xyzw
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return r @ np.asarray(axis, dtype=float)


def _xy_overlap(a_lo, a_hi, b_lo, b_hi) -> float:
    """Fraction of a's xy footprint inside b's xy footprint."""
    w = max(0.0, min(a_hi[0], b_hi[0]) - max(a_lo[0], b_lo[0]))
    h = max(0.0, min(a_hi[1], b_hi[1]) - max(a_lo[1], b_lo[1]))
    area = max(1e-9, (a_hi[0] - a_lo[0]) * (a_hi[1] - a_lo[1]))
    return w * h / area


def support_below(world, object_id: str) -> str | None:
    """The body carrying most of the object's weight (upward contact normals)."""
    native = getattr(world, "support_below", None)
    if callable(native):  # non-PyBullet worlds (e.g. SAPIEN in wo.models.lingbot_va_client)
        return native(object_id)
    names = {b: k for k, b in world.bodies.items()}
    if getattr(world, "robot", None) is not None:
        names[world.robot] = "robot"
    names[world.plane] = "floor"
    own = world.bodies[object_id]
    load: dict[str, float] = {}
    for cp in world.p.getContactPoints(bodyA=own, physicsClientId=world.cid):
        # normal points from B to A; an upward normal means B pushes A up
        if cp[7][2] > 0.5:
            key = names.get(cp[2], "other")
            load[key] = load.get(key, 0.0) + max(cp[9], 1e-6) * cp[7][2]
    return max(load, key=load.get) if load else None


def at_rest(world, object_id: str) -> bool:
    v, w = world.p.getBaseVelocity(world.bodies[object_id], physicsClientId=world.cid)
    return (np.linalg.norm(v) < THRESHOLDS["rest_speed_mps"]
            and np.linalg.norm(w) < THRESHOLDS["rest_angular_rps"])


def evaluate(world, pred: Predicate, *, initial: dict[str, Any] | None = None) -> tuple[bool, dict]:
    """Evaluate one predicate on the current world state."""

    t = THRESHOLDS
    s = pred.subject
    ev: dict[str, Any] = {"kind": pred.kind}
    if pred.kind in {"joint_open", "joint_closed", "pressed"}:
        frac = world.articulation_fraction(s)
        ev["joint_fraction"] = frac
        if pred.kind == "joint_closed":
            return frac <= pred.params.get("threshold", t["joint_closed_fraction"]), ev
        return frac >= pred.params.get("threshold", t["joint_open_fraction"]), ev

    s_lo, s_hi = world.aabb(s)
    _, s_orn = world.object_pose(s)
    # Assets are not guaranteed to encode their semantic base-to-cap axis as
    # local +Z.  Authors may bind a measured source-frame axis in the goal
    # params (for example [0, 1, 0] for a bottle whose mesh is Y-long).
    # Keep +Z as the backwards-compatible default.
    semantic_axis = pred.params.get("semantic_up_axis", (0, 0, 1))
    if len(semantic_axis) != 3 or not np.isfinite(semantic_axis).all():
        raise ValueError("semantic_up_axis must be a finite 3-vector")
    semantic_axis = np.asarray(semantic_axis, dtype=float)
    norm = float(np.linalg.norm(semantic_axis))
    if norm < 1e-9:
        raise ValueError("semantic_up_axis must be non-zero")
    semantic_axis /= norm
    up = float(_quat_axis(s_orn, semantic_axis)[2])
    ev.update(subject_aabb=[s_lo.tolist(), s_hi.tolist()], subject_up_cos=up,
              semantic_up_axis=semantic_axis.tolist())
    if pred.kind == "upright":
        return up >= t["upright_cos"], ev
    if pred.kind == "inverted":
        return up <= -t["inverted_cos"], ev
    if pred.kind == "pose_delta":
        if initial is None:
            raise ValueError("pose_delta needs the initial state")
        moved = float(np.linalg.norm(world.object_pose(s)[0] - initial[s][0]))
        ev["displacement_m"] = moved
        return moved >= pred.params.get("threshold", t["pose_delta_m"]), ev

    tgt = pred.target
    if tgt is None:
        raise ValueError(f"{pred.kind} needs a target")
    t_lo, t_hi = world.aabb(tgt)
    touching = bool(world.contacts(s, tgt))
    below = support_below(world, s)
    overlap = _xy_overlap(s_lo, s_hi, t_lo, t_hi)
    ev.update(target_aabb=[t_lo.tolist(), t_hi.tolist()], touching=touching,
              supported_by=below, xy_overlap=overlap)
    height = t_hi[2] - t_lo[2]

    if pred.kind == "contact":
        return touching, ev
    if pred.kind == "inside":
        depth_ok = s_lo[2] <= t_hi[2] - t["below_rim_fraction"] * height
        ev["bottom_below_rim_m"] = float(t_hi[2] - s_lo[2])
        return bool(overlap >= t["xy_overlap_fraction"] and depth_ok and below == tgt), ev
    if pred.kind in {"on", "stacked"}:
        gap = float(s_lo[2] - t_hi[2])
        ev["gap_to_target_top_m"] = gap
        return bool(below == tgt and touching and gap > -0.5 * (s_hi[2] - s_lo[2])
                    and abs(gap) < t["stack_gap_m"] + 0.02), ev
    if pred.kind == "on_region":
        cx, cy = (s_lo[:2] + s_hi[:2]) / 2
        inside = t_lo[0] <= cx <= t_hi[0] and t_lo[1] <= cy <= t_hi[1]
        low = s_lo[2] <= t_hi[2] + t["on_region_height_m"]
        return bool(inside and low and below in {tgt, world.support_id}), ev
    if pred.kind == "inserted":
        depth = float(t_hi[2] - s_lo[2])
        ev["insertion_depth_m"] = depth
        need = t["inserted_depth_fraction"] * (s_hi[2] - s_lo[2])
        cx, cy = (s_lo[:2] + s_hi[:2]) / 2
        inside = t_lo[0] <= cx <= t_hi[0] and t_lo[1] <= cy <= t_hi[1]
        return bool(inside and depth >= need), ev
    if pred.kind == "covering":
        cover = _xy_overlap(t_lo, t_hi, s_lo, s_hi)  # target footprint covered by subject
        ev["target_covered_fraction"] = cover
        return bool(cover >= t["xy_overlap_fraction"] and touching and below == tgt), ev
    if pred.kind == "hung":
        others = [k for k in world.bodies if k not in {s, tgt}]
        free = all(not world.contacts(s, k) for k in others)
        lifted = s_lo[2] >= world.support_top() + t["hung_clearance_m"]
        ev.update(only_touches_target=free, lifted=bool(lifted))
        return bool(touching and free and lifted), ev
    raise ValueError(f"no evaluator for {pred.kind!r}")


def evaluate_step(world, step, *, initial: dict[str, Any] | None = None) -> tuple[bool, dict]:
    """Evaluate a step endpoint, including explicit terminal constraints.

    Constraints are part of the authored contract, not planner hints.  In
    particular, ``keep_upright`` applies to placement/push endpoints too;
    applying it here keeps execution, final certification, and native replay
    on the same predicate path.
    """
    ok, evidence = evaluate(world, step.goal, initial=initial)
    if step.constraints.get("keep_upright"):
        axis = np.asarray(step.goal.params.get("semantic_up_axis", (0, 0, 1)), dtype=float)
        norm = float(np.linalg.norm(axis))
        if axis.shape != (3,) or norm < 1e-9 or not np.isfinite(axis).all():
            raise ValueError("semantic_up_axis must be a finite non-zero 3-vector")
        axis /= norm
        _, orn = world.object_pose(step.subject)
        upright = float(_quat_axis(orn, axis)[2])
        evidence["upright_cos"] = upright
        ok = bool(ok and upright >= THRESHOLDS["upright_cos"])
    return bool(ok), evidence
