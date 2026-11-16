"""Physics-checked goal-pose sampling.

For a transport step we search subject poses near the target: candidate
orientation x candidate (x, y), then sweep the subject down from above until it
first touches something (container floor, stack top, hook, region surface).
Each resting candidate is drop-tested in a scratch dynamic world without the
robot; we keep poses that stay put and satisfy the step's goal predicate.
The same code serves inside / on / stacked / on_region / covering / inserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import predicates
from ..spec import Step
from ..sim.world import World
from .kin import Pose

DROP_TEST_S = 1.0
MAX_DRIFT_M = 0.02
MAX_TILT_DEG = 10.0
TARGET_MAX_DRIFT_M = 0.01


@dataclass
class Goal:
    obj_pose: Pose
    score: float
    evidence: dict


def _rotz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def _rotx(a: float) -> np.ndarray:
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])


def _semantic_axis(step: Step) -> np.ndarray:
    axis = np.asarray(step.goal.params.get("semantic_up_axis", (0, 0, 1)), dtype=float)
    norm = float(np.linalg.norm(axis))
    if axis.shape != (3,) or norm < 1e-9 or not np.isfinite(axis).all():
        raise ValueError("semantic_up_axis must be a finite non-zero 3-vector")
    return axis / norm


def _axis_to_z(axis: np.ndarray) -> np.ndarray:
    """Return a deterministic rotation taking a source axis to world +Z."""
    z = np.array([0., 0., 1.])
    v = np.cross(axis, z)
    c = float(np.dot(axis, z))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else _rotx(math.pi)
    vx = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def candidate_orientations(step: Step, start: Pose, n_yaw: int = 8) -> list[np.ndarray]:
    """Orientations allowed by explicit constraints and measured asset frame."""
    yaws = [_rotz(2 * math.pi * k / n_yaw) for k in range(n_yaw)]
    if step.goal.kind == "upright" or step.constraints.get("keep_upright"):
        return [yaw @ _axis_to_z(_semantic_axis(step)) for yaw in yaws]
    if step.goal.kind == "inverted":
        # The declared semantic axis is the asset's physical "up" direction.
        # Inverted means that same axis points down, not that local +Z is
        # flipped.  Keep the yaw sampling used by upright goals.
        return [yaw @ _axis_to_z(-_semantic_axis(step)) for yaw in yaws]
    return [r @ start.R for r in yaws]


def goal_region(w: World, step: Step) -> tuple[np.ndarray, np.ndarray, str]:
    """xy sampling region and the body the subject should end up near."""
    target = step.goal.target or step.target
    if step.primitive != "reorient" and target is not None and target in w.bodies:
        lo, hi = w.aabb(target)
        return lo, hi, target
    # reorientation (and goals without an object relation): set the subject down
    # near where it lies, on whatever supports it
    lo, hi = w.aabb(step.subject)
    c = (lo + hi) / 2
    pad = np.array([0.12, 0.12, 0.0])
    return np.minimum(lo, c - pad), np.maximum(hi, c + pad), w.support_id


def _touching_any(w: World, body: int, skip: set[int], margin: float = 0.0015) -> bool:
    w.p.performCollisionDetection(physicsClientId=w.cid)
    for b in [*w.bodies.values(), w.plane]:
        if b == body or b in skip:
            continue
        for cp in w.p.getClosestPoints(body, b, margin, physicsClientId=w.cid):
            # This search moves downward. Keep the existing forward clearance
            # against approaching faces, but do not turn parallel cavity walls
            # into an artificial smaller opening. Every actual intersection
            # still blocks motion, regardless of its normal.
            projected_clearance = margin * max(0.0, float(cp[7][2]))
            if float(cp[8]) <= projected_clearance:
                return True
    return False


def sweep_down(w: World, oid: str, xy, R: np.ndarray, z_top: float, z_floor: float,
               *, skip_ids: set[str] = frozenset()) -> float | None:
    """Lowest collision-free link height when lowering the subject at (x, y)."""
    body = w.bodies[oid]
    skip_bodies = {w.bodies[x] for x in skip_ids if x in w.bodies}
    q = Pose(np.zeros(3), R).q

    def free(z: float) -> bool:
        w.set_link_pose(body, (xy[0], xy[1], z), q)
        return not _touching_any(w, body, skip_bodies)

    if not free(z_top):
        return None
    z_free, z = z_top, z_top - 0.01
    while z > z_floor:
        if not free(z):
            lo_c, hi_f = z, z_free
            for _ in range(8):
                mid = (lo_c + hi_f) / 2
                if free(mid):
                    hi_f = mid
                else:
                    lo_c = mid
            return hi_f
        z_free, z = z, z - 0.01
    return None


def cavity_xy_candidates(w: World, target: str, lo: np.ndarray, hi: np.ndarray,
                         *, grid_size: int = 17) -> list[np.ndarray]:
    """Probe actual collision geometry, prioritizing deep, central openings.

    Rays which hit another object are not evidence of a cavity. These are only
    search hints: sweep, settling, and predicate checks remain collision enabled.
    """
    centre = (lo[:2] + hi[:2]) / 2
    samples = []
    for x in np.linspace(lo[0], hi[0], grid_size):
        for y in np.linspace(lo[1], hi[1], grid_size):
            hit = w.p.rayTest([x, y, hi[2] + 0.01],
                              [x, y, lo[2] - 0.01], physicsClientId=w.cid)[0]
            if hit[0] != w.bodies[target]:
                continue
            depth = hi[2] - float(hit[3][2])
            if depth <= 0.0015:
                continue
            xy = np.array([x, y])
            samples.append((depth, float(np.linalg.norm(xy - centre)), xy))
    samples.sort(key=lambda v: (-round(v[0], 4), v[1]))
    return [v[2] for v in samples]


def sample_goals(w: World, step: Step, start: Pose, *, n_xy: int = 24, max_goals: int = 24,
                 seed: int = 0) -> tuple[list[Goal], dict]:
    """``w`` must be an ``objects``-mode world in its initial state."""
    if w.mode != "objects":
        raise ValueError("goal sampling needs an objects-mode world")
    rng = np.random.default_rng(seed)
    s = step.subject
    p, c = w.p, w.cid
    clean = p.saveState(physicsClientId=c)
    s_lo, s_hi = w.aabb(s)
    t_lo, t_hi, t = goal_region(w, step)
    s_size = s_hi - s_lo
    link_p, _ = w.object_pose(s)
    # offset from the link origin to the AABB centre, so sampled xy positions the AABB
    centre_off = (s_lo + s_hi) / 2 - link_p
    target_pose0 = w.object_pose(t)[0]
    stats = {"tried": 0, "swept": 0, "settled": 0, "predicate": 0, "reject": {}}

    def rej(why: str) -> None:
        stats["reject"][why] = stats["reject"].get(why, 0) + 1

    goals: list[Goal] = []
    orientations = candidate_orientations(step, start)
    centre = np.array([(t_lo[0] + t_hi[0]) / 2, (t_lo[1] + t_hi[1]) / 2])
    pts = [centre]
    if step.goal.kind == "inserted" and t in w.bodies:
        pts += cavity_xy_candidates(w, t, t_lo, t_hi)[:n_xy - 1]
    pts += [rng.uniform(t_lo[:2], t_hi[:2]) for _ in range(max(0, n_xy - len(pts)))]
    for R in orientations:
        rot_off = R @ start.R.T @ centre_off
        for xy_c in pts:
            stats["tried"] += 1
            p.restoreState(clean, physicsClientId=c)
            xy = xy_c - rot_off[:2]
            z_top = max(t_hi[2], s_hi[2]) + s_size[2] + 0.05
            z = sweep_down(w, s, xy, R, z_top, w.support_top() - 0.02)
            if z is None:
                rej("no_free_height")
                continue
            stats["swept"] += 1
            pose = Pose(np.array([xy[0], xy[1], z + 0.002]), R)
            w.set_link_pose(w.bodies[s], pose.p, pose.q)
            w.step_physics(int(DROP_TEST_S * 240))
            end = Pose.from_pq(*w.object_pose(s))
            drift = float(np.linalg.norm(end.p - pose.p))
            tilt = _angle_deg(end.R, pose.R)
            t_drift = float(np.linalg.norm(w.object_pose(t)[0] - target_pose0))
            if drift > MAX_DRIFT_M:
                rej("drift")
                continue
            if tilt > MAX_TILT_DEG:
                rej("tilt")
                continue
            if t_drift > TARGET_MAX_DRIFT_M:
                rej("target_moved")
                continue
            if not predicates.at_rest(w, s):
                rej("not_at_rest")
                continue
            stats["settled"] += 1
            ok, ev = predicates.evaluate(w, step.goal)
            if step.constraints.get("keep_upright"):
                up = float((end.R @ _semantic_axis(step))[2])
                ev["upright_cos"] = up
                ok = ok and up >= predicates.THRESHOLDS["upright_cos"]
            if not ok:
                rej(f"predicate:{step.goal.kind}")
                stats.setdefault("example_predicate_failure", {
                    k: v for k, v in ev.items() if k not in {"subject_aabb", "target_aabb"}})
                continue
            stats["predicate"] += 1
            centre_dist = float(np.linalg.norm(end.p[:2] - (t_lo[:2] + t_hi[:2]) / 2))
            score = -drift * 10 - tilt / 20 - centre_dist
            ev.update(drift_m=drift, tilt_deg=tilt, target_drift_m=t_drift)
            goals.append(Goal(pose, score, ev))
    p.restoreState(clean, physicsClientId=c)
    goals.sort(key=lambda g: -g.score)
    return goals[:max_goals], stats
