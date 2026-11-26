"""Monitored expert rollout for transport-family steps.

Planning happens in a kinematic mirror of the dynamic world; execution happens
in the dynamic world with 15 Hz DROID joint-position actions.  Before the final
descent the in-hand object pose is *measured* and the release pose re-solved,
so slip or tilt during the carry is corrected instead of being baked in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import predicates
from ..spec import Step
from ..sim.world import CONTROL_HZ, DROID_HOME, World
from .goals import Goal, sample_goals
from .grasps import Grasp, sample_grasps, approach_fraction
from .kin import Kinematics, Pose, ee_to_tcp, tcp_to_ee, aperture_tcp_offset

FREE_JOINT_SPEED = 0.9  # rad/s
FINE_SPEED = 0.08  # m/s for Cartesian approach / descent / retreat
PREGRASP_BACKOFF = 0.10
LIFT_M = 0.10
HOLD_FRAMES = 12
SETTLE_TOL_RAD = 0.01


class StepFailed(RuntimeError):
    def __init__(self, stage: str, detail: str, evidence: dict | None = None):
        super().__init__(f"{stage}: {detail}")
        self.stage = stage
        self.detail = detail
        self.evidence = evidence or {}


@dataclass
class Plan:
    grasp: Grasp
    goal: Goal
    q_pre: np.ndarray
    q_grasp: np.ndarray
    approach_path: list[np.ndarray]
    score: float
    q_release: np.ndarray | None = None
    # Reorientation may require releasing above the support: a closed gripper at
    # the final table pose is itself a collision.  Mark this bounded fallback so
    # execution can drop only after a collision-checked carry.
    release_drop: bool = False
    approach_gripper: float = 0.0
    overhead_rise: float = 0.0


@dataclass
class StepReport:
    step_index: int
    primitive: str
    passed: bool
    stage: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


def sync(k: World, d: World) -> None:
    """Copy object poses and arm state from the dynamic world into a mirror."""
    for oid, body in d.bodies.items():
        if oid in k.bodies:
            pp, qq = d.object_pose(oid)
            k.set_link_pose(k.bodies[oid], pp, qq)
    for comp in getattr(d, "articulation", {}):
        if comp in getattr(k, "articulation", {}):
            k.set_joint_q(comp, d.joint_q(comp))
    if k.robot is not None and d.robot is not None:
        k.set_arm(d.arm_q())
        k.set_gripper_state(d.gripper_fraction())


# ------------------------------------------------------------------ actions
def _resample(qs: list[np.ndarray], duration_s: float) -> list[np.ndarray]:
    qs = [np.asarray(q) for q in qs]
    # A deliberately skipped motion (e.g. overhead release) has no waypoints.
    # Preserve it as a no-op rather than indexing a nonexistent segment.
    if len(qs) <= 1:
        return qs
    seg = [np.max(np.abs(qs[i + 1] - qs[i])) + 1e-9 for i in range(len(qs) - 1)]
    cum = np.concatenate([[0], np.cumsum(seg)])
    n = max(1, int(math.ceil(duration_s * CONTROL_HZ)))
    out = []
    for s in np.linspace(0, cum[-1], n + 1)[1:]:
        i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(qs) - 2)
        t = (s - cum[i]) / (cum[i + 1] - cum[i])
        out.append(qs[i] + t * (qs[i + 1] - qs[i]))
    return out


def free_actions(qs: list[np.ndarray]) -> list[np.ndarray]:
    total = sum(np.max(np.abs(np.asarray(qs[i + 1]) - qs[i])) for i in range(len(qs) - 1))
    return _resample(qs, max(0.4, total / FREE_JOINT_SPEED * 1.3))


def fine_actions(qs: list[np.ndarray], length_m: float) -> list[np.ndarray]:
    return _resample(qs, max(0.3, length_m / FINE_SPEED))


class Executor:
    def __init__(self, d: World, k: World, kin: Kinematics) -> None:
        self.d, self.k, self.kin = d, k, kin
        self.gripper = 0.0

    def run(self, qs: list[np.ndarray], gripper: float | None = None, phase: str = "") -> None:
        if gripper is not None:
            self.gripper = gripper
        if phase:
            self.d.phase = phase
        for q in qs:
            self.d.act([*np.asarray(q).tolist(), self.gripper])

    def settle(self, q: np.ndarray, max_frames: int = 20) -> None:
        for _ in range(max_frames):
            if np.max(np.abs(self.d.arm_q() - q)) < SETTLE_TOL_RAD:
                return
            self.d.act([*np.asarray(q).tolist(), self.gripper])

    @property
    def target(self) -> np.ndarray:
        cmd = self.d.frames[-1]["policy_source"]["commanded_action"] if self.d.frames else None
        return np.asarray(cmd[:7]) if cmd else self.d.arm_q()


def stabilized_transport_actions(actions: list[np.ndarray], repeats: int = 2) -> list[np.ndarray]:
    """Repeat each transport waypoint to reduce inertial slip in load-bearing grasps.

    Position-controlled fingers need several physics frames to react to changing
    contact normals. Holding each already collision-certified waypoint briefly
    is preferable to forcing the jaws closed, which can expel light or confined
    objects.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    return [q for q in actions for _ in range(repeats)]


def close_transport_gripper(ex: Executor, q: np.ndarray, subject: str) -> list[dict]:
    """Ramp to bilateral contact and hold that aperture, not through the object.

    Fully shutting a position-controlled jaw after acquiring a convex object
    can expel it. Stop only on actual bilateral contact; the caller still
    requires contact after the hold, physical lift, and successful carry.
    No attachment or object repositioning is performed.
    """
    trace = []
    start = ex.d.gripper_fraction()
    for fraction in np.linspace(start, 1.0, HOLD_FRAMES + 1)[1:]:
        ex.run([q], gripper=float(fraction), phase="grasp")
        loaded = getattr(ex.d, "load_bearing_grasped", ex.d.grasped)
        bilateral = bool(loaded(subject))
        trace.append({"command": float(fraction),
                      "measured_fraction": float(ex.d.gripper_fraction()),
                      "bilateral": bilateral,
                      "load_bearing": bilateral})
        if bilateral:
            break
    # A contact report can become load-bearing during the ramp but still be
    # marginal once the object settles between the pads.  Give that certified
    # contact a short measured settling hold, then make one bounded tightening
    # pass only if the load-bearing predicate was lost.  This stays physical:
    # no attachment or pose correction is performed, and the caller still
    # rechecks the grasp after lift and carry.
    ex.run([q] * HOLD_FRAMES, gripper=float(fraction), phase="grasp")
    loaded = getattr(ex.d, "load_bearing_grasped", ex.d.grasped)
    if trace and not loaded(subject):
        start = float(ex.d.gripper_fraction())
        for fraction in np.linspace(start, 1.0, HOLD_FRAMES + 1)[1:]:
            ex.run([q], gripper=float(fraction), phase="grasp_retain")
            bilateral = bool(loaded(subject))
            trace.append({"command": float(fraction),
                          "measured_fraction": float(ex.d.gripper_fraction()),
                          "bilateral": bilateral,
                          "load_bearing": bilateral,
                          "retention_pass": True})
            if bilateral:
                ex.run([q] * HOLD_FRAMES, gripper=float(fraction), phase="grasp_retain")
                break
    return trace


def resolve_target(w: World, step: Step) -> str:
    """Body the subject ends on/in/near.  Reorientation and other goals without an
    object relation (upright, inverted) resolve to the support surface."""
    for t in (step.goal.target, step.target):
        if t is not None and t in w.bodies:
            return t
    return w.support_id


# ------------------------------------------------------------------ planning
def plan_transport(k: World, kin: Kinematics, o: World, step: Step, *, seed: int = 0,
                   max_grasps: int = 24, max_goals: int = 16,
                   _tilted_top: bool = False, _aperture: bool = False) -> tuple[list[Plan], dict]:
    s = step.subject
    obj = Pose.from_pq(*k.object_pose(s))
    sampling = {}
    if _tilted_top:
        sampling['tilted_top'] = True
    if _aperture:
        sampling['aperture_sampling'] = True
    grasps = sample_grasps(k, s, **sampling)
    goals, goal_stats = sample_goals(o, step, Pose.from_pq(*o.object_pose(s)), seed=seed)
    info: dict[str, Any] = {"grasps": len(grasps), "goals": len(goals), "goal_sampling": goal_stats,
                            "rejections": {}}

    def reject(why: str) -> None:
        info["rejections"][why] = info["rejections"].get(why, 0) + 1

    if not grasps:
        raise StepFailed("planning", "no antipodal grasp on the subject", info)
    if not goals:
        raise StepFailed("planning", "no physically stable goal pose satisfies the predicate", info)
    q_home = k.arm_q()
    t_lo, t_hi = k.aabb(resolve_target(k, step))
    s_lo, _ = k.aabb(s)
    plans: list[Plan] = []
    for g in grasps[:max_grasps]:
        aperture = approach_fraction(k, g.width) if _aperture else 0.0
        tcp = g.world(obj)
        a = tcp.R[:, 2]
        offset = aperture_tcp_offset(k, aperture)
        ee_grasp = tcp_to_ee(tcp, offset)
        ee_pre = ee_grasp.moved(-a * PREGRASP_BACKOFF)
        kin.attached, kin.ignore = None, set()
        q_pre = kin.ik(ee_pre, seed=q_home)
        if q_pre is None:
            reject(f"pregrasp_ik:{g.kind}")
            continue
        hit = kin.in_collision(q_pre, gripper=aperture)
        if hit is not None:
            reject(f"pregrasp_collision:{g.kind}:{hit}")
            # Preserve the usual successful branch; only search alternate
            # converged branches when the first one is physically blocked.
            q_pre = (kin.ik(ee_pre, seed=q_home,
                           accept_solution=lambda q: kin.in_collision(q, gripper=aperture) is None)
                     if aperture else kin.ik(ee_pre, seed=q_home, collision_free=True))
            if q_pre is None or kin.in_collision(q_pre, gripper=aperture) is not None:
                continue
            info["pregrasp_branch_recoveries"] = info.get("pregrasp_branch_recoveries", 0) + 1
        kin.ignore = {s}
        approach_detail: dict[str, Any] = {}
        approach = kin.cartesian_path(q_pre, ee_pre, ee_grasp, gripper=aperture,
                                      diagnostics=approach_detail)
        kin.ignore = set()
        if approach is None:
            info.setdefault("approach_failures", []).append({"grasp_kind": g.kind,
                                                             **approach_detail})
            reject("approach_path")
            recovered = []
            def accept_approach(q):
                # The pregrasp must be free with ALL bodies present. Only the
                # subsequent original grasp descent exempts the subject.
                kin.ignore = set()
                if kin.in_collision(q, gripper=aperture) is not None:
                    return False
                detail = {}
                kin.ignore = {s}
                try:
                    path = kin.cartesian_path(q, ee_pre, ee_grasp, gripper=aperture,
                                              diagnostics=detail)
                finally:
                    kin.ignore = set()
                if path is None:
                    info.setdefault("alternate_approach_failures", []).append(detail)
                    return False
                recovered.append(path)
                return True
            # Endpoint-clear IK alone does not certify a branch: the elbow can
            # intersect a container midway through descent. Search only after
            # the ordinary approach fails, preserving existing successful plans.
            alternate = kin.ik(ee_pre, seed=q_home, accept_solution=accept_approach)
            if alternate is None or not recovered:
                continue
            q_pre, approach = alternate, recovered[-1]
            info["approach_branch_recoveries"] = info.get("approach_branch_recoveries", 0) + 1
        q_grasp = approach[-1]
        # open fingers must not penetrate the subject at the grasp pose
        k.set_arm(q_grasp)
        k.set_gripper_state(aperture)
        if any(cp[8] < -0.002 for cp in k.p.getClosestPoints(k.robot, k.bodies[s], 0.0,
                                                            physicsClientId=k.cid)):
            reject("open_finger_penetration")
            continue
        t_ee_obj = ee_grasp.inv() @ obj
        for goal in goals[:max_goals]:
            ee_rel = tcp_to_ee(goal.obj_pose @ g.tcp_in_obj)
            bottom_rel = s_lo[2] - obj.p[2] + goal.obj_pose.p[2]
            rise = max(LIFT_M, t_hi[2] + 0.04 - bottom_rel)
            ee_prerel = ee_rel.moved([0, 0, rise])
            # release configuration: subject sits at the goal, fingers around it
            k.set_link_pose(k.bodies[s], goal.obj_pose.p, goal.obj_pose.q)
            kin.ignore = {s}
            why = None
            release_drop = False
            overhead_rise = 0.0
            q_rel = kin.ik(ee_rel, seed=q_grasp)
            if q_rel is None:
                why = "release_ik"
            else:
                hit = kin.in_collision(q_rel, gripper=1.0)
                if hit is not None:
                    why = f"release_collision:{hit}"
                elif kin.cartesian_path(q_rel, ee_rel, ee_prerel, gripper=1.0) is None:
                    why = "release_descent_path"
            if why is not None:
                # A source-grasp rest pose can select an unusable release IK
                # branch, especially across large orientation changes. Try the
                # independent home-seeded bounded search only after failure;
                # certify both endpoint and original checked retreat throughout.
                original_why = why
                def accept_release(q):
                    return (kin.in_collision(q, gripper=1.0) is None and
                            kin.cartesian_path(q, ee_rel, ee_prerel,
                                               gripper=1.0) is not None)
                q_alt = kin.ik(ee_rel, seed=q_home, accept_solution=accept_release)
                if q_alt is not None and accept_release(q_alt):
                    q_rel, why = q_alt, None
                    info["release_branch_recoveries"] = info.get("release_branch_recoveries", 0) + 1
                    info.setdefault("recovered_release_rejections", []).append(original_why)
                # Some open containers are intentionally narrower than the
                # closed gripper. Requiring the fingers to enter the container
                # before opening makes a physically valid drop impossible. A
                # collision-certified overhead release is the general fallback
                # for inside goals (and for reorientation), not a container or
                # object-specific exemption: both closed and open held poses
                # remain checked against every scene body.
                if why is not None and allows_overhead_release(step):
                    def accept_open(q):
                        return overhead_release_free(kin, s, t_ee_obj, q)
                    suspended = search_overhead_release(kin, ee_rel, rise, q_home, accept_open)
                    if suspended is not None:
                        q_rel, overhead_rise = suspended
                        why, release_drop = None, True
                        info["overhead_drop_releases"] = info.get(
                            "overhead_drop_releases", 0) + 1
                        info.setdefault("overhead_rises_m", []).append(overhead_rise)
            kin.ignore = set()
            k.set_link_pose(k.bodies[s], obj.p, obj.q)
            if why is not None:
                reject(why)
                continue
            plans.append(Plan(g, goal, q_pre, q_grasp, approach, g.score + goal.score,
                              q_rel, release_drop, aperture, overhead_rise))
            break
        if len(plans) >= 10:
            break
    plans.sort(key=lambda pl: -pl.score)
    info["feasible_plans"] = len(plans)
    if not plans:
        # Ranking is an optimisation, not evidence of infeasibility. Preserve
        # successful bounded searches exactly, but exhaust the already sampled
        # certified candidates before declaring the task unreachable.
        if len(grasps) > max_grasps or len(goals) > max_goals:
            k.set_arm(q_home)
            k.set_gripper_state(0.0)
            kin.attached, kin.ignore = None, set()
            try:
                plans, expanded_info = plan_transport(
                    k, kin, o, step, seed=seed,
                    max_grasps=max(max_grasps, len(grasps)),
                    max_goals=max(max_goals, len(goals)), _tilted_top=_tilted_top,
                    _aperture=_aperture)
            except StepFailed as exc:
                exc.evidence["initial_ranked_search"] = info
                raise
            expanded_info["initial_ranked_search"] = info
            expanded_info["expanded_candidate_search"] = True
            return plans, expanded_info
        if not _aperture and any(f.get('reason') == 'collision'
                                 for f in info.get('approach_failures', [])):
            k.set_arm(q_home)
            kin.attached, kin.ignore = None, set()
            try:
                plans, aperture_info = plan_transport(
                    k, kin, o, step, seed=seed, max_grasps=max(max_grasps, len(grasps)),
                    max_goals=max(max_goals, len(goals)), _tilted_top=_tilted_top,
                    _aperture=True)
            except StepFailed as exc:
                exc.evidence['full_open_search'] = info
                raise
            aperture_info['full_open_search'] = info
            aperture_info['measured_aperture_fallback'] = True
            return plans, aperture_info
        if not _tilted_top and needs_tilted_contact_search(step, info):
            # IK branch search cannot remove a rim crossing shared by every
            # vertical TCP path. Try new certified contact frames only after
            # exhausting the ordinary ranked and expanded candidate searches.
            k.set_arm(q_home)
            k.set_gripper_state(0.0)
            kin.attached, kin.ignore = None, set()
            try:
                plans, tilted_info = plan_transport(
                    k, kin, o, step, seed=seed, max_grasps=64,
                    max_goals=max(max_goals, len(goals)), _tilted_top=True,
                    _aperture=_aperture)
            except StepFailed as exc:
                exc.evidence["ordinary_search"] = info
                raise
            tilted_info["ordinary_search"] = info
            tilted_info["tilted_top_fallback"] = True
            return plans, tilted_info
        raise StepFailed("planning", "no grasp/goal pair is reachable without collision", info)
    return plans, info


def allows_overhead_release(step):
    """Whether a failed support-level release may be replaced by a drop.

    Inside predicates describe a container drop, so the object need not be
    carried through the container wall with closed fingers.  This is a
    semantic goal rule, independent of any particular asset or task case.
    """
    return (step.primitive == "reorient" or
            getattr(step.goal, "kind", None) == "inside")


def needs_tilted_contact_search(step, info):
    """A blocked approach or inverted release may require a new contact frame.

    IK restarts cannot change the rigid grasp transform.  Reorientation can
    turn a vertical source grasp into an upward-pointing, unreachable wrist.
    Search the existing certified tilted contacts only after ordinary contacts
    are exhausted; this does not alter goal poses or collision checks.
    """
    approach_blocked = any(
        f.get('grasp_kind') == 'top' and f.get('reason') == 'collision'
        for f in info.get('approach_failures', []))
    release_blocked = step.primitive == 'reorient' and any(
        count and (reason == 'release_ik' or reason.startswith('release_collision:')
                   or reason == 'release_descent_path')
        for reason, count in info.get('rejections', {}).items())
    return approach_blocked or release_blocked


def search_overhead_release(kin, ee_rel, minimum_rise, seed, accept):
    """Try lowest certified suspension first, bounded to four lift increments.

    A flipped wrist can extend below the held object; object-only clearance
    therefore does not certify robot clearance. Every candidate retains full
    held-object and closed/open robot checks supplied by the caller.
    """
    for extra in range(4):
        rise = minimum_rise + extra * LIFT_M
        pose = ee_rel.moved([0, 0, rise])
        q = kin.ik(pose, seed=seed, accept_solution=accept)
        if q is not None and accept(q):
            return q, rise
    return None


def overhead_release_free(kin, subject, t_ee_obj, q, gripper=1.0):
    """Certify suspended hold/open endpoints without support exemptions."""
    saved = kin.attached, kin.ignore
    kin.attached, kin.ignore = (subject, t_ee_obj), set()
    try:
        return (kin.in_collision(q, gripper=gripper) is None and
                kin.in_collision(q, gripper=0.0) is None)
    finally:
        kin.attached, kin.ignore = saved


# ------------------------------------------------------------------ execution
def release_snapshot(d: World, subject: str, target: str) -> dict:
    """Persist measured release evidence even when failed attempts are restored.

    These are observations only: no commands, collision exemptions or predicate
    changes. Contact link IDs distinguish a trapped finger from palm support.
    """
    def pose(oid):
        p, q = d.object_pose(oid)
        return {"position": np.asarray(p).tolist(), "quaternion": np.asarray(q).tolist()}

    ep, eq = d.ee_pose()
    names = {bid: oid for oid, bid in d.bodies.items()}
    names[d.robot] = "robot"
    contacts = [{"body": names.get(cp[2], str(cp[2])),
                 "subject_link": int(cp[3]), "other_link": int(cp[4]),
                 "position_on_subject": list(cp[5]), "position_on_other": list(cp[6]),
                 "normal_on_other": list(cp[7]), "distance_m": float(cp[8]),
                 "normal_force_n": float(cp[9])}
                for cp in d.contacts(subject)]
    return {"subject": pose(subject), "target": pose(target),
            "ee": {"position": np.asarray(ep).tolist(), "quaternion": np.asarray(eq).tolist()},
            "gripper_fraction": float(d.gripper_fraction()),
            "support": predicates.support_below(d, subject), "contacts": contacts}


def execute_transport(d: World, k: World, kin: Kinematics, step: Step, plan: Plan) -> dict:
    s, t = step.subject, resolve_target(d, step)
    ex = Executor(d, k, kin)
    ev: dict[str, Any] = {"grasp_kind": plan.grasp.kind, "grasp_width_m": plan.grasp.width}
    d.active_subject, d.active_target = s, t

    # 1. reach the pre-grasp
    sync(k, d)
    kin.attached, kin.ignore = None, set()
    aperture = plan.approach_gripper
    ev['approach_gripper_fraction'] = aperture
    for fraction in np.linspace(d.gripper_fraction(), aperture, 12):
        hit = kin.in_collision(d.arm_q(), gripper=float(fraction))
        if hit is not None:
            raise StepFailed('reach', 'initial aperture transition collision', {'collision': hit})
    path = kin.plan(d.arm_q(), plan.q_pre, gripper=aperture)
    if path is None:
        raise StepFailed("reach", "no collision-free path to the pre-grasp")
    if aperture:
        ex.run([d.arm_q()] * HOLD_FRAMES, gripper=aperture, phase='prepare_aperture')
    ex.run(free_actions(path), gripper=aperture, phase="approach")
    ex.settle(plan.q_pre)
    # 2. approach along the grasp axis
    ex.run(fine_actions(plan.approach_path, PREGRASP_BACKOFF), phase="descend")
    ex.settle(plan.q_grasp)
    # 3. close. Persist observations for failed grasps as well as releases;
    # restoring a candidate otherwise erases the actual contact geometry.
    ev['grasp_snapshots'] = {'before_close': release_snapshot(d, s, t)}
    ev['closure_trace'] = close_transport_gripper(ex, plan.q_grasp, s)
    closure = ev['closure_trace'][-1]['command']
    ev['hold_gripper_fraction'] = closure
    ev['grasp_snapshots']['after_close'] = release_snapshot(d, s, t)
    loaded_grasp = getattr(d, "load_bearing_grasped", d.grasped)
    if not loaded_grasp(s):
        raise StepFailed("grasp", "fingers do not establish a loaded bilateral grasp after closing", ev)
    # 4. lift straight up
    sync(k, d)
    ee0 = Pose.from_pq(*k.ee_pose())
    z0 = d.object_pose(s)[0][2]
    kin.ignore = {s, t, d.support_id}
    lift = kin.cartesian_path(d.arm_q(), ee0, ee0.moved([0, 0, LIFT_M]), gripper=closure, check=False)
    kin.ignore = set()
    if lift is None:
        raise StepFailed("lift", "no IK solution for the vertical lift")
    ex.run(fine_actions(lift, LIFT_M), phase="lift")
    ex.settle(lift[-1])
    lifted = d.object_pose(s)[0][2] - z0
    ev["lift_m"] = float(lifted)
    ev['grasp_snapshots']['after_lift'] = release_snapshot(d, s, t)
    if not loaded_grasp(s) or lifted < 0.5 * LIFT_M:
        raise StepFailed("lift", f"subject not carried (rose {lifted:.3f} m)", ev)

    def measured_release() -> tuple[Pose, Pose, Pose]:
        """Release EE poses from the *measured* in-hand transform."""
        sync(k, d)
        ee = Pose.from_pq(*d.ee_pose())
        obj = Pose.from_pq(*d.object_pose(s))
        t_ee_obj = ee.inv() @ obj
        ee_rel = plan.goal.obj_pose @ t_ee_obj.inv()
        t_hi = k.aabb(t)[1]
        s_lo = k.aabb(s)[0]
        bottom_rel = s_lo[2] - obj.p[2] + plan.goal.obj_pose.p[2]
        rise = max(LIFT_M, t_hi[2] + 0.04 - bottom_rel)
        if plan.release_drop:
            rise = max(rise, plan.overhead_rise)
        return ee_rel, ee_rel.moved([0, 0, rise]), t_ee_obj

    # 5. transfer to the pre-release pose (subject attached for collision checks)
    ee_rel, ee_prerel, t_ee_obj = measured_release()
    # solve the release, then walk straight up to the pre-release (same as planning)
    q_prerel, why = None, "ik"
    kin.ignore = {s}
    for seed in (plan.q_release, d.arm_q(), plan.q_grasp):
        if seed is None:
            continue
        if plan.release_drop:
            q_suspended = kin.ik(
                ee_prerel, seed=seed, restarts=4,
                accept_solution=lambda q: overhead_release_free(kin, s, t_ee_obj, q, closure))
            if q_suspended is not None and overhead_release_free(kin, s, t_ee_obj, q_suspended, closure):
                q_prerel = q_suspended
                break
            why = "overhead_collision_or_ik"
            continue
        q_rel = kin.ik(ee_rel, seed=seed, restarts=4)
        if q_rel is None:
            continue
        up = kin.cartesian_path(q_rel, ee_rel, ee_prerel, gripper=closure, check=False)
        if up is None:
            why = "lift_line_ik"
            continue
        kin.ignore = set()
        kin.attached = (s, t_ee_obj)
        hit = kin.in_collision(up[-1], closure)
        kin.attached = None
        kin.ignore = {s}
        if hit is None:
            q_prerel = up[-1]
            break
        why = f"collision:{hit}"
    kin.ignore = set()
    if q_prerel is None:
        ev["prerelease_failure"] = why
        raise StepFailed("transfer", f"pre-release pose unreachable with the measured grasp ({why})", ev)
    kin.attached = (s, t_ee_obj)
    path = kin.plan(d.arm_q(), q_prerel, gripper=closure)
    kin.attached = None
    if path is None:
        raise StepFailed("transfer", "no collision-free carry path", ev)
    ex.run(stabilized_transport_actions(free_actions(path)), phase="transfer")
    ex.settle(q_prerel)
    if not loaded_grasp(s):
        raise StepFailed("transfer", "subject dropped during the carry", ev)
    # 6. re-measure the in-hand pose and descend with a contact guard
    ee_rel, ee_prerel, t_ee_obj = measured_release()
    ev["in_hand_tilt_deg_before_descent"] = float(math.degrees(math.acos(np.clip(
        (np.trace(t_ee_obj.R.T @ (plan.grasp.tcp_in_obj.inv()).R) - 1) / 2, -1, 1))))
    ev["release_snapshots"] = {"before_descent": release_snapshot(d, s, t)}
    if plan.release_drop:
        ex.run([d.arm_q()] * HOLD_FRAMES, gripper=0.0, phase="release_overhead")
        descent = []
        length = 0.0
        ev["descent_contact_stop"] = False
        ev["support_at_release"] = predicates.support_below(d, s)
    else:
        sync(k, d)
        kin.ignore = {s, t}
        descent = kin.cartesian_path(d.arm_q(), Pose.from_pq(*d.ee_pose()), ee_rel, gripper=closure,
                                     check=False)
        kin.ignore = set()
        if descent is None:
            raise StepFailed("descend", "no IK along the measured descent line", ev)
        length = float(np.linalg.norm(ee_prerel.p - ee_rel.p))
        d.phase = "lower"
    stopped_early = False
    for q in fine_actions(descent, length):
        d.act([*q.tolist(), closure])
        sup = predicates.support_below(d, s)
        if sup is not None and sup != "robot":
            stopped_early = True
            break
    ev["descent_contact_stop"] = stopped_early
    ev["support_at_release"] = predicates.support_below(d, s)
    ev["release_snapshots"]["before_open"] = release_snapshot(d, s, t)
    # 7. release once supported, then wait until the subject is at rest
    hold_q = d.arm_q()
    ex.run([hold_q] * HOLD_FRAMES, gripper=0.0, phase="release")
    for _ in range(30):
        if predicates.at_rest(d, s):
            break
        d.act([*hold_q.tolist(), 0.0])
    ev["release_snapshots"]["after_open"] = release_snapshot(d, s, t)
    # 8. retreat back along the approach axis, then up
    sync(k, d)
    ee = Pose.from_pq(*d.ee_pose())
    back = ee.moved(-ee.R[:, 2] * 0.08)
    up = back.moved([0, 0, 0.08])
    kin.ignore = {s}
    r1 = kin.cartesian_path(d.arm_q(), ee, back, check=False)
    r2 = kin.cartesian_path(r1[-1], back, up, check=False) if r1 else None
    kin.ignore = set()
    if r1 and r2:
        ex.run(fine_actions(r1 + r2[1:], 0.16), gripper=0.0, phase="retreat")
    for _ in range(int(CONTROL_HZ)):
        d.act([*ex.target.tolist(), 0.0], record=True)
    ev["release_snapshots"]["after_retreat"] = release_snapshot(d, s, t)
    return ev


def verify_tool_goal(d: World, step: Step, ev: dict[str, Any], *, initial=None) -> tuple[bool, dict[str, Any]]:
    """Require the declared tool result, preserving transient contact evidence.

    A contact goal is an event: ``run_strike`` records its predicate at the
    first guarded contact, before the tool is put down. Endpoint goals are
    evaluated on the post-action world and receive the rollout initial state
    for pose_delta.
    """
    if step.goal.kind == "contact" and "contact_predicate" in ev:
        pev = ev["contact_predicate"]
        ok = bool(ev.get("contact_goal_passed", False))
    else:
        ok, pev = predicates.evaluate_step(d, step, initial=initial)
    return ok, {**ev, "predicate": pev}


def run_steps(d: World, k: World, o_factory, steps: tuple[Step, ...], *, max_plans: int = 6,
              snapshot=None) -> list[StepReport]:
    """Execute every step; on an execution failure restore and try the next plan."""
    kin = Kinematics(k)
    reports: list[StepReport] = []
    initial = {oid: d.object_pose(oid) for oid in d.bodies}
    for i, step in enumerate(steps):
        d.step_index = i
        if step.primitive == "push":
            from .push import run_push

            sync(k, d)
            o = o_factory(d)
            try:
                ev = run_push(d, k, kin, o, step)
                ok, pev = predicates.evaluate_step(d, step, initial=initial)
                reports.append(StepReport(i, step.primitive, ok, "done" if ok else "verify",
                                          "" if ok else "goal predicate false after pushing",
                                          {**ev, "predicate": pev}))
            except StepFailed as exc:
                reports.append(StepReport(i, step.primitive, False, exc.stage, exc.detail,
                                          exc.evidence))
            finally:
                o.close()
            if not reports[-1].passed:
                break
            continue
        if step.primitive == "articulate":
            from .articulate import run_articulate

            sync(k, d)
            state = d.p.saveState(physicsClientId=d.cid)
            n_frames = len(d.frames)
            try:
                ev = run_articulate(d, k, kin, step)
                ok, pev = predicates.evaluate(d, step.goal)
                reports.append(StepReport(i, step.primitive, ok, "done" if ok else "verify",
                                          "" if ok else "joint goal not reached",
                                          {**ev, "predicate": pev}))
            except StepFailed as exc:
                d.p.restoreState(state, physicsClientId=d.cid)
                del d.frames[n_frames:]
                reports.append(StepReport(i, step.primitive, False, exc.stage, exc.detail,
                                          exc.evidence))
            if not reports[-1].passed:
                break
            continue
        if step.primitive == "tool_contact":
            from .tool import run_strike

            sync(k, d)
            state = d.p.saveState(physicsClientId=d.cid)
            n_frames = len(d.frames)
            try:
                ev = run_strike(d, k, kin, step)
                ok, ev = verify_tool_goal(d, step, ev, initial=initial)
                if not ok:
                    d.p.restoreState(state, physicsClientId=d.cid)
                    del d.frames[n_frames:]
                reports.append(StepReport(i, step.primitive, ok, "done" if ok else "verify",
                                          "" if ok else "tool action did not achieve the declared goal",
                                          ev))
            except StepFailed as exc:
                d.p.restoreState(state, physicsClientId=d.cid)
                del d.frames[n_frames:]
                reports.append(StepReport(i, step.primitive, False, exc.stage, exc.detail,
                                          exc.evidence))
            if not reports[-1].passed:
                break
            continue
        if step.primitive not in {"transport", "reorient"}:
            reports.append(StepReport(i, step.primitive, False, "unsupported",
                                      f"primitive {step.primitive!r} not implemented yet"))
            break
        sync(k, d)
        o = o_factory(d)
        try:
            plans, info = plan_transport(k, kin, o, step, seed=i)
        except StepFailed as exc:
            reports.append(StepReport(i, step.primitive, False, exc.stage, exc.detail, exc.evidence))
            break
        finally:
            o.close()
        state = d.p.saveState(physicsClientId=d.cid)
        n_frames = len(d.frames)
        attempts = []
        report = None
        for j, pl in enumerate(plans[:max_plans]):
            try:
                ev = execute_transport(d, k, kin, step, pl)
            except StepFailed as exc:
                attempts.append({"plan": j, "stage": exc.stage, "detail": exc.detail,
                                 **exc.evidence})
                d.p.restoreState(state, physicsClientId=d.cid)
                del d.frames[n_frames:]
                continue
            ok, pev = predicates.evaluate_step(d, step, initial=initial)
            attempts.append({"plan": j, "stage": "verify", "passed": ok,
                             "predicate": pev, **ev})
            if ok:
                report = StepReport(i, step.primitive, True, "done", "",
                                    {"planning": info, "attempts": attempts, "predicate": pev})
                break
            d.p.restoreState(state, physicsClientId=d.cid)
            del d.frames[n_frames:]
        if report is None:
            report = StepReport(i, step.primitive, False, "execution",
                                "every feasible plan failed in execution",
                                {"planning": info, "attempts": attempts})
            reports.append(report)
            break
        reports.append(report)
    return reports
