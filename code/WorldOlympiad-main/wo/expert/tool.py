"""Tool-contact primitive (strike / tap with a grasped tool).

The tool is grasped far from its centre of mass (the handle end of a hammer,
spatula, stick...).  Its working end is the slab of the tool farthest from the
grasp along the tool's long axis.  The working end is brought above the target's
top centre and lowered quickly with a contact guard; the step succeeds when the
tool touched the target with non-trivial force.  The tool is then set back down.
"""

from __future__ import annotations

import numpy as np

from ..spec import Step
from .. import predicates
from ..sim.world import World
from .grasps import sample_grasps
from .kin import Kinematics, Pose, tcp_to_ee
from .rollout import (LIFT_M, PREGRASP_BACKOFF, Executor, StepFailed, fine_actions,
                      free_actions, sync)

STRIKE_HEIGHT_M = 0.10
STRIKE_SPEED = 0.35  # m/s
MIN_STRIKE_FORCE_N = 1.0


def local_aabb(w: World, oid: str) -> tuple[np.ndarray, np.ndarray]:
    body = w.bodies[oid]
    pose = w.object_pose(oid)
    w.set_link_pose(body, (0, 0, 5.0), (0, 0, 0, 1))
    lo, hi = w.aabb(oid)
    w.set_link_pose(body, *pose)
    return lo - np.array([0, 0, 5.0]), hi - np.array([0, 0, 5.0])


def run_strike(d: World, k: World, kin: Kinematics, step: Step) -> dict:
    tool, target = step.subject, step.goal.target or step.target
    d.active_subject, d.active_target = tool, target
    sync(k, d)
    lo_l, hi_l = local_aabb(k, tool)
    axis = int(np.argmax(hi_l - lo_l))  # long axis in the tool frame
    obj = Pose.from_pq(*k.object_pose(tool))
    grasps = sample_grasps(k, tool)
    if not grasps:
        raise StepFailed("planning", "no grasp on the tool")
    com_l = obj.inv() @ Pose(np.asarray(k.p.getBasePositionAndOrientation(
        k.bodies[tool], physicsClientId=k.cid)[0]), np.eye(3))

    def handle_score(g):
        return abs(g.tcp_in_obj.p[axis] - com_l.p[axis]) + 0.2 * g.score

    grasps.sort(key=lambda g: -handle_score(g))
    ev = {"grasps": len(grasps), "long_axis": axis}
    t_lo, t_hi = k.aabb(target)
    target_top = np.array([(t_lo[0] + t_hi[0]) / 2, (t_lo[1] + t_hi[1]) / 2, t_hi[2]])
    for g in grasps[:16]:
        # working end: the slab at the far end of the long axis from the grasp
        far_hi = g.tcp_in_obj.p[axis] < (lo_l[axis] + hi_l[axis]) / 2
        end_l = (lo_l + hi_l) / 2
        end_l[axis] = hi_l[axis] - 0.03 if far_hi else lo_l[axis] + 0.03
        end_l[2] = lo_l[2]  # strike with the underside of the working end
        tcp = g.world(obj)
        ee_grasp = tcp_to_ee(tcp)
        ee_pre = ee_grasp.moved(-tcp.R[:, 2] * PREGRASP_BACKOFF)
        q_pre = kin.ik(ee_pre, seed=k.arm_q())
        if q_pre is None or kin.in_collision(q_pre) is not None:
            continue
        kin.ignore = {tool}
        approach = kin.cartesian_path(q_pre, ee_pre, ee_grasp)
        kin.ignore = set()
        if approach is None:
            continue
        # strike pose: tool keeps its grasp-time orientation, working end above target top
        end_world0 = obj @ Pose(end_l, np.eye(3))
        shift = target_top + np.array([0, 0, STRIKE_HEIGHT_M]) - end_world0.p
        ee_strike_hi = ee_grasp.moved(shift)
        q_hi = kin.ik(ee_strike_hi, seed=approach[-1], restarts=6)
        if q_hi is None:
            continue
        down = kin.cartesian_path(q_hi, ee_strike_hi, ee_strike_hi.moved([0, 0, -STRIKE_HEIGHT_M - 0.02]),
                                  check=False)
        if down is None:
            continue
        ev.update(grasp_kind=g.kind, grasp_width_m=g.width)
        break
    else:
        raise StepFailed("planning", "no handle grasp reaches a strike pose over the target", ev)

    ex = Executor(d, k, kin)
    path = kin.plan(d.arm_q(), q_pre)
    if path is None:
        raise StepFailed("reach", "no path to the tool pre-grasp", ev)
    ex.run(free_actions(path), gripper=0.0, phase="approach")
    ex.settle(q_pre)
    ex.run(fine_actions(approach, PREGRASP_BACKOFF), phase="descend")
    ex.settle(approach[-1])
    ex.run([approach[-1]] * 12, gripper=1.0, phase="grasp")
    if not d.grasped(tool):
        raise StepFailed("grasp", "fingers do not hold the tool", ev)
    sync(k, d)
    ee0 = Pose.from_pq(*d.ee_pose())
    lift = kin.cartesian_path(d.arm_q(), ee0, ee0.moved([0, 0, LIFT_M]), check=False)
    if lift is None:
        raise StepFailed("lift", "no IK for lifting the tool", ev)
    ex.run(fine_actions(lift, LIFT_M), phase="lift")
    kin.attached = (tool, ee0.inv() @ Pose.from_pq(*d.object_pose(tool)))
    path = kin.plan(d.arm_q(), q_hi, gripper=1.0)
    kin.attached = None
    if path is None:
        raise StepFailed("transfer", "no path to the strike pose", ev)
    ex.run(free_actions(path), phase="transfer")
    ex.settle(q_hi)
    # strike: fast descent, stop at first tool-target contact
    d.phase = "strike"
    peak = 0.0
    contact_checked = False
    for q in fine_actions(down, (STRIKE_HEIGHT_M + 0.02) * 0.08 / STRIKE_SPEED):
        d.act([*q.tolist(), 1.0])
        force = sum(cp[9] for cp in d.contacts(tool, target))
        peak = max(peak, force)
        if force > MIN_STRIKE_FORCE_N:
            # Contact is transient. Capture the declared contact predicate now,
            # before recoil and putting the tool back down change the state.
            # Only contact is a transient event. Endpoint goals are certified
            # after the action by run_steps, where the rollout initial state is
            # available; evaluating them here can raise without ``initial``.
            if step.goal.kind == "contact":
                ok, pev = predicates.evaluate(d, step.goal)
                ev.update(contact_goal_passed=bool(ok), contact_predicate=pev)
            contact_checked = True
            break
    if not contact_checked:
        ev.update(contact_goal_passed=False,
                  contact_predicate={"kind": step.goal.kind, "contact_event": False})
    ev["peak_contact_force_n"] = float(peak)
    ex.run(fine_actions(list(reversed(down)), STRIKE_HEIGHT_M), phase="recoil")
    d.step_contact_peak = peak
    # put the tool back down where it was picked up
    sync(k, d)
    ee = Pose.from_pq(*d.ee_pose())
    back = kin.plan(d.arm_q(), lift[-1], gripper=1.0)
    if back is not None:
        ex.run(free_actions(back), phase="return")
        down2 = kin.cartesian_path(d.arm_q(), Pose.from_pq(*d.ee_pose()), ee0.moved([0, 0, 0.005]),
                                   check=False)
        if down2:
            ex.run(fine_actions(down2, LIFT_M), phase="lower")
    ex.run([d.arm_q()] * 10, gripper=0.0, phase="release")
    sync(k, d)
    ee = Pose.from_pq(*d.ee_pose())
    up = kin.cartesian_path(d.arm_q(), ee, ee.moved(-ee.R[:, 2] * 0.10), check=False)
    if up:
        ex.run(fine_actions(up, 0.10), phase="retreat")
    if peak <= MIN_STRIKE_FORCE_N:
        raise StepFailed("strike", f"tool never pressed on the target (peak {peak:.2f} N)", ev)
    return ev
