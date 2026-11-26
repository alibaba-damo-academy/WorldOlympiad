"""Articulate primitive: open / close a revolute or prismatic object joint.

The grasp is sampled on the moving link (and links rigidly attached to it, e.g.
a handle).  The hand then follows the joint's own kinematics: for joint values
q_i the grasp frame is T_world_link(q_i) * T_link_tcp, solved by sequential IK.
During execution the measured joint value re-indexes the path, so a lagging or
overshooting joint is tracked instead of fought.
"""

from __future__ import annotations

import numpy as np

from .. import predicates
from ..spec import Step
from ..sim.world import World
from .grasps import sample_grasps
from .kin import Kinematics, Pose, tcp_to_ee
from .rollout import Executor, StepFailed, fine_actions, free_actions, sync

TARGET_FRACTION = 0.85
PATH_STEP_RAD = 0.03
PATH_STEP_M = 0.01
CHUNK = 6


def rigid_links(w: World, body: int, root_link: int) -> set[int]:
    """root_link plus every link attached to it through fixed joints."""
    p, c = w.p, w.cid
    out = {root_link}
    changed = True
    while changed:
        changed = False
        for j in range(p.getNumJoints(body, physicsClientId=c)):
            info = p.getJointInfo(body, j, physicsClientId=c)
            if info[2] == p.JOINT_FIXED and info[16] in out and j not in out:
                out.add(j)
                changed = True
    return out


def link_pose(w: World, body: int, link: int) -> Pose:
    s = w.p.getLinkState(body, link, computeForwardKinematics=True, physicsClientId=w.cid)
    return Pose.from_pq(s[4], s[5])


def run_articulate(d: World, k: World, kin: Kinematics, step: Step) -> dict:
    comp = step.subject
    if comp not in k.articulation:
        raise StepFailed("planning", f"no movable joint bound to component {comp!r}")
    art = k.articulation[comp]
    body, joint = art["body"], art["joint"]
    host = art["host"]
    d.active_target = host
    links = rigid_links(k, body, joint)
    sync(k, d)
    k.set_joint_q(comp, d.joint_q(comp))
    q_now = k.joint_q(comp)
    closing = step.goal.kind == "joint_closed"
    q_goal_end = art["q_closed"] if closing else art["q_closed"] + TARGET_FRACTION * (
        art["q_open"] - art["q_closed"])
    step_size = PATH_STEP_RAD if k.p.getJointInfo(body, joint, physicsClientId=k.cid)[2] == \
        k.p.JOINT_REVOLUTE else PATH_STEP_M
    n = max(2, int(np.ceil(abs(q_goal_end - q_now) / step_size)) + 1)
    qs_joint = np.linspace(q_now, q_goal_end, n)
    t_link0 = link_pose(k, body, joint)
    # prefer features attached to the moving link (handles, knobs) over the panel itself
    grasps = sample_grasps(k, host, links=links, frame=t_link0, prefer_links=links - {joint})
    ev = {"grasps": len(grasps), "q_start": q_now, "q_target": float(q_goal_end), "rejections": {}}
    if not grasps:
        raise StepFailed("planning", "no antipodal grasp on the moving part", ev)

    best = None
    for g in grasps[:24]:
        # follow the joint arc with sequential IK
        kin.ignore = {host}
        arm_path, reached = [], q_now
        seed = k.arm_q()
        ok = True
        for qj in qs_joint:
            k.set_joint_q(comp, qj)
            ee = tcp_to_ee(link_pose(k, body, joint) @ g.tcp_in_obj)
            q = kin.ik(ee, seed=seed, restarts=0 if arm_path else 4)
            if q is None or (arm_path and np.max(np.abs(q - arm_path[-1])) > 0.3):
                break
            if kin.in_collision(q, 1.0) is not None:
                break
            arm_path.append(q)
            seed = q
            reached = qj
        k.set_joint_q(comp, q_now)
        kin.ignore = set()
        frac = (reached - art["q_closed"]) / (art["q_open"] - art["q_closed"])
        if not arm_path:
            ev["rejections"]["grasp_ik"] = ev["rejections"].get("grasp_ik", 0) + 1
            continue
        # the pre-grasp and the straight approach must be feasible too
        tcp0 = t_link0 @ g.tcp_in_obj
        ee_g = tcp_to_ee(tcp0)
        ee_p = ee_g.moved(-tcp0.R[:, 2] * 0.10)
        q_p = kin.ik(ee_p, seed=arm_path[0])
        if q_p is None or kin.in_collision(q_p) is not None:
            ev["rejections"]["pregrasp"] = ev["rejections"].get("pregrasp", 0) + 1
            continue
        kin.ignore = {host}
        appr = kin.cartesian_path(q_p, ee_p, ee_g)
        kin.ignore = set()
        if appr is None:
            ev["rejections"]["approach"] = ev["rejections"].get("approach", 0) + 1
            continue
        if best is None or abs(reached - q_now) > abs(best[2] - q_now):
            best = (g, arm_path, reached, frac)
        if (frac <= 0.05) if closing else (frac >= TARGET_FRACTION - 0.01):
            break
    if best is None:
        raise StepFailed("planning", "no reachable grasp on the moving part", ev)
    g, arm_path, reached, frac = best
    ev.update(planned_fraction=float(frac), grasp_kind=g.kind, grasp_width_m=g.width)
    need = predicates.THRESHOLDS["joint_open_fraction"]
    if not closing and frac < need:
        raise StepFailed("planning", f"reachable arc only opens to {frac:.2f}", ev)

    ex = Executor(d, k, kin)
    q_grasp = arm_path[0]
    t_grasp = tcp_to_ee(link_pose(k, body, joint) @ g.tcp_in_obj)
    a = (link_pose(k, body, joint) @ g.tcp_in_obj).R[:, 2]
    pre = t_grasp.moved(-a * 0.10)
    q_pre = kin.ik(pre, seed=q_grasp)
    path = kin.plan(d.arm_q(), q_pre)
    if path is None:
        raise StepFailed("reach", "no path to the pre-grasp", ev)
    ex.run(free_actions(path), gripper=0.0, phase="approach")
    ex.settle(q_pre)
    kin.ignore = {host}
    approach = kin.cartesian_path(q_pre, pre, t_grasp)
    kin.ignore = set()
    if approach is None:
        raise StepFailed("reach", "approach line to the handle blocked", ev)
    ex.run(fine_actions(approach, 0.10), phase="descend")
    ex.settle(approach[-1])
    ex.run([approach[-1]] * 12, gripper=1.0, phase="grasp")
    contacts = {f: [cp for cp in d.p.getContactPoints(bodyA=d.robot, bodyB=body, linkIndexA=f,
                                                      physicsClientId=d.cid)
                    if cp[4] in links] for f in d.fingers}
    if not all(contacts.values()):
        raise StepFailed("grasp", "fingers do not both hold the moving part", ev)
    # follow the arc, re-indexing by the measured joint value
    d.phase = "articulate"
    idx = 0
    stalls = 0
    while idx < len(arm_path) - 1:
        nxt = min(len(arm_path) - 1, idx + CHUNK)
        for q in fine_actions(arm_path[idx:nxt + 1], 0.05):
            d.act([*q.tolist(), 1.0])
        q_meas = d.joint_q(comp)
        new_idx = int(np.argmin(np.abs(qs_joint[: len(arm_path)] - q_meas)))
        if new_idx <= idx:
            stalls += 1
            if stalls > 3:
                break
            idx = nxt  # push on along the planned arc
        else:
            stalls = 0
            idx = max(new_idx, idx + 1)
    ev["final_fraction"] = d.articulation_fraction(comp)
    # release and back away along the approach axis
    hold = d.arm_q()
    ex.run([hold] * 10, gripper=0.0, phase="release")
    sync(k, d)
    ee = Pose.from_pq(*d.ee_pose())
    back = kin.cartesian_path(d.arm_q(), ee, ee.moved(-ee.R[:, 2] * 0.10), check=False)
    if back:
        ex.run(fine_actions(back, 0.10), phase="retreat")
    return ev
