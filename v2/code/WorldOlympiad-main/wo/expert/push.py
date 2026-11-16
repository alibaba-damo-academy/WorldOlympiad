"""Non-prehensile push primitive.

Goal positions come from the same physics-checked sampler as transport (the
subject resting inside the goal region).  Execution pushes with the closed
gripper along the line from the subject's *current* centre to the goal and
re-aims every segment from the measured subject position (closed loop).
"""

from __future__ import annotations

import math

import numpy as np

from .. import predicates
from ..spec import Step
from ..sim.world import World
from .goals import sample_goals
from .kin import Kinematics, Pose, tcp_to_ee
from .rollout import Executor, StepFailed, fine_actions, free_actions, sync

PUSH_SEGMENT_M = 0.04
PUSH_SPEED_FRACTION = 0.6
CONTACT_GAP_M = 0.025
FINGERTIP_CLEARANCE_M = 0.026
MAX_SEGMENTS = 30


def _top_down(yaw: float) -> np.ndarray:
    # closing axis horizontal, approach straight down
    x = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    a = np.array([0.0, 0.0, -1.0])
    return np.stack([x, np.cross(a, x), a], axis=1)


def _footprint_radius(w: World, oid: str, d: np.ndarray) -> float:
    lo, hi = w.aabb(oid)
    half = (hi[:2] - lo[:2]) / 2
    return float(abs(d[0]) * half[0] + abs(d[1]) * half[1])


def _contact_radius(w: World, oid: str, direction: np.ndarray,
                    centre: np.ndarray, height: float) -> float:
    """Measure the rear surface on the actual push line, not an AABB corner.

    A diagonal AABB support distance overestimates rounded bodies and can
    consume the entire final stroke before contact. If another body occludes
    the ray (or the line misses the subject), retain the conservative bound;
    never use another object's surface as the subject contact.
    """
    bound = _footprint_radius(w, oid, direction)
    origin = np.array([*centre, height], dtype=float)
    start = origin - direction * (bound + CONTACT_GAP_M)
    hit = w.p.rayTest(start.tolist(), origin.tolist(), physicsClientId=w.cid)[0]
    if hit[0] == w.bodies[oid]:
        distance = float(np.dot(origin - np.asarray(hit[3]), direction))
        if 0.0 < distance <= bound + CONTACT_GAP_M:
            return distance
    return bound


def _separate_pusher(d: World, kin: Kinematics, ex: Executor, subject: str,
                     diagnostics: list | None = None) -> bool:
    """Leave intentional subject contact before invoking a free-space planner.

    RRT requires a collision-free starting configuration. Only the pushed body
    is exempt during this checked vertical separation; all other obstacles
    remain active, and the exemption is restored even when IK fails.
    """
    ee = Pose.from_pq(*d.ee_pose())
    previous = set(kin.ignore)
    # Try a vertical lift first, then retreat directions around the local
    # clutter. Every candidate still checks all obstacles except the subject.
    deltas = ([0.0, 0.0, 0.20], [0.0, 0.0, 0.15],
              [0.10, 0.0, 0.06], [-0.10, 0.0, 0.06],
              [0.0, 0.10, 0.06], [0.0, -0.10, 0.06])
    for delta in deltas:
        try:
            kin.ignore = previous | {subject}
            detail = {'delta': list(delta)}
            kwargs = {'diagnostics': detail} if diagnostics is not None else {}
            path = kin.cartesian_path(d.arm_q(), ee, ee.moved(delta), gripper=1.0,
                                      **kwargs)
            if diagnostics is not None:
                diagnostics.append(detail)
        finally:
            kin.ignore = previous
        if path is None:
            continue
        ex.run(fine_actions(path, float(np.linalg.norm(delta))), gripper=1.0,
               phase="separate")
        sync(kin.w, d)
        return True
    return False


def live_goal_position(sampled_goal: Pose, sampled_target: Pose, live_target: Pose) -> np.ndarray:
    """Preserve a certified goal's target-local coordinates as a receptacle moves.

    A movable mat/tray may translate or rotate under contact. Steering to its
    stale world position cannot satisfy the live destination predicate.
    """
    return (live_target @ sampled_target.inv() @ sampled_goal).p


def push_segment_evidence(before, after, goal_xy, start: Pose, end: Pose,
                          radius: float, ee: Pose) -> dict:
    """Record commanded contact geometry and measured progress, not success."""
    return {'subject_before_xy': np.asarray(before).tolist(),
            'subject_after_xy': np.asarray(after).tolist(),
            'goal_xy': np.asarray(goal_xy).tolist(),
            'remaining_distance_m': float(np.linalg.norm(np.asarray(goal_xy) - after)),
            'displacement_m': float(np.linalg.norm(np.asarray(after) - before)),
            'footprint_radius_m': float(radius),
            'start_tcp': start.p.tolist(), 'end_tcp': end.p.tolist(),
            'actual_ee': ee.p.tolist(),
            'endpoint_tracking_error_m': float(np.linalg.norm(ee.p - tcp_to_ee(end).p))}


def _needs_push_entry(actual_ee: Pose, start_tcp: Pose) -> bool:
    """Reuse a contact entry only when its complete measured EE pose matches.

    Alternate wrist frames are certified by entry planning, not by proximity
    of their origins. A Cartesian stroke cannot start in an unrelated frame.
    """
    desired = tcp_to_ee(start_tcp)
    angle = math.acos(float(np.clip((np.trace(actual_ee.R.T @ desired.R) - 1) / 2,
                                   -1.0, 1.0)))
    return (float(np.linalg.norm(actual_ee.p - desired.p)) > 0.02
            or angle > 0.05)


def _checked_push_line(kin: Kinematics, q: np.ndarray, start: Pose, end: Pose,
                       subject: str, target: str, evidence: dict,
                       segment: int) -> list | None:
    """Check the stroke and retain the exact obstruction, restoring exemptions."""
    previous = set(kin.ignore)
    diagnostics = {}
    try:
        kin.ignore = previous | {subject, target}
        line = kin.cartesian_path(q, tcp_to_ee(start), tcp_to_ee(end),
                                  gripper=1.0, check=True,
                                  diagnostics=diagnostics)
    finally:
        kin.ignore = previous
    evidence.setdefault('stroke_checks', []).append(
        {'segment': segment, 'start_tcp': start.p.tolist(),
         'end_tcp': end.p.tolist(), **diagnostics})
    return line


def _plan_push_entry(kin, q, start, end, subject, target, evidence, segment):
    """Select a fully checked descent AND stroke before committing to entry.

    A closed gripper has two equivalent across-stroke closing axes. The flipped
    wrist frame can reach a line the first frame cannot, without changing the
    object-centre route, contact height, or intentional-contact exemptions.
    """
    previous = set(kin.ignore)
    # Preserve preferred across-stroke frames first. A non-prehensile closed
    # gripper need not keep its closing axis across the line: orthogonal
    # vertical frames can avoid a wrist singularity. They are proposals only;
    # the complete robot approach, descent and stroke must still be checked.
    # Vertical-only wrists can exhaust the arm's inner workspace even when
    # the object route is clear. Propose bounded tilted closed-pusher frames
    # only after all vertical frames fail. TCP positions are unchanged; real
    # fingers/table and full robot clearance must still pass every check.
    frames = [(angle, 0.) for angle in (0., math.pi, math.pi / 2, -math.pi / 2)]
    frames += [(angle, math.radians(tilt)) for angle in (0., math.pi)
               for tilt in (-25., 25.)]
    # Keep the established first-branch path first; only exhaust alternative
    # IK branches after every contact frame has failed its complete entry.
    # A fixed high precontact pose can be outside the inner workspace while
    # the contact itself is reachable. Lower approaches are proposals, never
    # clearance exemptions: certify their full free path and descent as usual.
    for angle, tilt, branch_search, rise in [(a, t, b, h)
                                            for h in (.12, .08, .04)
                                            for b in (False, True)
                                            for a, t in frames]:
        c, sn = math.cos(angle), math.sin(angle)
        rotation = (start.R @ np.diag([-1., -1., 1.]) if angle == math.pi
                    else start.R @ np.array([[c, -sn, 0.], [sn, c, 0.], [0., 0., 1.]]))
        ct, st = math.cos(tilt), math.sin(tilt)
        rotation = rotation @ np.array([[1., 0., 0.], [0., ct, -st], [0., st, ct]])
        flip = angle == math.pi
        a, b = Pose(start.p.copy(), rotation), Pose(end.p.copy(), rotation)
        above = a.moved([0, 0, rise])
        try:
            kin.ignore = previous
            planned = {}
            attempt = {'segment': segment, 'wrist_rotation_rad': angle,
                       'wrist_tilt_rad': tilt, 'approach_rise_m': rise,
                       'alternate_ik_branch_search': branch_search, 'reason': 'above_ik'}
            evidence.setdefault('entry_checks', []).append(attempt)

            def accept_entry(qa):
                # IK endpoint convergence is not an entry certificate. Search
                # another bounded branch when its elbow or descent is blocked.
                kin.ignore = previous
                collision = kin.in_collision(qa, 1.0)
                if collision is not None:
                    attempt.update(reason='above_collision', collision=str(collision))
                    return False
                attempt['reason'] = 'approach_path'
                path = kin.plan(q, qa, gripper=1.0)
                if path is None:
                    return False
                attempt['reason'] = 'descent'
                try:
                    kin.ignore = previous | {subject}
                    down = kin.cartesian_path(qa, tcp_to_ee(above), tcp_to_ee(a),
                                              gripper=1.0, check=True)
                finally:
                    kin.ignore = previous
                if down is None:
                    return False
                attempt['reason'] = 'stroke'
                line = _checked_push_line(kin, down[-1], a, b, subject, target,
                                          evidence, segment)
                if line is None:
                    return False
                planned.update(path=path, down=down)
                attempt['reason'] = 'success'
                return True

            if branch_search:
                qa = kin.ik(tcp_to_ee(above), seed=q, accept_solution=accept_entry)
                accepted = qa is not None
            else:
                qa = kin.ik(tcp_to_ee(above), seed=q)
                accepted = qa is not None and accept_entry(qa)
            if accepted:
                path, down = planned['path'], planned['down']
                evidence.setdefault('entry_frames', []).append(
                    {'segment': segment, 'flipped': flip,
                     **({'approach_rise_m': rise} if rise != .12 else {}),
                     **({'alternate_ik_branch_search': True} if branch_search else {}),
                     **({'wrist_rotation_rad': angle} if angle not in (0., math.pi) else {}),
                     **({'wrist_tilt_rad': tilt} if tilt else {})})
                return a, b, qa, path, down
        finally:
            kin.ignore = previous
    return None


def _visible_segment(a, b, boxes) -> bool:
    """Reject segments crossing the interior of any inflated obstacle."""
    for lo, hi in boxes:
        enter, leave = 0.0, 1.0
        for axis in range(2):
            v = b[axis] - a[axis]
            if abs(v) < 1e-12:
                if not lo[axis] < a[axis] < hi[axis]:
                    enter, leave = 1.0, 0.0
                    break
            else:
                bounds = sorted(((lo[axis] - a[axis]) / v,
                                 (hi[axis] - a[axis]) / v))
                enter, leave = max(enter, bounds[0]), min(leave, bounds[1])
        if enter < leave - 1e-10:
            return False
    return True


def feedback_push(ex, world, subject, actions, before, direction, boxes):
    """Execute a certified stroke until measured lateral drift needs re-aiming.

    The visibility route is not a physics controller. Especially with tilted
    contact, a correct TCP line can move the subject sideways. Reserve half
    the measured starting obstacle clearance for that error, and replan from
    the live pose rather than finishing an obsolete open-loop stroke.
    """
    before = np.asarray(before)
    direction = np.asarray(direction)[:2]
    distances = [float(np.linalg.norm(np.maximum(np.maximum(lo - before,
                                                           before - hi), 0)))
                 for lo, hi in boxes]
    budget = min([CONTACT_GAP_M / 4] + [distance / 2 for distance in distances])
    executed = 0
    drift = 0.0
    for q in actions:
        ex.run([q], phase='push')
        executed += 1
        position = np.asarray(world.p.getBasePositionAndOrientation(
            world.bodies[subject], physicsClientId=world.cid)[0][:2])
        displacement = position - before
        drift = float(np.linalg.norm(displacement - direction *
                                     np.dot(displacement, direction)))
        if drift >= budget:
            break
    return {'executed_actions': executed, 'lateral_drift_m': drift,
            'drift_budget_m': budget, 'reaim': executed < len(actions)}


def push_route(start, goal, boxes, support_bounds=None):
    """Shortest visibility route around measured inflated XY AABBs.

    This is only an advisory object-centre route, NOT a collision certificate:
    all approach, stroke and retreat robot checks still execute unchanged.
    """
    import heapq
    start, goal = np.asarray(start), np.asarray(goal)
    def supported(point):
        return (support_bounds is None or
                bool(np.all(point >= support_bounds[0]) and
                     np.all(point <= support_bounds[1])))
    if not supported(start) or not supported(goal):
        return None
    nodes = [start, goal]
    for lo, hi in boxes:
        # Exact boundary corners leave zero allowance for physical lateral
        # drift during contact. Put advisory waypoints outside the unchanged
        # forbidden envelope, using the existing contact acquisition clearance.
        # This adds clearance; visibility and full robot checks still reject
        # the original measured boxes, including blocked starts/goals.
        outer_lo, outer_hi = lo - CONTACT_GAP_M, hi + CONTACT_GAP_M
        nodes.extend(point for x in (outer_lo[0], outer_hi[0])
                     for y in (outer_lo[1], outer_hi[1])
                     if supported(point := np.array([x, y])))
    queue = [(0.0, 0, [0])]
    best = {0: 0.0}
    while queue:
        cost, i, path = heapq.heappop(queue)
        if cost > best[i]:
            continue
        if i == 1:
            return [nodes[j] for j in path]
        for j, node in enumerate(nodes):
            if j == i or not _visible_segment(nodes[i], node, boxes):
                continue
            new = cost + float(np.linalg.norm(node - nodes[i]))
            if new < best.get(j, float('inf')):
                best[j] = new
                heapq.heappush(queue, (new, j, path + [j]))
    return None


def select_push_route(centre, goal_positions, boxes, support_bounds=None):
    """Keep the ranked goal when routable, otherwise try certified alternatives.

    Goal sampling certifies placement, not transit through clutter. No blocked
    region is removed: every candidate uses the same measured obstacles.
    """
    for index, goal in enumerate(goal_positions):
        route = push_route(centre, np.asarray(goal)[:2], boxes, support_bounds)
        if route is not None:
            return index, route
    return None, None


def _push_support_bounds(w, subject):
    """Keep the complete yaw-independent subject footprint on its support.

    The existing contact gap supplies a reserve for contact-induced drift.
    This is a conservative rectangular support envelope, not permission to
    skip any robot collision checks or the final physical task predicate.
    """
    lo, hi = w.aabb(subject)
    reserve = float(np.linalg.norm((hi[:2] - lo[:2]) / 2)) + CONTACT_GAP_M
    low, high = w.aabb(w.support_id)
    return low[:2] + reserve, high[:2] - reserve


def _push_obstacles(w, subject, target, height):
    lo, hi = w.aabb(subject)
    # A circular bound covers every subject yaw plus rear contact acquisition.
    clearance = float(np.linalg.norm((hi[:2] - lo[:2]) / 2)) + CONTACT_GAP_M
    boxes = []
    for oid in w.bodies:
        if oid in (subject, target):
            continue
        low, high = w.aabb(oid)
        if low[2] <= height <= high[2]:
            boxes.append((low[:2] - clearance, high[:2] + clearance))
    return boxes


def run_push(d: World, k: World, kin: Kinematics, o: World, step: Step) -> dict:
    s, t = step.subject, step.goal.target or step.target
    d.active_subject, d.active_target = s, t
    goals, stats = sample_goals(o, step, Pose.from_pq(*o.object_pose(s)), n_xy=16)
    if not goals:
        raise StepFailed("planning", "no stable subject pose inside the push goal region",
                         {"goal_sampling": stats})
    ex = Executor(d, k, kin)
    sampled_goal = goals[0].obj_pose
    sampled_target = Pose.from_pq(*o.object_pose(t)) if t in o.bodies else None
    lo, hi = d.aabb(s)
    # fingertips reach ~2 cm past the TCP: keep them clear of the surface below
    push_z = lo[2] + max(FINGERTIP_CLEARANCE_M, min(0.04, 0.5 * (hi[2] - lo[2])))
    ev = {"goal_sampling": stats, "segments": 0}
    initial = {oid: d.object_pose(oid) for oid in d.bodies}
    for seg in range(MAX_SEGMENTS):
        ok, _ = predicates.evaluate(d, step.goal, initial=initial)
        if ok:
            break
        sync(k, d)
        # push through the centre of mass (an AABB centre can lie outside an L-shaped part)
        centre = np.asarray(d.p.getBasePositionAndOrientation(d.bodies[s],
                                                             physicsClientId=d.cid)[0][:2])
        before = centre.copy()
        goal_xy = (live_goal_position(sampled_goal, sampled_target,
                                     Pose.from_pq(*d.object_pose(t)))[:2]
                   if sampled_target is not None else sampled_goal.p[:2])
        boxes = _push_obstacles(d, s, t, push_z)
        support_bounds = _push_support_bounds(d, s)
        ev['support_bounds_xy'] = [p.tolist() for p in support_bounds]
        route = push_route(centre, goal_xy, boxes, support_bounds)
        if route is None:
            # A stable placement may lie inside an inflated distractor's transit
            # envelope. Search the already predicate-certified goals, not a
            # relaxed collision envelope or a newly invented destination.
            positions = [(live_goal_position(g.obj_pose, sampled_target,
                          Pose.from_pq(*d.object_pose(t)))
                          if sampled_target is not None else g.obj_pose.p)
                         for g in goals]
            index, route = select_push_route(centre, positions, boxes, support_bounds)
            ev.setdefault('route_goal_search', []).append(
                {'segment': seg, 'candidate_count': len(positions),
                 'selected': index, 'start': centre.tolist(),
                 'goals': [p.tolist() for p in positions],
                 'inflated_obstacles': [[lo.tolist(), hi.tolist()] for lo, hi in boxes]})
            if route is None:
                raise StepFailed('push', 'no obstacle-free object-centre route', ev)
            sampled_goal = goals[index].obj_pose
            goal_xy = np.asarray(positions[index])[:2]
        waypoint = route[1]
        ev.setdefault('routes', []).append(
            {'segment': seg, 'waypoints': [p.tolist() for p in route],
             'inflated_obstacles': [[lo.tolist(), hi.tolist()] for lo, hi in boxes]})
        delta = waypoint - centre
        dist = float(np.linalg.norm(delta))
        if dist < 0.005:
            break
        u = np.append(delta / dist, 0.0)
        r = _contact_radius(d, s, u, centre, push_z)
        R = _top_down(math.atan2(u[1], u[0]) + math.pi / 2)  # fingers span across the push line
        start = Pose(np.array([*(centre - u[:2] * (r + CONTACT_GAP_M)), push_z]), R)
        end = start.moved(u * min(PUSH_SEGMENT_M + CONTACT_GAP_M, dist + CONTACT_GAP_M))
        above = start.moved([0, 0, 0.12])
        line = None
        needs_entry = seg == 0 or _needs_push_entry(Pose.from_pq(*d.ee_pose()), start)
        if not needs_entry:
            line = _checked_push_line(kin, d.arm_q(), start, end, s, t, ev, seg)
            needs_entry = line is None
            if needs_entry:
                ev.setdefault('reuse_stroke_reentry', []).append(seg)
        if needs_entry:
            # A completed segment intentionally ends in contact. Separate first:
            # an RRT from that contact configuration cannot leave its root.
            separation = []
            if seg > 0:
                separated = _separate_pusher(d, kin, ex, s, separation)
                ev.setdefault('separation_attempts', []).append(
                    {'segment': seg, 'attempts': separation})
                if not separated:
                    raise StepFailed("push", "cannot separate pusher from the subject", ev)
            # Do not commit to an entry wrist frame whose upcoming stroke is
            # unreachable. Both equivalent closing-axis frames retain every
            # original collision and continuity check.
            entry = _plan_push_entry(kin, d.arm_q(), start, end, s, t, ev, seg)
            if entry is None:
                raise StepFailed("push", "no checked entry and stroke frame", ev)
            start, end, q_above, path, down = entry
            ex.run(free_actions(path), gripper=1.0, phase="approach")
            ex.settle(q_above)
            ex.run(fine_actions(down, 0.12), phase="descend")
        if line is None:
            line = _checked_push_line(kin, d.arm_q(), start, end, s, t, ev, seg)
        if line is None:
            raise StepFailed("push", "checked push line rejected", ev)
        actions = fine_actions(line, float(np.linalg.norm(end.p - start.p)) / PUSH_SPEED_FRACTION)
        feedback = feedback_push(ex, d, s, actions, before, u, boxes)
        ev.setdefault('stroke_feedback', []).append({'segment': seg, **feedback})
        ev["segments"] = seg + 1
        after = np.asarray(d.p.getBasePositionAndOrientation(d.bodies[s],
                                                            physicsClientId=d.cid)[0][:2])
        ev.setdefault('segment_progress', []).append(
            push_segment_evidence(before, after, goal_xy, start, end, r,
                                  Pose.from_pq(*d.ee_pose())))
        if float(np.linalg.norm(after - before)) < 0.004:
            ev["stalls"] = ev.get("stalls", 0) + 1
            if ev["stalls"] >= 3:
                raise StepFailed("push", "pusher is not moving the subject", ev)
    # back off upward
    sync(k, d)
    ee = Pose.from_pq(*d.ee_pose())
    kin.ignore = {s, t}  # leave intentional push contact; other bodies stay checked
    try:
        up = kin.cartesian_path(d.arm_q(), ee, ee.moved([0, 0, 0.12]), gripper=1.0)
    finally:
        kin.ignore = set()
    if up:
        ex.run(fine_actions(up, 0.12), phase="retreat")
    for _ in range(10):
        d.act([*ex.target.tolist(), 1.0])
    return ev
