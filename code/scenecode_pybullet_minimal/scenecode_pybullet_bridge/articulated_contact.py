"""Contact-space planning for typed articulated handles.

Planning resets are confined to a saved pre-execution state. Every recorded
action uses robot motors; the furniture joint remains passive. Handle position,
normal and longitudinal axis belong to the immutable articulation binding.

A single IK solve cannot satisfy reachability, continuity and collision
clearance at once: a redundant arm has several discrete configuration branches
at one certified handle pose, and the branch that follows the whole handle arc
is not necessarily the one a seed lands on.  Planning therefore proceeds in two
stages.  It first enumerates genuine IK branches at the closed handle pose with
deterministic low-discrepancy joint seeds, then continues each branch through
the actuation arc with a continuity-regularised solve and keeps the first branch
that reaches the open limit with bounded pose error, joint jump and forbidden
penetration.  Both valid tool rolls of a symmetric handle pinch and reusable
workcell placements are searched; no case name, dimension or recipe is used.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# Deterministic low-discrepancy seeds.  The primes decorrelate the joint axes so
# a small number of samples still covers every configuration branch.
_PRIMES = (2, 3, 5, 7, 11, 13, 17)


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    while index > 0:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def _joint_seeds(lower: np.ndarray, upper: np.ndarray, count: int) -> list[np.ndarray]:
    span = np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)
    seeds = []
    for index in range(1, count + 1):
        unit = np.array(
            [_van_der_corput(index, _PRIMES[dimension]) for dimension in range(len(span))],
            dtype=np.float64,
        )
        seeds.append(np.asarray(lower, dtype=np.float64) + unit * span)
    return seeds


def run_contact_trajectory(scene):
    s, p = scene, scene.p
    binding = s.inputs.binding
    counts = dict(s.inputs.action_counts)
    initial_q = binding.open_position if s.inputs.operator == "close" else binding.closed_position
    target_q = binding.closed_position if s.inputs.operator == "close" else binding.open_position
    s._set_joint_state(initial_q)
    s.planning_evidence = {"planner": "articulated_contact_space_v2", "waypoints": []}
    lower = np.array([p.getJointInfo(s.panda, j)[8] for j in s.arm_joints])
    upper = np.array([p.getJointInfo(s.panda, j)[9] for j in s.arm_joints])
    # Registered Robotiq pad depth plus the finger-body standoff, distinct from
    # the finger-link origins.  Closing uses the safe standoff; the actuation
    # arc may sit deeper so the pads keep their normal load while the pull's
    # prying component tries to lift them off the handle.
    gripper_offset = float(s.inputs.gripper_contact_offset_m)
    tcp = np.array([0., 0., gripper_offset + s.inputs.grasp_standoff_m])
    actuation_offset = (s.inputs.grasp_standoff_m if s.inputs.actuation_standoff_m is None
                        else s.inputs.actuation_standoff_m)
    tcp_actuation = np.array([0., 0., gripper_offset + actuation_offset])
    position_tolerance = .005
    orientation_tolerance = .05
    penetration_tolerance = .003
    # A plan that only *touches* the fixed host is not collision-free: Bullet
    # keeps the shoulder at its torque ceiling while the arm is wedged against
    # the furniture and the tracked motion silently stops.  Require a positive
    # forbidden gap at every planned configuration.
    required_clearance = .004
    required_path_clearance = .008
    actuation_clearance_floor = (required_clearance if actuation_offset >= s.inputs.grasp_standoff_m - 1e-9
                                 else -penetration_tolerance + 5e-4)
    clearance_probe = .02
    continuity_limit = .25
    branch_position_tolerance = 5e-4
    branch_orientation_tolerance = 5e-3
    approach_distance = float(s.inputs.approach_clearance_m)
    # A handle pinch is symmetric: rolling the tool half a turn about the
    # approach axis swaps the fingers without changing which handle surface is
    # grasped.  Both rolls are physically valid and select different arm
    # branches, so both are planned.
    tool_rolls = (0, 1)

    def contact_pose(position, roll):
        s._set_joint_state(position)
        xyz, quat = s._child_link_pose()
        child = Rotation.from_quat(quat)
        contact = np.asarray(xyz) + child.apply(binding.contact_target_local_xyz)
        normal = child.apply(binding.contact_normal_local_xyz)
        normal /= np.linalg.norm(normal)
        longitudinal = child.apply(binding.handle_axis_local_xyz)
        longitudinal -= normal * np.dot(normal, longitudinal)
        longitudinal /= np.linalg.norm(longitudinal)
        tool_z = -normal
        tool_x = np.cross(longitudinal, tool_z)
        tool = Rotation.from_matrix(np.column_stack((tool_x, longitudinal, tool_z)))
        if roll:
            tool = tool * Rotation.from_euler("z", math.pi)
        return contact, normal, tool

    def end_effector_pose():
        state = p.getLinkState(s.panda, s.end_effector, computeForwardKinematics=True)
        rotation = Rotation.from_quat(state[1]) * Rotation.from_quat(state[3]).inv()
        return np.asarray(state[0]) - rotation.apply(state[2]), rotation

    def solve(point, rotation, seed, weight, max_nfev):
        seed = np.asarray(seed, dtype=np.float64)

        def residual(q):
            for joint, value in zip(s.arm_joints, q):
                p.resetJointState(s.panda, joint, float(value))
            xyz, actual_rotation = end_effector_pose()
            # The continuity term keeps a redundant solution on the branch that
            # contains the seed instead of jumping to a distant one.
            return np.r_[xyz - point, (rotation.inv() * actual_rotation).as_rotvec() * .25,
                         (q - seed) * weight]

        def jacobian(q):
            # Bullet FK contains float32 pose values. Relative finite steps
            # shrink to machine epsilon near zero-valued joints and can report
            # a false zero derivative. Use a fixed angular perturbation instead.
            columns = []
            for index in range(len(q)):
                before, after = q.copy(), q.copy()
                before[index] = max(lower[index] + 1e-6, q[index] - 1e-4)
                after[index] = min(upper[index] - 1e-6, q[index] + 1e-4)
                columns.append((residual(after) - residual(before)) / (after[index] - before[index]))
            return np.column_stack(columns)

        result = least_squares(
            residual, np.clip(seed, lower + 1e-5, upper - 1e-5),
            bounds=(lower + 1e-6, upper - 1e-6), max_nfev=max_nfev, jac=jacobian,
        )
        value = residual(result.x)
        return (
            result.x.copy(),
            float(np.linalg.norm(value[:3])),
            float(np.linalg.norm(value[3:6]) / .25),
        )

    def planning_clearance(configuration):
        """Smallest gap between the robot and the host outside the grasp pair.

        Negative values mean penetration.  Measured with the imported contact
        margin disabled, so a real clearance is a real clearance.
        """

        for joint, value in zip(s.arm_joints, configuration):
            p.resetJointState(s.panda, joint, float(value))
        p.performCollisionDetection()
        points = p.getClosestPoints(s.panda, s.host, clearance_probe)
        forbidden = [point for point in points
                     if not (point[3] in s.finger_contact_links and point[4] in s.contact_target_links)]
        if not forbidden:
            return clearance_probe
        return min(float(point[8]) for point in forbidden)

    def interpolated_clearance(start, target, samples=8):
        clearance = clearance_probe
        for index in range(1, samples + 1):
            alpha = index / samples
            smooth = alpha * alpha * (3. - 2. * alpha)
            clearance = min(clearance, planning_clearance(start + (target - start) * smooth))
        return clearance

    def place_robot(position, quaternion):
        dynamics = p.getDynamicsInfo(s.panda, -1)
        world_com = p.multiplyTransforms(position.tolist(), quaternion.tolist(),
                                         dynamics[3], dynamics[4])
        p.resetBasePositionAndOrientation(s.panda, *world_com)

    def host_footprint():
        bounds = [p.getAABB(s.host, index) for index in range(-1, p.getNumJoints(s.host))
                  if p.getCollisionShapeData(s.host, index)]
        lower_host = np.min([bound[0] for bound in bounds], axis=0)
        upper_host = np.max([bound[1] for bound in bounds], axis=0)
        return lower_host, upper_host

    def clear_of_host(xy, footprint, margin):
        lower_host, upper_host = footprint
        x0, x1 = lower_host[0] - margin, upper_host[0] + margin
        y0, y1 = lower_host[1] - margin, upper_host[1] + margin
        x, y = float(xy[0]), float(xy[1])
        if not (x0 < x < x1 and y0 < y < y1):
            return np.asarray(xy, dtype=np.float64)
        edges = ((x0, y), (x1, y), (x, y0), (x, y1))
        nearest = min(edges, key=lambda edge: (edge[0] - x) ** 2 + (edge[1] - y) ** 2)
        return np.array(nearest, dtype=np.float64)

    def joint_limit_margin(configuration):
        return float(np.min(np.minimum(configuration - lower, upper - configuration)))

    saved = p.saveState()
    initial_arm = s._arm_position()
    initial_target, initial_normal, _ = contact_pose(initial_q, 0)
    final_target, _, _ = contact_pose(target_q, 0)
    direction = final_target - initial_target
    direction /= np.linalg.norm(direction)

    def placement_candidates():
        base_position, base_quaternion = s._robot_base_link_pose()
        base_position = np.asarray(base_position, dtype=np.float64)
        base_quaternion = np.asarray(base_quaternion, dtype=np.float64)
        candidates = [(base_position.copy(), base_quaternion.copy(), roll) for roll in tool_rolls]
        if not s.inputs.automatic_robot_placement:
            return candidates
        # Sample the registered native workcell radius around the mean of the
        # handle arc.  The arc centre is measured from the certified binding, so
        # the candidates follow any host placement, orientation or opening
        # direction.  The mid-arc outward normal is the natural first side;
        # sweeping the full circle keeps the policy general when the panel
        # sweeps toward that side.
        arc = [
            contact_pose(initial_q + (target_q - initial_q) * fraction, 0)[0]
            for fraction in np.linspace(0., 1., 5)
        ]
        center = np.mean(arc, axis=0)
        _, mid_normal, _ = contact_pose(initial_q + (target_q - initial_q) * .5, 0)
        normal_xy = np.asarray(mid_normal[:2], dtype=np.float64)
        if float(np.linalg.norm(normal_xy)) <= 1e-6:
            return candidates
        base_angle = float(np.arctan2(normal_xy[1], normal_xy[0]))
        radius = float(np.linalg.norm(base_position[:2] - np.asarray(s.inputs.host_center_xy)))
        radius = min(max(radius, .45), .575)
        # Keep the robot base outside the host volume.  The host is a fixed
        # support, so a base sampled inside its footprint is not a placement at
        # all; the candidate is projected to the nearest footprint edge with a
        # standoff for the robot's own base width.
        footprint = host_footprint()
        for offset_degrees in (0, -45, 45, -90, 90, -135, 135, 180):
            angle = base_angle + math.radians(offset_degrees)
            outward = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
            for scale in (1., 1.2, .8):
                xy = center[:2] + outward * radius * scale
                xy = clear_of_host(xy, footprint, .05)
                yaw = float(np.arctan2(center[1] - xy[1], center[0] - xy[0]))
                quaternion = Rotation.from_euler("z", yaw).as_quat()
                for roll in tool_rolls:
                    candidates.append((np.r_[xy, base_position[2]], np.asarray(quaternion, dtype=np.float64), roll))
        return candidates

    def contact_branches(roll):
        contact, _, rotation = contact_pose(initial_q, roll)
        target = contact - rotation.apply(tcp)
        found: list[np.ndarray] = []
        for seed in (initial_arm, *_joint_seeds(lower, upper, 32)):
            solution, position_error, angle_error = solve(
                target, rotation, seed, .002, 150)
            if position_error > branch_position_tolerance or angle_error > branch_orientation_tolerance:
                continue
            if any(float(np.max(np.abs(solution - other))) < .25 for other in found):
                continue
            found.append(solution)
        # Prefer the branch with the widest joint-limit margin; ties go to the
        # configuration closest to the certified reset pose.
        found.sort(key=lambda q: (-joint_limit_margin(q), float(np.linalg.norm(q - initial_arm))))
        return found

    def plan_branch(contact_solution, roll):
        """Continue one IK branch from the closed handle to the open limit."""

        records: list[dict] = []
        contact, _, contact_rotation = contact_pose(initial_q, roll)
        contact_target = contact - contact_rotation.apply(tcp)
        contact_error, contact_angle = _pose_error(contact_target, contact_rotation, contact_solution)
        contact_clearance = planning_clearance(contact_solution)
        records.append(_record("contact", initial_q, contact, contact_solution, contact_solution,
                               contact_error, contact_angle, contact_clearance))
        if (contact_error > position_tolerance or contact_angle > orientation_tolerance
                or contact_clearance < required_clearance):
            return None, records
        # Approach keeps the selected branch: the arm travels to a nearby pose
        # further out along the certified contact normal.
        approach_contact, approach_normal, approach_rotation = contact_pose(initial_q, roll)
        approach_target = (approach_contact - approach_rotation.apply(tcp)
                           + approach_normal * approach_distance)
        approach_solution, approach_error, approach_angle = solve(
            approach_target, approach_rotation, contact_solution, .01, 200)
        approach_clearance = planning_clearance(approach_solution)
        approach_jump = float(np.max(np.abs(approach_solution - contact_solution)))
        records.append(_record("approach", initial_q, approach_contact, approach_solution, contact_solution,
                               approach_error, approach_angle, approach_clearance, approach_jump))
        if (approach_error > position_tolerance or approach_angle > orientation_tolerance
                or approach_clearance < required_clearance):
            return None, records
        # The free move from the certified reset pose to the approach pose is
        # executed by position motors; a branch whose joint-space interpolation
        # grazes the furniture would wedge the arm.
        for start, target_configuration, samples in (
            (initial_arm, approach_solution, 16),
            (approach_solution, contact_solution, 8),
        ):
            if interpolated_clearance(start, target_configuration, samples) < required_path_clearance:
                return None, records
        seed = contact_solution
        arc_solutions: list[np.ndarray] = []
        for index in range(counts["actuate"]):
            q = initial_q + (target_q - initial_q) * (index + 1) / counts["actuate"]
            point, normal, rotation = contact_pose(q, roll)
            target = point - rotation.apply(tcp_actuation)
            solution, position_error, angle_error = solve(target, rotation, seed, .01, 200)
            if position_error > 2e-3 or angle_error > .02:
                # Polish the pose accuracy without leaving the followed branch:
                # a fresh low-weight solve starts from the regularised result.
                polished, polished_position, polished_angle = solve(
                    target, rotation, solution, .001, 200)
                if (polished_position < position_error
                        and float(np.max(np.abs(polished - seed))) <= continuity_limit):
                    solution, position_error, angle_error = polished, polished_position, polished_angle
            clearance = planning_clearance(solution)
            jump = float(np.max(np.abs(solution - seed)))
            records.append(_record("actuate", q, point, solution, seed,
                                   position_error, angle_error, clearance, jump))
            if (position_error > position_tolerance or angle_error > orientation_tolerance
                    or clearance < actuation_clearance_floor or jump > continuity_limit):
                return None, records
            arc_solutions.append(solution)
            seed = solution
        # Retreat repeats the approach clearance at the open limit, still on the
        # followed branch.
        retreat_contact, retreat_normal, retreat_rotation = contact_pose(target_q, roll)
        retreat_target = (retreat_contact - retreat_rotation.apply(tcp)
                          + retreat_normal * approach_distance)
        retreat_solution, retreat_error, retreat_angle = solve(
            retreat_target, retreat_rotation, seed, .01, 200)
        retreat_clearance = planning_clearance(retreat_solution)
        records.append(_record("retreat", target_q, retreat_contact, retreat_solution, seed,
                               retreat_error, retreat_angle, retreat_clearance))
        if (retreat_error > position_tolerance or retreat_angle > orientation_tolerance
                or retreat_clearance < required_clearance):
            return None, records
        return {
            "contact_solution": contact_solution,
            "approach_solution": approach_solution,
            "arc_solutions": arc_solutions,
            "retreat_solution": retreat_solution,
        }, records

    def _pose_error(target, rotation, configuration):
        for joint, value in zip(s.arm_joints, configuration):
            p.resetJointState(s.panda, joint, float(value))
        xyz, actual_rotation = end_effector_pose()
        return (float(np.linalg.norm(xyz - target)),
                float(np.linalg.norm((rotation.inv() * actual_rotation).as_rotvec())))

    def _record(phase, joint_target, contact, solution, seed, position_error, angle_error,
                clearance, jump=None):
        return {
            "phase": phase,
            "joint_target": joint_target,
            "contact_world": contact.tolist(),
            "position_error_m": position_error,
            "orientation_error_rad": angle_error,
            "maximum_joint_delta_rad": (float(np.max(np.abs(solution - seed)))
                                        if jump is None else jump),
            "arm_joint_positions_rad": np.asarray(solution, dtype=np.float64).tolist(),
            "forbidden_clearance_m": clearance,
            "maximum_forbidden_penetration_m": max(0., -float(clearance)),
            "joint_limit_margin_rad": joint_limit_margin(np.asarray(solution, dtype=np.float64)),
        }

    placement_attempts = []
    selected_placement = None
    selected_plan = None
    try:
        for position, quaternion, roll in placement_candidates():
            p.restoreState(saved)
            place_robot(position, quaternion)
            s._set_joint_state(initial_q)
            attempt = {
                "base_position": np.asarray(position, dtype=np.float64).tolist(),
                "base_quaternion_xyzw": np.asarray(quaternion, dtype=np.float64).tolist(),
                "tool_roll_half_turn": roll,
                "passed": False,
                "initial_clearance_m": planning_clearance(initial_arm),
                "branches": [],
            }
            placement_attempts.append(attempt)
            if attempt["initial_clearance_m"] < required_clearance:
                continue
            branches = contact_branches(roll)
            attempt["branch_count"] = len(branches)
            for branch in branches[:16]:
                plan, records = plan_branch(branch, roll)
                attempt["branches"].append({
                    "joint_positions_rad": branch.tolist(),
                    "joint_limit_margin_rad": joint_limit_margin(branch),
                    "passed": plan is not None,
                    "failure": None if plan is not None else "arc_continuity_or_reachability",
                    "reached_phase": records[-1]["phase"] if records else None,
                    "reached_joint_target": records[-1]["joint_target"] if records else None,
                    "last_position_error_m": records[-1]["position_error_m"] if records else None,
                    "last_orientation_error_rad": records[-1]["orientation_error_rad"] if records else None,
                    "last_joint_delta_rad": records[-1]["maximum_joint_delta_rad"] if records else None,
                    "last_forbidden_clearance_m": records[-1]["forbidden_clearance_m"] if records else None,
                })
                if plan is not None:
                    attempt["passed"] = True
                    attempt["waypoints"] = records
                    selected_placement = (position, quaternion, roll)
                    selected_plan = plan
                    s.planning_evidence["waypoints"] = records
                    break
            if selected_placement is not None:
                break
    finally:
        p.restoreState(saved)
        p.removeState(saved)
    s.planning_evidence.update(
        passed=selected_placement is not None,
        placement_policy="frozen_host_native_workcell_reach" if s.inputs.automatic_robot_placement else "explicit_robot_pose",
        grasp_standoff_m=s.inputs.grasp_standoff_m,
        placement_attempts=placement_attempts,
    )
    if selected_placement is None:
        s.planning_evidence["failure"] = "reachability_or_ik_continuity"
    else:
        place_robot(selected_placement[0], selected_placement[1])
        s.panda_base = tuple(selected_placement[0])
        s.planning_evidence["selected_tool_roll_half_turn"] = selected_placement[2]
        # Planning resets are not recorded actions. Conditioning begins from
        # the selected, now-frozen actual robot root pose.
        s.frames.clear()
        s.contact_frame_count = 0
        s._record("conditioning", None)
    s._set_joint_state(initial_q)
    p.setJointMotorControl2(s.host, s.articulation_joint, p.VELOCITY_CONTROL, force=0.)
    s._travel_direction_world = direction
    s.contact_execution_evidence = {
        "stable_grasp": False, "actuation_frames": 0, "dual_contact_actuation_frames": 0,
        "maximum_arm_tracking_error_rad": 0., "forbidden_collision_frames": 0,
        "maximum_forbidden_penetration_m": 0.,
    }
    evidence = s.contact_execution_evidence
    if selected_plan is None:
        return s._finish(False, initial_q, direction, initial_target, target_q=target_q)

    sequence = [
        ("approach", selected_plan["approach_solution"], counts["approach"], 0.),
        ("contact", selected_plan["contact_solution"], counts["contact"], 0.),
        ("actuate", None, counts["actuate"], 1.),
        ("retreat", selected_plan["retreat_solution"], counts["retreat"], 0.),
    ]
    # Reachable arm configurations along the certified handle arc.  Actuation
    # commands the interpolated configuration for a bounded lead over the
    # measured articulation position instead of the open-loop reference, so the
    # pads cannot be walked off the handle by an accumulating tracking gap.
    actuate_table = [(initial_q, selected_plan["contact_solution"])]
    for index, solution in enumerate(selected_plan["arc_solutions"], start=1):
        actuate_table.append(
            (initial_q + (target_q - initial_q) * index / counts["actuate"], solution))

    def arc_solution_at(position):
        position = float(np.clip(position, min(initial_q, target_q), max(initial_q, target_q)))
        for index in range(1, len(actuate_table)):
            lower_q, lower_solution = actuate_table[index - 1]
            upper_q, upper_solution = actuate_table[index]
            if (lower_q <= position <= upper_q) or (upper_q <= position <= lower_q):
                span = upper_q - lower_q
                alpha = 0. if abs(span) < 1e-12 else float(np.clip((position - lower_q) / span, 0., 1.))
                return lower_solution + (upper_solution - lower_solution) * alpha
        return actuate_table[-1][1]

    def dual_contact():
        return len({c["robot_link_index"] for c in s._contact_evidence()
                    if c["normal_force_n"] > 0.}) == 2

    def record_tracking(command):
        evidence["maximum_arm_tracking_error_rad"] = max(
            evidence["maximum_arm_tracking_error_rad"],
            float(np.max(np.abs(s._arm_position() - command))),
        )
        forbidden = [c for c in p.getContactPoints(s.panda, s.host)
                     if not (c[3] in s.finger_contact_links and c[4] in s.contact_target_links)]
        penetration = max([-float(c[8]) for c in forbidden], default=0.)
        if penetration > .003:
            evidence["forbidden_collision_frames"] += 1
        evidence["maximum_forbidden_penetration_m"] = max(
            evidence["maximum_forbidden_penetration_m"], penetration)

    def move(command, count, phase, grip):
        start = s._arm_position()
        for i in range(count):
            alpha = (i + 1) / count
            smooth = alpha * alpha * (3. - 2. * alpha)
            action = start + (command - start) * smooth
            s._execute_action(action, grip, phase)
            record_tracking(action)

    for phase, command, count, grip in sequence:
        if phase == "contact":
            move(command, count, phase, 0.)
            stable = []
            for i in range(counts["contact"]):
                s._execute_action(command, (i + 1) / counts["contact"], "contact")
                record_tracking(command)
                stable.append(dual_contact())
            evidence["stable_grasp"] = all(stable[-3:])
            if not evidence["stable_grasp"]:
                evidence["failure"] = "no_stable_two_finger_grasp"
                return s._finish(False, initial_q, direction, initial_target, target_q=target_q)
        elif phase == "actuate":
            span = target_q - initial_q
            lead_fraction = s.inputs.grasp_lead_rad / abs(span)
            hold_tail = max(0, int(s.inputs.actuation_hold_frames))
            previous_command = s._arm_position()
            for step_index in range(1, count + 1):
                measured = (s._joint_position() - initial_q) / span
                progress = min(step_index / count, measured + lead_fraction)
                command = arc_solution_at(initial_q + progress * span)
                if hold_tail and step_index > count - hold_tail:
                    command = previous_command
                # Feed the planned joint velocity forward so the contact servo
                # tracks the moving handle instead of lagging behind it; the
                # bounded lead above still limits how far it may run ahead.
                feedforward = (
                    (command - previous_command) * s.control_hz
                    * float(s.inputs.arm_velocity_feedforward)
                )
                s._execute_action(command, grip, phase, arm_velocity=feedforward)
                previous_command = command
                record_tracking(command)
                evidence["actuation_frames"] += 1
                evidence["dual_contact_actuation_frames"] += int(dual_contact())
                if not dual_contact():
                    evidence["failure"] = "grasp_lost_during_actuation"
                    return s._finish(False, initial_q, direction, initial_target, target_q=target_q)
        elif phase == "retreat":
            settle_velocity = float(s.inputs.release_settle_velocity_rad_s)
            if settle_velocity > 0.:
                # A handle released while it is still rebounding keeps that
                # velocity and coasts away from the reached opening.  Hold the
                # grasp, still commanding the reached actuation pose, until the
                # articulation is at rest, then let go.
                settle_command = s._arm_position()
                for _ in range(counts["retreat"]):
                    if abs(float(p.getJointState(s.host, s.articulation_joint)[1])) < settle_velocity:
                        break
                    s._execute_action(settle_command, 1.0, "release")
                    record_tracking(settle_command)
            hold = s._arm_position()
            for i in range(counts["contact"]):
                alpha = (i + 1) / counts["contact"]
                # Unload the narrow handle gradually before the jaws clear it.
                jaw_open = alpha * alpha * alpha
                s._execute_action(hold, 1. - jaw_open, "release")
                record_tracking(hold)
            move(command, count, phase, 0.)
        else:
            move(command, count, phase, grip)

    for _ in range(counts["final_settle"]):
        s._execute_action(s._arm_position(), 0., "final_settle")
    span = abs(binding.open_position - binding.closed_position)
    success = bool(
        abs(s._joint_position() - target_q) <= (1. - s.inputs.open_fraction_required + 1e-9) * span
        and evidence["stable_grasp"]
        and evidence["actuation_frames"] == counts["actuate"]
        and evidence["dual_contact_actuation_frames"] == counts["actuate"]
        and evidence["forbidden_collision_frames"] == 0
        and evidence["maximum_arm_tracking_error_rad"] <= .10
        and s._attachment_constraint_count() == 0
        and not s._contact_evidence()
    )
    evidence["released"] = not bool(s._contact_evidence())
    evidence["passed"] = success
    return s._finish(success, initial_q, direction, initial_target, target_q=target_q)
