"""Kinematic queries on a ``World(mode="kinematic")``: IK, FK, collision checks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..sim.world import DROID_HOME, World

# TCP (centre between the finger pads) in the Robotiq base_link frame:
# +z is the approach direction, +x the closing axis (measured from the URDF).
TCP_OFFSET = np.array([0.0, 0.0, 0.150])
MAX_OPENING_M = 0.080
COLLISION_MARGIN_M = 0.004


def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(r: np.ndarray) -> np.ndarray:
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


@dataclass
class Pose:
    p: np.ndarray  # position (3,)
    R: np.ndarray  # rotation (3,3)

    @staticmethod
    def from_pq(p, q) -> "Pose":
        return Pose(np.asarray(p, float), quat_to_mat(q))

    @property
    def q(self) -> np.ndarray:
        return mat_to_quat(self.R)

    def __matmul__(self, other: "Pose") -> "Pose":
        return Pose(self.p + self.R @ other.p, self.R @ other.R)

    def inv(self) -> "Pose":
        return Pose(-self.R.T @ self.p, self.R.T)

    def moved(self, delta_world: np.ndarray) -> "Pose":
        return Pose(self.p + np.asarray(delta_world, float), self.R)


def aperture_tcp_offset(w: World, fraction: float) -> np.ndarray:
    """Calibrate the open TCP by measured pad translation at this aperture.

    Finger-link frames translate with the coupled knuckles. Their symmetric
    mean cancels jaw opening; express it in the EE frame (not world axes).
    Preserve the existing open-pad contact calibration and restore joint state.
    """
    if fraction == 0.0:
        return TCP_OFFSET.copy()
    previous = w.gripper_fraction()
    def centre(f):
        w.set_gripper_state(f)
        ee = Pose.from_pq(*w.ee_pose())
        positions = [w.p.getLinkState(w.robot, link,
                     computeForwardKinematics=True, physicsClientId=w.cid)[4]
                     for link in w.fingers]
        return ee.R.T @ (np.mean(positions, axis=0) - ee.p)
    try:
        baseline = centre(0.0)
        return TCP_OFFSET + centre(fraction) - baseline
    finally:
        w.set_gripper_state(previous)


def tcp_to_ee(tcp: Pose, offset: np.ndarray = TCP_OFFSET) -> Pose:
    return Pose(tcp.p - tcp.R @ offset, tcp.R)


def ee_to_tcp(ee: Pose) -> Pose:
    return Pose(ee.p + ee.R @ TCP_OFFSET, ee.R)


class Kinematics:
    def __init__(self, world: World) -> None:
        if world.mode != "kinematic":
            raise ValueError("Kinematics needs a kinematic world")
        self.w = world
        self.p = world.p
        self.lower = np.asarray(world.lower)
        self.upper = np.asarray(world.upper)
        self.home = np.asarray(DROID_HOME)
        self.attached: tuple[str, Pose] | None = None  # (object_id, T_ee_obj)
        self.ignore: set[str] = set()

    # --------------------------------------------------------------- FK / IK
    def fk(self, q) -> Pose:
        self.w.set_arm(q)
        return Pose.from_pq(*self.w.ee_pose())

    def ik(self, ee: Pose, seed=None, *, tol_pos=0.004, tol_rot_deg=3.0,
           restarts: int = 4, collision_free: bool = False, accept_solution=None) -> np.ndarray | None:
        p, w = self.p, self.w
        n_mov = len(w.movable)
        seeds = [np.asarray(seed if seed is not None else self.home)]
        rng = np.random.default_rng(0)
        for _ in range(restarts):
            seeds.append(rng.uniform(self.lower, self.upper) * 0.5 + self.home * 0.5)
        ranges = list(self.upper - self.lower) + [1.0] * (n_mov - 7)
        lows = list(self.lower) + [-1.0] * (n_mov - 7)
        highs = list(self.upper) + [1.0] * (n_mov - 7)
        for s in seeds:
            w.set_arm(s)
            q = s
            for _ in range(4):
                sol = p.calculateInverseKinematics(
                    w.robot, w.ee, ee.p.tolist(), ee.q.tolist(),
                    lowerLimits=lows, upperLimits=highs, jointRanges=ranges,
                    restPoses=list(s) + [0.0] * (n_mov - 7),
                    maxNumIterations=200, residualThreshold=1e-5,
                    physicsClientId=w.cid,
                )
                q = np.clip(np.asarray(sol[:7]), self.lower, self.upper)
                w.set_arm(q)
            got = Pose.from_pq(*w.ee_pose())
            dp = float(np.linalg.norm(got.p - ee.p))
            dr = math.degrees(math.acos(np.clip((np.trace(got.R.T @ ee.R) - 1) / 2, -1, 1)))
            if dp <= tol_pos and dr <= tol_rot_deg:
                # A converged IK branch need not be usable (e.g. elbow below
                # the support). Opt-in callers exhaust the same bounded seed
                # set rather than treating the first colliding branch as proof
                # that the target pose is unreachable. No collision exemptions.
                if collision_free and self.in_collision(q) is not None:
                    continue
                # Optional downstream feasibility query (e.g. an entire checked
                # approach), not merely endpoint clearance. Continue the same
                # bounded branch search if a converged solution cannot execute.
                if accept_solution is not None and not accept_solution(q):
                    continue
                return q
        return None

    # ------------------------------------------------------------- collision
    def _place_attached(self) -> None:
        if self.attached is None:
            return
        oid, t_ee_obj = self.attached
        ee = Pose.from_pq(*self.w.ee_pose())
        obj = ee @ t_ee_obj
        self.w.set_link_pose(self.w.bodies[oid], obj.p, obj.q)

    def in_collision(self, q, gripper: float = 0.0, margin: float = COLLISION_MARGIN_M) -> str | None:
        """Return the name of the first body in collision, else None."""
        w, p = self.w, self.p
        w.set_arm(q)
        w.set_gripper_state(gripper)
        self._place_attached()
        p.performCollisionDetection(physicsClientId=w.cid)
        attached = self.attached[0] if self.attached else None
        for oid, body in w.bodies.items():
            if oid in self.ignore or oid == attached:
                continue
            if oid == w.support_id:
                # the arm is mounted on the support: skip the base and first two links
                hit = any(p.getClosestPoints(w.robot, body, margin, linkIndexA=li,
                                             physicsClientId=w.cid)
                          for li in range(2, p.getNumJoints(w.robot, physicsClientId=w.cid)))
            else:
                hit = bool(p.getClosestPoints(w.robot, body, margin, physicsClientId=w.cid))
            if hit:
                return oid
            if attached is not None and p.getClosestPoints(
                w.bodies[attached], body, margin * 0.5, physicsClientId=w.cid
            ):
                return f"{attached}->{oid}"
        if p.getClosestPoints(w.robot, w.plane, margin, physicsClientId=w.cid):
            # only links far from the base can touch the floor
            return "floor"
        return None

    def path_free(self, qa, qb, gripper: float = 0.0, step_rad: float = 0.04) -> bool:
        qa, qb = np.asarray(qa), np.asarray(qb)
        n = max(2, int(np.ceil(np.max(np.abs(qb - qa)) / step_rad)) + 1)
        for t in np.linspace(0, 1, n):
            if self.in_collision(qa + t * (qb - qa), gripper) is not None:
                return False
        return True

    def cartesian_path(self, q0, ee_start: Pose, ee_end: Pose, step_m: float = 0.01,
                       gripper: float = 0.0, check: bool = True,
                       max_joint_jump: float = 0.25, diagnostics: dict | None = None) -> list[np.ndarray] | None:
        """IK along a straight EE line with slerped orientation."""
        from scipy.spatial.transform import Rotation, Slerp

        dist = float(np.linalg.norm(ee_end.p - ee_start.p))
        ang = math.acos(np.clip((np.trace(ee_start.R.T @ ee_end.R) - 1) / 2, -1, 1))
        n = max(2, int(np.ceil(max(dist / step_m, ang / 0.05))) + 1)
        slerp = Slerp([0, 1], Rotation.from_matrix([ee_start.R, ee_end.R]))
        qs = [np.asarray(q0)]
        for t in np.linspace(0, 1, n)[1:]:
            target = Pose(ee_start.p + t * (ee_end.p - ee_start.p), slerp([t]).as_matrix()[0])
            q = self.ik(target, seed=qs[-1], restarts=0)
            if q is None and check:
                # A local seed can fail near a singularity even when the next
                # small Cartesian step has a continuous solution. Exhaust the
                # usual bounded IK seeds, accepting ONLY a solution continuous
                # with this path and clear under the caller's original checks.
                # Do not recover collision rejects or permit a branch jump.
                def continuous_clear(candidate):
                    return (float(np.max(np.abs(candidate - qs[-1]))) <= max_joint_jump
                            and self.in_collision(candidate, gripper) is None)
                q = self.ik(target, seed=qs[-1], restarts=4,
                            accept_solution=continuous_clear)
                if q is not None and diagnostics is not None:
                    diagnostics['ik_recoveries'] = diagnostics.get('ik_recoveries', 0) + 1
            jump = None if q is None else float(np.max(np.abs(q - qs[-1])))
            hit = None
            reason = None
            if q is None:
                reason = 'ik'
            elif jump > max_joint_jump:
                reason = 'joint_jump'
            elif check:
                hit = self.in_collision(q, gripper)
                if hit is not None:
                    reason = 'collision'
            if reason is not None:
                if diagnostics is not None:
                    if reason == 'collision' and hit in self.w.bodies:
                        contacts = self.p.getClosestPoints(
                            self.w.robot, self.w.bodies[hit], COLLISION_MARGIN_M,
                            physicsClientId=self.w.cid)
                        diagnostics['robot_collision_links'] = sorted({
                            name for name, index in self.w.link_index.items()
                            if any(pt[3] == index for pt in contacts)})
                    diagnostics.update(reason=reason, progress=float(t),
                                       target_position=target.p.tolist(),
                                       joint_jump=jump, collision=hit)
                return None
            qs.append(q)
        if diagnostics is not None:
            diagnostics.update(reason='success', waypoints=len(qs))
        return qs

    # -------------------------------------------------------------- planning
    def plan(self, qa, qb, gripper: float = 0.0, *, iters: int = 3000,
             seed: int = 0) -> list[np.ndarray] | None:
        """Joint-space path: direct if free, else bidirectional RRT-Connect."""
        qa, qb = np.asarray(qa), np.asarray(qb)
        if self.path_free(qa, qb, gripper):
            return [qa, qb]
        for attempt in range(4):  # independent restarts: RRT failures are often unlucky trees
            path = self._rrt(qa, qb, gripper, iters, seed + 7919 * attempt)
            if path is not None:
                return path
        return None

    def _rrt(self, qa, qb, gripper, iters, seed):
        rng = np.random.default_rng(seed)
        ta: list[tuple[np.ndarray, int]] = [(qa, -1)]
        tb: list[tuple[np.ndarray, int]] = [(qb, -1)]
        step = 0.15

        def nearest(tree, q):
            d = [np.linalg.norm(n[0] - q) for n in tree]
            return int(np.argmin(d))

        def extend(tree, q):
            i = nearest(tree, q)
            near = tree[i][0]
            d = q - near
            dist = np.linalg.norm(d)
            new = q if dist <= step else near + d / dist * step
            if self.path_free(near, new, gripper):
                tree.append((new, i))
                return len(tree) - 1, dist <= step
            return None, False

        def connect(tree, q):
            while True:
                idx, reached = extend(tree, q)
                if idx is None:
                    return None
                if reached:
                    return idx

        def trace(tree, i):
            out = []
            while i >= 0:
                out.append(tree[i][0])
                i = tree[i][1]
            return out

        swapped = False  # True when ta is the goal-rooted tree
        for _ in range(iters):
            sample = rng.uniform(self.lower, self.upper)
            idx, _ = extend(ta, sample)
            if idx is not None:
                j = connect(tb, ta[idx][0])
                if j is not None:
                    from_a, from_b = trace(ta, idx), trace(tb, j)
                    start_branch, goal_branch = (from_b, from_a) if swapped else (from_a, from_b)
                    path = start_branch[::-1] + goal_branch
                    return self.shortcut(path, gripper, rng)
            ta, tb = tb, ta
            swapped = not swapped
        return None

    def shortcut(self, path, gripper, rng, rounds: int = 60):
        path = list(path)
        for _ in range(rounds):
            if len(path) <= 2:
                break
            i, j = sorted(rng.choice(len(path), 2, replace=False))
            if j - i < 2:
                continue
            if self.path_free(path[i], path[j], gripper):
                path = path[: i + 1] + path[j:]
        return path
