"""PyBullet world for the reference expert.

One class serves three uses, selected by ``mode``:
  * ``"dynamic"``  - the reference rollout: gravity, DROID-matched motor gains,
                     15 Hz joint-position actions, renderer-compatible frames.
  * ``"kinematic"``- planning: same geometry, used for IK and collision queries.
  * ``"objects"``  - scratch drop tests: objects + support only, no robot.

The trajectory schema matches what scripts/render_scenecode_robotics_multiview.py
consumes with ``--scene-objects-manifest`` (``frame["objects"]``, ``frame["robot"]``).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..scene import SceneInputs, SceneObject, WORKSPACE
from .joint_state import measured_object_joints

BRIDGE = WORKSPACE / "code/scenecode_pybullet_minimal"
if str(BRIDGE) not in sys.path:
    sys.path.insert(0, str(BRIDGE))
from scenecode_pybullet_bridge.sdf import prepare_physics_sdf  # noqa: E402

PHYSICS_HZ = 240
CONTROL_HZ = 15
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ

DROID_HOME = (0.0, -math.pi / 5, 0.0, -4 * math.pi / 5, 0.0, 3 * math.pi / 5, 0.0)
DROID_ARM_FORCES = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
DROID_ARM_GAIN = 0.5
GRIPPER_CLOSED_RAD = math.pi / 4
GRIPPER_FORCE = 16.5
GRIPPER_MIMIC = {
    "finger_joint": 1.0,
    "right_outer_knuckle_joint": -1.0,
    "left_inner_finger_knuckle_joint": 1.0,
    "right_inner_finger_knuckle_joint": -1.0,
    "left_inner_finger_joint": -1.0,
    "right_inner_finger_joint": 1.0,
}
FINGER_LINKS = ("left_inner_finger", "right_inner_finger")
JOINT_FRICTION = 0.4  # N*m (or N) passive friction on object joints
EE_LINK = "base_link"


@dataclass
class PreparedBody:
    object_id: str
    physics_sdf: Path
    artifact: dict[str, Any]


def _physics_cache_valid(source: Path, ok: Path, cached: Path, source_hash: Path) -> bool:
    """Validate a prepared collision cache against the actual source bytes.

    The source path is not an identity: SceneCode may regenerate a generation in
    place.  Reusing only a path-matching marker can therefore load an older URDF
    (including a different link topology) for a newer SDF.
    """
    import hashlib
    if not (ok.is_file() and cached.is_file() and source_hash.is_file()):
        return False
    if ok.read_text() != str(source):
        return False
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return source_hash.read_text().strip() == digest


def prepare_bodies(scene: SceneInputs, work_dir: Path) -> dict[str, PreparedBody]:
    """SceneCode SDF -> PyBullet-loadable SDF (visuals stripped, links collapsed)."""

    out: dict[str, PreparedBody] = {}
    physics_dir = work_dir / "physics_sdf"
    for obj in [scene.support, *scene.objects.values()]:
        key = f"{obj.object_id}_s{obj.scale:.3f}"
        dest = physics_dir / key / obj.sdf_path.name
        ok = physics_dir / key / ".ok"
        cached = physics_dir / key / ".path"
        source_hash = physics_dir / key / ".source_sha256"
        if _physics_cache_valid(obj.sdf_path, ok, cached, source_hash):
            from .convex_collision import simplify_convex_collision
            cached_dest = Path(cached.read_text())
            simplified = simplify_convex_collision(obj.sdf_path, cached_dest)
            evidence = {"convex_collision_simplification": simplified} if simplified else {}
            from .prismatic_collision import preserve_prismatic_cavities
            partition = preserve_prismatic_cavities(obj.sdf_path, cached_dest)
            if partition:
                evidence['prismatic_collision_partition'] = partition
            from .concave_collision import preserve_concave_shell
            shell = preserve_concave_shell(obj.sdf_path, cached_dest)
            if shell:
                evidence['concave_collision_decomposition'] = shell
            out[obj.object_id] = PreparedBody(obj.object_id, cached_dest, evidence)
            continue
        from .sdf_urdf import link_count, sdf_to_urdf

        if link_count(obj.sdf_path) > 1:
            # multi-link (articulated) assets: PyBullet's SDF importer mis-places links
            # below a moving link, so load an equivalent URDF instead
            dest = sdf_to_urdf(obj.sdf_path, dest.with_suffix(".urdf"))
            artifact = None
        else:
            artifact = prepare_physics_sdf(obj.sdf_path, dest)
        ok.write_text(str(obj.sdf_path))
        (physics_dir / key / ".source_sha256").write_text(
            __import__('hashlib').sha256(obj.sdf_path.read_bytes()).hexdigest())
        (physics_dir / key / ".path").write_text(str(dest))
        from .convex_collision import simplify_convex_collision
        simplified = simplify_convex_collision(obj.sdf_path, dest)
        evidence = artifact.to_dict() if artifact else {"format": "urdf"}
        if simplified is not None:
            evidence["convex_collision_simplification"] = simplified
        from .prismatic_collision import preserve_prismatic_cavities
        partition = preserve_prismatic_cavities(obj.sdf_path, dest)
        if partition:
            evidence['prismatic_collision_partition'] = partition
        from .concave_collision import preserve_concave_shell
        shell = preserve_concave_shell(obj.sdf_path, dest)
        if shell:
            evidence['concave_collision_decomposition'] = shell
        out[obj.object_id] = PreparedBody(obj.object_id, dest, evidence)
    return out


class World:
    def __init__(
        self,
        scene: SceneInputs,
        prepared: dict[str, PreparedBody],
        *,
        mode: str = "dynamic",
        include: Iterable[str] | None = None,
    ) -> None:
        import pybullet as p
        import pybullet_data

        if mode not in {"dynamic", "kinematic", "objects"}:
            raise ValueError(mode)
        self.p = p
        self.mode = mode
        self.scene = scene
        self.cid = p.connect(p.DIRECT)
        self.frames: list[dict[str, Any]] = []
        self.active_subject: str | None = None
        self.active_target: str | None = None
        self.phase = "conditioning"
        self.step_index = 0
        self._build(prepared, pybullet_data.getDataPath(), include)

    # ------------------------------------------------------------------ build
    def _build(self, prepared, data_path, include) -> None:
        p, c = self.p, self.cid
        p.resetSimulation(physicsClientId=c)
        p.setAdditionalSearchPath(data_path, physicsClientId=c)
        p.setGravity(0, 0, -9.81 if self.mode != "kinematic" else 0.0, physicsClientId=c)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / PHYSICS_HZ,
            numSolverIterations=200,
            deterministicOverlappingPairs=1,
            physicsClientId=c,
        )
        self.plane = p.loadURDF("plane.urdf", physicsClientId=c)
        self.bodies: dict[str, int] = {}
        self.roles: dict[str, str] = {}
        wanted = None if include is None else set(include)
        for obj in [self.scene.support, *self.scene.objects.values()]:
            if wanted is not None and obj.object_id not in wanted and obj.role != "support":
                continue
            path = prepared[obj.object_id].physics_sdf
            if path.suffix == ".urdf":
                body = p.loadURDF(str(path), useFixedBase=obj.fixed,
                                  flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=c)
            else:
                body = p.loadSDF(str(path), physicsClientId=c)[0]
            self.bodies[obj.object_id] = body
            self.roles[obj.object_id] = obj.role
            self.set_link_pose(body, obj.position, obj.orientation_xyzw)
            p.changeDynamics(
                body, -1,
                lateralFriction=0.8 if obj.role == "support" else 1.0,
                spinningFriction=0.002, rollingFriction=0.0005,
                restitution=0.0, physicsClientId=c,
            )
            if any(token in obj.sdf_path.name.lower() for token in ("goal_marker", "target_marker")):
                # Region markers stay fixed and do not obstruct pushes.
                p.changeDynamics(body, -1, mass=0.0, physicsClientId=c)
                for link in range(-1, p.getNumJoints(body, physicsClientId=c)):
                    p.setCollisionFilterGroupMask(body, link, 0, 0, physicsClientId=c)
            if obj.role in {"manipuland", "tool"} and not obj.fixed:
                mass = float(p.getDynamicsInfo(body, -1, physicsClientId=c)[0])
                if mass > 20.0:
                    alo, ahi = p.getAABB(body, -1, physicsClientId=c)
                    volume = float(np.prod(np.maximum(np.asarray(ahi) - np.asarray(alo), 1e-6)))
                    p.changeDynamics(body, -1, mass=max(0.02, volume * 2000.0),
                                     physicsClientId=c)
            if obj.fixed or self.mode == "kinematic":
                p.changeDynamics(body, -1, mass=0.0, physicsClientId=c)
        self.support_id = self.scene.support.object_id
        self._bind_articulations()
        self.robot = None
        if self.mode != "objects":
            self._load_robot()

    def _bind_articulations(self) -> None:
        """Make every object joint passive and map task components to joints."""
        p, c = self.p, self.cid
        self.articulation: dict[str, dict[str, Any]] = {}
        movable_by_body: dict[int, list[tuple[int, str, str, float, float]]] = {}
        for body in self.bodies.values():
            for j in range(p.getNumJoints(body, physicsClientId=c)):
                info = p.getJointInfo(body, j, physicsClientId=c)
                if info[2] not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    continue
                if self.mode == "dynamic" or self.mode == "objects":
                    p.setJointMotorControl2(body, j, p.VELOCITY_CONTROL, targetVelocity=0,
                                            force=JOINT_FRICTION, physicsClientId=c)
                movable_by_body.setdefault(body, []).append(
                    (j, info[1].decode(), info[12].decode(), float(info[8]), float(info[9]),
                     "revolute" if info[2] == p.JOINT_REVOLUTE else "prismatic"))
        for comp, spec in self.scene.articulations.items():
            body = self.bodies.get(spec["host"])
            joints = movable_by_body.get(body, []) if body is not None else []
            if not joints:
                continue
            chosen = None
            if spec.get("joint_name"):
                chosen = next((jt for jt in joints if jt[1] == spec["joint_name"]), None)
            if chosen is None and len(joints) == 1:
                chosen = joints[0]
            if chosen is None:
                tokens = set(comp.lower().split("_"))
                chosen = max(joints, key=lambda jt: len(tokens & set(jt[2].lower().split("_"))))
            j, name, child, lo, hi, jtype = chosen
            q0 = p.getJointState(body, j, physicsClientId=c)[0]
            far = hi if abs(hi - q0) >= abs(lo - q0) else lo
            self.articulation[comp] = {"body": body, "joint": j, "joint_name": name,
                                       "child_link": child, "lower": lo, "upper": hi,
                                       "q_closed": q0, "q_open": far, "host": spec["host"],
                                       "joint_type": jtype}

    def joint_q(self, comp: str) -> float:
        a = self.articulation[comp]
        return float(self.p.getJointState(a["body"], a["joint"], physicsClientId=self.cid)[0])

    def set_joint_q(self, comp: str, q: float) -> None:
        a = self.articulation[comp]
        self.p.resetJointState(a["body"], a["joint"], q, physicsClientId=self.cid)

    def articulation_fraction(self, comp: str) -> float:
        a = self.articulation[comp]
        span = a["q_open"] - a["q_closed"]
        return float((self.joint_q(comp) - a["q_closed"]) / span) if abs(span) > 1e-9 else 0.0

    def _load_robot(self) -> None:
        p, c = self.p, self.cid
        base = self.scene.robot_base_position
        orn = p.getQuaternionFromEuler((0, 0, self.scene.robot_base_yaw))
        self.robot = p.loadURDF(
            str(self.scene.robot_urdf), basePosition=base, baseOrientation=orn,
            useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=c,
        )
        self.joint_index: dict[str, int] = {}
        self.link_index: dict[str, int] = {}
        self.movable: list[int] = []
        self.movable_names: list[str] = []
        self.arm: list[int] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        for j in range(p.getNumJoints(self.robot, physicsClientId=c)):
            info = p.getJointInfo(self.robot, j, physicsClientId=c)
            name, ltype, link = info[1].decode(), info[2], info[12].decode()
            self.joint_index[name] = j
            self.link_index[link] = j
            if ltype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.movable.append(j)
                self.movable_names.append(name)
            if name.startswith("panda_joint") and len(self.arm) < 7:
                self.arm.append(j)
                self.lower.append(info[8])
                self.upper.append(info[9])
        self.ee = self.link_index[EE_LINK]
        self.fingers = [self.link_index[n] for n in FINGER_LINKS]
        for link in self.fingers:
            p.changeDynamics(self.robot, link, lateralFriction=2.0, physicsClientId=c)
        self.set_arm(DROID_HOME)
        self.set_gripper_state(0.0)
        self.command(DROID_HOME, 0.0)

    # ------------------------------------------------------------- utilities
    def close(self) -> None:
        self.p.disconnect(self.cid)

    def link_frame(self, body: int) -> tuple[np.ndarray, np.ndarray]:
        p, c = self.p, self.cid
        pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=c)
        ipos, iorn = p.getDynamicsInfo(body, -1, physicsClientId=c)[3:5]
        inv = p.invertTransform(ipos, iorn)
        lp, lo = p.multiplyTransforms(pos, orn, inv[0], inv[1])
        return np.asarray(lp), np.asarray(lo)

    def set_link_pose(self, body: int, position, orientation_xyzw) -> None:
        p, c = self.p, self.cid
        ipos, iorn = p.getDynamicsInfo(body, -1, physicsClientId=c)[3:5]
        cp, co = p.multiplyTransforms(position, orientation_xyzw, ipos, iorn)
        p.resetBasePositionAndOrientation(body, cp, co, physicsClientId=c)
        p.resetBaseVelocity(body, (0, 0, 0), (0, 0, 0), physicsClientId=c)

    def object_pose(self, object_id: str) -> tuple[np.ndarray, np.ndarray]:
        return self.link_frame(self.bodies[object_id])

    def aabb(self, object_id: str) -> tuple[np.ndarray, np.ndarray]:
        body = self.bodies[object_id]
        lo, hi = self.p.getAABB(body, -1, physicsClientId=self.cid)
        lo, hi = np.asarray(lo), np.asarray(hi)
        for j in range(self.p.getNumJoints(body, physicsClientId=self.cid)):
            a, b = self.p.getAABB(body, j, physicsClientId=self.cid)
            lo, hi = np.minimum(lo, a), np.maximum(hi, b)
        return lo, hi

    def support_top(self) -> float:
        return float(self.aabb(self.support_id)[1][2])

    # ------------------------------------------------------------------ robot
    def set_arm(self, q: Sequence[float]) -> None:
        for j, v in zip(self.arm, q):
            self.p.resetJointState(self.robot, j, float(v), physicsClientId=self.cid)

    def arm_q(self) -> np.ndarray:
        return np.array([self.p.getJointState(self.robot, j, physicsClientId=self.cid)[0]
                         for j in self.arm])

    def set_gripper_state(self, closed_fraction: float) -> None:
        target = float(np.clip(closed_fraction, 0, 1)) * GRIPPER_CLOSED_RAD
        for name, m in GRIPPER_MIMIC.items():
            self.p.resetJointState(self.robot, self.joint_index[name], m * target,
                                   physicsClientId=self.cid)

    def gripper_fraction(self) -> float:
        v = self.p.getJointState(self.robot, self.joint_index["finger_joint"],
                                 physicsClientId=self.cid)[0]
        return float(np.clip(v / GRIPPER_CLOSED_RAD, 0, 1))

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        s = self.p.getLinkState(self.robot, self.ee, computeForwardKinematics=True,
                                physicsClientId=self.cid)
        return np.asarray(s[4]), np.asarray(s[5])

    def command(self, q: Sequence[float], gripper: float) -> None:
        p, c = self.p, self.cid
        for j, v, f in zip(self.arm, q, DROID_ARM_FORCES):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=float(v),
                                    force=f, positionGain=DROID_ARM_GAIN, velocityGain=1.0,
                                    physicsClientId=c)
        target = float(np.clip(gripper, 0, 1)) * GRIPPER_CLOSED_RAD
        for name, m in GRIPPER_MIMIC.items():
            p.setJointMotorControl2(self.robot, self.joint_index[name], p.POSITION_CONTROL,
                                    targetPosition=m * target, force=GRIPPER_FORCE,
                                    positionGain=0.3, velocityGain=1.0, physicsClientId=c)

    def _cancel_robot_gravity(self) -> None:
        # RoboLab's DROID articulation has gravity disabled; objects keep it.
        p, c = self.p, self.cid
        for link in range(p.getNumJoints(self.robot, physicsClientId=c)):
            mass = p.getDynamicsInfo(self.robot, link, physicsClientId=c)[0]
            if mass > 0:
                com = p.getLinkState(self.robot, link, physicsClientId=c)[0]
                p.applyExternalForce(self.robot, link, (0, 0, mass * 9.81), com,
                                     p.WORLD_FRAME, physicsClientId=c)

    def step_physics(self, n: int = 1) -> None:
        for _ in range(n):
            if self.robot is not None and self.mode == "dynamic":
                self._cancel_robot_gravity()
            self.p.stepSimulation(physicsClientId=self.cid)

    def act(self, action: Sequence[float], *, record: bool = True) -> None:
        """Apply one 15 Hz DROID action [q1..q7, gripper] and advance physics."""

        a = np.asarray(action, dtype=float)
        q = np.clip(a[:7], self.lower, self.upper)
        self.command(q, a[7])
        self.step_physics(SUBSTEPS)
        if record:
            # Preserve the actual bounded command, including contact-limited apertures.
            # Binarizing this value makes action replay close through held objects.
            self.record(commanded=[*q.tolist(), float(np.clip(a[7], 0, 1))])

    # ---------------------------------------------------------------- contacts
    def contacts(self, a: str | int, b: str | int | None = None) -> list:
        ba = self.bodies[a] if isinstance(a, str) else a
        kwargs = {"bodyA": ba, "physicsClientId": self.cid}
        if b is not None:
            kwargs["bodyB"] = self.bodies[b] if isinstance(b, str) else b
        return list(self.p.getContactPoints(**kwargs))

    def finger_contacts(self, object_id: str) -> dict[int, list]:
        out: dict[int, list] = {f: [] for f in self.fingers}
        for cp in self.p.getContactPoints(bodyA=self.robot, bodyB=self.bodies[object_id],
                                          physicsClientId=self.cid):
            if cp[3] in out:
                out[cp[3]].append(cp)
        return out

    def grasped(self, object_id: str) -> bool:
        fc = self.finger_contacts(object_id)
        return all(len(v) > 0 for v in fc.values())

    def load_bearing_grasped(self, object_id: str, *, min_force: float = 0.1,
                             max_separation_m: float = 1e-4) -> bool:
        """Require bilateral contact with real normal load, not proximity."""
        fc = self.finger_contacts(object_id)
        for contacts in fc.values():
            if not any(float(cp[9]) >= min_force and float(cp[8]) <= max_separation_m
                       for cp in contacts):
                return False
        return True

    # --------------------------------------------------------------- recording
    def record(self, commanded: list[float] | None = None) -> None:
        p, c = self.p, self.cid
        objects = {}
        hosts = {a["host"] for a in getattr(self, "articulation", {}).values()}
        for oid, body in self.bodies.items():
            if oid == self.support_id and oid not in hosts:
                continue
            pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=c)
            lp, lo = self.link_frame(body)
            objects[oid] = {
                "position": list(pos),
                "orientation_xyzw": list(orn),
                "link_position": lp.tolist(),
                "link_orientation_xyzw": lo.tolist(),
                "role": self.roles[oid],
                # Measure all movable joints, not only task-bound articulation
                # components. Floating bottles/lids need their actual initial
                # passive joint state for faithful native reconstruction.
                "joint_positions": measured_object_joints(p, body, c),
            }
        ee_p, ee_o = self.ee_pose()
        base_p, base_o = self.link_frame(self.robot)  # URDF base link frame, not its COM
        frame = {
            "frame": len(self.frames),
            "time_s": len(self.frames) / CONTROL_HZ,
            "phase": self.phase,
            "operator_step_index": self.step_index,
            "active_subject_id": self.active_subject,
            "active_target_id": self.active_target,
            "policy_source": {"commanded_action": commanded},
            "robot": {
                "base_position": [float(v) for v in base_p],
                "base_orientation_xyzw": [float(v) for v in base_o],
                "joint_names": self.movable_names,
                "joint_positions": [p.getJointState(self.robot, j, physicsClientId=c)[0]
                                    for j in self.movable],
                "end_effector_position": ee_p.tolist(),
                "end_effector_orientation_xyzw": ee_o.tolist(),
                "finger_link_positions": {
                    n: list(p.getLinkState(self.robot, self.link_index[n],
                                           computeForwardKinematics=True,
                                           physicsClientId=c)[4])
                    for n in FINGER_LINKS
                },
                "droid_gripper_position": self.gripper_fraction(),
            },
            "objects": objects,
        }
        if self.active_subject in objects:
            frame["cup"] = objects[self.active_subject]
            contacts = self.finger_contacts(self.active_subject)
            frame["robot_cup_contact_links"] = sorted(k for k, v in contacts.items() if v)
        if self.active_target in objects:
            frame["tray"] = objects[self.active_target]
        if self.articulation:
            comp = next(iter(self.articulation))
            frame["articulated"] = {"component_id": comp,
                                    "joint_name": self.articulation[comp]["joint_name"],
                                    "position": self.joint_q(comp)}
            frame["articulation_positions"] = {k: self.joint_q(k) for k in self.articulation}
        self.frames.append(frame)
