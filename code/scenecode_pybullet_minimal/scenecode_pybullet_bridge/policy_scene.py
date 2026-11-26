"""PyBullet execution scene for typed DROID joint-position operator graphs.

The scene consumes task objects and roles from data, accepts the public RoboLab
eight-dimensional joint-position action contract, and records every 15 Hz state
required by the official SceneCode/Blender replay path.
"""

from __future__ import annotations

import json

from . import json_evidence
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .sdf import prepare_physics_sdf


@dataclass(frozen=True)
class PolicySceneObject:
    """One additional task object in a reusable operator-graph scene."""

    object_id: str
    sdf_path: Path
    center_xy: tuple[float, float]
    role: str
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    lateral_friction: float = 1.2
    collision_enabled: bool = True
    fixed: bool = False


@dataclass(frozen=True)
class PolicySceneInputs:
    table_sdf: Path
    cup_sdf: Path
    tray_sdf: Path
    panda_urdf: Path
    output_dir: Path
    robot_embodiment: str = "panda_stock"
    initial_arm_joint_positions: tuple[float, float, float, float, float, float, float] = (
        0.0,
        -0.45,
        0.0,
        -2.25,
        0.0,
        1.85,
        0.8,
    )
    disable_robot_gravity: bool = False
    arm_position_gain: float = 0.18
    arm_joint_forces: tuple[float, float, float, float, float, float, float] = (
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
        300.0,
    )
    action_authority: str = "Cosmos3-Nano-Policy-DROID official joint_pos head"
    cup_center_xy: tuple[float, float] = (0.24, -0.08)
    tray_center_xy: tuple[float, float] = (-0.18, 0.10)
    cup_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    tray_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    success_alignment_axis_local: tuple[float, float, float] | None = None
    success_alignment_axis_world: tuple[float, float, float] | None = None
    success_predicate: str = "in_container"
    operator: str = "pick_place"
    subject_lateral_friction: float = 1.2
    target_collision_enabled: bool = True
    subject_id: str = "cup"
    target_id: str = "tray"
    additional_objects: tuple[PolicySceneObject, ...] = ()


@dataclass(frozen=True)
class PolicySceneResult:
    trajectory_path: str
    frame_count: int
    simulation_seconds: float
    grasp_contact_left_ever: bool
    grasp_contact_right_ever: bool
    dual_finger_contact_frame_count: int
    carry_frame_count: int
    dual_finger_supported_carry_fraction: float
    opposed_external_carry_frame_count: int
    opposed_external_carry_fraction: float
    minimum_opposed_contact_margin_m: float
    cup_lift_m: float
    cup_inside_tray_xy: bool
    cup_in_tray_contact: bool
    cup_released: bool
    object_attachment_constraint_count: int
    gripper_coupling_constraint_count: int
    expected_gripper_coupling_constraint_count: int
    joint_limit_clamp_count: int
    robot_embodiment: str
    operator: str
    success_predicate: str
    push_contact_frame_count: int
    subject_horizontal_displacement_m: float
    predicate_evidence: dict[str, Any]
    success: bool
    # 2026-09-11 household-scale repair: the projection margin below measures the
    # contacts against one reference point at the subject root link.  For a tall
    # object grasped far above that point (a kettle carried by its handle), a
    # small legitimate pendulum swing in the grip moves the reference sideways
    # and breaks the projection even though the pinch is force-closed.  The
    # contact-normal opposition is the reference-frame-independent test; both are
    # recorded so a carry must still show dual opposed contacts in every frame.
    opposed_external_geometric_fraction: float = 0.0
    minimum_opposed_normal_margin: float = float("-inf")
    opposed_normal_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def finger_separation_axis(
    frame: dict[str, Any], *, finger_names: Sequence[str]
) -> np.ndarray | None:
    """Return the unit axis between the two finger contact link origins."""

    positions: list[np.ndarray] = []
    for name in finger_names:
        position = (frame.get("robot") or {}).get("finger_link_positions", {}).get(
            name
        )
        if position is None:
            return None
        positions.append(np.asarray(position, dtype=np.float64))
    if len(positions) != 2:
        return None
    axis = positions[1] - positions[0]
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-9:
        return None
    return axis / norm


def strongest_contact_per_finger(
    frame: dict[str, Any], *, finger_links: Sequence[int]
) -> list[dict[str, Any]] | None:
    """Return the strongest recorded contact for each finger contact link."""

    evidence = frame.get("robot_cup_contact_evidence") or []
    strongest: list[dict[str, Any]] = []
    for link in finger_links:
        candidates = [
            item
            for item in evidence
            if int(item.get("robot_link_index", -1)) == link
        ]
        if not candidates:
            return None
        strongest.append(
            max(
                candidates,
                key=lambda item: (
                    float(item.get("normal_force_n") or 0.0),
                    -abs(float(item.get("contact_distance_m") or 0.0)),
                ),
            )
        )
    return strongest


def opposed_projection_margin(
    frame: dict[str, Any],
    *,
    finger_names: Sequence[str],
    finger_links: Sequence[int],
) -> float:
    """Signed clearance of the finger contacts about the subject root link.

    Merely reporting one contact from each finger is insufficient: one finger can
    sit inside a hollow cup while the other pinches the same rim from outside.
    Project the strongest contact from each finger onto the instantaneous
    finger-separation axis.  A positive margin means the contacts lie on
    opposite sides of the subject link origin.

    This projection is exact for a subject whose grasp site is close to its root
    link, but the reference distance multiplies any residual rotation of a
    subject that settles inside the grip, so it is not scale-free; the
    contact-normal test below is the reference-frame-independent companion.
    """

    axis = finger_separation_axis(frame, finger_names=finger_names)
    if axis is None:
        return float("-inf")
    cup_center = np.asarray(
        (frame.get("cup") or {}).get("link_position"), dtype=np.float64
    )
    if cup_center.shape != (3,) or not np.isfinite(cup_center).all():
        return float("-inf")
    strongest = strongest_contact_per_finger(frame, finger_links=finger_links)
    if strongest is None:
        return float("-inf")
    projections: list[float] = []
    for contact_item in strongest:
        contact = np.asarray(
            contact_item.get("position_on_cup_world"), dtype=np.float64
        )
        if contact.shape != (3,) or not np.isfinite(contact).all():
            return float("-inf")
        projections.append(float(np.dot(contact - cup_center, axis)))
    return min(-projections[0], projections[1])


def opposed_normal_margin(
    frame: dict[str, Any],
    *,
    finger_names: Sequence[str],
    finger_links: Sequence[int],
) -> float:
    """Return the reference-frame-independent contact-normal opposition.

    A genuine opposed pinch is force-closed: each finger presses the subject
    from the opposite side, so the two contact normals are antiparallel and lie
    along the instantaneous finger-separation axis.  Unlike the projection this
    does not depend on any single reference point, so it stays valid when a tall
    subject settles or swings slightly inside the grip.  It still rejects a
    same-side "carry" (for example one finger inside a hollow cup and the other
    on the outside of the same rim), whose two normals point the same way.
    """

    axis = finger_separation_axis(frame, finger_names=finger_names)
    if axis is None:
        return float("-inf")
    strongest = strongest_contact_per_finger(frame, finger_links=finger_links)
    if strongest is None:
        return float("-inf")
    alignments: list[float] = []
    for contact_item in strongest:
        normal = np.asarray(
            contact_item.get("normal_on_cup_world"), dtype=np.float64
        )
        if normal.shape != (3,) or not np.isfinite(normal).all():
            return float("-inf")
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= 1.0e-9:
            return float("-inf")
        alignments.append(float(np.dot(normal / magnitude, axis)))
    if len(alignments) != 2:
        return float("-inf")
    strength = min(abs(value) for value in alignments)
    return strength if alignments[0] * alignments[1] < 0.0 else -strength


def inserted_geometry_passed(
    *,
    center_inside_target_xy: bool,
    target_contact: bool,
    subject_bottom_z_m: float,
    subject_top_z_m: float,
    target_bottom_z_m: float,
    target_top_z_m: float,
    alignment_cosine: float,
) -> bool:
    """Return whether a subject occupies the target's insertion volume.

    A valid insert may protrude above the opening or sit completely inside the
    target.  Requiring the subject top to remain above the rim incorrectly
    rejects the latter.  The lower overlap bound prevents an object that fell
    entirely below the target from passing.
    """

    subject_height_m = max(subject_top_z_m - subject_bottom_z_m, 1.0e-9)
    insertion_depth_m = target_top_z_m - subject_bottom_z_m
    return bool(
        center_inside_target_xy
        and target_contact
        and insertion_depth_m >= 0.20 * subject_height_m
        and subject_top_z_m >= target_bottom_z_m
        and alignment_cosine >= 0.85
    )


class PyBulletDroidPolicyScene:
    """Execute 15 Hz DROID absolute-joint commands in the reviewed scene."""

    physics_hz = 240
    control_hz = 15
    physics_steps_per_action = physics_hz // control_hz
    panda_opening_m = 0.04
    droid_gripper_closed_rad = math.pi / 4.0
    supported_robot_embodiments = {"panda_stock", "robolab_droid"}
    supported_success_predicates = {
        "in_container",
        "stacked",
        "inserted",
        "covering",
        "on_region",
        "contact",
    }
    supported_operators = {"pick_place", "push"}
    droid_gripper_multipliers = {
        "finger_joint": 1.0,
        "right_outer_knuckle_joint": -1.0,
        "left_inner_finger_knuckle_joint": 1.0,
        "right_inner_finger_knuckle_joint": -1.0,
        "left_inner_finger_joint": -1.0,
        "right_inner_finger_joint": 1.0,
    }

    def __init__(self, inputs: PolicySceneInputs) -> None:
        self.inputs = inputs
        if inputs.robot_embodiment not in self.supported_robot_embodiments:
            raise ValueError(
                f"unsupported robot embodiment: {inputs.robot_embodiment!r}"
            )
        if inputs.success_predicate not in self.supported_success_predicates:
            raise ValueError(
                f"unsupported physical success predicate: {inputs.success_predicate!r}"
            )
        if inputs.operator not in self.supported_operators:
            raise ValueError(f"unsupported controller operator: {inputs.operator!r}")
        if not math.isfinite(inputs.subject_lateral_friction) or not (
            0.0 <= inputs.subject_lateral_friction <= 2.0
        ):
            raise ValueError("subject lateral friction must lie in [0, 2]")
        if not inputs.action_authority.strip():
            raise ValueError("action authority must be non-empty")
        alignment_requested = (
            inputs.success_alignment_axis_local is not None
            or inputs.success_alignment_axis_world is not None
        )
        if alignment_requested and (
            inputs.success_alignment_axis_local is None
            or inputs.success_alignment_axis_world is None
        ):
            raise ValueError("success alignment requires both local and world axes")
        if alignment_requested:
            for axis in (
                inputs.success_alignment_axis_local,
                inputs.success_alignment_axis_world,
            ):
                vector = np.asarray(axis, dtype=np.float64)
                if (
                    vector.shape != (3,)
                    or not np.isfinite(vector).all()
                    or float(np.linalg.norm(vector)) <= 1.0e-9
                ):
                    raise ValueError("success alignment axes must be finite and non-zero")
        primary = (
            PolicySceneObject(
                object_id=inputs.subject_id,
                sdf_path=inputs.cup_sdf,
                center_xy=inputs.cup_center_xy,
                role="manipuland",
                orientation_xyzw=inputs.cup_orientation_xyzw,
                lateral_friction=inputs.subject_lateral_friction,
            ),
            PolicySceneObject(
                object_id=inputs.target_id,
                sdf_path=inputs.tray_sdf,
                center_xy=inputs.tray_center_xy,
                role="destination",
                orientation_xyzw=inputs.tray_orientation_xyzw,
                lateral_friction=0.9,
                collision_enabled=inputs.target_collision_enabled,
                fixed=not inputs.target_collision_enabled,
            ),
        )
        object_specs = (*primary, *inputs.additional_objects)
        object_ids = [item.object_id for item in object_specs]
        if any(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", object_id) is None
            for object_id in object_ids
        ):
            raise ValueError("scene object ids must be safe non-empty identifiers")
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("scene object ids must be unique")
        if any(item.role not in {"manipuland", "destination"} for item in object_specs):
            raise ValueError("scene object role must be manipuland or destination")
        for item in object_specs:
            center = np.asarray(item.center_xy, dtype=np.float64)
            if center.shape != (2,) or not np.isfinite(center).all():
                raise ValueError("scene object centers must be finite XY pairs")
            orientation = np.asarray(item.orientation_xyzw, dtype=np.float64)
            if (
                orientation.shape != (4,)
                or not np.isfinite(orientation).all()
                or not np.isclose(np.linalg.norm(orientation), 1.0, atol=1.0e-6)
            ):
                raise ValueError("scene object orientation must be a normalized quaternion")
            if not math.isfinite(item.lateral_friction) or not (
                0.0 <= item.lateral_friction <= 2.0
            ):
                raise ValueError("scene object lateral friction must lie in [0, 2]")
            if item.fixed and item.role == "manipuland":
                raise ValueError("a manipuland may not be fixed")
        self.object_specs = {item.object_id: item for item in object_specs}
        # Validate the data-only contract before importing the server-only
        # simulator.  This keeps local unit tests useful without PyBullet.
        try:
            import pybullet as bullet
            import pybullet_data
        except ImportError as exc:  # pragma: no cover - server-only dependency
            raise RuntimeError("PyBullet is required for policy execution") from exc
        self.p = bullet
        self.pybullet_data = pybullet_data
        self.active_subject_id = inputs.subject_id
        self.active_target_id = inputs.target_id
        self.active_success_predicate = inputs.success_predicate
        self.active_operator = inputs.operator
        self.active_success_alignment_axis_local = (
            inputs.success_alignment_axis_local
        )
        self.active_success_alignment_axis_world = (
            inputs.success_alignment_axis_world
        )
        self.operator_step_index = 0
        self.active_start_frame = 0
        self.output_dir = inputs.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        physics_dir = self.output_dir / "physics_sdf"
        self.prepared = {
            "table": prepare_physics_sdf(
                inputs.table_sdf, physics_dir / "table.sdf"
            )
        }
        for object_id, item in self.object_specs.items():
            self.prepared[object_id] = prepare_physics_sdf(
                item.sdf_path,
                physics_dir / f"object_{object_id}.sdf",
                allow_visual_bounds_fallback=not item.collision_enabled,
            )
        self.client = self.p.connect(self.p.DIRECT)
        if self.client < 0:
            raise RuntimeError("failed to connect to PyBullet in DIRECT mode")

        self.frames: list[dict[str, Any]] = []
        self.physics_step = 0
        self.max_cup_z = -math.inf
        self.initial_cup_z = 0.0
        self.joint_limit_clamp_count = 0
        self._closed = False
        try:
            self._build_scene()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "PyBulletDroidPolicyScene":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self.p.disconnect(self.client)
            self._closed = True

    @staticmethod
    def _finite_action(action: Sequence[float]) -> np.ndarray:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError(f"DROID action must be finite shape (8,), got {values}")
        return values

    def _build_scene(self) -> None:
        p = self.p
        p.resetSimulation()
        p.setAdditionalSearchPath(self.pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(1.0 / self.physics_hz)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / self.physics_hz,
            numSolverIterations=200,
            deterministicOverlappingPairs=1,
        )
        p.loadURDF("plane.urdf")
        self.table = p.loadSDF(self.prepared["table"].physics_sdf)[0]
        self.objects = {
            object_id: p.loadSDF(self.prepared[object_id].physics_sdf)[0]
            for object_id in self.object_specs
        }
        self.cup = self.objects[self.active_subject_id]
        self.tray = self.objects[self.active_target_id]
        self.panda = p.loadURDF(
            str(self.inputs.panda_urdf.resolve()),
            basePosition=(0.0, 0.0, 0.0),
            baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, math.pi / 2.0)),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )
        p.changeDynamics(self.table, -1, mass=0.0, lateralFriction=0.8)

        def place_aabb(body: int, center_xy: Sequence[float], bottom_z: float) -> None:
            position, orientation = p.getBasePositionAndOrientation(body)
            lower, upper = p.getAABB(body)
            current_center = (
                (lower[0] + upper[0]) / 2.0,
                (lower[1] + upper[1]) / 2.0,
            )
            delta = (
                center_xy[0] - current_center[0],
                center_xy[1] - current_center[1],
                bottom_z - lower[2],
            )
            p.resetBasePositionAndOrientation(
                body,
                tuple(position[index] + delta[index] for index in range(3)),
                orientation,
            )

        table_top = p.getAABB(self.table)[1][2]
        self.panda_base = (0.0, -0.43, table_top + 0.002)
        p.resetBasePositionAndOrientation(
            self.panda,
            self.panda_base,
            p.getQuaternionFromEuler((0.0, 0.0, math.pi / 2.0)),
        )
        for object_id, item in self.object_specs.items():
            body = self.objects[object_id]
            position, _ = p.getBasePositionAndOrientation(body)
            p.resetBasePositionAndOrientation(
                body, position, item.orientation_xyzw
            )
            place_aabb(body, item.center_xy, table_top + 0.002)
            p.changeDynamics(
                body,
                -1,
                lateralFriction=item.lateral_friction,
                spinningFriction=0.02,
                rollingFriction=0.002,
                restitution=0.0,
                collisionMargin=0.0,
            )
            if item.fixed or not item.collision_enabled:
                p.changeDynamics(body, -1, mass=0.0)
            if not item.collision_enabled:
                for link_index in range(-1, p.getNumJoints(body)):
                    p.setCollisionFilterGroupMask(body, link_index, 0, 0)

        self.movable_joints: list[int] = []
        self.arm_joints: list[int] = []
        self.finger_joints: list[int] = []
        self.gripper_joints: dict[str, int] = {}
        self.joint_names: list[str] = []
        self.arm_limits: list[tuple[float, float]] = []
        for joint_index in range(p.getNumJoints(self.panda)):
            info = p.getJointInfo(self.panda, joint_index)
            name = info[1].decode("utf-8")
            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.movable_joints.append(joint_index)
                self.joint_names.append(name)
            if name.startswith("panda_joint"):
                self.arm_joints.append(joint_index)
                self.arm_limits.append((float(info[8]), float(info[9])))
            if self.inputs.robot_embodiment == "panda_stock" and "finger_joint" in name:
                self.finger_joints.append(joint_index)
            if name in self.droid_gripper_multipliers:
                self.gripper_joints[name] = joint_index
        self.arm_joints = self.arm_joints[:7]
        self.arm_limits = self.arm_limits[:7]
        if len(self.arm_joints) != 7:
            raise RuntimeError(f"unexpected Franka arm layout: {self.arm_joints}")
        link_indices = {
            p.getJointInfo(self.panda, joint_index)[12].decode("utf-8"): joint_index
            for joint_index in range(p.getNumJoints(self.panda))
        }
        if self.inputs.robot_embodiment == "panda_stock":
            if len(self.finger_joints) != 2:
                raise RuntimeError(
                    f"unexpected Panda finger layout: {self.finger_joints}"
                )
            self.finger_contact_links = list(self.finger_joints)
            self.end_effector = link_indices["panda_grasptarget"]
        else:
            missing = set(self.droid_gripper_multipliers) - set(self.gripper_joints)
            if missing:
                raise RuntimeError(
                    f"DROID Franka+Robotiq URDF is missing joints: {sorted(missing)}"
                )
            try:
                self.finger_contact_links = [
                    link_indices["left_inner_finger"],
                    link_indices["right_inner_finger"],
                ]
                self.end_effector = link_indices["base_link"]
            except KeyError as exc:
                raise RuntimeError(
                    f"DROID Franka+Robotiq URDF is missing link: {exc.args[0]}"
                ) from exc

        neutral = self.inputs.initial_arm_joint_positions
        if len(neutral) != 7 or not np.isfinite(np.asarray(neutral)).all():
            raise ValueError("initial Panda arm position must contain seven finite values")
        if not 0.0 < self.inputs.arm_position_gain <= 1.0:
            raise ValueError("PyBullet arm position gain must lie in (0,1]")
        forces = self.inputs.arm_joint_forces
        if (
            len(forces) != 7
            or not np.isfinite(np.asarray(forces)).all()
            or any(force <= 0.0 for force in forces)
        ):
            raise ValueError("PyBullet arm forces must contain seven positive values")
        self.arm_force_by_joint = dict(zip(self.arm_joints, forces, strict=True))
        for joint, position in zip(self.arm_joints, neutral, strict=True):
            p.resetJointState(self.panda, joint, position)
            self._command_arm_joint(joint, position)
        if self.inputs.robot_embodiment == "panda_stock":
            for joint in self.finger_joints:
                p.resetJointState(self.panda, joint, self.panda_opening_m)
                p.changeDynamics(
                    self.panda, joint, lateralFriction=2.0, collisionMargin=0.0
                )
                self._command_finger_joint(joint, self.panda_opening_m)
            self.gear_constraint = p.createConstraint(
                self.panda,
                self.finger_joints[0],
                self.panda,
                self.finger_joints[1],
                jointType=p.JOINT_GEAR,
                jointAxis=(1, 0, 0),
                parentFramePosition=(0, 0, 0),
                childFramePosition=(0, 0, 0),
            )
            p.changeConstraint(
                self.gear_constraint, gearRatio=-1, erp=0.1, maxForce=50
            )
            self.expected_gripper_coupling_constraint_count = 1
        else:
            for joint in self.gripper_joints.values():
                p.resetJointState(self.panda, joint, 0.0)
            for link in self.finger_contact_links:
                p.changeDynamics(
                    self.panda, link, lateralFriction=2.0, collisionMargin=0.0
                )
            # RoboLab's USD uses one driven finger_joint plus five PhysX mimic
            # joints. PyBullet does not import USD mimic schemas, so issue the
            # same six generalized-coordinate targets explicitly.
            self._command_droid_gripper(0.0)
            self.gear_constraint = None
            self.expected_gripper_coupling_constraint_count = 0

        # Let collision geometry settle before defining benchmark frame zero.
        for _ in range(self.physics_hz):
            self._step_physics_once()
        self.physics_step = 0
        self._reset_active_tracking()
        self._record(
            phase="conditioning",
            source_window=None,
            source_action_index=None,
            commanded_action=None,
            clamped_joint_indices=[],
        )

    def _reset_active_tracking(self) -> None:
        position = self.p.getBasePositionAndOrientation(self.cup)[0]
        self.initial_cup_z = float(position[2])
        self.initial_cup_xy = np.asarray(position[:2], dtype=np.float64)
        self.max_cup_z = self.initial_cup_z

    def activate_pair(
        self,
        *,
        subject_id: str,
        target_id: str,
        success_predicate: str,
        operator: str = "pick_place",
        step_index: int,
        success_alignment_axis_local: tuple[float, float, float] | None = None,
        success_alignment_axis_world: tuple[float, float, float] | None = None,
    ) -> None:
        """Select one compiled operator step without rebuilding the scene."""

        if subject_id == target_id or subject_id not in self.objects or target_id not in self.objects:
            raise ValueError("active subject and target must be distinct scene objects")
        if self.object_specs[subject_id].role != "manipuland":
            raise ValueError("active subject must have manipuland role")
        if self.object_specs[target_id].role != "destination":
            raise ValueError("active target must have destination role")
        if success_predicate not in self.supported_success_predicates:
            raise ValueError(f"unsupported physical success predicate: {success_predicate!r}")
        if operator not in self.supported_operators:
            raise ValueError(f"unsupported controller operator: {operator!r}")
        if step_index < 0:
            raise ValueError("operator step index must be non-negative")
        alignment_requested = (
            success_alignment_axis_local is not None
            or success_alignment_axis_world is not None
        )
        if alignment_requested and (
            success_alignment_axis_local is None
            or success_alignment_axis_world is None
        ):
            raise ValueError("success alignment requires both local and world axes")
        if alignment_requested:
            for axis in (
                success_alignment_axis_local,
                success_alignment_axis_world,
            ):
                vector = np.asarray(axis, dtype=np.float64)
                if (
                    vector.shape != (3,)
                    or not np.isfinite(vector).all()
                    or float(np.linalg.norm(vector)) <= 1.0e-9
                ):
                    raise ValueError("success alignment axes must be finite and non-zero")
        self.active_subject_id = subject_id
        self.active_target_id = target_id
        self.active_success_predicate = success_predicate
        self.active_operator = operator
        self.active_success_alignment_axis_local = success_alignment_axis_local
        self.active_success_alignment_axis_world = success_alignment_axis_world
        self.operator_step_index = step_index
        self.active_start_frame = len(self.frames)
        self.cup = self.objects[subject_id]
        self.tray = self.objects[target_id]
        self._reset_active_tracking()

    def _command_arm_joint(self, joint: int, position: float) -> None:
        self.p.setJointMotorControl2(
            self.panda,
            joint,
            self.p.POSITION_CONTROL,
            targetPosition=position,
            force=self.arm_force_by_joint[joint],
            positionGain=self.inputs.arm_position_gain,
            velocityGain=1.0,
        )

    def _step_physics_once(self) -> None:
        """Step once, optionally cancelling gravity on robot links only.

        RoboLab's DROID articulation has ``disable_gravity=True`` while the
        task objects retain gravity. PyBullet gravity is global, so matching
        that contract requires one equal-and-opposite force at each robot
        link COM before every physics step.
        """

        if self.inputs.disable_robot_gravity:
            for link_index in range(self.p.getNumJoints(self.panda)):
                mass = float(self.p.getDynamicsInfo(self.panda, link_index)[0])
                if mass <= 0.0:
                    continue
                center_of_mass = self.p.getLinkState(self.panda, link_index)[0]
                self.p.applyExternalForce(
                    self.panda,
                    link_index,
                    (0.0, 0.0, mass * 9.81),
                    center_of_mass,
                    self.p.WORLD_FRAME,
                )
        self.p.stepSimulation()

    def _command_finger_joint(self, joint: int, opening: float) -> None:
        self.p.setJointMotorControl2(
            self.panda,
            joint,
            self.p.POSITION_CONTROL,
            targetPosition=opening,
            force=120,
            positionGain=0.3,
            velocityGain=1.0,
        )

    def _command_droid_gripper(self, closed_fraction: float) -> None:
        driver_target = float(
            np.clip(closed_fraction, 0.0, 1.0) * self.droid_gripper_closed_rad
        )
        for name, multiplier in self.droid_gripper_multipliers.items():
            self.p.setJointMotorControl2(
                self.panda,
                self.gripper_joints[name],
                self.p.POSITION_CONTROL,
                targetPosition=multiplier * driver_target,
                force=16.5,
                positionGain=0.3,
                velocityGain=1.0,
            )

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def joint_position(self) -> np.ndarray:
        return np.asarray(
            [self.p.getJointState(self.panda, joint)[0] for joint in self.arm_joints],
            dtype=np.float32,
        )

    def gripper_position(self) -> float:
        if self.inputs.robot_embodiment == "robolab_droid":
            driver = self.p.getJointState(
                self.panda, self.gripper_joints["finger_joint"]
            )[0]
            return float(
                np.clip(driver / self.droid_gripper_closed_rad, 0.0, 1.0)
            )
        opening_width = float(
            np.clip(
                sum(self.p.getJointState(self.panda, joint)[0] for joint in self.finger_joints),
                0.0,
                2.0 * self.panda_opening_m,
            )
        )
        return 1.0 - opening_width / (2.0 * self.panda_opening_m)

    def execute_action(
        self,
        action: Sequence[float],
        *,
        source_window: int,
        source_action_index: int,
        phase_override: str | None = None,
    ) -> dict[str, Any]:
        values = self._finite_action(action)
        arm_targets = values[:7].copy()
        clamped: list[int] = []
        for index, (lower, upper) in enumerate(self.arm_limits):
            target = float(np.clip(arm_targets[index], lower, upper))
            if target != float(arm_targets[index]):
                clamped.append(index)
                arm_targets[index] = target
        self.joint_limit_clamp_count += len(clamped)
        gripper_binary = float(values[7] > 0.5)
        for joint, target in zip(self.arm_joints, arm_targets, strict=True):
            self._command_arm_joint(joint, float(target))
        if self.inputs.robot_embodiment == "panda_stock":
            finger_target = 0.0 if gripper_binary else self.panda_opening_m
            for joint in self.finger_joints:
                self._command_finger_joint(joint, finger_target)
        else:
            self._command_droid_gripper(gripper_binary)
        for _ in range(self.physics_steps_per_action):
            self._step_physics_once()
            self.physics_step += 1
            self.max_cup_z = max(
                self.max_cup_z,
                self.p.getBasePositionAndOrientation(self.cup)[0][2],
            )
        contact_links = {
            int(contact[3])
            for contact in self.p.getContactPoints(bodyA=self.panda, bodyB=self.cup)
        }
        cup_z = self.p.getBasePositionAndOrientation(self.cup)[0][2]
        if phase_override is not None:
            phase = phase_override.strip()
            if not phase or any(character.isspace() for character in phase):
                raise ValueError("phase override must be one non-empty token")
        elif set(self.finger_contact_links).issubset(contact_links):
            phase = "lift" if cup_z > self.initial_cup_z + 0.02 else "grasp"
        elif gripper_binary:
            phase = "closed_motion"
        else:
            phase = "open_motion"
        self._record(
            phase=phase,
            source_window=source_window,
            source_action_index=source_action_index,
            commanded_action=[*arm_targets.tolist(), gripper_binary],
            clamped_joint_indices=clamped,
        )
        return {
            "actual_joint_position": self.joint_position().tolist(),
            "actual_gripper_position": self.gripper_position(),
            "clamped_joint_indices": clamped,
            "phase": phase,
        }

    def _link_pose(self, body: int) -> tuple[Sequence[float], Sequence[float]]:
        p = self.p
        com_position, com_orientation = p.getBasePositionAndOrientation(body)
        dynamics = p.getDynamicsInfo(body, -1)
        inertial_position, inertial_orientation = dynamics[3], dynamics[4]
        link_from_inertial = p.invertTransform(inertial_position, inertial_orientation)
        return p.multiplyTransforms(
            com_position,
            com_orientation,
            link_from_inertial[0],
            link_from_inertial[1],
        )

    def _record(
        self,
        *,
        phase: str,
        source_window: int | None,
        source_action_index: int | None,
        commanded_action: list[float] | None,
        clamped_joint_indices: list[int],
    ) -> None:
        p = self.p
        cup_pose = p.getBasePositionAndOrientation(self.cup)
        tray_pose = p.getBasePositionAndOrientation(self.tray)
        cup_link_pose = self._link_pose(self.cup)
        tray_link_pose = self._link_pose(self.tray)
        object_records = {}
        for object_id, body in self.objects.items():
            pose = p.getBasePositionAndOrientation(body)
            link_pose = self._link_pose(body)
            object_records[object_id] = {
                "position": list(pose[0]),
                "orientation_xyzw": list(pose[1]),
                "link_position": list(link_pose[0]),
                "link_orientation_xyzw": list(link_pose[1]),
                "role": self.object_specs[object_id].role,
            }
        panda_link_pose = self._link_pose(self.panda)
        contacts = p.getContactPoints(bodyA=self.panda, bodyB=self.cup)
        contact_links = sorted({int(contact[3]) for contact in contacts})
        contact_evidence = [
            {
                "robot_link_index": int(contact[3]),
                "cup_link_index": int(contact[4]),
                "position_on_robot_world": list(contact[5]),
                "position_on_cup_world": list(contact[6]),
                "normal_on_cup_world": list(contact[7]),
                "contact_distance_m": float(contact[8]),
                "normal_force_n": float(contact[9]),
            }
            for contact in contacts
            if int(contact[3]) in self.finger_contact_links
        ]
        ee_pose = p.getLinkState(
            self.panda, self.end_effector, computeForwardKinematics=True
        )[4:6]
        frame_index = len(self.frames)
        self.frames.append(
            {
                "frame": frame_index,
                "sim_step": self.physics_step,
                "time_s": frame_index / self.control_hz,
                "phase": phase,
                "operator_step_index": self.operator_step_index,
                "active_subject_id": self.active_subject_id,
                "active_target_id": self.active_target_id,
                "active_operator": self.active_operator,
                "active_success_predicate": self.active_success_predicate,
                "policy_source": {
                    "window_index": source_window,
                    "action_index": source_action_index,
                    "commanded_action": commanded_action,
                    "joint_limit_clamped_indices": clamped_joint_indices,
                },
                "robot": {
                    "base_position": list(panda_link_pose[0]),
                    "base_orientation_xyzw": list(panda_link_pose[1]),
                    "joint_names": self.joint_names,
                    "joint_positions": [
                        float(p.getJointState(self.panda, joint)[0])
                        for joint in self.movable_joints
                    ],
                    "end_effector_position": list(ee_pose[0]),
                    "end_effector_orientation_xyzw": list(ee_pose[1]),
                    "finger_link_positions": {
                        p.getJointInfo(self.panda, joint)[12].decode("utf-8"): list(
                            p.getLinkState(
                                self.panda, joint, computeForwardKinematics=True
                            )[4]
                        )
                        for joint in self.finger_contact_links
                    },
                    "droid_gripper_position": self.gripper_position(),
                },
                "cup": {
                    "position": list(cup_pose[0]),
                    "orientation_xyzw": list(cup_pose[1]),
                    "link_position": list(cup_link_pose[0]),
                    "link_orientation_xyzw": list(cup_link_pose[1]),
                },
                "tray": {
                    "position": list(tray_pose[0]),
                    "orientation_xyzw": list(tray_pose[1]),
                    "link_position": list(tray_link_pose[0]),
                    "link_orientation_xyzw": list(tray_link_pose[1]),
                },
                "objects": object_records,
                "robot_cup_contact_links": contact_links,
                "robot_cup_contact_evidence": contact_evidence,
                "subject_target_contact": bool(
                    p.getContactPoints(bodyA=self.cup, bodyB=self.tray)
                ),
            }
        )

    def write_outputs(self) -> PolicySceneResult:
        p = self.p
        trajectory_path = self.output_dir / "trajectory.json"
        trajectory_path.write_text(
            json_evidence.dumps(
                {
                    "schema_version": "scenecode_pybullet_droid_policy_rollout.v1",
                    "authority": {
                        "physics": "PyBullet",
                        "visual_assets": "official SceneCode SDF/glTF",
                        "robot": (
                            "RoboLab DROID Franka+Robotiq 2F-85 portable URDF"
                            if self.inputs.robot_embodiment == "robolab_droid"
                            else "PyBullet bundled Franka Panda"
                        ),
                        "actions": self.inputs.action_authority,
                    },
                    "physics_hz": self.physics_hz,
                    "sample_hz": self.control_hz,
                    "control_hz": self.control_hz,
                    "controller": {
                        "initial_arm_joint_positions": list(
                            self.inputs.initial_arm_joint_positions
                        ),
                        "disable_robot_gravity": self.inputs.disable_robot_gravity,
                        "arm_position_gain": self.inputs.arm_position_gain,
                        "arm_joint_forces": list(self.inputs.arm_joint_forces),
                        "robot_embodiment": self.inputs.robot_embodiment,
                        "action_authority": self.inputs.action_authority,
                        "success_predicate": self.active_success_predicate,
                        "operator": self.active_operator,
                    },
                    "prepared_assets": {
                        name: artifact.to_dict()
                        for name, artifact in self.prepared.items()
                    },
                    "frames": self.frames,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        cup_position, cup_orientation = p.getBasePositionAndOrientation(self.cup)
        tray_position = p.getBasePositionAndOrientation(self.tray)[0]
        cup_aabb = p.getAABB(self.cup)
        tray_aabb = p.getAABB(self.tray)
        cup_half_xy = (
            (cup_aabb[1][0] - cup_aabb[0][0]) / 2.0,
            (cup_aabb[1][1] - cup_aabb[0][1]) / 2.0,
        )
        # AABB-derived horizontal admission scales to arbitrary generated
        # subject/target assets.  Keep only a small geometric margin so a
        # legitimate rim contact does not fail a container or cover task.
        cup_inside_tray_xy = all(
            tray_aabb[0][axis] - 0.005
            <= cup_position[axis]
            <= tray_aabb[1][axis] + 0.005
            for axis in (0, 1)
        )
        cup_fully_over_target_xy = all(
            tray_aabb[0][axis] - 0.005 + cup_half_xy[axis]
            <= cup_position[axis]
            <= tray_aabb[1][axis] + 0.005 - cup_half_xy[axis]
            for axis in (0, 1)
        )
        cup_in_tray_contact = bool(p.getContactPoints(bodyA=self.cup, bodyB=self.tray))
        cup_released = not bool(p.getContactPoints(bodyA=self.panda, bodyB=self.cup))
        subject_bottom_z = float(cup_aabb[0][2])
        subject_top_z = float(cup_aabb[1][2])
        target_bottom_z = float(tray_aabb[0][2])
        target_top_z = float(tray_aabb[1][2])
        subject_height_m = max(subject_top_z - subject_bottom_z, 1.0e-9)
        insertion_depth_m = target_top_z - subject_bottom_z
        rotation = np.asarray(
            p.getMatrixFromQuaternion(cup_orientation), dtype=np.float64
        ).reshape(3, 3)
        upright_cosine = float(rotation[2, 2])
        if self.active_success_alignment_axis_local is None:
            alignment_axis_local = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
            alignment_axis_world = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        else:
            alignment_axis_local = np.asarray(
                self.active_success_alignment_axis_local, dtype=np.float64
            )
            alignment_axis_world = np.asarray(
                self.active_success_alignment_axis_world, dtype=np.float64
            )
        alignment_axis_local /= np.linalg.norm(alignment_axis_local)
        alignment_axis_world /= np.linalg.norm(alignment_axis_world)
        subject_alignment_cosine = float(
            np.dot(rotation @ alignment_axis_local, alignment_axis_world)
        )
        left_ever = any(
            self.finger_contact_links[0] in frame["robot_cup_contact_links"]
            for frame in self.frames
        )
        right_ever = any(
            self.finger_contact_links[1] in frame["robot_cup_contact_links"]
            for frame in self.frames
        )
        dual_count = sum(
            set(self.finger_contact_links).issubset(frame["robot_cup_contact_links"])
            for frame in self.frames
        )
        carry_frames = [
            frame
            for frame in self.frames
            if frame.get("phase") in {"lift", "orient", "transfer", "lower"}
        ]
        dual_carry_count = sum(
            set(self.finger_contact_links).issubset(frame["robot_cup_contact_links"])
            for frame in carry_frames
        )
        dual_carry_fraction = (
            dual_carry_count / len(carry_frames) if carry_frames else 0.0
        )

        finger_names = [
            p.getJointInfo(self.panda, link)[12].decode("utf-8")
            for link in self.finger_contact_links
        ]
        opposed_geometry = dict(
            finger_names=finger_names, finger_links=self.finger_contact_links
        )
        opposed_margins = [
            opposed_projection_margin(frame, **opposed_geometry)
            for frame in carry_frames
        ]
        normal_margins = [
            opposed_normal_margin(frame, **opposed_geometry)
            for frame in carry_frames
        ]
        opposed_threshold_m = 0.004
        opposed_normal_threshold = 0.5
        geometric_flags = [
            margin >= opposed_threshold_m for margin in opposed_margins
        ]
        opposed_flags = [
            geometric or normal >= opposed_normal_threshold
            for geometric, normal in zip(geometric_flags, normal_margins)
        ]
        opposed_carry_count = sum(opposed_flags)
        opposed_geometric_count = sum(geometric_flags)
        opposed_carry_fraction = (
            opposed_carry_count / len(carry_frames) if carry_frames else 0.0
        )
        opposed_geometric_fraction = (
            opposed_geometric_count / len(carry_frames) if carry_frames else 0.0
        )
        finite_opposed_margins = [
            margin for margin in opposed_margins if np.isfinite(margin)
        ]
        minimum_opposed_margin = (
            min(finite_opposed_margins) if finite_opposed_margins else float("-inf")
        )
        finite_normal_margins = [
            margin for margin in normal_margins if np.isfinite(margin)
        ]
        minimum_opposed_normal_margin = (
            min(finite_normal_margins) if finite_normal_margins else float("-inf")
        )
        object_attachment_constraint_count = 0
        gripper_coupling_constraint_count = 0
        for index in range(p.getNumConstraints()):
            constraint_id = p.getConstraintUniqueId(index)
            info = p.getConstraintInfo(constraint_id)
            parent_body, child_body = int(info[0]), int(info[2])
            if any(
                body in {parent_body, child_body}
                for object_id, body in self.objects.items()
                if self.object_specs[object_id].role == "manipuland"
            ):
                object_attachment_constraint_count += 1
            if parent_body == self.panda and child_body == self.panda:
                gripper_coupling_constraint_count += 1
        cup_lift = self.max_cup_z - self.initial_cup_z
        push_frames = [
            frame for frame in self.frames if frame.get("phase") == "push"
        ]
        push_contact_count = sum(
            bool(frame.get("robot_cup_contact_links")) for frame in push_frames
        )
        target_contact_frame_count = sum(
            bool(frame.get("subject_target_contact")) for frame in self.frames
        )
        subject_horizontal_displacement = float(
            np.linalg.norm(np.asarray(cup_position[:2]) - self.initial_cup_xy)
        )
        common_manipulation_checks = (
            cup_released
            and object_attachment_constraint_count == 0
            and gripper_coupling_constraint_count
            == self.expected_gripper_coupling_constraint_count
        )
        if self.active_operator == "push":
            manipulation_passed = (
                common_manipulation_checks
                and push_contact_count > 0
                and subject_horizontal_displacement > 0.02
                and cup_lift < 0.05
            )
        else:
            manipulation_passed = (
                common_manipulation_checks
                and left_ever
                and right_ever
                and dual_count > 0
                and opposed_carry_fraction == 1.0
                and cup_lift >= 0.12
            )
        predicate_checks = {
            "in_container": (
                cup_inside_tray_xy
                and cup_in_tray_contact
                and subject_bottom_z < target_top_z + 0.02
            ),
            "stacked": (
                cup_fully_over_target_xy
                and cup_in_tray_contact
                and abs(subject_bottom_z - target_top_z) <= 0.025
                and upright_cosine >= 0.85
            ),
            "inserted": (
                inserted_geometry_passed(
                    center_inside_target_xy=cup_inside_tray_xy,
                    target_contact=cup_in_tray_contact,
                    subject_bottom_z_m=subject_bottom_z,
                    subject_top_z_m=subject_top_z,
                    target_bottom_z_m=target_bottom_z,
                    target_top_z_m=target_top_z,
                    alignment_cosine=subject_alignment_cosine,
                )
            ),
            "covering": (
                cup_fully_over_target_xy
                and cup_in_tray_contact
                and abs(subject_bottom_z - target_top_z) <= 0.030
                and subject_alignment_cosine >= 0.85
            ),
            "on_region": (
                cup_inside_tray_xy
                and cup_released
                and upright_cosine >= 0.80
            ),
            "contact": target_contact_frame_count > 0,
        }
        predicate_evidence = {
            "subject_aabb_world": [list(cup_aabb[0]), list(cup_aabb[1])],
            "target_aabb_world": [list(tray_aabb[0]), list(tray_aabb[1])],
            "subject_center_inside_target_xy": cup_inside_tray_xy,
            "subject_fully_over_target_xy": cup_fully_over_target_xy,
            "subject_target_contact": cup_in_tray_contact,
            "target_contact_frame_count": target_contact_frame_count,
            "subject_released": cup_released,
            "subject_bottom_z_m": subject_bottom_z,
            "subject_top_z_m": subject_top_z,
            "target_bottom_z_m": target_bottom_z,
            "target_top_z_m": target_top_z,
            "insertion_depth_m": insertion_depth_m,
            "subject_height_m": subject_height_m,
            "subject_upright_cosine": upright_cosine,
            "subject_alignment_axis_local": alignment_axis_local.tolist(),
            "subject_alignment_axis_world": alignment_axis_world.tolist(),
            "subject_alignment_cosine": subject_alignment_cosine,
            "subject_horizontal_displacement_m": subject_horizontal_displacement,
            "predicate_passed": bool(predicate_checks[self.active_success_predicate]),
        }
        success = manipulation_passed and predicate_evidence["predicate_passed"]
        result = PolicySceneResult(
            trajectory_path=str(trajectory_path),
            frame_count=len(self.frames),
            simulation_seconds=len(self.frames) / self.control_hz,
            grasp_contact_left_ever=left_ever,
            grasp_contact_right_ever=right_ever,
            dual_finger_contact_frame_count=dual_count,
            carry_frame_count=len(carry_frames),
            dual_finger_supported_carry_fraction=dual_carry_fraction,
            opposed_external_carry_frame_count=opposed_carry_count,
            opposed_external_carry_fraction=opposed_carry_fraction,
            minimum_opposed_contact_margin_m=minimum_opposed_margin,
            opposed_external_geometric_fraction=opposed_geometric_fraction,
            minimum_opposed_normal_margin=minimum_opposed_normal_margin,
            cup_lift_m=cup_lift,
            cup_inside_tray_xy=cup_inside_tray_xy,
            cup_in_tray_contact=cup_in_tray_contact,
            cup_released=cup_released,
            object_attachment_constraint_count=object_attachment_constraint_count,
            gripper_coupling_constraint_count=gripper_coupling_constraint_count,
            expected_gripper_coupling_constraint_count=(
                self.expected_gripper_coupling_constraint_count
            ),
            joint_limit_clamp_count=self.joint_limit_clamp_count,
            robot_embodiment=self.inputs.robot_embodiment,
            operator=self.active_operator,
            success_predicate=self.active_success_predicate,
            push_contact_frame_count=push_contact_count,
            subject_horizontal_displacement_m=subject_horizontal_displacement,
            predicate_evidence=predicate_evidence,
            success=success,
        )
        (self.output_dir / "result.json").write_text(
            json_evidence.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return result
