"""Deterministic PyBullet pick-and-place using official SceneCode assets.

PyBullet is the state authority.  The output trajectory contains every state
needed to reproduce the motion with the official SceneCode rendering path.
"""

from __future__ import annotations

import json

from . import json_evidence
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .sdf import prepare_physics_sdf


@dataclass(frozen=True)
class PickPlaceInputs:
    table_sdf: Path
    cup_sdf: Path
    tray_sdf: Path
    panda_urdf: Path
    output_dir: Path


@dataclass(frozen=True)
class PickPlaceResult:
    trajectory_path: str
    frame_count: int
    simulation_seconds: float
    grasp_contact_left: bool
    grasp_contact_right: bool
    carry_sample_count: int
    dual_finger_supported_carry_sample_count: int
    dual_finger_supported_carry_fraction: float
    object_attachment_constraint_count: int
    gripper_coupling_constraint_count: int
    cup_lift_m: float
    cup_inside_tray_xy: bool
    cup_in_tray_contact: bool
    cup_released: bool
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _quat_wxyz_to_xyzw(value: Sequence[float]) -> tuple[float, float, float, float]:
    w, x, y, z = value
    return x, y, z, w


def run_pick_place(inputs: PickPlaceInputs) -> PickPlaceResult:
    """Run a cup-to-tray task and emit an auditable state trajectory.

    SceneCode source files are never modified.  The table is welded to the
    world, matching SceneCode's ``house_furniture_welded.dmd.yaml`` semantics.
    The cup and tray retain their generated mass, inertia and collision meshes.
    """

    try:
        import pybullet as p
        import pybullet_data
    except ImportError as exc:  # pragma: no cover - exercised on the server
        raise RuntimeError("run this command in an environment with pybullet") from exc

    output_dir = inputs.output_dir.resolve()
    physics_dir = output_dir / "physics_sdf"
    frames_path = output_dir / "trajectory.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = {}
    for name, source in (
        ("table", inputs.table_sdf),
        ("cup", inputs.cup_sdf),
        ("tray", inputs.tray_sdf),
    ):
        artifact = prepare_physics_sdf(source, physics_dir / f"{name}.sdf")
        prepared[name] = artifact

    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("failed to connect to PyBullet in DIRECT mode")

    hz = 240
    sample_hz = 15
    sample_every = hz // sample_hz
    trajectory: list[dict[str, Any]] = []
    sim_step = 0
    phase = "setup"

    try:
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(1.0 / hz)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / hz,
            numSolverIterations=200,
            deterministicOverlappingPairs=1,
        )
        p.loadURDF("plane.urdf")

        table = p.loadSDF(prepared["table"].physics_sdf)[0]
        cup = p.loadSDF(prepared["cup"].physics_sdf)[0]
        tray = p.loadSDF(prepared["tray"].physics_sdf)[0]
        panda = p.loadURDF(
            str(inputs.panda_urdf.resolve()),
            basePosition=(0.0, 0.0, 0.0),
            baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, math.pi / 2.0)),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        # ``loadSDF`` reports the base pose at the center of mass, not at the
        # SDF model origin.  Resetting that COM to (0, 0, 0) would sink this
        # table by its inertial offset.  Keep the generated SDF placement.
        p.changeDynamics(table, -1, mass=0.0, lateralFriction=0.8)

        def place_aabb(body: int, center_xy: Sequence[float], bottom_z: float) -> None:
            """Translate an SDF body using geometry bounds while preserving COM offset."""

            position, orientation = p.getBasePositionAndOrientation(body)
            lower, upper = p.getAABB(body)
            current_center = ((lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2)
            delta = (
                center_xy[0] - current_center[0],
                center_xy[1] - current_center[1],
                bottom_z - lower[2],
            )
            p.resetBasePositionAndOrientation(
                body,
                tuple(position[i] + delta[i] for i in range(3)),
                orientation,
            )

        table_top = p.getAABB(table)[1][2]
        # The bundled Panda is a table-mounted arm.  Mount it at the near edge
        # instead of asking a sub-meter arm to reach a 0.75 m table from the floor.
        panda_base = (0.0, -0.43, table_top + 0.002)
        panda_orientation = p.getQuaternionFromEuler((0.0, 0.0, math.pi / 2.0))
        p.resetBasePositionAndOrientation(panda, panda_base, panda_orientation)
        place_aabb(cup, (0.24, -0.08), table_top + 0.002)
        place_aabb(tray, (-0.18, 0.10), table_top + 0.002)
        p.changeDynamics(
            cup,
            -1,
            lateralFriction=1.2,
            spinningFriction=0.02,
            rollingFriction=0.002,
            restitution=0.0,
            # Keep the physics surface registered to the SceneCode mesh.
            # A positive mesh margin leaves a visible air gap in Blender even
            # while Bullet reports contact.
            collisionMargin=0.0,
        )
        p.changeDynamics(
            tray,
            -1,
            lateralFriction=0.9,
            spinningFriction=0.02,
            rollingFriction=0.002,
            restitution=0.0,
        )

        movable_joints: list[int] = []
        arm_joints: list[int] = []
        finger_joints: list[int] = []
        joint_names: list[str] = []
        for joint_index in range(p.getNumJoints(panda)):
            info = p.getJointInfo(panda, joint_index)
            joint_type = info[2]
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                movable_joints.append(joint_index)
                joint_names.append(info[1].decode("utf-8"))
            if info[1].decode("utf-8").startswith("panda_joint"):
                arm_joints.append(joint_index)
            if "finger_joint" in info[1].decode("utf-8"):
                finger_joints.append(joint_index)

        arm_joints = arm_joints[:7]
        if len(arm_joints) != 7 or len(finger_joints) != 2:
            raise RuntimeError(
                f"unexpected Panda layout: arm={arm_joints}, fingers={finger_joints}"
            )

        neutral = (0.0, -0.45, 0.0, -2.25, 0.0, 1.85, 0.8)
        for joint, position in zip(arm_joints, neutral, strict=True):
            p.resetJointState(panda, joint, position)
        for joint in finger_joints:
            p.resetJointState(panda, joint, 0.04)
            p.changeDynamics(panda, joint, lateralFriction=2.0, collisionMargin=0.0)

        # The URDF describes two independent fingers; this gear constraint is
        # PyBullet's standard mirrored-gripper coupling for the bundled Panda.
        gear = p.createConstraint(
            panda,
            finger_joints[0],
            panda,
            finger_joints[1],
            jointType=p.JOINT_GEAR,
            jointAxis=(1, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0),
        )
        p.changeConstraint(gear, gearRatio=-1, erp=0.1, maxForce=50)

        end_effector = 11
        down = p.getQuaternionFromEuler((math.pi, 0.0, 0.0))
        cup_z_initial = p.getBasePositionAndOrientation(cup)[0][2]
        max_cup_z = cup_z_initial
        carry_sample_count = 0
        dual_finger_supported_carry_sample_count = 0

        def record(force: bool = False) -> None:
            nonlocal carry_sample_count, dual_finger_supported_carry_sample_count
            if not force and sim_step % sample_every:
                return
            cup_pose = p.getBasePositionAndOrientation(cup)
            tray_pose = p.getBasePositionAndOrientation(tray)

            def link_pose(body: int) -> tuple[Sequence[float], Sequence[float]]:
                """Convert PyBullet's world-COM pose to the SDF link-frame pose."""

                com_position, com_orientation = p.getBasePositionAndOrientation(body)
                dynamics = p.getDynamicsInfo(body, -1)
                inertial_position, inertial_orientation = dynamics[3], dynamics[4]
                link_from_inertial = p.invertTransform(
                    inertial_position, inertial_orientation
                )
                return p.multiplyTransforms(
                    com_position,
                    com_orientation,
                    link_from_inertial[0],
                    link_from_inertial[1],
                )

            cup_link_pose = link_pose(cup)
            tray_link_pose = link_pose(tray)
            panda_link_pose = link_pose(panda)
            robot_positions = [p.getJointState(panda, j)[0] for j in movable_joints]
            ee_pose = p.getLinkState(
                panda, end_effector, computeForwardKinematics=True
            )[4:6]
            contacts = p.getContactPoints(bodyA=panda, bodyB=cup)
            contact_links = sorted({int(c[3]) for c in contacts})
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
                if int(contact[3]) in finger_joints
            ]
            if phase in {"lift", "transfer", "lower"}:
                carry_sample_count += 1
                if set(finger_joints).issubset(contact_links):
                    dual_finger_supported_carry_sample_count += 1
            finger_link_positions = {
                p.getJointInfo(panda, joint)[12].decode("utf-8"): list(
                    p.getLinkState(panda, joint, computeForwardKinematics=True)[4]
                )
                for joint in finger_joints
            }
            trajectory.append(
                {
                    "frame": len(trajectory),
                    "sim_step": sim_step,
                    "time_s": sim_step / hz,
                    "phase": phase,
                    "robot": {
                        # resetBasePositionAndOrientation uses the base COM
                        # pose, while Drake welds the URDF link frame.  The
                        # Panda link0 inertial offset is 5 cm, so replaying the
                        # COM as a link pose visibly floats the whole gripper.
                        "base_position": list(panda_link_pose[0]),
                        "base_orientation_xyzw": list(panda_link_pose[1]),
                        "joint_names": joint_names,
                        "joint_positions": robot_positions,
                        "end_effector_position": list(ee_pose[0]),
                        "end_effector_orientation_xyzw": list(ee_pose[1]),
                        "finger_link_positions": finger_link_positions,
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
                    "robot_cup_contact_links": contact_links,
                    "robot_cup_contact_evidence": contact_evidence,
                }
            )

        def step(count: int) -> None:
            nonlocal sim_step, max_cup_z
            for _ in range(count):
                p.stepSimulation()
                sim_step += 1
                max_cup_z = max(max_cup_z, p.getBasePositionAndOrientation(cup)[0][2])
                record()

        def command_gripper(opening: float, seconds: float) -> None:
            for joint in finger_joints:
                p.setJointMotorControl2(
                    panda,
                    joint,
                    p.POSITION_CONTROL,
                    targetPosition=opening,
                    force=120,
                    positionGain=0.3,
                    velocityGain=1.0,
                )
            step(round(seconds * hz))

        def move_ee(target: Sequence[float], seconds: float) -> None:
            start = p.getLinkState(panda, end_effector, computeForwardKinematics=True)[4]
            count = max(1, round(seconds * hz))
            for index in range(count):
                alpha = (index + 1) / count
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                point = [
                    float(start[axis] + (target[axis] - start[axis]) * smooth)
                    for axis in range(3)
                ]
                solution = p.calculateInverseKinematics(
                    panda,
                    end_effector,
                    point,
                    down,
                    maxNumIterations=200,
                    residualThreshold=1e-5,
                )
                for joint, position in zip(arm_joints, solution[:7], strict=True):
                    p.setJointMotorControl2(
                        panda,
                        joint,
                        p.POSITION_CONTROL,
                        targetPosition=position,
                        force=300,
                        positionGain=0.18,
                        velocityGain=1.0,
                    )
                step(1)

        phase = "settle"
        command_gripper(0.04, 1.0)
        step(hz)
        cup_z_initial = p.getBasePositionAndOrientation(cup)[0][2]
        max_cup_z = cup_z_initial

        cup_xy = p.getBasePositionAndOrientation(cup)[0][:2]
        phase = "approach"
        move_ee((cup_xy[0], cup_xy[1], 1.02), 1.4)
        phase = "descend"
        move_ee((cup_xy[0], cup_xy[1], 0.825), 1.2)
        phase = "grasp"
        command_gripper(0.0, 1.2)

        left_contact = bool(
            p.getContactPoints(bodyA=panda, bodyB=cup, linkIndexA=finger_joints[0])
        )
        right_contact = bool(
            p.getContactPoints(bodyA=panda, bodyB=cup, linkIndexA=finger_joints[1])
        )

        phase = "lift"
        move_ee((cup_xy[0], cup_xy[1], 1.02), 1.4)
        tray_xy = p.getBasePositionAndOrientation(tray)[0][:2]
        phase = "transfer"
        move_ee((tray_xy[0], tray_xy[1], 1.02), 1.8)
        phase = "lower"
        move_ee((tray_xy[0], tray_xy[1], 0.86), 1.0)
        phase = "release"
        command_gripper(0.04, 1.5)
        phase = "retreat_vertical"
        move_ee((tray_xy[0], tray_xy[1], 1.06), 1.0)
        phase = "retreat_back"
        move_ee((tray_xy[0], tray_xy[1] - 0.10, 1.06), 0.8)
        phase = "final_settle"
        step(round(1.5 * hz))
        record(force=True)

        cup_position = p.getBasePositionAndOrientation(cup)[0]
        tray_position = p.getBasePositionAndOrientation(tray)[0]
        cup_inside_tray_xy = (
            abs(cup_position[0] - tray_position[0]) < 0.085
            and abs(cup_position[1] - tray_position[1]) < 0.13
        )
        cup_released = not p.getContactPoints(bodyA=panda, bodyB=cup)
        cup_in_tray_contact = bool(p.getContactPoints(bodyA=cup, bodyB=tray))
        cup_lift = max_cup_z - cup_z_initial
        object_attachment_constraint_count = 0
        gripper_coupling_constraint_count = 0
        for index in range(p.getNumConstraints()):
            constraint_id = p.getConstraintUniqueId(index)
            info = p.getConstraintInfo(constraint_id)
            parent_body, child_body = int(info[0]), int(info[2])
            if cup in {parent_body, child_body}:
                object_attachment_constraint_count += 1
            if parent_body == panda and child_body == panda:
                gripper_coupling_constraint_count += 1
        dual_finger_supported_carry_fraction = (
            dual_finger_supported_carry_sample_count / carry_sample_count
            if carry_sample_count
            else 0.0
        )
        success = (
            left_contact
            and right_contact
            and cup_lift >= 0.12
            and cup_inside_tray_xy
            and cup_in_tray_contact
            and cup_released
            and object_attachment_constraint_count == 0
            and gripper_coupling_constraint_count == 1
            and dual_finger_supported_carry_fraction == 1.0
        )

        manifest = {
            "schema_version": "scenecode_pybullet_pick_place_v1",
            "authority": {
                "physics": "PyBullet",
                "visual_assets": "official SceneCode SDF/glTF",
                "robot": "PyBullet bundled Franka Panda (external to SceneCode)",
            },
            "physics_hz": hz,
            "sample_hz": sample_hz,
            "prepared_assets": {k: v.to_dict() for k, v in prepared.items()},
            "frames": trajectory,
        }
        frames_path.write_text(json_evidence.dumps(manifest, indent=2), encoding="utf-8")
        result = PickPlaceResult(
            trajectory_path=str(frames_path),
            frame_count=len(trajectory),
            simulation_seconds=sim_step / hz,
            grasp_contact_left=left_contact,
            grasp_contact_right=right_contact,
            carry_sample_count=carry_sample_count,
            dual_finger_supported_carry_sample_count=(
                dual_finger_supported_carry_sample_count
            ),
            dual_finger_supported_carry_fraction=dual_finger_supported_carry_fraction,
            object_attachment_constraint_count=object_attachment_constraint_count,
            gripper_coupling_constraint_count=gripper_coupling_constraint_count,
            cup_lift_m=cup_lift,
            cup_inside_tray_xy=cup_inside_tray_xy,
            cup_in_tray_contact=cup_in_tray_contact,
            cup_released=cup_released,
            success=success,
        )
        (output_dir / "result.json").write_text(
            json_evidence.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return result
    finally:
        p.disconnect(client)
