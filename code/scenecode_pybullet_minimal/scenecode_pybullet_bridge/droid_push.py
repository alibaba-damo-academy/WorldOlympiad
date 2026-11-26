"""Deterministic reusable DROID push-to-region reference operator."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json

from . import json_evidence
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .droid_pick_place import OFFICIAL_DROID_FORCES, OFFICIAL_DROID_RESET
from .policy_scene import PolicySceneInputs, PyBulletDroidPolicyScene


DROID_GRIPPER_CENTER_BELOW_BASE_M = 0.1436321380
PUSH_PHASE_ACTION_COUNTS = {
    "settle": 15,
    "close": 12,
    "approach": 30,
    "descend": 24,
    "precontact": 20,
    "push": 60,
    "retreat_vertical": 20,
    "open": 6,
    "retreat_back": 16,
    "final_settle": 30,
}


@dataclass(frozen=True)
class DroidPushInputs:
    table_sdf: Path
    subject_sdf: Path
    target_sdf: Path
    panda_urdf: Path
    output_dir: Path
    subject_center_xy: tuple[float, float] = (0.18, -0.03)
    target_center_xy: tuple[float, float] = (-0.16, 0.08)
    subject_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    target_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    subject_lateral_friction: float = 0.35


@dataclass(frozen=True)
class DroidPushResult:
    trajectory_path: str
    actions_path: str
    certificate_path: str
    frame_count: int
    simulation_seconds: float
    push_contact_frame_count: int
    subject_horizontal_displacement_m: float
    simulator_result: dict[str, Any]
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def push_waypoints_from_aabbs(
    *,
    subject_aabb: Sequence[Sequence[float]],
    target_aabb: Sequence[Sequence[float]],
) -> dict[str, np.ndarray | float]:
    """Derive a straight push corridor from runtime subject/target geometry."""

    subject = np.asarray(subject_aabb, dtype=np.float64)
    target = np.asarray(target_aabb, dtype=np.float64)
    if subject.shape != (2, 3) or target.shape != (2, 3):
        raise ValueError("push geometry requires two 3D AABBs")
    if not np.isfinite(subject).all() or not np.isfinite(target).all():
        raise ValueError("push geometry must be finite")
    if np.any(subject[1] <= subject[0]) or np.any(target[1] <= target[0]):
        raise ValueError("push geometry is degenerate")
    subject_center = (subject[0] + subject[1]) / 2.0
    target_center = (target[0] + target[1]) / 2.0
    delta = target_center[:2] - subject_center[:2]
    distance = float(np.linalg.norm(delta))
    if distance <= 0.05:
        raise ValueError("push subject and target are not spatially distinct")
    direction = delta / distance
    half_extent = float(
        np.dot(np.abs(direction), (subject[1, :2] - subject[0, :2]) / 2.0)
    )
    contact_xy = subject_center[:2] - direction * (half_extent + 0.004)
    precontact_xy = contact_xy - direction * 0.08
    finish_xy = target_center[:2] - direction * (half_extent + 0.003)
    push_z = float(subject_center[2] + DROID_GRIPPER_CENTER_BELOW_BASE_M)
    return {
        "direction_xy": direction,
        "precontact_xy": precontact_xy,
        "contact_xy": contact_xy,
        "finish_xy": finish_xy,
        "push_base_z_m": push_z,
        "subject_half_extent_along_push_m": half_extent,
    }


def run_droid_push(inputs: DroidPushInputs) -> DroidPushResult:
    """Execute one straight, contact-driven push and terminate after settling."""

    output = inputs.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite DROID push rollout: {output}")
    if not math.isfinite(inputs.subject_lateral_friction) or not (
        0.0 <= inputs.subject_lateral_friction <= 2.0
    ):
        raise ValueError("subject lateral friction must lie in [0, 2]")
    scene_inputs = PolicySceneInputs(
        table_sdf=inputs.table_sdf.resolve(),
        cup_sdf=inputs.subject_sdf.resolve(),
        tray_sdf=inputs.target_sdf.resolve(),
        panda_urdf=inputs.panda_urdf.resolve(),
        output_dir=output,
        robot_embodiment="robolab_droid",
        initial_arm_joint_positions=OFFICIAL_DROID_RESET,
        disable_robot_gravity=True,
        arm_position_gain=0.5,
        arm_joint_forces=OFFICIAL_DROID_FORCES,
        action_authority="deterministic Cartesian push operator in PyBullet",
        cup_center_xy=tuple(float(value) for value in inputs.subject_center_xy),
        tray_center_xy=tuple(float(value) for value in inputs.target_center_xy),
        cup_orientation_xyzw=tuple(
            float(value) for value in inputs.subject_orientation_xyzw
        ),
        tray_orientation_xyzw=tuple(
            float(value) for value in inputs.target_orientation_xyzw
        ),
        success_predicate="on_region",
        operator="push",
        subject_lateral_friction=inputs.subject_lateral_friction,
        target_collision_enabled=False,
    )
    actions: list[list[float]] = []
    action_index = 0

    with PyBulletDroidPolicyScene(scene_inputs) as scene:
        p = scene.p
        geometry = push_waypoints_from_aabbs(
            subject_aabb=p.getAABB(scene.cup), target_aabb=p.getAABB(scene.tray)
        )
        direction = np.asarray(geometry["direction_xy"], dtype=np.float64)
        initial_orientation = p.getLinkState(
            scene.panda, scene.end_effector, computeForwardKinematics=True
        )[5]
        yaw = math.atan2(float(direction[1]), float(direction[0])) + math.pi / 2.0
        fixed_orientation = p.multiplyTransforms(
            (0.0, 0.0, 0.0),
            p.getQuaternionFromEuler((0.0, 0.0, yaw)),
            (0.0, 0.0, 0.0),
            initial_orientation,
        )[1]

        def execute(arm: Sequence[float], gripper: float, phase: str) -> None:
            nonlocal action_index
            action = [*(float(value) for value in arm), float(gripper)]
            scene.execute_action(
                action,
                source_window=-1,
                source_action_index=action_index,
                phase_override=phase,
            )
            actions.append(action)
            action_index += 1

        def hold(count: int, gripper: float, phase: str) -> None:
            target = scene.joint_position().astype(np.float64)
            for _ in range(count):
                execute(target, gripper, phase)

        def move(target: Sequence[float], count: int, gripper: float, phase: str) -> None:
            start = np.asarray(
                p.getLinkState(
                    scene.panda, scene.end_effector, computeForwardKinematics=True
                )[4],
                dtype=np.float64,
            )
            destination = np.asarray(target, dtype=np.float64)
            if destination.shape != (3,) or not np.isfinite(destination).all():
                raise ValueError(f"invalid push Cartesian target: {destination}")
            for local_index in range(count):
                alpha = (local_index + 1) / count
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                point = start + (destination - start) * smooth
                solution = p.calculateInverseKinematics(
                    scene.panda,
                    scene.end_effector,
                    point.tolist(),
                    fixed_orientation,
                    maxNumIterations=300,
                    residualThreshold=1e-6,
                )
                if len(solution) < 7 or not np.isfinite(solution[:7]).all():
                    raise RuntimeError(f"PyBullet IK failed for {phase}: {solution}")
                execute(solution[:7], gripper, phase)

        pre = np.asarray(geometry["precontact_xy"], dtype=np.float64)
        contact = np.asarray(geometry["contact_xy"], dtype=np.float64)
        finish = np.asarray(geometry["finish_xy"], dtype=np.float64)
        push_z = float(geometry["push_base_z_m"])
        high_z = push_z + 0.14
        hold(PUSH_PHASE_ACTION_COUNTS["settle"], 0.0, "settle")
        hold(PUSH_PHASE_ACTION_COUNTS["close"], 1.0, "close")
        move((*pre, high_z), PUSH_PHASE_ACTION_COUNTS["approach"], 1.0, "approach")
        move((*pre, push_z), PUSH_PHASE_ACTION_COUNTS["descend"], 1.0, "descend")
        move((*contact, push_z), PUSH_PHASE_ACTION_COUNTS["precontact"], 1.0, "precontact")
        move((*finish, push_z), PUSH_PHASE_ACTION_COUNTS["push"], 1.0, "push")
        move((*finish, high_z), PUSH_PHASE_ACTION_COUNTS["retreat_vertical"], 1.0, "retreat_vertical")
        hold(PUSH_PHASE_ACTION_COUNTS["open"], 0.0, "open")
        back = finish - direction * 0.10
        move((*back, high_z), PUSH_PHASE_ACTION_COUNTS["retreat_back"], 0.0, "retreat_back")
        hold(PUSH_PHASE_ACTION_COUNTS["final_settle"], 0.0, "final_settle")
        simulator_result = scene.write_outputs()

    actions_path = output / "expert_actions.npy"
    np.save(actions_path, np.asarray(actions, dtype=np.float32), allow_pickle=False)
    trajectory_path = Path(simulator_result.trajectory_path)
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    phase_counts = Counter(frame["phase"] for frame in trajectory["frames"])
    success = bool(
        simulator_result.success
        and simulator_result.push_contact_frame_count > 0
        and simulator_result.subject_horizontal_displacement_m > 0.02
    )
    certificate = {
        "schema_version": "scenecode_pybullet_droid_push_certificate.v1",
        "operator": "push",
        "success_predicate": "on_region",
        "frame_count": len(trajectory["frames"]),
        "fps": PyBulletDroidPolicyScene.control_hz,
        "duration_sec": len(trajectory["frames"]) / PyBulletDroidPolicyScene.control_hz,
        "phase_action_counts": PUSH_PHASE_ACTION_COUNTS,
        "recorded_phase_counts": dict(sorted(phase_counts.items())),
        "runtime_push_geometry": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in geometry.items()
        },
        "push_contact_frame_count": simulator_result.push_contact_frame_count,
        "subject_horizontal_displacement_m": (
            simulator_result.subject_horizontal_displacement_m
        ),
        "simulation_result": simulator_result.to_dict(),
        "trajectory_sha256": _sha256(trajectory_path),
        "actions_sha256": _sha256(actions_path),
        "success": success,
    }
    certificate_path = output / "expert_certificate.json"
    certificate_path.write_text(
        json_evidence.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return DroidPushResult(
        trajectory_path=str(trajectory_path),
        actions_path=str(actions_path),
        certificate_path=str(certificate_path),
        frame_count=len(trajectory["frames"]),
        simulation_seconds=len(trajectory["frames"]) / PyBulletDroidPolicyScene.control_hz,
        push_contact_frame_count=simulator_result.push_contact_frame_count,
        subject_horizontal_displacement_m=(
            simulator_result.subject_horizontal_displacement_m
        ),
        simulator_result=simulator_result.to_dict(),
        success=success,
    )
