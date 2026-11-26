"""Execute a reusable sequence of DROID pick-plus-terminal operator steps."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json

from . import json_evidence
import math
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from .droid_pick_place import (
    ALIGNMENT_ACTION_COUNT,
    OFFICIAL_DROID_FORCES,
    OFFICIAL_DROID_RESET,
    PHASE_ACTION_COUNTS,
    STACK_RELEASE_CLEARANCE_M,
    axis_alignment_quaternion_xyzw,
    compensated_place_end_effector_target,
    stack_subject_link_target_from_aabbs,
)
from .policy_scene import (
    PolicySceneInputs,
    PolicySceneObject,
    PyBulletDroidPolicyScene,
)


TERMINAL_OPERATORS = {"place", "stack", "insert", "cover"}
TERMINAL_PREDICATES = {
    "place": {"in_container", "on_region"},
    "stack": {"stacked"},
    "insert": {"inserted"},
    "cover": {"covering"},
}


@dataclass(frozen=True)
class DroidGraphObject:
    object_id: str
    sdf_path: Path
    center_xy: tuple[float, float]
    role: str
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    lateral_friction: float = 1.2
    collision_enabled: bool = True
    fixed: bool = False


@dataclass(frozen=True)
class DroidPickTerminalStep:
    subject_id: str
    target_id: str
    terminal_operator: str
    success_predicate: str
    grasp_base_offset_z_m: float
    hover_base_offset_z_m: float
    lift_base_offset_z_m: float
    place_target_offset_xy_m: tuple[float, float] = (0.0, 0.0)
    place_target_offset_z_m: float = 0.0
    grasp_yaw_deg: float = 90.0
    terminal_alignment_axis_local: tuple[float, float, float] | None = None
    terminal_alignment_axis_world: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class DroidOperatorGraphInputs:
    table_sdf: Path
    panda_urdf: Path
    output_dir: Path
    objects: tuple[DroidGraphObject, ...]
    steps: tuple[DroidPickTerminalStep, ...]


@dataclass(frozen=True)
class DroidOperatorGraphResult:
    trajectory_path: str
    actions_path: str
    certificate_path: str
    scene_objects_manifest_path: str
    frame_count: int
    simulation_seconds: float
    step_count: int
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


def _validate_inputs(inputs: DroidOperatorGraphInputs) -> dict[str, DroidGraphObject]:
    if len(inputs.steps) < 2:
        raise ValueError("operator-graph executor requires at least two terminal steps")
    objects: dict[str, DroidGraphObject] = {}
    for item in inputs.objects:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", item.object_id) is None:
            raise ValueError("operator-graph object id is unsafe")
        if item.object_id in objects:
            raise ValueError("operator-graph object ids must be unique")
        if item.role not in {"manipuland", "destination"}:
            raise ValueError("operator-graph object role is unsupported")
        center = np.asarray(item.center_xy, dtype=np.float64)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("operator-graph object center must be a finite XY pair")
        orientation = np.asarray(item.orientation_xyzw, dtype=np.float64)
        if (
            orientation.shape != (4,)
            or not np.isfinite(orientation).all()
            or not np.isclose(np.linalg.norm(orientation), 1.0, atol=1.0e-6)
        ):
            raise ValueError("operator-graph object orientation must be normalized")
        objects[item.object_id] = item
    for step in inputs.steps:
        if step.subject_id not in objects or step.target_id not in objects:
            raise ValueError("operator step references an unknown object")
        if step.subject_id == step.target_id:
            raise ValueError("operator step subject and target must differ")
        if objects[step.subject_id].role != "manipuland":
            raise ValueError("operator step subject must be a manipuland")
        if objects[step.target_id].role != "destination":
            raise ValueError("operator step target must be a destination")
        if step.terminal_operator not in TERMINAL_OPERATORS:
            raise ValueError("operator graph contains an unsupported terminal operator")
        if step.success_predicate not in TERMINAL_PREDICATES[step.terminal_operator]:
            raise ValueError("terminal operator and physical predicate disagree")
        offsets = np.asarray(
            [
                step.grasp_base_offset_z_m,
                step.hover_base_offset_z_m,
                step.lift_base_offset_z_m,
                step.place_target_offset_z_m,
                step.grasp_yaw_deg,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(offsets).all() or not (
            0.12
            <= step.grasp_base_offset_z_m
            < step.hover_base_offset_z_m
            <= step.lift_base_offset_z_m
            <= 0.55
        ):
            raise ValueError("operator step execution geometry is invalid")
        if not -0.10 <= step.place_target_offset_z_m <= 0.50:
            raise ValueError("operator step target offset is invalid")
        target_offset_xy = np.asarray(
            step.place_target_offset_xy_m, dtype=np.float64
        )
        if target_offset_xy.shape != (2,) or not np.isfinite(target_offset_xy).all():
            raise ValueError("operator step target XY offset is invalid")
        if abs(step.grasp_yaw_deg) > 180.0:
            raise ValueError("operator step grasp yaw is invalid")
        alignment_requested = (
            step.terminal_alignment_axis_local is not None
            or step.terminal_alignment_axis_world is not None
        )
        if alignment_requested and (
            step.terminal_alignment_axis_local is None
            or step.terminal_alignment_axis_world is None
        ):
            raise ValueError("operator step terminal alignment is incomplete")
        if alignment_requested:
            axis_alignment_quaternion_xyzw(
                step.terminal_alignment_axis_local,
                step.terminal_alignment_axis_world,
            )
    return objects


def run_droid_operator_graph(
    inputs: DroidOperatorGraphInputs,
) -> DroidOperatorGraphResult:
    """Run every compiled step in one persistent scene and one trajectory."""

    output = inputs.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite DROID operator graph: {output}")
    objects = _validate_inputs(inputs)
    first = inputs.steps[0]
    first_subject = objects[first.subject_id]
    first_target = objects[first.target_id]
    additional = tuple(
        PolicySceneObject(
            object_id=item.object_id,
            sdf_path=item.sdf_path,
            center_xy=item.center_xy,
            role=item.role,
            orientation_xyzw=item.orientation_xyzw,
            lateral_friction=item.lateral_friction,
            collision_enabled=item.collision_enabled,
            fixed=item.fixed,
        )
        for item in inputs.objects
        if item.object_id not in {first.subject_id, first.target_id}
    )
    scene_inputs = PolicySceneInputs(
        table_sdf=inputs.table_sdf.resolve(),
        cup_sdf=first_subject.sdf_path.resolve(),
        tray_sdf=first_target.sdf_path.resolve(),
        panda_urdf=inputs.panda_urdf.resolve(),
        output_dir=output,
        robot_embodiment="robolab_droid",
        initial_arm_joint_positions=OFFICIAL_DROID_RESET,
        disable_robot_gravity=True,
        arm_position_gain=0.5,
        arm_joint_forces=OFFICIAL_DROID_FORCES,
        action_authority="deterministic compiled DROID operator graph in PyBullet",
        cup_center_xy=first_subject.center_xy,
        tray_center_xy=first_target.center_xy,
        cup_orientation_xyzw=first_subject.orientation_xyzw,
        tray_orientation_xyzw=first_target.orientation_xyzw,
        success_alignment_axis_local=first.terminal_alignment_axis_local,
        success_alignment_axis_world=first.terminal_alignment_axis_world,
        success_predicate=first.success_predicate,
        operator="pick_place",
        subject_lateral_friction=first_subject.lateral_friction,
        target_collision_enabled=first_target.collision_enabled,
        subject_id=first.subject_id,
        target_id=first.target_id,
        additional_objects=additional,
    )
    actions: list[list[float]] = []
    action_index = 0
    step_results: list[dict[str, Any]] = []
    step_certificates: list[dict[str, Any]] = []

    with PyBulletDroidPolicyScene(scene_inputs) as scene:
        p = scene.p
        reference_end_effector_orientation = p.getLinkState(
            scene.panda, scene.end_effector, computeForwardKinematics=True
        )[5]

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

        def move(
            target: Sequence[float],
            count: int,
            gripper: float,
            phase: str,
            fixed_orientation: Sequence[float],
        ) -> None:
            start = np.asarray(
                p.getLinkState(
                    scene.panda,
                    scene.end_effector,
                    computeForwardKinematics=True,
                )[4],
                dtype=np.float64,
            )
            destination = np.asarray(target, dtype=np.float64)
            if destination.shape != (3,) or not np.isfinite(destination).all():
                raise ValueError(f"invalid operator-graph Cartesian target: {destination}")
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

        def orient_in_place(target_orientation: Sequence[float]) -> None:
            state = p.getLinkState(
                scene.panda,
                scene.end_effector,
                computeForwardKinematics=True,
            )
            for local_index in range(ALIGNMENT_ACTION_COUNT):
                alpha = (local_index + 1) / ALIGNMENT_ACTION_COUNT
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                orientation = p.getQuaternionSlerp(
                    state[5], target_orientation, smooth
                )
                solution = p.calculateInverseKinematics(
                    scene.panda,
                    scene.end_effector,
                    state[4],
                    orientation,
                    maxNumIterations=300,
                    residualThreshold=1e-6,
                )
                if len(solution) < 7 or not np.isfinite(solution[:7]).all():
                    raise RuntimeError(
                        f"PyBullet IK failed for terminal orientation: {solution}"
                    )
                execute(solution[:7], 1.0, "orient")

        for step_index, step in enumerate(inputs.steps):
            if step_index:
                scene.activate_pair(
                    subject_id=step.subject_id,
                    target_id=step.target_id,
                    success_predicate=step.success_predicate,
                    operator="pick_place",
                    step_index=step_index,
                    success_alignment_axis_local=(
                        step.terminal_alignment_axis_local
                    ),
                    success_alignment_axis_world=(
                        step.terminal_alignment_axis_world
                    ),
                )
            world_yaw = p.getQuaternionFromEuler(
                (0.0, 0.0, float(np.deg2rad(step.grasp_yaw_deg)))
            )
            fixed_orientation = p.multiplyTransforms(
                (0.0, 0.0, 0.0),
                world_yaw,
                (0.0, 0.0, 0.0),
                reference_end_effector_orientation,
            )[1]
            subject_link_position = np.asarray(scene._link_pose(scene.cup)[0])
            target_link_position = np.asarray(scene._link_pose(scene.tray)[0])
            subject_xy = subject_link_position[:2]
            grasp_z = float(
                subject_link_position[2] + step.grasp_base_offset_z_m
            )
            hover_z = float(
                subject_link_position[2] + step.hover_base_offset_z_m
            )
            lift_z = float(subject_link_position[2] + step.lift_base_offset_z_m)

            hold(PHASE_ACTION_COUNTS["settle"], 0.0, "settle")
            move(
                (*subject_xy, hover_z),
                PHASE_ACTION_COUNTS["approach"],
                0.0,
                "approach",
                fixed_orientation,
            )
            move(
                (*subject_xy, grasp_z),
                PHASE_ACTION_COUNTS["descend"],
                0.0,
                "descend",
                fixed_orientation,
            )
            hold(PHASE_ACTION_COUNTS["grasp"], 1.0, "grasp")
            move(
                (*subject_xy, lift_z),
                PHASE_ACTION_COUNTS["lift"],
                1.0,
                "lift",
                fixed_orientation,
            )
            terminal_alignment = None
            if step.terminal_alignment_axis_local is not None:
                subject_link_pose = scene._link_pose(scene.cup)
                subject_rotation = np.asarray(
                    p.getMatrixFromQuaternion(subject_link_pose[1]),
                    dtype=np.float64,
                ).reshape(3, 3)
                source_axis_world = subject_rotation @ np.asarray(
                    step.terminal_alignment_axis_local, dtype=np.float64
                )
                alignment_quaternion = axis_alignment_quaternion_xyzw(
                    source_axis_world,
                    step.terminal_alignment_axis_world,
                )
                desired_subject_orientation = p.multiplyTransforms(
                    (0.0, 0.0, 0.0),
                    alignment_quaternion,
                    (0.0, 0.0, 0.0),
                    subject_link_pose[1],
                )[1]
                ee_state = p.getLinkState(
                    scene.panda,
                    scene.end_effector,
                    computeForwardKinematics=True,
                )
                inverse_ee = p.invertTransform(ee_state[4], ee_state[5])
                subject_from_ee = p.multiplyTransforms(
                    inverse_ee[0],
                    inverse_ee[1],
                    subject_link_pose[0],
                    subject_link_pose[1],
                )
                inverse_subject_from_ee = p.invertTransform(
                    (0.0, 0.0, 0.0), subject_from_ee[1]
                )
                fixed_orientation = p.multiplyTransforms(
                    (0.0, 0.0, 0.0),
                    desired_subject_orientation,
                    (0.0, 0.0, 0.0),
                    inverse_subject_from_ee[1],
                )[1]
                orient_in_place(fixed_orientation)
                terminal_alignment = {
                    "local_axis": list(step.terminal_alignment_axis_local),
                    "source_axis_world": source_axis_world.tolist(),
                    "target_axis_world": list(step.terminal_alignment_axis_world),
                    "alignment_quaternion_xyzw": list(alignment_quaternion),
                    "desired_subject_orientation_xyzw": list(
                        desired_subject_orientation
                    ),
                    "terminal_end_effector_orientation_xyzw": list(
                        fixed_orientation
                    ),
                }
            measured_ee_position = np.asarray(
                p.getLinkState(
                    scene.panda,
                    scene.end_effector,
                    computeForwardKinematics=True,
                )[4],
                dtype=np.float64,
            )
            measured_subject_pose = p.getBasePositionAndOrientation(scene.cup)
            measured_subject_position = np.asarray(
                measured_subject_pose[0], dtype=np.float64
            )
            desired_subject_link_position = target_link_position.copy()
            desired_subject_link_position[:2] += np.asarray(
                step.place_target_offset_xy_m, dtype=np.float64
            )
            desired_subject_link_position[2] += step.place_target_offset_z_m
            stack_release_geometry = None
            if step.success_predicate == "stacked":
                subject_aabb = p.getAABB(scene.cup)
                lowest_finger_z = min(
                    float(p.getAABB(scene.panda, link_index)[0][2])
                    for link_index in scene.finger_contact_links
                )
                finger_overhang = max(
                    0.0, float(subject_aabb[0][2]) - lowest_finger_z
                )
                desired_subject_link_position, stack_release_geometry = (
                    stack_subject_link_target_from_aabbs(
                        subject_link_position=measured_subject_position,
                        subject_aabb=subject_aabb,
                        target_aabb=p.getAABB(scene.tray),
                        release_clearance_m=(
                            finger_overhang + STACK_RELEASE_CLEARANCE_M
                        ),
                    )
                )
            place_end_effector, subject_from_end_effector = (
                compensated_place_end_effector_target(
                    current_end_effector_position=measured_ee_position,
                    current_subject_link_position=measured_subject_position,
                    target_link_position=desired_subject_link_position,
                    target_vertical_offset_m=0.0,
                )
            )
            move(
                (
                    float(place_end_effector[0]),
                    float(place_end_effector[1]),
                    lift_z,
                ),
                PHASE_ACTION_COUNTS["transfer"],
                1.0,
                "transfer",
                fixed_orientation,
            )
            move(
                place_end_effector,
                PHASE_ACTION_COUNTS["lower"],
                1.0,
                "lower",
                fixed_orientation,
            )
            hold(PHASE_ACTION_COUNTS["release"], 0.0, "release")
            move(
                (
                    float(place_end_effector[0]),
                    float(place_end_effector[1]),
                    hover_z,
                ),
                PHASE_ACTION_COUNTS["retreat_vertical"],
                0.0,
                "retreat_vertical",
                fixed_orientation,
            )
            move(
                (
                    float(place_end_effector[0]),
                    float(place_end_effector[1] - 0.12),
                    hover_z,
                ),
                PHASE_ACTION_COUNTS["retreat_back"],
                0.0,
                "retreat_back",
                fixed_orientation,
            )
            hold(PHASE_ACTION_COUNTS["final_settle"], 0.0, "final_settle")
            full_frames = scene.frames
            scene.frames = full_frames[scene.active_start_frame :]
            try:
                step_result = scene.write_outputs().to_dict()
            finally:
                scene.frames = full_frames
            step_results.append(step_result)
            step_certificates.append(
                {
                    "step_index": step_index,
                    "subject_id": step.subject_id,
                    "target_id": step.target_id,
                    "terminal_operator": step.terminal_operator,
                    "success_predicate": step.success_predicate,
                    "frame_start": 0 if step_index == 0 else scene.active_start_frame,
                    "frame_end": scene.frame_count - 1,
                    "measured_subject_from_end_effector_m": (
                        subject_from_end_effector.tolist()
                    ),
                    "desired_subject_link_target_m": (
                        desired_subject_link_position.tolist()
                    ),
                    "place_target_offset_xy_m": list(
                        step.place_target_offset_xy_m
                    ),
                    "stack_release_geometry": stack_release_geometry,
                    "terminal_alignment": terminal_alignment,
                    "predicate_evidence": step_result["predicate_evidence"],
                    "success": step_result["success"],
                }
            )
        global_result = scene.write_outputs().to_dict()

    trajectory_path = output / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory["schema_version"] = "scenecode_pybullet_droid_operator_graph.v1"
    trajectory["controller"]["operator"] = "operator_graph"
    trajectory["controller"]["success_predicate"] = "all_steps"
    trajectory["operator_steps"] = step_certificates
    trajectory_path.write_text(
        json_evidence.dumps(trajectory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    aggregate = dict(global_result)
    aggregate["operator"] = "operator_graph"
    aggregate["success_predicate"] = "all_steps"
    aggregate["step_count"] = len(step_results)
    aggregate["step_results"] = step_results
    aggregate["grasp_contact_left_ever"] = all(
        item["grasp_contact_left_ever"] for item in step_results
    )
    aggregate["grasp_contact_right_ever"] = all(
        item["grasp_contact_right_ever"] for item in step_results
    )
    aggregate["dual_finger_contact_frame_count"] = sum(
        int(item["dual_finger_contact_frame_count"]) for item in step_results
    )
    aggregate["carry_frame_count"] = sum(
        int(item["carry_frame_count"]) for item in step_results
    )
    aggregate["dual_finger_supported_carry_fraction"] = min(
        float(item["dual_finger_supported_carry_fraction"])
        for item in step_results
    )
    aggregate["opposed_external_carry_frame_count"] = sum(
        int(item["opposed_external_carry_frame_count"]) for item in step_results
    )
    aggregate["opposed_external_carry_fraction"] = min(
        float(item["opposed_external_carry_fraction"]) for item in step_results
    )
    aggregate["minimum_opposed_contact_margin_m"] = min(
        float(item["minimum_opposed_contact_margin_m"]) for item in step_results
    )
    aggregate["cup_lift_m"] = max(float(item["cup_lift_m"]) for item in step_results)
    aggregate["cup_inside_tray_xy"] = all(
        item["cup_inside_tray_xy"] for item in step_results
    )
    aggregate["cup_in_tray_contact"] = all(
        item["cup_in_tray_contact"] for item in step_results
    )
    aggregate["cup_released"] = all(item["cup_released"] for item in step_results)
    aggregate["object_attachment_constraint_count"] = max(
        int(item["object_attachment_constraint_count"]) for item in step_results
    )
    aggregate["subject_horizontal_displacement_m"] = sum(
        float(item["subject_horizontal_displacement_m"]) for item in step_results
    )
    aggregate["predicate_evidence"] = {
        "predicate_passed": all(item["success"] for item in step_results),
        "steps": [item["predicate_evidence"] for item in step_results],
    }
    aggregate["success"] = all(item["success"] for item in step_results)
    result_path = output / "result.json"
    result_path.write_text(
        json_evidence.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    actions_path = output / "expert_actions.npy"
    np.save(actions_path, np.asarray(actions, dtype=np.float32), allow_pickle=False)
    phase_counts = Counter(frame["phase"] for frame in trajectory["frames"])
    scene_objects_manifest = output / "scene_objects_manifest.json"
    scene_objects_manifest.write_text(
        json_evidence.dumps(
            {
                "schema_version": "robotics_scene_object_table.v1",
                "objects": [
                    {
                        "object_id": item.object_id,
                        "sdf_path": str(item.sdf_path.resolve()),
                        "role": item.role,
                    }
                    for item in inputs.objects
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    certificate = {
        "schema_version": "scenecode_pybullet_droid_operator_graph_certificate.v1",
        "operator": "operator_graph",
        "step_count": len(step_certificates),
        "steps": step_certificates,
        "frame_count": len(trajectory["frames"]),
        "fps": PyBulletDroidPolicyScene.control_hz,
        "duration_sec": len(trajectory["frames"])
        / PyBulletDroidPolicyScene.control_hz,
        "recorded_phase_counts": dict(sorted(phase_counts.items())),
        "object_ids": sorted(objects),
        "actions": str(actions_path),
        "actions_sha256": _sha256(actions_path),
        "trajectory": str(trajectory_path),
        "trajectory_sha256": _sha256(trajectory_path),
        "scene_objects_manifest": str(scene_objects_manifest),
        "scene_objects_manifest_sha256": _sha256(scene_objects_manifest),
        "success": aggregate["success"],
    }
    certificate_path = output / "expert_certificate.json"
    certificate_path.write_text(
        json_evidence.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return DroidOperatorGraphResult(
        trajectory_path=str(trajectory_path),
        actions_path=str(actions_path),
        certificate_path=str(certificate_path),
        scene_objects_manifest_path=str(scene_objects_manifest),
        frame_count=len(trajectory["frames"]),
        simulation_seconds=len(trajectory["frames"])
        / PyBulletDroidPolicyScene.control_hz,
        step_count=len(step_certificates),
        simulator_result=aggregate,
        success=bool(aggregate["success"]),
    )
    axis_alignment_quaternion_xyzw,
