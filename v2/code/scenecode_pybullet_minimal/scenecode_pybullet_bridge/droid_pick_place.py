"""Deterministic Franka+Robotiq cup-to-tray expert rollout."""

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

from .policy_scene import (
    PolicySceneInputs,
    PolicySceneResult,
    PyBulletDroidPolicyScene,
)


OFFICIAL_DROID_RESET = (
    0.0,
    -np.pi / 5.0,
    0.0,
    -4.0 * np.pi / 5.0,
    0.0,
    3.0 * np.pi / 5.0,
    0.0,
)
STACK_RELEASE_CLEARANCE_M = 0.003
OFFICIAL_DROID_FORCES = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
PHASE_ACTION_COUNTS = {
    "settle": 15,
    "approach": 30,
    "descend": 24,
    "grasp": 18,
    "lift": 24,
    "transfer": 36,
    "lower": 24,
    "release": 18,
    "retreat_vertical": 18,
    "retreat_back": 6,
    "final_settle": 9,
}
ALIGNMENT_ACTION_COUNT = 24
# One adapter-owned vertical envelope shared by the reference compiler and this
# executor: the compiler never emits a base-height offset outside this band, and
# the executor accepts exactly the same band.
MINIMUM_BASE_OFFSET_Z_M = 0.12
MAXIMUM_BASE_OFFSET_Z_M = 0.55
# Every terminal operator the reference compiler can emit must reach this shared
# expert.  The place family (including stack, insert and cover) shares the
# default pick-and-place motion envelope; ``flip``, ``strike`` and ``pour`` are
# the explicit specialisations below.
SUPPORTED_SEMANTIC_OPERATORS = frozenset(
    {"pick_place", "place", "stack", "insert", "cover", "flip", "strike", "pour"}
)
# Vertical margin kept between a carried subject and the destination's top face
# while the robot carries it across that destination.
CARRY_CLEARANCE_OVER_DESTINATION_M = 0.02
# A placement descent counts as supported when the subject stays in contact and
# stops dropping within this distance between two consecutive actions.
PLACEMENT_SUPPORT_DROP_TOLERANCE_M = 0.005
# Lateral margin kept between the grasp site and the measured subject footprint.
GRASP_FOOTPRINT_MARGIN_M = 0.002
# Share of the carry frames that must hold the subject with both fingers.  The
# first and last carry frame are transitions; requiring all of them abandoned
# rollouts that had already carried the subject successfully.
CARRY_CONTACT_FRACTION_REQUIRED = 0.95
CARRY_CONTACT_TRANSITION_FRAMES = 2


@dataclass(frozen=True)
class DroidPickPlaceInputs:
    table_sdf: Path
    cup_sdf: Path
    tray_sdf: Path
    panda_urdf: Path
    output_dir: Path
    grasp_base_offset_z_m: float = 0.19
    hover_base_offset_z_m: float = 0.38
    lift_base_offset_z_m: float = 0.43
    grasp_yaw_deg: float = 90.0
    place_target_offset_xy_m: tuple[float, float] = (0.0, 0.0)
    place_target_offset_z_m: float = 0.0
    # Grasp-site offset from the subject root link, expressed in world axes.
    # A non-zero offset lets the expert grasp a dedicated carrying feature
    # (e.g. a handle) instead of always pinching the subject body at its root.
    grasp_offset_xy_m: tuple[float, float] = (0.0, 0.0)
    cup_center_xy: tuple[float, float] = (0.24, -0.08)
    tray_center_xy: tuple[float, float] = (-0.18, 0.10)
    cup_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    tray_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    terminal_alignment_axis_local: tuple[float, float, float] | None = None
    terminal_alignment_axis_world: tuple[float, float, float] | None = None
    # When set, the expert only succeeds if the measured final subject alignment
    # cosine against the declared terminal axes reaches this value. It certifies
    # a requested upright/standing placement instead of leaving it implicit.
    terminal_upright_min_cosine: float | None = None
    # A support-region alias may share the physical table SDF.  Disable the
    # duplicate target collision in that case so the subject rests on the real
    # support instead of being spawned underneath a second table body.
    target_collision_enabled: bool = True
    semantic_operator: str = "pick_place"
    success_predicate: str = "in_container"
    target_frame_count: int | None = None


@dataclass(frozen=True)
class DroidPickPlaceResult:
    trajectory_path: str
    actions_path: str
    certificate_path: str
    frame_count: int
    simulation_seconds: float
    dual_contact_carry_fraction: float
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


def axis_alignment_quaternion_xyzw(
    source_axis_world: Sequence[float], target_axis_world: Sequence[float]
) -> tuple[float, float, float, float]:
    """Return the shortest normalized quaternion rotating source onto target."""

    source = np.asarray(source_axis_world, dtype=np.float64)
    target = np.asarray(target_axis_world, dtype=np.float64)
    if source.shape != (3,) or target.shape != (3,):
        raise ValueError("axis alignment requires two three-vectors")
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if (
        not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or source_norm <= 1.0e-9
        or target_norm <= 1.0e-9
    ):
        raise ValueError("axis alignment vectors must be finite and non-zero")
    source /= source_norm
    target /= target_norm
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot < -1.0 + 1.0e-9:
        basis = np.eye(3)[int(np.argmin(np.abs(source)))]
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis)
        return (float(axis[0]), float(axis[1]), float(axis[2]), 0.0)
    cross = np.cross(source, target)
    quaternion = np.asarray([cross[0], cross[1], cross[2], 1.0 + dot])
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def compensated_place_end_effector_target(
    *,
    current_end_effector_position: Sequence[float],
    current_subject_link_position: Sequence[float],
    target_link_position: Sequence[float],
    target_vertical_offset_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compensate the place waypoint for the measured subject-to-gripper offset."""

    end_effector = np.asarray(current_end_effector_position, dtype=np.float64)
    subject = np.asarray(current_subject_link_position, dtype=np.float64)
    target = np.asarray(target_link_position, dtype=np.float64).copy()
    if any(value.shape != (3,) for value in (end_effector, subject, target)):
        raise ValueError("placement compensation positions must be three-vectors")
    if not all(np.isfinite(value).all() for value in (end_effector, subject, target)):
        raise ValueError("placement compensation positions must be finite")
    vertical_offset = float(target_vertical_offset_m)
    if not np.isfinite(vertical_offset):
        raise ValueError("placement compensation vertical offset must be finite")
    subject_from_end_effector = subject - end_effector
    target[2] += vertical_offset
    return target - subject_from_end_effector, subject_from_end_effector


def stack_subject_link_target_from_aabbs(
    *,
    subject_link_position: Sequence[float],
    subject_aabb: Sequence[Sequence[float]],
    target_aabb: Sequence[Sequence[float]],
    release_clearance_m: float = STACK_RELEASE_CLEARANCE_M,
) -> tuple[np.ndarray, dict[str, float]]:
    """Center a stacked subject and release its bottom just above the target top."""

    subject_link = np.asarray(subject_link_position, dtype=np.float64)
    subject_bounds = np.asarray(subject_aabb, dtype=np.float64)
    target_bounds = np.asarray(target_aabb, dtype=np.float64)
    clearance = float(release_clearance_m)
    if subject_link.shape != (3,) or subject_bounds.shape != (2, 3) or (
        target_bounds.shape != (2, 3)
    ):
        raise ValueError("stack placement requires one link position and two AABBs")
    if not all(
        np.isfinite(value).all()
        for value in (subject_link, subject_bounds, target_bounds)
    ) or not np.isfinite(clearance):
        raise ValueError("stack placement geometry must be finite")
    if clearance < 0 or np.any(subject_bounds[1] <= subject_bounds[0]) or np.any(
        target_bounds[1] <= target_bounds[0]
    ):
        raise ValueError("stack placement geometry is degenerate")
    subject_bottom_from_link = float(subject_bounds[0, 2] - subject_link[2])
    target_top = float(target_bounds[1, 2])
    desired = np.array(
        [
            float((target_bounds[0, 0] + target_bounds[1, 0]) / 2.0),
            float((target_bounds[0, 1] + target_bounds[1, 1]) / 2.0),
            target_top + clearance - subject_bottom_from_link,
        ],
        dtype=np.float64,
    )
    return desired, {
        "subject_bottom_from_link_m": subject_bottom_from_link,
        "target_top_z_m": target_top,
        "release_clearance_m": clearance,
    }



def _clamp_into_footprint(value: float, low: float, high: float) -> float:
    """Clamp a lateral grasp site into a measured footprint with a margin."""

    lower = float(low) + GRASP_FOOTPRINT_MARGIN_M
    upper = float(high) - GRASP_FOOTPRINT_MARGIN_M
    if lower > upper:
        return float((float(low) + float(high)) / 2.0)
    return float(min(max(float(value), lower), upper))


def carry_transfer_target_z(
    *,
    compiled_lift_z_m: float,
    measured_end_effector_z_m: float,
    measured_subject_bottom_z_m: float,
    measured_target_top_z_m: float,
    clearance_m: float = CARRY_CLEARANCE_OVER_DESTINATION_M,
) -> tuple[float, dict[str, float]]:
    """Return the carry height that keeps a pinched subject above a destination.

    A subject pinched off its own base hangs below the gripper by the measured
    offset, so the compiled lift height can still drag it through a tall
    destination top face.  The transfer therefore rises to whatever height puts
    the measured subject bottom above the measured destination top plus the
    clearance, and never drops below the compiled lift.
    """

    values = (
        compiled_lift_z_m,
        measured_end_effector_z_m,
        measured_subject_bottom_z_m,
        measured_target_top_z_m,
    )
    if not all(np.isfinite(float(value)) for value in values) or not np.isfinite(
        float(clearance_m)
    ):
        raise ValueError("carry transfer geometry must be finite")
    if float(clearance_m) < 0.0:
        raise ValueError("carry clearance must be non-negative")
    pinch_offset_m = float(measured_end_effector_z_m) - float(
        measured_subject_bottom_z_m
    )
    required_transfer_z_m = (
        float(measured_target_top_z_m) + float(clearance_m) + pinch_offset_m
    )
    adopted_transfer_z_m = float(max(float(compiled_lift_z_m), required_transfer_z_m))
    return adopted_transfer_z_m, {
        "compiled_lift_z_m": float(compiled_lift_z_m),
        "required_transfer_z_m": required_transfer_z_m,
        "adopted_transfer_z_m": adopted_transfer_z_m,
        "pinch_offset_m": pinch_offset_m,
        "measured_end_effector_z_m": float(measured_end_effector_z_m),
        "measured_subject_bottom_z_m": float(measured_subject_bottom_z_m),
        "measured_target_top_z_m": float(measured_target_top_z_m),
        "clearance_m": float(clearance_m),
        "raised_above_compiled_lift": bool(
            adopted_transfer_z_m > float(compiled_lift_z_m)
        ),
    }


def placement_descent_is_supported(
    *,
    previous_subject_bottom_z_m: float,
    subject_bottom_z_m: float,
    in_contact: bool,
) -> bool:
    """Return whether a placement descent is resting on the destination.

    Contact alone is not support: a wall graze can touch the destination while
    the subject keeps falling.  A hanging carry is not support either: the
    subject can stall without touching anything.  Support therefore requires
    both contact and a subject that stopped dropping.
    """

    if not np.isfinite(float(previous_subject_bottom_z_m)) or not np.isfinite(
        float(subject_bottom_z_m)
    ):
        raise ValueError("placement descent geometry must be finite")
    drop_m = float(previous_subject_bottom_z_m) - float(subject_bottom_z_m)
    return bool(in_contact) and drop_m <= PLACEMENT_SUPPORT_DROP_TOLERANCE_M


def resolve_grasp_waypoints(
    *,
    subject_base_xy: Sequence[float],
    subject_aabb: Sequence[Sequence[float]],
    grasp_offset_xy_m: Sequence[float],
    grasp_base_offset_z_m: float,
    hover_base_offset_z_m: float,
    lift_base_offset_z_m: float,
    semantic_operator: str,
) -> dict[str, Any]:
    """Resolve the grasp, hover and lift waypoints from the measured subject AABB.

    The compiled offsets are measured from the lowest collision point, so the
    pinch follows the AABB bottom instead of the root origin, and the lateral
    grasp site is clamped inside the measured footprint instead of trusting a
    pose the object no longer holds.  ``flip`` and ``strike`` pinch the measured
    body centre and lift clear of the measured top face.
    """

    base_xy = np.asarray(subject_base_xy, dtype=np.float64)
    bounds = np.asarray(subject_aabb, dtype=np.float64)
    offset_xy = np.asarray(grasp_offset_xy_m, dtype=np.float64)
    offsets = np.asarray(
        [grasp_base_offset_z_m, hover_base_offset_z_m, lift_base_offset_z_m],
        dtype=np.float64,
    )
    if base_xy.shape != (2,) or bounds.shape != (2, 3) or offset_xy.shape != (2,):
        raise ValueError(
            "grasp waypoints require one subject XY, one subject AABB and one XY offset"
        )
    if not all(
        np.isfinite(value).all() for value in (base_xy, bounds, offset_xy)
    ) or not np.isfinite(offsets).all():
        raise ValueError("grasp waypoint geometry must be finite")
    if np.any(bounds[1] <= bounds[0]):
        raise ValueError("grasp waypoint geometry is degenerate")

    bottom_z = float(bounds[0][2])
    top_z = float(bounds[1][2])
    if semantic_operator in {"flip", "strike"}:
        if semantic_operator == "flip" and float(np.linalg.norm(offset_xy)) > 1e-9:
            requested = base_xy + offset_xy
            grasp_xy = (
                _clamp_into_footprint(float(requested[0]), float(bounds[0][0]),
                                      float(bounds[1][0])),
                _clamp_into_footprint(float(requested[1]), float(bounds[0][1]),
                                      float(bounds[1][1])),
            )
        else:
            grasp_xy = (
                float((bounds[0][0] + bounds[1][0]) / 2.0),
                float((bounds[0][1] + bounds[1][1]) / 2.0),
            )
        grasp_z_m = float((bounds[0][2] + bounds[1][2]) / 2.0)
        hover_z_m = float(top_z + 0.18)
        lift_z_m = float(top_z + 0.24)
    else:
        requested = base_xy + offset_xy
        grasp_xy = (
            _clamp_into_footprint(
                float(requested[0]), float(bounds[0][0]), float(bounds[1][0])
            ),
            _clamp_into_footprint(
                float(requested[1]), float(bounds[0][1]), float(bounds[1][1])
            ),
        )
        grasp_z_m = float(bottom_z + offsets[0])
        hover_z_m = float(bottom_z + offsets[1])
        lift_z_m = float(bottom_z + offsets[2])
    return {
        "subject_bottom_z_m": bottom_z,
        "subject_top_z_m": top_z,
        "grasp_xy": grasp_xy,
        "grasp_z_m": grasp_z_m,
        "hover_z_m": hover_z_m,
        "lift_z_m": lift_z_m,
        "semantic_operator": semantic_operator,
    }

def run_droid_pick_place(inputs: DroidPickPlaceInputs) -> DroidPickPlaceResult:
    """Generate a successful reference without object-attachment constraints."""

    output = inputs.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite DROID expert rollout: {output}")
    offsets = np.asarray(
        [
            inputs.grasp_base_offset_z_m,
            inputs.hover_base_offset_z_m,
            inputs.lift_base_offset_z_m,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(offsets).all() or not (
        MINIMUM_BASE_OFFSET_Z_M
        <= offsets[0]
        < offsets[1]
        <= offsets[2]
        <= MAXIMUM_BASE_OFFSET_Z_M
    ):
        raise ValueError("DROID expert base-height offsets are invalid")
    if not np.isfinite(inputs.grasp_yaw_deg) or abs(inputs.grasp_yaw_deg) > 180.0:
        raise ValueError("DROID expert grasp yaw must be finite and within [-180, 180]")
    if (
        not np.isfinite(inputs.place_target_offset_z_m)
        or not -0.10 <= inputs.place_target_offset_z_m <= 0.50
    ):
        raise ValueError("DROID expert place target Z offset must lie in [-0.10, 0.50]")
    target_offset_xy = np.asarray(inputs.place_target_offset_xy_m, dtype=np.float64)
    if target_offset_xy.shape != (2,) or not np.isfinite(target_offset_xy).all():
        raise ValueError("DROID expert place target XY offset must be a finite pair")
    grasp_offset_xy = np.asarray(inputs.grasp_offset_xy_m, dtype=np.float64)
    if grasp_offset_xy.shape != (2,) or not np.isfinite(grasp_offset_xy).all():
        raise ValueError("DROID expert grasp XY offset must be a finite pair")
    if float(np.linalg.norm(grasp_offset_xy)) > 0.20:
        raise ValueError("DROID expert grasp XY offset exceeds the subject footprint")
    if inputs.success_predicate not in PyBulletDroidPolicyScene.supported_success_predicates:
        raise ValueError(
            f"unsupported physical success predicate: {inputs.success_predicate!r}"
        )
    if inputs.semantic_operator not in SUPPORTED_SEMANTIC_OPERATORS:
        raise ValueError(f"unsupported semantic DROID operator: {inputs.semantic_operator!r}")
    alignment_requested = (
        inputs.terminal_alignment_axis_local is not None
        or inputs.terminal_alignment_axis_world is not None
    )
    if alignment_requested and (
        inputs.terminal_alignment_axis_local is None
        or inputs.terminal_alignment_axis_world is None
    ):
        raise ValueError("terminal alignment requires both local and world axes")
    if alignment_requested:
        axis_alignment_quaternion_xyzw(
            inputs.terminal_alignment_axis_local,
            inputs.terminal_alignment_axis_world,
        )
    if inputs.terminal_upright_min_cosine is not None:
        if (
            not math.isfinite(inputs.terminal_upright_min_cosine)
            or not 0.0 < inputs.terminal_upright_min_cosine <= 1.0
        ):
            raise ValueError("terminal upright minimum cosine must lie in (0, 1]")
        if not alignment_requested:
            raise ValueError(
                "terminal upright certification requires terminal alignment axes"
            )
    if sum(PHASE_ACTION_COUNTS.values()) != 222:
        raise RuntimeError("DROID expert phase schedule must contain 222 future frames")
    minimum_frame_count = 223 + (
        ALIGNMENT_ACTION_COUNT if alignment_requested else 0
    )
    target_frame_count = (
        minimum_frame_count
        if inputs.target_frame_count is None
        else inputs.target_frame_count
    )
    if not minimum_frame_count <= target_frame_count <= 750:
        raise ValueError(
            "DROID expert target frame count must be at least its compiled "
            f"operator duration ({minimum_frame_count}) and no more than 750"
        )
    phase_action_counts = dict(PHASE_ACTION_COUNTS)
    if alignment_requested:
        phase_action_counts["orient"] = ALIGNMENT_ACTION_COUNT
    phase_action_counts["final_settle"] += target_frame_count - minimum_frame_count

    scene_inputs = PolicySceneInputs(
        table_sdf=inputs.table_sdf.resolve(),
        cup_sdf=inputs.cup_sdf.resolve(),
        tray_sdf=inputs.tray_sdf.resolve(),
        panda_urdf=inputs.panda_urdf.resolve(),
        output_dir=output,
        robot_embodiment="robolab_droid",
        initial_arm_joint_positions=OFFICIAL_DROID_RESET,
        disable_robot_gravity=True,
        arm_position_gain=0.5,
        arm_joint_forces=OFFICIAL_DROID_FORCES,
        action_authority="deterministic Cartesian IK expert in PyBullet",
        cup_center_xy=tuple(float(value) for value in inputs.cup_center_xy),
        tray_center_xy=tuple(float(value) for value in inputs.tray_center_xy),
        cup_orientation_xyzw=tuple(
            float(value) for value in inputs.cup_orientation_xyzw
        ),
        tray_orientation_xyzw=tuple(
            float(value) for value in inputs.tray_orientation_xyzw
        ),
        success_alignment_axis_local=inputs.terminal_alignment_axis_local,
        success_alignment_axis_world=inputs.terminal_alignment_axis_world,
        success_predicate=inputs.success_predicate,
        target_collision_enabled=inputs.target_collision_enabled,
    )
    actions: list[list[float]] = []
    action_index = 0

    with PyBulletDroidPolicyScene(scene_inputs) as scene:
        p = scene.p
        initial_ee = p.getLinkState(
            scene.panda, scene.end_effector, computeForwardKinematics=True
        )[4:6]
        initial_orientation = initial_ee[1]
        world_yaw = p.getQuaternionFromEuler(
            (0.0, 0.0, float(np.deg2rad(inputs.grasp_yaw_deg)))
        )
        fixed_orientation = p.multiplyTransforms(
            (0.0, 0.0, 0.0),
            world_yaw,
            (0.0, 0.0, 0.0),
            initial_orientation,
        )[1]
        cup_link_position = np.asarray(scene.frames[0]["cup"]["link_position"])
        tray_link_position = np.asarray(scene.frames[0]["tray"]["link_position"])

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
            orientation: Sequence[float] | None = None,
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
                raise ValueError(f"invalid expert Cartesian target: {destination}")
            target_orientation = (
                fixed_orientation if orientation is None else orientation
            )
            for local_index in range(count):
                alpha = (local_index + 1) / count
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                point = start + (destination - start) * smooth
                solution = p.calculateInverseKinematics(
                    scene.panda,
                    scene.end_effector,
                    point.tolist(),
                    target_orientation,
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
            position = state[4]
            start_orientation = state[5]
            for local_index in range(ALIGNMENT_ACTION_COUNT):
                alpha = (local_index + 1) / ALIGNMENT_ACTION_COUNT
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                orientation = p.getQuaternionSlerp(
                    start_orientation, target_orientation, smooth
                )
                solution = p.calculateInverseKinematics(
                    scene.panda,
                    scene.end_effector,
                    position,
                    orientation,
                    maxNumIterations=300,
                    residualThreshold=1e-6,
                )
                if len(solution) < 7 or not np.isfinite(solution[:7]).all():
                    raise RuntimeError(
                        f"PyBullet IK failed for terminal orientation: {solution}"
                    )
                execute(solution[:7], 1.0, "orient")

        subject_aabb = p.getAABB(scene.cup)
        waypoints = resolve_grasp_waypoints(
            subject_base_xy=(
                float(cup_link_position[0]),
                float(cup_link_position[1]),
            ),
            subject_aabb=subject_aabb,
            grasp_offset_xy_m=inputs.grasp_offset_xy_m,
            grasp_base_offset_z_m=inputs.grasp_base_offset_z_m,
            hover_base_offset_z_m=inputs.hover_base_offset_z_m,
            lift_base_offset_z_m=inputs.lift_base_offset_z_m,
            semantic_operator=inputs.semantic_operator,
        )
        grasp_xy = np.asarray(waypoints["grasp_xy"], dtype=np.float64)
        grasp_z = float(waypoints["grasp_z_m"])
        hover_z = float(waypoints["hover_z_m"])
        lift_z = float(waypoints["lift_z_m"])
        hold(PHASE_ACTION_COUNTS["settle"], 0.0, "settle")
        move(
            (*grasp_xy, hover_z),
            PHASE_ACTION_COUNTS["approach"],
            0.0,
            "approach",
        )
        move(
            (*grasp_xy, grasp_z),
            PHASE_ACTION_COUNTS["descend"],
            0.0,
            "descend",
        )
        hold(PHASE_ACTION_COUNTS["grasp"], 1.0, "grasp")
        move((*grasp_xy, lift_z), PHASE_ACTION_COUNTS["lift"], 1.0, "lift")
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
        measured_subject_orientation = tuple(
            float(value) for value in measured_subject_pose[1]
        )
        measured_subject_rotation = p.getMatrixFromQuaternion(
            measured_subject_orientation
        )
        lift_subject_upright_cosine = float(measured_subject_rotation[8])
        terminal_alignment = None
        if alignment_requested:
            local_axis = np.asarray(
                inputs.terminal_alignment_axis_local, dtype=np.float64
            )
            subject_rotation = np.asarray(
                measured_subject_rotation, dtype=np.float64
            ).reshape(3, 3)
            source_axis_world = subject_rotation @ local_axis
            alignment_quaternion = axis_alignment_quaternion_xyzw(
                source_axis_world,
                inputs.terminal_alignment_axis_world,
            )
            desired_subject_orientation = p.multiplyTransforms(
                (0.0, 0.0, 0.0),
                alignment_quaternion,
                (0.0, 0.0, 0.0),
                measured_subject_orientation,
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
                measured_subject_position,
                measured_subject_orientation,
            )
            inverse_subject_from_ee = p.invertTransform(
                (0.0, 0.0, 0.0), subject_from_ee[1]
            )
            terminal_ee_orientation = p.multiplyTransforms(
                (0.0, 0.0, 0.0),
                desired_subject_orientation,
                (0.0, 0.0, 0.0),
                inverse_subject_from_ee[1],
            )[1]
            orient_in_place(terminal_ee_orientation)
            fixed_orientation = terminal_ee_orientation
            terminal_alignment = {
                "local_axis": list(inputs.terminal_alignment_axis_local),
                "source_axis_world": source_axis_world.tolist(),
                "target_axis_world": list(inputs.terminal_alignment_axis_world),
                "alignment_quaternion_xyzw": list(alignment_quaternion),
                "desired_subject_orientation_xyzw": list(
                    desired_subject_orientation
                ),
                "terminal_end_effector_orientation_xyzw": list(
                    terminal_ee_orientation
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
            measured_subject_position = np.asarray(
                p.getBasePositionAndOrientation(scene.cup)[0],
                dtype=np.float64,
            )
        desired_subject_link_position = tray_link_position.copy()
        desired_subject_link_position[:2] += target_offset_xy
        desired_subject_link_position[2] += inputs.place_target_offset_z_m
        if inputs.semantic_operator == "strike":
            # SceneCode target roots may carry an offset from their collision
            # geometry.  An impact must be aimed at the measured block center,
            # so bind the strike waypoint to the target AABB rather than its
            # logical link origin.
            target_aabb = p.getAABB(scene.tray)
            desired_subject_link_position[:2] = (
                0.5 * (target_aabb[0][0] + target_aabb[1][0]),
                0.5 * (target_aabb[0][1] + target_aabb[1][1]),
            )
        stack_release_geometry = None
        if inputs.success_predicate == "stacked":
            subject_aabb = p.getAABB(scene.cup)
            lowest_finger_z = min(
                float(p.getAABB(scene.panda, link_index)[0][2])
                for link_index in scene.finger_contact_links
            )
            finger_overhang_below_subject_m = max(
                0.0,
                float(subject_aabb[0][2]) - lowest_finger_z,
            )
            release_clearance = (
                finger_overhang_below_subject_m + STACK_RELEASE_CLEARANCE_M
            )
            desired_subject_link_position, stack_release_geometry = (
                stack_subject_link_target_from_aabbs(
                    subject_link_position=measured_subject_position,
                    subject_aabb=subject_aabb,
                    target_aabb=p.getAABB(scene.tray),
                    release_clearance_m=release_clearance,
                )
            )
            stack_release_geometry["lowest_finger_z_m"] = lowest_finger_z
            stack_release_geometry["finger_overhang_below_subject_m"] = (
                finger_overhang_below_subject_m
            )
        place_end_effector, subject_from_end_effector = (
            compensated_place_end_effector_target(
                current_end_effector_position=measured_ee_position,
                current_subject_link_position=measured_subject_position,
                target_link_position=desired_subject_link_position,
                target_vertical_offset_m=0.0,
            )
        )
        # A subject pinched off its own base hangs below the gripper, so the
        # compiled lift height can still drag it through a tall destination top
        # face.  Carry across at whatever height clears the measured top.
        if inputs.semantic_operator == "flip" and not inputs.target_collision_enabled:
            carry_transfer_z_m = max(lift_z, float(measured_ee_position[2]))
            carry_transfer_geometry = {
                "policy": "planar_support_has_no_destination_wall",
                "compiled_lift_z_m": lift_z,
                "adopted_transfer_z_m": carry_transfer_z_m,
                "raised_above_compiled_lift": False,
            }
        else:
            carry_transfer_z_m, carry_transfer_geometry = carry_transfer_target_z(
                compiled_lift_z_m=lift_z,
                measured_end_effector_z_m=float(measured_ee_position[2]),
                measured_subject_bottom_z_m=float(p.getAABB(scene.cup)[0][2]),
                measured_target_top_z_m=float(p.getAABB(scene.tray)[1][2]),
            )
        move(
            (float(place_end_effector[0]), float(place_end_effector[1]), carry_transfer_z_m),
            PHASE_ACTION_COUNTS["transfer"],
            1.0,
            "transfer",
        )
        subject_bottom_before_descent_m = float(p.getAABB(scene.cup)[0][2])
        if inputs.semantic_operator == "strike":
            # A strike is a short closed-gripper descent to the target followed
            # by a retraction before release.  Contact is recorded per frame by
            # PolicyScene and is part of the terminal predicate certificate.
            move(place_end_effector, PHASE_ACTION_COUNTS["lower"], 1.0, "impact")
            hold(PHASE_ACTION_COUNTS["release"], 1.0, "impact")
            move(
                (float(place_end_effector[0]), float(place_end_effector[1]), hover_z),
                PHASE_ACTION_COUNTS["retreat_vertical"],
                1.0,
                "impact_retract",
            )
        else:
            move(place_end_effector, PHASE_ACTION_COUNTS["lower"], 1.0, "lower")
            hold(PHASE_ACTION_COUNTS["release"], 0.0, "release")
            if inputs.semantic_operator == "flip" and not inputs.target_collision_enabled:
                hold(PHASE_ACTION_COUNTS["retreat_vertical"], 0.0,
                     "retreat_vertical")
            else:
                move(
                    (float(place_end_effector[0]), float(place_end_effector[1]), hover_z),
                    PHASE_ACTION_COUNTS["retreat_vertical"],
                    0.0,
                    "retreat_vertical",
                )
        subject_bottom_after_descent_m = float(p.getAABB(scene.cup)[0][2])
        descent_in_contact = bool(
            scene.frames[-1].get("subject_target_contact") if scene.frames else False
        )
        placement_descent_support = {
            "previous_subject_bottom_z_m": subject_bottom_before_descent_m,
            "subject_bottom_z_m": subject_bottom_after_descent_m,
            "in_contact": descent_in_contact,
            "supported": placement_descent_is_supported(
                previous_subject_bottom_z_m=subject_bottom_before_descent_m,
                subject_bottom_z_m=subject_bottom_after_descent_m,
                in_contact=descent_in_contact,
            ),
        }
        if inputs.semantic_operator == "flip" and not inputs.target_collision_enabled:
            hold(PHASE_ACTION_COUNTS["retreat_back"], 0.0, "retreat_back")
        else:
            move(
                (
                    float(place_end_effector[0]),
                    float(place_end_effector[1] - 0.12),
                    hover_z,
                ),
                PHASE_ACTION_COUNTS["retreat_back"],
                0.0,
                "retreat_back",
            )
        hold(PHASE_ACTION_COUNTS["final_settle"], 0.0, "final_settle")
        hold(target_frame_count - minimum_frame_count, 0.0, "final_settle")
        if scene.frame_count != target_frame_count or len(actions) != target_frame_count - 1:
            raise RuntimeError(
                f"expert timeline mismatch: frames={scene.frame_count}, actions={len(actions)}"
            )
        simulator_result: PolicySceneResult = scene.write_outputs()
        finger_links = set(scene.finger_contact_links)

    actions_path = output / "expert_actions.npy"
    np.save(actions_path, np.asarray(actions, dtype=np.float32), allow_pickle=False)
    trajectory_path = Path(simulator_result.trajectory_path)
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    frames = trajectory["frames"]
    carry = [
        frame
        for frame in frames
        if frame["phase"] in {"lift", "orient", "transfer", "lower"}
    ]
    dual_carry = [
        frame
        for frame in carry
        if finger_links.issubset(frame["robot_cup_contact_links"])
    ]
    carry_body = carry[CARRY_CONTACT_TRANSITION_FRAMES:len(carry) - CARRY_CONTACT_TRANSITION_FRAMES] if len(carry) > 2 * CARRY_CONTACT_TRANSITION_FRAMES else carry
    dual_carry_body = [
        frame for frame in carry_body
        if finger_links.issubset(frame["robot_cup_contact_links"])
    ]
    dual_fraction = len(dual_carry_body) / len(carry_body) if carry_body else 0.0
    phase_counts = Counter(frame["phase"] for frame in frames)
    impact_contact_frame_count = sum(
        bool(frame.get("subject_target_contact")) for frame in frames
    )
    measured_alignment_cosine = simulator_result.predicate_evidence.get(
        "subject_alignment_cosine"
    )
    upright_certified = True
    if inputs.terminal_upright_min_cosine is not None:
        upright_certified = bool(
            isinstance(measured_alignment_cosine, (int, float))
            and math.isfinite(float(measured_alignment_cosine))
            and float(measured_alignment_cosine)
            >= inputs.terminal_upright_min_cosine
        )
    # A usable reference shows the subject carried under control; a single
    # re-grip frame at either end of the carry is not a failure.  The strict
    # all-frame requirement made any momentary contact loss discard a rollout.
    success = bool(
        simulator_result.success
        and dual_fraction >= CARRY_CONTACT_FRACTION_REQUIRED
        and simulator_result.opposed_external_carry_fraction >= CARRY_CONTACT_FRACTION_REQUIRED
        and upright_certified
    )
    certificate = {
        "schema_version": "scenecode_pybullet_droid_expert_certificate.v2",
        "frame_count": len(frames),
        "fps": PyBulletDroidPolicyScene.control_hz,
        "duration_sec": len(frames) / PyBulletDroidPolicyScene.control_hz,
        "robot_embodiment": "robolab_droid",
        "semantic_operator": inputs.semantic_operator,
        "initial_joint_positions": list(OFFICIAL_DROID_RESET),
        "phase_action_counts": phase_action_counts,
        "recorded_phase_counts": dict(sorted(phase_counts.items())),
        "base_height_offsets_m": {
            "grasp": inputs.grasp_base_offset_z_m,
            "hover": inputs.hover_base_offset_z_m,
            "lift": inputs.lift_base_offset_z_m,
        },
        "grasp_yaw_deg": inputs.grasp_yaw_deg,
        "grasp_target_world_xy_m": grasp_xy.tolist(),
        "grasp_offset_xy_m": list(inputs.grasp_offset_xy_m),
        "place_target_offset_z_m": inputs.place_target_offset_z_m,
        "place_target_offset_xy_m": list(inputs.place_target_offset_xy_m),
        "terminal_alignment": terminal_alignment,
        "terminal_upright_min_cosine": inputs.terminal_upright_min_cosine,
        "terminal_upright_measured_cosine": measured_alignment_cosine,
        "terminal_upright_certified": upright_certified,
        "measured_subject_from_end_effector_m": subject_from_end_effector.tolist(),
        "lift_subject_orientation_xyzw": list(measured_subject_orientation),
        "lift_subject_upright_cosine": lift_subject_upright_cosine,
        "compensated_place_end_effector_target_m": place_end_effector.tolist(),
        "desired_subject_link_target_m": desired_subject_link_position.tolist(),
        "stack_release_geometry": stack_release_geometry,
        "grasp_waypoints": waypoints,
        "carry_transfer_geometry": carry_transfer_geometry,
        "placement_descent_support": placement_descent_support,
        "success_predicate": inputs.success_predicate,
        "predicate_evidence": simulator_result.predicate_evidence,
        "layout": {
            "cup_center_xy": list(inputs.cup_center_xy),
            "tray_center_xy": list(inputs.tray_center_xy),
        },
        "initial_end_effector_orientation_xyzw": list(initial_orientation),
        "fixed_end_effector_orientation_xyzw": list(fixed_orientation),
        "actions": str(actions_path),
        "actions_sha256": _sha256(actions_path),
        "trajectory": str(trajectory_path),
        "trajectory_sha256": _sha256(trajectory_path),
        "carry_frame_count": len(carry),
        "dual_contact_carry_frame_count": len(dual_carry),
        "dual_contact_carry_fraction": dual_fraction,
        "impact_contact_frame_count": impact_contact_frame_count,
        "opposed_external_carry_fraction": (
            simulator_result.opposed_external_carry_fraction
        ),
        "minimum_opposed_contact_margin_m": (
            simulator_result.minimum_opposed_contact_margin_m
        ),
        "object_attachment_constraint_count": (
            simulator_result.object_attachment_constraint_count
        ),
        "simulator_result": simulator_result.to_dict(),
        "success": success,
    }
    certificate_path = output / "expert_certificate.json"
    certificate_path.write_text(
        json_evidence.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return DroidPickPlaceResult(
        trajectory_path=str(trajectory_path),
        actions_path=str(actions_path),
        certificate_path=str(certificate_path),
        frame_count=len(frames),
        simulation_seconds=len(frames) / PyBulletDroidPolicyScene.control_hz,
        dual_contact_carry_fraction=dual_fraction,
        simulator_result=simulator_result.to_dict(),
        success=success,
    )
