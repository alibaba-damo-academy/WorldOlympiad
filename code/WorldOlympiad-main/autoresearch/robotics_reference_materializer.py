"""Materialize a recipe-free certified reference from one selected task graph.

The task graph selects reusable controller operators. Model/robot adapter
constants live here as a versioned embodiment profile. A task-bound expert pose
profile can revise supported native grasp parameters before admission; it cannot
change task semantics, asset identity, physics, duration or success criteria.
The executable adapter dispatches reusable pick-plus-terminal and
push-to-region operators. Other compiled operators fail closed until their
reusable executor is implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from .io import sha256_file, write_json
from .robotics_contracts import (
    RoboticsTaskBatchManifest,
    TaskInstructionSpec,
)
from .robotics_views import load_reference_manifest
from .wam_adapter_contracts import reference_requirements_for_wam
from .robotics_resource_lock import exclusive_blender_job
from .robotics_task_intent import ensure_task_intent
from .robotics_native_cameras import (
    DROID_NATIVE_FIXED_CAMERA_PRESET,
    ROBOLAB_DROID_FIXED_CAMERAS,
    DROID_NATIVE_FIXED_FOV_Y_DEG,
    DROID_NATIVE_FRAME_MARGIN_PX,
    DROID_TABLE_FRAME_SEAT_OFFSET_FRACTIONS,
    box_corners_table,
    framing_measurement,
    native_fixed_camera_table_poses,
    pinhole_intrinsics,
)
from .robotics_robot_asset import (
    ROBOT_ASSET_DIR, ROBOT_ASSET_MANIFEST_SHA256, ROBOT_URDF_SHA256,
    validate_robot_asset,
)


HARNESS_ID = "agentic_operator_graph_droid_v1"


class MissingExecutionCapability(ValueError):
    """A declared task operator has no admitted physical executor."""
MODEL_ADAPTER = "cosmos3_nano_droid"
GRIPPER_MAX_APERTURE_M = 0.085
MINIMUM_CLOSING_TRAVEL_M = 0.008
POSE_MARGIN_M = 0.005
GEOMETRY_NUMERICAL_TOLERANCE_M = 1.0e-6
UNIFORM_SCALE_STEP = 0.05
STACK_SUPPORT_MARGIN_SCALE_STEPS = 1
PYBULLET_COLLISION_MARGIN_PER_SIDE_M = 0.004
# Default centre separation used to fit two task objects into the DROID workcell.
DROID_TWO_OBJECT_SEPARATION_M = 0.44
# Widest separation both objects may be laid out at without leaving the DROID
# reachable workcell.
DROID_MAXIMUM_REACHABLE_SEPARATION_M = 0.95
INSERT_APERTURE_CLEARANCE_FRACTION = 0.85
MINIMUM_OPPOSED_GRASP_EXTENT_M = 0.020
# A grasp feature may sit anywhere inside the manipuland footprint, but the two
# opposing finger contacts still have to straddle the subject root axis for the
# carry certificate to admit the episode.
MAXIMUM_GRASP_FEATURE_LATERAL_OFFSET_M = 0.10
MINIMUM_GRASP_STRADDLE_MARGIN_M = 0.004
# Robot-adapter calibration from the RoboLab DROID Robotiq collision model.
# The inner-finger side-grasp band sits 0.103632138 m below ``base_link``.
# Using the fingertip extremum (the previous 0.143632138 m value) only touched
# short objects on their top edge and could not sustain a physical carry.
DROID_SIDE_GRASP_CENTER_BELOW_BASE_M = 0.1036321380
DROID_MINIMUM_BASE_HEIGHT_M = 0.12
# Adapter-owned ceiling for the vertical base-height offsets.  This mirrors
# ``scenecode_pybullet_bridge.droid_pick_place.MAXIMUM_BASE_OFFSET_Z_M``: the
# executor rejects anything above it, so the compiler must never emit one.
DROID_MAXIMUM_BASE_HEIGHT_M_OFFSET = 0.55
# Extra clearance the carried subject must keep above the destination top face.
CARRY_OVER_DESTINATION_CLEARANCE_M = 0.05
# Measured workcell XY of the DROID reset end effector (base at (0, -0.43) in
# the compiled frame). A task object placed here is already inside the reset
# gripper, so the expert's first settle action collides with it and pushes it.
DROID_RESET_END_EFFECTOR_XY_M = (0.0, -0.06544)
# Horizontal half-footprint of the open DROID gripper plus a safety margin; the
# destination rectangle must stay this far from the reset end-effector XY.
DROID_GRASP_FOOTPRINT_CLEARANCE_M = 0.11
DROID_MAXIMUM_LAYOUT_SHIFT_M = 0.30
# Reorientation on a planar support uses a nearby landing site and a bounded
# offset inside the two-finger pinch.  These belong to the DROID embodiment
# profile; the task predicate and physical rollout remain independent gates.
DROID_PLANAR_REORIENTATION_PROFILE = "droid_planar_reorientation_v1"
DROID_PLANAR_REORIENTATION_GRASP_OFFSET_CAP_M = 0.006
DROID_PLANAR_REORIENTATION_GRASP_AXIS_DEADBAND_M = 0.0005
DROID_PLANAR_REORIENTATION_LANDING_SHIFT_M = 0.08
DROID_BASE_Y_M = -0.43
REFERENCE_VIEWS = ("top", "wrist", "exterior_left", "exterior_right")
COSMOS_NATIVE_SIZE_WH_BY_VIEW = {
    "top": (640, 360),
    "wrist": (640, 360),
    "exterior_left": (320, 180),
    "exterior_right": (320, 180),
}
REFERENCE_CACHE_SCHEMA = "robotics_reference_render_cache.v1"


def droid_uniform_grasp_scale(
    horizontal_extent_m: float,
    *,
    placement_horizontal_limit_m: float | None = None,
    minimum_horizontal_extent_m: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return the largest uniform scale satisfying grasp and placement geometry."""

    extent = float(horizontal_extent_m)
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("manipuland horizontal extent must be positive and finite")
    usable = GRIPPER_MAX_APERTURE_M - MINIMUM_CLOSING_TRAVEL_M - POSE_MARGIN_M
    if usable <= 0:
        raise RuntimeError("DROID gripper profile has no usable aperture")
    placement_limit = None
    if placement_horizontal_limit_m is not None:
        placement_limit = float(placement_horizontal_limit_m)
        if not math.isfinite(placement_limit) or placement_limit <= 0:
            raise ValueError("placement horizontal limit must be positive and finite")
    minimum_extent = None
    if minimum_horizontal_extent_m is not None:
        minimum_extent = float(minimum_horizontal_extent_m)
        if not math.isfinite(minimum_extent) or minimum_extent <= 0.0:
            raise ValueError("minimum grasp extent must be positive and finite")
    limiting_extent = min(
        usable,
        placement_limit if placement_limit is not None else usable,
    )
    raw = min(1.0, limiting_extent / extent)
    # Quantize downward to a stable 5% scale step. SceneCode/VTK bounds are
    # float32; admit only sub-micron measurement noise at an exact boundary so
    # a nominal 90 mm object is not spuriously rounded from 0.80 to 0.75.
    step_width_m = extent * UNIFORM_SCALE_STEP
    scale_steps = math.floor(
        raw / UNIFORM_SCALE_STEP
        + GEOMETRY_NUMERICAL_TOLERANCE_M / step_width_m
    )
    scale = max(0.1, scale_steps * UNIFORM_SCALE_STEP)
    scaled_extent = extent * scale
    grasp_passed = scaled_extent <= usable + GEOMETRY_NUMERICAL_TOLERANCE_M
    placement_passed = placement_limit is None or (
        scaled_extent <= placement_limit + GEOMETRY_NUMERICAL_TOLERANCE_M
    )
    minimum_grasp_passed = minimum_extent is None or (
        scaled_extent + GEOMETRY_NUMERICAL_TOLERANCE_M >= minimum_extent
    )
    passed = grasp_passed and placement_passed and minimum_grasp_passed
    report = {
        "schema_version": "robotics_embodiment_scale_compatibility.v2",
        "adapter": MODEL_ADAPTER,
        "robot_embodiment": "franka_robotiq_2f85",
        "source_horizontal_extent_m": extent,
        "gripper_max_aperture_m": GRIPPER_MAX_APERTURE_M,
        "minimum_closing_travel_m": MINIMUM_CLOSING_TRAVEL_M,
        "pose_margin_m": POSE_MARGIN_M,
        "usable_extent_m": usable,
        "placement_horizontal_limit_m": placement_limit,
        "minimum_opposed_grasp_extent_m": minimum_extent,
        "limiting_horizontal_extent_m": limiting_extent,
        "geometry_numerical_tolerance_m": GEOMETRY_NUMERICAL_TOLERANCE_M,
        "uniform_scale": scale,
        "scaled_horizontal_extent_m": scaled_extent,
        "grasp_passed": grasp_passed,
        "placement_passed": placement_passed,
        "minimum_opposed_grasp_passed": minimum_grasp_passed,
        "policy": (
            "largest_downward_quantized_uniform_scale_satisfying_grasp_and_"
            "terminal_placement_geometry"
        ),
        "passed": passed,
    }
    if not passed:
        raise ValueError(
            f"manipuland cannot satisfy DROID/operator geometry scale policy: {report}"
        )
    return scale, report


def stack_destination_visibility_scale(
    *, subject_horizontal_extent_m: float, destination_support_extent_m: float
) -> float:
    """Enlarge a tiny stack base before shrinking an otherwise graspable object."""

    subject_extent = float(subject_horizontal_extent_m)
    destination_extent = float(destination_support_extent_m)
    usable = GRIPPER_MAX_APERTURE_M - MINIMUM_CLOSING_TRAVEL_M - POSE_MARGIN_M
    if min(subject_extent, destination_extent) <= 0 or not all(
        math.isfinite(value) for value in (subject_extent, destination_extent)
    ):
        raise ValueError("stack visibility extents must be positive and finite")
    desired_subject_extent = min(subject_extent, usable)
    required_support_extent = (
        desired_subject_extent
        - 2.0 * PYBULLET_COLLISION_MARGIN_PER_SIDE_M
        + desired_subject_extent
        * UNIFORM_SCALE_STEP
        * STACK_SUPPORT_MARGIN_SCALE_STEPS
    )
    raw = max(1.0, required_support_extent / destination_extent)
    scale = math.ceil(raw / UNIFORM_SCALE_STEP - 1.0e-9) * UNIFORM_SCALE_STEP
    return min(2.0, scale)


def droid_destination_fit_scale(
    *,
    destination_half_extents_m: tuple[float, float],
    orientation_xyzw: tuple[float, float, float, float],
    layout_direction_xy: tuple[float, float],
) -> tuple[float, dict[str, Any]]:
    """Down-scale an oversized destination into the reachable two-object workcell.

    SceneCode normalizes some assets to roughly one metre regardless of the task
    scale. A destination whose *oriented* footprint along the layout direction
    cannot clear the robot's reset pose inside the reachable envelope is
    uniformly scaled down. This is an embodiment-compatibility derivative
    recorded in the physical manifest, not a task or success change.
    """

    half_x, half_y = (float(value) for value in destination_half_extents_m)
    direction = np.asarray(layout_direction_xy, dtype=np.float64)
    if (
        not math.isfinite(half_x)
        or not math.isfinite(half_y)
        or half_x <= 0.0
        or half_y <= 0.0
    ):
        raise ValueError("destination half extents must be positive and finite")
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise ValueError("layout direction must be finite and non-zero")
    direction = direction / norm
    rotation = _quaternion_rotation_matrix(orientation_xyzw)
    local_x = rotation[:2, 0]
    local_y = rotation[:2, 1]
    separation = DROID_TWO_OBJECT_SEPARATION_M
    center = (0.0, 0.01)
    home = DROID_RESET_END_EFFECTOR_XY_M
    dest0 = (
        center[0] + direction[0] * separation / 2.0,
        center[1] + direction[1] * separation / 2.0,
    )
    # The layout shifts the pair away from the reset pose, so the relevant
    # support direction is the home->destination direction, not the
    # subject->destination direction.
    away = np.asarray([dest0[0] - home[0], dest0[1] - home[1]], dtype=np.float64)
    away_norm = float(np.linalg.norm(away))
    if away_norm <= 1.0e-9:
        away = direction
    else:
        away = away / away_norm
    support_unit = half_x * abs(float(away @ local_x)) + half_y * abs(
        float(away @ local_y)
    )
    # The layout widens the separation (not a shift) until both objects clear
    # the reset pose, so the destination support must fit half of the widest
    # reachable separation minus the gripper footprint.
    available = (
        DROID_MAXIMUM_REACHABLE_SEPARATION_M / 2.0
        - DROID_GRASP_FOOTPRINT_CLEARANCE_M
    )
    if support_unit <= available + GEOMETRY_NUMERICAL_TOLERANCE_M:
        scale = 1.0
    else:
        raw = available / support_unit
        scale_steps = math.floor(
            raw / UNIFORM_SCALE_STEP
            + GEOMETRY_NUMERICAL_TOLERANCE_M / UNIFORM_SCALE_STEP
        )
        scale = max(0.1, scale_steps * UNIFORM_SCALE_STEP)
    scaled_support = support_unit * scale
    report = {
        "schema_version": "robotics_destination_workcell_fit.v1",
        "adapter": MODEL_ADAPTER,
        "robot_embodiment": "franka_robotiq_2f85",
        "source_half_extents_m": [half_x, half_y],
        "orientation_xyzw": [
            float(value) for value in orientation_xyzw
        ],
        "layout_direction_xy": [float(value) for value in direction],
        "clearance_direction_xy": [float(value) for value in away],
        "source_oriented_support_m": support_unit,
        "available_clearance_envelope_m": available,
        "geometry_numerical_tolerance_m": GEOMETRY_NUMERICAL_TOLERANCE_M,
        "uniform_scale": scale,
        "scaled_oriented_support_m": scaled_support,
        "policy": (
            "largest_downward_quantized_uniform_scale_whose_oriented_footprint_"
            "clears_the_reset_pose_in_the_reachable_workcell"
        ),
        "passed": scaled_support <= available + GEOMETRY_NUMERICAL_TOLERANCE_M,
    }
    if not report["passed"]:
        raise ValueError(f"destination cannot fit the DROID workcell: {report}")
    return scale, report


def native_destination_view_requirements(
    task: TaskInstructionSpec,
) -> dict[str, dict[str, Any]]:
    """Native views whose destination legibility the admission gate measures.

    The packaging gate reads a deterministic frame-zero semantic mask per native
    view and requires the destination to be fully in frame above a minimum
    linear fraction.  A framing derivative must therefore see the same views,
    with the strictest floor any requested world-action model asks for.
    """

    requirements: dict[str, dict[str, Any]] = {}
    for wam in task.requested_wams:
        for camera in reference_requirements_for_wam(wam).cameras:
            role = next(
                (item for item in camera.required_roles if item.role == "destination"),
                None,
            )
            if role is None:
                continue
            size = [int(camera.size_wh[0]), int(camera.size_wh[1])]
            record = requirements.setdefault(
                camera.source_view,
                {"size_wh": size, "minimum_linear_fraction": 0.0, "wams": []},
            )
            if record["size_wh"] != size:
                raise ValueError(
                    "requested models disagree on the native size of "
                    f"{camera.source_view}"
                )
            record["minimum_linear_fraction"] = max(
                float(record["minimum_linear_fraction"]),
                float(role.minimum_linear_fraction),
            )
            record["wams"].append(wam)
    if not requirements:
        raise ValueError("no requested model declares a native destination view")
    return requirements


# An articulated host is *placed* at its compiled base pose instead of being
# dropped and settled onto the table the way a task manipuland is, so its
# framing is evaluated at that placed pose only.  Reusing the pick-place
# settle-drift interval here would shrink every host for a separation the
# articulated path never produces.
DROID_ARTICULATED_HOST_SEAT_OFFSET_FRACTIONS = (0.0,)


def droid_destination_native_view_scale(
    *,
    candidate_scale: float,
    subject_xy: tuple[float, float],
    destination_xy: tuple[float, float],
    subject_half_extents_m: tuple[float, float],
    subject_orientation_xyzw: tuple[float, float, float, float],
    destination_half_extents_m: tuple[float, float],
    destination_orientation_xyzw: tuple[float, float, float, float],
    destination_height_m: float,
    required_views: dict[str, dict[str, Any]],
    home_xy: tuple[float, float] = DROID_RESET_END_EFFECTOR_XY_M,
    home_clearance_m: float = DROID_GRASP_FOOTPRINT_CLEARANCE_M,
    maximum_shift_m: float = DROID_MAXIMUM_LAYOUT_SHIFT_M,
    maximum_reachable_separation_m: float = DROID_MAXIMUM_REACHABLE_SEPARATION_M,
    minimum_scale: float = 0.1,
    step: float = UNIFORM_SCALE_STEP,
    margin_px: float = DROID_NATIVE_FRAME_MARGIN_PX,
    minimum_fraction_safety: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    """Scale an oversized destination until every native view frames it.

    The workcell fit keeps a destination inside the robot's reachable envelope,
    but a destination an order of magnitude larger than the manipuland can still
    swallow the native over-shoulder camera: the object then crops at the frame
    border and the model-native admission gate rejects the reference.  This
    derivative walks the quantised scale ladder downwards and returns the
    largest scale whose post-layout destination box stays fully inside every
    required native view, using the same layout function the materializer
    compiles, so the decision is a property of the reference rather than of a
    particular camera accident.
    """

    if not math.isfinite(destination_height_m) or destination_height_m <= 0.0:
        raise ValueError("destination height must be positive and finite")
    if not 0.0 < candidate_scale <= 2.0:
        raise ValueError("destination candidate scale is out of range")
    if not 0.0 < minimum_scale <= candidate_scale:
        raise ValueError("minimum scale must not exceed the candidate scale")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("scale step must be positive and finite")

    modelled = dict(ROBOLAB_DROID_FIXED_CAMERAS)
    required = {
        name: {
            "size_wh": [int(spec["size_wh"][0]), int(spec["size_wh"][1])],
            "minimum_linear_fraction": float(spec["minimum_linear_fraction"]),
        }
        for name, spec in required_views.items()
    }
    unmodelled = sorted(set(required) - set(modelled))
    checkable = {name: spec for name, spec in required.items() if name in modelled}
    if not checkable:
        raise ValueError("no required native view has a registered camera preset")
    poses = native_fixed_camera_table_poses(DROID_NATIVE_FIXED_CAMERA_PRESET)
    intrinsics = {
        name: pinhole_intrinsics(
            spec["size_wh"], DROID_NATIVE_FIXED_FOV_Y_DEG
        )
        for name, spec in checkable.items()
    }

    ladder: list[float] = []
    current = candidate_scale
    while current >= minimum_scale - GEOMETRY_NUMERICAL_TOLERANCE_M:
        ladder.append(round(current, 10))
        current -= step
    candidates: list[dict[str, Any]] = []
    selected: float | None = None
    for scale in ladder:
        _, compiled_destination_xy, layout = normalized_two_object_layout(
            subject_xy,
            destination_xy,
            maximum_reachable_separation_m=maximum_reachable_separation_m,
            subject_half_extents_m=(
                float(subject_half_extents_m[0]) * scale,
                float(subject_half_extents_m[1]) * scale,
            ),
            subject_orientation_xyzw=subject_orientation_xyzw,
            destination_half_extents_m=(
                float(destination_half_extents_m[0]) * scale,
                float(destination_half_extents_m[1]) * scale,
            ),
            destination_orientation_xyzw=destination_orientation_xyzw,
            home_xy=home_xy,
            home_clearance_m=home_clearance_m,
            maximum_shift_m=maximum_shift_m,
        )
        height = destination_height_m * scale
        views: dict[str, Any] = {}
        passed = True
        for seat_offset in DROID_TABLE_FRAME_SEAT_OFFSET_FRACTIONS:
            bottom = float(seat_offset) * height
            corners = box_corners_table(
                center_xy_m=compiled_destination_xy,
                half_extents_m=(
                    float(destination_half_extents_m[0]) * scale,
                    float(destination_half_extents_m[1]) * scale,
                ),
                bottom_z_m=bottom,
                top_z_m=bottom + height,
            )
            for name, spec in checkable.items():
                measurement = framing_measurement(
                    camera_from_table=poses[name],
                    intrinsics=intrinsics[name],
                    points_table=corners,
                    size_wh=spec["size_wh"],
                    margin_px=margin_px,
                )
                floor = (
                    spec["minimum_linear_fraction"] * minimum_fraction_safety
                )
                measurement["minimum_linear_fraction"] = floor
                measurement["passed"] = bool(
                    measurement["fully_in_frame"]
                    and measurement["linear_fraction"] >= floor
                )
                seat_key = f"seat_offset_fraction_{float(seat_offset):.3f}"
                views.setdefault(name, {})[seat_key] = measurement
                passed = passed and measurement["passed"]
        candidates.append(
            {
                "scale": scale,
                "destination_xy_m": [float(value) for value in compiled_destination_xy],
                "destination_height_m": height,
                "layout_separation_m": float(layout.get("source_separation_m", 0.0)),
                "views": views,
                "passed": passed,
            }
        )
        if passed:
            selected = scale
            break

    report: dict[str, Any] = {
        "schema_version": "robotics_destination_native_view_fit.v1",
        "policy": (
            "largest_downward_quantized_uniform_scale_whose_post_layout_destination"
            "_box_stays_fully_inside_every_required_native_view"
        ),
        "native_camera_preset": DROID_NATIVE_FIXED_CAMERA_PRESET,
        "native_fov_y_deg": DROID_NATIVE_FIXED_FOV_Y_DEG,
        "frame_margin_px": margin_px,
        "minimum_fraction_safety": minimum_fraction_safety,
        "candidate_scale": candidate_scale,
        "minimum_scale": minimum_scale,
        "scale_step": step,
        "required_views": required,
        "unmodelled_views": unmodelled,
        "candidates": candidates,
        "selected_scale": selected,
        "passed": selected is not None,
    }
    if selected is None:
        raise ValueError(
            f"destination cannot be framed by the native reference views: {report}"
        )
    return selected, report


def droid_articulated_host_native_view_scale(
    *,
    host_half_extents_m: tuple[float, float],
    host_height_m: float,
    host_xy: tuple[float, float],
    required_views: dict[str, dict[str, Any]],
    seat_offset_fractions: tuple[float, ...] = (
        DROID_ARTICULATED_HOST_SEAT_OFFSET_FRACTIONS
    ),
    candidate_scale: float = 1.0,
    minimum_scale: float = 0.1,
    step: float = UNIFORM_SCALE_STEP,
    margin_px: float = DROID_NATIVE_FRAME_MARGIN_PX,
    minimum_fraction_safety: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    """Scale a self-standing articulated host until the native views frame it.

    An articulated host is not a tabletop manipuland.  The registered DROID
    over-shoulder cameras are mounted on the robot base, so a full-height
    cupboard seated at the reachable workcell centre overflows the frame and the
    model-native admission gate rejects the reference.  The host is therefore
    fitted exactly like an oversized pick-place destination: the largest
    quantised uniform scale whose host box still stays fully inside every native
    view that the requested models measure, using the same camera preset, frame
    margin and linear-fraction floor.  The held-out robot reach and the
    acceptance threshold stay untouched: only the physical size of the host
    changes, and it changes with the geometry it frames.
    """

    if not math.isfinite(host_height_m) or host_height_m <= 0.0:
        raise ValueError("articulated host height must be positive and finite")
    if not 0.0 < candidate_scale <= 2.0:
        raise ValueError("articulated host candidate scale is out of range")
    if not 0.0 < minimum_scale <= candidate_scale:
        raise ValueError("minimum scale must not exceed the candidate scale")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("scale step must be positive and finite")
    center_xy = tuple(float(value) for value in host_xy)
    half_extents = tuple(float(value) for value in host_half_extents_m)
    if not all(math.isfinite(value) for value in (*center_xy, *half_extents)):
        raise ValueError("articulated host framing inputs must be finite")
    if half_extents[0] <= 0.0 or half_extents[1] <= 0.0:
        raise ValueError("articulated host footprint must be non-degenerate")

    modelled = dict(ROBOLAB_DROID_FIXED_CAMERAS)
    required = {
        name: {
            "size_wh": [int(spec["size_wh"][0]), int(spec["size_wh"][1])],
            "minimum_linear_fraction": float(spec["minimum_linear_fraction"]),
        }
        for name, spec in required_views.items()
    }
    unmodelled = sorted(set(required) - set(modelled))
    checkable = {name: spec for name, spec in required.items() if name in modelled}
    if not checkable:
        raise ValueError("no required native view has a registered camera preset")
    poses = native_fixed_camera_table_poses(DROID_NATIVE_FIXED_CAMERA_PRESET)
    intrinsics = {
        name: pinhole_intrinsics(spec["size_wh"], DROID_NATIVE_FIXED_FOV_Y_DEG)
        for name, spec in checkable.items()
    }

    ladder: list[float] = []
    current = candidate_scale
    while current >= minimum_scale - GEOMETRY_NUMERICAL_TOLERANCE_M:
        ladder.append(round(current, 10))
        current -= step
    candidates: list[dict[str, Any]] = []
    selected: float | None = None
    for scale in ladder:
        height = host_height_m * scale
        views: dict[str, Any] = {}
        passed = True
        for seat_offset in seat_offset_fractions:
            bottom = float(seat_offset) * height
            corners = box_corners_table(
                center_xy_m=center_xy,
                half_extents_m=(
                    half_extents[0] * scale,
                    half_extents[1] * scale,
                ),
                bottom_z_m=bottom,
                top_z_m=bottom + height,
            )
            for name, spec in checkable.items():
                measurement = framing_measurement(
                    camera_from_table=poses[name],
                    intrinsics=intrinsics[name],
                    points_table=corners,
                    size_wh=spec["size_wh"],
                    margin_px=margin_px,
                )
                floor = spec["minimum_linear_fraction"] * minimum_fraction_safety
                measurement["minimum_linear_fraction"] = floor
                measurement["passed"] = bool(
                    measurement["fully_in_frame"]
                    and measurement["linear_fraction"] >= floor
                )
                seat_key = f"seat_offset_fraction_{float(seat_offset):.3f}"
                views.setdefault(name, {})[seat_key] = measurement
                passed = passed and measurement["passed"]
        candidates.append(
            {
                "scale": scale,
                "host_xy_m": [center_xy[0], center_xy[1]],
                "host_height_m": height,
                "views": views,
                "passed": passed,
            }
        )
        if passed:
            selected = scale
            break

    report: dict[str, Any] = {
        "schema_version": "robotics_articulated_host_native_view_fit.v1",
        "policy": (
            "largest_downward_quantized_uniform_scale_whose_articulated_host_box"
            "_stays_fully_inside_every_required_native_view"
        ),
        "native_camera_preset": DROID_NATIVE_FIXED_CAMERA_PRESET,
        "native_fov_y_deg": DROID_NATIVE_FIXED_FOV_Y_DEG,
        "frame_margin_px": margin_px,
        "minimum_fraction_safety": minimum_fraction_safety,
        "seat_offset_fractions": [float(value) for value in seat_offset_fractions],
        "candidate_scale": candidate_scale,
        "minimum_scale": minimum_scale,
        "scale_step": step,
        "required_views": required,
        "unmodelled_views": unmodelled,
        "candidates": candidates,
        "selected_scale": selected,
        "passed": selected is not None,
    }
    # The same measurement that rejects an oversized host states the size the
    # generation side has to produce, so no sample is ever repaired by hand.
    requirement_scale = selected
    if requirement_scale is None:
        requirement_scale = max(
            ladder, key=lambda value: value if value > 0.0 else 0.0
        )
        requirement_scale = min(requirement_scale, 1.0)
    report["generation_requirement"] = {
        "policy": "generate_the_articulated_host_inside_the_registered_native_views",
        "host_xy_m": [center_xy[0], center_xy[1]],
        "maximum_host_extent_xyz_m": [
            float(2.0 * half_extents[0] * requirement_scale),
            float(2.0 * half_extents[1] * requirement_scale),
            float(host_height_m * requirement_scale),
        ],
        "measured_host_extent_xyz_m": [
            float(2.0 * half_extents[0]),
            float(2.0 * half_extents[1]),
            float(host_height_m),
        ],
        "required_handle_geometry": (
            "keep the declared handle contact geometry at its generated native "
            "size: a shrunken handle no longer matches the certified grasp"
        ),
        "placement": (
            "the host is a placed static body, so it is framed at its compiled "
            "base pose; the same registered views must also frame the "
            "manipulated component and the end effector, which a task-object"
            "_scale host satisfies and a floor-standing host does not"
        ),
        "required_views": sorted(required),
    }
    if selected is None:
        raise ValueError(
            "articulated host cannot be framed by the native reference views: "
            f"{report}"
        )
    return selected, report


def _oriented_rectangle_clearance_m(
    point: tuple[float, float],
    center: tuple[float, float],
    half_extents: tuple[float, float],
    orientation_xyzw: tuple[float, float, float, float],
) -> float:
    """Return the distance from an XY point to an oriented rectangle (0 inside)."""

    half_x, half_y = (float(value) for value in half_extents)
    if half_x <= 0.0 or half_y <= 0.0:
        raise ValueError("destination footprint extents must be positive")
    rotation = _quaternion_rotation_matrix(orientation_xyzw)
    delta = np.asarray(
        [point[0] - center[0], point[1] - center[1]], dtype=np.float64
    )
    local = rotation[:2, :2].T @ delta
    gap_x = max(abs(float(local[0])) - half_x, 0.0)
    gap_y = max(abs(float(local[1])) - half_y, 0.0)
    return math.hypot(gap_x, gap_y)


def normalized_two_object_layout(
    subject_xy: tuple[float, float],
    destination_xy: tuple[float, float],
    *,
    maximum_separation_m: float = DROID_TWO_OBJECT_SEPARATION_M,
    maximum_reachable_separation_m: float = DROID_MAXIMUM_REACHABLE_SEPARATION_M,
    subject_half_extents_m: tuple[float, float] | None = None,
    subject_orientation_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        1.0,
    ),
    destination_half_extents_m: tuple[float, float] | None = None,
    destination_orientation_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        1.0,
    ),
    home_xy: tuple[float, float] | None = None,
    home_clearance_m: float = 0.0,
    maximum_shift_m: float = 0.30,
) -> tuple[tuple[float, float], tuple[float, float], dict[str, Any]]:
    """Preserve SceneCode direction while fitting the DROID reachable workcell.

    The DROID expert starts from a fixed reset pose whose gripper occupies a
    known workcell footprint. Every task object whose oriented footprint
    overlaps that point is contacted by the first settle action, so the layout
    first widens the separation (up to the reachable maximum) until both the
    subject and the destination clear the reset pose by ``home_clearance_m``,
    then rigidly shifts the pair along the away-from-home direction for any
    residual asymmetry. Relative direction is preserved exactly; the separation
    is widened only as far as the workcell allows, and a layout that cannot
    clear both objects fails closed.
    """

    sx, sy = (float(value) for value in subject_xy)
    dx, dy = (float(value) for value in destination_xy)
    values = (sx, sy, dx, dy)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("SceneCode task-object layout must be finite")
    vector = (dx - sx, dy - sy)
    distance = math.hypot(*vector)
    if distance <= 0.05:
        raise ValueError("SceneCode subject and destination are not spatially distinct")
    if maximum_reachable_separation_m < maximum_separation_m:
        raise ValueError("reachable separation cannot be smaller than the base cap")
    scale = min(1.0, maximum_separation_m / distance)
    unit = (vector[0] / distance, vector[1] / distance)
    half = (vector[0] * scale / 2.0, vector[1] * scale / 2.0)
    center = (0.0, 0.01)
    subject = (center[0] - half[0], center[1] - half[1])
    destination = (center[0] + half[0], center[1] + half[1])
    shift = (0.0, 0.0)
    clearance_report: dict[str, Any] | None = None
    if home_xy is not None and destination_half_extents_m is not None:
        clearance = float(home_clearance_m)
        home = (float(home_xy[0]), float(home_xy[1]))
        if not all(math.isfinite(value) for value in (*home, clearance)):
            raise ValueError("workcell clearance geometry must be finite")
        if clearance < 0.0 or maximum_shift_m < 0.0:
            raise ValueError("workcell clearance distances must not be negative")

        def _support(
            half_extents: tuple[float, float],
            orientation_xyzw: tuple[float, float, float, float],
            direction: tuple[float, float],
        ) -> float:
            rotation = _quaternion_rotation_matrix(orientation_xyzw)
            return float(
                float(half_extents[0])
                * abs(float(np.dot(direction, rotation[:2, 0])))
                + float(half_extents[1])
                * abs(float(np.dot(direction, rotation[:2, 1])))
            )

        subject_half = (
            (float(subject_half_extents_m[0]), float(subject_half_extents_m[1]))
            if subject_half_extents_m is not None
            else (1.0e-6, 1.0e-6)
        )
        destination_half = (
            float(destination_half_extents_m[0]),
            float(destination_half_extents_m[1]),
        )
        support_subject = _support(subject_half, subject_orientation_xyzw, unit)
        support_destination = _support(
            destination_half, destination_orientation_xyzw, unit
        )
        required_separation = (
            2.0 * max(support_subject, support_destination) + 2.0 * clearance
        )
        current_separation = 2.0 * math.hypot(*half)
        separation_before = current_separation
        if required_separation > current_separation + GEOMETRY_NUMERICAL_TOLERANCE_M:
            separation = min(required_separation, maximum_reachable_separation_m)
            half = (unit[0] * separation / 2.0, unit[1] * separation / 2.0)
            subject = (center[0] - half[0], center[1] - half[1])
            destination = (center[0] + half[0], center[1] + half[1])

        away = (center[0] - home[0], center[1] - home[1])
        away_norm = math.hypot(*away)
        away = (
            (1.0, 0.0)
            if away_norm <= 1.0e-9
            else (away[0] / away_norm, away[1] / away_norm)
        )
        total_shift = 0.0
        for _ in range(32):
            gap_subject = _oriented_rectangle_clearance_m(
                home, subject, subject_half, subject_orientation_xyzw
            )
            gap_destination = _oriented_rectangle_clearance_m(
                home, destination, destination_half, destination_orientation_xyzw
            )
            if min(gap_subject, gap_destination) >= clearance - 1.0e-9:
                break
            shortfall = clearance - min(gap_subject, gap_destination)
            step = min(shortfall + 1.0e-3, maximum_shift_m - total_shift)
            if step <= 1.0e-9:
                break
            total_shift += step
            subject = (
                subject[0] + away[0] * step,
                subject[1] + away[1] * step,
            )
            destination = (
                destination[0] + away[0] * step,
                destination[1] + away[1] * step,
            )
        shift = (away[0] * total_shift, away[1] * total_shift)
        gap_subject = _oriented_rectangle_clearance_m(
            home, subject, subject_half, subject_orientation_xyzw
        )
        gap_destination = _oriented_rectangle_clearance_m(
            home, destination, destination_half, destination_orientation_xyzw
        )
        clearance_report = {
            "home_xy_m": list(home),
            "subject_half_extents_m": list(subject_half),
            "destination_half_extents_m": list(destination_half),
            "home_clearance_m": clearance,
            "separation_before_m": separation_before,
            "separation_after_m": math.hypot(
                destination[0] - subject[0], destination[1] - subject[1]
            ),
            "required_separation_m": required_separation,
            "applied_shift_m": list(shift),
            "subject_gap_m": gap_subject,
            "destination_gap_m": gap_destination,
            "clearance_satisfied": bool(
                min(gap_subject, gap_destination) >= clearance - 1.0e-6
            ),
        }
    report = {
        "schema_version": "robotics_reachability_layout_normalization.v1",
        "source_subject_xy_m": [sx, sy],
        "source_destination_xy_m": [dx, dy],
        "source_separation_m": distance,
        "maximum_reachable_separation_m": maximum_reachable_separation_m,
        "uniform_relative_layout_scale": scale,
        "compiled_subject_xy_m": list(subject),
        "compiled_destination_xy_m": list(destination),
        "direction_preserved": True,
    }
    if clearance_report is not None:
        report["workcell_clearance"] = clearance_report
    return subject, destination, report



def _cli_float(value: Any) -> str:
    """Serialize a finite float without exponent syntax for argparse nargs."""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("CLI numeric arguments must be finite")
    rendered = f"{number:.17f}".rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


def normalized_object_layout(
    positions: dict[str, tuple[float, float]],
    *,
    maximum_span_m: float = 0.52,
) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
    """Preserve one SceneCode object arrangement inside the DROID workcell."""

    if len(positions) < 2:
        raise ValueError("object layout requires at least two task objects")
    if not math.isfinite(maximum_span_m) or maximum_span_m <= 0.10:
        raise ValueError("maximum object-layout span must be finite and usable")
    object_ids = sorted(positions)
    points = {
        object_id: tuple(float(value) for value in positions[object_id])
        for object_id in object_ids
    }
    if any(
        len(point) != 2 or not all(math.isfinite(value) for value in point)
        for point in points.values()
    ):
        raise ValueError("SceneCode object layout must contain finite XY pairs")
    pair_distances = [
        math.dist(points[left], points[right])
        for index, left in enumerate(object_ids)
        for right in object_ids[index + 1 :]
    ]
    if min(pair_distances) <= 0.05:
        raise ValueError("SceneCode task objects are not spatially distinct")
    source_span = max(pair_distances)
    scale = min(1.0, maximum_span_m / source_span)
    centroid = (
        sum(point[0] for point in points.values()) / len(points),
        sum(point[1] for point in points.values()) / len(points),
    )
    workcell_center = (0.0, 0.01)
    compiled = {
        object_id: (
            workcell_center[0] + (point[0] - centroid[0]) * scale,
            workcell_center[1] + (point[1] - centroid[1]) * scale,
        )
        for object_id, point in points.items()
    }
    return compiled, {
        "schema_version": "robotics_reachability_layout_normalization.v1",
        "source_xy_m": {key: list(value) for key, value in points.items()},
        "compiled_xy_m": {key: list(value) for key, value in compiled.items()},
        "source_span_m": source_span,
        "maximum_reachable_span_m": maximum_span_m,
        "uniform_relative_layout_scale": scale,
        "relative_geometry_preserved": True,
    }


def normalized_articulated_layout(
    host_xy: tuple[float, float],
    *,
    workcell_center_xy: tuple[float, float] = (0.0, 0.55),
) -> tuple[tuple[float, float], dict[str, Any]]:
    """Place one articulated host at the registered DROID workcell center."""

    source = tuple(float(value) for value in host_xy)
    center = tuple(float(value) for value in workcell_center_xy)
    if not all(math.isfinite(value) for value in (*source, *center)):
        raise ValueError("articulated host layout must be finite")
    return center, {
        "schema_version": "robotics_articulated_layout.v1",
        "source_host_xy_m": list(source),
        "compiled_host_xy_m": list(center),
        "workcell_center_xy_m": list(center),
        "relative_geometry_preserved": True,
    }


def compiled_execution_steps(plan: dict[str, Any], *,
                             verified_operators: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Collapse a compiled controller program into reusable terminal steps."""

    from .robotics_task_compiler import UNVERIFIED_EXECUTION_OPERATORS
    controller = plan.get("controller_program") or []
    success_program = plan.get("success_program") or []
    unverified = [str(step.get("operator")) for step in controller
                  if step.get("operator") not in verified_operators and (
                      step.get("capability_status") in {"missing", "declared_unverified"}
                      or step.get("operator") in UNVERIFIED_EXECUTION_OPERATORS)]
    if unverified:
        raise MissingExecutionCapability("reference executor lacks verified operations: " +
                                         ", ".join(sorted(set(unverified))))
    required_predicates = {
        "place": {"in_container", "inside", "on_region"},
        "stack": {"stacked"},
        "insert": {"inserted"},
        "cover": {"covering"},
        # Semantic operators currently share the deterministic DROID motion
        # envelope; their explicit constraints remain in the returned step for
        # task-intent review and downstream evidence.
        "flip": {"on_region"},
        "strike": {"on_region", "contact"},
        "pour": {"inside", "in_container"},
        "push": {"on_region"},
        "pull": {"drawer_open", "door_open"},
        "press": {"pressed", "joint_at_open_limit"},
        "close": {"closed", "joint_at_closed_limit"},
    }

    def make_step(item: dict[str, Any], terminal_operator: str) -> dict[str, Any]:
        subjects = item.get("subjects") or []
        target = item.get("target")
        if len(subjects) != 1:
            raise ValueError("operator step requires exactly one subject")
        subject = str(subjects[0])
        if terminal_operator in {"pull", "press", "close"}:
            # Articulation subjects are task-addressable components, while the
            # host object is the physical target used by the simulator.
            if not isinstance(target, str):
                components = {
                    str(component.get("component_id")): component
                    for component in plan.get("articulation_components", [])
                }
                target = components.get(subject, {}).get("host_object_id")
            if not isinstance(target, str):
                raise ValueError("articulated operator requires a host target")
        elif not isinstance(target, str):
            raise ValueError("operator step requires one subject and one target")
        predicates = [
            str(record.get("predicate"))
            for record in success_program
            if record.get("subject") == subject
            and record.get("object") == target
            and record.get("predicate") in required_predicates[terminal_operator]
        ]
        # A legacy reorientation task names the logical wooden table while the
        # generated SceneCode inventory exposes its executable table surface as
        # a separate destination object.  Keep the predicate and target
        # identity intact while binding the physical step to that surface.
        if not predicates and terminal_operator == "flip":
            predicates = [
                str(record.get("predicate"))
                for record in success_program
                if record.get("subject") == subject
                and record.get("predicate") in required_predicates[terminal_operator]
            ]
        if not predicates and terminal_operator == "pour":
            # A pour moves the source container, while the terminal predicate
            # names the contents that must arrive in the destination.
            contents_object = (item.get("constraints") or {}).get("contents_object")
            if isinstance(contents_object, str):
                predicates = [
                    str(record.get("predicate"))
                    for record in success_program
                    if record.get("subject") == contents_object
                    and record.get("object") == target
                    and record.get("predicate") in required_predicates[terminal_operator]
                ]
        if len(predicates) != 1:
            raise ValueError(
                "terminal operator requires exactly one matching physical predicate"
            )
        constraints = item.get("constraints") or {}
        terminal_upright = bool(constraints.get("terminal_upright"))
        min_cosine = constraints.get("terminal_upright_min_cosine")
        step = {
            "subject_id": subject,
            "destination_id": target,
            "terminal_operator": terminal_operator,
            "terminal_predicate": predicates[0],
        }
        if terminal_operator in {"pull", "press", "close"}:
            goal_records = [record for record in success_program
                if record.get("subject") == subject and record.get("object") == target
                and record.get("predicate") == predicates[0]]
            fraction = goal_records[0].get("terminal_joint_fraction", constraints.get("terminal_joint_fraction"))
            if fraction is not None:
                step["terminal_joint_fraction"] = fraction
        if terminal_upright:
            # Only declared upright placements carry the extra certification so
            # existing non-upright plans keep their exact shape.
            step["terminal_upright"] = True
            step["terminal_upright_min_cosine"] = (
                float(min_cosine) if min_cosine is not None else None
            )
            local_axis = constraints.get("terminal_upright_axis_local")
            world_axis = constraints.get("terminal_upright_axis_world")
            if local_axis is not None:
                step["terminal_alignment_axis_local"] = tuple(
                    float(value) for value in local_axis
                )
            if world_axis is not None:
                step["terminal_alignment_axis_world"] = tuple(
                    float(value) for value in world_axis
                )
        if constraints and terminal_operator in {"insert", "flip", "strike", "pour"}:
            step["constraints"] = dict(constraints)
        return step

    operators = [str(item.get("operator")) for item in controller]
    if operators == ["push"]:
        return [make_step(controller[0], "push")]
    if operators in (["pull"], ["press"], ["close"]):
        return [make_step(controller[0], operators[0])]
    if len(controller) < 2 or len(controller) % 2:
        raise ValueError("pick-plus-terminal graph must contain complete operator pairs")
    steps = []
    for index in range(0, len(controller), 2):
        pick, terminal = controller[index : index + 2]
        terminal_operator = str(terminal.get("operator"))
        if pick.get("operator") != "pick":
            raise ValueError("operator graph contains a non-pick opening step")
        if terminal_operator not in {
            "place",
            "stack",
            "insert",
            "cover",
            "flip",
            "strike",
            "pour",
        }:
            raise MissingExecutionCapability(
                f"operator graph has no verified {terminal_operator} executor"
            )
        if pick.get("subjects") != terminal.get("subjects"):
            raise ValueError("pick and terminal operators must preserve one subject")
        steps.append(make_step(terminal, terminal_operator))
    return steps


def _sdf_has_collision_geometry(path: Path) -> bool:
    model = ET.parse(path).getroot().find("model")
    if model is None:
        raise ValueError("SceneCode SDF has no model")
    return any(list(node) for node in model.findall(".//collision/geometry"))


# Drake's deterministic VTK render engine produces the label image that is the
# authority for model-native reference admission, and it only loads ``.obj`` and
# ``.gltf`` visual meshes (other mesh types are silently ignored, see
# RenderEngineVtk).  A component repair can legitimately install a ``.glb``
# visual, which then has no label pixels at all: admission rejects the task
# object as "too small, cropped, or occluded" even though the physics executor
# and the Blender appearance renderer both consume the mesh.  The registered
# asset is never modified; the declared scene graph is projected into ``.gltf``
# in a derived directory and only the deterministic renderer is pointed at it.
DETERMINISTIC_LABEL_VISUAL_SUFFIXES = frozenset({".obj", ".gltf"})
LABEL_RENDER_PROJECTION_SCHEMA = "robotics_label_render_projection.v1"


def _sdf_mesh_uris(sdf_path: Path, *, visual_only: bool) -> list[str]:
    root = ET.parse(sdf_path).getroot()
    pattern = ".//visual/geometry/mesh/uri" if visual_only else ".//mesh/uri"
    return [
        str(node.text).strip()
        for node in root.findall(pattern)
        if node.text and str(node.text).strip()
    ]


def _declared_visual_world_points(mesh_path: Path) -> np.ndarray:
    """Return the declared visual's triangle corners with node transforms baked."""

    import trimesh

    scene = trimesh.load(str(mesh_path), force="scene", process=False)
    if not scene.graph.nodes_geometry:
        raise ValueError(f"declared visual has no triangle geometry: {mesh_path}")
    points = []
    for node in sorted(scene.graph.nodes_geometry):
        matrix, name = scene.graph[node]
        mesh = scene.geometry[name]
        triangles = np.asarray(mesh.triangles, dtype=np.float64)
        if triangles.size == 0:
            raise ValueError("declared visual conversion requires triangle meshes")
        transform = np.asarray(matrix, dtype=np.float64)
        points.append(triangles.reshape(-1, 3) @ transform[:3, :3].T + transform[:3, 3])
    return np.concatenate(points, axis=0)


def _project_declared_visual_to_gltf(source: Path, destination: Path) -> dict[str, Any]:
    """Write the declared glTF/GLB scene graph as a ``.gltf`` bundle."""

    from scipy.spatial import cKDTree
    import trimesh

    source_points = _declared_visual_world_points(source)
    scene = trimesh.load(str(source), force="scene", process=False)
    files = scene.export(file_type="gltf")
    if not files:
        raise ValueError(f"declared visual conversion produced no files: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for name, payload in files.items():
        target = (destination.parent / name).resolve()
        if target.parent != destination.parent.resolve():
            raise ValueError("converted visual escaped its projection directory")
        target.write_bytes(payload.encode() if isinstance(payload, str) else payload)
        written.append(target)
    exported = destination.parent / "model.gltf"
    if not exported.is_file():
        raise ValueError("declared visual conversion produced no glTF document")
    exported.replace(destination)
    converted_points = _declared_visual_world_points(destination)
    forward = cKDTree(source_points).query(converted_points)[0].max()
    backward = cKDTree(converted_points).query(source_points)[0].max()
    error = float(max(forward, backward))
    extent = float(np.ptp(source_points, axis=0).max())
    tolerance = max(1.0e-7, extent * 1.0e-6)
    passed = bool(
        len(source_points) == len(converted_points) and error <= tolerance
    )
    if not passed:
        raise ValueError(
            "globally projected visual changed declared geometry: "
            f"{source} (corners {len(source_points)} -> {len(converted_points)}, "
            f"max_corner_error_m={error} > tolerance_m={tolerance})"
        )
    return {
        "declared_uri": str(source),
        "projected_uri": str(destination),
        "declared_triangle_corner_count": int(len(source_points)),
        "projected_triangle_corner_count": int(len(converted_points)),
        "max_triangle_corner_error_m": error,
        "tolerance_m": tolerance,
        "source_bounds_m": [
            source_points.min(axis=0).tolist(),
            source_points.max(axis=0).tolist(),
        ],
        "projected_bounds_m": [
            converted_points.min(axis=0).tolist(),
            converted_points.max(axis=0).tolist(),
        ],
        "files": [str(path) for path in written],
        "passed": passed,
    }


def project_visual_meshes_for_label_renderer(
    *, sdf_path: Path, output_dir: Path
) -> tuple[Path, dict[str, Any]]:
    """Return an SDF whose visual meshes the deterministic label renderer loads.

    Only visual URIs with an extension the VTK label renderer ignores (a
    ``.glb`` from a component repair, for instance) are converted; every other
    asset is returned unchanged so no working case changes behaviour.  The
    projection copies the declared dependencies it needs beside the projected
    SDF and certifies the exported geometry against the declared scene graph.
    """

    sdf_path = Path(sdf_path).resolve()
    output_dir = Path(output_dir).resolve()
    visual_uris = _sdf_mesh_uris(sdf_path, visual_only=True)
    unsupported = [
        uri
        for uri in visual_uris
        if Path(uri).suffix.lower() not in DETERMINISTIC_LABEL_VISUAL_SUFFIXES
    ]
    evidence: dict[str, Any] = {
        "schema_version": LABEL_RENDER_PROJECTION_SCHEMA,
        "sdf": str(sdf_path),
        "supported_visual_suffixes": sorted(DETERMINISTIC_LABEL_VISUAL_SUFFIXES),
        "visual_mesh_uris": visual_uris,
        "projected_visuals": [],
        "passed": True,
    }
    if not unsupported:
        evidence["projection"] = "not_required"
        evidence["projected_sdf"] = str(sdf_path)
        return sdf_path, evidence

    declared = _sdf_mesh_uris(sdf_path, visual_only=False)
    source_root = sdf_path.parent
    text = sdf_path.read_text(encoding="utf-8")
    for uri in declared:
        relative = Path(uri)
        source = (source_root / relative).resolve()
        destination = (output_dir / relative).resolve()
        if Path(uri).is_absolute():
            source = Path(uri)
            relative = Path(source.name)
            destination = (output_dir / relative).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"declared mesh dependency is missing: {source}")
        if output_dir not in destination.parents and destination.parent != output_dir:
            raise ValueError("declared mesh dependency escaped its projection directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    projected_sdf = output_dir / sdf_path.name
    for uri in unsupported:
        relative = Path(uri) if not Path(uri).is_absolute() else Path(Path(uri).name)
        source = output_dir / relative
        projected_relative = relative.with_suffix(".gltf")
        projected = output_dir / projected_relative
        if projected.exists():
            projected.unlink()
        result = _project_declared_visual_to_gltf(source, projected)
        marker = f"<uri>{uri}</uri>"
        replacement = f"<uri>{projected_relative.as_posix()}</uri>"
        if marker not in text:
            raise ValueError(
                f"declared SDF does not reference its visual mesh literally: {uri}"
            )
        text = text.replace(marker, replacement)
        evidence["projected_visuals"].append(
            {"declared_uri": uri, "projected_uri": projected_relative.as_posix(), **result}
        )
    projected_sdf.write_text(text, encoding="utf-8")
    remaining = [
        uri
        for uri in _sdf_mesh_uris(projected_sdf, visual_only=True)
        if Path(uri).suffix.lower() not in DETERMINISTIC_LABEL_VISUAL_SUFFIXES
    ]
    if remaining:
        raise ValueError(
            f"projected SDF still declares unsupported visuals: {remaining}"
        )
    if not all(item["passed"] for item in evidence["projected_visuals"]):
        raise ValueError("label-renderer visual projection did not certify geometry")
    evidence["projection"] = "declared_visual_scene_graph_to_gltf"
    evidence["projected_sdf"] = str(projected_sdf)
    evidence["projected_sdf_sha256"] = sha256_file(projected_sdf)
    write_json(output_dir / "label_render_projection.json", evidence)
    return projected_sdf, evidence


def label_render_visual_sdfs(
    *, work: Path, assets: dict[str, Path]
) -> dict[str, Path]:
    """Project every deterministic-renderer asset that needs a supported visual."""

    projected: dict[str, Path] = {}
    for label, sdf_path in assets.items():
        path, _ = project_visual_meshes_for_label_renderer(
            sdf_path=sdf_path,
            output_dir=work / "label_render_projection" / label,
        )
        projected[label] = path
    return projected


def pick_place_execution_geometry(
    *,
    subject_height_m: float,
    destination_height_m: float,
    terminal_operator: str,
    grasp_band_center_offset_m: float | None = None,
) -> dict[str, float]:
    """Compile reusable terminal operators into DROID vertical waypoints."""

    subject_height = float(subject_height_m)
    destination_height = float(destination_height_m)
    if not all(
        math.isfinite(value) and value > 0
        for value in (subject_height, destination_height)
    ):
        raise ValueError("pick/place object heights must be positive and finite")
    if terminal_operator not in {
        "place",
        "stack",
        "insert",
        "cover",
        "flip",
        "strike",
        "pour",
    }:
        raise ValueError(f"unsupported pick/place terminal operator: {terminal_operator}")
    if grasp_band_center_offset_m is None:
        # Fallback for objects whose grasp band could not be measured: the
        # side-grasp band sits just under the object's vertical centre.
        contact_height = 0.5 * subject_height
    else:
        contact_height = float(grasp_band_center_offset_m)
        if (
            not math.isfinite(contact_height)
            or contact_height < 0.0
            or contact_height > subject_height + 1.0e-6
        ):
            raise ValueError("grasp band centre must lie inside the subject height")
    grasp = max(
        DROID_MINIMUM_BASE_HEIGHT_M,
        contact_height + DROID_SIDE_GRASP_CENTER_BELOW_BASE_M,
    )
    if grasp >= DROID_MAXIMUM_BASE_HEIGHT_M_OFFSET:
        # The executor rejects a grasp offset at or above its ceiling, so fail
        # here with the geometry rather than emitting an unexecutable program.
        raise ValueError(
            "grasp band centre leaves no room inside the DROID vertical "
            "envelope: "
            f"{grasp} >= {DROID_MAXIMUM_BASE_HEIGHT_M_OFFSET}"
        )
    if terminal_operator in {"place", "flip", "strike"}:
        target_offset = 0.0
    elif terminal_operator in {"stack", "cover"}:
        target_offset = destination_height
    else:
        # Lower the peg base below the top face while keeping a visible rounded
        # end above the hole block.  The actual SDF collision/contact predicate
        # remains the pass authority.
        target_offset = destination_height * 0.35
    # The adapter accepts a bounded vertical envelope; keep a real hover and
    # lift clearance whenever the grasp band sits near the envelope ceiling.
    # The executor additionally raises the lateral transfer above the measured
    # destination top face, so this compiled figure no longer has to encode the
    # container wall height.  A planar re-orientation/strike destination (flip)
    # has no wall to clear, and adding its support-surface height would force a
    # needlessly fast ceiling-height lift that shakes a small manipuland out of
    # the pinch.
    if terminal_operator in {"place", "stack", "insert", "cover", "pour"}:
        carry_offset = (
            destination_height
            + 0.5 * subject_height
            + CARRY_OVER_DESTINATION_CLEARANCE_M
        )
    else:
        carry_offset = 0.0
    hover = min(
        DROID_MAXIMUM_BASE_HEIGHT_M_OFFSET,
        max(grasp + 0.19, grasp + 0.10, carry_offset),
    )
    lift = min(
        DROID_MAXIMUM_BASE_HEIGHT_M_OFFSET,
        max(hover + 0.03, grasp + 0.24, carry_offset),
    )
    return {
        "grasp_base_offset_z_m": grasp,
        "hover_base_offset_z_m": hover,
        "lift_base_offset_z_m": lift,
        "place_target_offset_z_m": target_offset,
    }


def scale_grasp_offset_xy(
    offset_xy_m: Sequence[float] | None, scale: float
) -> list[float]:
    """Scale a measured base-frame grasp offset onto the physical variant.

    The grasp band (and its lateral centre) is measured on the registered
    native asset, while the executor loads the uniformly scaled physical
    variant.  Every other band quantity is scaled before it reaches the
    controller, so the lateral offset must be scaled by the same factor or the
    expert grasps at a point outside the loaded collision footprint.
    """

    if offset_xy_m is None:
        offset = np.zeros(2, dtype=np.float64)
    else:
        offset = np.asarray(offset_xy_m, dtype=np.float64)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("grasp XY offset must be a finite pair")
    factor = float(scale)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("grasp offset scale must be positive and finite")
    return [float(value) * factor for value in offset]


def planar_reorientation_profile(
    subject_xy: Sequence[float], measured_grasp_offset_xy: Sequence[float], scale: float
) -> dict[str, object]:
    """Fit one DROID planar flip to measured asset geometry, independent of case ID."""
    subject = np.asarray(subject_xy, dtype=np.float64)
    if subject.shape != (2,) or not np.isfinite(subject).all():
        raise ValueError("planar reorientation needs a finite subject XY")
    offset = np.asarray(scale_grasp_offset_xy(measured_grasp_offset_xy, scale))
    # Geometry-derived lateral centres carry sub-millimetre mesh noise.  A
    # spurious orthogonal offset breaks an otherwise symmetric finger pinch.
    offset[np.abs(offset) < DROID_PLANAR_REORIENTATION_GRASP_AXIS_DEADBAND_M] = 0.0
    norm = float(np.linalg.norm(offset))
    if norm > DROID_PLANAR_REORIENTATION_GRASP_OFFSET_CAP_M:
        offset *= DROID_PLANAR_REORIENTATION_GRASP_OFFSET_CAP_M / norm
    # The landing point remains on the same support, towards the DROID base.
    # The independent simulator verifies support contact and task completion.
    direction_y = -1.0 if subject[1] > DROID_BASE_Y_M else 1.0
    landing = subject + np.array([
        0.0, direction_y * DROID_PLANAR_REORIENTATION_LANDING_SHIFT_M
    ])
    return {
        "profile": DROID_PLANAR_REORIENTATION_PROFILE,
        "landing_xy_m": [float(value) for value in landing],
        "grasp_offset_xy_m": [float(value) for value in offset],
        "measured_grasp_offset_xy_m": [float(value) for value in measured_grasp_offset_xy],
        "grasp_offset_cap_m": DROID_PLANAR_REORIENTATION_GRASP_OFFSET_CAP_M,
        "grasp_axis_deadband_m": DROID_PLANAR_REORIENTATION_GRASP_AXIS_DEADBAND_M,
        "landing_shift_m": DROID_PLANAR_REORIENTATION_LANDING_SHIFT_M,
    }


def _run(argv: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        stdout = (completed.stdout or "")[-4_000:]
        stderr = (completed.stderr or "")[-8_000:]
        raise RuntimeError(
            "subprocess failed "
            f"(exit={completed.returncode}, argv0={argv[0]!r})\n"
            f"stdout_tail:\n{stdout}\n"
            f"stderr_tail:\n{stderr}"
        )


def _stabilized_grasp_band_plan(
    payload: dict[str, Any],
    *,
    usable_aperture_m: float,
    minimum_extent_m: float,
    maximum_lateral_offset_m: float,
    minimum_straddle_margin_m: float,
) -> dict[str, Any]:
    """Re-centre a fallback grasp band on the manipuland's mass centroid.

    The reusable profile planner maximizes scale, so when *no* native band fits
    the gripper aperture it can fall back to a narrow taper near the very top of
    a tall subject (a bottle neck, a vase shoulder).  Such a band is above the
    subject's centre of mass: the expert can still close the jaws, but the
    certified carry gate then fails because the hanging subject pivots out of
    the pinch.  When the planner reports no natively feasible band, select the
    feasible cross-section closest to the vertical centroid of the collision
    volume instead; the existing uniform-scale step then shrinks the manipuland
    so that the *stable* band, not the taper, fits the aperture.
    """

    plan = payload.get("plan")
    profile = payload.get("profile")
    if not isinstance(plan, dict) or not isinstance(profile, dict):
        return payload
    if plan.get("feature_grasp_feasible") is not False:
        return payload
    bands = profile.get("bands") or []
    if len(bands) < 2:
        return payload
    weights = [
        max(float(band.get("x_extent_m") or 0.0), 0.0)
        * max(float(band.get("y_extent_m") or 0.0), 0.0)
        for band in bands
    ]
    total_weight = math.fsum(weights)
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        return payload
    centroid_z = math.fsum(
        float(band["base_offset_center_m"]) * weight
        for band, weight in zip(bands, weights, strict=True)
    ) / total_weight
    height = float(profile.get("object_height_m") or 0.0)
    planned_center = float(plan.get("band_center_offset_m") or 0.0)
    if height <= 0.0 or planned_center - centroid_z <= 0.15 * height:
        return payload
    band_height = float(profile.get("band_height_m") or 0.06)
    candidates: list[dict[str, float]] = []
    center = float(bands[0]["base_offset_low_m"]) + 0.5 * band_height
    top = float(bands[-1]["base_offset_high_m"])
    while center <= top + GEOMETRY_NUMERICAL_TOLERANCE_M:
        low = center - 0.5 * band_height
        high = center + 0.5 * band_height
        selected = [
            band
            for band in bands
            if float(band["base_offset_high_m"]) > low
            and float(band["base_offset_low_m"]) < high
        ]
        if selected:
            x_low = min(float(band["x_low_m"]) for band in selected)
            x_high = max(float(band["x_high_m"]) for band in selected)
            y_low = min(float(band["y_low_m"]) for band in selected)
            y_high = max(float(band["y_high_m"]) for band in selected)
            x_extent = x_high - x_low
            y_extent = y_high - y_low
            axis = 0 if x_extent <= y_extent else 1
            lateral_x = 0.5 * (x_low + x_high)
            lateral_y = 0.5 * (y_low + y_high)
            lateral = lateral_x if axis == 0 else lateral_y
            candidates.append(
                {
                    "band_center_offset_m": center,
                    "band_low_offset_m": low,
                    "band_high_offset_m": high,
                    "grip_axis": float(axis),
                    "grip_extent_m": float(min(x_extent, y_extent)),
                    "x_extent_m": float(x_extent),
                    "y_extent_m": float(y_extent),
                    "lateral_center_x_m": float(lateral_x),
                    "lateral_center_y_m": float(lateral_y),
                    "grip_axis_lateral_center_m": float(lateral),
                }
            )
        center += 0.005
    feasible = [
        candidate
        for candidate in candidates
        if candidate["grip_extent_m"] >= minimum_extent_m - GEOMETRY_NUMERICAL_TOLERANCE_M
        and abs(candidate["grip_axis_lateral_center_m"])
        <= maximum_lateral_offset_m + GEOMETRY_NUMERICAL_TOLERANCE_M
        and 0.5 * candidate["grip_extent_m"]
        - abs(candidate["grip_axis_lateral_center_m"])
        >= minimum_straddle_margin_m - GEOMETRY_NUMERICAL_TOLERANCE_M
    ]
    if not feasible:
        return payload
    best = min(
        feasible,
        key=lambda candidate: (
            abs(candidate["band_center_offset_m"] - centroid_z),
            candidate["grip_extent_m"],
        ),
    )
    stabilized = dict(plan)
    stabilized.update(
        {
            "band_center_offset_m": best["band_center_offset_m"],
            "band_low_offset_m": best["band_low_offset_m"],
            "band_high_offset_m": best["band_high_offset_m"],
            "grip_axis": int(best["grip_axis"]),
            "grip_extent_m": best["grip_extent_m"],
            "grip_axis_lateral_center_m": best["grip_axis_lateral_center_m"],
            "lateral_center_xy_m": [
                best["lateral_center_x_m"],
                best["lateral_center_y_m"],
            ],
            "predicted_straddle_margin_m": 0.5 * best["grip_extent_m"]
            - abs(best["grip_axis_lateral_center_m"]),
            "admissible_uniform_scale": math.floor(
                min(1.0, usable_aperture_m / best["grip_extent_m"]) / 0.05
                + 1.0e-9
            )
            * 0.05,
            "x_extent_m": best["x_extent_m"],
            "y_extent_m": best["y_extent_m"],
            "policy": "centroid_stabilized_uniform_scale",
        }
    )
    result = dict(payload)
    result["plan"] = stabilized
    result["stabilized_from_band_center_offset_m"] = planned_center
    result["stabilized_centroid_offset_m"] = centroid_z
    return result


def _grasp_band_plan(
    *,
    python: Path,
    bridge: Path,
    env: dict[str, str],
    sdf_path: Path,
    orientation_xyzw: tuple[float, float, float, float],
    usable_aperture_m: float,
    minimum_extent_m: float,
    maximum_lateral_offset_m: float,
    minimum_straddle_margin_m: float,
    output: Path,
) -> dict[str, Any]:
    """Measure the manipuland's real collision cross-sections for grasp planning.

    A parallel-jaw gripper only has to span the material inside the band where
    it closes, so sizing the manipuland from its overall horizontal AABB shrinks
    every object that has a usable handle or neck.  The profile comes from the
    reusable bridge measurement (``scenecode_pybullet_bridge.grasp_profile``) on
    the same collision geometry the simulator loads, so the resulting plan is
    reproducible evidence rather than a heuristic constant.
    """

    argv = [
        str(python),
        "-m",
        "scenecode_pybullet_bridge.grasp_profile",
        "--sdf",
        str(sdf_path),
        "--usable-aperture",
        _cli_float(usable_aperture_m),
        "--minimum-extent",
        _cli_float(minimum_extent_m),
        "--maximum-lateral-offset",
        _cli_float(maximum_lateral_offset_m),
        "--minimum-straddle-margin",
        _cli_float(minimum_straddle_margin_m),
        "--orientation-xyzw="
        + ",".join(_cli_float(value) for value in orientation_xyzw),
        "--out",
        str(output),
    ]
    _run(argv, cwd=bridge, env=env)
    payload = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "plan" not in payload:
        raise ValueError("grasp-band measurement returned no plan")
    return _stabilized_grasp_band_plan(
        payload,
        usable_aperture_m=usable_aperture_m,
        minimum_extent_m=minimum_extent_m,
        maximum_lateral_offset_m=maximum_lateral_offset_m,
        minimum_straddle_margin_m=minimum_straddle_margin_m,
    )


def _bounded_log_tails(root: Path) -> dict[str, str]:
    tails: dict[str, str] = {}
    remaining = 20_000
    if not root.is_dir():
        return tails
    for path in sorted(root.rglob("*.log")):
        if remaining <= 0 or not path.is_file():
            break
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        value = "\n".join(lines[-40:])[-remaining:]
        if value:
            tails[path.relative_to(root).as_posix()] = value
            remaining -= len(value)
    return tails


def _preserve_failure_evidence(work: Path, destination: Path) -> dict[str, Any]:
    """Keep measured diagnostics independently of disposable render staging."""
    paths = sorted(path for path in work.glob("*.json") if path.is_file())
    paths += sorted(path for path in (work / "simulation").glob("*.json") if path.is_file())
    paths += sorted(path for path in (work / "bundle").glob("*.json") if path.is_file())
    if not paths:
        return {}
    digests = {path.relative_to(work).as_posix(): sha256_file(path) for path in paths}
    fingerprint = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
    retained = destination / fingerprint
    for path in paths:
        target = retained / path.relative_to(work)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(retained / "evidence_manifest.json", {
        "schema_version": "robotics_reference_failure_evidence.v1",
        "files": digests,
        "passed": False,
    })
    return {"directory": str(retained), "files": digests}


def _cache_input_digests(
    *,
    repository: Path,
    bridge: Path,
    source: dict[str, Any],
    robot_manifest: Path,
) -> dict[str, str]:
    """Identify a reusable completed render by every executable input it consumed."""

    paths = {
        "materializer": Path(__file__).resolve(),
        "task": source["task_path"],
        "task_graph": source["plan_path"],
        "task_intent_policy": repository / "autoresearch" / "robotics_task_intent.py",
        "scene_generation": source["generation_path"],
        "semantic_identity": source["identity_path"],
        "robot_asset": robot_manifest,
        "renderer": repository / "scripts" / "render_scenecode_robotics_multiview.py",
        "parallel_renderer": repository
        / "scripts"
        / "render_scenecode_robotics_multiview_parallel.py",
        "blender_settings": repository
        / "deployment"
        / "ziplab5090"
        / "scenecode_robotics_render_settings.py",
        "policy_scene": bridge / "scenecode_pybullet_bridge" / "policy_scene.py",
        "sdf_bridge": bridge / "scenecode_pybullet_bridge" / "sdf.py",
        "resource_lock": repository
        / "autoresearch"
        / "robotics_resource_lock.py",
    }
    if source.get("is_articulated"):
        paths["native_camera_planner"] = (
            repository / "autoresearch" / "robotics_articulated_camera.py"
        )
        paths["operator_executor"] = (
            bridge / "scenecode_pybullet_bridge" / "articulated.py"
        )
        paths["contact_planner"] = bridge / "scenecode_pybullet_bridge" / "articulated_contact.py"
        paths["operator_driver"] = bridge / "scripts" / "run_droid_articulated.py"
        paths["articulation_binding"] = Path(source["articulation_binding_path"])
    elif source["operator_sequence"] == ["pick", "pour"]:
        paths["operator_executor"] = (
            repository / "autoresearch" / "robotics_pour_execution.py"
        )
        paths["operator_driver"] = (
            repository / "autoresearch" / "robotics_pour_controller.py"
        )
        paths["pour_asset_derivative_verifier"] = (
            repository / "autoresearch" / "robotics_asset_derivative_verifier.py"
        )
    elif len(source["execution_steps"]) > 1:
        paths["operator_executor"] = (
            bridge / "scenecode_pybullet_bridge" / "droid_operator_graph.py"
        )
        paths["operator_driver"] = (
            bridge / "scripts" / "run_droid_operator_graph.py"
        )
        paths["scene_object_contract"] = (
            repository / "autoresearch" / "robotics_scene_objects.py"
        )
    elif source["operator_sequence"] == ["push"]:
        paths["operator_executor"] = (
            bridge / "scenecode_pybullet_bridge" / "droid_push.py"
        )
        paths["operator_driver"] = bridge / "scripts" / "run_droid_push.py"
    else:
        paths["operator_executor"] = (
            bridge / "scenecode_pybullet_bridge" / "droid_pick_place.py"
        )
        paths["operator_driver"] = bridge / "scripts" / "run_droid_pick_place.py"
    result: dict[str, str] = {}
    for label, path in paths.items():
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        result[label] = sha256_file(path)
    # This version is bumped only when materializer decisions that affect
    # simulation or rendering change. Packaging-only fixes must not invalidate
    # an already completed, fully hashed render cache.
    result["execution_adapter_profile"] = (
        "agentic_articulated_camera_fit_v13" if source.get("is_articulated")
        else "agentic_droid_pour_v1" if source["operator_sequence"] == ["pick", "pour"]
        else "agentic_operator_graph_droid_v12"
    )
    return result


def _cache_artifact_digests(cache: Path) -> dict[str, str]:
    return {
        path.relative_to(cache).as_posix(): sha256_file(path)
        for path in sorted(cache.rglob("*"))
        if path.is_file() and path.name != "cache_manifest.json"
    }


def _write_completed_cache(cache: Path, input_digests: dict[str, str]) -> None:
    render_path = cache / "render" / "render_manifest.json"
    trajectory_path = cache / "simulation" / "trajectory.json"
    result_path = cache / "simulation" / "result.json"
    for path in (render_path, trajectory_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    render = json.loads(render_path.read_text(encoding="utf-8"))
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("success") is not True:
        raise ValueError("cannot cache a failed reference execution")
    if not _measured_goal_cache_valid(cache, input_digests, trajectory):
        raise ValueError("cannot cache an articulated reference without its measured terminal goal")
    if set(render.get("views", {})) != set(REFERENCE_VIEWS):
        raise ValueError("cannot cache an incomplete reference render")
    frame_count = len(trajectory.get("frames") or [])
    if frame_count <= 0 or any(
        len((render["views"][view].get("frames") or [])) != frame_count
        for view in REFERENCE_VIEWS
    ):
        raise ValueError("cannot cache a render whose views differ from its trajectory")
    write_json(
        cache / "cache_manifest.json",
        {
            "schema_version": REFERENCE_CACHE_SCHEMA,
            "completed": True,
            "input_sha256": input_digests,
            "frame_count": frame_count,
            "views": list(REFERENCE_VIEWS),
            "artifact_sha256": _cache_artifact_digests(cache),
        },
    )


def _measured_goal_cache_valid(cache: Path, input_digests: dict[str, str], trajectory: dict) -> bool:
    expected=input_digests.get("articulation_goal_contract")
    if expected is None:
        return True
    from .articulated_goal_contract import assess_measured_reference
    certificate=json.loads((cache/"simulation/measured_goal_certificate.json").read_text())
    goal=certificate["goal_contract"]
    if hashlib.sha256(json.dumps(goal,sort_keys=True).encode()).hexdigest()!=expected:
        return False
    measured=assess_measured_reference(trajectory,goal,trajectory["binding"])
    return measured["passed"] is True and measured==certificate


def _validate_completed_cache(cache: Path, input_digests: dict[str, str]) -> bool:
    manifest_path = cache / "cache_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        def execution_inputs(values):
            values = dict(values or {})
            if "execution_adapter_profile" in values:
                values.pop("materializer", None)
            return values
        if (
            payload.get("schema_version") != REFERENCE_CACHE_SCHEMA
            or payload.get("completed") is not True
            or execution_inputs(payload.get("input_sha256")) != execution_inputs(input_digests)
            or payload.get("views") != list(REFERENCE_VIEWS)
            or payload.get("artifact_sha256") != _cache_artifact_digests(cache)
        ):
            return False
        result = json.loads(
            (cache / "simulation" / "result.json").read_text(encoding="utf-8")
        )
        render = json.loads(
            (cache / "render" / "render_manifest.json").read_text(encoding="utf-8")
        )
        trajectory = json.loads(
            (cache / "simulation" / "trajectory.json").read_text(encoding="utf-8")
        )
        frame_count = len(trajectory.get("frames") or [])
        return bool(
            result.get("success") is True
            and _measured_goal_cache_valid(cache, input_digests, trajectory)
            and frame_count == payload.get("frame_count")
            and set(render.get("views", {})) == set(REFERENCE_VIEWS)
            and all(
                len(render["views"][view].get("frames") or []) == frame_count
                for view in REFERENCE_VIEWS
            )
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _cached_physical_manifest(cache: Path, source: dict) -> Path | None:
    manifests = sorted(cache.glob("physical_variant/*/physical_variant_manifest.json"))
    # Match fresh execution: a subject variant is assigned after a destination
    # variant. Every prepared asset is archived separately by the packager.
    for key in ("subject_id", "destination_id"):
        if source.get(key):
            selected = cache / "physical_variant" / source[key] / "physical_variant_manifest.json"
            if selected in manifests:
                return selected
    if len(manifests) > 1:
        raise ValueError("cached physical variants do not match the selected task objects")
    return manifests[0] if manifests else None


def _scene_object_records(house_state: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(house_state.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for room in payload.get("rooms", {}).values():
        for object_id, item in (room.get("objects") or {}).items():
            if isinstance(item, dict):
                result[str(object_id)] = item
    return result


def _object_positions(house_state: Path) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for object_id, item in _scene_object_records(house_state).items():
        translation = ((item.get("transform") or {}).get("translation") or [])
        if len(translation) >= 2:
            result[str(object_id)] = (float(translation[0]), float(translation[1]))
    return result


def _object_orientations(
    house_state: Path,
) -> dict[str, tuple[float, float, float, float]]:
    result: dict[str, tuple[float, float, float, float]] = {}
    for object_id, item in _scene_object_records(house_state).items():
        rotation = ((item.get("transform") or {}).get("rotation_wxyz") or [])
        if len(rotation) != 4:
            continue
        values = np.asarray(rotation, dtype=np.float64)
        norm = float(np.linalg.norm(values))
        if not np.isfinite(values).all() or norm <= 1.0e-9:
            raise ValueError(f"SceneCode object has invalid rotation: {object_id}")
        w, x, y, z = (float(value / norm) for value in values)
        result[object_id] = (x, y, z, w)
    return result


def _oriented_box_extent(
    low_xyz: Any,
    high_xyz: Any,
    orientation_xyzw: Any,
) -> tuple[float, float, float]:
    """Return world-axis extents of a local AABB under one quaternion."""

    low = np.asarray(low_xyz, dtype=np.float64)
    high = np.asarray(high_xyz, dtype=np.float64)
    quaternion = np.asarray(orientation_xyzw, dtype=np.float64)
    if (
        low.shape != (3,)
        or high.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or not np.isfinite(quaternion).all()
        or np.any(high <= low)
    ):
        raise ValueError("oriented bounds require finite non-degenerate 3D data")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-9:
        raise ValueError("oriented bounds quaternion is degenerate")
    x, y, z, w = quaternion / norm
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    extent = np.abs(rotation) @ (high - low)
    if not np.isfinite(extent).all() or np.any(extent <= 0.0):
        raise ValueError("oriented bounds produced a degenerate extent")
    return tuple(float(value) for value in extent)


def _quaternion_rotation_matrix(orientation_xyzw: Any) -> np.ndarray:
    quaternion = np.asarray(orientation_xyzw, dtype=np.float64)
    if (
        quaternion.shape != (4,)
        or not np.isfinite(quaternion).all()
        or float(np.linalg.norm(quaternion)) <= 1.0e-9
    ):
        raise ValueError("rotation matrix requires a finite quaternion")
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _closed_section_aperture(section: Any) -> tuple[np.ndarray, float]:
    """Return center and conservative radius of one inner section loop."""

    loops: list[tuple[float, np.ndarray]] = []
    vertices = np.asarray(section.vertices, dtype=np.float64)
    for entity in section.entities:
        if not bool(getattr(entity, "closed", False)):
            continue
        points = vertices[np.asarray(entity.points, dtype=np.int64), :2]
        if len(points) < 4:
            continue
        if not np.allclose(points[0], points[-1], atol=1.0e-8):
            points = np.vstack((points, points[0]))
        area = 0.5 * abs(
            float(
                np.dot(points[:-1, 0], points[1:, 1])
                - np.dot(points[1:, 0], points[:-1, 1])
            )
        )
        loops.append((area, points[:-1]))
    if len(loops) < 2:
        raise ValueError("insert destination has no closed inner aperture loop")
    loops.sort(key=lambda item: item[0], reverse=True)
    outer_area = loops[0][0]
    inner = [item for item in loops[1:] if 0.0 < item[0] < 0.8 * outer_area]
    if len(inner) != 1:
        raise ValueError(
            "insert destination must expose exactly one vertical aperture"
        )
    _area, points = inner[0]
    center = points.mean(axis=0)
    radius = float(np.min(np.linalg.norm(points - center, axis=1)))
    if not np.isfinite(center).all() or not math.isfinite(radius) or radius <= 1.0e-4:
        raise ValueError("insert destination aperture is degenerate")
    return center, radius


def _sdf_vertical_aperture(sdf_path: Path) -> dict[str, Any]:
    """Measure one through-aperture from the exact SceneCode visual mesh."""

    root = ET.parse(sdf_path).getroot()
    visuals = [
        node
        for node in root.findall(".//visual")
        if node.find("geometry/mesh/uri") is not None
    ]
    if len(visuals) != 1:
        raise ValueError("insert destination requires exactly one visual mesh")
    visual = visuals[0]
    uri = str(visual.findtext("geometry/mesh/uri"))
    mesh_path = Path(uri)
    if not mesh_path.is_absolute():
        mesh_path = (sdf_path.parent / mesh_path).resolve()
    if not mesh_path.is_file() or mesh_path.suffix.lower() not in {".glb", ".gltf"}:
        raise ValueError("insert aperture requires one local glTF visual mesh")
    import trimesh

    loaded = trimesh.load(mesh_path, force="scene", process=False)
    mesh = (
        trimesh.util.concatenate(tuple(loaded.geometry.values()))
        if isinstance(loaded, trimesh.Scene)
        else loaded
    )
    # glTF is Y-up; SceneCode's SDF/Drake convention is Z-up.
    mesh.apply_transform(
        np.asarray(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
    )
    scale_text = visual.findtext("geometry/mesh/scale", "1 1 1")
    scale = np.asarray([float(value) for value in scale_text.split()], dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("insert destination visual scale is invalid")
    mesh.apply_scale(scale)
    low, high = np.asarray(mesh.bounds, dtype=np.float64)
    height = float(high[2] - low[2])
    if not math.isfinite(height) or height <= 1.0e-4:
        raise ValueError("insert destination visual height is degenerate")
    # A destination is not required to be a through-hole.  The vertical closed
    # aperture this module demanded in its first revision was an invention of
    # our own pipeline -- no upstream SceneCode contract, exactly one caller --
    # and it failed every legitimate open destination, a hook above all.
    # Measure what the destination actually offers and record which kind of
    # opening it has instead of promoting one shape into an admission gate.
    measurements = []
    for fraction in (0.50, 0.98):
        z = float(low[2] + fraction * height)
        section = mesh.section(
            plane_origin=(0.0, 0.0, z), plane_normal=(0.0, 0.0, 1.0)
        )
        if section is None:
            continue
        try:
            measurements.append(_closed_section_aperture(section))
        except ValueError:
            continue
    aperture_kind = "open_destination"
    center = None
    radius = None
    if len(measurements) == 2:
        centers = np.stack([item[0] for item in measurements])
        mean_center = centers.mean(axis=0)
        if float(np.max(np.linalg.norm(centers - mean_center, axis=1))) <= 0.001:
            aperture_kind = "through_hole"
            center = mean_center
            radius = min(item[1] for item in measurements)
    if center is None:
        # Open destination: the subject is hung on or lowered onto the
        # structure itself, so the horizontal target is the centre of the
        # destination's own footprint and the clearance is half its narrower
        # horizontal extent.
        center = np.asarray(
            [(low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0], dtype=np.float64
        )
        radius = 0.5 * float(min(high[0] - low[0], high[1] - low[1]))
    return {
        "schema_version": "robotics_vertical_aperture_measurement.v1",
        "source_visual_mesh": str(mesh_path),
        "aperture_kind": aperture_kind,
        "aperture_measured": aperture_kind == "through_hole",
        "center_local_xy_m": center.tolist(),
        "radius_m": radius,
        "diameter_m": 2.0 * radius,
        "sampled_height_fractions": [0.50, 0.98],
    }


def _terminal_alignment_axes(
    *,
    terminal_operator: str,
    primary_axis: int,
    thin_axis: int,
    upright: bool = False,
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """Compile semantic terminal orientation into object-local/world axes."""

    if primary_axis not in {0, 1, 2} or thin_axis not in {0, 1, 2}:
        raise ValueError("terminal alignment axis index must be a 3D basis index")
    if terminal_operator == "insert":
        local_index = primary_axis
    elif terminal_operator == "cover":
        local_index = thin_axis
    elif upright and terminal_operator in {"place", "stack", "flip"}:
        # The instruction requires an upright/standing final placement: align the
        # manipuland's dominant axis with world up and certify the measured
        # terminal alignment cosine instead of leaving the placement implicit.
        local_index = primary_axis
    else:
        return None, None
    local_axis = tuple(1.0 if index == local_index else 0.0 for index in range(3))
    return local_axis, (0.0, 0.0, 1.0)


def _primary_support_id(
    *, task: TaskInstructionSpec, matches: dict[str, dict[str, Any]], house_state: Path
) -> str:
    """Select the task support that owns a real SceneCode support surface."""

    records = _scene_object_records(house_state)
    candidates = []
    for item in task.objects:
        if item.role != "support":
            continue
        match = matches[item.object_id]
        record = records.get(str(match["scene_object_id"])) or {}
        surfaces = record.get("support_surfaces") or []
        if isinstance(surfaces, list) and surfaces:
            candidates.append(item.object_id)
    if len(candidates) != 1:
        raise ValueError(
            "DROID reference requires exactly one matched task support with a "
            f"declared SceneCode support surface, found {sorted(candidates)}"
        )
    return candidates[0]


def _validate_sources(
    *, output_root: Path, batch_id: str, case_id: str,
    compilation_plan: Path | None = None,
    asset_revision: Path | None = None,
) -> dict[str, Any]:
    batch_path = output_root / "robotics_task_batches" / batch_id / "batch_manifest.json"
    batch = RoboticsTaskBatchManifest.model_validate_json(
        batch_path.read_text(encoding="utf-8")
    )
    records = [record for record in batch.cases if record.case_id == case_id]
    if len(records) != 1:
        raise ValueError("selected batch must contain the requested case exactly once")
    record = records[0]
    task_path = (output_root / record.task_path).resolve()
    if sha256_file(task_path) != record.task_sha256:
        raise ValueError("selected task file digest changed")
    task = TaskInstructionSpec.model_validate_json(task_path.read_text(encoding="utf-8"))
    if task.requested_wams != [MODEL_ADAPTER]:
        raise ValueError("this reference adapter requires an exact Cosmos-only task request")

    plan_path = (compilation_plan.resolve() if compilation_plan is not None else
                 output_root / "robotics_compilation_plans" / batch_id / f"{case_id}.json")
    if compilation_plan is not None:
        plan_path.relative_to((output_root / "robotics_compilation_revisions" / case_id).resolve())
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("case_id") != case_id or plan.get("task_sha256") != task.content_hash():
        raise ValueError("compiled task graph differs from the selected task")
    task_intent_review = ensure_task_intent(task=task, plan=plan, output_root=output_root)
    operators = [item.get("operator") for item in plan.get("controller_program", [])]
    try:
        compiled_steps = compiled_execution_steps(
            plan, verified_operators=frozenset({"pour"}) if asset_revision else frozenset())
    except MissingExecutionCapability as exc:
        raise MissingExecutionCapability(
            f"{HARNESS_ID} does not yet implement operator sequence {operators}: {exc}"
        ) from exc
    except ValueError as exc:
        raise ValueError(
            f"{HARNESS_ID} does not yet implement operator sequence {operators}: {exc}"
        ) from exc

    generation_root = output_root / "robotics_scenecode_generations" / batch_id / case_id
    generation_path = generation_root / "generation_manifest.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    identity_path = generation_root / "asset_identity" / "semantic_identity_verdict.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        generation.get("task_sha256") != record.task_sha256
        or generation.get("passed") is not True
        or generation.get("semantic_identity_passed") is not True
        or identity.get("overall_passed") is not True
    ):
        raise ValueError("SceneCode candidate has not passed semantic identity admission")
    if (generation.get("asset_identity") or {}).get(
        "semantic_verdict_sha256"
    ) != sha256_file(identity_path):
        raise ValueError("SceneCode semantic identity digest changed")
    inventory_path = generation_root / "task_inventory_certificate.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("passed") is not True:
        raise ValueError("SceneCode task inventory is not passing")
    matches = {item["task_object_id"]: item for item in inventory.get("matches", [])}
    task_ids = {item.object_id for item in task.objects}
    if set(matches) != task_ids:
        raise ValueError("SceneCode task inventory does not cover exact task objects")
    for object_id, item in matches.items():
        path = Path(item["sdf_path"])
        if not path.is_file() or sha256_file(path) != item.get("sdf_sha256"):
            raise ValueError(f"SceneCode SDF digest changed: {object_id}")

    house_state = Path(inventory["house_state"])
    if not house_state.is_file() or sha256_file(house_state) != inventory.get(
        "house_state_sha256"
    ):
        raise ValueError("SceneCode house state digest changed")
    positions = _object_positions(house_state)
    orientations = _object_orientations(house_state)
    identity_report_path = generation_root / "asset_identity" / "asset_identity_report.json"
    identity_report = json.loads(identity_report_path.read_text(encoding="utf-8"))
    from scripts.render_scenecode_asset_identity import sdf_collision_bounds

    articulated_steps = [
        step
        for step in compiled_steps
        if step["terminal_operator"] in {"pull", "press", "close"}
    ]
    if articulated_steps:
        if len(compiled_steps) != 1 or len(articulated_steps) != 1:
            raise ValueError(
                "articulated reference currently requires one standalone articulated step"
            )
        step = articulated_steps[0]
        component = next(
            (
                item
                for item in task.articulation_components
                if item.component_id == step["subject_id"]
            ),
            None,
        )
        if component is None:
            raise ValueError(
                f"articulated step references unknown component {step['subject_id']!r}"
            )
        if component.host_object_id != step["destination_id"]:
            raise ValueError("articulation step target differs from its declared host")
        binding_path = (
            output_root
            / "robotics_articulation_bindings"
            / batch_id
            / case_id
            / f"{component.component_id}.json"
        )
        if not binding_path.is_file():
            raise FileNotFoundError(
                f"articulation binding certificate is missing: {binding_path}"
            )
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if binding.get("passed") is not True:
            raise ValueError("articulation binding certificate is not passing")
        host = matches.get(component.host_object_id)
        if host is None or host.get("status") != "matched":
            raise ValueError("SceneCode inventory has no matched articulated host")
        host_sdf = Path(str(host["sdf_path"])).resolve()
        if not host_sdf.is_file() or sha256_file(host_sdf) != host.get("sdf_sha256"):
            raise ValueError("articulated host SDF digest changed")
        if binding.get("host_sdf_sha256") != host.get("sdf_sha256"):
            raise ValueError("articulation binding does not bind the inventory host SDF")
        host_scene_id = str(host["scene_object_id"])
        if host_scene_id not in positions or host_scene_id not in orientations:
            raise ValueError("articulated host placement or orientation is missing")
        source_host_xy = tuple(float(value) for value in positions[host_scene_id])
        host_xy, articulated_layout = normalized_articulated_layout(source_host_xy)
        host_orientation = tuple(
            float(value) for value in orientations[host_scene_id]
        )
        # Keep the certified articulated geometry and handle at native size.
        # The actual robot mount is selected by the articulated executor, so a
        # pick/place camera prediction here uses the wrong base pose. After the
        # physical reference, fit the fixed lenses to its whole motion and
        # freeze that rig before rendering or running a WAM.
        host_bounds_low, host_bounds_high = sdf_collision_bounds(host_sdf)
        host_world_extent = _oriented_box_extent(
            host_bounds_low, host_bounds_high, host_orientation
        )
        articulated_layout["source_host_local_extent_xyz"] = [
            float(value) for value in host_world_extent
        ]
        articulated_layout["native_view_fit"] = {
            "policy": "fit_registered_fixed_lenses_to_reference_motion",
            "source_host_extent_xyz_m": [float(value) for value in host_world_extent],
            "asset_rescaled": False,
        }
        execution_step = {
            **step,
            "subject_xy": host_xy,
            "destination_xy": host_xy,
            "subject_orientation_xyzw": host_orientation,
            "destination_orientation_xyzw": host_orientation,
            "articulation_component_id": component.component_id,
            "articulation_host_object_id": component.host_object_id,
            "articulation_binding": binding,
            "articulation_binding_path": str(binding_path),
            "articulated_host_sdf": str(host_sdf),
        }
        return {
            "batch": batch,
            "record": record,
            "task": task,
            "task_path": task_path,
            "plan_path": plan_path,
            "task_intent_review_path": task_intent_review,
            "generation_root": generation_root,
            "generation_path": generation_path,
            "identity_path": identity_path,
            "matches": matches,
            # Articulated hosts are themselves the physical support and do
            # not require a second task-object table with a support surface.
            "support_id": None,
            "subject_id": component.component_id,
            "destination_id": component.host_object_id,
            "terminal_operator": step["terminal_operator"],
            "operator_sequence": operators,
            "terminal_predicate": step["terminal_predicate"],
            "subject_xy": host_xy,
            "destination_xy": host_xy,
            "subject_orientation_xyzw": host_orientation,
            "destination_orientation_xyzw": host_orientation,
            "execution_steps": [execution_step],
            "step_object_ids": [component.host_object_id],
            "is_articulated": True,
            "articulation_binding_path": str(binding_path),
            "articulation_binding": binding,
            "articulated_host_sdf": str(host_sdf),
            "identity_report": identity_report,
            "layout": articulated_layout,
        }

    support_id = _primary_support_id(
        task=task, matches=matches, house_state=house_state
    )
    execution_steps = []
    step_object_ids: set[str] = set()
    for step in compiled_steps:
        subject_id = step["subject_id"]
        destination_id = step["destination_id"]
        subject_scene_id = matches[subject_id]["scene_object_id"]
        destination_scene_id = matches[destination_id]["scene_object_id"]
        if subject_scene_id not in positions or destination_scene_id not in positions:
            raise ValueError("SceneCode task-object placement is missing")
        if (
            subject_scene_id not in orientations
            or destination_scene_id not in orientations
        ):
            raise ValueError("SceneCode task-object orientation is missing")
        bounds = identity_report["results"][subject_id]
        low, high = bounds["bounds_low"], bounds["bounds_high"]
        subject_local_extent = tuple(
            float(high[axis]) - float(low[axis]) for axis in range(3)
        )
        subject_world_extent = _oriented_box_extent(
            low,
            high,
            orientations[subject_scene_id],
        )
        horizontal_extent = max(
            subject_world_extent[0],
            subject_world_extent[1],
        )
        # The DROID parallel-jaw aperture only has to span the manipuland along
        # the jaw closing axis, so the grasp axis is the narrower of the two
        # horizontal world axes.  Sizing and grasping across that axis keeps the
        # manipuland as large as the gripper physically allows instead of
        # shrinking it (and grasping it) across its widest feature (a kettle's
        # spout, a handle, a long flange, ...).
        grasp_horizontal_axis = (
            0 if subject_world_extent[0] <= subject_world_extent[1] else 1
        )
        grasp_horizontal_extent = subject_world_extent[grasp_horizontal_axis]
        # yaw 0.0 closes the jaws along world +Y, yaw 90.0 along world +X.
        grasp_yaw_deg = 0.0 if grasp_horizontal_axis == 1 else 90.0
        subject_height = subject_world_extent[2]
        destination_bounds = identity_report["results"][destination_id]
        destination_local_extent = tuple(
            float(destination_bounds["bounds_high"][axis])
            - float(destination_bounds["bounds_low"][axis])
            for axis in range(3)
        )
        destination_collision_low, destination_collision_high = (
            sdf_collision_bounds(Path(matches[destination_id]["sdf_path"]))
        )
        destination_collision_extent = _oriented_box_extent(
            destination_collision_low,
            destination_collision_high,
            orientations[destination_scene_id],
        )
        destination_horizontal_support_extent = min(
            destination_collision_extent[0],
            destination_collision_extent[1],
        )
        destination_height = _oriented_box_extent(
            destination_bounds["bounds_low"],
            destination_bounds["bounds_high"],
            orientations[destination_scene_id],
        )[2]
        primary_axis = int(np.argmax(np.asarray(subject_local_extent)))
        thin_axis = int(np.argmin(np.asarray(subject_local_extent)))
        subject_cross_section_extent = max(
            subject_local_extent[axis] for axis in range(3) if axis != primary_axis
        )
        terminal_alignment_axis_local, terminal_alignment_axis_world = (
            _terminal_alignment_axes(
                terminal_operator=step["terminal_operator"],
                primary_axis=primary_axis,
                thin_axis=thin_axis,
                upright=bool(step.get("terminal_upright")),
            )
        )
        insert_aperture = None
        terminal_target_offset_world_xy = (0.0, 0.0)
        if step["terminal_operator"] == "insert":
            insert_aperture = _sdf_vertical_aperture(
                Path(matches[destination_id]["sdf_path"])
            )
            local_offset = np.asarray(
                [*insert_aperture["center_local_xy_m"], 0.0], dtype=np.float64
            )
            world_offset = _quaternion_rotation_matrix(
                orientations[destination_scene_id]
            ) @ local_offset
            terminal_target_offset_world_xy = (
                float(world_offset[0]),
                float(world_offset[1]),
            )
        if not all(
            math.isfinite(value) and value > 0
            for value in (
                horizontal_extent,
                subject_height,
                destination_horizontal_support_extent,
                destination_height,
            )
        ):
            raise ValueError("operator-step asset geometry is invalid")
        execution_steps.append(
            {
                **step,
                "subject_xy": positions[subject_scene_id],
                "destination_xy": positions[destination_scene_id],
                "subject_orientation_xyzw": orientations[subject_scene_id],
                "destination_orientation_xyzw": orientations[
                    destination_scene_id
                ],
                "horizontal_extent": horizontal_extent,
                "grasp_horizontal_extent": grasp_horizontal_extent,
                "grasp_horizontal_axis": grasp_horizontal_axis,
                "grasp_yaw_deg": grasp_yaw_deg,
                "destination_horizontal_support_extent": (
                    destination_horizontal_support_extent
                ),
                "subject_height": subject_height,
                "destination_height": destination_height,
                "destination_local_extent_xyz": destination_local_extent,
                "subject_local_extent_xyz": subject_local_extent,
                "subject_primary_axis_local": primary_axis,
                "subject_thin_axis_local": thin_axis,
                "subject_cross_section_extent_m": subject_cross_section_extent,
                "terminal_alignment_axis_local": terminal_alignment_axis_local,
                "terminal_alignment_axis_world": terminal_alignment_axis_world,
                "terminal_target_offset_world_xy_m": (
                    terminal_target_offset_world_xy
                ),
                "insert_aperture": insert_aperture,
            }
        )
        step_object_ids.update({subject_id, destination_id})
    first_step = execution_steps[0]
    return {
        "batch": batch,
        "record": record,
        "task": task,
        "task_path": task_path,
        "plan_path": plan_path,
        "task_intent_review_path": task_intent_review,
        "generation_root": generation_root,
        "generation_path": generation_path,
        "identity_path": identity_path,
        "matches": matches,
        "support_id": support_id,
        "subject_id": first_step["subject_id"],
        "destination_id": first_step["destination_id"],
        "terminal_operator": first_step["terminal_operator"],
        "operator_sequence": operators,
        "terminal_predicate": first_step["terminal_predicate"],
        "subject_xy": first_step["subject_xy"],
        "destination_xy": first_step["destination_xy"],
        "subject_orientation_xyzw": first_step["subject_orientation_xyzw"],
        "destination_orientation_xyzw": first_step[
            "destination_orientation_xyzw"
        ],
        "horizontal_extent": first_step["horizontal_extent"],
        "grasp_horizontal_extent": first_step["grasp_horizontal_extent"],
        "grasp_horizontal_axis": first_step["grasp_horizontal_axis"],
        "grasp_yaw_deg": first_step["grasp_yaw_deg"],
        "destination_horizontal_support_extent": first_step[
            "destination_horizontal_support_extent"
        ],
        "subject_height": first_step["subject_height"],
        "destination_height": first_step["destination_height"],
        "destination_local_extent_xyz": first_step[
            "destination_local_extent_xyz"
        ],
        "subject_local_extent_xyz": first_step["subject_local_extent_xyz"],
        "subject_cross_section_extent_m": first_step[
            "subject_cross_section_extent_m"
        ],
        "terminal_alignment_axis_local": first_step[
            "terminal_alignment_axis_local"
        ],
        "terminal_alignment_axis_world": first_step[
            "terminal_alignment_axis_world"
        ],
        "terminal_upright_min_cosine": first_step.get(
            "terminal_upright_min_cosine"
        ),
        "terminal_target_offset_world_xy_m": first_step[
            "terminal_target_offset_world_xy_m"
        ],
        "insert_aperture": first_step["insert_aperture"],
        "execution_steps": execution_steps,
        "step_object_ids": sorted(step_object_ids),
        "is_articulated": False,
    }


def _robot_asset(output_root: Path) -> tuple[Path, Path]:
    return validate_robot_asset(output_root)


def _package_completed_cache(
    *,
    python: Path,
    repository: Path,
    env: dict[str, str],
    output_root: Path,
    batch_id: str,
    case_id: str,
    source: dict[str, Any],
    simulation: Path,
    render: Path,
    physical_manifest: Path | None,
    visual_manifest: Path | None,
    package: Path,
    final: Path,
    grasp_band_evidence: Path | None = None,
) -> Path:
    prompt_root = output_root / "robotics_scene_prompts" / batch_id / case_id
    package_args = [
        str(python),
        str(repository / "scripts" / "package_robotics_reference.py"),
        "--task",
        str(source["task_path"]),
        "--trajectory-json",
        str(simulation / "trajectory.json"),
        "--simulation-result",
        str(simulation / "result.json"),
        "--render-manifest",
        str(render / "render_manifest.json"),
        "--scene-prompt-bundle",
        str(prompt_root),
        "--scenecode-release",
        f"{batch_id}/{case_id}",
        "--harness-id",
        HARNESS_ID,
        "--task-graph",
        str(source["plan_path"]),
        "--generation-manifest",
        str(source["generation_path"]),
        "--identity-certificate",
        str(source["identity_path"]),
        "--output-dir",
        str(package),
    ]
    if physical_manifest is not None:
        package_args.extend(["--physical-variant-manifest", str(physical_manifest)])
    if visual_manifest is not None:
        package_args.extend(["--visual-variant-manifest", str(visual_manifest)])
    if grasp_band_evidence is not None and grasp_band_evidence.is_dir():
        package_args.extend(["--grasp-band-evidence", str(grasp_band_evidence)])
    _run(package_args, cwd=repository, env=env)
    manifest_path, manifest = load_reference_manifest(
        package / "robotics_reference_manifest.json"
    )
    if manifest.schema_version != "robotics_reference_bundle.v6":
        raise ValueError("packager did not emit agentic reference v6")
    package.replace(final)
    return final / manifest_path.name


def _prepare_reference_prerequisites(*, output_root: Path, batch_id: str, case_id: str,
                                     compilation_plan: Path | None = None) -> None:
    """Build missing deterministic prerequisites once before reference admission.

    Published plans/certificates are still checked by _validate_sources; this
    does not replace mismatched evidence or bypass a failed physical gate.
    """
    from .robotics_task_compiler import compile_batch
    from .robotics_articulation import certify_case_articulation

    batch_path = output_root / "robotics_task_batches" / batch_id / "batch_manifest.json"
    batch = RoboticsTaskBatchManifest.model_validate_json(batch_path.read_text(encoding="utf-8"))
    records = [record for record in batch.cases if record.case_id == case_id]
    if len(records) != 1:
        raise ValueError("selected batch must contain the requested case exactly once")
    task_path = (output_root / records[0].task_path).resolve()
    task_path.relative_to(output_root)
    if sha256_file(task_path) != records[0].task_sha256:
        raise ValueError("selected task file digest changed")
    task = TaskInstructionSpec.model_validate_json(task_path.read_text(encoding="utf-8"))
    plan_path = output_root / "robotics_compilation_plans" / batch_id / f"{case_id}.json"
    if compilation_plan is None and not plan_path.exists():
        compile_batch(batch_manifest=batch_path, output_root=output_root)
    for component in task.articulation_components:
        binding = output_root / "robotics_articulation_bindings" / batch_id / case_id / f"{component.component_id}.json"
        certify_case_articulation(output_root=output_root, batch_id=batch_id,
                                  case_id=case_id, component_id=component.component_id)


def _certify_published_graph_compatibility(
    *, final: Path, source: dict[str, Any], output_root: Path,
    manifest_sha256: str,
) -> Path:
    """Admit a frozen older plan only when its task and execution agree."""
    old_path = final / "agentic_provenance/task_graph.json"
    new_path = Path(source["plan_path"])
    if sha256_file(final / "task_instruction.json") != sha256_file(source["task_path"]):
        raise ValueError("published reference task bytes differ from the selected task")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    if (old.get("case_id") != new.get("case_id")
            or old.get("schema_version") != new.get("schema_version")
            or new.get("status") != "static_plan_validated"
            or old.get("requested_wams") != new.get("requested_wams")):
        raise ValueError("published and selected task graphs differ in identity")

    added_upright = []
    def compare_program(key: str, status_key: str) -> None:
        before = old.get(key) or []
        after = new.get(key) or []
        if len(before) != len(after):
            raise ValueError(f"published task graph changes {key} length")
        for prior, current in zip(before, after, strict=True):
            prior = dict(prior)
            current = dict(current)
            if current.pop(status_key, None) not in {"native_registered", "registered"}:
                raise ValueError("selected graph capability is not independently registered")
            old_constraints = (
                prior.setdefault("constraints", {}) if key == "controller_program"
                else prior
            )
            new_constraints = (
                current.setdefault("constraints", {}) if key == "controller_program"
                else current
            )
            keys = ("terminal_upright", "terminal_upright_axis_world",
                    "terminal_upright_min_cosine")
            if (not old_constraints.get("terminal_upright")
                    and new_constraints.get("terminal_upright") is True):
                if not all(item in new_constraints for item in keys):
                    raise ValueError("selected upright constraint is incomplete")
                added_upright.append({item: new_constraints[item] for item in keys})
                for item in keys:
                    new_constraints.pop(item)
            if prior != current:
                raise ValueError(f"published task graph changes {key} semantics")
    compare_program("controller_program", "capability_status")
    compare_program("success_program", "evaluator_status")

    evidence = None
    if added_upright:
        first = added_upright[0]
        if any(item != first for item in added_upright):
            raise ValueError("selected graph has inconsistent upright constraints")
        from scipy.spatial.transform import Rotation
        trajectory_path = final / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        frames = trajectory.get("frames") or []
        state = (frames[-1].get("cup") or {}) if frames else {}
        quaternion = np.asarray(state.get("orientation_xyzw"), dtype=np.float64)
        local = np.asarray(source.get("terminal_alignment_axis_local"), dtype=np.float64)
        world = np.asarray(first["terminal_upright_axis_world"], dtype=np.float64)
        threshold = float(first["terminal_upright_min_cosine"])
        if (quaternion.shape != (4,) or local.shape != (3,) or world.shape != (3,)
                or not np.isfinite(quaternion).all() or not np.isfinite(local).all()
                or not np.isfinite(world).all() or not 0.0 < threshold <= 1.0
                or np.linalg.norm(local) < 1e-9 or np.linalg.norm(world) < 1e-9):
            raise ValueError("selected upright axis or frozen pose is invalid")
        observed = Rotation.from_quat(quaternion).apply(local)
        cosine = float(np.dot(observed, world) / (
            np.linalg.norm(observed) * np.linalg.norm(world)
        ))
        if cosine < threshold:
            raise ValueError("published physical trajectory does not satisfy new upright constraint")
        evidence = {
            "terminal_upright_cosine": cosine,
            "minimum_cosine": threshold,
            "trajectory_sha256": sha256_file(trajectory_path),
        }
    output = (output_root / "harness_refactor/reference_graph_compatibility"
              / str(source["task"].case_id) / f"{sha256_file(new_path)[:16]}.json")
    return write_json(output, {
        "schema_version": "robotics_published_graph_compatibility.v1",
        "passed": True,
        "published_reference_manifest_sha256": manifest_sha256,
        "published_task_graph_sha256": sha256_file(old_path),
        "selected_task_graph_sha256": sha256_file(new_path),
        "unchanged_task_file_sha256": sha256_file(source["task_path"]),
        "proof": "same frozen task bytes and execution/evaluator programs",
        "added_upright_evidence": evidence,
    })


def _reuse_published_reference_without_executor(
    *, output_root: Path, batch_id: str, case_id: str,
    selected_plan: Path | None,
) -> Path:
    """Admit an archived execution only for the same frozen task and plan."""
    batch_path = output_root / "robotics_task_batches" / batch_id / "batch_manifest.json"
    batch = RoboticsTaskBatchManifest.model_validate_json(
        batch_path.read_text(encoding="utf-8")
    )
    rows = [item for item in batch.cases if item.case_id == case_id]
    if len(rows) != 1:
        raise ValueError("published reference has no unambiguous frozen batch row")
    row = rows[0]
    task_path = (output_root / row.task_path).resolve()
    task_path.relative_to(output_root)
    if sha256_file(task_path) != row.task_sha256:
        raise ValueError("frozen task file digest changed")
    task = TaskInstructionSpec.model_validate_json(task_path.read_text(encoding="utf-8"))
    final = output_root / "robotics_reference_bundles" / row.reference_run_id
    manifest_path, manifest = load_reference_manifest(
        final / "robotics_reference_manifest.json"
    )
    if (manifest.schema_version != "robotics_reference_bundle.v6"
            or manifest.case_id != case_id
            or manifest.quality_passed is not True
            or manifest.physical_render_passed is not True
            or manifest.visual_context_passed is not True):
        raise ValueError("published reference is not independently certified")
    if sha256_file(final / "task_instruction.json") != sha256_file(task_path):
        raise ValueError("published reference changes the frozen task bytes")
    plan_path = selected_plan or (
        output_root / "robotics_compilation_plans" / batch_id / f"{case_id}.json"
    )
    if manifest.task_graph_sha256 != sha256_file(plan_path):
        _certify_published_graph_compatibility(
            final=final,
            source={"plan_path": plan_path, "task_path": task_path,
                    "task": task, "terminal_alignment_axis_local": None},
            output_root=output_root, manifest_sha256=sha256_file(manifest_path),
        )
    return manifest_path


def materialize_reference(
    *, workspace: str | Path, repository: str | Path, batch_id: str, case_id: str,
    articulation_open_fraction_required: float | None = None,
    compilation_plan: str | Path | None = None,
    asset_revision: str | Path | None = None,
    visual_revision_manifest: str | Path | None = None,
    visual_revision_verdict: str | Path | None = None,
    execution_profile: str | Path | None = None,
) -> Path:
    if articulation_open_fraction_required is not None and not 0.5 <= articulation_open_fraction_required <= 1.0:
        raise ValueError("articulation acceptance fraction must lie in [0.5, 1.0]")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", batch_id):
        raise ValueError("unsafe batch id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", case_id):
        raise ValueError("unsafe case id")
    workspace = Path(workspace).resolve()
    repository = Path(repository).resolve()
    output_root = workspace / "outputs"
    selected_execution_profile = Path(execution_profile).resolve() if execution_profile is not None else None
    if selected_execution_profile is not None:
        selected_execution_profile.relative_to((output_root / "robotics_reference_execution_profiles" / case_id).resolve())
    selected_plan = Path(compilation_plan).resolve() if compilation_plan is not None else None
    selected_asset_revision = Path(asset_revision).resolve() if asset_revision is not None else None
    selected_visual_manifest = (Path(visual_revision_manifest).resolve()
                                if visual_revision_manifest is not None else None)
    selected_visual_verdict = (Path(visual_revision_verdict).resolve()
                               if visual_revision_verdict is not None else None)
    if (selected_visual_manifest is None) != (selected_visual_verdict is None):
        raise ValueError("visual revision manifest and independent verdict must be paired")
    if selected_plan is not None:
        selected_plan.relative_to((output_root / "robotics_compilation_revisions" / case_id).resolve())
    _prepare_reference_prerequisites(output_root=output_root, batch_id=batch_id,
                                     case_id=case_id, compilation_plan=selected_plan)
    try:
        source = _validate_sources(
            output_root=output_root, batch_id=batch_id, case_id=case_id,
            compilation_plan=selected_plan,
            asset_revision=selected_asset_revision,
        )
    except MissingExecutionCapability:
        if (selected_execution_profile is not None or selected_asset_revision is not None or selected_visual_manifest is not None
                or selected_visual_verdict is not None):
            raise
        return _reuse_published_reference_without_executor(
            output_root=output_root, batch_id=batch_id, case_id=case_id,
            selected_plan=selected_plan,
        )
    is_pour = source["operator_sequence"] == ["pick", "pour"]
    if is_pour != (selected_asset_revision is not None):
        raise MissingExecutionCapability(
            "pour requires a separately verified graspable asset revision")
    pour_case = None
    if is_pour:
        from .robotics_asset_derivative_verifier import verify_derivative
        from .robotics_pour_replay import resolve_pour_case
        assert selected_asset_revision is not None
        verdict_path = selected_asset_revision / "derivative_verdict.json"
        verify_derivative(selected_asset_revision, verdict_path)
        pour_case = resolve_pour_case(
            workspace=workspace, task_path=source["task_path"],
            generation_path=source["generation_path"],
            asset_revision=selected_asset_revision,
        )
        source["pour_asset_revision"] = selected_asset_revision
        source["pour_asset_derivative_verdict"] = verdict_path
    if selected_visual_manifest is not None:
        from .robotics_visual_revision_verifier import verify_visual_revision
        assert selected_visual_verdict is not None
        allowed_revision_roots = (
            output_root / "harness_refactor/candidates",
            output_root / "agentic_rsi/candidates",
        )
        if not any(root.resolve() in selected_visual_manifest.parents
                   for root in allowed_revision_roots):
            raise ValueError("visual revision must be isolated under candidate outputs")
        if selected_visual_verdict != (
            selected_visual_manifest.parent.parent / "visual_revision_verdict.json"
        ).resolve():
            raise ValueError("visual revision verdict must be beside its asset directory")
        declared = json.loads(selected_visual_manifest.read_text())
        source_sdf = Path(declared["source_sdf"]).resolve()
        matching = [object_id for object_id, item in source["matches"].items()
                    if Path(item["sdf_path"]).resolve() == source_sdf]
        if len(matching) != 1:
            raise ValueError("visual revision source does not name one certified task asset")
        verify_visual_revision(
            manifest_path=selected_visual_manifest,
            output=selected_visual_verdict, expected_source_sdf=source_sdf)
        source["visual_revision"] = {
            "object_id": matching[0],
            "sdf": selected_visual_manifest.parent / source_sdf.name,
            "manifest": selected_visual_manifest,
            "verdict": selected_visual_verdict,
        }
    record = source["record"]
    run_id = record.reference_run_id
    final = output_root / "robotics_reference_bundles" / run_id
    cache = output_root / "robotics_reference_caches" / run_id
    failure_path = output_root / "robotics_reference_failures" / f"{run_id}.json"
    if final.is_dir():
        if selected_execution_profile is not None:
            raise ValueError("an admitted reference is immutable; execution profiles only repair unadmitted references")
        manifest_path, manifest = load_reference_manifest(
            final / "robotics_reference_manifest.json"
        )
        if manifest.schema_version != "robotics_reference_bundle.v6":
            raise ValueError("cached agentic reference uses another schema")
        if manifest.task_graph_sha256 != sha256_file(source["plan_path"]):
            _certify_published_graph_compatibility(
                final=final, source=source, output_root=output_root,
                manifest_sha256=sha256_file(manifest_path),
            )
        for label, published, current in (
            ("SceneCode generation", manifest.scene_generation_manifest_sha256, source["generation_path"]),
            ("asset identity", manifest.semantic_identity_certificate_sha256, source["identity_path"]),
        ):
            if published != sha256_file(current):
                raise ValueError(f"published reference binds another {label}")
        if manifest.visual_variant_manifest:
            from scripts.package_robotics_reference import _validate_visual_variant_manifest
            _validate_visual_variant_manifest(final / manifest.visual_variant_manifest, final)
        if selected_visual_manifest is not None:
            if not manifest.visual_variant_manifest:
                raise ValueError("published reference has no selected visual revision")
            published_revision = final / manifest.visual_variant_manifest
            original = json.loads(selected_visual_manifest.read_text())
            bundled = json.loads(published_revision.read_text())
            if ({key: value for key, value in original.items() if key != "source_sdf"}
                    != {key: value for key, value in bundled.items() if key != "source_sdf"}):
                raise ValueError("published reference binds another visual revision")
        if is_pour:
            assert pour_case is not None
            packaged_result = json.loads((final / "simulation_result.json").read_text())
            if packaged_result.get("asset_revision_sha256") != pour_case["asset_revision_sha256"]:
                raise ValueError("published pour reference binds another asset revision")
        shutil.rmtree(cache, ignore_errors=True)
        return manifest_path
    if final.exists():
        raise RuntimeError(f"incomplete reference destination requires inspection: {final}")

    staging = final.parent / f".{run_id}.partial.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    work = cache
    simulation = work / "simulation"
    render = work / "render"
    package = staging / "bundle"
    staging.mkdir(parents=True)
    measurements = staging / "measurements"
    measurements.mkdir(parents=True)
    work_for_measurements = measurements
    stage = "source_validation"
    cache_completed = False
    try:
        stage = "embodiment_compatibility"
        robot_manifest, robot_urdf = _robot_asset(output_root)
        task: TaskInstructionSpec = source["task"]
        env = dict(os.environ)
        bridge = workspace / "code" / "scenecode_pybullet_minimal"
        from .runtime_paths import scenecode_runtime as selected_scenecode_runtime
        from .runtime_paths import scenecode_python as selected_scenecode_python
        scenecode_runtime = selected_scenecode_runtime(workspace)
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(scenecode_runtime),
                str(repository),
                str(bridge),
                env.get("PYTHONPATH", ""),
            ]
        ).rstrip(os.pathsep)
        python = workspace / "envs" / "autoresearch" / "bin" / "python"
        scenecode_python = selected_scenecode_python(workspace)
        is_articulated = bool(source.get("is_articulated"))
        is_multi = len(source["execution_steps"]) > 1
        is_push = source["operator_sequence"] == ["push"]
        expert_pose_profile = None
        if selected_execution_profile is not None:
            if is_multi or is_articulated or is_push or is_pour:
                raise MissingExecutionCapability("expert pose profiles currently support native rigid single-subject references")
            from .robotics_reference_execution_profile import load_profile
            expert_pose_profile = load_profile(selected_execution_profile,
                task_sha256=sha256_file(source["task_path"]), batch_id=batch_id,
                case_id=case_id, object_id=source["subject_id"])
        driver = bridge / "scripts" / (
            "run_droid_articulated.py"
            if is_articulated
            else ("run_droid_operator_graph.py"
            if is_multi
            else ("run_droid_push.py" if is_push else "run_droid_pick_place.py")
            )
        )
        # Grasp-feature measurement: the aperture the jaws must span is the
        # material inside the grasp band, not the manipuland's whole footprint.
        # This runs before the render cache lookup so the cached decision inputs
        # stay identical to a fresh execution.
        grasp_band_reports: dict[str, Any] = {}
        if not is_articulated and not is_push and not is_pour:
            for step in source["execution_steps"]:
                match = source["matches"][step["subject_id"]]
                report = _grasp_band_plan(
                    python=python,
                    bridge=bridge,
                    env=env,
                    sdf_path=Path(match["sdf_path"]),
                    orientation_xyzw=tuple(
                        float(value) for value in step["subject_orientation_xyzw"]
                    ),
                    usable_aperture_m=(
                        GRIPPER_MAX_APERTURE_M
                        - MINIMUM_CLOSING_TRAVEL_M
                        - POSE_MARGIN_M
                    ),
                    minimum_extent_m=MINIMUM_OPPOSED_GRASP_EXTENT_M,
                    maximum_lateral_offset_m=(
                        MAXIMUM_GRASP_FEATURE_LATERAL_OFFSET_M
                    ),
                    minimum_straddle_margin_m=MINIMUM_GRASP_STRADDLE_MARGIN_M,
                    output=work_for_measurements / f"grasp_band_{step['subject_id']}.json",
                )
                plan = report["plan"]
                profile = report.get("profile") or {}
                step["grasp_band_plan"] = plan
                step["grasp_band_profile_summary"] = {
                    key: profile.get(key)
                    for key in (
                        "schema_version",
                        "sdf_sha256",
                        "object_height_m",
                        "aabb_low_m",
                        "aabb_high_m",
                        "profile_step_m",
                        "ray_step_m",
                        "band_height_m",
                    )
                }
                step["whole_object_grasp_horizontal_extent_m"] = step[
                    "grasp_horizontal_extent"
                ]
                step["whole_object_grasp_horizontal_axis"] = step[
                    "grasp_horizontal_axis"
                ]
                step["whole_object_grasp_yaw_deg"] = step["grasp_yaw_deg"]
                step["grasp_horizontal_extent"] = float(plan["grip_extent_m"])
                step["grasp_horizontal_axis"] = int(plan["grip_axis"])
                # Bridge convention, measured with the real finger-pad AABBs:
                # jaw yaw 0 closes along world +X, jaw yaw 90 along world +Y.
                step["grasp_yaw_deg"] = 0.0 if plan["grip_axis"] == 0 else 90.0
                step["grasp_offset_xy_m"] = [
                    float(value) for value in plan["lateral_center_xy_m"]
                ]
                grasp_band_reports[step["subject_id"]] = {
                    "plan": plan,
                    "whole_object_extent_m": step[
                        "whole_object_grasp_horizontal_extent_m"
                    ],
                    "whole_object_axis": step["whole_object_grasp_horizontal_axis"],
                    "whole_object_yaw_deg": step["whole_object_grasp_yaw_deg"],
                }
            first = source["execution_steps"][0]
            source["grasp_horizontal_extent"] = first["grasp_horizontal_extent"]
            source["grasp_horizontal_axis"] = first["grasp_horizontal_axis"]
            source["grasp_yaw_deg"] = first["grasp_yaw_deg"]
            source["grasp_offset_xy_m"] = first["grasp_offset_xy_m"]
            source["grasp_band_center_offset_m"] = first["grasp_band_plan"][
                "band_center_offset_m"
            ]
            source["grasp_band_reports"] = grasp_band_reports
        cache_inputs = _cache_input_digests(
            repository=repository,
            bridge=bridge,
            source=source,
            robot_manifest=robot_manifest,
        )
        if selected_execution_profile is not None:
            cache_inputs.update(execution_profile=sha256_file(selected_execution_profile),
                execution_profile_engine=sha256_file(repository / "autoresearch/robotics_reference_execution_profile.py"))
        if source.get("visual_revision"):
            revision = source["visual_revision"]
            cache_inputs.update({
                "visual_revision_manifest": sha256_file(revision["manifest"]),
                "visual_revision_verdict": sha256_file(revision["verdict"]),
                "visual_revision_sdf": sha256_file(revision["sdf"]),
                "visual_revision_verifier": sha256_file(
                    repository / "autoresearch/robotics_visual_revision_verifier.py"),
            })
        if is_pour:
            assert selected_asset_revision is not None and pour_case is not None
            cache_inputs.update({
                "pour_asset_manifest": sha256_file(selected_asset_revision / "candidate_manifest.json"),
                "pour_asset_derivative_verdict": sha256_file(source["pour_asset_derivative_verdict"]),
                "pour_source_sdf": sha256_file(pour_case["source_sdf"]),
                "pour_content_sdf": sha256_file(pour_case["content_sdf"]),
                "pour_controller": sha256_file(repository / "autoresearch/robotics_pour_controller.py"),
                "pour_replay": sha256_file(repository / "autoresearch/robotics_pour_replay.py"),
                "pour_verifier": sha256_file(repository / "autoresearch/robotics_pour_verifier.py"),
                "pour_execution": sha256_file(repository / "autoresearch/robotics_pour_execution.py"),
            })
        articulation_goal = None
        articulation_timing = None
        if is_articulated:
            from .articulated_goal_contract import reference_goal
            from .articulated_reference_timing import reference_timing
            first = source["execution_steps"][0]
            articulation_goal = reference_goal(first["articulation_binding"], first,
                requested_fraction=articulation_open_fraction_required)
            cache_inputs["articulation_goal_contract"] = hashlib.sha256(
                json.dumps(articulation_goal, sort_keys=True).encode()).hexdigest()
            articulation_timing = reference_timing(first["articulation_binding"])
            cache_inputs["articulation_timing_contract"] = hashlib.sha256(
                json.dumps(articulation_timing,sort_keys=True).encode()).hexdigest()
        elif articulation_open_fraction_required is not None:
            raise ValueError("articulation acceptance applies only to an articulated reference")
        cache_completed = _validate_completed_cache(cache, cache_inputs)
        if cache_completed:
            physical_manifest = _cached_physical_manifest(cache, source)
            stage = "reference_packaging"
            published = _package_completed_cache(
                python=python,
                repository=repository,
                env=env,
                output_root=output_root,
                batch_id=batch_id,
                case_id=case_id,
                source=source,
                simulation=simulation,
                render=render,
                physical_manifest=physical_manifest,
                visual_manifest=selected_visual_manifest,
                package=package,
                final=final,
                grasp_band_evidence=work_for_measurements,
            )
            shutil.rmtree(cache, ignore_errors=True)
            failure_path.unlink(missing_ok=True)
            return published
        if cache.exists():
            shutil.rmtree(cache)
        cache.mkdir(parents=True)
        simulation.mkdir(parents=True)
        render.mkdir(parents=True)
        physical_manifest: Path | None = None
        stage = "reference_execution"
        if is_articulated:
            step = source["execution_steps"][0]
            binding_path = Path(source["articulation_binding_path"]).resolve()
            host_sdf = Path(source["articulated_host_sdf"]).resolve()
            host_xy = tuple(float(value) for value in step["destination_xy"])
            contact_frame = step["articulation_binding"]
            if any(contact_frame.get(key) is None for key in (
                "contact_target_local_xyz", "contact_normal_local_xyz", "handle_axis_local_xyz"
            )):
                raise ValueError("articulated reference requires a certified handle contact frame")
            placement = (task.case_parameters or {}).get("articulated_execution", {}) if isinstance(task.case_parameters, dict) else {}

            execution_geometry = {
                "schema_version": "robotics_articulated_execution_geometry.v1",
                "required_joint_travel_fraction": articulation_goal["required_joint_travel_fraction"],
                "measured_goal_contract": articulation_goal,
                "reference_timing": articulation_timing,
                "operator": step["terminal_operator"],
                "success_predicate": step["terminal_predicate"],
                "component_id": step["articulation_component_id"],
                "host_object_id": step["articulation_host_object_id"],
                "binding": step["articulation_binding"],
                "binding_sha256": sha256_file(binding_path),
                "host_sdf": str(host_sdf),
                "host_sdf_sha256": sha256_file(host_sdf),
                "host_center_xy": list(host_xy),
            }
            compatibility = {
                "schema_version": "robotics_embodiment_articulated_compatibility.v1",
                "adapter": MODEL_ADAPTER,
                "robot_embodiment": "franka_robotiq_2f85",
                "operator": step["terminal_operator"],
                "policy": "contact_only_no_attachment_constraint",
                "passed": True,
            }
            layout = source["layout"]
            execution_args = [
                str(python),
                str(driver),
                "--binding",
                str(binding_path),
                "--host-sdf",
                str(host_sdf),
                "--panda-urdf",
                str(robot_urdf),
                "--output-dir",
                str(simulation),
                "--host-center-xy",
                *(_cli_float(value) for value in host_xy),
                "--operator",
                step["terminal_operator"],
                "--success-predicate",
                step["terminal_predicate"],
            ]
            execution_args.extend(["--open-fraction-required", str(articulation_goal["required_joint_travel_fraction"])])
            execution_args.extend(["--actuate-steps", str(articulation_timing["actuate_steps"])])
            for key, flag, count in (
                ("robot_base_xy_m", "--robot-base-xy", 2),
                ("robot_base_z_m", "--robot-base-z", 1),
                ("robot_base_yaw_rad", "--robot-base-yaw", 1),
            ):
                if key in placement:
                    values = placement[key] if count == 2 else [placement[key]]
                    if len(values) != count or not all(math.isfinite(float(value)) for value in values):
                        raise ValueError("articulated robot placement must be finite and dimensionally valid")
                    execution_args.extend([flag, *(_cli_float(value) for value in values)])
            if placement.get("preserve_host_orientation") is True:
                execution_args.append("--preserve-host-orientation")
            execution_geometry["robot_placement"] = placement
        elif is_pour:
            assert selected_asset_revision is not None and pour_case is not None
            layout = pour_case["layout"]
            execution_geometry = {
                "schema_version": "robotics_pour_execution_geometry.v1",
                "source_id": pour_case["source_id"],
                "target_id": pour_case["target_id"],
                "content_instances": pour_case["content_instances"],
                "source_sdf_sha256": sha256_file(pour_case["source_sdf"]),
                "asset_revision_sha256": pour_case["asset_revision_sha256"],
                "independent_asset_verdict_sha256": sha256_file(
                    source["pour_asset_derivative_verdict"]),
            }
            compatibility = {
                "schema_version": "robotics_embodiment_pour_compatibility.v1",
                "adapter": MODEL_ADAPTER,
                "robot_embodiment": "franka_robotiq_2f85",
                "operator": "pour",
                "policy": "trusted_joint_replay_and_independent_contents_verifier",
                "passed": True,
            }
            execution_args = [
                str(python), "-m", "autoresearch.robotics_pour_execution",
                "--workspace", str(workspace),
                "--task", str(source["task_path"]),
                "--generation", str(source["generation_path"]),
                "--asset-revision", str(selected_asset_revision),
                "--output", str(simulation),
            ]
        elif is_multi:
            source_positions = {
                step["subject_id"]: step["subject_xy"]
                for step in source["execution_steps"]
            }
            source_positions.update(
                {
                    step["destination_id"]: step["destination_xy"]
                    for step in source["execution_steps"]
                }
            )
            compiled_positions, layout = normalized_object_layout(source_positions)
            execution_geometry = []
            compatibility_steps = []
            for step in source["execution_steps"]:
                placement_limit = None
                if step["terminal_operator"] == "stack":
                    placement_limit = (
                        step["destination_horizontal_support_extent"]
                        + 2.0 * PYBULLET_COLLISION_MARGIN_PER_SIDE_M
                        - step["horizontal_extent"]
                        * UNIFORM_SCALE_STEP
                        * STACK_SUPPORT_MARGIN_SCALE_STEPS
                    )
                elif step["terminal_operator"] == "insert":
                    placement_limit = (
                        step["insert_aperture"]["diameter_m"]
                        * INSERT_APERTURE_CLEARANCE_FRACTION
                    )
                step_scale, step_compatibility = droid_uniform_grasp_scale(
                    step["horizontal_extent"],
                    placement_horizontal_limit_m=placement_limit,
                    minimum_horizontal_extent_m=(
                        MINIMUM_OPPOSED_GRASP_EXTENT_M
                        if step["terminal_operator"] == "insert"
                        else None
                    ),
                )
                if step_scale < 1.0:
                    raise ValueError(
                        "multi-object operator graph requires native-scale manipulands"
                    )
                step_compatibility["terminal_operator"] = step["terminal_operator"]
                step_compatibility["subject_id"] = step["subject_id"]
                step_compatibility["destination_id"] = step["destination_id"]
                compatibility_steps.append(step_compatibility)
                geometry = pick_place_execution_geometry(
                    subject_height_m=step["subject_height"],
                    destination_height_m=step["destination_height"],
                    terminal_operator=step["terminal_operator"],
                )
                execution_geometry.append(
                    {
                        **geometry,
                        "subject_id": step["subject_id"],
                        "target_id": step["destination_id"],
                        "terminal_operator": step["terminal_operator"],
                        "success_predicate": step["terminal_predicate"],
                        "terminal_alignment_axis_local": step[
                            "terminal_alignment_axis_local"
                        ],
                        "terminal_alignment_axis_world": step[
                            "terminal_alignment_axis_world"
                        ],
                        "place_target_offset_xy_m": list(
                            step["terminal_target_offset_world_xy_m"]
                        ),
                        "insert_aperture": step["insert_aperture"],
                    }
                )
            compatibility = {
                "schema_version": (
                    "robotics_embodiment_operator_graph_compatibility.v1"
                ),
                "adapter": MODEL_ADAPTER,
                "robot_embodiment": "franka_robotiq_2f85",
                "steps": compatibility_steps,
                "passed": True,
            }
            task_roles = {item.object_id: item.role for item in task.objects}
            graph_manifest = work / "operator_graph_input.json"
            write_json(
                graph_manifest,
                {
                    "schema_version": "robotics_droid_operator_graph_input.v1",
                    "objects": [
                        {
                            "object_id": object_id,
                            "sdf_path": source["matches"][object_id]["sdf_path"],
                            "center_xy": list(compiled_positions[object_id]),
                            "orientation_xyzw": list(
                                next(
                                    step[
                                        "subject_orientation_xyzw"
                                        if step["subject_id"] == object_id
                                        else "destination_orientation_xyzw"
                                    ]
                                    for step in source["execution_steps"]
                                    if object_id
                                    in {step["subject_id"], step["destination_id"]}
                                )
                            ),
                            "role": task_roles[object_id],
                            "lateral_friction": (
                                1.2 if task_roles[object_id] == "manipuland" else 0.9
                            ),
                            "collision_enabled": True,
                            "fixed": False,
                        }
                        for object_id in source["step_object_ids"]
                    ],
                    "steps": execution_geometry,
                },
            )
            execution_args = [
                str(python),
                str(driver),
                "--graph-manifest",
                str(graph_manifest),
                "--table-sdf",
                source["matches"][source["support_id"]]["sdf_path"],
                "--panda-urdf",
                str(robot_urdf),
                "--output-dir",
                str(simulation),
            ]
        else:
            destination_scale = 1.0
            destination_sdf = Path(
                source["matches"][source["destination_id"]]["sdf_path"]
            )
            if is_push:
                scale = 1.0
                compatibility = {
                    "schema_version": "robotics_embodiment_push_compatibility.v1",
                    "adapter": MODEL_ADAPTER,
                    "robot_embodiment": "franka_robotiq_2f85",
                    "operator": "push",
                    "source_horizontal_extent_m": source["horizontal_extent"],
                    "uniform_scale": scale,
                    "target_collision_policy": "visual_goal_region_no_collision",
                    "passed": True,
                }
            else:
                placement_limit = None
                compatibility_destination_fit = None
                compatibility_destination_native_view_fit = None
                if source["terminal_operator"] == "stack":
                    destination_scale = stack_destination_visibility_scale(
                        subject_horizontal_extent_m=source["horizontal_extent"],
                        destination_support_extent_m=source[
                            "destination_horizontal_support_extent"
                        ],
                    )
                    placement_limit = (
                        source["destination_horizontal_support_extent"]
                        * destination_scale
                        + 2.0 * PYBULLET_COLLISION_MARGIN_PER_SIDE_M
                        - source["horizontal_extent"]
                        * UNIFORM_SCALE_STEP
                        * STACK_SUPPORT_MARGIN_SCALE_STEPS
                    )
                elif source["terminal_operator"] == "insert":
                    placement_limit = (
                        source["insert_aperture"]["diameter_m"]
                        * INSERT_APERTURE_CLEARANCE_FRACTION
                    )
                elif source["terminal_operator"] in {"place", "cover"}:
                    # A destination whose oriented footprint cannot clear the
                    # reset pose inside the reachable envelope is scaled down
                    # before layout; otherwise the task cannot be laid out.
                    direction = (
                        float(source["destination_xy"][0])
                        - float(source["subject_xy"][0]),
                        float(source["destination_xy"][1])
                        - float(source["subject_xy"][1]),
                    )
                    destination_scale, destination_fit = droid_destination_fit_scale(
                        destination_half_extents_m=(
                            0.5 * float(source["destination_local_extent_xyz"][0]),
                            0.5 * float(source["destination_local_extent_xyz"][1]),
                        ),
                        orientation_xyzw=tuple(
                            float(value)
                            for value in source["destination_orientation_xyzw"]
                        ),
                        layout_direction_xy=direction,
                    )
                    compatibility_destination_fit = destination_fit
                    # Clearing the reset pose is necessary but not sufficient: a
                    # destination that is an order of magnitude larger than the
                    # manipuland also swallows the native over-shoulder camera
                    # and crops at the frame border, which the model-native
                    # admission gate measures on the rendered mask.
                    destination_scale, native_view_fit = droid_destination_native_view_scale(
                        candidate_scale=destination_scale,
                        subject_xy=tuple(float(value) for value in source["subject_xy"]),
                        destination_xy=tuple(
                            float(value) for value in source["destination_xy"]
                        ),
                        subject_half_extents_m=(
                            0.5 * float(source["subject_local_extent_xyz"][0]),
                            0.5 * float(source["subject_local_extent_xyz"][1]),
                        ),
                        subject_orientation_xyzw=tuple(
                            float(value)
                            for value in source["subject_orientation_xyzw"]
                        ),
                        destination_half_extents_m=(
                            0.5 * float(source["destination_local_extent_xyz"][0]),
                            0.5 * float(source["destination_local_extent_xyz"][1]),
                        ),
                        destination_orientation_xyzw=tuple(
                            float(value)
                            for value in source["destination_orientation_xyzw"]
                        ),
                        destination_height_m=float(
                            source["destination_local_extent_xyz"][2]
                        ),
                        required_views=native_destination_view_requirements(task),
                    )
                    compatibility_destination_native_view_fit = native_view_fit
                scale, compatibility = droid_uniform_grasp_scale(
                    source["grasp_horizontal_extent"],
                    placement_horizontal_limit_m=placement_limit,
                    minimum_horizontal_extent_m=(
                        MINIMUM_OPPOSED_GRASP_EXTENT_M
                        if source["terminal_operator"] == "insert"
                        else None
                    ),
                )
                compatibility["grasp_horizontal_axis"] = source["grasp_horizontal_axis"]
                compatibility["subject_horizontal_extent_m"] = source[
                    "horizontal_extent"
                ]
                compatibility["grasp_yaw_deg"] = source["grasp_yaw_deg"]
                compatibility["terminal_operator"] = source["terminal_operator"]
                compatibility["destination_horizontal_support_extent_m"] = source[
                    "destination_horizontal_support_extent"
                ]
                compatibility["destination_uniform_scale"] = destination_scale
                compatibility["scaled_destination_horizontal_support_extent_m"] = (
                    source["destination_horizontal_support_extent"]
                    * destination_scale
                )
                compatibility["stack_support_margin_scale_steps"] = (
                    STACK_SUPPORT_MARGIN_SCALE_STEPS
                    if source["terminal_operator"] == "stack"
                    else None
                )
                compatibility["pybullet_collision_margin_per_side_m"] = (
                    PYBULLET_COLLISION_MARGIN_PER_SIDE_M
                    if source["terminal_operator"] == "stack"
                    else None
                )
                compatibility["insert_aperture"] = source["insert_aperture"]
                compatibility["insert_aperture_clearance_fraction"] = (
                    INSERT_APERTURE_CLEARANCE_FRACTION
                    if source["terminal_operator"] == "insert"
                    else None
                )
                compatibility["destination_workcell_fit"] = (
                    compatibility_destination_fit
                )
                compatibility["destination_native_view_fit"] = (
                    compatibility_destination_native_view_fit
                )
            subject_sdf = Path(source["matches"][source["subject_id"]]["sdf_path"])
            if (
                not is_push
                and source["terminal_predicate"] == "on_region"
                and not _sdf_has_collision_geometry(destination_sdf)
            ):
                if physical_manifest is not None or destination_scale != 1.0:
                    raise ValueError(
                        "planar collision derivation conflicts with another physical variant"
                    )
                destination_variant = (
                    work / "physical_variant" / source["destination_id"]
                )
                from scripts.derive_scenecode_planar_collision_variant import (
                    derive_planar_collision_variant,
                )

                destination_sdf = derive_planar_collision_variant(
                    destination_sdf,
                    destination_variant,
                    variant_id=(
                        f"{source['destination_id']}_{MODEL_ADAPTER}_"
                        "planar_collision_v1"
                    ),
                    size_xyz_m=tuple(source["destination_local_extent_xyz"]),
                )
                physical_manifest = (
                    destination_variant / "physical_variant_manifest.json"
                )
                physical_payload = json.loads(
                    physical_manifest.read_text(encoding="utf-8")
                )
                physical_payload["embodiment_compatibility"] = compatibility
                physical_payload["robot_asset_manifest_sha256"] = sha256_file(
                    robot_manifest
                )
                write_json(physical_manifest, physical_payload)
            if destination_scale != 1.0:
                destination_variant = work / "physical_variant" / source["destination_id"]
                destination_variant_id = (
                    f"{source['destination_id']}_{MODEL_ADAPTER}_"
                    + (
                        "stack_support_v1"
                        if destination_scale > 1.0
                        else "native_compat_v1"
                    )
                )
                from scripts.derive_scenecode_uniform_scale_variant import (
                    derive_uniform_scale_variant,
                )

                destination_sdf = derive_uniform_scale_variant(
                    destination_sdf,
                    destination_variant,
                    variant_id=destination_variant_id,
                    scale=destination_scale,
                )
                physical_manifest = (
                    destination_variant / "physical_variant_manifest.json"
                )
                physical_payload = json.loads(
                    physical_manifest.read_text(encoding="utf-8")
                )
                physical_payload["embodiment_compatibility"] = compatibility
                physical_payload["robot_asset_manifest_sha256"] = sha256_file(
                    robot_manifest
                )
                write_json(physical_manifest, physical_payload)
            if scale < 1.0:
                variant = work / "physical_variant" / source["subject_id"]
                variant_id = (
                    f"{source['subject_id']}_{MODEL_ADAPTER}_operator_geometry_compat_v3"
                )
                from scripts.derive_scenecode_uniform_scale_variant import (
                    derive_uniform_scale_variant,
                )

                subject_sdf = derive_uniform_scale_variant(
                    subject_sdf,
                    variant,
                    variant_id=variant_id,
                    scale=scale,
                )
                physical_manifest = variant / "physical_variant_manifest.json"
                physical_payload = json.loads(
                    physical_manifest.read_text(encoding="utf-8")
                )
                physical_payload["embodiment_compatibility"] = compatibility
                physical_payload["robot_asset_manifest_sha256"] = sha256_file(
                    robot_manifest
                )
                write_json(physical_manifest, physical_payload)

            subject_xy, destination_xy, layout = normalized_two_object_layout(
                source["subject_xy"],
                source["destination_xy"],
                subject_half_extents_m=(
                    (
                        0.5 * float(source["subject_local_extent_xyz"][0]) * scale,
                        0.5 * float(source["subject_local_extent_xyz"][1]) * scale,
                    )
                    if not is_push
                    else None
                ),
                subject_orientation_xyzw=tuple(
                    float(value) for value in source["subject_orientation_xyzw"]
                ),
                destination_half_extents_m=(
                    (
                        0.5
                        * float(source["destination_local_extent_xyz"][0])
                        * destination_scale,
                        0.5
                        * float(source["destination_local_extent_xyz"][1])
                        * destination_scale,
                    )
                    if not is_push and source["terminal_operator"] != "flip"
                    else None
                ),
                destination_orientation_xyzw=tuple(
                    float(value)
                    for value in source["destination_orientation_xyzw"]
                ),
                home_xy=(
                    DROID_RESET_END_EFFECTOR_XY_M if not is_push else None
                ),
                home_clearance_m=(
                    DROID_GRASP_FOOTPRINT_CLEARANCE_M if not is_push else 0.0
                ),
                maximum_shift_m=DROID_MAXIMUM_LAYOUT_SHIFT_M,
            )
            planar_profile = None
            if (source["terminal_operator"] == "flip"
                    and source["terminal_predicate"] == "on_region"):
                planar_profile = planar_reorientation_profile(
                    subject_xy, source.get("grasp_offset_xy_m") or (0.0, 0.0), scale
                )
                destination_xy = tuple(planar_profile["landing_xy_m"])
                layout["planar_reorientation"] = planar_profile
            if (layout.get("workcell_clearance") or {}).get(
                "clearance_satisfied"
            ) is False:
                raise ValueError(
                    "DROID reset-pose clearance is unsatisfiable inside the "
                    "reachable workcell; the destination would still overlap "
                    "the expert's starting gripper"
                )
            execution_geometry = (
                {"authority": "runtime_aabb_push_corridor"}
                if is_push
                else pick_place_execution_geometry(
                    subject_height_m=source["subject_height"] * scale,
                    destination_height_m=(
                        source["destination_height"] * destination_scale
                    ),
                    terminal_operator=source["terminal_operator"],
                    grasp_band_center_offset_m=(
                        source["grasp_band_center_offset_m"] * scale
                        if source.get("grasp_band_center_offset_m") is not None
                        else None
                    ),
                )
            )
            if not is_push:
                execution_geometry["terminal_alignment_axis_local"] = source[
                    "terminal_alignment_axis_local"
                ]
                execution_geometry["terminal_alignment_axis_world"] = source[
                    "terminal_alignment_axis_world"
                ]
                execution_geometry["place_target_offset_xy_m"] = list(
                    source["terminal_target_offset_world_xy_m"]
                )
                execution_geometry["insert_aperture"] = source["insert_aperture"]
                execution_geometry["grasp_yaw_deg"] = source["grasp_yaw_deg"]
                # The grasp feature was measured on the native asset, but the
                # executor loads the uniformly scaled physical variant.  Scale
                # the lateral grasp-site offset with the same factor as the
                # vertical band, otherwise a shrunk subject is grasped at a
                # point that lies outside its actual footprint.
                execution_geometry["grasp_offset_xy_m"] = scale_grasp_offset_xy(
                    source.get("grasp_offset_xy_m") or (0.0, 0.0), scale
                )
                if planar_profile is not None:
                    execution_geometry["grasp_offset_xy_m"] = planar_profile[
                        "grasp_offset_xy_m"
                    ]
                    execution_geometry["planar_reorientation"] = planar_profile
                execution_geometry["grasp_band_plan"] = source.get("grasp_band_reports")
            execution_args = [
                str(python),
                str(driver),
                "--table-sdf",
                source["matches"][source["support_id"]]["sdf_path"],
                "--panda-urdf",
                str(robot_urdf),
                "--output-dir",
                str(simulation),
            ]
            if is_push:
                execution_args.extend(
                    [
                        "--subject-sdf",
                        str(subject_sdf),
                        "--target-sdf",
                        str(destination_sdf),
                        "--subject-center-xy",
                        *(_cli_float(value) for value in subject_xy),
                        "--target-center-xy",
                        *(_cli_float(value) for value in destination_xy),
                        "--subject-orientation-xyzw",
                        *(
                            _cli_float(value)
                            for value in source["subject_orientation_xyzw"]
                        ),
                        "--target-orientation-xyzw",
                        *(
                            _cli_float(value)
                            for value in source["destination_orientation_xyzw"]
                        ),
                    ]
                )
            else:
                execution_args.extend(
                    [
                        "--cup-sdf",
                        str(subject_sdf),
                        "--tray-sdf",
                        str(destination_sdf),
                        "--grasp-base-offset-z-m",
                        _cli_float(execution_geometry["grasp_base_offset_z_m"]),
                        "--hover-base-offset-z-m",
                        _cli_float(execution_geometry["hover_base_offset_z_m"]),
                        "--lift-base-offset-z-m",
                        _cli_float(execution_geometry["lift_base_offset_z_m"]),
                        "--place-target-offset-z-m",
                        _cli_float(execution_geometry["place_target_offset_z_m"]),
                        "--grasp-yaw-deg",
                        _cli_float(execution_geometry["grasp_yaw_deg"]),
                        "--grasp-offset-xy-m",
                        *(
                            _cli_float(value)
                            for value in execution_geometry.get(
                                "grasp_offset_xy_m", (0.0, 0.0)
                            )
                        ),
                        "--place-target-offset-xy-m",
                        *(
                            _cli_float(value)
                            for value in execution_geometry[
                                "place_target_offset_xy_m"
                            ]
                        ),
                        "--success-predicate",
                        source["terminal_predicate"],
                        "--semantic-operator",
                        source["terminal_operator"],
                        "--cup-center-xy",
                        *(_cli_float(value) for value in subject_xy),
                        "--tray-center-xy",
                        *(_cli_float(value) for value in destination_xy),
                        "--cup-orientation-xyzw",
                        *(
                            _cli_float(value)
                            for value in source["subject_orientation_xyzw"]
                        ),
                        "--tray-orientation-xyzw",
                        *(
                            _cli_float(value)
                        for value in source["destination_orientation_xyzw"]
                        ),
                    ]
                )
                if source["terminal_operator"] == "flip":
                    execution_args.extend(["--target-collision-enabled", "0"])
                if source["terminal_alignment_axis_local"] is not None:
                    execution_args.extend(
                        [
                            "--terminal-alignment-axis-local",
                            *(
                                _cli_float(value)
                                for value in source[
                                    "terminal_alignment_axis_local"
                                ]
                            ),
                            "--terminal-alignment-axis-world",
                            *(
                                _cli_float(value)
                                for value in source[
                                    "terminal_alignment_axis_world"
                                ]
                            ),
                        ]
                    )
                if source.get("terminal_upright_min_cosine") is not None:
                    execution_args.extend(
                        [
                            "--terminal-upright-min-cosine",
                            _cli_float(source["terminal_upright_min_cosine"]),
                        ]
                    )
        if expert_pose_profile is not None:
            from .robotics_reference_execution_profile import apply_profile
            parameters = expert_pose_profile["parameters"]
            if "grasp_offset_xy_m" in parameters:
                extent = source["subject_local_extent_xyz"]
                radius = 0.5 * math.hypot(float(extent[0]), float(extent[1])) * scale
                if math.hypot(*parameters["grasp_offset_xy_m"]) > radius + GEOMETRY_NUMERICAL_TOLERANCE_M:
                    raise ValueError("expert grasp offset is outside the projected subject envelope")
            if "grasp_base_offset_z_m" in parameters:
                height = parameters["grasp_base_offset_z_m"] - DROID_SIDE_GRASP_CENTER_BELOW_BASE_M
                if not 0 <= height <= source["subject_height"] * scale:
                    raise ValueError("expert grasp band is outside the projected subject height")
                if parameters["grasp_base_offset_z_m"] >= execution_geometry["hover_base_offset_z_m"]:
                    raise ValueError("expert grasp must remain below its native approach height")
            execution_args = apply_profile(execution_args, parameters)
            execution_geometry.update(parameters)
            execution_geometry["execution_profile_sha256"] = cache_inputs["execution_profile"]
        write_json(work / "execution_request.json", {
            "argv": execution_args,
            "cwd": str(bridge),
            "input_digests": cache_inputs,
            "execution_geometry": execution_geometry,
            "embodiment_compatibility": compatibility,
            "execution_profile": expert_pose_profile,
        })
        _run(
            execution_args,
            cwd=bridge,
            env=env,
        )
        result = json.loads((simulation / "result.json").read_text(encoding="utf-8"))
        if result.get("success") is not True:
            raise ValueError("DROID expert simulation did not complete the task")
        if is_articulated:
            from .articulated_goal_contract import assess_measured_reference
            trajectory = json.loads((simulation / "trajectory.json").read_text())
            goal_certificate = assess_measured_reference(trajectory, articulation_goal, step["articulation_binding"])
            write_json(simulation / "measured_goal_certificate.json", goal_certificate)
            if goal_certificate["passed"] is not True:
                raise ValueError("reference failed native target, release or terminal stability; see measured_goal_certificate.json")
        write_json(
            simulation / "agentic_execution_manifest.json",
            {
                "schema_version": "robotics_agentic_execution_manifest.v1",
                "harness_id": HARNESS_ID,
                "model_adapter": MODEL_ADAPTER,
                "task_graph": str(source["plan_path"]),
                "task_graph_sha256": sha256_file(source["plan_path"]),
                "operator_sequence": source["operator_sequence"],
                "success_predicate": (
                    "all_steps" if is_multi else source["terminal_predicate"]
                ),
                "execution_geometry": execution_geometry,
                "embodiment_compatibility": compatibility,
                "layout_normalization": layout,
                "termination": "expert_program_complete_and_success_predicates_true",
                "robot_asset_manifest": str(robot_manifest),
                "robot_asset_manifest_sha256": sha256_file(robot_manifest),
                "passed": True,
            },
        )

        camera_rig: Path | None = None
        if is_articulated:
            from .robotics_articulated_camera import fit_articulated_camera_rig
            camera_rig = fit_articulated_camera_rig(
                trajectory=simulation / "trajectory.json",
                host_sdf=Path(source["articulated_host_sdf"]),
                binding=Path(source["articulation_binding_path"]),
                required_views=native_destination_view_requirements(task),
                output=work / "camera_rig.json",
            )
            from .articulated_exterior_framing import fit_measured_exterior_rig
            camera_rig = fit_measured_exterior_rig(
                parent_rig=camera_rig,
                trajectory=simulation / "trajectory.json",
                physics_sdf=simulation / "physics_sdf" / "host.sdf",
                binding=Path(source["articulation_binding_path"]),
                robot_urdf=robot_urdf,
                output=work / "camera_rig_measured_motion.json",
            )

        stage = "camera_observability"
        render_args = [
            str(scenecode_python),
            str(repository / "scripts" / "render_scenecode_robotics_multiview_parallel.py"),
            "--render-script",
            str(repository / "scripts" / "render_scenecode_robotics_multiview.py"),
            "--trajectory",
            str(simulation / "trajectory.json"),
            "--panda-urdf",
            str(robot_urdf),
            "--output-dir",
            str(render),
            "--blender-settings-file",
            str(
                repository
                / "deployment"
                / "ziplab5090"
                / "scenecode_robotics_render_settings.py"
            ),
            "--width",
            str(COSMOS_NATIVE_SIZE_WH_BY_VIEW["top"][0]),
            "--height",
            str(COSMOS_NATIVE_SIZE_WH_BY_VIEW["top"][1]),
            "--port",
            os.environ.get("WORLDOLYMPIAD_RENDER_PORT", "8166"),
            "--instruction",
            task.instruction,
            "--object-name",
            (
                "active task manipuland"
                if is_multi
                else (
                    next(
                        item.visual_description
                        for item in task.articulation_components
                        if item.component_id == source["subject_id"]
                    )
                    if is_articulated
                    else next(
                        item.visual_description
                        for item in task.objects
                        if item.object_id == source["subject_id"]
                    )
                )
            ),
            "--canonical-object-name",
            "operator_graph_active_subject" if is_multi else source["subject_id"],
            "--robot-embodiment",
            "robolab_droid",
            "--wrist-camera-preset",
            "robolab_droid",
            "--fixed-camera-preset",
            "robolab_droid",
            "--fov-y-deg",
            "71.5077745089",
            "--wrist-fov-y-deg",
            "56.74",
            "--semantic-evidence-mode",
            (
                "articulation_contact"
                if is_articulated
                else ("push_contact" if is_push else
                      "dual_contact" if is_pour else "carry_phases")
            ),
        ]
        if camera_rig is not None:
            render_args.extend(["--camera-rig", str(camera_rig)])
        if not is_multi and not is_articulated:
            render_args.extend(
                [
                    "--semantic-subject-object-id",
                    source["subject_id"],
                    "--semantic-destination-object-id",
                    source["destination_id"],
                ]
            )
            if source.get("support_id"):
                # Register the support surface too so the rendered in-context
                # review can measure the manipuland/destination against the
                # tabletop it actually sits on.
                render_args.extend(
                    ["--semantic-support-object-id", source["support_id"]]
                )
        if is_articulated:
            scene_objects_manifest = simulation / "scene_objects_manifest.json"
            write_json(
                scene_objects_manifest,
                {
                    "schema_version": "robotics_scene_object_table.v1",
                    "objects": [
                        {
                            "object_id": source["destination_id"],
                            "sdf_path": str(source["visual_revision"]["sdf"])
                            if source.get("visual_revision") else str(host_sdf),
                            "role": "support",
                        }
                    ],
                    "articulation_binding": str(binding_path),
                    "articulation_binding_sha256": sha256_file(binding_path),
                },
            )
            render_args.extend(
                [
                    "--scene-objects-manifest",
                    str(scene_objects_manifest),
                    "--articulation-binding",
                    str(binding_path),
                ]
            )
        elif is_multi or is_pour:
            render_scene_objects = simulation / "scene_objects_manifest.json"
            if source.get("visual_revision"):
                revision = source["visual_revision"]
                payload = json.loads(render_scene_objects.read_text())
                selected = [row for row in payload.get("objects", [])
                            if row.get("object_id") == revision["object_id"]]
                if len(selected) != 1:
                    raise ValueError("visual revision object is absent from render scene")
                selected[0]["sdf_path"] = str(revision["sdf"])
                render_scene_objects = work / "render_scene_objects_manifest.json"
                write_json(render_scene_objects, payload)
            render_args.extend(
                [
                    "--scene-objects-manifest",
                    str(render_scene_objects),
                ]
            )
            if is_pour:
                render_args.extend([
                    "--table-sdf", str(pour_case["table_sdf"]),
                ])
        else:
            render_assets = label_render_visual_sdfs(
                work=work,
                assets={
                    "support": Path(source["visual_revision"]["sdf"])
                        if source.get("visual_revision", {}).get("object_id") == source["support_id"]
                        else Path(source["matches"][source["support_id"]]["sdf_path"]),
                    "subject": Path(source["visual_revision"]["sdf"])
                        if source.get("visual_revision", {}).get("object_id") == source["subject_id"]
                        else Path(subject_sdf),
                    "destination": Path(source["visual_revision"]["sdf"])
                        if source.get("visual_revision", {}).get("object_id") == source["destination_id"]
                        else Path(destination_sdf),
                },
            )
            render_args.extend(
                [
                    # The parallel legacy renderer requires the support SDF
                    # together with the subject and destination SDFs.
                    "--table-sdf",
                    str(render_assets["support"]),
                    "--cup-sdf",
                    str(render_assets["subject"]),
                    "--tray-sdf",
                    str(render_assets["destination"]),
                ]
            )
        for view in REFERENCE_VIEWS:
            width, height = COSMOS_NATIVE_SIZE_WH_BY_VIEW[view]
            render_args.extend(
                [
                    "--view-size",
                    f"{view}={width}x{height}",
                ]
            )
        with exclusive_blender_job(output_root):
            _run(render_args, cwd=repository, env=env)
        _write_completed_cache(cache, cache_inputs)
        cache_completed = True
        stage = "reference_packaging"
        published = _package_completed_cache(
            python=python,
            repository=repository,
            env=env,
            output_root=output_root,
            batch_id=batch_id,
            case_id=case_id,
            source=source,
            simulation=simulation,
            render=render,
            physical_manifest=physical_manifest,
            visual_manifest=selected_visual_manifest,
            package=package,
            final=final,
            grasp_band_evidence=work_for_measurements,
        )
        shutil.rmtree(cache, ignore_errors=True)
        failure_path.unlink(missing_ok=True)
        return published
    except BaseException as exc:
        log_tails = _bounded_log_tails(work)
        try:
            retained_evidence = _preserve_failure_evidence(
                work, output_root / "robotics_reference_failure_evidence" / run_id
            )
        except (OSError, ValueError) as evidence_error:
            retained_evidence = {"preservation_error": str(evidence_error)}
        classification = {
            "camera_observability": "camera_observability",
            "reference_packaging": "reference_packaging",
            "embodiment_compatibility": "embodiment_compatibility",
            "source_validation": "source_validation",
        }.get(stage, "reference_execution")
        if isinstance(exc, MissingExecutionCapability):
            classification = "native_capability"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        history_root = output_root / "robotics_reference_failures/history" / run_id
        history_root.mkdir(parents=True, exist_ok=True)
        if failure_path.is_file():
            previous = failure_path.read_bytes()
            historical = history_root / (hashlib.sha256(previous).hexdigest() + ".json")
            if not historical.exists():
                historical.write_bytes(previous)
        completed_cache_retained = cache_completed and _validate_completed_cache(
            cache, cache_inputs
        )
        if not completed_cache_retained and (
            os.environ.get("WORLDOLYMPIAD_KEEP_REFERENCE_STAGING") != "1"
        ):
            shutil.rmtree(cache, ignore_errors=True)
        write_json(
            failure_path,
            {
                "schema_version": "robotics_agentic_reference_failure.v1",
                "batch_id": batch_id,
                "case_id": case_id,
                "run_id": run_id,
                "harness_id": HARNESS_ID,
                "classification": classification,
                "owner_stage": stage,
                "reason": str(exc).replace(str(staging), "<discarded-staging>"),
                "log_tails": log_tails,
                "retained_evidence": retained_evidence,
                "heavy_staging_discarded": not completed_cache_retained,
                "completed_render_cache_retained": completed_cache_retained,
            },
        )
        current = failure_path.read_bytes()
        historical = history_root / (hashlib.sha256(current).hexdigest() + ".json")
        if not historical.exists():
            historical.write_bytes(current)
        raise
    finally:
        # Bounded debug aid: keep the staging tree when a caller explicitly asks
        # for it so a failed expert episode can be inspected (trajectory.json,
        # result.json, prepared SDFs).  Default behaviour is unchanged.
        if os.environ.get("WORLDOLYMPIAD_KEEP_REFERENCE_STAGING") == "1":
            print(f"[materializer] retained staging for inspection: {staging}")
        else:
            shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--compilation-plan", type=Path)
    parser.add_argument("--asset-revision", type=Path)
    parser.add_argument("--visual-revision-manifest", type=Path)
    parser.add_argument("--visual-revision-verdict", type=Path)
    parser.add_argument("--execution-profile", type=Path)
    parser.add_argument("--articulation-open-fraction-required", type=float, default=None)
    args = parser.parse_args()
    print(materialize_reference(**vars(args)))


if __name__ == "__main__":
    main()
