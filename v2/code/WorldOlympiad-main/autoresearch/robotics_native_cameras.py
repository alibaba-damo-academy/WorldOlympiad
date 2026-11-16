"""Native RoboLab DROID fixed-camera framing shared by renderer and policy.

The DROID native cameras are rigidly attached to the registered robot base.
The bridge mounts that base on the workspace table and seats every task object
on the same table surface, so all framing geometry is expressed in a *table
frame*: world-aligned axes with ``z = 0`` at the table surface.  Table height
therefore cancels out and a framing decision stays valid for any generated
scene.

``scripts/render_scenecode_robotics_multiview.py`` composes the same presets in
Drake and stays the authority for the rendered pose; the preset constants live
here so the materializer can predict that pose before spending a render.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

# RoboLab commit f72d39445ab699c9cfd9bf551e8d21b6075293b9 publishes these DROID
# over-shoulder camera offsets in the robot-root frame.  Quaternions are
# OpenGL wxyz; the renderer converts them to Drake's +Z-forward/+Y-down camera
# axes and composes the registered Panda base pose.
ROBOLAB_DROID_FIXED_CAMERAS = {
    "exterior_left": ((0.05, 0.57, 0.66), (-0.393, -0.195, 0.399, 0.805)),
    "exterior_right": ((0.05, -0.57, 0.66), (0.805, 0.399, -0.195, -0.393)),
}
# Registered native fixed-camera preset of the DROID reference recipe.
DROID_NATIVE_FIXED_CAMERA_PRESET = "robolab_droid"
# Every projected box corner must stay this far inside the native frame, so a
# one-pixel label dilation cannot move the measured mask onto the border.
DROID_NATIVE_FRAME_MARGIN_PX = 2.0
# Native fixed-view vertical field of view of the RoboLab DROID cameras, as
# rendered at the registered 320x180 exterior size.
DROID_NATIVE_FIXED_FOV_Y_DEG = 71.5077745089

# Registered DROID mount: ``run_droid_pick_place`` welds the Panda at
# (0, -0.43, table_top + 0.002) with a +90 degree yaw.
DROID_TABLE_FRAME_BASE_XY_M = (0.0, -0.43)
DROID_TABLE_FRAME_BASE_Z_M = 0.002
DROID_TABLE_FRAME_BASE_YAW_RAD = math.pi / 2.0
# ``run_droid_pick_place`` seats a task object's bounding box on the table, but
# the replayed settle leaves the rendered mesh above the table by a fraction of
# its own height (the physics collision geometry starts above the visual mesh
# root).  The published pick_place reference measured 0.047 m on a 0.43 m
# destination, so framing is evaluated at both ends of this interval.
DROID_TABLE_FRAME_SEAT_OFFSET_FRACTIONS = (0.0, 0.19)


def _quaternion_xyzw_rotation(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("camera quaternion must be finite and non-zero")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rigid(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def native_fixed_camera_table_poses(
    preset: str = "robolab_droid",
) -> dict[str, np.ndarray]:
    """Return camera-to-table transforms for every native fixed view."""

    if preset != "robolab_droid":
        raise ValueError(f"unknown native fixed camera preset: {preset}")
    robot_to_table = _rigid(
        np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        (*DROID_TABLE_FRAME_BASE_XY_M, DROID_TABLE_FRAME_BASE_Z_M),
    )
    opengl_from_drake_camera = np.diag([1.0, -1.0, -1.0, 1.0])
    poses: dict[str, np.ndarray] = {}
    for name, (position, quaternion_wxyz) in ROBOLAB_DROID_FIXED_CAMERAS.items():
        camera_from_robot = _rigid(
            _quaternion_xyzw_rotation(
                (quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0])
            ),
            position,
        )
        poses[name] = robot_to_table @ camera_from_robot @ opengl_from_drake_camera
    return poses


def pinhole_intrinsics(
    size_wh: Sequence[int], fov_y_deg: float
) -> np.ndarray:
    """Return the pinhole matrix of a native view rendered at ``size_wh``."""

    width, height = (int(value) for value in size_wh)
    if width <= 0 or height <= 0:
        raise ValueError("native view size must be positive")
    if not math.isfinite(fov_y_deg) or not 0.0 < fov_y_deg < 180.0:
        raise ValueError("native view vertical FOV must lie in (0, 180)")
    focal = (height / 2.0) / math.tan(math.radians(float(fov_y_deg)) / 2.0)
    return np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def project_table_points(
    camera_from_table: np.ndarray,
    intrinsics: np.ndarray,
    points_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project table-frame points into the camera image plane.

    Drake's camera axes are +X right, +Y down, +Z forward, so the projection is
    ``u = fx * Xc / Zc + cx`` and ``v = fy * Yc / Zc + cy`` with no axis flip.
    """

    camera_from_table = np.asarray(camera_from_table, dtype=np.float64)
    rotation = camera_from_table[:3, :3]
    origin = camera_from_table[:3, 3]
    points = np.asarray(points_table, dtype=np.float64)
    if np.linalg.det(rotation) < 0.0:
        raise ValueError("camera-to-table rotation must be proper")
    camera_points = (rotation.T @ (points - origin).T).T
    depth = camera_points[:, 2]
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    pixels = (intrinsics @ camera_points.T).T[:, :2] / camera_points[:, 2:3]
    return pixels, depth


def native_view_box_envelope(
    *,
    center_xy_m: tuple[float, float],
    required_views: dict[str, dict[str, object]],
    minimum_fraction_safety: float = 1.0,
    margin_px: float = DROID_NATIVE_FRAME_MARGIN_PX,
    step: float = 0.05,
) -> list[float]:
    """Return the largest box a registered native view can frame at a site.

    The registered over-shoulder cameras see a bounded volume of the workcell.
    Sizing a generated asset by trial and error against the renderer wastes a
    whole generation, so the same projection that frames a measured asset is
    applied to a unit box: because framing scales linearly with box size, the
    largest fitting unit-box scale *is* the admissible envelope, in metres, for
    any asset placed at that site.
    """

    poses = native_fixed_camera_table_poses(DROID_NATIVE_FIXED_CAMERA_PRESET)
    checkable = {
        name: {
            "size_wh": [int(spec["size_wh"][0]), int(spec["size_wh"][1])],  # type: ignore[index]
            "minimum_linear_fraction": float(spec["minimum_linear_fraction"]),  # type: ignore[index]
        }
        for name, spec in required_views.items()
        if name in poses
    }
    if not checkable:
        raise ValueError("no required native view has a registered camera preset")
    intrinsics = {
        name: pinhole_intrinsics(spec["size_wh"], DROID_NATIVE_FIXED_FOV_Y_DEG)
        for name, spec in checkable.items()
    }
    ladder: list[float] = []
    scale = 1.0
    while scale >= 0.05 - 1.0e-9:
        ladder.append(round(scale, 10))
        scale -= step
    for scale in ladder:
        corners = box_corners_table(
            center_xy_m=center_xy_m,
            half_extents_m=(0.5 * scale, 0.5 * scale),
            bottom_z_m=0.0,
            top_z_m=float(scale),
        )
        framed = True
        for name, spec in checkable.items():
            measurement = framing_measurement(
                camera_from_table=poses[name],
                intrinsics=intrinsics[name],
                points_table=corners,
                size_wh=spec["size_wh"],
                margin_px=margin_px,
            )
            floor = spec["minimum_linear_fraction"] * minimum_fraction_safety
            framed = framed and bool(
                measurement["fully_in_frame"]
                and measurement["linear_fraction"] >= floor
            )
        if framed:
            return [float(scale), float(scale), float(scale)]
    raise ValueError(
        "the registered native views cannot frame any box at "
        f"{center_xy_m}; required views: {sorted(checkable)}"
    )


def box_corners_table(
    center_xy_m: Sequence[float],
    half_extents_m: Sequence[float],
    bottom_z_m: float,
    top_z_m: float,
) -> np.ndarray:
    """Return the eight corners of an axis-aligned task-object box."""

    center_x, center_y = (float(value) for value in center_xy_m)
    half_x, half_y = (float(value) for value in half_extents_m)
    bottom, top = float(bottom_z_m), float(top_z_m)
    if half_x <= 0.0 or half_y <= 0.0 or top < bottom:
        raise ValueError("task-object framing box must be non-degenerate")
    return np.asarray(
        [
            [x, y, z]
            for x in (center_x - half_x, center_x + half_x)
            for y in (center_y - half_y, center_y + half_y)
            for z in (bottom, top)
        ],
        dtype=np.float64,
    )


def framing_measurement(
    *,
    camera_from_table: np.ndarray,
    intrinsics: np.ndarray,
    points_table: np.ndarray,
    size_wh: Sequence[int],
    margin_px: float,
) -> dict[str, Any]:
    """Measure whether the projected box is fully inside the native frame."""

    width, height = (int(value) for value in size_wh)
    pixels, depth = project_table_points(camera_from_table, intrinsics, points_table)
    behind = bool(np.any(depth <= 0.0))
    x_min, y_min = (float(value) for value in pixels.min(axis=0))
    x_max, y_max = (float(value) for value in pixels.max(axis=0))
    in_frame = bool(
        not behind
        and x_min >= margin_px
        and y_min >= margin_px
        and x_max <= (width - 1.0) - margin_px
        and y_max <= (height - 1.0) - margin_px
    )
    linear_fraction = float(
        max((x_max - x_min) / float(width), (y_max - y_min) / float(height))
    )
    return {
        "projected_bbox_xyxy": [x_min, y_min, x_max, y_max],
        "minimum_depth_m": float(depth.min()),
        "point_behind_camera": behind,
        "fully_in_frame": in_frame,
        "linear_fraction": linear_fraction,
    }
