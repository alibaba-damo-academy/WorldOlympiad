"""Small quaternion/pose helpers shared by import and compilation."""

from __future__ import annotations

import math

from .models import Pose


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quat_multiply(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = left
    bx, by, bz, bw = right
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate_vector(
    quat: tuple[float, float, float, float], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
    x, y, z, w = quat
    vx, vy, vz = vector
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def compose(left: Pose, right: Pose) -> Pose:
    translated = rotate_vector(left.quat_xyzw, right.xyz)
    return Pose(
        xyz=tuple(left.xyz[i] + translated[i] for i in range(3)),
        quat_xyzw=quat_multiply(left.quat_xyzw, right.quat_xyzw),
    )


def pose_from_sdf(text: str | None) -> Pose:
    values = [float(value) for value in (text or "").split()]
    values += [0.0] * (6 - len(values))
    return Pose(xyz=tuple(values[:3]), quat_xyzw=rpy_to_quat(*values[3:6]))


def pose_from_scenecode(data: dict | None) -> Pose:
    if not data:
        return Pose()
    w, x, y, z = data.get("rotation_wxyz", [1.0, 0.0, 0.0, 0.0])
    return Pose(
        xyz=tuple(float(value) for value in data.get("translation", [0.0, 0.0, 0.0])),
        quat_xyzw=(float(x), float(y), float(z), float(w)),
    )

