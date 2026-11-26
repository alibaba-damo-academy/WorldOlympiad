"""Small, simulator-independent pieces of the official RoboLab DROID contract."""

from __future__ import annotations

import math

import numpy as np


ROBOLAB_FINGER_JOINT_CLOSED_RAD = math.pi / 4.0


def normalize_robolab_gripper_position(joint_position_rad: float) -> float:
    """Return measured RoboLab gripper state where 0=open and 1=closed."""

    value = float(joint_position_rad)
    if not math.isfinite(value):
        raise ValueError("RoboLab finger joint position must be finite")
    return float(np.clip(value / ROBOLAB_FINGER_JOINT_CLOSED_RAD, 0.0, 1.0))


def executable_robolab_gripper_target(command: float) -> float:
    """Apply RoboLab's official >0.5 binary command threshold in radians."""

    value = float(command)
    if not math.isfinite(value):
        raise ValueError("RoboLab gripper command must be finite")
    return ROBOLAB_FINGER_JOINT_CLOSED_RAD if value > 0.5 else 0.0
