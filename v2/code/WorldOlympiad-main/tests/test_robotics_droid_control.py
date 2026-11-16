import math

import pytest

from autoresearch.robotics_droid_control import (
    executable_robolab_gripper_target,
    normalize_robolab_gripper_position,
)


def test_measured_gripper_feedback_stays_continuous() -> None:
    assert normalize_robolab_gripper_position(0.0) == 0.0
    assert normalize_robolab_gripper_position(math.pi / 8.0) == pytest.approx(0.5)
    assert normalize_robolab_gripper_position(math.pi / 4.0) == 1.0
    assert normalize_robolab_gripper_position(-0.1) == 0.0
    assert normalize_robolab_gripper_position(1.0) == 1.0


def test_future_gripper_commands_use_official_binary_threshold() -> None:
    assert executable_robolab_gripper_target(0.5) == 0.0
    assert executable_robolab_gripper_target(0.5001) == pytest.approx(math.pi / 4.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_gripper_values_fail_closed(value: float) -> None:
    with pytest.raises(ValueError):
        normalize_robolab_gripper_position(value)
    with pytest.raises(ValueError):
        executable_robolab_gripper_target(value)
