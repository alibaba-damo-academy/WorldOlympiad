import math
import pytest
from autoresearch.articulated_reference_timing import reference_timing


@pytest.mark.parametrize("kind,travel,steps",[("revolute",math.pi/2,300),("prismatic",.3,120),("prismatic",.005,30)])
def test_declared_motion_is_smooth_and_time_is_set_before_model(kind,travel,steps):
    result=reference_timing({"joint_type":kind,"open_position":-travel,"closed_position":0})
    assert result["actuate_steps"]==steps
    assert travel/result["actuation_duration_s"]<=result["commanded_average_speed_limit"]
    assert result["model_output_consulted"] is False


@pytest.mark.parametrize("travel",[0,float("nan"),float("inf"),10])
def test_invalid_or_over_budget_travel_is_rejected(travel):
    with pytest.raises(ValueError):
        reference_timing({"joint_type":"revolute","open_position":travel,"closed_position":0})
