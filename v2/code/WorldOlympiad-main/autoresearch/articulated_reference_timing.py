"""Predeclare smooth articulated reference timing before model inference."""
import math


def reference_timing(binding: dict) -> dict:
    kind=binding["joint_type"]
    speeds={"revolute":0.08,"prismatic":0.04}
    if kind not in speeds: raise ValueError("unsupported articulated timing joint")
    travel=abs(float(binding["open_position"])-float(binding["closed_position"]))
    if not math.isfinite(travel) or travel<=0: raise ValueError("articulated travel must be finite and positive")
    # Whole seconds make the schedule explicit on the shared 15 Hz clock.
    seconds=max(2,math.ceil(travel/speeds[kind]))
    steps=seconds*15
    if steps>900: raise ValueError("smooth articulated reference exceeds the executor budget")
    return {"policy":"declared_articulated_reference_speed.v1","joint_type":kind,
            "control_hz":15,"actuate_steps":steps,"actuation_duration_s":seconds,
            "declared_joint_travel":travel,"commanded_average_speed_limit":speeds[kind],
            "instantaneous_measured_speed_bounded":False,"model_output_consulted":False}
