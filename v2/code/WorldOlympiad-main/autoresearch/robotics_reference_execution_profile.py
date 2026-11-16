"""Versioned expert pose parameters; task semantics and physics stay authoritative."""
from __future__ import annotations

import json
import math
from pathlib import Path

FLAGS = {"grasp_offset_xy_m": ("--grasp-offset-xy-m", 2),
         "grasp_base_offset_z_m": ("--grasp-base-offset-z-m", 1),
         "grasp_yaw_deg": ("--grasp-yaw-deg", 1)}

# Content-owned declarations are also the agent tool's parameter contract.
PARAMETER_PROPERTIES = {
    "grasp_offset_xy_m": {"type": ["array", "null"], "minItems": 2, "maxItems": 2,
        "items": {"type": "number"}, "description": "Projected simulation world XY offset from the controlled subject root, metres; native feature envelope applies."},
    "grasp_base_offset_z_m": {"type": ["number", "null"], "minimum": 0.12, "maximum": 0.55,
        "description": "Native end-effector base height above the support; also validated against the projected subject and approach height."},
    "grasp_yaw_deg": {"type": ["number", "null"], "minimum": -180, "maximum": 180},
}


def validate_parameters(parameters: dict) -> dict:
    if not isinstance(parameters, dict) or set(parameters) - set(FLAGS):
        raise ValueError("expert profile only supports declared grasp pose parameters")
    selected = {key: value for key, value in parameters.items() if value is not None}
    if not selected:
        raise ValueError("expert profile must propose an actual pose change")
    for key, value in selected.items():
        values = value if key == "grasp_offset_xy_m" else [value]
        if (not isinstance(values, (list, tuple)) or len(values) != FLAGS[key][1]
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) for x in values)):
            raise ValueError(f"invalid finite pose parameter: {key}")
        if key == "grasp_offset_xy_m" and math.hypot(*values) > 0.10:
            raise ValueError("grasp offset exceeds the existing native feature envelope")
        if key == "grasp_base_offset_z_m" and not 0.12 <= value <= 0.55:
            raise ValueError("grasp base offset exceeds the existing native adapter envelope")
        if key == "grasp_yaw_deg" and not -180 <= value <= 180:
            raise ValueError("grasp yaw must be within [-180,180] degrees")
        selected[key] = [float(x) for x in values] if key == "grasp_offset_xy_m" else float(value)
    return selected


def load_profile(path: Path, *, task_sha256: str, batch_id: str, case_id: str,
                 object_id: str) -> dict:
    profile = json.loads(path.read_text())
    if (profile.get("schema_version") != "robotics_reference_execution_profile.v1"
            or profile.get("task_file_sha256") != task_sha256
            or profile.get("batch_id") != batch_id or profile.get("case_id") != case_id
            or profile.get("object_id") != object_id):
        raise ValueError("expert pose profile belongs to another task or controlled subject")
    profile["parameters"] = validate_parameters(profile["parameters"])
    return profile


def apply_profile(argv: list[str], parameters: dict) -> list[str]:
    """Replace existing pose flags, never insert physics or predicate flags."""
    selected = validate_parameters(parameters)
    output = list(argv)
    changed = False
    for key, value in selected.items():
        flag, count = FLAGS[key]
        if output.count(flag) != 1:
            raise ValueError(f"this native reference does not support {flag}")
        index = output.index(flag) + 1
        values = list(value) if count == 2 else [value]
        previous = [float(x) for x in output[index:index + count]]
        if len(previous) != count:
            raise ValueError("native pose flag is truncated")
        changed |= previous != values
        output[index:index + count] = [format(x, ".17g") for x in values]
    if not changed:
        raise ValueError("expert pose profile does not change the measured native parameters")
    return output
