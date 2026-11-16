"""Record all declared dynamic scene objects from current native physics."""
from __future__ import annotations
from copy import deepcopy
from typing import Any

STATE_FIELDS = ("position", "orientation_xyzw", "link_position",
                "link_orientation_xyzw", "linear_velocity", "angular_velocity")


def live_object_records(reference_objects: dict[str, dict], bindings: list,
                        body_states: dict[str, dict]) -> dict[str, dict]:
    result = deepcopy(reference_objects)
    for binding in bindings:
        name = binding.trajectory_object_name
        if name not in result:
            continue
        current = body_states.get(binding.body_name)
        if current is None:
            if binding.dynamic:
                raise ValueError(f"dynamic object has no live native state: {name}")
            continue
        for key in (*STATE_FIELDS, 'link_states', 'joint_states', 'measurement_source',
                    'initial_joint_limit_excursions'):
            if key in current:
                result[name][key] = deepcopy(current[key])
        if 'joint_states' in current:
            result[name]['joint_positions'] = {joint: value['position']
                                              for joint, value in current['joint_states'].items()}
    return result


def dynamic_state_checkpoint(bindings: list, body_states: dict[str, dict]) -> dict[str, list]:
    result = {}
    for binding in bindings:
        if not binding.dynamic:
            continue
        current = body_states[binding.body_name]
        for key in STATE_FIELDS:
            if key in current:
                result[f"scene:{binding.trajectory_object_name}:{key}"] = list(current[key])
    return result
