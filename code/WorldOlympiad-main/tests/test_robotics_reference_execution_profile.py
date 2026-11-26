import json
import math
import pytest

from autoresearch.robotics_reference_execution_profile import (
    validate_parameters, load_profile, apply_profile, FLAGS, PARAMETER_PROPERTIES,
)


def test_declared_content_parameters_have_native_cli_bindings():
    assert set(PARAMETER_PROPERTIES) == set(FLAGS)


def native_command():
    return ["python", "native.py", "--grasp-offset-xy-m", "0", "0",
            "--grasp-base-offset-z-m", "0.15", "--grasp-yaw-deg", "0",
            "--success-predicate", "inside", "--frame-count", "200"]


def test_profile_changes_only_existing_pose_flags_and_retains_protocol():
    command = native_command()
    output = apply_profile(command, {"grasp_yaw_deg": 20, "grasp_offset_xy_m": [0.01, 0]})
    assert command == native_command()
    assert output[output.index("--grasp-yaw-deg") + 1] == "20"
    assert output[output.index("--success-predicate") + 1] == "inside"
    assert output[output.index("--frame-count") + 1] == "200"
    with pytest.raises(ValueError, match="does not change"):
        apply_profile(command, {"grasp_yaw_deg": 0.0})
    with pytest.raises(ValueError, match="does not support"):
        apply_profile(["python", "other.py"], {"grasp_yaw_deg": 20})


@pytest.mark.parametrize("parameters", [
    {}, {"gravity": 0}, {"success_threshold": 0}, {"duration_sec": 100},
    {"grasp_yaw_deg": True}, {"grasp_yaw_deg": math.nan},
    {"grasp_yaw_deg": 181}, {"grasp_offset_xy_m": [0.1, 0.1]},
    {"grasp_offset_xy_m": [0]}, {"grasp_base_offset_z_m": 0.9},
])
def test_profile_rejects_physics_predicate_time_or_invalid_pose_changes(parameters):
    with pytest.raises(ValueError):
        validate_parameters(parameters)


def test_profile_binds_task_case_and_controlled_object(tmp_path):
    path = tmp_path / "pose.json"
    payload = {"schema_version": "robotics_reference_execution_profile.v1",
        "task_file_sha256": "frozen-task", "batch_id": "batch", "case_id": "case",
        "object_id": "subject", "parameters": {"grasp_yaw_deg": 20}}
    path.write_text(json.dumps(payload))
    args = dict(task_sha256="frozen-task", batch_id="batch", case_id="case", object_id="subject")
    before = path.read_bytes()
    assert load_profile(path, **args)["parameters"]["grasp_yaw_deg"] == 20
    assert path.read_bytes() == before
    for key in args:
        with pytest.raises(ValueError, match="another task"):
            load_profile(path, **{**args, key: "another"})
