import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from autoresearch.lingbot_world2_adapter import (
    MODEL_ID,
    PROTOCOL,
    compile_event_prompt,
    prepare_world2_case,
    validate_event_prompt,
    validate_input_contract,
)
from autoresearch.lingbot_world2_harness import _finalize, _safe_run_id, validate_artifacts
from autoresearch.lingbot_world_physics import digest, profile_for_model_id


PROMPT = (
    "A red rubber ball is visible above a wooden table. "
    "The ball is released immediately. "
    "It falls straight down, strikes the table, bounces once, and settles."
)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    reference = source / "reference"
    reference.mkdir(parents=True)
    Image.new("RGB", (832, 464), (20, 30, 40)).save(reference / "camera_rgb.png")
    calibration = {
        "width": 832,
        "height": 464,
        "intrinsics": [[600, 0, 416], [0, 600, 232], [0, 0, 1]],
        "camera_to_world_opencv": np.eye(4).tolist(),
    }
    (reference / "calibration.json").write_text(json.dumps(calibration))
    (reference / "scene_data.json").write_text(json.dumps({
        "initial_component_region": {"bbox_xyxy": [300, 100, 500, 300]},
    }))
    (reference / "rgb.mp4").write_bytes(b"reference-video-placeholder")
    np.savez(
        reference / "state_trajectory.npz",
        body_names=np.array(["ball"]),
        positions=np.zeros((2, 1, 3)),
        linear_velocities=np.zeros((2, 1, 3)),
    )
    (source / "simulation.json").write_text(json.dumps({
        "duration_s": 4.0,
        "task_id": "drop_ball",
        "instruction": "Release the ball.",
        "fps": 16,
        "frames": 65,
    }))
    return source


def test_14b_profile_is_not_the_old_13b_sampling_profile() -> None:
    assert profile_for_model_id(MODEL_ID) == {"task": "i2v-A14B", "sample_shift": 10.0}


def test_event_prompt_compiler_and_static_scene_rejection() -> None:
    prompt = compile_event_prompt(
        "A blue ball is visible above a table",
        "The ball is released immediately",
        ["It falls", "It hits the table", "It bounces and settles"],
    )
    assert "released immediately" in prompt
    assert validate_event_prompt(prompt) == prompt
    with pytest.raises(ValueError, match="static-scene"):
        validate_event_prompt(prompt + " The camera and background remain stationary.")


def test_prepare_emits_a_strict_14b_contract(tmp_path: Path) -> None:
    output = tmp_path / "inputs"
    receipt = prepare_world2_case(_source(tmp_path), output, PROMPT)
    assert receipt["model_id"] == MODEL_ID
    assert receipt["sampling"]["task"] == "i2v-A14B"
    assert receipt["sampling"]["sample_shift"] == 10.0
    assert receipt["sampling"]["output_wh"] == [832, 464]
    assert validate_input_contract(output / "input_contract.json") == receipt


def test_contract_rejects_static_prompt_even_with_recomputed_digest(tmp_path: Path) -> None:
    output = tmp_path / "inputs"
    prepare_world2_case(_source(tmp_path), output, PROMPT)
    path = output / "input_contract.json"
    value = json.loads(path.read_text())
    prompt = value["conditioning"]["prompt"] + " The camera remains stationary."
    value["conditioning"]["prompt"] = prompt
    value["conditioning"]["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="static-scene"):
        validate_input_contract(path)


def test_published_artifacts_are_digest_bound_and_resumable(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    artifacts = run_root / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "imagination.mp4").write_bytes(b"video")
    request = {"identity_sha256": "a" * 64}
    result = {
        "status": "complete",
        "request_identity_sha256": request["identity_sha256"],
        "finite": True,
        "full_decode_verified": True,
        "artifact_sha256": {"imagination.mp4": digest(artifacts / "imagination.mp4")},
    }
    (artifacts / "result.json").write_text(json.dumps(result))
    assert validate_artifacts(artifacts, request["identity_sha256"]) == result
    manifest = _finalize(run_root, request, result)
    assert manifest["protocol"] == PROTOCOL
    (artifacts / "imagination.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_artifacts(artifacts, request["identity_sha256"])


@pytest.mark.parametrize("value", ["", "../escape", "has space", "x;touch"])
def test_run_id_rejects_path_or_shell_fragments(value: str) -> None:
    with pytest.raises(ValueError):
        _safe_run_id(value)
