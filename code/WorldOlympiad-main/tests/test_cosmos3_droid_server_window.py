from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from PIL import Image

from autoresearch.cosmos3_droid_window import (
    checkpoint_frames_sha256,
    restore_decoded_droid_panels,
    run_server_window,
    recorded_prompt_format,
    validate_resume_identity,
    validate_resume_ledger,
    validate_replayed_state,
)


def test_prompt_format_provenance_cannot_be_invented_or_mixed():
    assert recorded_prompt_format([]) is None
    assert recorded_prompt_format([{'model_metadata': {}}]) is None
    def row(value):
        return {'model_metadata': {'format_prompt_as_json': value}}
    assert recorded_prompt_format([row(False), row(False)]) is False
    assert recorded_prompt_format([row(True), row(True)]) is True
    for values in [(False, True), (False, None), (True, 'True')]:
        with pytest.raises(RuntimeError, match='prompt format'):
            recorded_prompt_format([row(x) for x in values])


def test_decoded_droid_panel_restoration_preserves_exterior_coordinates() -> None:
    video = np.zeros((2, 528, 640, 3), dtype=np.uint8)
    video[:, 450, 160] = (10, 20, 30)
    video[:, 450, 480] = (40, 50, 60)
    wrist, left, right, metadata = restore_decoded_droid_panels(video)

    assert wrist.shape == (2, 360, 640, 3)
    assert left.shape == right.shape == (2, 180, 320, 3)
    np.testing.assert_array_equal(left[:, 90, 160], [[10, 20, 30]] * 2)
    np.testing.assert_array_equal(right[:, 90, 160], [[40, 50, 60]] * 2)
    assert not left[:, 168:].any()
    assert not right[:, 168:].any()
    assert metadata["exterior_bottom_padding"] == 12


@pytest.mark.parametrize('json_prompt', [False, True])
def test_server_window_matches_executable_droid_timeline(tmp_path, monkeypatch, json_prompt) -> None:
    future = np.zeros((32, 8), dtype=np.float32)
    future[:, :7] = np.arange(7, dtype=np.float32)
    future[:, 7] = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    video = np.zeros((33, 528, 640, 3), dtype=np.uint8)

    class FakeClient:
        def __init__(self, host: str, port: int) -> None:
            assert (host, port) == ("127.0.0.1", 28000)

        def infer(self, request):
            assert request["observation/image"].shape == (540, 640, 3)
            assert request["observation/joint_position"].shape == (7,)
            assert request["observation/gripper_position"].shape == (1,)
            return {
                "action": future,
                "video": video,
                "worldolympiad_request_index": 3,
                "worldolympiad_request_seed": 1234,
                "worldolympiad_format_prompt_as_json": json_prompt,
            }

    client_module = types.ModuleType("websocket_client_policy")
    client_module.WebsocketClientPolicy = FakeClient
    package = types.ModuleType("openpi_client")
    package.websocket_client_policy = client_module
    monkeypatch.setitem(sys.modules, "openpi_client", package)

    image_path = tmp_path / "conditioning.png"
    Image.fromarray(np.zeros((540, 640, 3), dtype=np.uint8)).save(image_path)
    joints = np.linspace(-0.3, 0.3, 7, dtype=np.float32)
    actions, raw, metadata = run_server_window(
        window_root=tmp_path / "window",
        input_image=image_path,
        joints=joints,
        gripper=0.25,
        prompt="Put the cup in the tray.",
        host="127.0.0.1",
        port=28000,
        request_index=3,
        seed_schedule_seed=7,
        expected_request_seed=1234,
    )

    assert actions.shape == raw.shape == (33, 8)
    np.testing.assert_allclose(actions[0, :7], joints)
    assert actions[0, 7] == 0.25
    assert set(np.unique(actions[1:, 7])) == {0.0, 1.0}
    np.testing.assert_allclose(raw[1:, 7], future[:, 7])
    assert metadata["request_index"] == 3
    assert metadata["request_seed"] == 1234
    assert metadata['format_prompt_as_json'] is json_prompt
    assert metadata["decoded_video_shape"] == [33, 528, 640, 3]
    assert (tmp_path / "window/model/actions.npy").is_file()
    assert (tmp_path / "window/model/droid_concat_rollout.npy").is_file()

    original_infer = FakeClient.infer

    def missing_format(self, request):
        response = original_infer(self, request)
        response.pop('worldolympiad_format_prompt_as_json')
        return response

    FakeClient.infer = missing_format
    with pytest.raises(RuntimeError, match='effective prompt format'):
        run_server_window(window_root=tmp_path/'missing_format', input_image=image_path,
            joints=joints,gripper=.25,prompt='Put the cup in the tray.',
            host='127.0.0.1',port=28000,request_index=3,seed_schedule_seed=7,
            expected_request_seed=1234)

    def diagnostic_infer(self, request):
        response = original_infer(self, request)
        response['worldolympiad_diagnostic_only'] = True
        return response

    FakeClient.infer = diagnostic_infer
    with pytest.raises(RuntimeError, match='diagnostic server'):
        run_server_window(window_root=tmp_path/'diagnostic', input_image=image_path,
            joints=joints,gripper=.25,prompt='Put the cup in the tray.',
            host='127.0.0.1',port=28000,request_index=3,seed_schedule_seed=7,
            expected_request_seed=1234)
    assert not (tmp_path/'diagnostic/model/actions.npy').exists()

    def drifted_infer(self, request):
        response = original_infer(self, request)
        response["worldolympiad_request_seed"] = 999
        return response

    FakeClient.infer = drifted_infer
    with pytest.raises(RuntimeError, match="request schedule differs"):
        run_server_window(
            window_root=tmp_path / "drifted_window",
            input_image=image_path,
            joints=joints,
            gripper=0.25,
            prompt="Put the cup in the tray.",
            host="127.0.0.1",
            port=28000,
            request_index=3,
            seed_schedule_seed=7,
            expected_request_seed=1234,
        )


def test_resume_identity_binds_task_world_prompt_and_seed() -> None:
    progress = {
        "schema_version": "cosmos_worldir_isaac_progress.v2",
        "official_runtime_task": "task.py:Task",
        "official_task_file_sha256": "a" * 64,
        "worldir_hash": "b" * 64,
        "prompt": "Put the red cup in the blue tray.",
        "seed_schedule_seed": 0,
        "runtime_identity_sha256": "c" * 64,
    }
    kwargs = {key: value for key, value in progress.items() if key != "schema_version"}
    validate_resume_identity(progress, **kwargs)
    for key, replacement in (
        ("official_runtime_task", "other.py:Task"),
        ("official_task_file_sha256", "c" * 64),
        ("worldir_hash", "d" * 64),
        ("prompt", "Use a different task."),
        ("seed_schedule_seed", 1),
        ("runtime_identity_sha256", "e" * 64),
    ):
        changed = dict(kwargs)
        changed[key] = replacement
        with pytest.raises(RuntimeError, match="resume identity differs"):
            validate_resume_identity(progress, **changed)


def test_resume_ledger_accepts_only_an_uncommitted_future_suffix() -> None:
    committed = [{
        "frame": 0,
        "sim_step": 0,
        "time_s": 0.0,
        "policy_source": "initial_worldir_state",
    }]
    committed.extend(
        {
            "frame": index,
            "sim_step": index * 8,
            "time_s": index / 15,
            "policy_source": "cosmos_window_000000",
        }
        for index in range(1, 33)
    )
    progress = {
        "seed_schedule_seed": 0,
        "completed_windows": 1,
        "current_frame_count": 33,
        "windows": [{
            "window_index": 0,
            "conditioning_frame": 0,
            "planned_action_count": 32,
            "executed_action_count": 32,
            "request_seed": 1826701615,
            "executed_actions_sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
        }],
        "checkpoint_frames_sha256": checkpoint_frames_sha256(committed),
    }
    future = {
        "frame": 33,
        "sim_step": 264,
        "time_s": 33 / 15,
        "policy_source": "cosmos_window_000001",
    }
    assert validate_resume_ledger(
        progress, {"frames": committed + [future]}, official_max_steps=750
    ) == committed

    changed = [dict(item) for item in committed]
    changed[1]["frame"] = 99
    with pytest.raises(RuntimeError, match="prefix digest changed"):
        validate_resume_ledger(progress, {"frames": changed}, official_max_steps=750)


def test_official_request_seed_schedule_has_frozen_resume_anchor() -> None:
    seeds = np.random.default_rng(0).integers(0, 2**31, size=5).tolist()
    assert seeds == [1826701615, 1367864807, 1097657232, 579362556, 661058652]


def test_replayed_state_checks_all_fields_and_quaternion_sign() -> None:
    expected = {
        "robot_joint_positions": [0.1, 0.2],
        "robot_joint_velocities": [0.0, 0.0],
        "robot_base_position": [0.0, 0.0, 0.0],
        "robot_base_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "robot_base_linear_velocity": [0.0, 0.0, 0.0],
        "robot_base_angular_velocity": [0.0, 0.0, 0.0],
        "cup_position": [0.1, 0.2, 0.8],
        "cup_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "cup_linear_velocity": [0.0, 0.0, 0.0],
        "cup_angular_velocity": [0.0, 0.0, 0.0],
        "tray_position": [0.4, 0.2, 0.7],
        "tray_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "tray_linear_velocity": [0.0, 0.0, 0.0],
        "tray_angular_velocity": [0.0, 0.0, 0.0],
    }
    actual = {key: list(value) for key, value in expected.items()}
    actual["cup_orientation_xyzw"] = [0.0, 0.0, 0.0, -1.0]
    validate_replayed_state(expected, actual)
    actual["robot_joint_velocities"][0] = 0.01
    with pytest.raises(RuntimeError, match="robot_joint_velocities"):
        validate_replayed_state(expected, actual)
