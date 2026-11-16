import numpy as np
import pytest

from wo.models.fastwam_server import CAMERAS, decode_observation, encode_observation, infer_payload


def test_local_asset_config_preserves_upstream_model(tmp_path):
    from wo.models.fastwam_server import local_asset_config
    (tmp_path / 'configs').mkdir()
    with local_asset_config(tmp_path) as path:
        text = path.read_text()
        assert '  - sim_robotwin' in text
        assert 'redirect_common_files: false' in text
        assert 'skip_dit' not in text
    assert not path.exists()


def observation():
    return {"observation": {name: {"rgb": np.full((8, 10, 3), i, dtype=np.uint8)}
                            for i, name in enumerate(CAMERAS)},
            "joint_action": {"vector": np.linspace(-1, 1, 14)}}


def test_raw_native_observation_roundtrip():
    original = observation()
    decoded, prompt = decode_observation(encode_observation(original, "move cube"))
    assert prompt == "move cube"
    np.testing.assert_allclose(decoded["joint_action"]["vector"], original["joint_action"]["vector"])
    for name in CAMERAS:
        np.testing.assert_array_equal(decoded["observation"][name]["rgb"], original["observation"][name]["rgb"])


@pytest.mark.parametrize("qpos", [[0] * 13, [float("nan")] * 14])
def test_reject_invalid_native_state(qpos):
    payload = encode_observation(observation(), "move")
    payload["joint_qpos"] = qpos
    with pytest.raises(ValueError, match="14-D"):
        decode_observation(payload)


def test_reject_truncated_camera():
    payload = encode_observation(observation(), "move")
    payload["cameras"][CAMERAS[0]]["data"] = ""
    with pytest.raises(ValueError, match="byte count"):
        decode_observation(payload)


def test_upstream_inference_is_not_renormalized_or_converted_to_ee():
    class Policy:
        replan_steps = 24

        def _infer_action_chunk(self, observation, instruction):
            assert instruction == "move"
            assert observation["joint_action"]["vector"].shape == (14,)
            return np.tile(np.arange(14), (32, 1))

    result = infer_payload(Policy(), encode_observation(observation(), "move"))
    assert result["action_type"] == "qpos"
    assert result["replan_steps"] == 24
    assert result["actions"] == [list(range(14))] * 32


def test_bad_policy_outputs_are_infrastructure_errors():
    class Policy:
        replan_steps = 24

        def _infer_action_chunk(self, **kwargs):
            return np.zeros((32, 7))

    with pytest.raises(RuntimeError, match="invalid absolute joint"):
        infer_payload(Policy(), encode_observation(observation(), "move"))
