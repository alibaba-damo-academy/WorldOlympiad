import io
import json

import numpy as np
import pytest

from wo.models.fastwam_client import FastWAMClient, executable_prefix
from wo.models.fastwam_server import CAMERAS


def response():
    return {"action_type": "qpos", "actions": np.arange(32 * 14).reshape(32, 14).tolist(),
            "replan_steps": 24}


def test_prefix_preserves_absolute_native_actions():
    value = response()
    np.testing.assert_array_equal(executable_prefix(value), np.asarray(value["actions"])[:24])


@pytest.mark.parametrize("patch", [
    {"action_type": "ee"}, {"actions": [[0] * 7]},
    {"actions": [[float("nan")] * 14]}, {"replan_steps": 0},
    {"replan_steps": 33}, {"replan_steps": True}, {"replan_steps": 2.5},
])
def test_protocol_errors_not_task_failures(patch):
    value = response()
    value.update(patch)
    with pytest.raises(RuntimeError):
        executable_prefix(value)


def test_http_transports_native_cameras_and_joint_state(monkeypatch):
    obs = {"observation": {name: {"rgb": np.zeros((8, 10, 3), dtype=np.uint8)}
                            for name in CAMERAS}, "joint_action": {"vector": np.arange(14)}}

    def open_request(request, timeout):
        assert request.full_url == "http://127.0.0.1:29058/infer"
        assert request.method == "POST"
        assert timeout == 300
        payload = json.loads(request.data)
        assert payload["joint_qpos"] == list(range(14))
        assert set(payload["cameras"]) == set(CAMERAS)
        return io.BytesIO(json.dumps(response()).encode())

    monkeypatch.setattr("wo.models.fastwam_client.urlopen", open_request)
    assert FastWAMClient().infer(obs, "move cube").shape == (24, 14)
