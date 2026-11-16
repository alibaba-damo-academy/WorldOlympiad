from unittest import mock

import numpy as np
import pytest

from autoresearch.articulated_input_contract import audit_initial_view


def test_native_wrist_target_may_be_offscreen_when_exterior_handle_is_visible():
    config = {"source_sdf": "unused"}
    state = {
        "child_link_xyzw": [0, 0, 0, 1],
        "child_link_position": [0, 0, 0],
        "physical_valid": True,
        "finger_contacts": [False, False],
        "robot": {"gripper": 0.0},
        "frame": 0,
    }
    calibration = {"cameras": {name: {"name": name} for name in (
        "wrist", "exterior_left", "exterior_right")}}
    projected = [
        {"center_in_frame": False, "bbox_visible_fraction": 0.0},
        {"center_in_frame": True, "whole_handle_in_frame": True},
        {"center_in_frame": False, "whole_handle_in_frame": False},
    ]
    with mock.patch(
        "autoresearch.articulated_input_contract.target_corners",
        return_value=np.zeros((8, 3)),
    ), mock.patch(
        "autoresearch.articulated_input_contract.project_target",
        side_effect=projected,
    ):
        report = audit_initial_view(config, state, calibration, {
            "contact_target_local_xyz": [0, 0, 0]
        })
    assert report["status"] == "passed"
    assert report["wrist_handle_visible"] is False
    assert report["context_contains_whole_handle"] is True
    projected[1] = {"center_in_frame": False, "whole_handle_in_frame": False}
    with mock.patch(
        "autoresearch.articulated_input_contract.target_corners",
        return_value=np.zeros((8, 3)),
    ), mock.patch(
        "autoresearch.articulated_input_contract.project_target",
        side_effect=projected,
    ), pytest.raises(ValueError, match="not observable"):
        audit_initial_view(config, state, calibration, {
            "contact_target_local_xyz": [0, 0, 0]
        })
