from __future__ import annotations

import pytest

from continuous_action.workspaces import (
    robot_workspace_contract_for_model,
    validate_robot_workspace_contract,
)


@pytest.mark.parametrize(
    ("model", "diameter"),
    [
        ("cosmos3_nano_droid", 1.71),
        ("lingbot_va_robotwin", 1.824341085722686),
    ],
)
def test_reviewed_workspace_contract_round_trip(model: str, diameter: float) -> None:
    contract = robot_workspace_contract_for_model(model)
    assert contract["diameter_m"] == diameter
    assert validate_robot_workspace_contract(contract, source_model=model) == contract


def test_workspace_contract_rejects_case_tuning() -> None:
    contract = robot_workspace_contract_for_model("cosmos3_nano_droid")
    contract["diameter_m"] = 2.0
    with pytest.raises(ValueError, match="contract_sha256"):
        validate_robot_workspace_contract(contract, source_model="cosmos3_nano_droid")


def test_unknown_adapter_cannot_fall_back_to_video_scale() -> None:
    with pytest.raises(ValueError, match="no reviewed robot-workspace contract"):
        robot_workspace_contract_for_model("unknown_wam")
