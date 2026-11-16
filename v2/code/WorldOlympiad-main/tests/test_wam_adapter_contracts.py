from __future__ import annotations

from autoresearch.wam_adapter_contracts import (
    LINGBOT_ROBOTWIN_WAM,
    build_lingbot_robotwin_adapter_contract,
)


def test_lingbot_robotwin_contract_is_native_closed_loop() -> None:
    contract = build_lingbot_robotwin_adapter_contract(
        case_id="place_empty_cup_seed10000",
        reference_fps=15.0,
        reference_frame_count=187,
    )

    assert contract.wam == LINGBOT_ROBOTWIN_WAM
    assert contract.effective_rollout_protocol == "simulator_closed_loop"
    assert contract.native_input_size_wh == (320, 384)
    assert contract.source_views == {
        "observation.images.cam_high": "native_high",
        "observation.images.cam_left_wrist": "native_left_wrist",
        "observation.images.cam_right_wrist": "native_right_wrist",
    }
    assert contract.output.formal_view == "native_high"
    assert contract.output.formal_size_wh == (320, 256)
    assert contract.temporal.duration_policy == "upstream_runtime_owned_v1"


def test_lingbot_robotwin_contract_round_trips_json() -> None:
    contract = build_lingbot_robotwin_adapter_contract(
        case_id="adjust_bottle_seed10000",
        reference_fps=15.0,
        reference_frame_count=145,
    )

    assert type(contract).model_validate_json(contract.model_dump_json()) == contract
