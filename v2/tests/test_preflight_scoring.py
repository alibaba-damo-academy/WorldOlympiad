from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "scripts/preflight.py"
spec = importlib.util.spec_from_file_location("wov2_preflight", PRE)
preflight = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(preflight)


def inventory(eligible: set[int], *, eligible_free_mib: int = 31_000):
    cards = []
    for index in range(7):
        free = eligible_free_mib if index in eligible else 4_000
        pids = [] if index in eligible else [90_000 + index]
        cards.append({
            "index": index,
            "name": "NVIDIA A100",
            "uuid": f"gpu-{index}",
            "total_mib": 32_607,
            "free_mib": free,
            "utilization": 0 if index in eligible else 90,
            "compute_apps": [],
            "compute_app_pids": pids,
            "compute_process_count": len(pids),
            "process_query_ok": True,
        })
    return {"queried_gpu_indices": list(range(7)), "cards": cards}


def profile():
    return {
        "tested_model": {"adapter": "lingbot_va", "gpu": "auto"},
        "scoring": {"mode": "auto", "gpu": "auto", "compute_gpu": "auto"},
    }


def test_split_score_preflight_selects_two_distinct_eligible_cards():
    result = preflight.assess_resources(inventory({1, 4}), "scoring", profile())
    assert result["ready"] is True
    assert result["scoring_mode"] == "split"
    assert result["required_scoring_gpu_count"] == 2
    assert result["selected_vlm_gpu_index"] != result["selected_compute_gpu_index"]
    assert {result["selected_vlm_gpu_index"], result["selected_compute_gpu_index"]} == {1, 4}


def test_split_score_preflight_blocks_with_only_one_eligible_card():
    result = preflight.assess_resources(inventory({5}), "scoring", profile())
    assert result["ready"] is False
    assert result["eligible_scoring_gpu_indices"] == [5]
    assert result["selected_vlm_gpu_index"] is None
    assert result["selected_compute_gpu_index"] is None
    assert "found 1/2" in result["blocking_reason"]
    assert len(result["card_deficits"]) == 7


def test_preparation_preflight_uses_one_card_with_eight_gib_headroom():
    result = preflight.assess_resources(
        inventory({6}, eligible_free_mib=9_000), "preparation", profile())
    assert result["ready"] is True
    assert result["required_preparation_gpu_count"] == 1
    assert result["preparation_memory_threshold_mib"] == 8_192
    assert result["selected_preparation_gpu_index"] == 6
