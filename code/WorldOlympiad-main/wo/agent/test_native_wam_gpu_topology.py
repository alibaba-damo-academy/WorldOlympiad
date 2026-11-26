"""Native WAM topology is selected from fresh model and simulator headroom."""
import pytest

from wo.agent import config


@pytest.fixture(autouse=True)
def allowlist(monkeypatch):
    monkeypatch.setenv("WOV2_GPU_ALLOWLIST", "0,1")


def inventory(*, model_free=70_000, busy_second=False):
    return [{"index": index, "total_mib": 80_000,
             "free_mib": model_free if index == 0 else 70_000,
             "utilization": 0 if index == 0 or not busy_second else 100,
             "compute_process_count": 0 if index == 0 or not busy_second else 1,
             "compute_pids": [] if index == 0 or not busy_second else [321],
             "process_query_ok": True} for index in (0, 1)]


def profile(adapter, tmp_path):
    return {"workspace_root": str(tmp_path), "_config_path": str(tmp_path / "config.json"),
            "tested_model": {"adapter": adapter, "checkpoint": "weights/model"},
            "agent_api": {"base_url": "https://example.invalid/v1", "model": "test"}}


@pytest.mark.parametrize("adapter", ["fastwam", "longwam"])
def test_native_wam_splits_when_model_card_cannot_hold_both(adapter, tmp_path, monkeypatch):
    settings = profile(adapter, tmp_path)
    cards = inventory()
    assert config.resolve_simulator_gpu(settings, 0, inventory=cards) == 1
    env = config.apply_environment({}, settings, inventory=cards, phase="episode")
    assert env["WO_TESTED_MODEL_GPU"] == "0"
    assert env["WO_SIMULATOR_GPU"] == "1"
    monkeypatch.setattr(config, "all_gpu_info", lambda: cards)
    assert config.job_defaults("episode", {"model": adapter}, profile=settings)["simulator_gpu"] == 1


@pytest.mark.parametrize("adapter", ["fastwam", "longwam"])
def test_native_wam_shares_only_with_combined_headroom(adapter, tmp_path, monkeypatch):
    settings = profile(adapter, tmp_path)
    cards = inventory(model_free=78_000, busy_second=True)
    assert config.resolve_simulator_gpu(settings, 0, inventory=cards) == 0
    env = config.apply_environment({}, settings, inventory=cards, phase="episode")
    assert env["WO_TESTED_MODEL_GPU"] == env["WO_SIMULATOR_GPU"] == "0"
    monkeypatch.setattr(config, "all_gpu_info", lambda: cards)
    assert config.job_defaults("episode", {"model": adapter}, profile=settings)["simulator_gpu"] == 0
    settings["tested_model"]["simulator_gpu"] = 0
    assert config.resolve_simulator_gpu(settings, 0, inventory=cards) == 0


@pytest.mark.parametrize("adapter", ["fastwam", "longwam"])
def test_native_wam_rejects_insufficient_combined_and_second_card(adapter, tmp_path):
    settings = profile(adapter, tmp_path)
    cards = inventory(busy_second=True)
    with pytest.raises(RuntimeError, match="combined headroom or a second idle"):
        config.apply_environment({}, settings, inventory=cards, phase="episode")
    settings["tested_model"]["simulator_gpu"] = 0
    with pytest.raises(RuntimeError, match="shared native WAM GPU needs"):
        config.resolve_simulator_gpu(settings, 0, inventory=cards)
