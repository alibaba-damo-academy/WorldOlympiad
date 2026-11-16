from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _gpu_allowlist(monkeypatch):
    monkeypatch.setenv("WOV2_GPU_ALLOWLIST", "0,1,2,3,4,5,6")

from wo.agent import config


def card(index: int, free_mib: int, *, utilization: int = 0, pids=()):
    return {
        "index": index,
        "uuid": f"gpu-{index}",
        "total_mib": 32_607,
        "free_mib": free_mib,
        "utilization": utilization,
        "compute_process_count": len(pids),
        "compute_pids": list(pids),
        "process_query_ok": True,
    }


def profile(gpu="auto"):
    return {
        "_config_path": "/candidate/config/two-interface.json",
        "workspace_root": "/candidate",
        "agent_api": {"base_url": "https://example.invalid/v1", "model": "test", "api_key_env": "WO_AGENT_API_KEY"},
        "tested_model": {
            "adapter": "lingbot_va", "checkpoint": "/weights/model",
            "gpu": gpu, "policy_gpus": "auto", "simulator_gpu": "auto",
        },
        "scoring": {"mode": "auto", "gpu": "auto", "compute_gpu": "auto"},
    }


def test_preparation_uses_one_idle_card_without_episode_fsdP_floor(monkeypatch):
    inventory = [
        card(i, 9_000 if i == 6 else 8_500 if i == 0 else 2_000,
             utilization=0 if i in {0, 6} else 70,
             pids=() if i in {0, 6} else (1000+i,))
        for i in range(7)
    ]
    assert config.resolve_preparation_gpu(profile(), inventory=inventory) == 6
    with pytest.raises(RuntimeError, match="30720"):
        config.resolve_gpu(profile(gpu=6), "tested_model", inventory=inventory)
    monkeypatch.setenv("WOV2_GPU_ALLOWLIST", "6")
    applied = config.apply_environment({}, profile(), inventory=inventory, phase="preparation")
    assert applied["WO_TESTED_MODEL_GPU"] == "6"
    assert applied["WO_TESTED_MODEL_GPUS"] == "6"


def test_preparation_rejects_card_with_compute_process_even_if_memory_is_free():
    inventory = [card(i, 31_000, utilization=0, pids=(2000+i,)) for i in range(7)]
    with pytest.raises(RuntimeError, match="found 0"):
        config.resolve_preparation_gpu(profile(), inventory=inventory)
