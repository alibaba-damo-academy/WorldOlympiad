"""Focused CPU contracts for the registered LingBot adapter preparation."""
from __future__ import annotations

import ast
import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ["WO_EVAL_CONFIG"] = "/tmp/wo-lingbot-prep-profile-missing.json"

import numpy as np
import pytest

from wo.agent import config, stages
from wo.models import lingbot_va, lingbot_va_client


def test_adapter_is_registered_and_exposes_native_ee_wire_contract():
    assert stages.MODELS["lingbot_va"] == "wo.models.lingbot_va"
    assert callable(lingbot_va.main)
    initial = np.array([0.1, 0.2, 0.9, 0, 0, 0, 1, 1.0,
                        -0.1, 0.2, 0.9, 0, 0, 0, 1, 0.0])
    raw = np.array([0.01, 0, 0, 0, 0, 0, 1, 0.5,
                    0, 0.02, 0, 0, 0, 0, 1, 0.2])
    decoded = lingbot_va_client.decode_action(raw, initial)
    assert decoded[:3] == pytest.approx([0.11, 0.2, 0.9])
    assert decoded[7] == pytest.approx(0.5)
    assert decoded[8:11] == pytest.approx([-0.1, 0.22, 0.9])
    assert decoded[15] == pytest.approx(0.2)
    assert lingbot_va_client.executable_actions({"action_type": "ee", "actions": [raw.tolist()]}) .shape == (1, 16)
    with pytest.raises(RuntimeError, match="action_type"):
        lingbot_va_client.executable_actions({"action_type": "qpos", "actions": [raw.tolist()]})


def test_gpu_inventory_queries_are_allowlisted_and_scheduler_filters_indices(monkeypatch):
    monkeypatch.setenv("WOV2_GPU_ALLOWLIST", "2,8")
    calls = []

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(command, **kwargs):
        calls.append(command)
        if any(str(part).startswith("--query-gpu=") for part in command):
            return Result("2, GPU-2, 81920, 40000, 1\n8, GPU-8, 81920, 81919, 0\n")
        return Result("")

    monkeypatch.setattr(config.subprocess, "run", fake_run)
    inventory = config.all_gpu_info()
    assert len(calls) == 2
    assert all(command[command.index("-i") + 1] == "2,8" for command in calls)
    assert all("nvidia-smi" == command[0] for command in calls)
    assert [item["index"] for item in inventory] == [2, 8]
    # Even caller-supplied stale inventories cannot make index 7 schedulable.
    injected = [
        {"index": 7, "total_mib": 90000, "free_mib": 89999, "utilization": 0},
        {"index": 2, "total_mib": 81920, "free_mib": 40000, "utilization": 0},
        {"index": 8, "total_mib": 81920, "free_mib": 80000, "utilization": 0},
    ]
    profile = {"tested_model": {"gpu": "auto", "adapter": "cosmos3"}}
    assert config.resolve_gpu(profile, inventory=injected) == 8
    with pytest.raises(ValueError, match="WOV2_GPU_ALLOWLIST"):
        config.resolve_gpu({"tested_model": {"gpu": 7, "adapter": "cosmos3"}}, inventory=injected)
    with patch.object(config, "load_profile", return_value=None):
        with pytest.raises(ValueError, match="WOV2_GPU_ALLOWLIST"):
            config.job_defaults("episode", {"run": "r", "case": "c", "model": "cosmos3", "gpu": 7})


def test_config_rejects_gpu_indices_outside_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("WOV2_GPU_ALLOWLIST", "0,1,2,3,4,5,6")
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({
        "schema_version": 1, "workspace_root": str(tmp_path),
        "agent_api": {"base_url": "https://agent.example/v1", "model": "agent"},
        "tested_model": {"adapter": "lingbot_va", "checkpoint": "weights", "gpu": 7},
        "scoring": {"gpu": 0},
    }))
    with pytest.raises(ValueError, match="WOV2_GPU_ALLOWLIST"):
        config.load_profile(path)


def test_import_time_jobs_and_stage_resources_do_not_query_hardware(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("module initialization attempted a hardware query")

    monkeypatch.setattr(config.subprocess, "run", forbidden)
    import wo.agent.jobs as jobs
    import wo.agent.stages as stage_module
    importlib.reload(config)
    importlib.reload(jobs)
    importlib.reload(stage_module)
    assert jobs.GPU_BUDGET_GB <= 16
    assert stage_module.STAGES["episode"].gpu_gb <= 8


def test_official_upstream_server_command_uses_supported_entrypoint_and_checkpoint(tmp_path):
    repo = tmp_path / "lingbot-va"
    output = tmp_path / "attempt"
    command = lingbot_va._official_server_command(
        tmp_path / "env/bin/python", repo, output, tmp_path / "checkpoint", 28003, (1, 2)
    )
    assert command[:3] == [str(tmp_path / "env/bin/python"), "-m", "torch.distributed.run"]
    assert command[command.index("--nproc_per_node") + 1] == "2"
    assert command[command.index("--master_port") + 1] == "29003"
    entry = output / "lingbot_va_server_entry.py"
    assert entry.is_file()
    source = entry.read_text()
    ast.parse(source)
    assert "wan_va_server" in source
    assert "VA_CONFIGS['robotwin']" in source
    assert str(tmp_path / "checkpoint") in source
    assert "_cfg.infer_mode = 'server'" in source


def test_source_bridge_keeps_configured_robotwin_venv_launcher(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    repo = workspace / "runtime/RoboTwin"
    repo.mkdir(parents=True)
    launcher = workspace / "envs/robotwin/bin/python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(sys.executable)
    bridge = workspace / "code/WorldOlympiad-main/wo/models/lingbot_va_native_bridge.py"
    bridge.parent.mkdir(parents=True)
    bridge.write_text("# bridge fixture\\n")
    profile = workspace / "two-interface.json"
    profile.write_text(json.dumps({
        "workspace_root": str(workspace),
        "tested_model": {
            "robotwin_repo": "runtime/RoboTwin",
            "robotwin_python": "envs/robotwin/bin/python",
        },
    }))
    monkeypatch.setenv("WO_EVAL_CONFIG", str(profile))
    # The outer launcher may export the resolved symlink target. The bridge
    # must still execute via the configured venv launcher to retain site-packages.
    monkeypatch.setenv("WO_ROBOTWIN_PYTHON", os.path.realpath(launcher))

    command = lingbot_va.source_bridge_command(["--stock-task", "stack_blocks_two"])
    assert command[0] == str(launcher)
    assert command[0] != os.path.realpath(launcher)
    assert command[1] == str(bridge)


def test_policy_check_registered_timeout_is_at_most_1800_seconds():
    assert stages.STAGES["policy_check"].timeout_s <= 1800
