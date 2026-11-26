from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _gpu_allowlist(monkeypatch):
    monkeypatch.setenv("WOV2_GPU_ALLOWLIST", "0,1,2,3,4,5,6")

from wo.agent import config, jobs


def _inventory(*, free=(30_000, 28_000, 0, 0, 0, 0, 0), occupied=()):
    rows = []
    occupied = set(occupied)
    for index in range(7):
        rows.append({
            "index": index,
            "total_mib": 32_607,
            "free_mib": free[index],
            "utilization": 0 if index not in occupied else 100,
            "compute_process_count": 0 if index not in occupied else 1,
            "compute_pids": [] if index not in occupied else [90_000 + index],
            "process_query_ok": True,
        })
    # A tempting high-memory device outside the configured list must be ignored.
    rows.append({
        "index": 7, "total_mib": 80_000, "free_mib": 80_000,
        "utilization": 0, "compute_process_count": 0,
        "compute_pids": [], "process_query_ok": True,
    })
    return rows


def _profile(tmp_path: Path, *, scoring=None):
    return {
        "_config_path": str(tmp_path / "config.json"),
        "workspace_root": str(tmp_path),
        "agent_api": {
            "base_url": "https://agent.example/v1",
            "model": "test-model",
            "api_key_env": "WO_AGENT_API_KEY",
        },
        "tested_model": {"adapter": "lingbot_va", "checkpoint": "weights/test"},
        "scoring": {"mode": "auto", "gpu": "auto", "compute_gpu": "auto",
                    **(scoring or {})},
    }


def test_auto_split_scoring_selects_two_distinct_idle_allowlisted_gpus(tmp_path, monkeypatch):
    inventory = _inventory()
    monkeypatch.setattr(config, "all_gpu_info", lambda: inventory)

    selected = config.resolve_score_gpus(_profile(tmp_path))

    assert selected == {
        "mode": "split", "gpu": 0, "compute_gpu": 1,
        "required_free_mib_per_gpu": config.SPLIT_SCORE_MIN_FREE_MIB,
    }


def test_insufficient_score_topology_is_blocked_before_queue_insert(tmp_path, monkeypatch):
    inventory = _inventory(free=(30_000, 12_000, 0, 0, 0, 0, 0), occupied=(1, 2, 3, 4, 5, 6))
    profile = _profile(tmp_path)
    monkeypatch.setattr(config, "all_gpu_info", lambda: inventory)
    monkeypatch.setattr(config, "load_profile", lambda: profile)
    monkeypatch.setattr(jobs, "job_defaults",
                        lambda kind, args: config.job_defaults(kind, args, profile=profile))
    monkeypatch.setattr(jobs, "connect",
                        lambda: pytest.fail("insufficient-resource preflight must stop before SQLite insert"))

    with pytest.raises(RuntimeError, match=r"needs 2 distinct idle allowlisted NVIDIA GPUs.*found 1/2"):
        jobs.submit("score", {"run": "test-run", "case": "test-case", "model": "lingbot_va"})


def test_score_preflight_reports_only_allowed_inventory(tmp_path):
    inventory = _inventory()
    selected = config.preflight(_profile(tmp_path), env={}, phase="scoring", inventory=inventory)

    assert [card["index"] for card in selected["gpu_inventory"]] == list(range(7))
    assert selected["resources"]["scoring"]["distinct_gpus"] is True


def test_exact_score_gpus_are_rechecked_when_stage_environment_is_built(monkeypatch, tmp_path):
    profile = _profile(tmp_path, scoring={"gpu": 0, "compute_gpu": 1, "mode": "split"})
    busy = _inventory(free=(30_000, 20_000, 30_000, 0, 0, 0, 0), occupied=(1, 3, 4, 5, 6))
    monkeypatch.setattr(config, "all_gpu_info", lambda: busy)

    with pytest.raises(RuntimeError, match=r"compute GPU 1 is not eligible"):
        config.apply_environment({}, profile, phase="scoring")



def test_resource_query_uses_only_custom_allowlisted_device_indices(monkeypatch):
    monkeypatch.setenv("WOV2_GPU_ALLOWLIST", "2,8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "--query-gpu=" in command[3]:
            output = (
                "2, GPU-2, 81920, 31000, 0\n"
                "8, GPU-8, 46080, 29000, 1\n"
            )
        else:
            output = ""
        return type("Result", (), {"stdout": output})()

    monkeypatch.setattr(config.subprocess, "run", fake_run)
    inventory = config.all_gpu_info()

    assert [card["index"] for card in inventory] == [2, 8]
    assert len(calls) == 2
    assert all(command[:3] == ["nvidia-smi", "-i", "2,8"] for command in calls)


def test_inventory_filters_out_indices_not_named_by_user(monkeypatch):
    monkeypatch.setenv("WOV2_GPU_ALLOWLIST", "3,9")
    inventory = [{"index": index} for index in (0, 3, 7, 9)]
    assert config._allowed_inventory(inventory) == [{"index": 3}, {"index": 9}]

def test_split_score_runner_masks_service_and_compute_to_separate_cards(tmp_path, monkeypatch):
    from scripts import score_robotics_cases as runner

    inventory = _inventory()
    monkeypatch.setattr(config, "all_gpu_info", lambda: inventory)
    starts = []
    phases = []
    commits = []
    class Handle:
        pass
    monkeypatch.setattr(
        runner, "_start_qwenvl",
        lambda repo, workspace, weights, python, gpu, env, log: (
            starts.append((gpu, env["CUDA_VISIBLE_DEVICES"])) or Handle(), Handle()
        ),
    )
    monkeypatch.setattr(runner, "_stop_qwenvl", lambda service, log: None)
    monkeypatch.setattr(
        runner, "_run_phase",
        lambda repo, workspace, weights, python, root, data, attempt, phase, env, **kwargs:
            phases.append((phase, env["CUDA_VISIBLE_DEVICES"], env["WOV2_SCORE_COMPUTE_GPU"])) or {},
    )
    monkeypatch.setattr(runner, "_merge_split", lambda parts, attempt: {"score": "fixture"})
    monkeypatch.setattr(
        runner, "_commit_score",
        lambda root, summary, hashes, case_id: commits.append((root, summary, case_id)),
    )
    case_root = tmp_path / "case"
    case_root.mkdir()
    case = {"case_id": "fixture"}
    dummy = case_root / "dummy"
    dummy.write_text("x")
    data = (case, dummy, dummy, dummy, dummy, {"dummy": "sha"})
    monkeypatch.setattr(runner, "inputs", lambda root: data)

    runner.run([case_root], tmp_path, "0", "split", "1")

    assert starts == [("0", "0")]
    assert phases == [("physical", "1", "1"), ("interaction", "1", "1"), ("geometry", "1", "1")]
    assert commits == [(case_root, {"score": "fixture"}, "fixture")]


def test_split_runner_refuses_busy_devices_before_vlm_start(tmp_path, monkeypatch):
    from scripts import score_robotics_cases as runner

    inventory = _inventory(free=(30_000, 20_000, 0, 0, 0, 0, 0), occupied=(1, 2, 3, 4, 5, 6))
    monkeypatch.setattr(config, "all_gpu_info", lambda: inventory)
    monkeypatch.setattr(runner, "_start_qwenvl",
                        lambda *args, **kwargs: pytest.fail("VLM must not start after failed preflight"))
    case_root = tmp_path / "case"
    case_root.mkdir()
    dummy = case_root / "dummy"
    dummy.write_text("x")
    data = ({"case_id": "fixture"}, dummy, dummy, dummy, dummy, {"dummy": "sha"})
    monkeypatch.setattr(runner, "inputs", lambda root: data)

    with pytest.raises(RuntimeError, match=r"fresh scoring preflight blocked GPU"):
        runner.run([case_root], tmp_path, "0", "split", "1")


def test_config_allowlist_requires_ascii_unique_nonnegative_indices(monkeypatch):
    monkeypatch.setenv("WOV2_GPU_ALLOWLIST", "8, 2")
    assert config.configured_gpu_indices() == (2, 8)
    monkeypatch.setenv("WOV2_GPU_ALLOWLIST", "2,²")
    with pytest.raises(ValueError, match="nonnegative integer indices"):
        config.configured_gpu_indices()
    monkeypatch.setenv("WOV2_GPU_ALLOWLIST", "2,02")
    with pytest.raises(ValueError, match="duplicate indices"):
        config.configured_gpu_indices()


def test_explicit_profile_gpu_requires_a_configured_allowlist():
    assert not config._valid_profile_gpu(2, set())
    assert config._valid_profile_gpu(2, {2, 8})
    assert not config._valid_profile_gpu(7, {2, 8})


def test_score_runner_uses_user_allowlist_instead_of_lab_indices(tmp_path, monkeypatch):
    from scripts import score_robotics_cases as runner

    monkeypatch.setenv("WOV2_GPU_ALLOWLIST", "2,8")
    runner.run([], tmp_path, "2", "split", "8")
    with pytest.raises(ValueError, match="WOV2_GPU_ALLOWLIST"):
        runner.run([], tmp_path, "1", "split", "8")
    monkeypatch.delenv("WOV2_GPU_ALLOWLIST")
    with pytest.raises(ValueError, match="WOV2_GPU_ALLOWLIST"):
        runner.run([], tmp_path, "2", "split", "8")
