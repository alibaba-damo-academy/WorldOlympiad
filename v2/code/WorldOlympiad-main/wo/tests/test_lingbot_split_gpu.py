import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from wo.agent import config
from wo.models import lingbot_va as L


def _profile(root, *, simulator_gpu="auto"):
    path = root / "two-interface.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "workspace_root": str(root),
        "agent_api": {"base_url": "https://agent.example/v1", "model": "agent",
                      "api_key_env": "WO_AGENT_API_KEY"},
        "tested_model": {"adapter": "lingbot_va", "checkpoint": "weights/model",
                         "gpu": "auto", "simulator_gpu": simulator_gpu,
                         "robotwin_repo": "runtime/RoboTwin", "lingbot_repo": "runtime/lingbot-va",
                         "robotwin_python": "envs/robotwin/bin/python",
                         "lingbot_python": "envs/lingbot/bin/python",
                         "scenecode_source": "runtime/SceneCode",
                         "scenecode_runtime": "tmp/scenecode_runtime"},
        "scoring": {"mode": "auto", "gpu": "auto"},
    }))
    return config.load_profile(path, required=True)


def test_small_gpu_selection_splits_policy_and_simulator_and_large_stays_single(tmp_path):
    profile = _profile(tmp_path)
    inventory = [
        {"index": 4, "total_mib": 32607, "free_mib": 32000, "utilization": 0},
        {"index": 5, "total_mib": 32607, "free_mib": 32000, "utilization": 1},
        {"index": 6, "total_mib": 32607, "free_mib": 30000, "utilization": 50},
    ]
    assert config.resolve_gpu(profile, inventory=inventory) == 4
    assert config.resolve_simulator_gpu(profile, 4, inventory=inventory) == 5
    assert config.resolve_simulator_gpu(profile, 4, inventory=inventory[:1]) is None
    large = [{"index": 0, "total_mib": 85651, "free_mib": 80000, "utilization": 0}]
    assert config.resolve_simulator_gpu(profile, 0, inventory=large) == 0


def test_job_defaults_and_episode_command_pass_distinct_gpus(tmp_path):
    profile = _profile(tmp_path)
    inventory = [
        {"index": 4, "total_mib": 32607, "free_mib": 32000, "utilization": 0},
        {"index": 5, "total_mib": 32607, "free_mib": 32000, "utilization": 0},
        {"index": 6, "total_mib": 32607, "free_mib": 32000, "utilization": 0},
    ]
    with patch.object(config, "all_gpu_info", return_value=inventory):
        args = config.job_defaults("episode", {"run": "run0", "case": "case0"}, profile)
        from wo.agent import stages
        command = stages.STAGES["episode"].command(args | {"model": "lingbot_va"}, 7)
    assert args["gpu"] == 4
    assert args["simulator_gpu"] == 5
    assert args["policy_gpus"] == [4, 6]
    assert command[command.index("--gpu") + 1] == "4"
    assert command[command.index("--simulator-gpu") + 1] == "5"
    assert command[command.index("--policy-gpus") + 1:command.index("--policy-gpus") + 3] == ["4", "6"]


def test_lingbot_starts_policy_and_renderer_on_separate_gpus(tmp_path, monkeypatch):
    ref, out = tmp_path / "ref", tmp_path / "out"
    ref.mkdir()
    started, client = {}, {}
    monkeypatch.setattr(L, "build_plan", lambda *a, **k: ({"prompt": "move block"}, object()))
    monkeypatch.setattr(L, "start_server", lambda *a, **k:
                        (started.update(k) or SimpleNamespace(pid=1, poll=lambda: 0, wait=lambda *_: None)))
    monkeypatch.setattr(L, "_wait_port", lambda *a, **k: None)

    def run_native_client(output, command, *, gpu):
        client.update(command=command, gpu=gpu)
        (Path(output) / "episode.json").write_text(json.dumps({"model": "lingbot_va"}))
        return 0

    monkeypatch.setattr(L, "run_client", run_native_client)
    monkeypatch.setattr(L.os, "killpg", lambda *_: None)
    assert L.run_episode(ref, out, gpu=4, simulator_gpu=5, policy_gpus=(4, 6)) == {"model": "lingbot_va"}
    assert started["policy_gpus"] == (4, 6)
    assert client["gpu"] == 5



def test_policy_server_launches_with_all_configured_policy_gpus(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    captured = {}
    monkeypatch.setenv("WO_TESTED_MODEL_CHECKPOINT", str(tmp_path / "checkpoint"))
    monkeypatch.setenv("WO_LINGBOT_PYTHON", str(tmp_path / "lingbot-python"))
    monkeypatch.delenv("WO_LINGBOT_SERVER_COMMAND", raising=False)
    monkeypatch.setattr(L.subprocess, "Popen", lambda command, **kwargs: captured.update(
        command=command, kwargs=kwargs) or SimpleNamespace(pid=1))

    L.start_server(out, policy_gpus=(4, 5), port=28001, seed=7)

    command = captured["command"]
    assert command[:3] == [str(tmp_path / "lingbot-python"), "-m", "lingbot_va.server"]
    assert command[command.index("--checkpoint") + 1] == str(tmp_path / "checkpoint")
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "4,5"

def test_preflight_reports_robotwin_files_and_second_gpu(tmp_path):
    profile = _profile(tmp_path)
    for relative in ["weights/model", "runtime/RoboTwin/assets/embodiments/aloha-agilex",
                     "runtime/RoboTwin/assets/objects/objaverse", "runtime/lingbot-va",
                     "envs/robotwin/bin", "envs/lingbot/bin", "code/SceneCode-main",
                     "tmp/scenecode_runtime"]:
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    (tmp_path / "runtime/RoboTwin/assets/objects/objaverse/list.json").write_text("[]")
    (tmp_path / "code/SceneCode-main/main.py").touch()
    (tmp_path / "tmp/scenecode_runtime/.worldolympiad_scenecode_runtime_v1").touch()
    inventory = [{"index": 0, "total_mib": 85651, "free_mib": 80000, "utilization": 0}]
    with patch.object(config, "all_gpu_info", return_value=inventory):
        check = config.preflight(profile, {"WO_AGENT_API_KEY": "test-secret"})
    assert check["tested_model"]["simulator_gpu_ready"]
    assert not check["tested_model"]["robotwin_runtime_ready"]
    assert not check["tested_model"]["robotwin_required_files"][
        "assets/embodiments/aloha-agilex/config.yml"]


def test_job_defaults_refuses_episode_without_second_idle_small_gpu(tmp_path):
    profile = _profile(tmp_path)
    only_one = [{"index": 5, "total_mib": 32607, "free_mib": 32000, "utilization": 0}]
    with patch.object(config, "all_gpu_info", return_value=only_one):
        try:
            config.job_defaults("episode", {"run": "run0", "case": "case0"}, profile)
        except RuntimeError as exc:
            assert "separate idle GPUs" in str(exc)
        else:
            raise AssertionError("episode queued without a second idle GPU")


def _run_positive_control_with_env(tmp_path, monkeypatch, *, primary, policy, simulator):
    out = tmp_path / f"positive-{primary}-{simulator}"
    monkeypatch.setenv("WO_TESTED_MODEL_GPU", str(primary))
    monkeypatch.setenv("WO_TESTED_MODEL_GPUS", ",".join(map(str, policy)))
    monkeypatch.setenv("WO_SIMULATOR_GPU", str(simulator))
    started, client = {}, {}

    class Server:
        pid = 123
        def poll(self):
            return None

        def wait(self, _timeout):
            return 0

    def start_server(_out, **kwargs):
        started.update(kwargs)
        return Server()

    def run_client(_out, _args, *, gpu):
        client["gpu"] = gpu
        capture = out / "native_run/capture"
        capture.mkdir(parents=True)
        (capture / "capture_metadata.json").write_text(json.dumps({"task_success": True}))
        return 0

    monkeypatch.setattr(L, "start_server", start_server)
    monkeypatch.setattr(L, "_wait_port", lambda *_args: None)
    monkeypatch.setattr(L, "run_client", run_client)
    monkeypatch.setattr(L.os, "killpg", lambda *_args: None)
    L.positive_control("stack_blocks_two", out, port=28001)
    return started, client


def test_positive_control_uses_allocated_split_gpus(tmp_path, monkeypatch):
    started, client = _run_positive_control_with_env(
        tmp_path, monkeypatch, primary=1, policy=(1, 4), simulator=3)
    assert started["policy_gpus"] == (1, 4)
    assert client["gpu"] == 3


def test_positive_control_allows_one_large_gpu_for_policy_and_simulator(tmp_path, monkeypatch):
    started, client = _run_positive_control_with_env(
        tmp_path, monkeypatch, primary=0, policy=(0,), simulator=0)
    assert started["policy_gpus"] == (0,)
    assert client["gpu"] == 0
