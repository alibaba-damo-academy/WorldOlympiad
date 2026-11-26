"""Registered LingBot-VA adapter boundary for native RoboTwin Aloha episodes.

The policy uses its upstream EE-action convention (not FastWAM's qpos schema).
The inference process and native RoboTwin process are separate so deployment
profiles can supply their own isolated upstream runtimes without importing
heavy policy dependencies in the Agent API process.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from ..paths import WORKSPACE
from .lingbot_va_client import LingBotVAClient, decode_action, executable_actions

W = WORKSPACE
ROOT = WORKSPACE / "code/WorldOlympiad-main"
SERVER_STARTUP_TIMEOUT_ENV = "WO_LINGBOT_SERVER_STARTUP_TIMEOUT"
MIN_LAYOUT_SCALE = 0.5
TABLE_TOP_RT = 0.741
WORKSPACE_RT = ((-0.55, 0.55), (-0.70, 0.45))


class Unsupported(RuntimeError):
    """The current upstream runtime cannot represent this input safely."""


def _python_launcher(env_name, profile_name, fallback):
    """Resolve an interpreter without discarding a configured venv launcher.

    The user-loop may export the real interpreter behind a venv symlink. If
    that resolves to the same executable as the configured launcher, retain
    the launcher path so Python still discovers that venv's site-packages.
    """
    value = os.environ.get(env_name)
    if value:
        configured = None
        profile_path = os.environ.get("WO_EVAL_CONFIG")
        if profile_path:
            try:
                profile = json.loads(Path(profile_path).read_text())
                configured = (profile.get("tested_model", {}) or {}).get(profile_name)
            except (OSError, ValueError, TypeError, AttributeError):
                configured = None
        if configured:
            configured_path = Path(configured).expanduser()
            if not configured_path.is_absolute():
                configured_path = Path(profile.get("workspace_root", ".")).expanduser() / configured_path
            # Keep the launcher symlink intact: resolving it can bypass the
            # venv's pyvenv.cfg and lose that runtime's installed packages.
            configured_path = Path(os.path.abspath(configured_path))
            if os.path.realpath(value) == os.path.realpath(configured_path):
                return configured_path
        return Path(value)
    return Path(fallback)


def _server_startup_timeout():
    value = float(os.environ.get(SERVER_STARTUP_TIMEOUT_ENV, "3600"))
    if value <= 0:
        raise ValueError(f"{SERVER_STARTUP_TIMEOUT_ENV} must be a positive number")
    return value


def client_env(gpu):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    return env


def _wait_port(port, timeout, process=None):
    import socket
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError("LingBot-VA inference service exited before becoming ready")
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=1):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"LingBot-VA inference service did not open port {port}")


def _configured_command(name, default):
    value = os.environ.get(name)
    return shlex.split(value) if value else list(default)


def _official_server_command(python, repo, out, checkpoint, port, policy_gpus):
    """Build a launcher for the pinned upstream RoboTwin inference server.

    The upstream checkout exposes ``wan_va/wan_va_server.py`` rather than a
    ``lingbot_va.server`` package. Its server also reads the pretrained model
    directory from the Python config (not a ``--checkpoint`` CLI option), so a
    small per-attempt entrypoint binds that configured local path without
    editing the upstream checkout.
    """
    repo = Path(repo).resolve()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    entry = out / "lingbot_va_server_entry.py"
    entry.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"_repo = Path({str(repo)!r})\n"
        f"_checkpoint = {str(checkpoint)!r}\n"
        f"_port = {int(port)!r}\n"
        f"_save_root = {str(out / 'server_output')!r}\n"
        "sys.path.insert(0, str(_repo / 'wan_va'))\n"
        "import wan_va_server\n"
        "from configs import VA_CONFIGS\n"
        "_cfg = VA_CONFIGS['robotwin']\n"
        "_cfg.infer_mode = 'server'\n"
        "_cfg.wan22_pretrained_model_name_or_path = _checkpoint\n"
        "wan_va_server.run(type('Args', (), {'config_name': 'robotwin', "
        "'port': _port, 'save_root': _save_root})())\n",
        encoding="utf-8",
    )
    return [
        str(python), "-m", "torch.distributed.run",
        "--nproc_per_node", str(len(tuple(policy_gpus))),
        "--master_port", str(29000 + int(port) % 1000), str(entry),
    ]


def build_plan(ref, out, *, seed=0, window_s=None, prompt=None):
    """The installed native bridge owns scene decoding and returns its plan."""
    raise Unsupported("scene-plan compilation is provided by the configured RoboTwin bridge")


def run_scene_check(ref, out, *, gpu=0, seed=0):
    """Run the configured native bridge's camera audit without starting policy inference."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec_archive, state_archive = _reference_bridge_inputs(ref, out)
    native_out = out / "native_run"
    command = ["--ref", str(Path(ref).resolve()), "--task-spec", str(spec_archive.resolve()),
               "--reference-state", str(state_archive.resolve()), "--out", str(native_out),
               "--port", "28991", "--seed", str(int(seed)), "--scene-check"]
    rc = run_client(out, command, gpu=gpu)
    if rc:
        raise RuntimeError(f"LingBot native scene check failed with rc={rc}")
    result = json.loads((native_out / "scene_check.json").read_text(encoding="utf-8"))
    if result.get("status") != "ready":
        raise RuntimeError("LingBot native scene check did not verify all task objects")
    return result


def episode_record(out, ref, spec, prompt, seed):
    """Read the bridge's already-bound native episode receipt."""
    path = Path(out) / "episode.json"
    if not path.is_file():
        raise FileNotFoundError(f"native bridge did not write {path}")
    return json.loads(path.read_text())


def not_applicable(ref, out, reason, seed=0, window_s=None):
    """Create an explicit unsupported-input receipt without claiming failure."""
    from ..eval.case import not_applicable as _not_applicable
    return _not_applicable(ref, out, reason, seed, window_s)


def start_server(out, *, policy_gpus=(0,), port=28000, seed=0):
    """Start the configured upstream inference server on its assigned GPU set.

    Deployments may override the command with WO_LINGBOT_SERVER_COMMAND. The
    default launches the pinned upstream RoboTwin server through torch.distributed
    and binds the configured checkpoint path in a temporary entrypoint; this
    module never loads model weights itself.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = os.environ.get("WO_TESTED_MODEL_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError("WO_TESTED_MODEL_CHECKPOINT is not configured")
    python = _python_launcher("WO_LINGBOT_PYTHON", "lingbot_python", sys.executable)
    gpus = tuple(int(gpu) for gpu in policy_gpus)
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 or gpu > 6 for gpu in gpus):
        raise ValueError("policy_gpus must be unique allowed GPU indices")
    configured = os.environ.get("WO_LINGBOT_SERVER_COMMAND")
    if configured:
        command = shlex.split(configured)
        command += ["--checkpoint", checkpoint, "--port", str(int(port)), "--seed", str(int(seed))]
    else:
        repo = os.environ.get("WO_LINGBOT_REPO")
        if not repo:
            raise RuntimeError("WO_LINGBOT_REPO is required for the upstream RoboTwin inference server")
        command = _official_server_command(python, repo, out, checkpoint, port, gpus)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)))
    log_path = out / "logs/policy_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        return subprocess.Popen(command, cwd=os.environ.get("WO_LINGBOT_REPO", None), env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def source_bridge_command(args):
    """Build the source-owned bridge command from pinned profile paths."""
    from .lingbot_va_bridge import runtime_from_profile
    profile_path = os.environ.get("WO_EVAL_CONFIG")
    if not profile_path:
        raise RuntimeError("WO_EVAL_CONFIG is required for the source-owned RoboTwin bridge")
    runtime = runtime_from_profile(profile_path)
    source_root = runtime.workspace / "code" / "WorldOlympiad-main"
    script = source_root / "wo" / "models" / "lingbot_va_native_bridge.py"
    if not script.is_file():
        raise FileNotFoundError(f"source-owned native bridge is missing: {script}")
    python = _python_launcher("WO_ROBOTWIN_PYTHON", "robotwin_python", runtime.python)
    return [str(python), str(script), "--robotwin-repo", str(runtime.repo),
            "--wov2-source", str(source_root), *map(str, args)]


def run_client(out, args, *, gpu):
    """Run the source-owned native bridge (or an explicit deployment override)."""
    policy = args[args.index("--policy") + 1] if "--policy" in args else "lingbot_va"
    configured = os.environ.get("WO_ROBOTWIN_BRIDGE_COMMAND")
    if not configured and policy == "lingbot_va":
        configured = os.environ.get("WO_LINGBOT_ROBOTWIN_COMMAND")
    if configured:
        argv = [*shlex.split(configured), *map(str, args)]
    else:
        argv = source_bridge_command(args)
    python = _python_launcher("WO_ROBOTWIN_PYTHON", "robotwin_python", sys.executable)
    env = client_env(gpu)
    env["PYTHON"] = str(python)
    log_path = Path(out) / "logs/client.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        return subprocess.run(argv, env=env, check=False, stdout=log,
                              stderr=subprocess.STDOUT).returncode


def _reference_bridge_inputs(ref, out):
    """Archive the exact TaskSpec and frame-zero state before crossing runtimes."""
    from ..eval.reference_binding import verify_task
    from ..spec import TaskSpec
    from .reference_state import collect
    ref, out = Path(ref).resolve(), Path(out)
    certificate = verify_task(ref)
    spec_path = Path(certificate["task"]).resolve()
    spec = TaskSpec.load(spec_path)
    spec.validate()
    state = collect(ref)
    archive = out / "lingbot"
    archive.mkdir(parents=True, exist_ok=True)
    spec_archive = archive / "task_spec.json"
    state_archive = archive / "reference_state.json"
    for path, value in ((spec_archive, spec.to_dict()), (state_archive, state)):
        encoded = json.dumps(value, indent=2)
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"immutable LingBot reference archive changed: {path}")
        if not path.exists():
            path.write_text(encoded, encoding="utf-8")
    return spec_archive, state_archive


def run_episode(ref, out, *, gpu=0, simulator_gpu=None, policy_gpus=None,
                port=28000, seed=0, window_s=None, prompt=None):
    """Run one episode through the configured native bridge and bind its receipt."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    policy_gpus = tuple(policy_gpus or (gpu,))
    # Bind and archive the authoritative TaskSpec/frame-zero state before any
    # model process starts. The native bridge must receive these exact objects;
    # a task-name-only fallback would reset RoboTwin's randomized stock actors.
    spec_archive, state_archive = _reference_bridge_inputs(ref, out)
    server = start_server(out, policy_gpus=policy_gpus, port=port, seed=seed)
    try:
        _wait_port(port, _server_startup_timeout(), server)
        command = ["--ref", str(Path(ref).resolve()), "--task-spec", str(spec_archive.resolve()),
                   "--reference-state", str(state_archive.resolve()), "--out", str(out),
                   "--port", str(port), "--policy", "lingbot_va", "--seed", str(seed)]
        if window_s is not None:
            command += ["--window-s", str(window_s)]
        if prompt:
            command += ["--prompt", prompt]
        rc = run_client(out, command, gpu=gpu if simulator_gpu is None else simulator_gpu)
    finally:
        if server.poll() is None:
            import os
            os.killpg(server.pid, 15)
            server.wait(60)
    if rc:
        raise RuntimeError(f"LingBot-VA RoboTwin bridge failed with rc={rc}")
    episode = json.loads((out / "episode.json").read_text())
    if episode.get("model") != "lingbot_va":
        raise ValueError("native bridge episode receipt is not for lingbot_va")
    return episode


def run_stock_task(stock_task, out, *, port=28000, seed=0, window_s=60, prompt=None):
    """Run a stock RoboTwin control for native policy-check qualification."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    primary = int(os.environ.get("WO_TESTED_MODEL_GPU", "0"))
    policy_gpus = tuple(int(g) for g in os.environ.get("WO_TESTED_MODEL_GPUS", str(primary)).split(","))
    simulator = int(os.environ.get("WO_SIMULATOR_GPU", str(primary)))
    server = start_server(out, policy_gpus=policy_gpus, port=port, seed=seed)
    try:
        _wait_port(port, _server_startup_timeout(), server)
        command = ["--stock-task", str(stock_task), "--out", str(out / "native_run"),
                   "--port", str(port), "--policy", "lingbot_va", "--seed", str(seed),
                   "--window-s", str(window_s)]
        if prompt:
            command += ["--prompt", prompt]
        rc = run_client(out, command, gpu=simulator)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, 15)
            server.wait(60)
    if rc:
        raise RuntimeError(f"LingBot-VA stock RoboTwin bridge failed with rc={rc}")
    receipt = out / "native_run/capture/capture_metadata.json"
    if not receipt.is_file():
        raise FileNotFoundError(f"native bridge did not write {receipt}")
    return json.loads(receipt.read_text())


def positive_control(stock_task, out, *, port=28000, seed=0, window_s=60, prompt=None):
    return run_stock_task(stock_task, out, port=port, seed=seed, window_s=window_s, prompt=prompt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("WO_TESTED_MODEL_GPU", "0")))
    parser.add_argument("--simulator-gpu", type=int,
                        default=int(os.environ.get("WO_SIMULATOR_GPU",
                                                   os.environ.get("WO_TESTED_MODEL_GPU", "0"))))
    parser.add_argument("--policy-gpus", type=int, nargs="+")
    parser.add_argument("--port", type=int, default=28000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-s", type=float)
    parser.add_argument("--prompt")
    parser.add_argument("--stock-task")
    args = parser.parse_args()
    if args.stock_task:
        result = run_stock_task(args.stock_task, args.out, port=args.port, seed=args.seed,
                                window_s=60 if args.window_s is None else args.window_s,
                                prompt=args.prompt)
    else:
        if args.ref is None:
            parser.error("--ref is required")
        result = run_episode(args.ref, args.out, gpu=args.gpu, simulator_gpu=args.simulator_gpu,
                             policy_gpus=args.policy_gpus, port=args.port, seed=args.seed,
                             window_s=args.window_s, prompt=args.prompt)
    print(json.dumps(result, indent=1))


__all__ = ["LingBotVAClient", "decode_action", "executable_actions", "main", "run_episode"]

if __name__ == "__main__":
    main()
