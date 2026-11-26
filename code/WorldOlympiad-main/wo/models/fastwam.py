"""FastWAM native RoboTwin adapter. Registration awaits a native control.

Shares scene/predicate/evidence machinery with LingBot, NOT its action protocol.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess

from . import lingbot_va as rt

W = rt.W
PYTHON = W / "envs/fastwam-rt/bin/python"
SOURCE = W / "code/baselines/FastWAM-7faa711"
WEIGHTS = W / "models/fastwam/8eaceeb24c3cc92ff2a9c9a9d266a4941b836705"
CHECKPOINT = WEIGHTS / "robotwin_uncond_3cam_384.pt"
STATS = WEIGHTS / "robotwin_uncond_3cam_384_dataset_stats.json"


def server_command(port: int, seed: int) -> list[str]:
    return [str(PYTHON), "-u", "-m", "wo.models.fastwam_server",
            "--source", str(SOURCE), "--checkpoint", str(CHECKPOINT),
            "--dataset-stats", str(STATS), "--port", str(port), "--seed", str(seed)]


def start_server(out: Path, *, gpu: int, port: int, seed: int):
    for path in (PYTHON, CHECKPOINT, STATS):
        if not path.is_file():
            raise FileNotFoundError(path)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    env = dict(rt.client_env(gpu), PYTHONPATH=f"{rt.ROOT}:{SOURCE}",
               DIFFSYNTH_MODEL_BASE_PATH=str(W / "models/fastwam-assets"))
    with (out / "logs/policy_server.log").open("w") as log:
        return subprocess.Popen(server_command(port, seed), cwd=SOURCE, env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def execute(out: Path, args: list[str], *, gpu: int, port: int, seed: int,
            simulator_gpu: int | None = None, server_factory=None,
            policy: str = "fastwam") -> dict:
    if simulator_gpu is None:
        raise ValueError("native WAM simulator GPU must be selected by resource preflight")
    out = Path(out)
    native_out = Path(args[args.index("--out") + 1]) if "--out" in args else out / "native_run"
    if native_out not in {out, out / "native_run"}:
        raise ValueError("native output must be the episode directory or its native_run child")
    if (native_out / "capture/capture_metadata.json").exists():
        raise FileExistsError("native capture already exists; use a fresh episode attempt")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    server = (server_factory or start_server)(out, gpu=gpu, port=port, seed=seed)
    try:
        rt._wait_port(port, 1200, server)
        command = [*args, "--policy", policy, "--port", str(port)]
        if "--out" not in args:
            command += ["--out", str(native_out)]
        rc = rt.run_client(out, command, gpu=simulator_gpu)
    finally:
        # Owned process group only; a failure here must not turn an infra error
        # into a task failure or not_applicable record.
        if server.poll() is None:
            os.killpg(server.pid, 15)
            server.wait(60)
    na = native_out / "not_applicable.json"
    if na.is_file():
        raise rt.Unsupported(json.loads(na.read_text())["reason"])
    if rc:
        log = out / "logs/client.log"
        tail = log.read_text(errors="replace").splitlines()[-20:] if log.is_file() else []
        raise RuntimeError(f"{policy} RoboTwin client failed (rc={rc}):\n" + "\n".join(tail))
    return json.loads((native_out / "capture/capture_metadata.json").read_text())


def run_episode(ref: Path, out: Path, *, gpu=0, simulator_gpu=None, port=29058,
                seed=0, window_s=None, prompt=None):
    out.mkdir(parents=True, exist_ok=True)
    task_spec, state = rt._reference_bridge_inputs(ref, out)
    args = ["--ref", str(Path(ref).resolve()), "--task-spec", str(task_spec),
            "--reference-state", str(state), "--out", str(out), "--seed", str(seed)]
    if window_s is not None:
        args += ["--window-s", str(window_s)]
    if prompt:
        args += ["--prompt", prompt]
    execute(out, args, gpu=gpu, simulator_gpu=simulator_gpu, port=port, seed=seed)
    result = rt.episode_record(out, ref, None, prompt, seed)
    if result.get("model") != "fastwam":
        raise ValueError("native bridge episode receipt is not for fastwam")
    result["action_type"] = "qpos"
    (out / "episode.json").write_text(json.dumps(result, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=int(os.environ.get("WO_TESTED_MODEL_GPU", "0")))
    ap.add_argument("--simulator-gpu", type=int,
                    default=(int(os.environ["WO_SIMULATOR_GPU"])
                             if "WO_SIMULATOR_GPU" in os.environ else None))
    ap.add_argument("--port", type=int, default=29058)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window-s", type=float, default=None,
                     help="optional simulated-time cutoff; default runs until native success or the original RoboTwin action limit")
    ap.add_argument("--prompt")
    ap.add_argument("--stock-task")
    a = ap.parse_args()
    if a.stock_task:
        result = execute(a.out, ["--stock-task", a.stock_task, "--seed", str(a.seed),
                                *( ["--window-s", str(a.window_s)] if a.window_s is not None else []),
                                *( ["--prompt", a.prompt] if a.prompt else [])],
                         gpu=a.gpu, simulator_gpu=a.simulator_gpu, port=a.port, seed=a.seed)
    else:
        if a.ref is None:
            ap.error("--ref is required unless --stock-task is supplied")
        try:
            result = run_episode(a.ref, a.out, gpu=a.gpu, simulator_gpu=a.simulator_gpu, port=a.port,
                                 seed=a.seed, window_s=a.window_s, prompt=a.prompt)
        except rt.Unsupported as exc:
            result = rt.not_applicable(a.ref, a.out, str(exc), a.seed, a.window_s)
            result["model"] = "fastwam"
            (a.out / "episode.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
