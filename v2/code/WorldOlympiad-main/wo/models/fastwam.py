"""FastWAM native RoboTwin adapter. Registration awaits a native control.

Shares scene/predicate/evidence machinery with LingBot, NOT its action protocol.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
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
            server_factory=None, policy: str = "fastwam") -> dict:
    shutil.rmtree(out / "native_run", ignore_errors=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    server = (server_factory or start_server)(out, gpu=gpu, port=port, seed=seed)
    try:
        rt._wait_port(port, 1200, server)
        rc = rt.run_client(out, [*args, "--policy", policy, "--port", str(port),
                                 "--out", str(out / "native_run")], gpu=gpu)
    finally:
        # Owned process group only; a failure here must not turn an infra error
        # into a task failure or not_applicable record.
        if server.poll() is None:
            os.killpg(server.pid, 15)
            server.wait(60)
    na = out / "native_run/not_applicable.json"
    if na.is_file():
        raise rt.Unsupported(json.loads(na.read_text())["reason"])
    if rc:
        tail = (out / "logs/client.log").read_text(errors="replace").splitlines()[-20:]
        raise RuntimeError(f"{policy} RoboTwin client failed (rc={rc}):\n" + "\n".join(tail))
    return json.loads((out / "native_run/capture/capture_metadata.json").read_text())


def run_episode(ref: Path, out: Path, *, gpu=0, port=29058, seed=0, window_s=None, prompt=None):
    out.mkdir(parents=True, exist_ok=True)
    plan, spec = rt.build_plan(ref, out, seed=seed, window_s=window_s, prompt=prompt)
    execute(out, ["--plan", str(out / "lingbot/scene_plan.json")],
            gpu=gpu, port=port, seed=seed)
    result = rt.episode_record(out, ref, spec, plan["prompt"], seed)
    result["model"] = "fastwam"
    result["action_type"] = "qpos"
    (out / "episode.json").write_text(json.dumps(result, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
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
                         gpu=a.gpu, port=a.port, seed=a.seed)
    else:
        if a.ref is None:
            ap.error("--ref is required unless --stock-task is supplied")
        try:
            result = run_episode(a.ref, a.out, gpu=a.gpu, port=a.port,
                                 seed=a.seed, window_s=a.window_s, prompt=a.prompt)
        except rt.Unsupported as exc:
            result = rt.not_applicable(a.ref, a.out, str(exc), a.seed, a.window_s)
            result["model"] = "fastwam"
            (a.out / "episode.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
