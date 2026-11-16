"""Single-command launcher for an isolated two-interface evaluation run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def _bootstrap(config_path: Path) -> tuple[Path, dict]:
    profile = json.loads(config_path.read_text(encoding="utf-8"))
    workspace = Path(profile["workspace_root"]).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    os.environ["WO_WORKSPACE"] = str(workspace)
    os.environ["WO_EVAL_CONFIG"] = str(config_path.resolve())
    os.environ["PYTHONPATH"] = str(repo) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if not (workspace / "code/WorldOlympiad-main").resolve() == repo.resolve():
        raise ValueError(
            "candidate must be installed at workspace_root/code/WorldOlympiad-main; "
            "refusing to run against a different source tree"
        )
    return workspace, profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--goal", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--min-gap-s", type=int, default=600)
    args = parser.parse_args()

    workspace, _raw_profile = _bootstrap(args.config.expanduser().resolve())
    from .config import (
        apply_environment,
        load_private_environment,
        load_profile,
        preflight,
        scenecode_runtime_status,
    )

    load_private_environment(os.environ)
    profile = load_profile(args.config, required=True)
    apply_environment(os.environ, profile)
    check = preflight(profile)
    if not check["agent_api"]["api_key_present"]:
        raise SystemExit("Agent API key is not present in the configured environment variable.")
    if not check["tested_model"]["checkpoint_exists"]:
        raise SystemExit("Configured tested-model checkpoint does not exist.")
    if profile["tested_model"]["adapter"] == "lingbot_va":
        required_paths = {
            "RoboTwin repository": os.environ.get("WO_ROBOTWIN_REPO", ""),
            "LingBot-VA repository": os.environ.get("WO_LINGBOT_REPO", ""),
            "RoboTwin Python": os.environ.get("WO_ROBOTWIN_PYTHON", ""),
            "LingBot-VA Python": os.environ.get("WO_LINGBOT_PYTHON", ""),
            "SceneCode source": os.environ.get("SCENECODE_SOURCE_ROOT", ""),
        }
        missing = [f"{label}: {path}" for label, path in required_paths.items()
                   if not path or not Path(path).exists()]
        scene_source = required_paths["SceneCode source"]
        if scene_source and not (Path(scene_source) / "main.py").is_file():
            missing.append(f"SceneCode source main.py: {Path(scene_source) / 'main.py'}")
        runtime_root = os.environ.get("SCENECODE_RUNTIME_ROOT", "")
        if not runtime_root:
            missing.append("SceneCode runtime path is not configured")
        elif scenecode_runtime_status(runtime_root) == "unrecognized":
            missing.append(f"SceneCode runtime has no WorldOlympiad marker: {runtime_root}")
        requirements = Path(required_paths["RoboTwin repository"]) / "scripts/requirements.txt"
        if not requirements.exists():
            missing.append(f"RoboTwin scripts/requirements.txt: {requirements}")
        if missing:
            raise SystemExit("Configured LingBot-VA runtime is incomplete:\n" + "\n".join(missing))
    worker_python = workspace / "envs/autoresearch/bin/python"
    if not worker_python.is_file():
        raise SystemExit(f"registered-stage Python environment not found: {worker_python}")
    run = args.run
    logroot = workspace / "outputs/wo_agent/launcher"
    logroot.mkdir(parents=True, exist_ok=True)
    worker_log = logroot / f"{run}_{time.time_ns()}_worker.log"
    with worker_log.open("w") as log:
        worker = subprocess.Popen(
            [str(worker_python), "-m", "wo.agent.jobs", "worker"],
            cwd=Path(__file__).resolve().parents[2], env=os.environ.copy(),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            command = [sys.executable, "-m", "wo.agent.run", "--run", run,
                       "--goal", str(args.goal.resolve()), "--rounds", str(args.rounds),
                       "--min-gap-s", str(args.min_gap_s), "--no-engineering"]
            result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                    env=os.environ.copy(), check=False)
            if result.returncode:
                raise SystemExit(result.returncode)
        finally:
            worker.send_signal(signal.SIGTERM)
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()


if __name__ == "__main__":
    main()
