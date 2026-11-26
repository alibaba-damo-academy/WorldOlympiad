#!/usr/bin/env python3
"""Sequential one-prompt-per-process runner for subset_sub.csv."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

DEFAULT_CSV_PATH = Path(
    "examples/prompts.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs"
)
COMPLETION_MARKERS = (
    "ALL SCENES COMPLETED!",
    "Experiment execution completed in",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "experiment" / "base_experiment.yaml"
)
SERVER_PORT_KEYS = (
    "geometry_generation_server",
    "hssd_retrieval_server",
    "articulated_retrieval_server",
    "materials_retrieval_server",
    "objaverse_retrieval_server",
)
FALLBACK_SERVER_PORTS = {
    "geometry_generation_server": 7005,
    "hssd_retrieval_server": 7006,
    "articulated_retrieval_server": 7007,
    "materials_retrieval_server": 7008,
    "objaverse_retrieval_server": 7009,
}
PIPELINE_STAGE_CHOICES = (
    "floor_plan",
    "furniture",
    "wall_mounted",
    "ceiling_mounted",
    "manipuland",
)


@dataclass(frozen=True)
class PromptJob:
    """A single prompt-to-scene generation job."""

    scene_id: int
    prompt: str
    output_dir: Path


def format_scene_dir(scene_id: int, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Build the final output directory for a scene id."""
    return output_root / f"scene_{scene_id:03d}"


def load_prompt_jobs(
    csv_path: Path = DEFAULT_CSV_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> list[PromptJob]:
    """Load prompt jobs from subset_sub.csv."""
    jobs: list[PromptJob] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            try:
                scene_id = int(row["ID"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"CSV row {row_num} has invalid ID: {row}") from exc

            prompt = row.get("Description")
            if prompt is None or not prompt.strip():
                raise ValueError(f"CSV row {row_num} is missing Description: {row}")

            jobs.append(
                PromptJob(
                    scene_id=scene_id,
                    prompt=prompt,
                    output_dir=format_scene_dir(scene_id, output_root),
                )
            )
    return jobs


def should_skip_scene_dir(scene_dir: Path) -> bool:
    """Return True when the scene directory already contains outputs."""
    return scene_dir.exists() and any(scene_dir.iterdir())


def build_child_command(
    job: PromptJob,
    run_name: str,
    materials_retrieval_server_port: int | None = None,
    start_stage: str | None = None,
    stop_stage: str | None = None,
) -> list[str]:
    """Build the Hydra command for a single child run."""
    command = [
        sys.executable,
        "main.py",
        f"+name={run_name}",
        f"hydra.run.dir={job.output_dir}",
        "experiment.num_workers=1",
        "experiment.csv_path=null",
        f"experiment.single_scene_id={job.scene_id}",
        f"experiment.single_prompt={json.dumps(job.prompt)}",
        f"experiment.fixed_scene_output_dir={job.output_dir}",
    ]
    if materials_retrieval_server_port is not None:
        command.append(
            "experiment.materials_retrieval_server.port="
            f"{materials_retrieval_server_port}"
        )
    if start_stage is not None:
        command.append(f"experiment.pipeline.start_stage={start_stage}")
    if stop_stage is not None:
        command.append(f"experiment.pipeline.stop_stage={stop_stage}")
    return command


def load_default_server_ports(
    config_path: Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
) -> dict[str, int]:
    """Load default server ports from the experiment config file."""
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_path)
        ports: dict[str, int] = {}
        for key in SERVER_PORT_KEYS:
            server_cfg = cfg.get(key)
            if server_cfg is None or server_cfg.get("port") is None:
                raise ValueError(f"Missing port configuration for {key}")
            ports[key] = int(server_cfg["port"])
        return ports
    except Exception:
        return FALLBACK_SERVER_PORTS.copy()


def resolve_cleanup_ports(
    command: list[str],
    default_ports: dict[str, int] | None = None,
) -> tuple[int, ...]:
    """Resolve the exact ports used by this child command."""
    port_map = dict(default_ports or load_default_server_ports())

    for arg in command:
        if not arg.startswith("experiment.") or "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        if not key.endswith(".port"):
            continue

        server_key = key.removeprefix("experiment.").removesuffix(".port")
        if server_key not in port_map:
            continue

        try:
            port_map[server_key] = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid port override '{arg}'") from exc

    return tuple(port_map[key] for key in SERVER_PORT_KEYS if key in port_map)


def cleanup_listening_ports(ports: tuple[int, ...]) -> list[int]:
    """Kill processes listening on the provided TCP ports."""
    target_inodes: set[str] = set()
    target_ports = set(ports)

    for proc_net_path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(proc_net_path) as f:
                next(f)
                for line in f:
                    parts = line.split()
                    local = parts[1]
                    state = parts[3]
                    inode = parts[9]
                    _, port_hex = local.split(":")
                    port = int(port_hex, 16)
                    if port in target_ports and state == "0A":
                        target_inodes.add(inode)
        except FileNotFoundError:
            continue

    killed: list[int] = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        fd_dir = Path("/proc") / pid / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    link = os.readlink(fd)
                except OSError:
                    continue
                if link.startswith("socket:[") and link[8:-1] in target_inodes:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        killed.append(int(pid))
                    except (ProcessLookupError, PermissionError):
                        pass
                    break
        except OSError:
            continue

    return sorted(set(killed))


def _reader_loop(stream, line_queue: Queue[str], log_file) -> None:
    for line in iter(stream.readline, ""):
        print(line, end="")
        log_file.write(line)
        log_file.flush()
        line_queue.put(line)
    stream.close()


def _terminate_process_group(process_group_id: int) -> None:
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return


def _kill_process_group(process_group_id: int) -> None:
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return


def run_job(
    job: PromptJob,
    run_name: str,
    cuda_visible_devices: str,
    grace_seconds: float,
    overwrite: bool,
    materials_retrieval_server_port: int | None = None,
    start_stage: str | None = None,
    stop_stage: str | None = None,
) -> bool:
    """Run one prompt job in a fresh child process."""
    is_resume = start_stage is not None and start_stage != "floor_plan"
    if not is_resume:
        if overwrite and job.output_dir.exists():
            shutil.rmtree(job.output_dir)
        elif should_skip_scene_dir(job.output_dir):
            print(f"[skip] scene_{job.scene_id:03d} already exists at {job.output_dir}")
            return True

    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.output_dir / "sequential_runner.log"
    command = build_child_command(
        job,
        run_name,
        materials_retrieval_server_port=materials_retrieval_server_port,
        start_stage=start_stage,
        stop_stage=stop_stage,
    )
    cleanup_ports = resolve_cleanup_ports(command)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    with open(log_path, "a", buffering=1) as log_file:
        log_file.write(f"[command] {' '.join(command)}\n")
        log_file.write(f"[cleanup_ports] {cleanup_ports}\n")
        log_file.write(f"[cwd] {REPO_ROOT}\n")
        line_queue: Queue[str] = Queue()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        assert process.stdout is not None
        process_group_id = os.getpgid(process.pid)

        reader = threading.Thread(
            target=_reader_loop,
            args=(process.stdout, line_queue, log_file),
            daemon=True,
        )
        reader.start()

        completion_seen = False
        completion_deadline: float | None = None
        forced_cleanup = False

        while True:
            if process.poll() is not None and not reader.is_alive() and line_queue.empty():
                break

            try:
                line = line_queue.get(timeout=0.2)
                if (not completion_seen) and any(
                    marker in line for marker in COMPLETION_MARKERS
                ):
                    completion_seen = True
                    completion_deadline = time.monotonic() + grace_seconds
                    log_file.write(
                        f"[info] completion marker seen; waiting {grace_seconds}s for exit\n"
                    )
            except Empty:
                pass

            if (
                completion_deadline is not None
                and time.monotonic() >= completion_deadline
            ):
                forced_cleanup = True
                log_file.write(
                    "[warn] child did not exit after completion marker; terminating\n"
                )
                if process.poll() is None:
                    _terminate_process_group(process_group_id)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        log_file.write("[warn] SIGTERM timed out; sending SIGKILL\n")
                        _kill_process_group(process_group_id)
                        process.wait(timeout=5)
                    reader.join(timeout=1)
                    if reader.is_alive():
                        log_file.write(
                            "[warn] stdout reader still alive after child exit; "
                            "sending SIGKILL to process group\n"
                        )
                        _kill_process_group(process_group_id)
                elif reader.is_alive():
                    log_file.write(
                        "[warn] child exited but stdout reader is still alive; "
                        "cleaning up inherited pipe holders\n"
                    )
                    _terminate_process_group(process_group_id)
                    reader.join(timeout=1)
                    if reader.is_alive():
                        log_file.write(
                            "[warn] stdout reader still alive after SIGTERM; "
                            "sending SIGKILL to process group\n"
                        )
                        _kill_process_group(process_group_id)

                killed_pids = cleanup_listening_ports(cleanup_ports)
                if killed_pids:
                    log_file.write(f"[warn] cleaned listener pids: {killed_pids}\n")
                break

        reader.join(timeout=1)
        return_code = process.poll()
        if forced_cleanup and completion_seen:
            print(f"[ok] scene_{job.scene_id:03d} completed with forced cleanup")
            return True
        if return_code == 0:
            print(f"[ok] scene_{job.scene_id:03d} completed")
            return True

        log_file.write(f"[error] child exited with code {return_code}\n")
        print(f"[error] scene_{job.scene_id:03d} failed with exit code {return_code}")
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse sequential runner CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="subset_sequential")
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )
    parser.add_argument("--grace-seconds", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--materials-retrieval-server-port",
        type=int,
        default=None,
        help=(
            "Forwarded as experiment.materials_retrieval_server.port. "
            "Example: --materials-retrieval-server-port 6992"
        ),
    )
    parser.add_argument(
        "--start-stage",
        choices=PIPELINE_STAGE_CHOICES,
        default=None,
        help=(
            "Forwarded as experiment.pipeline.start_stage. "
            "When set to a non-floor_plan stage, existing scene outputs are "
            "preserved so the pipeline can resume from the previous checkpoint."
        ),
    )
    parser.add_argument(
        "--stop-stage",
        choices=PIPELINE_STAGE_CHOICES,
        default=None,
        help="Forwarded as experiment.pipeline.stop_stage.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the subset CSV sequentially."""
    args = parse_args(argv)
    if (
        args.start_stage is not None
        and args.stop_stage is not None
        and PIPELINE_STAGE_CHOICES.index(args.start_stage)
        > PIPELINE_STAGE_CHOICES.index(args.stop_stage)
    ):
        raise SystemExit(
            f"--start-stage '{args.start_stage}' cannot be after "
            f"--stop-stage '{args.stop_stage}'"
        )

    jobs = load_prompt_jobs(args.csv_path, args.output_root)
    for job in jobs:
        success = run_job(
            job=job,
            run_name=args.run_name,
            cuda_visible_devices=args.cuda_visible_devices,
            grace_seconds=args.grace_seconds,
            overwrite=args.overwrite,
            materials_retrieval_server_port=args.materials_retrieval_server_port,
            start_stage=args.start_stage,
            stop_stage=args.stop_stage,
        )
        if not success:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
