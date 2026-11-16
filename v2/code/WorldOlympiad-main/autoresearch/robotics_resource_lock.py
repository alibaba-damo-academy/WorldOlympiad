"""Process-safe locks for shared robotics rendering services."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Iterator
import uuid


@contextmanager
def exclusive_case_asset_edit(output_root: str | Path, batch_id: str, case_id: str) -> Iterator[Path]:
    """Serialize published-asset edits for one case without reserving a renderer."""
    key = hashlib.sha256((batch_id + "\0" + case_id).encode()).hexdigest()
    path = Path(output_root).resolve() / ".locks" / f"case_asset_{key}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield path
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def exclusive_blender_job(output_root: str | Path) -> Iterator[Path]:
    """Serialize jobs that own SceneCode/Blender server lifecycles.

    SceneCode generation and formal reference rendering both launch managed
    Blender HTTP servers. Individual request locks cannot protect concurrent
    server startup/shutdown, so one host-level job lock owns that lifecycle.
    """

    root = Path(output_root).resolve()
    lock_path = root / ".locks" / "scenecode_blender_job.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield lock_path
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _gpu_memory(gpu_id: str) -> tuple[int, int, dict[int, int]]:
    """Read physical-device memory; missing process accounting is conservative."""
    command = ["nvidia-smi", f"--id={gpu_id}", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(
            command + ["--query-gpu=memory.total,memory.free"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        total, free = (int(value.strip()) for value in result.stdout.strip().split(","))
        if not 0 <= free <= total or total <= 0:
            raise ValueError("invalid GPU memory values")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot safely read GPU {gpu_id} memory") from exc
    processes: dict[int, int] = {}
    try:
        result = subprocess.run(
            command + ["--query-compute-apps=pid,used_gpu_memory"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            pid, used = (int(value.strip()) for value in line.split(","))
            if pid <= 0 or used < 0:
                raise ValueError("invalid GPU process memory")
            processes[pid] = processes.get(pid, 0) + used
    except (OSError, ValueError, subprocess.SubprocessError):
        # Graphics contexts and unavailable PID accounting remain included in
        # physical usage without reducing pending reservations.
        processes = {}
    return total, free, processes


def _process_parent(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def _pending_memory(leases: list[dict], processes: dict[int, int]) -> int:
    """Reserve each lease's estimated peak minus its observed resident memory.

    A child belongs to the nearest lease-owning ancestor. Unknown ancestry is
    never credited, so external jobs and graphics memory cannot create phantom
    capacity. A second device snapshot caps how much credit can be used.
    """
    observed = [0] * len(leases)
    owners = {int(lease["pid"]): index for index, lease in enumerate(leases)}
    for pid, memory in processes.items():
        visited: set[int] = set()
        while pid and pid not in visited:
            visited.add(pid)
            if pid in owners:
                observed[owners[pid]] += memory
                break
            pid = _process_parent(pid)
    return sum(max(0, int(lease["estimated_memory_mib"]) - used)
               for lease, used in zip(leases, observed))


def _active_gpu_leases(directory: Path) -> list[dict]:
    """Called under admission mutex; kernel locks reclaim crashed owners."""
    leases = []
    for path in directory.glob("*.json"):
        try:
            descriptor = path.open("r+")
        except FileNotFoundError:
            # A normally finishing job may remove its own locked record.
            continue
        with descriptor:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                leases.append(json.load(descriptor))
            else:
                path.unlink(missing_ok=True)
    return leases


@contextmanager
def exclusive_gpu_job(
    output_root: str | Path, gpu_id: str = "0", *,
    estimated_memory_mib: int | None = None, job_kind: str = "gpu_job",
    headroom_mib: int | None = None, poll_interval: float = 1.0,
    timeout: float | None = None,
) -> Iterator[Path]:
    """Admit GPU jobs by peak memory while preserving the historical API.

    Set a measured peak estimate per invocation (keyword or the per-job
    ROBOTICS_GPU_ESTIMATED_MEMORY_MIB environment variable) to share a device.
    Unconfigured jobs keep exclusive ownership, including compatibility with
    older workers. All cooperating jobs must use the same output root and
    physical GPU index. Keep CUDA_VISIBLE_DEVICES consistent with that index.

    ROBOTICS_GPU_HEADROOM_MIB defaults to 2048 MiB. Reservations cover memory
    not yet allocated by admitted jobs, while current physical free memory
    includes unrelated processes. Kernel-held lease locks reclaim reservations
    after a crash; surviving GPU children still count as physical usage.
    This is admission control, not an enforced GPU allocation limit: estimates
    must cover the job's peak, including all subprocesses.
    """
    if not gpu_id.isdigit():
        raise ValueError("a GPU job must name exactly one device")
    if estimated_memory_mib is None:
        configured = os.environ.get("ROBOTICS_GPU_ESTIMATED_MEMORY_MIB")
        estimated_memory_mib = int(configured) if configured else None
    if headroom_mib is None:
        headroom_mib = int(os.environ.get("ROBOTICS_GPU_HEADROOM_MIB", "2048"))
    if (estimated_memory_mib is not None and estimated_memory_mib <= 0) or headroom_mib < 0:
        raise ValueError("GPU estimate must be positive and headroom nonnegative")
    if poll_interval <= 0 or (timeout is not None and timeout < 0):
        raise ValueError("invalid GPU admission wait settings")
    gpu_id = str(int(gpu_id))
    path = Path(output_root).resolve() / ".locks" / f"robotics_gpu_{gpu_id}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        print(f"Waiting for GPU {gpu_id} job lease", flush=True)
        deadline = None if timeout is None else time.monotonic() + timeout
        mode = fcntl.LOCK_EX if estimated_memory_mib is None else fcntl.LOCK_SH
        while True:
            try:
                fcntl.flock(lock, mode | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"GPU {gpu_id} ownership admission timed out")
                time.sleep(poll_interval)
        lease = None
        lease_path = None
        try:
            if estimated_memory_mib is not None:
                directory = path.parent / f"robotics_gpu_{gpu_id}_leases"
                directory.mkdir(mode=0o700, exist_ok=True)
                with (directory / "admission.lock").open("a+") as mutex:
                    while True:
                        fcntl.flock(mutex, fcntl.LOCK_EX)
                        try:
                            leases = _active_gpu_leases(directory)
                            total, free, processes = _gpu_memory(gpu_id)
                            if estimated_memory_mib + headroom_mib > total:
                                raise ValueError(f"GPU {gpu_id} job estimate exceeds device capacity")
                            pending = _pending_memory(leases, processes)
                            # Process and device queries are not atomic. Re-read
                            # device usage after PID accounting; cap credited
                            # allocations by the usage present in BOTH samples.
                            current_total, current_free, _ = _gpu_memory(gpu_id)
                            if current_total != total:
                                raise RuntimeError(f"GPU {gpu_id} capacity changed during admission")
                            credited = sum(int(item["estimated_memory_mib"]) for item in leases) - pending
                            credited = min(credited, total - max(free, current_free))
                            pending = sum(int(item["estimated_memory_mib"]) for item in leases) - credited
                            if min(free, current_free) >= estimated_memory_mib + headroom_mib + pending:
                                lease_path = directory / f"{uuid.uuid4().hex}.json"
                                lease = lease_path.open("x+")
                                fcntl.flock(lease, fcntl.LOCK_EX)
                                json.dump({"pid": os.getpid(), "job_kind": job_kind,
                                           "estimated_memory_mib": estimated_memory_mib}, lease)
                                lease.flush()
                                break
                        finally:
                            fcntl.flock(mutex, fcntl.LOCK_UN)
                        if deadline is not None and time.monotonic() >= deadline:
                            raise TimeoutError(f"GPU {gpu_id} memory admission timed out")
                        time.sleep(poll_interval)
            print(f"Acquired GPU {gpu_id} job lease", flush=True)
            yield path
        finally:
            if lease is not None:
                # Unlink while still locked; admissions see either an active
                # reservation or its absence, never a partly written record.
                if lease_path is not None:
                    lease_path.unlink(missing_ok=True)
                lease.close()
            fcntl.flock(lock, fcntl.LOCK_UN)
