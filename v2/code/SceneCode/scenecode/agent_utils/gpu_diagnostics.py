from __future__ import annotations

import csv
import os
import shutil
import subprocess

from pathlib import Path
from typing import Any


def _read_process_name(pid: int) -> str:
    comm_path = Path(f"/proc/{pid}/comm")
    if comm_path.exists():
        try:
            return comm_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass

    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if cmdline_path.exists():
        try:
            raw = cmdline_path.read_bytes().replace(b"\x00", b" ").strip()
            if raw:
                return raw.decode("utf-8", errors="replace")
        except OSError:
            pass

    return "<unknown>"


def _normalize_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalize_mib(value: Any) -> int | None:
    if value is None:
        return None
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return None
    if numeric_value < 0:
        return None
    return numeric_value


def _collect_with_nvml() -> dict[str, Any] | None:
    try:
        import pynvml
    except Exception:
        return None

    try:
        pynvml.nvmlInit()
    except Exception:
        return None

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        devices: list[dict[str, Any]] = []
        processes: list[dict[str, Any]] = []

        process_getters = (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        )

        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            uuid = _normalize_text(pynvml.nvmlDeviceGetUUID(handle))
            devices.append(
                {
                    "index": index,
                    "uuid": uuid,
                    "name": _normalize_text(pynvml.nvmlDeviceGetName(handle)),
                    "total_mib": int(memory_info.total // (1024 * 1024)),
                    "used_mib": int(memory_info.used // (1024 * 1024)),
                    "free_mib": int(memory_info.free // (1024 * 1024)),
                }
            )

            running_processes = None
            for getter_name in process_getters:
                getter = getattr(pynvml, getter_name, None)
                if getter is None:
                    continue
                try:
                    running_processes = getter(handle)
                    break
                except pynvml.NVMLError_NotSupported:
                    running_processes = []
                    break
                except Exception:
                    continue

            if running_processes is None:
                running_processes = []

            for process_info in running_processes:
                pid = int(process_info.pid)
                used_mib = _normalize_mib(getattr(process_info, "usedGpuMemory", None))
                processes.append(
                    {
                        "gpu_index": index,
                        "gpu_uuid": uuid,
                        "pid": pid,
                        "process_name": _read_process_name(pid),
                        "used_mib": used_mib,
                    }
                )

        return {
            "backend": "nvml",
            "devices": devices,
            "processes": processes,
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _run_nvidia_smi_query(fields: str) -> list[list[str]] | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    result = subprocess.run(
        [
            nvidia_smi,
            f"--query-{fields}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    rows: list[list[str]] = []
    for row in csv.reader(line for line in result.stdout.splitlines() if line.strip()):
        rows.append([item.strip() for item in row])
    return rows


def _collect_with_nvidia_smi() -> dict[str, Any] | None:
    gpu_rows = _run_nvidia_smi_query(
        "gpu=index,uuid,name,memory.total,memory.used,memory.free"
    )
    if gpu_rows is None:
        return None

    devices: list[dict[str, Any]] = []
    uuid_to_index: dict[str, int] = {}
    for row in gpu_rows:
        if len(row) < 6:
            continue
        index = int(row[0])
        uuid = row[1]
        uuid_to_index[uuid] = index
        devices.append(
            {
                "index": index,
                "uuid": uuid,
                "name": row[2],
                "total_mib": int(row[3]),
                "used_mib": int(row[4]),
                "free_mib": int(row[5]),
            }
        )

    process_rows = _run_nvidia_smi_query(
        "compute-apps=gpu_uuid,pid,process_name,used_memory"
    )
    if process_rows is None:
        process_rows = _run_nvidia_smi_query("compute-apps=gpu_uuid,pid,used_memory")

    processes: list[dict[str, Any]] = []
    for row in process_rows or []:
        if len(row) < 3:
            continue

        gpu_uuid = row[0]
        pid = int(row[1])
        if len(row) >= 4:
            process_name = row[2] or _read_process_name(pid)
            used_raw = row[3]
        else:
            process_name = _read_process_name(pid)
            used_raw = row[2]

        processes.append(
            {
                "gpu_index": uuid_to_index.get(gpu_uuid),
                "gpu_uuid": gpu_uuid,
                "pid": pid,
                "process_name": process_name,
                "used_mib": _normalize_mib(used_raw),
            }
        )

    return {
        "backend": "nvidia-smi",
        "devices": devices,
        "processes": processes,
    }


def collect_gpu_snapshot(label: str) -> dict[str, Any]:
    snapshot = {
        "label": label,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "backend": "unavailable",
        "devices": [],
        "processes": [],
    }

    for collector in (_collect_with_nvml, _collect_with_nvidia_smi):
        result = collector()
        if result is None:
            continue
        snapshot.update(result)
        return snapshot

    snapshot["error"] = "Neither NVML nor nvidia-smi was available"
    return snapshot


def format_gpu_snapshot(snapshot: dict[str, Any]) -> str:
    lines = [
        "GPU snapshot "
        f"[{snapshot['label']}] pid={snapshot['pid']} "
        f"CUDA_VISIBLE_DEVICES={snapshot['cuda_visible_devices']} "
        f"backend={snapshot['backend']}"
    ]

    if snapshot.get("error"):
        lines.append(f"reason: {snapshot['error']}")

    devices = snapshot.get("devices", [])
    processes = snapshot.get("processes", [])
    if not devices:
        lines.append("No GPU device information available.")
        return "\n".join(lines)

    for device in devices:
        lines.append(
            "GPU {index} ({name}): used {used_mib} MiB / {total_mib} MiB, "
            "free {free_mib} MiB".format(**device)
        )
        device_processes = [
            process
            for process in processes
            if process.get("gpu_index") == device["index"]
            or process.get("gpu_uuid") == device.get("uuid")
        ]
        if not device_processes:
            lines.append("  processes: none")
            continue

        for process in device_processes:
            used_value = process.get("used_mib")
            used_display = "unknown" if used_value is None else f"{used_value} MiB"
            lines.append(
                "  pid={pid} name={process_name} used={used}".format(
                    pid=process["pid"],
                    process_name=process["process_name"],
                    used=used_display,
                )
            )

    return "\n".join(lines)


def describe_gpu_snapshot(label: str) -> str:
    return format_gpu_snapshot(collect_gpu_snapshot(label))
