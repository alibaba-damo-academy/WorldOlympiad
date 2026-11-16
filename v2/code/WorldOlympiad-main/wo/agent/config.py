"""One non-secret profile for the upper Agent API and the local model under test."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..paths import WORKSPACE

ADAPTERS = {"cosmos3", "lingbot_va", "gr00t", "fastwam", "longwam"}
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DEFAULT_AGENT_MODEL = "gpt-5.6-sol"
DEFAULT_SCORE_MODE = "full"
def configured_gpu_indices() -> tuple[int, ...]:
    """Return the user's explicit NVIDIA device-index allowlist."""
    raw = os.environ.get("WOV2_GPU_ALLOWLIST", "")
    if not raw.strip():
        return ()
    parts = [part.strip() for part in raw.split(",")]
    if any(not re.fullmatch(r"[0-9]+", part) for part in parts):
        raise ValueError("WOV2_GPU_ALLOWLIST must contain only nonnegative integer indices")
    indices = tuple(int(part) for part in parts)
    if len(set(indices)) != len(indices):
        raise ValueError("WOV2_GPU_ALLOWLIST must not contain duplicate indices")
    return tuple(sorted(indices))


def _valid_profile_gpu(value: Any, allowed: set[int]) -> bool:
    if isinstance(value, bool):
        return False
    try:
        index = int(value)
    except (TypeError, ValueError):
        return False
    return index >= 0 and bool(allowed) and index in allowed
SCENECODE_RUNTIME_MARKER = ".worldolympiad_scenecode_runtime_v1"


def config_path(path: str | Path | None = None) -> Path:
    configured = path or os.environ.get("WO_EVAL_CONFIG")
    return Path(configured).expanduser() if configured else WORKSPACE / "config/two-interface.json"


def _reject_secrets(value: Any, where: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key).lower()
            if name in {"api_key", "token", "password", "authorization", "secret_value"}:
                raise ValueError(f"{where}.{key} must be supplied through an environment variable")
            _reject_secrets(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{where}[{index}]")


def load_profile(path: str | Path | None = None, *, required: bool = False) -> dict | None:
    target = config_path(path)
    if not target.is_file():
        if required:
            raise FileNotFoundError(f"two-interface config not found: {target}")
        return None
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("two-interface config requires schema_version 1")
    _reject_secrets(data)

    api = data.get("agent_api")
    model = data.get("tested_model")
    scoring = data.get("scoring", {})
    if not isinstance(api, dict) or not isinstance(model, dict) or not isinstance(scoring, dict):
        raise ValueError("config requires agent_api, tested_model, and optional scoring objects")
    base = str(api.get("base_url", "")).strip()
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or "your_" in parsed.hostname.lower():
        raise ValueError("agent_api.base_url must be a configured http(s) endpoint")
    if parsed.username or parsed.password:
        raise ValueError("do not embed credentials in agent_api.base_url")
    if not str(api.get("model", "")).strip() or "your_" in str(api["model"]).lower():
        raise ValueError("agent_api.model must be configured")
    key_env = str(api.get("api_key_env", "WO_AGENT_API_KEY"))
    if not ENV_NAME.fullmatch(key_env):
        raise ValueError("agent_api.api_key_env must be an environment-variable name")
    adapter = model.get("adapter")
    if adapter not in ADAPTERS:
        raise ValueError(f"tested_model.adapter must be one of {sorted(ADAPTERS)}")
    checkpoint = str(model.get("checkpoint", "")).strip()
    if not checkpoint:
        raise ValueError("tested_model.checkpoint is required")
    if "scenecode_source" in model and (
        not isinstance(model["scenecode_source"], str) or not model["scenecode_source"].strip()
    ):
        raise ValueError("tested_model.scenecode_source must be a nonempty path")
    workspace = Path(data.get("workspace_root", "")).expanduser()
    if not workspace.is_absolute():
        raise ValueError("workspace_root must be an absolute path")
    allowed_gpus = set(configured_gpu_indices())
    for section in (model, scoring):
        fields = ("gpu", "compute_gpu") if section is scoring else ("gpu",)
        for field in fields:
            value = section.get(field, "auto")
            if value != "auto" and not _valid_profile_gpu(value, allowed_gpus):
                raise ValueError(f"{field} must be a nonnegative index in WOV2_GPU_ALLOWLIST")
    simulator_gpu = model.get("simulator_gpu", "auto")
    if simulator_gpu != "auto" and not _valid_profile_gpu(simulator_gpu, allowed_gpus):
        raise ValueError("tested_model.simulator_gpu must be in WOV2_GPU_ALLOWLIST")
    policy_gpus = model.get("policy_gpus", "auto")
    if policy_gpus != "auto" and (
        not isinstance(policy_gpus, list) or not policy_gpus
        or any(not _valid_profile_gpu(gpu, allowed_gpus) for gpu in policy_gpus)
        or len({int(gpu) for gpu in policy_gpus}) != len(policy_gpus)
    ):
        raise ValueError("tested_model.policy_gpus must be a nonempty list of unique allowlisted indices")
    if isinstance(policy_gpus, list) and model.get("gpu", "auto") != "auto":
        if int(policy_gpus[0]) != int(model["gpu"]):
            raise ValueError("tested_model.policy_gpus must start with tested_model.gpu")
    mode = scoring.get("mode", "auto")
    if mode not in {"auto", "full", "split"}:
        raise ValueError("scoring.mode must be auto, full, or split")
    data["_config_path"] = str(target.resolve())
    return data


def private_environment_path(env: dict[str, str] | None = None,
                             workspace: str | Path = WORKSPACE) -> Path:
    """Resolve the caller-selected private env file without reading its contents."""
    target_env = os.environ if env is None else env
    configured = target_env.get("WO_AGENT_ENV_FILE")
    return Path(configured).expanduser() if configured else Path(workspace) / ".secrets/scenecode.env"


def load_private_environment(env: dict[str, str] | None = None, path: str | Path | None = None) -> Path:
    """Load the existing private env profile without echoing its contents."""
    target_env = os.environ if env is None else env
    target = Path(path).expanduser() if path is not None else private_environment_path(target_env)
    if target.is_file():
        result = subprocess.run(
            ["bash", "-c", 'set -a; . "$1"; env -0', "wo-agent-profile", str(target)],
            capture_output=True, check=True,
        )
        for entry in result.stdout.decode(errors="replace").split("\0"):
            if "=" in entry:
                key, value = entry.split("=", 1)
                target_env.setdefault(key, value)
    return target


def gpu_inventory(text: str) -> list[dict[str, int]]:
    allowed = set(configured_gpu_indices())
    rows = []
    try:
        for line in text.splitlines():
            parts = [int(value.strip()) for value in line.split(",")]
            if len(parts) == 3:
                rows.append({"index": parts[0], "total_mib": parts[1], "free_mib": parts[2]})
            elif len(parts) == 4:
                rows.append({"index": parts[0], "total_mib": parts[1],
                             "free_mib": parts[2], "utilization": parts[3]})
    except ValueError:
        return []
    return [row for row in rows if row["index"] in allowed]


def _allowed_inventory(inventory: list[dict[str, int]]) -> list[dict[str, int]]:
    """Fail closed for missing allowlists and unselected/stale inventory rows."""
    allowed = set(configured_gpu_indices())
    if not allowed:
        return []
    return [item for item in inventory if item.get("index") in allowed]


def all_gpu_info() -> list[dict[str, int]]:
    allowlist = configured_gpu_indices()
    if not allowlist:
        return []
    query_indices = ",".join(map(str, allowlist))
    common = ["nvidia-smi", "-i", query_indices]
    try:
        result = subprocess.run(
            common + ["--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
                      "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        by_uuid = {}
        for line in result.stdout.splitlines():
            parts = [value.strip() for value in line.split(",")]
            if len(parts) != 5:
                return []
            index, uuid, total, free, utilization = parts
            index = int(index)
            if index not in allowlist:
                return []
            by_uuid[uuid] = {"index": index, "total_mib": int(total),
                             "free_mib": int(free), "utilization": int(utilization)}
        if {item["index"] for item in by_uuid.values()} != set(allowlist):
            return []
    except (OSError, ValueError, subprocess.SubprocessError):
        return []

    try:
        result = subprocess.run(
            common + ["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        process_pids = {}
        for line in result.stdout.splitlines():
            parts = [value.strip() for value in line.split(",")]
            if len(parts) != 2:
                return []
            uuid, pid = parts
            if uuid not in by_uuid:
                return []
            process_pids.setdefault(uuid, []).append(int(pid))
        process_query_ok = True
    except (OSError, ValueError, subprocess.SubprocessError):
        process_pids = {}
        process_query_ok = False

    return _allowed_inventory([
        dict(info, compute_process_count=len(process_pids.get(uuid, [])),
             compute_pids=process_pids.get(uuid, []), process_query_ok=process_query_ok)
        for uuid, info in by_uuid.items()
    ])



def _gpu_available(info: dict[str, int], min_free_mib: int) -> bool:
    return (info.get("process_query_ok", True)
            and info.get("compute_process_count", 0) == 0
            and info["free_mib"] >= min_free_mib
            and info.get("utilization", 0) <= 5)


def gpu_info(gpu: int = 0, *, inventory: list[dict[str, int]] | None = None) -> dict[str, int] | None:
    if isinstance(gpu, bool) or int(gpu) not in configured_gpu_indices():
        return None
    inventory = all_gpu_info() if inventory is None else _allowed_inventory(inventory)
    return next((item for item in inventory if item["index"] == int(gpu)), None)


PREPARATION_MIN_FREE_MIB = 8 * 1024


def resolve_preparation_gpu(profile: dict | None, *,
                            inventory: list[dict[str, int]] | None = None) -> int:
    """Choose one idle card for SceneCode/reference stages, independent of policy FSDP."""
    settings = (profile or {}).get("tested_model", {})
    requested = settings.get("gpu", "auto")
    inventory = all_gpu_info() if inventory is None else _allowed_inventory(inventory)
    if not inventory:
        raise RuntimeError("preparation preflight could not query the configured WOV2_GPU_ALLOWLIST")
    if requested != "auto":
        index = int(requested)
        if index not in configured_gpu_indices():
            raise ValueError("preparation GPU must be in the WOV2_GPU_ALLOWLIST")
        info = gpu_info(index, inventory=inventory)
        if info is None or not _gpu_available(info, PREPARATION_MIN_FREE_MIB):
            raise RuntimeError(f"preparation GPU {index} is occupied or below {PREPARATION_MIN_FREE_MIB} MiB free")
        return index
    eligible = [item for item in inventory if _gpu_available(item, PREPARATION_MIN_FREE_MIB)]
    if not eligible:
        raise RuntimeError(f"preparation needs 1 idle allowlisted NVIDIA GPU with {PREPARATION_MIN_FREE_MIB} MiB free; found 0")
    return min(eligible, key=lambda item: (-item["free_mib"], item["index"]))["index"]


def resolve_gpu(profile: dict | None, section: str = "tested_model",
                *, inventory: list[dict[str, int]] | None = None) -> int:
    settings = (profile or {}).get(section, {})
    requested = settings.get("gpu", "auto")
    policy_gpus = settings.get("policy_gpus", "auto")
    if section == "tested_model" and requested == "auto" and isinstance(policy_gpus, list):
        requested = policy_gpus[0]
    if requested != "auto":
        index = int(requested)
        if index not in configured_gpu_indices():
            raise ValueError("configured GPU must be in the WOV2_GPU_ALLOWLIST")
        inventory = all_gpu_info() if inventory is None else _allowed_inventory(inventory)
        info = gpu_info(index, inventory=inventory)
        if info is None:
            raise RuntimeError(f"configured GPU {index} is not present or could not be queried")
        total_gb = info["total_mib"] / 1024
        model = (profile or {}).get("tested_model", {})
        if section == "tested_model" and model.get("adapter") == "lingbot_va" and total_gb <= 40:
            min_free_mib = 30 * 1024
        else:
            min_free_mib = 25_600 if total_gb <= 40 else 48 * 1024
        if not _gpu_available(info, min_free_mib):
            raise RuntimeError(f"configured GPU {index} is occupied or below its {min_free_mib} MiB free threshold")
        return index
    inventory = all_gpu_info() if inventory is None else _allowed_inventory(inventory)
    if not inventory:
        raise RuntimeError("no GPU is available within WOV2_GPU_ALLOWLIST")
    model = (profile or {}).get("tested_model", {})
    candidates = []
    for item in inventory:
        total_gb = item["total_mib"] / 1024
        if section == "tested_model" and model.get("adapter") == "lingbot_va" and total_gb <= 40:
            min_free_mib = 30 * 1024
        else:
            min_free_mib = 25_600 if total_gb <= 40 else 48 * 1024
        if _gpu_available(item, min_free_mib):
            candidates.append(item)
    if not candidates:
        raise RuntimeError(f"no GPU has enough free memory for {section}")
    return min(candidates, key=lambda item: (-item["free_mib"], item["index"]))["index"]


def resolve_simulator_gpu(profile: dict | None, model_gpu: int, *,
                          inventory: list[dict[str, int]] | None = None) -> int | None:
    """Put a 32-GB-card simulator on another idle GPU; 80-GB cards stay single-GPU."""
    model = (profile or {}).get("tested_model", {})
    requested = model.get("simulator_gpu", "auto")
    if requested != "auto":
        index = int(requested)
        if index not in configured_gpu_indices():
            raise ValueError("simulator GPU must be in the WOV2_GPU_ALLOWLIST")
        inventory = all_gpu_info() if inventory is None else _allowed_inventory(inventory)
        info = gpu_info(index, inventory=inventory)
        if info is None or not _gpu_available(info, 25_600):
            raise RuntimeError(f"configured simulator GPU {index} is occupied or below 25,600 MiB free")
        return index
    inventory = all_gpu_info() if inventory is None else _allowed_inventory(inventory)
    model_info = next((item for item in inventory if item["index"] == int(model_gpu)), None)
    if model_info is None:
        return None
    if model_info["total_mib"] > 40 * 1024:
        return int(model_gpu)
    candidates = [item for item in inventory
                  if item["index"] != int(model_gpu) and _gpu_available(item, 25_600)]
    return min(candidates, key=lambda item: (-item["free_mib"], item["index"]))["index"] if candidates else None


def resolve_policy_gpus(profile: dict | None, model_gpu: int, simulator_gpu: int | None, *,
                        inventory: list[dict[str, int]] | None = None) -> tuple[int, ...]:
    """Keep low-memory LingBot policy ranks separate from its RoboTwin process."""
    model = (profile or {}).get("tested_model", {})
    inventory = all_gpu_info() if inventory is None else _allowed_inventory(inventory)
    requested = model.get("policy_gpus", "auto")
    model_info = gpu_info(model_gpu, inventory=inventory)
    if isinstance(requested, list):
        policy_gpus = tuple(int(gpu) for gpu in requested)
    elif model.get("adapter") == "lingbot_va" and model_info is not None \
            and model_info["total_mib"] <= 40 * 1024:
        if simulator_gpu is None:
            raise RuntimeError("LingBot on 32-GB cards needs a separate idle GPU for RoboTwin rendering")
        simulator_info = gpu_info(simulator_gpu, inventory=inventory)
        if simulator_info is None or not _gpu_available(simulator_info, 25_600):
            raise RuntimeError("RoboTwin simulator GPU is unavailable or below the idle-GPU threshold")
        eligible = [item for item in inventory
                    if item["index"] not in {int(model_gpu), int(simulator_gpu)}
                    and _gpu_available(item, 30 * 1024)]
        if not eligible:
            raise RuntimeError(
                "LingBot on 32-GB cards needs three idle GPUs: two for policy FSDP and one for RoboTwin; "
                "each must have at least 25,600 MiB free and <=5% utilization"
            )
        policy_gpus = (int(model_gpu), min(eligible, key=lambda item: (-item["free_mib"], item["index"]))["index"])
    else:
        policy_gpus = (int(model_gpu),)

    if not policy_gpus or policy_gpus[0] != int(model_gpu):
        raise ValueError("tested_model.policy_gpus must start with the resolved policy GPU")
    if len(set(policy_gpus)) != len(policy_gpus):
        raise ValueError("policy GPU indices must be unique")
    small_lingbot = (model.get("adapter") == "lingbot_va" and model_info is not None
                     and model_info["total_mib"] <= 40 * 1024)
    if small_lingbot:
        if len(policy_gpus) < 2:
            raise RuntimeError("LingBot on 32-GB cards needs two policy GPUs plus a separate simulator GPU")
        if simulator_gpu in policy_gpus:
            raise RuntimeError("32-GB LingBot needs separate GPUs for two FSDP ranks and RoboTwin rendering")
    for gpu in policy_gpus:
        info = gpu_info(gpu, inventory=inventory)
        min_free_mib = 30 * 1024 if small_lingbot else 25_600
        if info is None or not _gpu_available(info, min_free_mib):
            raise RuntimeError(f"policy GPU {gpu} is unavailable or below the memory/utilization threshold")
    return policy_gpus


SPLIT_SCORE_MIN_FREE_MIB = 25_600
FULL_SCORE_MIN_FREE_MIB = 48 * 1024


def score_mode(profile: dict | None, *, info: dict[str, int] | None = None,
               inventory: list[dict[str, int]] | None = None) -> str:
    """Choose split scoring on 32-GB devices; full mode requires a larger card."""
    requested = ((profile or {}).get("scoring") or {}).get("mode", "auto")
    if requested != "auto":
        return requested
    if info is None:
        inventory = all_gpu_info() if inventory is None else _allowed_inventory(inventory)
        configured = ((profile or {}).get("scoring") or {}).get("gpu", "auto")
        if configured != "auto":
            info = next((item for item in inventory if item["index"] == int(configured)), None)
        elif inventory:
            info = max(inventory, key=lambda item: item["total_mib"])
    if info is None:
        return DEFAULT_SCORE_MODE
    return "split" if info["total_mib"] <= 40 * 1024 else "full"


def _score_gpu_reason(info: dict[str, int] | None, required_mib: int) -> str:
    if info is None:
        return "not visible or query failed"
    if not info.get("process_query_ok", True):
        return "compute-process query failed"
    parts = []
    if info.get("compute_process_count", 0):
        pids = info.get("compute_pids", [])
        parts.append(f"compute PIDs={pids or 'present'}")
    if info.get("utilization", 0) > 5:
        parts.append(f"utilization={info['utilization']}%")
    if info.get("free_mib", 0) < required_mib:
        parts.append(f"free={info.get('free_mib', 0)} MiB < required={required_mib} MiB")
    return ", ".join(parts) if parts else "eligible"


def resolve_score_gpus(profile: dict | None, *,
                       inventory: list[dict[str, int]] | None = None) -> dict[str, int | str]:
    """Reserve an automatic single-card or split VLM/compute topology within the user allowlist."""
    inventory = _allowed_inventory(all_gpu_info() if inventory is None else inventory)
    if not inventory:
        raise RuntimeError("score preflight could not query the configured WOV2_GPU_ALLOWLIST")
    settings = (profile or {}).get("scoring", {})
    mode = score_mode(profile, inventory=inventory)
    if mode not in {"full", "split"}:
        raise ValueError("scoring.mode must resolve to full or split")
    min_free = FULL_SCORE_MIN_FREE_MIB if mode == "full" else SPLIT_SCORE_MIN_FREE_MIB
    if mode == "full" and max(item["total_mib"] for item in inventory) <= 40 * 1024:
        raise RuntimeError("full scoring requires a single GPU larger than 40 GiB; use split mode on smaller GPUs")

    requested_service = settings.get("gpu", "auto")
    requested_compute = settings.get("compute_gpu", "auto")
    for label, value in (("scoring.gpu", requested_service),
                         ("scoring.compute_gpu", requested_compute)):
        if value != "auto" and (isinstance(value, bool) or int(value) not in configured_gpu_indices()):
            raise ValueError(f"{label} must be auto or a nonnegative allowlisted GPU index")

    def card(index: int) -> dict[str, int] | None:
        return next((item for item in inventory if item["index"] == index), None)

    def require_card(index: int, label: str) -> int:
        info = card(index)
        reason = _score_gpu_reason(info, min_free)
        if reason != "eligible":
            raise RuntimeError(f"{label} GPU {index} is not eligible for {mode} scoring: {reason}")
        return index

    eligible = [item for item in inventory
                if _score_gpu_reason(item, min_free) == "eligible"]
    eligible.sort(key=lambda item: (-item["free_mib"], item["index"]))

    if mode == "full":
        if requested_compute != "auto" and int(requested_compute) != int(requested_service):
            raise ValueError("full scoring uses one GPU; compute_gpu must match scoring.gpu")
        if requested_service == "auto":
            if not eligible:
                raise RuntimeError(
                    f"full scoring needs 1 idle GPU with {min_free} MiB free; found 0. "
                    + "; ".join(f"GPU {item['index']}: {_score_gpu_reason(item, min_free)}"
                                for item in inventory)
                )
            service = eligible[0]["index"]
        else:
            service = require_card(int(requested_service), "VLM")
        return {"mode": mode, "gpu": service, "compute_gpu": service,
                "required_free_mib_per_gpu": min_free}

    service = None if requested_service == "auto" else require_card(int(requested_service), "VLM")
    compute = None if requested_compute == "auto" else require_card(int(requested_compute), "compute")
    if service is not None and compute is not None and service == compute:
        raise ValueError("split scoring needs distinct VLM-service and SAM/DA compute GPUs")
    if service is None:
        service = next((item["index"] for item in eligible if item["index"] != compute), None)
    if compute is None:
        compute = next((item["index"] for item in eligible if item["index"] != service), None)
    if service is None or compute is None:
        details = "; ".join(
            f"GPU {item['index']}: {_score_gpu_reason(item, min_free)}" for item in inventory
        )
        available = len(eligible)
        raise RuntimeError(
            f"split scoring needs 2 distinct idle allowlisted NVIDIA GPUs, each with at least "
            f"{min_free} MiB free, utilization <=5%, and no compute PID; found {available}/2. "
            f"GPU status: {details}"
        )
    return {"mode": mode, "gpu": int(service), "compute_gpu": int(compute),
            "required_free_mib_per_gpu": min_free}

def resource_limits(profile: dict | None = None, *, info: dict[str, int] | None = None,
                    inventory: list[dict[str, int]] | None = None) -> dict[str, int]:
    """Single-GPU stage reservations with headroom for the display/runtime stack."""
    inventory = all_gpu_info() if inventory is None else _allowed_inventory(inventory)
    if not inventory and info is None:
        return {"budget": 16, "episode": 8, "policy_check": 8, "score": 8, "scene_check": 4}
    gpu = resolve_gpu(profile, "tested_model", inventory=inventory)
    info = gpu_info(gpu, inventory=inventory) if info is None else info
    if info is None:
        # Query-free/import-time fallback. Live preflight supplies measured inventory.
        return {"budget": 16, "episode": 8, "policy_check": 8, "score": 8, "scene_check": 4}
    total_gb = info["total_mib"] / 1024
    if total_gb <= 40:
        return {"budget": max(22, int(total_gb * 0.80)), "episode": 22,
                "policy_check": 22, "score": 22, "scene_check": 6}
    return {"budget": max(52, int(total_gb * 0.86)), "episode": 48,
            "policy_check": 48, "score": 50, "scene_check": 8}


def scenecode_runtime_status(path: str | Path) -> str:
    """A missing runtime can be staged; an existing one must carry our marker."""
    root = Path(path)
    if not root.exists():
        return "not_staged"
    return "ready" if (root / SCENECODE_RUNTIME_MARKER).is_file() else "unrecognized"


def apply_environment(env: dict[str, str], profile: dict | None = None, *,
                      inventory: list[dict[str, int]] | None = None,
                      phase: str = "evaluation") -> dict[str, str]:
    phases = {"worker", "preparation", "episode", "scoring", "evaluation"}
    if phase not in phases:
        raise ValueError(f"resource phase must be one of {sorted(phases)}")
    profile = load_profile() if profile is None else profile
    if profile is None:
        return env
    inventory = ([] if phase == "worker" else
                 all_gpu_info() if inventory is None else _allowed_inventory(inventory))
    api = profile["agent_api"]
    model = profile["tested_model"]
    workspace_root = Path(profile["workspace_root"]).expanduser().resolve()
    env["WO_WORKSPACE"] = str(workspace_root)
    env["WORLDOLYMPIAD_WORKSPACE"] = str(workspace_root)
    configured_base = env.get("WO_AGENT_API_BASE") or api["base_url"]
    env["WO_AGENT_API_BASE"] = configured_base
    env["OPENAI_API_BASE"] = configured_base
    env["OPENAI_BASE_URL"] = configured_base
    key = env.get(api.get("api_key_env", "WO_AGENT_API_KEY")) or env.get("OPENAI_API_KEY")
    if key:
        env["OPENAI_API_KEY"] = key
    env["WO_AGENT_MODEL"] = env.get("WO_AGENT_API_MODEL") or api["model"]
    env["WO_TESTED_MODEL"] = model["adapter"]

    def workspace_path(value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (workspace_root / path).resolve()

    def workspace_executable(value):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = workspace_root / path
        return str(Path(os.path.abspath(path)))

    env["WO_TESTED_MODEL_CHECKPOINT"] = str(workspace_path(model["checkpoint"]))
    scenecode_source = model.get("scenecode_source")
    if scenecode_source is not None:
        env["SCENECODE_SOURCE_ROOT"] = str(workspace_path(scenecode_source))
    else:
        env.setdefault("SCENECODE_SOURCE_ROOT", str(workspace_path("code/SceneCode-main")))
    env["SCENECODE_RUNTIME_ROOT"] = str(workspace_path(
        model.get("scenecode_runtime", "tmp/scenecode_runtime")
    ))
    ffmpeg = workspace_path(model.get("ffmpeg", "runtime/ffmpeg"))
    if ffmpeg.exists():
        env.setdefault("WO_FFMPEG", str(ffmpeg))
    for key_name, name in (("robotwin_repo", "WO_ROBOTWIN_REPO"),
                           ("lingbot_repo", "WO_LINGBOT_REPO"),
                           ("robotwin_python", "WO_ROBOTWIN_PYTHON"),
                           ("lingbot_python", "WO_LINGBOT_PYTHON")):
        if model.get(key_name):
            resolver = workspace_executable if key_name.endswith("_python") else workspace_path
            env[name] = str(resolver(model[key_name]))

    model_gpu = None
    simulator_gpu = None
    policy_gpus: tuple[int, ...] = ()
    score_gpus = None
    if phase in {"preparation", "episode", "evaluation"}:
        model_gpu = (resolve_preparation_gpu(profile, inventory=inventory)
                     if phase == "preparation" else
                     resolve_gpu(profile, "tested_model", inventory=inventory))
        if phase == "preparation":
            policy_gpus = (model_gpu,)
        else:
            simulator_gpu = resolve_simulator_gpu(profile, model_gpu, inventory=inventory)
            policy_gpus = resolve_policy_gpus(profile, model_gpu, simulator_gpu,
                                              inventory=inventory)
    if phase in {"scoring", "evaluation"}:
        score_gpus = resolve_score_gpus(profile, inventory=inventory)

    for name in ("WO_TESTED_MODEL_GPU", "WO_TESTED_MODEL_GPUS", "WO_SIMULATOR_GPU",
                 "WO_SCORE_GPU", "WO_SCORE_COMPUTE_GPU"):
        env.pop(name, None)
    if model_gpu is not None:
        env["WO_TESTED_MODEL_GPU"] = str(model_gpu)
        env["WO_TESTED_MODEL_GPUS"] = ",".join(map(str, policy_gpus))
    if simulator_gpu is not None:
        env["WO_SIMULATOR_GPU"] = str(simulator_gpu)
    if score_gpus is not None:
        env["WO_SCORE_GPU"] = str(score_gpus["gpu"])
        env["WO_SCORE_COMPUTE_GPU"] = str(score_gpus["compute_gpu"])
    env["WO_SCORE_MODE"] = str(score_gpus["mode"]) if score_gpus else "deferred"
    env["WO_RESOURCE_PHASE"] = phase
    env["WO_EVAL_CONFIG"] = profile["_config_path"]
    return env

def job_defaults(kind: str, args: dict, profile: dict | None = None) -> dict:
    profile = load_profile() if profile is None else profile
    result = dict(args)
    if kind in {"episode", "score"}:
        for field in ("gpu", "simulator_gpu", "compute_gpu"):
            value = result.get(field)
            if value is not None and (isinstance(value, bool) or int(value) not in configured_gpu_indices()):
                raise ValueError(f"{field} must use a nonnegative index from WOV2_GPU_ALLOWLIST")
        for value in result.get("policy_gpus", ()):
            if isinstance(value, bool) or int(value) not in configured_gpu_indices():
                raise ValueError("policy_gpus must use allowlisted GPU indices")
    if profile is None or kind not in {"episode", "score"}:
        return result
    model = profile["tested_model"]
    requested = result.get("model")
    if requested and requested != model["adapter"]:
        raise ValueError(f"configured tested model is {model['adapter']!r}, got {requested!r}")
    result["model"] = model["adapter"]

    if kind == "score":
        selection_profile = dict(profile)
        selection_scoring = dict(profile.get("scoring", {}))
        for key in ("gpu", "compute_gpu"):
            if result.get(key) is not None:
                selection_scoring[key] = result[key]
        if result.get("score_mode") is not None:
            selection_scoring["mode"] = result["score_mode"]
        selection_profile["scoring"] = selection_scoring
        selection = resolve_score_gpus(selection_profile, inventory=all_gpu_info())
        result["gpu"] = int(selection["gpu"])
        result["compute_gpu"] = int(selection["compute_gpu"])
        result["score_mode"] = str(selection["mode"])
        return result

    inventory = all_gpu_info()
    result.setdefault("gpu", resolve_gpu(profile, "tested_model", inventory=inventory))
    if kind == "episode" and model["adapter"] == "lingbot_va":
        simulator_gpu = resolve_simulator_gpu(profile, result["gpu"], inventory=inventory)
        if simulator_gpu is None:
            raise RuntimeError(
                "LingBot on a 32-GB GPU needs separate idle GPUs for its policy and RoboTwin rendering"
            )
        if simulator_gpu == result["gpu"]:
            info = gpu_info(result["gpu"], inventory=inventory)
            if info is not None and info["total_mib"] <= 40 * 1024:
                raise RuntimeError("LingBot model and RoboTwin simulator must use different GPUs on 32-GB cards")
        result.setdefault("simulator_gpu", simulator_gpu)
        requested_policy_gpus = result.get("policy_gpus")
        if requested_policy_gpus is not None and (
            not isinstance(requested_policy_gpus, (list, tuple)) or not requested_policy_gpus
        ):
            raise ValueError("episode policy_gpus must be a nonempty list of GPU indices")
        if requested_policy_gpus is not None and model.get("policy_gpus", "auto") == "auto":
            selection_profile = dict(profile)
            selection_model = dict(model)
            selection_model["policy_gpus"] = requested_policy_gpus
            selection_profile["tested_model"] = selection_model
            policy_gpus = resolve_policy_gpus(selection_profile, result["gpu"],
                                              result["simulator_gpu"], inventory=inventory)
        else:
            policy_gpus = resolve_policy_gpus(profile, result["gpu"], result["simulator_gpu"],
                                              inventory=inventory)
            if requested_policy_gpus is not None \
                    and tuple(map(int, requested_policy_gpus)) != policy_gpus:
                raise ValueError("episode policy_gpus must match the configured GPU selection")
        result["policy_gpus"] = list(policy_gpus)
    return result

def preflight(profile: dict, env: dict[str, str] | None = None, *,
              phase: str = "evaluation",
              inventory: list[dict[str, int]] | None = None) -> dict:
    if phase not in {"worker", "preparation", "episode", "scoring", "evaluation"}:
        raise ValueError("invalid resource phase")
    env = os.environ if env is None else env
    inventory = ([] if phase == "worker" else
                 all_gpu_info() if inventory is None else _allowed_inventory(inventory))
    applied = apply_environment(dict(env), profile, inventory=inventory, phase=phase)
    model = profile["tested_model"]
    checkpoint = Path(applied["WO_TESTED_MODEL_CHECKPOINT"])
    scenecode_source = Path(applied["SCENECODE_SOURCE_ROOT"])
    scenecode_runtime = Path(applied["SCENECODE_RUNTIME_ROOT"])
    model_gpu = int(applied["WO_TESTED_MODEL_GPU"]) if "WO_TESTED_MODEL_GPU" in applied else None
    model_info = gpu_info(model_gpu, inventory=inventory) if model_gpu is not None else None
    simulator_gpu = (int(applied["WO_SIMULATOR_GPU"])
                     if "WO_SIMULATOR_GPU" in applied else None)
    policy_gpus = tuple(map(int, applied.get("WO_TESTED_MODEL_GPUS", "").split(","))) \
        if applied.get("WO_TESTED_MODEL_GPUS") else ()
    score_selection = None
    if phase in {"scoring", "evaluation"}:
        score_selection = resolve_score_gpus(profile, inventory=inventory)
    robotwin_repo_value = applied.get("WO_ROBOTWIN_REPO")
    robotwin_required_files = {}
    if model["adapter"] == "lingbot_va" and robotwin_repo_value:
        robotwin_repo = Path(robotwin_repo_value)
        robotwin_required_files = {
            relative: (robotwin_repo / relative).is_file()
            for relative in (
                "assets/embodiments/aloha-agilex/config.yml",
                "assets/objects/objaverse/list.json",
            )
        }
    score_payload = ({"mode": score_selection["mode"],
                      "vlm_gpu": score_selection["gpu"],
                      "compute_gpu": score_selection["compute_gpu"],
                      "distinct_gpus": score_selection["gpu"] != score_selection["compute_gpu"],
                      "required_free_mib_per_gpu": score_selection["required_free_mib_per_gpu"]}
                     if score_selection else {"deferred": True})
    resources = {
        "episode": (resource_limits(profile, info=model_info, inventory=inventory)
                    if phase in {"episode", "evaluation"} else {"deferred": True}),
        "scoring": score_payload,
    }
    return {
        "config": profile["_config_path"],
        "phase": phase,
        "agent_api": {
            "base_url_configured": bool(applied.get("OPENAI_BASE_URL")),
            "model": applied["WO_AGENT_MODEL"],
            "api_key_present": bool(applied.get("OPENAI_API_KEY")),
        },
        "tested_model": {
            "adapter": model["adapter"],
            "checkpoint_exists": checkpoint.exists(),
            "scenecode_source": str(scenecode_source),
            "scenecode_source_exists": (scenecode_source / "main.py").is_file(),
            "scenecode_runtime": str(scenecode_runtime),
            "scenecode_runtime_status": scenecode_runtime_status(scenecode_runtime),
            "gpu": model_gpu,
            "policy_gpus": list(policy_gpus),
            "simulator_gpu": simulator_gpu,
            "episode_resources_deferred": phase not in {"episode", "evaluation"},
            "simulator_gpu_ready": simulator_gpu is not None,
            "robotwin_required_files": robotwin_required_files,
            "robotwin_runtime_ready": all(robotwin_required_files.values())
            if robotwin_required_files else model["adapter"] != "lingbot_va",
        },
        "gpu_inventory": [
            {key: item.get(key) for key in ("index", "total_mib", "free_mib", "utilization",
                                             "compute_process_count", "compute_pids",
                                             "process_query_ok")}
            for item in inventory
        ],
        "resources": resources,
    }

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--phase", choices=("worker", "preparation", "episode", "scoring", "evaluation"), default="evaluation")
    parser.add_argument("command", choices=("check",))
    args = parser.parse_args()
    profile = load_profile(args.config, required=True)
    env = os.environ
    env.setdefault("WO_WORKSPACE", str(Path(profile["workspace_root"]).expanduser().resolve()))
    load_private_environment(env)
    print(json.dumps(preflight(profile, phase=args.phase), indent=2))


if __name__ == "__main__":
    main()
