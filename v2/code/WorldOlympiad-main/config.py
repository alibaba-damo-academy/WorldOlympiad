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
    for section in (model, scoring):
        value = section.get("gpu", "auto")
        if value != "auto" and (isinstance(value, bool) or int(value) < 0):
            raise ValueError("gpu must be auto or a nonnegative GPU index")
    mode = scoring.get("mode", "auto")
    if mode not in {"auto", "full", "split"}:
        raise ValueError("scoring.mode must be auto, full, or split")
    data["_config_path"] = str(target.resolve())
    return data


def load_private_environment(env: dict[str, str] | None = None, path: str | Path | None = None) -> Path:
    """Load the existing private env profile without echoing its contents."""
    target_env = os.environ if env is None else env
    target = Path(path or target_env.get("WO_AGENT_ENV_FILE", WORKSPACE / ".secrets/scenecode.env"))
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
    rows = []
    try:
        for line in text.splitlines():
            parts = [int(value.strip()) for value in line.split(",")]
            if len(parts) == 3:
                rows.append({"index": parts[0], "total_mib": parts[1], "free_mib": parts[2]})
    except ValueError:
        return []
    return rows


def all_gpu_info() -> list[dict[str, int]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return gpu_inventory(result.stdout)
    except (OSError, subprocess.SubprocessError):
        return []


def gpu_info(gpu: int = 0) -> dict[str, int] | None:
    return next((item for item in all_gpu_info() if item["index"] == int(gpu)), None)


def resolve_gpu(profile: dict | None, section: str = "tested_model",
                *, inventory: list[dict[str, int]] | None = None) -> int:
    requested = (profile or {}).get(section, {}).get("gpu", "auto")
    if requested != "auto":
        return int(requested)
    inventory = all_gpu_info() if inventory is None else inventory
    if not inventory:
        return 0
    candidates = []
    for item in inventory:
        total_gb = item["total_mib"] / 1024
        min_free_gb = 24 if total_gb <= 40 else 48
        if item["free_mib"] / 1024 >= min_free_gb:
            candidates.append(item)
    if not candidates:
        raise RuntimeError(f"no GPU has enough free memory for {section}")
    return min(candidates, key=lambda item: item["index"])["index"]


def score_mode(profile: dict | None, *, info: dict[str, int] | None = None) -> str:
    requested = ((profile or {}).get("scoring") or {}).get("mode", "auto")
    if requested != "auto":
        return requested
    target = resolve_gpu(profile, "scoring")
    info = gpu_info(int(target)) if info is None else info
    if info is None:
        return DEFAULT_SCORE_MODE
    return "split" if info["total_mib"] <= 40 * 1024 else "full"


def resource_limits(profile: dict | None = None, *, info: dict[str, int] | None = None) -> dict[str, int]:
    """Single-GPU stage reservations with headroom for the display/runtime stack."""
    gpu = resolve_gpu(profile, "tested_model")
    info = gpu_info(gpu) if info is None else info
    if info is None:
        return {"budget": 92, "episode": 48, "policy_check": 48, "score": 50, "scene_check": 8}
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


def apply_environment(env: dict[str, str], profile: dict | None = None) -> dict[str, str]:
    profile = load_profile() if profile is None else profile
    if profile is None:
        return env
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

    model_gpu = resolve_gpu(profile, "tested_model")
    score_gpu = resolve_gpu(profile, "scoring")
    env["WO_TESTED_MODEL_CHECKPOINT"] = str(workspace_path(model["checkpoint"]))
    env["WO_TESTED_MODEL_GPU"] = str(model_gpu)
    scenecode_source = model.get("scenecode_source")
    if scenecode_source is not None:
        env["SCENECODE_SOURCE_ROOT"] = str(workspace_path(scenecode_source))
    else:
        env.setdefault("SCENECODE_SOURCE_ROOT", str(workspace_path("code/SceneCode-main")))
    env["SCENECODE_RUNTIME_ROOT"] = str(workspace_path(
        model.get("scenecode_runtime", "runtime/scenecode")
    ))
    ffmpeg = workspace_path(model.get("ffmpeg", "runtime/ffmpeg"))
    if ffmpeg.exists():
        env.setdefault("WO_FFMPEG", str(ffmpeg))
    for key, name in (("robotwin_repo", "WO_ROBOTWIN_REPO"),
                      ("lingbot_repo", "WO_LINGBOT_REPO"),
                      ("robotwin_python", "WO_ROBOTWIN_PYTHON"),
                      ("lingbot_python", "WO_LINGBOT_PYTHON")):
        if model.get(key):
            env[name] = str(workspace_path(model[key]))
    scoring_gpu = score_gpu
    env["WO_SCORE_GPU"] = str(scoring_gpu)
    env["WO_SCORE_MODE"] = score_mode(profile)
    env["WO_EVAL_CONFIG"] = profile["_config_path"]
    return env


def job_defaults(kind: str, args: dict, profile: dict | None = None) -> dict:
    profile = load_profile() if profile is None else profile
    result = dict(args)
    if profile is None or kind not in {"episode", "score"}:
        return result
    model = profile["tested_model"]
    requested = result.get("model")
    if requested and requested != model["adapter"]:
        raise ValueError(f"configured tested model is {model['adapter']!r}, got {requested!r}")
    result["model"] = model["adapter"]
    result.setdefault("gpu", resolve_gpu(profile, "scoring" if kind == "score" else "tested_model"))
    return result


def preflight(profile: dict, env: dict[str, str] | None = None) -> dict:
    env = os.environ if env is None else env
    applied = apply_environment(dict(env), profile)
    model = profile["tested_model"]
    gpu = resolve_gpu(profile, "tested_model")
    checkpoint = Path(applied["WO_TESTED_MODEL_CHECKPOINT"])
    scenecode_source = Path(applied["SCENECODE_SOURCE_ROOT"])
    scenecode_runtime = Path(applied["SCENECODE_RUNTIME_ROOT"])
    info = gpu_info(gpu)
    return {
        "config": profile["_config_path"],
        "agent_api": {
            "base_url_configured": bool(applied.get("OPENAI_BASE_URL")),
            "model": applied["WO_AGENT_MODEL"],
            "api_key_present": bool(applied.get("OPENAI_API_KEY")),
        },
        "tested_model": {"adapter": model["adapter"], "checkpoint_exists": checkpoint.exists(),
                         "scenecode_source": str(scenecode_source),
                         "scenecode_source_exists": (scenecode_source / "main.py").is_file(),
                         "scenecode_runtime": str(scenecode_runtime),
                         "scenecode_runtime_status": scenecode_runtime_status(scenecode_runtime),
                         "gpu": gpu},
        "gpu": info,
        "score_mode": score_mode(profile, info=info),
        "resources": resource_limits(profile, info=info),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("command", choices=("check",))
    args = parser.parse_args()
    profile = load_profile(args.config, required=True)
    env = os.environ
    env.setdefault("WO_WORKSPACE", str(Path(profile["workspace_root"]).expanduser().resolve()))
    load_private_environment(env, Path(env["WO_WORKSPACE"]) / ".secrets/scenecode.env")
    print(json.dumps(preflight(profile), indent=2))


if __name__ == "__main__":
    main()
