#!/usr/bin/env python3
"""Fail-closed source, runtime, API-config and phase-specific preflight."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import sqlite3
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "code/WorldOlympiad-main"
CONFIG = ROOT / "config/two-interface.json"
SECRET_FILE = ROOT / ".secrets/scenecode.env"
BUILDER_SHA256 = "d46198abb1f937ddae474b86f5980219b6d780f937c08eeea1b907a42f48d225"
SAFE_RUN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def fail(message: str, code: int = 2) -> None:
    raise SystemExit(f"preflight blocked: {message}")


def parse_gpu_allowlist(raw: str | None = None) -> list[int]:
    """Parse the user's explicit local NVIDIA device-index allowlist."""
    if raw is None:
        raw = os.environ.get("WOV2_GPU_ALLOWLIST")
    if raw is None or not raw.strip():
        raise ValueError("set WOV2_GPU_ALLOWLIST to comma-separated nvidia-smi GPU indices")
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(not re.fullmatch(r"[0-9]+", part) for part in parts):
        raise ValueError("WOV2_GPU_ALLOWLIST must contain only nonnegative integer indices")
    indices = [int(part) for part in parts]
    if len(set(indices)) != len(indices):
        raise ValueError("WOV2_GPU_ALLOWLIST must not contain duplicate indices")
    return sorted(indices)


def query_gpu_resources(allowlist: list[int] | None = None) -> dict:
    """Query only user-allowlisted NVIDIA GPUs and fail closed on incomplete data."""
    if allowlist is None:
        try:
            allowlist = parse_gpu_allowlist()
        except ValueError as exc:
            fail(str(exc))
    else:
        try:
            allowlist = parse_gpu_allowlist(",".join(str(index) for index in allowlist))
        except ValueError as exc:
            fail(str(exc))
    query_indices = ",".join(map(str, allowlist))
    common = ["nvidia-smi", "-i", query_indices]
    try:
        gpu_result = subprocess.run(
            common + ["--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu",
                      "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        cards = {}
        for row in csv.reader(gpu_result.stdout.splitlines(), skipinitialspace=True):
            if len(row) != 6:
                fail("NVIDIA GPU inventory query returned malformed data")
            index, name, uuid, total, free, utilization = [field.strip() for field in row]
            index = int(index)
            if index not in allowlist:
                fail(f"GPU {index} was returned outside WOV2_GPU_ALLOWLIST")
            cards[uuid] = {"index": index, "name": name, "uuid": uuid,
                           "total_mib": int(total), "free_mib": int(free),
                           "utilization": int(utilization), "compute_apps": [],
                           "compute_app_pids": [], "compute_process_count": 0,
                           "process_query_ok": False}
        if {item["index"] for item in cards.values()} != set(allowlist):
            fail("NVIDIA GPU inventory did not return every requested allowlisted index")
        proc_result = subprocess.run(
            common + ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                      "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        for row in csv.reader(proc_result.stdout.splitlines(), skipinitialspace=True):
            if not row or (len(row) == 1 and "no running processes" in row[0].lower()):
                continue
            if len(row) != 4:
                fail("compute-app PID query returned malformed data")
            uuid, pid, name, used = [field.strip() for field in row]
            if uuid not in cards:
                fail("compute-app query returned a device outside WOV2_GPU_ALLOWLIST")
            try:
                pid_int = int(pid)
            except ValueError:
                fail("compute-app PID query returned a nonnumeric PID")
            app = {"pid": pid_int, "process_name": Path(name).name or name,
                   "used_memory_mib": int(used) if used.isdigit() else None}
            cards[uuid]["compute_apps"].append(app)
            cards[uuid]["compute_app_pids"].append(pid_int)
        for card in cards.values():
            card["compute_app_pids"].sort()
            card["compute_process_count"] = len(card["compute_app_pids"])
            card["process_query_ok"] = True
        return {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "queried_gpu_indices": allowlist,
                "cards": sorted(cards.values(), key=lambda item: item["index"])}
    except SystemExit:
        raise
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        fail(f"allowlisted NVIDIA GPU resource query failed closed ({type(exc).__name__})")


DEFAULT_NVIDIA_EGL_VENDOR = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")


def renderer_config_warning(phase: str) -> str | None:
    if phase not in {"preparation", "evaluation"}:
        return None
    configured = os.environ.get("WOV2_EGL_VENDOR_JSON")
    path = Path(configured).expanduser() if configured else DEFAULT_NVIDIA_EGL_VENDOR
    if path.is_file() and os.access(path, os.R_OK):
        return None
    return f"NVIDIA EGL vendor JSON is unavailable at {path}; replay will use host EGL discovery."


def _available(card: dict, min_free_mib: int) -> bool:
    pids = card.get("compute_app_pids", card.get("compute_pids"))
    process_count = card.get("compute_process_count")
    return (card.get("process_query_ok") is True
            and pids == []
            and process_count == 0
            and card.get("utilization", 101) <= 5
            and card.get("free_mib", 0) >= min_free_mib)


def assess_resources(snapshot: dict, phase: str, profile: dict) -> dict:
    """Summarize the topology this locked adapter can use; source preflight is final authority."""
    model = profile["tested_model"]
    adapter = model["adapter"]
    cards = snapshot["cards"]
    is_small_lingbot = adapter == "lingbot_va" and cards and max(c["total_mib"] for c in cards) <= 40 * 1024
    if phase == "scoring":
        sys.path.insert(0, str(WORLD))
        from wo.agent.config import resolve_score_gpus, score_mode
        inventory = [{"index": c["index"], "uuid": c["uuid"],
                      "total_mib": c["total_mib"], "free_mib": c["free_mib"],
                      "utilization": c["utilization"],
                      "compute_process_count": c["compute_process_count"],
                      "compute_pids": list(c["compute_app_pids"]),
                      "process_query_ok": c["process_query_ok"]} for c in cards]
        mode = score_mode(profile, inventory=inventory)
        threshold = 48 * 1024 if mode == "full" else 25_600
        eligible_score = [c for c in inventory if _available(c, threshold)]
        try:
            chosen = resolve_score_gpus(profile, inventory=inventory)
            ready, reason = True, None
        except (RuntimeError, ValueError) as exc:
            chosen, ready, reason = None, False, str(exc)
        deficits = [{"index": c["index"], "free_mib": c["free_mib"],
                     "score_memory_shortfall_mib": max(0, threshold-c["free_mib"]),
                     "compute_app_pids": c["compute_pids"],
                     "utilization": c["utilization"],
                     "eligible_for_scoring": _available(c, threshold)} for c in inventory]
        return {"phase": phase, "adapter": adapter, "ready": ready,
                "scoring_mode": mode, "required_scoring_gpu_count": 1 if mode == "full" else 2,
                "score_memory_threshold_mib": threshold,
                "eligible_scoring_gpu_indices": [c["index"] for c in eligible_score],
                "selected_vlm_gpu_index": chosen["gpu"] if chosen else None,
                "selected_compute_gpu_index": chosen["compute_gpu"] if chosen else None,
                "card_deficits": deficits, "blocking_reason": reason}

    if phase == "preparation":
        sys.path.insert(0, str(WORLD))
        from wo.agent.config import PREPARATION_MIN_FREE_MIB, resolve_preparation_gpu
        inventory = [{"index": c["index"], "uuid": c["uuid"],
                      "total_mib": c["total_mib"], "free_mib": c["free_mib"],
                      "utilization": c["utilization"],
                      "compute_process_count": c["compute_process_count"],
                      "compute_pids": list(c["compute_app_pids"]),
                      "process_query_ok": c["process_query_ok"]} for c in cards]
        eligible = [c for c in inventory if _available(c, PREPARATION_MIN_FREE_MIB)]
        try:
            selected = resolve_preparation_gpu(profile, inventory=inventory)
            ready, reason = True, None
        except (RuntimeError, ValueError) as exc:
            selected, ready, reason = None, False, str(exc)
        deficits = [{"index": c["index"], "free_mib": c["free_mib"],
                     "preparation_memory_shortfall_mib": max(0, PREPARATION_MIN_FREE_MIB-c["free_mib"]),
                     "compute_app_pids": c["compute_pids"], "utilization": c["utilization"],
                     "eligible_for_preparation": _available(c, PREPARATION_MIN_FREE_MIB)} for c in inventory]
        return {"phase": phase, "adapter": adapter, "ready": ready,
                "required_preparation_gpu_count": 1,
                "preparation_memory_threshold_mib": PREPARATION_MIN_FREE_MIB,
                "eligible_preparation_gpu_indices": [c["index"] for c in eligible],
                "selected_preparation_gpu_index": selected,
                "card_deficits": deficits, "blocking_reason": reason}

    policy_min = 30 * 1024 if is_small_lingbot else 25_600
    sim_min = 25_600
    policy_candidates = [c for c in cards if _available(c, policy_min)]
    simulator_candidates = [c for c in cards if _available(c, sim_min)]
    requested_gpu = model.get("gpu", "auto")
    if requested_gpu == "auto":
        model_card = min(policy_candidates, key=lambda c: (-c["free_mib"], c["index"])) if policy_candidates else None
    else:
        model_card = next((c for c in cards if c["index"] == int(requested_gpu)), None)
    selected_policy = []
    selected_sim = None
    if phase == "preparation":
        ready = model_card is not None and _available(model_card, policy_min)
        selected_policy = [model_card["index"]] if ready else []
        required_policy_count = 1
        reason = None if ready else "preparation requires one idle allowlisted NVIDIA GPU with the adapter's configured free-memory threshold"
    elif is_small_lingbot:
        if model_card is not None and _available(model_card, policy_min):
            requested_sim = model.get("simulator_gpu", "auto")
            sim_pool = [c for c in cards if c["index"] != model_card["index"] and _available(c, sim_min)]
            if requested_sim != "auto":
                sim_card = next((c for c in cards if c["index"] == int(requested_sim)), None)
                if sim_card and _available(sim_card, sim_min): sim_pool = [sim_card]
                else: sim_pool = []
            sim_card = min(sim_pool, key=lambda c: (-c["free_mib"], c["index"])) if sim_pool else None
            if sim_card:
                requested_policy = model.get("policy_gpus", "auto")
                if isinstance(requested_policy, list):
                    policy_pool = [next((c for c in cards if c["index"] == int(g)), None) for g in requested_policy]
                    policy_pool = [c for c in policy_pool if c and _available(c, policy_min)]
                else:
                    policy_pool = [c for c in cards if c["index"] not in {model_card["index"], sim_card["index"]}
                                   and _available(c, policy_min)]
                    policy_pool = [model_card] + sorted(policy_pool, key=lambda c: (-c["free_mib"], c["index"]))
                if len(policy_pool) >= 2 and all(c["index"] != sim_card["index"] for c in policy_pool):
                    selected_policy = [c["index"] for c in policy_pool]
                    selected_sim = sim_card["index"]
        ready = bool(selected_policy and selected_sim is not None)
        required_policy_count = 2
        reason = None if ready else "LingBot on GPUs up to 40 GiB needs two policy GPUs (each >=30,720 MiB free) plus a separate simulator GPU (>=25,600 MiB free); all must be allowlisted, idle, and <=5% utilized"
    else:
        ready = model_card is not None and _available(model_card, policy_min)
        selected_policy = [model_card["index"]] if ready else []
        required_policy_count = 1
        reason = None if ready else "evaluation requires at least one idle eligible allowlisted NVIDIA GPU"
    deficits = []
    for card in cards:
        min_required = policy_min if phase == "preparation" else sim_min
        deficits.append({"index": card["index"], "free_mib": card["free_mib"],
                         "policy_memory_shortfall_mib": max(0, policy_min-card["free_mib"]),
                         "simulator_memory_shortfall_mib": max(0, sim_min-card["free_mib"]),
                         "compute_app_pids": card["compute_app_pids"],
                         "utilization": card["utilization"],
                         "eligible_for_policy": _available(card, policy_min),
                         "eligible_for_simulator": _available(card, sim_min)})
    return {"phase": phase, "adapter": adapter, "ready": ready,
            "required_policy_gpu_count": required_policy_count,
            "policy_memory_threshold_mib": policy_min,
            "simulator_memory_threshold_mib": sim_min,
            "eligible_policy_gpu_indices": [c["index"] for c in policy_candidates],
            "eligible_simulator_gpu_indices": [c["index"] for c in simulator_candidates],
            "selected_policy_gpu_indices": selected_policy,
            "selected_simulator_gpu_index": selected_sim,
            "card_deficits": deficits, "blocking_reason": reason}


def write_resource_report(run_id: str | None, phase: str, snapshot: dict | None,
                          assessment: dict | None, status: str, reason: str | None,
                          pipeline_result: dict | None = None) -> Path:
    directory=ROOT/"outputs/evaluator/resource_preflight"
    directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    os.chmod(directory,0o700)
    payload={"schema_version":1,"status":status,"run":run_id,"phase":phase,
             "recorded_at_utc":datetime.now(timezone.utc).isoformat(),
             "resource_snapshot":snapshot,"assessment":assessment,"reason":reason,
             "pipeline_result":pipeline_result}
    target=directory/f"{phase}-{time.time_ns()}.json"
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"w",encoding="utf-8") as stream:
        json.dump(payload,stream,indent=2);stream.write("\n")
    return target


def run(args: list[str], *, env=None, timeout=120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        fail(f"{args[0]} could not complete ({type(exc).__name__})")


def config_data() -> dict:
    if not CONFIG.is_file():
        fail("config/two-interface.json is missing; configure the two interfaces first")
    try:
        value = json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"two-interface config is not valid JSON ({type(exc).__name__})")
    return value


def lock_checks() -> None:
    # Both manifest modes scan source bundles on cold/shared filesystems.
    check = run([sys.executable, str(ROOT / "scripts/setup_sources.py"), "--check-only"], timeout=1800)
    if check.returncode:
        fail("source lock is incomplete or invalid; no checkout was created")
    lock = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))
    for item in lock["sources"]:
        repo = ROOT / item["path"]
        if not repo.exists():
            fail(f"locked source checkout is missing: {item['name']}")
    # A cold full-bundle audit can take well over two minutes on shared NFS.
    check = run([sys.executable, str(ROOT / "scripts/setup_sources.py"), "--verify-only"], timeout=1800)
    if check.returncode:
        fail("source checkout does not match the lock or has unapproved changes")


def builder_check() -> None:
    path = ROOT / "code/scenecode_pybullet_minimal/scripts/build_robolab_droid_urdf.py"
    if not path.is_file():
        fail(f"DROID URDF builder is missing: {path.relative_to(ROOT)}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BUILDER_SHA256:
        fail("DROID URDF builder checksum differs from the locked WorldOlympiad contract")


def safe_api_file() -> Path:
    target=Path(os.environ.get("WO_AGENT_ENV_FILE", str(SECRET_FILE))).expanduser()
    if target.exists():
        if not target.is_file(): fail("configured Agent API environment path is not a regular file")
        mode=stat.S_IMODE(target.stat().st_mode)
        if mode!=0o600: fail("configured Agent API environment file must have mode 0600")
    return target


def process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return True


def queue_check(run_id: str | None, allow_same_run: bool) -> None:
    db = ROOT / "outputs/wo_agent/jobs.sqlite3"
    if not db.is_file():
        return
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=3)
        rows = con.execute("select id,kind,status,pid,args from jobs where status in ('queued','running') order by id").fetchall()
    except sqlite3.Error as exc:
        fail(f"cannot read the application job queue ({type(exc).__name__})")
    finally:
        try:
            con.close()
        except Exception:
            pass
    foreign, live = [], []
    for job_id, kind, status, pid, args in rows:
        try:
            job_run = json.loads(args).get("run")
        except Exception:
            job_run = None
        if job_run != run_id or not allow_same_run:
            foreign.append(f"#{job_id} {kind} {status}")
        if status == "running" and process_alive(pid):
            live.append(f"#{job_id} {kind}")
    if foreign:
        fail("application queue has unrelated or unapproved queued/running jobs: " + ", ".join(foreign[:12]))
    if live:
        fail("jobs are still running; do not start a second launcher: " + ", ".join(live[:12]))


def import_probe(label: str, python: Path, modules: list[str], *, extra_path: Path | None = None,
                 expected: tuple[int, int] | None = None) -> None:
    if not python.is_file() or not os.access(python, os.X_OK):
        fail(f"{label} Python interpreter is missing or not executable: {python}")
    script = r'''import importlib, json, sys
modules=json.loads(sys.argv[1]); errors=[]
for name in modules:
    try: importlib.import_module(name)
    except Exception as exc: errors.append(name + ":" + type(exc).__name__)
print("IMPORTS_OK" if not errors else "IMPORTS_FAIL " + ",".join(errors))
if errors: raise SystemExit(1)
'''
    env = os.environ.copy()
    if extra_path:
        env["PYTHONPATH"] = str(extra_path) + os.pathsep + env.get("PYTHONPATH", "")
    # Shared runtime trees are on NFS; cold imports can exceed fifteen minutes.
    p = run([str(python), "-c", script, json.dumps(modules)], env=env, timeout=1800)
    if p.returncode:
        fail(f"{label} import probe failed: {p.stdout.strip() or p.stderr.strip()}")
    if expected:
        version = run([str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], env=env)
        if version.returncode or version.stdout.strip() != f"{expected[0]}.{expected[1]}":
            fail(f"{label} requires Python {expected[0]}.{expected[1]}")
    print(f"PASS: {label} imports and interpreter version")


def runtime_checks(profile: dict | None) -> None:
    model = (profile or {}).get("tested_model", {
        "robotwin_repo": "runtime/RoboTwin", "lingbot_repo": "runtime/lingbot-va",
        "robotwin_python": "envs/robotwin/bin/python", "lingbot_python": "envs/lingbot/bin/python",
        "scenecode_source": "code/SceneCode", "scenecode_runtime": "runtime/scenecode",
    })
    repo_keys = ("robotwin_repo", "lingbot_repo")
    for key in repo_keys:
        value = Path(model[key]).expanduser()
        if not value.is_absolute(): value = ROOT / value
        if not value.is_dir(): fail(f"configured source path is missing: {key}")
    scene_source = Path(model["scenecode_source"]).expanduser()
    if not scene_source.is_absolute(): scene_source = ROOT / scene_source
    if not (scene_source / "main.py").is_file(): fail("SceneCode source main.py is missing")
    scene_runtime = Path(model["scenecode_runtime"]).expanduser()
    if not scene_runtime.is_absolute(): scene_runtime = ROOT / scene_runtime
    if not (scene_runtime / ".worldolympiad_scenecode_runtime_v1").is_file():
        fail("SceneCode runtime was not staged by the pinned bootstrap")
    robotwin = Path(model["robotwin_repo"]).expanduser()
    if not robotwin.is_absolute(): robotwin = ROOT / robotwin
    for rel in ("scripts/requirements.txt", "assets/embodiments/aloha-agilex/config.yml",
                "assets/objects/objaverse/list.json"):
        if not (robotwin / rel).is_file(): fail(f"RoboTwin required file is missing: {rel}")

    import_probe("autoresearch", ROOT / "envs/autoresearch/bin/python",
                 ["agents", "openai", "cv2", "numpy", "pybullet", "pydantic", "scipy", "trimesh"], expected=(3, 11))
    import_probe("WorldOlympiad evaluator", ROOT / "envs/worldeval/bin/python",
                 ["clip", "cv2", "depth_anything_3", "numpy", "open3d", "timm", "torch", "transformers", "ultralytics"], expected=(3, 11))
    import_probe("SceneCode", ROOT / "envs/scenecode/bin/python",
                 ["agents", "bpy", "coacd", "drake", "hydra", "mathutils", "torch"],
                 extra_path=scene_source, expected=(3, 11))
    robotwin_python = Path(model["robotwin_python"]).expanduser()
    lingbot_python = Path(model["lingbot_python"]).expanduser()
    if not robotwin_python.is_absolute(): robotwin_python = ROOT / robotwin_python
    if not lingbot_python.is_absolute(): lingbot_python = ROOT / lingbot_python
    import_probe("RoboTwin", robotwin_python,
                 ["cv2", "curobo", "gymnasium", "h5py", "mplib", "numpy", "sapien", "torch", "trimesh", "warp"], expected=(3, 10))
    lingbot_source = Path(model["lingbot_repo"]).expanduser()
    if not lingbot_source.is_absolute(): lingbot_source = ROOT / lingbot_source
    import_probe("LingBot-VA", lingbot_python,
                 ["diffusers", "einops", "msgpack", "numpy", "safetensors", "torch", "transformers", "websockets"],
                 extra_path=lingbot_source / "wan_va", expected=(3, 10))


def api_transport(base_url: str) -> None:
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url, headers={"User-Agent": "wov2-evaluator-preflight/1"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except Exception as exc:
        fail(f"Agent API transport did not complete ({type(exc).__name__}); no credential was sent")
    if status >= 500:
        fail(f"Agent API returned HTTP {status} without credentials; transport responded but service failed")
    print(f"Agent API transport responded with HTTP {status}; no key or generation request was sent")


def source_config_check(phase: str) -> dict:
    env = os.environ.copy()
    for name in ("WO_AGENT_API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL", "WO_AGENT_API_MODEL"):
        env.pop(name, None)
    env.update({"WO_WORKSPACE": str(ROOT), "WORLDOLYMPIAD_WORKSPACE": str(ROOT),
                "WO_EVAL_CONFIG": str(CONFIG),
                "WO_AGENT_ENV_FILE": os.environ.get("WO_AGENT_ENV_FILE", str(SECRET_FILE)),
                "WO_RESOURCE_PHASE": phase, "PYTHONPATH": str(WORLD)})
    cmd = [str(ROOT / "envs/autoresearch/bin/python"), "-m", "wo.agent.config",
           "--config", str(CONFIG), "--phase", phase, "check"]
    p = run(cmd, env=env, timeout=30)
    if p.returncode:
        fail("pinned pipeline preflight failed: " + (p.stderr.strip() or p.stdout.strip())[-1800:])
    try:
        result = json.loads(p.stdout)
    except Exception:
        fail("pinned pipeline preflight returned invalid JSON")
    if phase in {"preparation", "evaluation"} and not result.get("agent_api", {}).get("api_key_present"):
        fail("Agent API key is absent from the configured private environment; its value was not inspected")
    if not result.get("tested_model", {}).get("checkpoint_exists"):
        fail("configured tested-model checkpoint does not exist; no download was attempted")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("static", "runtimes", "preparation", "evaluation", "scoring"))
    parser.add_argument("--run")
    parser.add_argument("--allow-same-run-queue", action="store_true")
    parser.add_argument("--api-transport", action="store_true",
                        help="send an unauthenticated GET /models probe; no generation call")
    args = parser.parse_args()
    lock_checks()
    builder_check()
    if args.phase == "static":
        print("PASS: source pins, checkouts and DROID builder contract")
        return 0
    if args.phase == "runtimes":
        runtime_checks(config_data() if CONFIG.is_file() else None)
        return 0
    profile = config_data()
    secret_path=safe_api_file()
    runtime_checks(profile)
    if args.run and not SAFE_RUN.fullmatch(args.run):
        fail("run id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
    queue_check(args.run, args.allow_same_run_queue)
    try:
        snapshot=query_gpu_resources()
    except SystemExit as exc:
        report=write_resource_report(args.run,args.phase,None,None,"blocked",str(exc))
        fail(f"GPU query blocked; evidence: {report}")
    assessment=assess_resources(snapshot,args.phase,profile)
    renderer_warning = renderer_config_warning(args.phase)
    if renderer_warning:
        print(f"Warning: {renderer_warning}", file=sys.stderr)
    if not assessment["ready"]:
        report=write_resource_report(args.run,args.phase,snapshot,assessment,"blocked",assessment["blocking_reason"])
        details=[]
        for card in assessment["card_deficits"]:
            if args.phase == "scoring":
                details.append(f"GPU {card['index']}: free={card['free_mib']} MiB, "
                               f"score shortage={card['score_memory_shortfall_mib']} MiB, "
                               f"PIDs={card['compute_app_pids']}, util={card['utilization']}%")
            elif args.phase == "preparation":
                details.append(f"GPU {card['index']}: free={card['free_mib']} MiB, "
                               f"preparation shortage={card['preparation_memory_shortfall_mib']} MiB, "
                               f"PIDs={card['compute_app_pids']}, util={card['utilization']}%")
            else:
                details.append(f"GPU {card['index']}: free={card['free_mib']} MiB, "
                               f"policy shortage={card['policy_memory_shortfall_mib']} MiB, "
                               f"PIDs={card['compute_app_pids']}, util={card['utilization']}%")
        fail(assessment["blocking_reason"] + "; " + "; ".join(details)
             + f". No job was submitted. Evidence: {report}")
    try:
        result=source_config_check(args.phase)
    except SystemExit as exc:
        report=write_resource_report(args.run,args.phase,snapshot,assessment,"blocked",str(exc))
        fail(f"pinned pipeline preflight blocked; no job was submitted. Evidence: {report}")
    if args.api_transport:
        api_transport(profile["agent_api"]["base_url"])
    data=result.get("tested_model",{})
    pipeline_summary={"agent_model":result.get("agent_api",{}).get("model"),
                      "adapter":data.get("adapter"),"checkpoint_exists":data.get("checkpoint_exists"),
                      "gpu":data.get("gpu"),"policy_gpus":data.get("policy_gpus"),
                      "simulator_gpu":data.get("simulator_gpu"),
                      "episode_resources_deferred":data.get("episode_resources_deferred"),
                      "scoring_mode":assessment.get("scoring_mode"),
                      "selected_vlm_gpu_index":assessment.get("selected_vlm_gpu_index"),
                      "selected_compute_gpu_index":assessment.get("selected_compute_gpu_index")}
    report=write_resource_report(args.run,args.phase,snapshot,assessment,"passed",None,pipeline_summary)
    print(json.dumps({"phase":args.phase,"agent_model":pipeline_summary["agent_model"],
                      "adapter":pipeline_summary["adapter"],"checkpoint_exists":pipeline_summary["checkpoint_exists"],
                      "gpu":pipeline_summary["gpu"],"policy_gpus":pipeline_summary["policy_gpus"],
                      "simulator_gpu":pipeline_summary["simulator_gpu"],
                      "episode_resources_deferred":pipeline_summary["episode_resources_deferred"],
                      "scoring_mode":pipeline_summary["scoring_mode"],
                      "selected_vlm_gpu_index":pipeline_summary["selected_vlm_gpu_index"],
                      "selected_compute_gpu_index":pipeline_summary["selected_compute_gpu_index"],
                      "resource_preflight_evidence":str(report),
                      "resource_snapshot":snapshot,"assessment":assessment},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
