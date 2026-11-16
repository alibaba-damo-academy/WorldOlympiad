"""Registered pipeline stages.  Each stage = command + resources + artifact check.

Layout of a run (outputs/wo_runs/<run>/):
    tasks/<case>.json                      wo_task_spec.v1 (with "generation")
    references/<case>/                     expert reference + replay videos
    <case>/<model>_s<seed>/episode.json    native model episode
    <case>/<model>_s<seed>/case/score.json three-track judge scores
    results.json / results.md              report
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..egl import configure_egl_vendor
from ..scene import WORKSPACE
from .config import apply_environment, load_private_environment, load_profile, resource_limits

R = Path(os.environ.get("WO_CODE_ROOT", str(WORKSPACE / "code/WorldOlympiad-main"))).expanduser().resolve()
RUNS = WORKSPACE / "outputs/wo_runs"
PY = str(WORKSPACE / "envs/autoresearch/bin/python")
_PROFILE = load_profile()
# Stage registration is safe on CPU-only hosts and must not query live hardware.
_RESOURCES = resource_limits(_PROFILE, inventory=[])
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

# model adapters available to the "episode" stage: name -> python module
# Native execution controls reviewed for GR00T (#630) and FastWAM (#632).
# These receipts establish execution, not successful task manipulation.
MODELS = {"cosmos3": "wo.models.cosmos", "lingbot_va": "wo.models.lingbot_va",
          "gr00t": "wo.models.gr00t", "fastwam": "wo.models.fastwam", "longwam": "wo.models.longwam"}

# Experimental adapters may execute a bound native positive control, but are
# not eligible for matrix scheduling until the evidence has been reviewed.
CONTROL_MODELS = {}


def episode_adapter(args: dict) -> str:
    model = args["model"]
    if model in MODELS:
        return MODELS[model]
    if args.get("control_only") is True and model in CONTROL_MODELS:
        return CONTROL_MODELS[model]
    raise ValueError(f"no adapter for model {model!r}; native control requires control_only=true")


def _safe(value: str, what: str) -> str:
    if not isinstance(value, str) or not SAFE.match(value):
        raise ValueError(f"unsafe {what}: {value!r}")
    return value


def run_dir(run: str) -> Path:
    return RUNS / _safe(run, "run")


def bind_scene_source_profile(args: dict) -> dict:
    """Bind a prepared same-run SceneCode profile, if the launcher supplied one."""
    configured = os.environ.get("WO_SCENECODE_SOURCE_CONFIG_ROOT")
    if not configured:
        return args
    if "run" not in args:
        raise ValueError("SceneCode profile binding requires a run id")
    expected = Path(configured).expanduser().resolve()
    run_root = run_dir(args["run"]).resolve()
    try:
        expected.relative_to(run_root)
    except ValueError as exc:
        raise ValueError("configured SceneCode profile must live inside this run") from exc
    supplied = args.get("source_config_root")
    if supplied and Path(supplied).expanduser().resolve() != expected:
        raise ValueError("submitted SceneCode profile differs from the launcher profile")
    bound = dict(args)
    bound["source_config_root"] = str(expected)
    return bound


@dataclass
class Stage:
    name: str
    required: tuple[str, ...]
    build: Callable[[dict, int], list[str]]
    check: Callable[[dict], dict]
    gpu_gb: float = 0.0
    timeout_s: int = 3600
    cwd: Path = R
    extra_env: dict = field(default_factory=dict)

    def validate(self, args: dict) -> None:
        missing = [k for k in self.required if k not in args]
        if missing:
            raise ValueError(f"{self.name} needs {missing}")
        for k in ("run", "case", "model", "stock_task"):
            if k in args:
                _safe(args[k], k)
        if self.name == "policy_check" and args.get("model") not in {"lingbot_va", "fastwam", "longwam"}:
            raise ValueError("stock RoboTwin control requires lingbot_va, fastwam or longwam")
        if self.name == "score":
            score_mode = args.get("score_mode", os.environ.get("WO_SCORE_MODE", "full"))
            if score_mode not in {"full", "split"}:
                raise ValueError("score_mode must be full or split")
        if "model" in args:
            if self.name == "episode":
                episode_adapter(args)
            elif args["model"] not in MODELS and not (self.name == "policy_check" and args["model"] in CONTROL_MODELS):
                raise ValueError(f"no adapter for model {args['model']!r}; available: {sorted(MODELS)}")

    def command(self, args: dict, job_id: int) -> list[str]:
        return [str(x) for x in self.build(args, job_id)]

    def env(self, args: dict | None = None) -> dict:
        env = dict(os.environ, PYTHONPATH=str(R), **self.extra_env)
        load_private_environment(env)
        args = args or {}
        profile = load_profile() or _PROFILE
        phase = "worker"
        if self.name in {"scene", "reference", "scene_check", "expert_check"}:
            phase = "preparation"
        elif self.name in {"policy_check", "episode"}:
            phase = "episode"
            if profile is not None:
                profile = dict(profile)
                model = dict(profile["tested_model"])
                if args.get("gpu") is not None:
                    model["gpu"] = int(args["gpu"])
                if args.get("simulator_gpu") is not None:
                    model["simulator_gpu"] = int(args["simulator_gpu"])
                if args.get("policy_gpus") is not None:
                    model["policy_gpus"] = list(map(int, args["policy_gpus"]))
                profile["tested_model"] = model
        elif self.name == "score":
            phase = "scoring"
            if profile is not None:
                profile = dict(profile)
                scoring = dict(profile.get("scoring", {}))
                if args.get("gpu") is not None:
                    scoring["gpu"] = int(args["gpu"])
                if args.get("compute_gpu") is not None:
                    scoring["compute_gpu"] = int(args["compute_gpu"])
                if args.get("score_mode") is not None:
                    scoring["mode"] = args["score_mode"]
                profile["scoring"] = scoring
        apply_environment(env, profile, phase=phase)
        env["WO_CODE_ROOT"] = str(R)
        env["WORLD_REPO"] = str(R)
        configure_egl_vendor(env)
        # the root filesystem is only 30 GB: all temp files go to the data disk
        tmp = WORKSPACE / "tmp/jobs"
        tmp.mkdir(parents=True, exist_ok=True)
        env.update(TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp))
        if self.name in {"reference", "expert_check"}:
            env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        return env

    def collect(self, args: dict) -> dict:
        return self.check(args)


# ------------------------------------------------------------------ stages
def _task(args) -> Path:
    return run_dir(args["run"]) / "tasks" / f"{args['case']}.json"


def _ref(args) -> Path:
    return run_dir(args["run"]) / "references" / args["case"]


def _ep(args) -> Path:
    return run_dir(args["run"]) / args["case"] / f"{args['model']}_s{int(args.get('seed', 0))}"


def _episode_window_args(args: dict) -> list[str]:
    # FastWAM runs to its stable native success predicate by default.
    # A fixed window remains available only for explicit protocol controls.
    if args.get("model") in {"fastwam", "longwam"} and args.get("fixed_window") is not True:
        return []
    if args.get("window_s") is None:
        return []
    return ["--window-s", str(args["window_s"])]


def _check_scene(args):
    spec = json.loads(_task(args).read_text())
    gen = Path(spec.get("generation", ""))
    if not (gen / "task_inventory_certificate.json").is_file():
        raise FileNotFoundError(f"no SceneCode generation for {args['case']}")
    return {"generation": str(gen)}


def _check_reference(args):
    cert = json.loads((_ref(args) / "certificate.json").read_text())
    last = cert["steps"][-1] if cert["steps"] else {}
    if not cert["success"]:
        raise RuntimeError(f"expert failed at {last.get('stage')}: {last.get('detail')}")
    from ..eval.reference_binding import verify_task
    verify_task(_ref(args), _task(args))
    if not (_ref(args) / "replay/exterior_left.mp4").is_file():
        raise FileNotFoundError("reference rendered no exterior_left video")
    return {"success": True, "frames": cert["frame_count"], "variants": cert.get("embodiment_variants")}


def _check_episode(args):
    from ..eval.reference_binding import verify_episode_binding
    from .episode_stage import verify_evaluation_view
    binding = verify_episode_binding(_ep(args), _ref(args))
    if binding["status"] != "verified":
        raise ValueError("model stage produced no complete reference binding")
    view = verify_evaluation_view(_ep(args), _ref(args))
    ep = json.loads((_ep(args) / "episode.json").read_text())
    return {"task_success": ep.get("task_success"), "termination": ep.get("termination_reason"),
            "view": view["view"], "frames": view["frame_count"],
            "video_sha256": view["video_sha256"]}


def _check_score(args):
    from .score_stage import valid_receipt
    case = _ep(args) / "case"
    if not valid_receipt(case):
        raise ValueError("scoring stage produced no valid three-track receipt")
    record = json.loads((case / "worldolympiad_case.json").read_text())
    if (record.get("view_contract") != "matched_single_physical_view_v2"
            or record.get("reference_view_id") != record.get("generated_view_id")):
        raise ValueError("score case violates the baseline same-view contract")
    s = json.loads((case / "score.json").read_text())
    result = {"physical": (s.get("physical") or {}).get("score"),
              "geometry": (s.get("three_d") or {}).get("final_score_normalized"),
              "interaction": (s.get("interaction") or {}).get("score")}
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in result.values()):
        raise ValueError("one or more required three-track measurements are missing")
    return result


def _check_showcase(args):
    ref = _ref(args)
    plan = json.loads((ref / "showcase_plan.json").read_text())
    video = ref / "showcase/showcase.mp4"
    if not video.is_file() or not video.stat().st_size:
        raise FileNotFoundError("no reference showcase video")
    if len(plan["frames"]) != len(json.loads((ref / "trajectory.json").read_text())["frames"]):
        raise ValueError("showcase did not preserve the full reference trajectory")
    return {"video": str(video), "presentation_only": True, "frames": len(plan["frames"])}


def _check_report(args):
    root = run_dir(args["run"])
    document = json.loads((root / "results.json").read_text())
    v3_path = root / "results_v3.json"
    v3 = json.loads(v3_path.read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    cases, models, seeds = manifest.get("cases"), manifest.get("models"), manifest.get("seeds")
    if not all(isinstance(values, list) and values for values in (cases, models, seeds)):
        raise ValueError("report acceptance requires explicit non-empty case/model/seed scope")
    expected = {(case, model, int(seed)) for case in cases for model in models for seed in seeds}
    rows = document.get("episodes", [])
    rows_v3 = v3.get("episodes", [])
    key = lambda row: (row.get("case_id"), row.get("model"), row.get("seed"))
    selected = {key(row): row for row in rows if key(row) in expected}
    selected_v3 = {key(row): row for row in rows_v3 if key(row) in expected}
    if set(selected) != expected or set(selected_v3) != expected:
        raise ValueError("report is missing a declared case/model/seed result row")
    coverage = v3.get("coverage") or document.get("coverage") or {}
    if (coverage.get("matrix_complete") is not True or coverage.get("missing_references")
            or coverage.get("pending_matrix_entries")):
        raise ValueError("report coverage is incomplete or has pending reference/episode entries")
    for row_key in sorted(expected):
        row, final = selected[row_key], selected_v3[row_key]
        score = row.get("worldolympiad2_score")
        if (row.get("status") != "complete" or row.get("pending_tracks")
                or not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError(f"v2 result is incomplete for {row_key}")
        tracks = row.get("tracks") or {}
        for name in ("physical", "geometry", "interaction"):
            track = tracks.get(name) or {}
            if track.get("status") != "scored" or not isinstance(track.get("score"), (int, float)):
                raise ValueError(f"required v2 {name} track is incomplete for {row_key}")
        cae = row.get("cae_details") or {}
        continuous = (tracks.get("interaction") or {}).get("submetrics", {}).get("continuous_action") or {}
        if (cae.get("status") != "scored" or cae.get("source") != "simulator_execution"
                or continuous.get("status") != "scored"
                or not isinstance(cae.get("score"), (int, float)) or not math.isfinite(cae["score"])):
            raise ValueError(f"baseline native CAE track is incomplete for {row_key}")
        if (final.get("status") != "complete" or final.get("pending_tracks")
                or not isinstance(final.get("worldolympiad2_score"), (int, float))
                or not math.isfinite(final["worldolympiad2_score"])):
            raise ValueError(f"report-v3 has no complete numeric result for {row_key}")
        v3_tracks = final.get("tracks") or {}
        for name in ("physical", "geometry", "interaction"):
            if (v3_tracks.get(name) or {}).get("status") != "scored":
                raise ValueError(f"report-v3 {name} track is incomplete for {row_key}")
        if ((v3_tracks.get("geometry") or {}).get("submetrics", {}).get("action_trajectory") or {}).get("status") != "scored":
            raise ValueError(f"report-v3 CAE action-trajectory component is incomplete for {row_key}")
        if ((v3_tracks.get("interaction") or {}).get("submetrics", {}).get("action_speed") or {}).get("status") != "scored":
            raise ValueError(f"report-v3 CAE action-speed component is incomplete for {row_key}")
        case_id, model, seed = row_key
        episode_dir = root / case_id / f"{model}_s{seed}"
        case_path = episode_dir / "case"
        if not (episode_dir / "native_run/simulation/trajectory.json").is_file():
            raise FileNotFoundError(f"native trajectory is missing for {row_key}")
        from .score_stage import valid_receipt
        if not valid_receipt(case_path):
            raise ValueError(f"score receipt is missing or stale for {row_key}")
        case_manifest = json.loads((case_path / "worldolympiad_case.json").read_text())
        if (case_manifest.get("view_contract") != "matched_single_physical_view_v2"
                or case_manifest.get("reference_view_id") != case_manifest.get("generated_view_id")):
            raise ValueError(f"score case has a mismatched camera contract for {row_key}")
        from ..eval.reference_binding import verify_binding, verify_episode_binding
        from .episode_stage import verify_evaluation_view
        reference = Path(json.loads((episode_dir / "episode.json").read_text())["reference"])
        if verify_episode_binding(episode_dir, reference)["status"] != "verified":
            raise ValueError(f"native episode is not bound to its expert reference for {row_key}")
        verify_evaluation_view(episode_dir, reference)
        verify_binding(episode_dir, reference, case=case_path)
    return {"episodes": len(expected), "results_v3": str(v3_path), "complete": True}


def _expert_check_dir(args):
    base = run_dir(args["run"]) / "expert_checks" / args["case"]
    if args.get("canonical_initialization"):
        evidence_id = args.get("evidence_id", "")
        if not SAFE.fullmatch(evidence_id):
            raise ValueError("canonical initialization requires safe unique evidence_id")
        return base / evidence_id
    return base


def _expert_check_command(args, job):
    if args.get("canonical_initialization"):
        return [PY, "-m", "wo.expert.initialization_check", "--task", _task(args),
                "--ref", _ref(args), "--out", _expert_check_dir(args)]
    return [PY, "-m", "wo.expert.run", "--task", _task(args),
            "--out", _expert_check_dir(args), "--layout", args.get("layout", "droid")]


def _check_expert_check(args):
    if args.get("canonical_initialization"):
        evidence = json.loads((_expert_check_dir(args) / "initialization.json").read_text())
        if not evidence.get("compatible"):
            raise RuntimeError("canonical initialization mismatch; supplement not publishable")
        return {"initialization_compatible": True, "control_only": True,
                "task_success_claim": False}
    cert = json.loads((_expert_check_dir(args) / "certificate.json").read_text())
    if not cert.get("success"):
        last = (cert.get("steps") or [{}])[-1]
        raise RuntimeError(f"expert control failed at {last.get('stage')}: {last.get('detail')}")
    return {"success": True, "frames": cert.get("frame_count"), "control_only": True}


def _scene_check_dir(args):
    return run_dir(args["run"]) / "scene_checks" / args["case"]


def _check_native_scene(args):
    result = json.loads((_scene_check_dir(args) / "scene_ready.json").read_text())
    if not result.get("scene_ready") or result["framing"].get("outside"):
        raise ValueError("native scene/camera checks did not pass")
    return {"scene_ready":True,"required_visible":result["required_visible"]}


def _policy_check_dir(args):
    return run_dir(args["run"]) / "policy_checks" / args["case"] / args["model"]


def _check_policy(args):
    result = json.loads((_policy_check_dir(args) / "policy_check.json").read_text())
    return {"control_only": True, "task_success": result["native_result"]["task_success"]}


STAGES: dict[str, Stage] = {
    "showcase": Stage(
        "showcase", ("run", "case"),
        lambda a, j: [PY, "-m", "wo.render", "--ref", _ref(a), "--out", "showcase", "--views", "showcase"],
        _check_showcase, gpu_gb=2, timeout_s=7200),
    "policy_check": Stage(
        "policy_check", ("run", "case", "model", "stock_task"),
        lambda a, j: [PY, "-m", "wo.models.policy_check", "--model", a["model"],
                      "--stock-task", a["stock_task"], "--out", _policy_check_dir(a),
                      "--window-s", str(a.get("window_s", 60)), "--seed", str(a.get("seed", 0)),
                      "--port", str(28000 + j % 1000),
                      *(["--prompt", a["prompt"]] if a.get("prompt") else [])],
        _check_policy, gpu_gb=_RESOURCES["policy_check"], timeout_s=1800),
    "scene_check": Stage(
        "scene_check", ("run", "case"),
        lambda a, j: [PY, "-m", "wo.models.scene_check", "--ref", _ref(a), "--out", _scene_check_dir(a),
                      "--task", _task(a)],
        _check_native_scene, gpu_gb=_RESOURCES["scene_check"], timeout_s=600),
    "expert_check": Stage(
        "expert_check", ("run", "case"),
        _expert_check_command,
        _check_expert_check, gpu_gb=0, timeout_s=1800),
    "scene": Stage(
        "scene", ("run", "case"),
        lambda a, j: [PY, "-m", "wo.tasks.scene", "--task", _task(a),
                      *(["--source-config-root", a["source_config_root"]]
                        if a.get("source_config_root") else [])],
        _check_scene, gpu_gb=4, timeout_s=3 * 3600),
    "reference": Stage(
        "reference", ("run", "case"),
        lambda a, j: ["bash", "-c",
                      f"{PY} -m wo.expert.run --task {_task(a)} --out {_ref(a)} "
                      f"--layout {a.get('layout', 'droid')} --require-success && {PY} -m wo.render --ref {_ref(a)}"],
        _check_reference, gpu_gb=2, timeout_s=2 * 3600),
    "episode": Stage(
        "episode", ("run", "case", "model"),
        lambda a, j: [PY, "-m", "wo.agent.episode_stage", "--run-root", run_dir(a["run"]),
                      "--ref", _ref(a), "--out", _ep(a), "--render-gpu-id", "0", "--",
                      PY, "-m", episode_adapter(a), "--ref", _ref(a), "--out", _ep(a),
                      *_episode_window_args(a), "--seed", str(int(a.get("seed", 0))),
                      "--port", str(28000 + j % 1000),
                      *(["--gpu", str(int(a.get("gpu", 0)))] if episode_adapter(a) == "wo.models.lingbot_va" else []),
                      *(["--simulator-gpu", str(int(a.get("simulator_gpu", a.get("gpu", 0))))]
                        if episode_adapter(a) == "wo.models.lingbot_va" else []),
                      *( ["--policy-gpus", *[str(int(gpu)) for gpu in a["policy_gpus"]]]
                        if episode_adapter(a) == "wo.models.lingbot_va" and a.get("policy_gpus") else [])],
        _check_episode, gpu_gb=_RESOURCES["episode"], timeout_s=2 * 3600),
    "score": Stage(
        "score", ("run", "case", "model"),
        # The receipt binds the final score to exact input hashes. score_stage
        # archives an incomplete prior case before rebuilding; it never deletes
        # a receipt or score to force a recomputation.
        lambda a, j: [PY, "-m", "wo.agent.score_stage", "--ref", str(_ref(a)),
                      "--episode", str(_ep(a)), "--out", str(_ep(a) / "case"),
                      "--workspace", str(WORKSPACE), "--gpu-id",
                      str(int(a.get("gpu", os.environ.get("WO_SCORE_GPU", 0)))),
                      "--compute-gpu-id",
                      str(int(a.get("compute_gpu", os.environ.get("WO_SCORE_COMPUTE_GPU",
                                                                  a.get("gpu", 0))))),
                      "--score-mode", a.get("score_mode", os.environ.get("WO_SCORE_MODE", "full"))],
        _check_score, gpu_gb=_RESOURCES["score"], timeout_s=2 * 3600),
    "report": Stage(
        "report", ("run",),
        lambda a, j: [PY, "-m", "wo.eval.report", "--runs", run_dir(a["run"]),
                      "--out", run_dir(a["run"]) / "results.json"],
        _check_report, timeout_s=1800),
}
