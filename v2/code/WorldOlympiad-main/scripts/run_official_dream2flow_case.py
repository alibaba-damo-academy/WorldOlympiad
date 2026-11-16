#!/usr/bin/env python3
"""Run the complete official Dream2Flow CAE chain for one prepared case.

The case must already contain a calibrated reference/imagination pair and a
frame-zero registration. This entry point owns the intermediate flow format,
the released optimizer invocation, and the final report, so callers do not
need to remember conversion or timing steps.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


FRONTEND_CONTRACT = {
    "mask_erosion_kernel_size": 3,
    "particle_ratio": 0.01,
    "minimum_particles": 64,
    "maximum_particles": 1024,
    "cotracker_visibility_threshold": 0.8,
    "depth_chunk_frames": 128,
    "depth_chunk_overlap": 8,
}


def _run(command: list[str], *, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def _resolve(value: str, *, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _video_info(path: Path) -> tuple[float, int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frames < 2:
        raise ValueError(f"video has invalid timing: {path} fps={fps} frames={frames}")
    return fps, frames


def _verify_flow(path: Path) -> None:
    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        required = {"position", "valid_mask", "initial_pixel_coords"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"official flow archive is missing {missing}: {path}")
        position = payload["position"]
        valid = payload["valid_mask"]
        if position.ndim != 3 or position.shape[-1] != 3:
            raise ValueError(f"invalid official flow position shape: {position.shape}")
        if valid.shape != position.shape[:2] or not bool(valid[0].any()):
            raise ValueError(f"official flow has no observable initial particles: {path}")


def _single_case_report(case_root: Path) -> dict[str, Any]:
    from autoresearch.official_dream2flow_action_report import (
        PROTOCOL,
        SCHEMA_VERSION,
        score_case,
    )

    case = score_case(case_root)
    continuous_action = case["continuous_action"]
    scored = continuous_action is not None
    coverage_failure = bool(case["evaluator_coverage_failure"])
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_protocol": PROTOCOL,
        "policy": {
            "real_score_requirement": "released Dream2Flow trajectory optimizer output for both roles",
            "missing_or_invalid_score": None,
            "partial_observation": "score complete optimizer trajectories; visibility is diagnostic only",
            "unmeasured_semantics": "missing or invalid evidence is null, never a model-quality zero",
            "historical_object_flow_median_sidecars_used": False,
        },
        "case_count": 1,
        "scored_count": int(scored),
        "coverage_failure_count": int(coverage_failure),
        "mean_continuous_action": float(continuous_action) if scored else None,
        "cases": [case],
    }


def run_case(case_root: Path, workspace: Path, *, force_flow: bool,
             pair_validation: Path | None = None,
             diagnostic_pair_validation: Path | None = None,
             spec_path_override: Path | None = None,
             prepare_only: bool = False) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    case_root = case_root.resolve()
    selected_manifest = os.environ.get("WORLDEVAL_SELECTED_MODEL_CASE_SET_MANIFEST")
    if pair_validation is not None and diagnostic_pair_validation is not None:
        raise ValueError("choose one per-case validation scope")
    selected_validation = pair_validation or diagnostic_pair_validation
    case_identity = json.loads((case_root / "worldolympiad_case.json").read_text())
    if (case_identity.get("task_type_id") == "articulated_door_open"
            and case_identity.get("model_id") == "cosmos3_nano_droid"
            and selected_validation is None):
        raise ValueError("articulated Cosmos CAE requires independent per-case validation")
    if selected_validation is not None:
        from autoresearch.io import sha256_file

        validation_path = Path(selected_validation).resolve()
        validation = json.loads(validation_path.read_text())
        pair = json.loads((case_root / "native_pair_evidence.json").read_text())
        case = json.loads((case_root / "worldolympiad_case.json").read_text())
        if (case.get("task_type_id") != "articulated_door_open"
                or validation.get("passed") is not True
                or validation.get("status")
                != "validated_native_model_run_formal_scoring_pending"
                or pair.get("native_run_validation_sha256")
                != sha256_file(validation_path)
                or pair.get("reference_manifest_sha256")
                != validation.get("reference_manifest_sha256")
                or pair.get("reference_view_sha256")
                != sha256_file(case_root / case["reference_video"])
                or pair.get("generated_video_sha256")
                != sha256_file(case_root / case["generated_video"])):
            raise ValueError("diagnostic CAE pair lacks immutable native/reference evidence")
    elif selected_manifest:
        from scripts.build_selected_model_case_set import validate_selected_case_root
        validate_selected_case_root(
            output_root=workspace / "outputs",
            manifest_path=Path(selected_manifest),
            case_root=case_root, expected_model="lingbot_va_robotwin",
        )
    if spec_path_override is not None:
        spec_path = Path(spec_path_override).resolve()
        spec_path.relative_to(case_root)
    else:
        spec_path = case_root / "official_cae_pair_spec.json"
        if not spec_path.is_file():
            spec_path = case_root / "cae_pair_spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"missing case contract: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    for key in ("output_dir", "source_model", "tracked_component_id", "registration"):
        if not isinstance(spec.get(key), str) or not spec[key]:
            raise ValueError(f"cae_pair_spec.json requires {key}")

    world_repo = Path(os.environ.get("WORLD_REPO", workspace / "code/WorldOlympiad-main"))
    d2f_repo = Path(
        os.environ.get(
            "D2F_REPO", workspace / "code/Dream2Flow-main-cae-min64-vis08-v1"
        )
    )
    official_repo = Path(
        os.environ.get(
            "D2F_OFFICIAL_ROOT", workspace / "code/Dream2Flow-main-official-20260809"
        )
    )
    python = Path(os.environ.get("D2F_PYTHON", workspace / "envs/dream2flow/bin/python"))
    required_paths = (world_repo, d2f_repo, official_repo, python)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Dream2Flow runtime resources: " + ", ".join(missing))
    if str(world_repo) not in sys.path:
        sys.path.insert(0, str(world_repo))

    registration = _resolve(spec["registration"], root=case_root)
    if not registration.is_file():
        raise FileNotFoundError(f"missing frame-zero registration: {registration}")

    pair_root = case_root / spec["output_dir"]
    component_id = spec["tracked_component_id"]
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(world_repo), str(d2f_repo), env.get("PYTHONPATH", "")]
            ).strip(os.pathsep),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "SPATRACKER_FRONT_MODEL_PATH": env.get(
                "SPATRACKER_FRONT_MODEL_PATH", str(workspace / "models/SpatialTrackerV2_Front")
            ),
            "SPATRACKER_OFFLINE_MODEL_PATH": env.get(
                "SPATRACKER_OFFLINE_MODEL_PATH", str(workspace / "models/SpatialTrackerV2-Offline")
            ),
        }
    )
    grounding = env.get("D2F_GROUNDING_DINO_MODEL", str(workspace / "models/grounding-dino-base"))
    sam2 = env.get("D2F_SAM2_MODEL", str(workspace / "models/sam2-hiera-large"))
    required_models = {
        Path(grounding): "config.json",
        Path(sam2): "config.json",
        Path(env["SPATRACKER_FRONT_MODEL_PATH"]): "model.safetensors",
        Path(env["SPATRACKER_OFFLINE_MODEL_PATH"]): "model.safetensors",
    }
    for model_path, marker in required_models.items():
        if not (model_path / marker).is_file():
            raise FileNotFoundError(f"missing local offline model asset: {model_path / marker}")

    for role in ("reference", "imagination"):
        scene = pair_root / role
        video = scene / "rgb.mp4"
        for name in (
            "scene_data.yaml",
            "camera_calibration_info.json",
            "initial_depth.pt",
            "camera_rgb.png",
        ):
            if not (scene / name).is_file():
                raise FileNotFoundError(f"{role} scene is missing {name}: {scene}")
        scene_data = yaml.safe_load((scene / "scene_data.yaml").read_text())
        initial_region = scene_data.get("initial_component_region") or {}
        # The registered grounding name is the single-object open-vocabulary
        # prompt; the scene caption may describe the whole task instead.
        grounding_name = str(
            spec.get("grounding_name")
            or scene_data.get("grounding_name")
            or scene_data.get("object_name", component_id)
        ).strip()
        if initial_region and str(initial_region.get("component_id", "")).strip() != component_id:
            raise ValueError(
                f"tracked component mismatch in {role}: spec={component_id!r} scene={initial_region.get('component_id')!r}"
            )
        fps, frame_count = _video_info(video)
        d2f_dir = scene / "dream2flow"
        d2f_dir.mkdir(parents=True, exist_ok=True)
        flow_pt = d2f_dir / "object_flow_result.pt"
        flow_npz = d2f_dir / "object_flow_result.npz"
        flow_args = [
            str(python),
            str(world_repo / "scripts/run_dream2flow_reference.py"),
            "--dream2flow-repo",
            str(d2f_repo),
            "--scene-dir",
            str(scene),
            "--object-name",
            grounding_name,
            "--video-role",
            role,
            "--source-model",
            str(spec["source_model"]),
            "--depth-mode",
            "playback",
            "--flow-only",
            "--flow-output",
            str(flow_pt),
            "--mask-erosion-kernel-size",
            str(FRONTEND_CONTRACT["mask_erosion_kernel_size"]),
            "--particle-ratio",
            str(FRONTEND_CONTRACT["particle_ratio"]),
            "--minimum-particles",
            str(FRONTEND_CONTRACT["minimum_particles"]),
            "--maximum-particles",
            str(FRONTEND_CONTRACT["maximum_particles"]),
            "--expected-cotracker-visibility-threshold",
            str(FRONTEND_CONTRACT["cotracker_visibility_threshold"]),
            "--grounding-dino-model-path",
            grounding,
            "--sam2-model-path",
            sam2,
            "--target-box-fraction-min",
            str((spec.get("dream2flow_target_box_fraction_range") or [0.03, 0.20])[0]),
            "--target-box-fraction-max",
            str((spec.get("dream2flow_target_box_fraction_range") or [0.03, 0.20])[1]),
        ]
        depth_path = scene / "depth_frames.pt"
        depth_audit = depth_path.with_suffix(".json")
        if not flow_pt.is_file() and (not depth_path.is_file() or not depth_audit.is_file()):
            depth_args = flow_args.copy()
            depth_args[depth_args.index("playback")] = "generate"
            depth_args.remove("--flow-only")
            output_index = depth_args.index("--flow-output")
            del depth_args[output_index:output_index + 2]
            _run(depth_args + ["--depth-only", "--depth-chunk-frames",
                str(FRONTEND_CONTRACT["depth_chunk_frames"]), "--depth-chunk-overlap",
                str(FRONTEND_CONTRACT["depth_chunk_overlap"])], env=env)
        if force_flow or not flow_pt.is_file():
            _run(flow_args + (["--force-flow"] if force_flow else []), env=env)
        _run(
            [
                str(python),
                str(world_repo / "scripts/export_dream2flow_object_flow_npz.py"),
                str(flow_pt),
                str(flow_npz),
            ],
            env=env,
        )
        _verify_flow(flow_npz)
        plan = d2f_dir / "trajectory_optimization_plan.npz"
        plan_args = [
            str(python),
            str(world_repo / "scripts/run_official_dream2flow_plan.py"),
            "--dream2flow-repo",
            str(official_repo),
            "--registration",
            str(registration),
            "--flow",
            str(flow_npz),
            "--video",
            str(video),
            "--output",
            str(plan),
            "--object-name",
            grounding_name,
            "--role",
            role,
            "--fps",
            str(fps),
            "--frame-count",
            str(frame_count),
        ]
        if force_flow or not plan.is_file():
            _run(plan_args, env=env)

    if prepare_only:
        from autoresearch.io import sha256_file

        case_id = case_identity["case_id"]
        receipt = {
            "schema_version": "official_dream2flow_pair_preparation.v1",
            "case_id": case_id,
            "spec_sha256": sha256_file(spec_path),
            "registration_sha256": sha256_file(registration),
            "roles": {
                role: {
                    "plan_sha256": sha256_file(
                        pair_root / role / "dream2flow/trajectory_optimization_plan.npz"),
                    "video_sha256": sha256_file(pair_root / role / "rgb.mp4"),
                }
                for role in ("reference", "imagination")
            },
            "model_episode_rerun": False,
            "score": None,
            "status": "prepared_only",
        }
        receipt_path = case_root / f"official_cae_prepared_{spec_path.stem}.json"
        text = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
        if receipt_path.exists() and receipt_path.read_text() != text:
            raise FileExistsError("prepared CAE receipt changed")
        if not receipt_path.exists():
            receipt_path.write_text(text)
        return {"cases": [receipt]}
    report = _single_case_report(case_root)
    report_path = case_root / (
        "official_dream2flow_diagnostic_report.json"
        if diagnostic_pair_validation is not None
        else "official_dream2flow_action_report.v2.json"
        if pair_validation is not None
        else "official_dream2flow_action_report.json")
    if diagnostic_pair_validation is not None:
        report["diagnostic_pair_validation_sha256"] = sha256_file(
            Path(diagnostic_pair_validation))
        report["formal_suite_admission"] = False
    elif pair_validation is not None:
        report["per_case_validation_sha256"] = sha256_file(
            Path(pair_validation))
        report["per_case_admission"] = True
        report["full_suite_total"] = None
    report_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if report_path.exists() and report_path.read_text(encoding="utf-8") != report_text:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        previous_cases = previous.get("cases") or []
        can_complete_unmeasured = (
            previous.get("case_count") == 1
            and previous.get("scored_count") == 0
            and report.get("scored_count") == 1
            and len(previous_cases) == 1
            and previous_cases[0].get("case_id") == report["cases"][0].get("case_id")
        )
        if not can_complete_unmeasured:
            raise FileExistsError(f"CAE report changed; choose a versioned path: {report_path}")
        report_path.write_text(report_text, encoding="utf-8")
    elif not report_path.exists():
        report_path.write_text(report_text, encoding="utf-8")
    if report["coverage_failure_count"]:
        reason = report["cases"][0].get("reason", "unknown")
        raise RuntimeError(f"official Dream2Flow report did not score the case: {reason}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("WORLDOLYMPIAD_WORKSPACE", "/mnt/data/v-wangchengzhi/worldolympiad2")),
    )
    parser.add_argument("--force-flow", action="store_true")
    parser.add_argument("--diagnostic-pair-validation", type=Path,
                        help="Independently validated single articulated pair; save a separately named diagnostic report.")
    parser.add_argument("--pair-validation", type=Path,
                        help="Independently validated articulated model pair for direct per-case CAE.")
    parser.add_argument("--spec-path", type=Path,
                        help="Versioned per-case CAE spec inside the selected case directory.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Produce both official optimizer plans and stop before case scoring.")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    from autoresearch.robotics_resource_lock import exclusive_gpu_job
    with exclusive_gpu_job(args.workspace / "outputs", os.environ.get("CUDA_VISIBLE_DEVICES", "0")):
        report = run_case(
            args.case_root, args.workspace, force_flow=args.force_flow,
            pair_validation=args.pair_validation,
            diagnostic_pair_validation=args.diagnostic_pair_validation,
            spec_path_override=args.spec_path,
            prepare_only=args.prepare_only)
    case = report["cases"][0]
    print(
        json.dumps(
            {
                "case_id": case["case_id"],
                "continuous_action": case.get("continuous_action"),
                "path_score": case.get("path_score"),
                "velocity_score": case.get("velocity_score"),
                "status": case["status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
