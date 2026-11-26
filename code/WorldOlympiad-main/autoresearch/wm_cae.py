"""Finalize a Dream2Flow trajectory-alignment manifest for a real World Model case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from continuous_action.current_metric import (
    CURRENT_METRIC_VERSION,
    CURRENT_METRIC_PROFILE,
    CURRENT_MANIFEST_NAME,
    CURRENT_PAIR_METADATA_NAME,
    current_metric_config,
    current_metric_config_sha256,
)
from continuous_action.workspaces import validate_robot_workspace_contract

from .io import write_json


def _write_immutable_json(path: Path, payload: dict) -> Path:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(
                f"refusing to overwrite immutable CAE provenance at {path}; "
                "use a new versioned case or output name"
            )
        return path
    return write_json(path, payload)


def finalize_manifest(case_dir: str | Path) -> Path:
    root = Path(case_dir).resolve()
    spec = json.loads((root / "cae_pair_spec.json").read_text(encoding="utf-8"))
    pair_root = root / str(spec["output_dir"])
    metadata = json.loads((pair_root / "pair_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("timing_policy") != "independent_native_v1":
        raise ValueError(
            "formal independent-action CAE requires timing_policy=independent_native_v1"
        )
    role_videos = metadata.get("role_videos")
    if not isinstance(role_videos, dict):
        raise ValueError("CAE pair metadata is missing per-role native timing")
    reference_duration = float(role_videos["reference"]["duration_sec"])
    imagination_duration = float(role_videos["imagination"]["duration_sec"])
    if reference_duration <= 0 or imagination_duration <= 0:
        raise ValueError("CAE role durations must be positive")

    def select_flow(role: str) -> Path:
        directory = pair_root / role / "dream2flow"
        candidates = (
            directory / "trajectory_optimization_plan.npz",
        )
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        raise FileNotFoundError(
            "Official Dream2Flow optimizer output is missing under: "
            f"{directory}"
        )

    reference_flow = select_flow("reference")
    imagination_flow = select_flow("imagination")

    from continuous_action.official_plan import validate_plan
    ref_meta = validate_plan(reference_flow, pair_root / 'reference/rgb.mp4', 'reference')
    pred_meta = validate_plan(imagination_flow, pair_root / 'imagination/rgb.mp4', 'imagination')
    if ref_meta['registration_sha256'] != pred_meta['registration_sha256']:
        raise ValueError('both video roles must use the same robot registration')

    model = str(spec["source_model"])
    raw_workspace_contract = spec.get("robot_workspace_contract")
    if not isinstance(raw_workspace_contract, dict):
        raise ValueError(
            "formal CAE requires cae_pair_spec.robot_workspace_contract"
        )
    robot_workspace_contract = validate_robot_workspace_contract(
        raw_workspace_contract,
        source_model=model,
    )
    object_name = str(spec["object_name"])
    benchmark_status = str(spec.get("benchmark_status", "formal"))
    manifest = {
        "evaluation_protocol": "video_imagination_dream2flow_official_action_trajectory_v2",
        "metric_version": CURRENT_METRIC_VERSION,
        "metric_profile": CURRENT_METRIC_PROFILE,
        "metric_config_sha256": current_metric_config_sha256(),
        "metrics": current_metric_config(),
        "robot_workspace_contract": robot_workspace_contract,
        "rollout": {
            "start_sec": 0.0,
            # The manifest envelope is descriptive only. Each role uses its
            # native video FPS; active CAE never samples one role on the
            # other's wall-clock horizon.
            "end_sec": max(reference_duration, imagination_duration),
            "reference": {
                "type": "dream2flow_video_plan",
                "path": reference_flow.relative_to(root).as_posix(),
                "video_path": (pair_root / "reference" / "rgb.mp4")
                .relative_to(root)
                .as_posix(),
                "video_role": "reference",
                "object_name": object_name,
                "sample_rate": float(role_videos["reference"]["fps"]),
                "benchmark_status": benchmark_status,
            },
            "predictions": {
                model: {
                    "type": "dream2flow_video_plan",
                    "path": imagination_flow.relative_to(root).as_posix(),
                    "video_path": (pair_root / "imagination" / "rgb.mp4")
                    .relative_to(root)
                    .as_posix(),
                    "video_role": "imagination",
                    "object_name": object_name,
                    "sample_rate": float(role_videos["imagination"]["fps"]),
                    "source_model": model,
                    "benchmark_status": benchmark_status,
                }
            },
        },
        "benchmark_status": benchmark_status,
        "evaluation_scope": spec.get(
            "evaluation_scope", "synthetic_cross_domain_real_wam"
        ),
    }
    metric_metadata = dict(metadata)
    metric_metadata["evaluation_protocol"] = manifest["evaluation_protocol"]
    metric_metadata["metric_version"] = CURRENT_METRIC_VERSION
    metric_metadata["metric_profile"] = CURRENT_METRIC_PROFILE
    metric_metadata["metric_config_sha256"] = current_metric_config_sha256()
    metric_metadata["robot_workspace_contract"] = robot_workspace_contract
    metric_metadata["cae_trajectory_source"] = (
        "official_dream2flow_trajectory_optimizer_ee_trajectory"
    )
    metric_metadata["action_head_used"] = False
    metric_metadata["object_to_robot_ik_used"] = True
    _write_immutable_json(pair_root / CURRENT_PAIR_METADATA_NAME, metric_metadata)
    path = _write_immutable_json(root / CURRENT_MANIFEST_NAME, manifest)

    case_manifest_path = root / "worldolympiad_case.json"
    case_manifest = json.loads(case_manifest_path.read_text(encoding="utf-8"))
    case_manifest["tracks"]["continuous_action"] = "required"
    # The harness exposes one active formal path. Retired pointers are not
    # copied forward, discovered, resumed, or offered for replay.
    case_manifest["continuous_action_manifests"] = {
        CURRENT_METRIC_VERSION: path.name
    }
    case_manifest["continuous_action_manifest"] = path.name
    write_json(case_manifest_path, case_manifest)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir")
    args = parser.parse_args()
    print(finalize_manifest(args.case_dir))


if __name__ == "__main__":
    main()
