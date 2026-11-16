"""Prepare certified reference/real-WM pairs for the official evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import cv2

from .io import write_json
from .evaluation_harness import plan_case_evaluation
from .agentic_contracts import AgentTask
from .lingbot_world2_harness import validate_artifacts


def prepare_agentic_physics_case(
    *, task: AgentTask, reference_root: str | Path,
    wm_run_manifest: str | Path, input_contract: str | Path,
    output_root: str | Path,
) -> Path:
    """Build one frozen passive-physics pair without task-name recipes."""
    if task.embodiment is not None:
        raise ValueError("passive physics case cannot contain a robot embodiment")
    reference_root = Path(reference_root).resolve()
    wm_run_manifest = Path(wm_run_manifest).resolve()
    input_contract = Path(input_contract).resolve()
    output_root = Path(output_root).resolve()
    reference = reference_root / "reference"
    simulation = json.loads((reference_root / "simulation.json").read_text())
    physical = json.loads((reference_root / "physical_certificate.json").read_text())
    if (physical.get("passed") is not True
            or physical.get("task_sha256") != task.content_hash()
            or simulation.get("physical_certificate_sha256") !=
            hashlib.sha256((reference_root / "physical_certificate.json").read_bytes()).hexdigest()):
        raise ValueError("passive reference lacks independent physical admission")
    render = json.loads((reference_root / "render.json").read_text())
    source_video = reference / "rgb.mp4"
    if render.get("video_sha256") != hashlib.sha256(source_video.read_bytes()).hexdigest():
        raise ValueError("reference video differs from its render receipt")
    wm = json.loads(wm_run_manifest.read_text())
    if (wm.get("schema_version") != "lingbot_world2_run_manifest.v1"
            or wm.get("status") != "complete"
            or wm.get("profile_id") != "lingbot_world2_14b_causal_fast"):
        raise ValueError("World2 model run is not a complete registered profile")
    request = json.loads((wm_run_manifest.parent / wm["request"]).read_text())
    if request.get("input_contract_sha256") != hashlib.sha256(input_contract.read_bytes()).hexdigest():
        raise ValueError("World2 run consumed another frozen reference input")
    artifacts = wm_run_manifest.parent / wm["artifacts"]
    validate_artifacts(artifacts, wm["request_identity_sha256"])
    generated = artifacts / "imagination.mp4"
    if not generated.is_file():
        raise ValueError("World2 run lacks a video")
    case_id = _safe_name("task_id", task.task_id)
    destination = output_root / case_id
    source_binding = {"task_sha256": task.content_hash(),
                      "reference_video_sha256": render["video_sha256"],
                      "world_model_run_sha256": hashlib.sha256(wm_run_manifest.read_bytes()).hexdigest(),
                      "input_contract_sha256": hashlib.sha256(input_contract.read_bytes()).hexdigest()}
    if destination.exists():
        binding_path = destination / "source_binding.json"
        manifest_path = destination / "worldolympiad_case.json"
        if (binding_path.is_file() and manifest_path.is_file()
                and json.loads(binding_path.read_text()) == source_binding):
            return manifest_path
        raise FileExistsError("formal physics case already exists with incomplete or different inputs")
    destination.mkdir(parents=True)
    ref_name = f"ref_{case_id}.mp4"
    model_name = f"lingbot_world_gen_{case_id}.mp4"
    shutil.copy2(source_video, destination / ref_name)
    shutil.copy2(generated, destination / model_name)
    base = destination / "cae_base_scene"
    base.mkdir()
    for name in ("camera_calibration_info.json", "initial_depth.pt", "scene_data.yaml"):
        shutil.copy2(reference / name, base / name)
    duration, fps, frames = _video_timing(destination / model_name)
    prompt = [{"chunk_index": 0, "interval": f"[00:00, 00:{duration:05.2f})",
               "action": task.instruction, "caption": task.instruction}]
    write_json(destination / "prompt.json", prompt)
    write_json(destination / f"{Path(model_name).stem}_chunk_timestamps.json", {
        "version": 1, "video_path": model_name, "fps": fps,
        "total_frames": frames, "duration_sec": duration,
        "chunks": [{"chunk_index": 0, "source_interval": prompt[0]["interval"],
                    "frame_start": 0, "frame_end": frames,
                    "generated_start_sec": 0.0, "generated_end_sec": duration}],
    })
    event = task.operation_program[0].operation
    interaction_applicable = any(
        predicate.kind in {"contact", "rebounded", "inside", "in_container"}
        for predicate in task.success_predicates
    )
    applicability = {"schema_version": "worldolympiad_track_applicability.v1",
        "cases": [{"case_id": case_id, "task_sha256": task.content_hash(),
                   "tracks": {"physical": True, "geometry": True,
                              "interaction": (True if interaction_applicable else {
                                  "applicable": False,
                                  "reason": "task has no declared interaction event"}),
                              "continuous_action": True}}]}
    applicability_path = output_root / "applicability_plan.json"
    if applicability_path.exists():
        if json.loads(applicability_path.read_text()) != applicability:
            raise ValueError("frozen track applicability changed")
    else:
        write_json(applicability_path, applicability)
    case = {"schema_version": "worldolympiad_case.v1", "case_id": case_id,
        "event_type": event, "interaction_mode": "external_intervention",
        "model_id": "lingbot_world2_14b_causal_fast",
        "reference_video": ref_name, "generated_video": model_name,
        "view_contract": "single_main_view_v1",
        "reference_view_count": 1, "generated_view_count": 1,
        "reference_view_id": "main", "generated_view_id": "main",
        "benchmark_status": "agentic_passive_physics",
        "evaluation_scope": "measured_physics_to_wm",
        "leaderboard_eligible": False,
        "task_sha256": task.content_hash(),
        "world_model_run_sha256": hashlib.sha256(wm_run_manifest.read_bytes()).hexdigest(),
        "tracks": {"physical": "required", "geometry": "required",
                   "interaction": "required" if interaction_applicable else "not_applicable",
                   "continuous_action": "not_applicable_to_official_rigid_grasp_optimizer"}}
    plan = destination / "evaluation_plan.json"
    plan_case_evaluation(case).write_json(plan)
    case["evaluation_plan"] = plan.name
    manifest = write_json(destination / "worldolympiad_case.json", case)
    write_json(destination / "source_binding.json", source_binding)
    return manifest


EVENT_TEXT = {
    "block_sort_tray": "Pick the red cube and place it inside the blue tray.",
    "cup_tip": "A robotic arm pushes the upper side of the blue ceramic cup until the cup tips over.",
    "block_push_target": "A robotic arm pushes the wooden block across the table into the green target zone and leaves it on the table.",
    "stack_collapse": "A robotic arm pushes the middle block sideways until the three-block stack collapses.",
}
EVENT_OBJECT = {
    "block_sort_tray": "small red sorting block",
    "cup_tip": "small blue object",
    "block_push_target": "small brown block",
    "stack_collapse": "upper yellow block",
}
PIPELINE_PREFIX = {
    "cosmos-predict": "cosmos",
    "lingbot-world": "lingbot_world",
}
PASSING_STATES = {"pass", "repaired_pass"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _video_timing(path: Path) -> tuple[float, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open World Model video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"World Model video has invalid timing metadata: {path}")
    return frames / fps, fps, frames


def _safe_name(label: str, value: str) -> str:
    if not SAFE_NAME.fullmatch(value):
        raise ValueError(f"{label} must be a filesystem-safe identifier")
    return value


def prepare_case(
    reference_dir: str | Path,
    wm_video: str | Path,
    output_root: str | Path,
    *,
    model_id: str,
    imagination_includes_condition: bool = True,
) -> Path:
    """Materialize one official WorldOlympiad case from a certified reference."""

    reference = Path(reference_dir).resolve()
    wm_path = Path(wm_video).resolve()
    model = _safe_name("model_id", model_id)
    spec = json.loads((reference / "episode_spec.json").read_text(encoding="utf-8"))
    event_id = _safe_name("episode_id", str(spec["episode_id"]))
    benchmark = json.loads(
        (reference / "benchmark_certificate.json").read_text(encoding="utf-8")
    )
    if benchmark.get("status") not in PASSING_STATES:
        raise ValueError(f"reference is not certified: {reference}")
    reference_video = reference / "rgb.mp4"
    if not reference_video.is_file() or not wm_path.is_file():
        raise FileNotFoundError("reference rgb.mp4 and World Model video are required")
    cae_input_names = (
        "camera_calibration_info.json",
        "initial_depth.pt",
        "scene_data.yaml",
    )
    for name in cae_input_names:
        source = reference / name
        if not source.is_file():
            raise FileNotFoundError(f"certified reference is missing CAE input: {source}")

    case_dir = Path(output_root).resolve() / event_id
    if case_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation case: {case_dir}")
    case_dir.mkdir(parents=True)
    ref_target = case_dir / f"ref_{event_id}.mp4"
    wm_target = case_dir / f"{model}_gen_{event_id}.mp4"
    shutil.copy2(reference_video, ref_target)
    shutil.copy2(wm_path, wm_target)

    cae_base = case_dir / "cae_base_scene"
    cae_base.mkdir()
    for name in cae_input_names:
        source = reference / name
        shutil.copy2(source, cae_base / name)

    duration, fps, total_frames = _video_timing(wm_target)
    description = EVENT_TEXT.get(event_id, f"Execute the {event_id} physical event.")
    prompt = [
        {
            "chunk_index": 0,
            "interval": f"[00:00, 00:{duration:05.2f})",
            "action": description,
            "caption": description,
        }
    ]
    write_json(case_dir / "prompt.json", prompt)
    write_json(
        case_dir / f"{wm_target.stem}_chunk_timestamps.json",
        {
            "version": 1,
            "video_path": wm_target.name,
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration,
            "chunks": [
                {
                    "chunk_index": 0,
                    "source_interval": prompt[0]["interval"],
                    "frame_start": 0,
                    "frame_end": total_frames,
                    "generated_start_sec": 0.0,
                    "generated_end_sec": duration,
                }
            ],
        },
    )
    write_json(
        case_dir / "cae_pair_spec.json",
        {
            "schema_version": "worldolympiad_cae_pair_spec.v1",
            "reference_video": ref_target.name,
            "imagination_video": wm_target.name,
            "base_scene": cae_base.name,
            "source_model": model,
            "object_name": EVENT_OBJECT[event_id],
            "imagination_includes_condition": imagination_includes_condition,
            "output_dir": "cae_pair",
            "benchmark_status": "pipeline_validation",
            "evaluation_scope": "synthetic_cross_domain_real_wam",
        },
    )
    case_payload = {
            "schema_version": "worldolympiad_case.v1",
            "case_id": event_id,
            "event_type": event_id,
            "interaction_mode": (
                "robot_contact" if spec.get("robot_program") is not None else "external_intervention"
            ),
            "model_id": model,
            "reference_video": ref_target.name,
            "generated_video": wm_target.name,
            "view_contract": "single_main_view_v1",
            "reference_view_count": 1,
            "generated_view_count": 1,
            "reference_view_id": "main",
            "generated_view_id": "main",
            "benchmark_status": "pipeline_validation",
            "evaluation_scope": "synthetic_cross_domain_real_wam",
            "leaderboard_eligible": False,
            "tracks": {
                "physical": "required",
                "geometry": "required",
                "interaction": "required",
                "continuous_action": "pending_symmetric_dream2flow_manifest",
            },
        }
    plan_path = case_dir / "evaluation_plan.json"
    plan_case_evaluation(case_payload).write_json(plan_path)
    case_payload["evaluation_plan"] = plan_path.name
    return write_json(case_dir / "worldolympiad_case.json", case_payload)


def prepare_case_set(
    release_root: str | Path,
    wm_root: str | Path,
    output_root: str | Path,
    *,
    model_id: str,
    imagination_includes_condition: bool = True,
) -> dict[str, Any]:
    """Prepare the four Pilot events for one real World Model."""

    release = Path(release_root).resolve()
    wm_dir = Path(wm_root).resolve()
    output = Path(output_root).resolve()
    events_root = release / "pilot" / "episodes"
    if not events_root.is_dir():
        raise FileNotFoundError(f"release has no pilot episodes: {events_root}")
    cases = []
    for event_id in EVENT_TEXT:
        manifest = prepare_case(
            events_root / event_id / "reference",
            wm_dir / f"{event_id}.mp4",
            output,
            model_id=model_id,
            imagination_includes_condition=imagination_includes_condition,
        )
        cases.append(str(manifest))
    return {
        "schema_version": "worldolympiad_case_set.v1",
        "model_id": model_id,
        "case_count": len(cases),
        "case_manifests": cases,
        "required_tracks": ["physical", "geometry", "interaction"],
        "conditional_tracks": ["continuous_action"],
        "view_contract": "single_main_view_v1",
        "evaluation_plans": [
            str(Path(path).with_name("evaluation_plan.json")) for path in cases
        ],
        "benchmark_status": "pipeline_validation",
        "evaluation_scope": "synthetic_cross_domain_real_wam",
        "leaderboard_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_root")
    parser.add_argument("wm_root")
    parser.add_argument("output_root")
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--imagination-excludes-condition",
        action="store_true",
        help="Prepend the exact reference conditioning frame before CAE alignment.",
    )
    args = parser.parse_args()
    result = prepare_case_set(
        args.release_root,
        args.wm_root,
        args.output_root,
        model_id=args.model_id,
        imagination_includes_condition=not args.imagination_excludes_condition,
    )
    path = write_json(Path(args.output_root) / "case_set.json", result)
    print(path)


if __name__ == "__main__":
    main()
