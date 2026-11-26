"""Formal Physics→WM evaluation with independent object-motion CAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2

from .agentic_contracts import AgentTask
from .full_track_report import build_report_v3
from .lingbot_world2_harness import validate_artifacts
from .robotics_resource_lock import exclusive_gpu_job
from .worldolympiad_case import prepare_agentic_physics_case


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_case_admission(case_root: Path, workspace: Path) -> None:
    """Admit only a frozen passive case to the shared video judges."""
    case_root = case_root.resolve()
    allowed = (workspace.resolve() / "outputs/evaluation_cases")
    if case_root.parent != allowed:
        raise ValueError("physics evaluation root is outside the case workspace")
    applicability = json.loads((case_root / "applicability_plan.json").read_text())
    cases = applicability.get("cases", [])
    if applicability.get("schema_version") != "worldolympiad_track_applicability.v1" or len(cases) != 1:
        raise ValueError("physics evaluation requires one frozen applicability case")
    case_id = cases[0]["case_id"]
    case_dir = case_root / case_id
    manifest = json.loads((case_dir / "worldolympiad_case.json").read_text())
    binding = json.loads((case_dir / "source_binding.json").read_text())
    if (manifest.get("case_id") != case_id
            or manifest.get("benchmark_status") != "agentic_passive_physics"
            or manifest.get("task_sha256") != cases[0].get("task_sha256")
            or manifest.get("task_sha256") != binding.get("task_sha256")
            or manifest.get("world_model_run_sha256") != binding.get("world_model_run_sha256")
            or binding.get("reference_video_sha256") !=
                _sha(case_dir / manifest["reference_video"])):
        raise ValueError("physics case is not bound to its admitted reference and model")
    for key in ("reference_video", "generated_video"):
        value = manifest[key]
        if Path(value).name != value or not (case_dir / value).is_file():
            raise ValueError("physics case video is missing or escapes its case")


def _run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write("\n=== WorldOlympiad evaluation attempt ===\n")
        stream.flush()
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"evaluation stage failed ({result.returncode}); see {log}")


def _imagination_flow(*, task: AgentTask, workspace: Path, repository: Path,
                      case_dir: Path, gpu_id: str) -> Path:
    case = json.loads((case_dir / "worldolympiad_case.json").read_text())
    video = case_dir / case["generated_video"]
    source = case_dir / "cae_base_scene"
    scene = case_dir / "physics_object_cae" / "imagination"
    scene.mkdir(parents=True, exist_ok=True)
    selected_video = scene / "rgb.mp4"
    if selected_video.exists():
        if _sha(selected_video) != _sha(video):
            raise ValueError("imagination video changed after flow preparation")
    else:
        shutil.copy2(video, selected_video)
    for name in ("camera_calibration_info.json", "initial_depth.pt", "scene_data.yaml"):
        target = scene / name
        if not target.exists():
            shutil.copy2(source / name, target)
        elif _sha(target) != _sha(source / name):
            raise ValueError("imagination calibration differs from frozen reference")
    image = scene / "camera_rgb.png"
    if not image.exists():
        capture = cv2.VideoCapture(str(video))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None or not cv2.imwrite(str(image), frame):
            raise ValueError("World Model first video frame is undecodable")
    result = scene / "dream2flow" / "object_flow_result.npz"
    if result.is_file():
        receipt = scene / "flow_binding.json"
        if not receipt.is_file() or json.loads(receipt.read_text()).get("video_sha256") != _sha(video):
            raise ValueError("cached imagination flow belongs to another video")
        return result
    flow = result.with_suffix(".pt")
    flow.parent.mkdir(parents=True, exist_ok=True)
    dream = workspace / "code/Dream2Flow-main-cae-min64-vis08-v1"
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join([str(repository), str(dream)]),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "CUDA_VISIBLE_DEVICES": gpu_id,
           "SPATRACKER_FRONT_MODEL_PATH": str(workspace / "models/SpatialTrackerV2_Front"),
           "SPATRACKER_OFFLINE_MODEL_PATH": str(workspace / "models/SpatialTrackerV2-Offline")}
    for name in ("SPATRACKER_FRONT_MODEL_PATH", "SPATRACKER_OFFLINE_MODEL_PATH"):
        if not (Path(env[name]) / "model.safetensors").is_file():
            raise FileNotFoundError(f"registered Dream2Flow dependency is missing: {name}")
    with exclusive_gpu_job(workspace / "outputs", gpu_id, job_kind="physics_object_flow"):
        _run([str(workspace / "envs/dream2flow/bin/python"),
              str(repository / "scripts/run_dream2flow_reference.py"),
              "--dream2flow-repo", str(dream), "--scene-dir", str(scene),
              "--video-role", "imagination", "--source-model", "lingbot_world2_14b_causal_fast",
              "--depth-mode", "generate", "--flow-only", "--force-flow",
              "--flow-output", str(flow),
              "--mask-erosion-kernel-size", "3", "--particle-ratio", "0.01",
              "--minimum-particles", "64", "--maximum-particles", "1024",
              "--expected-cotracker-visibility-threshold", "0.8",
              "--grounding-dino-model-path", str(workspace / "models/grounding-dino-base"),
              "--sam2-model-path", str(workspace / "models/sam2-hiera-large")],
             cwd=repository, env=env, log=scene / "flow.log")
        _run([str(workspace / "envs/dream2flow/bin/python"),
              str(repository / "scripts/export_dream2flow_object_flow_npz.py"),
              str(flow), str(result)], cwd=repository, env=env,
             log=scene / "flow.log")
    (scene / "flow_binding.json").write_text(json.dumps({
        "task_sha256": task.content_hash(), "video_sha256": _sha(video),
        "flow_sha256": _sha(result)}, sort_keys=True, indent=2) + "\n")
    return result


def evaluate(*, task_path: Path, workspace: Path, repository: Path,
             reference_root: Path, input_contract: Path,
             wm_run_manifest: Path, evaluation_root: Path, gpu_id: str) -> Path:
    task = AgentTask.model_validate_json(task_path.read_text())
    workspace, repository = workspace.resolve(), repository.resolve()
    evaluation_root = evaluation_root.resolve()
    manifest = prepare_agentic_physics_case(task=task, reference_root=reference_root,
        wm_run_manifest=wm_run_manifest, input_contract=input_contract,
        output_root=evaluation_root)
    case_dir = manifest.parent
    if not list(case_dir.glob("*_judge_*.json")):
        env = {**os.environ, "WORLDOLYMPIAD_WORKSPACE": str(workspace),
               "WORLD_REPO": str(repository),
               "WORLDEVAL_CAE_PROFILE": "physics_object"}
        _run([str(repository / "deployment/ziplab5090/run_worldolympiad_evaluation.sh"),
              str(evaluation_root), "lingbot-world"],
             cwd=repository, env=env, log=evaluation_root / "judge.log")
    flow = _imagination_flow(task=task, workspace=workspace,
                             repository=repository, case_dir=case_dir, gpu_id=gpu_id)
    reference_flow = reference_root / "reference/dream2flow/object_flow_result.npz"
    if not reference_flow.is_file():
        raise FileNotFoundError("certified reference object flow is missing")
    from continuous_action.physics_partial import score_partial_pair
    reference = json.loads((reference_root / "simulation.json").read_text())
    wm = json.loads(wm_run_manifest.read_text())
    validate_artifacts(wm_run_manifest.parent / wm["artifacts"],
                       wm["request_identity_sha256"])
    model_video = case_dir / json.loads(manifest.read_text())["generated_video"]
    reference_video = reference_root / "reference/rgb.mp4"
    pair = {"tracked_entity": task.objects[0].object_id,
        "roles": {
            "reference": {"dir": str(reference_flow.parent),
                          "fps": reference["fps"], "frames": reference["frames"],
                          "video_sha256": _sha(reference_video)},
            "imagination": {"dir": str(flow.parent),
                            "fps": wm["result"]["fps"],
                            "frames": wm["result"]["frames"],
                            "video_sha256": _sha(model_video)},
        }}
    measurement, _ = score_partial_pair(pair)
    action = {"cases": [{"case_id": task.task_id,
                         "continuous_score": measurement["continuous_score"],
                         "source": str(flow), "evaluator_coverage_failure": False}]}
    action_path = evaluation_root / "physics_object_action_report.json"
    action_path.write_text(json.dumps({"schema_version": "physics_object_action_report.v1",
                                       "measurement": measurement, **action},
                                      indent=2, sort_keys=True) + "\n")
    applicability = json.loads((evaluation_root / "applicability_plan.json").read_text())
    report = build_report_v3([evaluation_root], applicability_plan=applicability,
                             action_report=action)
    if report["complete_count"] != report["case_count"]:
        raise RuntimeError("required WorldOlympiad track is unmeasured")
    result = evaluation_root / "three_track_report_v1.json"
    result.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--wm-run-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--gpu-id", required=True)
    args = parser.parse_args()
    print(evaluate(task_path=args.task, workspace=args.workspace,
                   repository=args.repository, reference_root=args.reference_root,
                   input_contract=args.input_contract,
                   wm_run_manifest=args.wm_run_manifest,
                   evaluation_root=args.evaluation_root, gpu_id=args.gpu_id))


if __name__ == "__main__":
    main()
