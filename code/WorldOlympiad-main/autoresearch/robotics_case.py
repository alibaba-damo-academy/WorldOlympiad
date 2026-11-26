"""Materialize one matched robotics v2 pair for the existing evaluators."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import yaml

from continuous_action.workspaces import robot_workspace_contract_for_model

from .evaluation_harness import plan_case_evaluation
from .io import write_json
from .robotics_contracts import (
    EvaluationPairManifest,
    WAMInputManifest,
    WAMOutputManifest,
)
from .robotics_cosmos_registration import registration_for_cosmos_pair
from .robotics_views import sha256_file


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
EVALUATOR_PREFIX = {
    "lingbot_va_base": "lingbot_world",
    "lingbot_va_robotwin": "lingbot_world",
    "cosmos3_nano_droid": "cosmos",
}


def _dream2flow_target_fraction_range(wam: str) -> list[float]:
    """Return a profile-owned observability range, not a scoring threshold."""
    if wam in {"cosmos3_nano_droid", "lingbot_va_robotwin"}:
        return [64.0 / (320 * (256 if wam == "lingbot_va_robotwin" else 180)), 0.30]
    return [0.03, 0.20]


def prepare_robotics_case(
    pair_manifest: str | Path,
    output_root: str | Path,
    *,
    object_name: str,
    interaction_prompt: str | Path | None = None,
    interaction_chunks: str | Path | None = None,
    wam_output_manifest: str | Path | None = None,
    continuous_action_applicable: bool = True,
) -> Path:
    pair_path = Path(pair_manifest).resolve()
    pair = EvaluationPairManifest.model_validate_json(pair_path.read_text(encoding="utf-8"))
    if not SAFE_NAME.fullmatch(pair.case_id):
        raise ValueError("case_id must be filesystem-safe")
    reference = Path(pair.reference_video).resolve()
    generated = Path(pair.generated_video).resolve()
    if sha256_file(reference) != pair.reference_sha256 or sha256_file(generated) != pair.generated_sha256:
        raise ValueError("evaluation-pair video digest changed")
    certificate = json.loads(Path(pair.alignment_certificate).read_text(encoding="utf-8"))
    if certificate.get("passed") is not True:
        raise ValueError("alignment certificate did not pass")
    benchmark_status = str(certificate.get("benchmark_status", "formal"))
    base = Path(pair.cae_base_scene).resolve()
    required = ["camera_calibration_info.json", "initial_depth.pt", "scene_data.yaml"]
    generated_registration = None
    registration_urdf = None
    if continuous_action_applicable:
        if (base / "robot_registration.json").is_file():
            required.append("robot_registration.json")
        elif pair.wam == "cosmos3_nano_droid" and wam_output_manifest is not None:
            generated_registration, registration_urdf = registration_for_cosmos_pair(
                pair, Path(wam_output_manifest)
            )
        else:
            raise FileNotFoundError("CAE base scene is missing robot_registration.json")
    for name in required:
        if not (base / name).is_file():
            raise FileNotFoundError(f"CAE base scene is missing {name}")

    output = Path(output_root).resolve() / pair.case_id
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation case: {output}")
    output.mkdir(parents=True)
    ref_name = f"ref_{pair.case_id}.mp4"
    evaluator_prefix = EVALUATOR_PREFIX[pair.wam]
    gen_name = f"{evaluator_prefix}_gen_{pair.case_id}.mp4"
    shutil.copy2(reference, output / ref_name)
    shutil.copy2(generated, output / gen_name)
    base_out = output / "cae_base_scene"
    base_out.mkdir()
    for name in required:
        shutil.copy2(base / name, base_out / name)
    if generated_registration is not None:
        assert registration_urdf is not None
        urdf_out = base_out / "registered_droid.urdf"
        shutil.copy2(registration_urdf, urdf_out)
        generated_registration["source_urdf"] = str(urdf_out)
        write_json(base_out / "robot_registration.json", generated_registration)
    if (base / 'grasp_pose.json').is_file():
        shutil.copy2(base / 'grasp_pose.json', base_out / 'grasp_pose.json')
    scene_payload = yaml.safe_load((base / "scene_data.yaml").read_text(encoding="utf-8"))
    planner_urdf_value = scene_payload.get("planner_urdf_path")
    if planner_urdf_value is not None:
        planner_relative = Path(str(planner_urdf_value))
        if (
            planner_relative.is_absolute()
            or ".." in planner_relative.parts
            or len(planner_relative.parts) != 1
        ):
            raise ValueError("CAE planner URDF must be one base-scene-relative file")
        planner_source = base / planner_relative
        expected_digest = str(scene_payload.get("planner_urdf_sha256", ""))
        if (
            not planner_source.is_file()
            or len(expected_digest) != 64
            or sha256_file(planner_source) != expected_digest
        ):
            raise ValueError("CAE planner URDF is missing or differs from scene_data")
        target_link = str(scene_payload.get("target_link_name", ""))
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", target_link):
            raise ValueError("CAE planner target link is unsafe or missing")
        shutil.copy2(planner_source, base_out / planner_relative)

    capture = cv2.VideoCapture(str(output / gen_name))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if fps <= 0 or frames < 2:
        raise RuntimeError("generated evaluation video has invalid timing")
    duration = frames / fps
    if (interaction_prompt is None) != (interaction_chunks is None):
        raise ValueError(
            "interaction prompt and chunk provenance must be supplied together"
        )
    if wam_output_manifest is not None and interaction_prompt is not None:
        raise ValueError("use WAM runtime provenance or explicit interaction files, not both")

    def interval_text(start_sec: float, end_sec: float) -> str:
        def stamp(value: float) -> str:
            minutes = int(value // 60.0)
            seconds = value - 60.0 * minutes
            return f"{minutes:02d}:{seconds:05.2f}"

        return f"[{stamp(start_sec)}, {stamp(end_sec)})"

    interaction_provenance = None
    if wam_output_manifest is not None:
        wam_output_path = Path(wam_output_manifest).resolve()
        wam_output = WAMOutputManifest.model_validate_json(
            wam_output_path.read_text(encoding="utf-8")
        )
        if wam_output.case_id != pair.case_id or wam_output.wam != pair.wam:
            raise ValueError("WAM runtime provenance belongs to a different pair")
        if wam_output.frame_count != frames or abs(wam_output.fps - fps) > 1e-6:
            raise ValueError("WAM runtime provenance timing differs from the paired video")
        metadata_path = (wam_output_path.parent / wam_output.model_metadata).resolve()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if pair.wam == "cosmos3_nano_droid":
            input_manifest_path = (wam_output_path.parent / wam_output.input_manifest).resolve()
            input_spec = WAMInputManifest.model_validate_json(
                input_manifest_path.read_text(encoding="utf-8")
            )
            windows = metadata.get("windows")
            graph_runtime = input_spec.runtime_controller == "task_graph_runtime"
            legacy_graph = graph_runtime and input_spec.official_runtime_task == "worldolympiad/task_graph_fixed_horizon"
            expected_rollout = ("task_graph_fixed_horizon" if legacy_graph else
                "task_graph_native_episode" if graph_runtime else "official_robolab_task")
            if (
                metadata.get("rollout_controller") != expected_rollout
                or not isinstance(windows, list)
                or len(windows) != (frames + 30) // 32
                or (legacy_graph and (
                    metadata.get("termination_reason") != "reference_horizon_complete"
                    or metadata.get("task_success") is not False
                    or frames != int(metadata.get("official_task_max_steps") or 0) + 1
                ))
            ):
                raise ValueError("WAM runtime lacks exact bound Cosmos execution windows")
            ranges = [
                (start, min(start + 32, frames)) for start in range(0, frames, 32)
            ]
            rollout_controller = expected_rollout
        elif pair.wam == "lingbot_va_base":
            num_chunks = metadata.get("num_chunks")
            frame_chunk_size = metadata.get("frame_chunk_size")
            vae_temporal_scale = metadata.get("vae_temporal_scale")
            if (
                not isinstance(num_chunks, int)
                or isinstance(num_chunks, bool)
                or num_chunks <= 0
                or not isinstance(frame_chunk_size, int)
                or isinstance(frame_chunk_size, bool)
                or frame_chunk_size <= 0
                or not isinstance(vae_temporal_scale, int)
                or isinstance(vae_temporal_scale, bool)
                or vae_temporal_scale <= 0
                or metadata.get("rollout_length_source") != "upstream_model_config"
                or metadata.get("termination_reason") != "upstream_stopped"
                or metadata.get("task_success") is not False
            ):
                raise ValueError("WAM runtime lacks exact upstream LingBot chunks")
            video_frames_per_chunk = frame_chunk_size * vae_temporal_scale
            if frames != num_chunks * video_frames_per_chunk:
                raise ValueError("LingBot upstream chunks do not cover its video timeline")
            ranges = [
                (index * video_frames_per_chunk, (index + 1) * video_frames_per_chunk)
                for index in range(num_chunks)
            ]
            rollout_controller = "upstream_model_config"
        else:
            raise ValueError(f"unsupported WAM runtime provenance: {pair.wam}")
        chunks = [
            {
                "chunk_index": index,
                "frame_start": start,
                "frame_end": end,
                "generated_start_sec": start / fps,
                "generated_end_sec": end / fps,
            }
            for index, (start, end) in enumerate(ranges)
        ]
        prompts = [
            {
                "chunk_index": index,
                "action": pair.instruction,
                "caption": pair.instruction,
            }
            for index in range(len(chunks))
        ]
        interaction_provenance = {
            "source_wam_output": str(wam_output_path),
            "source_wam_output_sha256": sha256_file(wam_output_path),
            "source_runtime_metadata": str(metadata_path),
            "source_runtime_metadata_sha256": sha256_file(metadata_path),
            "native_chunk_count": len(chunks),
            "rollout_controller": rollout_controller,
        }
    elif interaction_prompt is None:
        chunks = [{
            "chunk_index": 0,
            "frame_start": 0,
            "frame_end": frames,
            "generated_start_sec": 0.0,
            "generated_end_sec": duration,
        }]
        prompts = [{
            "chunk_index": 0,
            "action": pair.instruction,
            "caption": pair.instruction,
        }]
    else:
        prompt_path = Path(interaction_prompt).resolve()
        chunk_path = Path(interaction_chunks).resolve()  # type: ignore[arg-type]
        prompt_payload = json.loads(prompt_path.read_text(encoding="utf-8"))
        chunk_payload = json.loads(chunk_path.read_text(encoding="utf-8"))
        prompts = prompt_payload.get("chunks")
        chunks = chunk_payload.get("chunks")
        if not isinstance(prompts, list) or not isinstance(chunks, list):
            raise ValueError("interaction provenance must contain chunk lists")
        if len(prompts) != len(chunks) or not chunks:
            raise ValueError("interaction prompt and timestamp chunk counts differ")
        if chunk_payload.get("frame_count") != frames or abs(
            float(chunk_payload.get("fps") or 0.0) - fps
        ) > 1e-6:
            raise ValueError("interaction provenance timing differs from generated video")
        cursor = 0
        for index, (prompt_record, chunk) in enumerate(
            zip(prompts, chunks, strict=True)
        ):
            if not isinstance(prompt_record, dict) or not isinstance(chunk, dict):
                raise ValueError("interaction chunk records must be objects")
            start = int(chunk.get("frame_start", -1))
            end = int(chunk.get("frame_end", -1))
            expected_interval = [start / fps, end / fps]
            if (
                prompt_record.get("chunk_index") != index
                or chunk.get("chunk_index") != index
                or start != cursor
                or not start < end <= frames
                or abs(float(chunk.get("generated_start_sec", -1.0)) - expected_interval[0]) > 1e-6
                or abs(float(chunk.get("generated_end_sec", -1.0)) - expected_interval[1]) > 1e-6
                or any(
                    not isinstance(actual, list)
                    or len(actual) != 2
                    or any(
                        abs(float(value) - expected) > 1e-6
                        for value, expected in zip(actual, expected_interval)
                    )
                    for actual in (
                        prompt_record.get("source_interval"),
                        chunk.get("source_interval"),
                    )
                )
            ):
                raise ValueError(f"interaction chunk {index} is not contiguous and exact")
            if not str(prompt_record.get("action") or "").strip() or not str(
                prompt_record.get("caption") or ""
            ).strip():
                raise ValueError(f"interaction chunk {index} has an empty task prompt")
            cursor = end
        if cursor != frames:
            raise ValueError("interaction chunks do not cover the exact generated timeline")
        if pair.wam == "cosmos3_nano_droid":
            expected = [
                (start, min(start + 32, frames)) for start in range(0, frames, 32)
            ]
            actual = [
                (int(item["frame_start"]), int(item["frame_end"])) for item in chunks
            ]
            if actual != expected:
                raise ValueError("Cosmos interaction chunks do not preserve native windows")
        interaction_provenance = {
            "source_prompt": str(prompt_path),
            "source_prompt_sha256": sha256_file(prompt_path),
            "source_chunks": str(chunk_path),
            "source_chunks_sha256": sha256_file(chunk_path),
            "native_chunk_count": len(chunks),
        }

    prompt_records = []
    timestamp_records = []
    for prompt_record, chunk in zip(prompts, chunks, strict=True):
        start_sec = float(chunk["generated_start_sec"])
        end_sec = float(chunk["generated_end_sec"])
        interval = interval_text(start_sec, end_sec)
        prompt_records.append(
            {
                "chunk_index": int(chunk["chunk_index"]),
                "interval": interval,
                "action": str(prompt_record["action"]),
                "caption": str(prompt_record["caption"]),
            }
        )
        timestamp_records.append(
            {
                "chunk_index": int(chunk["chunk_index"]),
                "source_interval": interval,
                "frame_start": int(chunk["frame_start"]),
                "frame_end": int(chunk["frame_end"]),
                "generated_start_sec": start_sec,
                "generated_end_sec": end_sec,
            }
        )
    write_json(output / "prompt.json", prompt_records)
    write_json(
        output / f"{Path(gen_name).stem}_chunk_timestamps.json",
        {
            "version": 1,
            "video_path": gen_name,
            "fps": fps,
            "total_frames": frames,
            "duration_sec": duration,
            "chunks": timestamp_records,
        },
    )
    if continuous_action_applicable:
        write_json(
            output / "cae_pair_spec.json",
            {
            "schema_version": "worldolympiad_cae_pair_spec.v2",
            "reference_video": ref_name,
            "imagination_video": gen_name,
            "base_scene": base_out.name,
            "robot_registration": str(Path(base_out.name) / "robot_registration.json"),
            **({"grasp_pose": str(Path(base_out.name) / "grasp_pose.json")} if (base_out / "grasp_pose.json").is_file() else {}),
            "source_model": pair.wam,
            "evaluator_pipeline_prefix": evaluator_prefix,
            "object_name": object_name,
            "robot_workspace_contract": robot_workspace_contract_for_model(pair.wam),
            "imagination_includes_condition": True,
            "output_dir": "cae_pair",
            "physical_view": pair.reference_view,
            "dream2flow_target_box_fraction_range": (
                _dream2flow_target_fraction_range(pair.wam)
            ),
            "benchmark_status": benchmark_status,
            "evaluation_scope": "robotics_imagination_matched_view_v2",
            },
        )
    case = {
        "schema_version": "worldolympiad_robotics_case.v2",
        "benchmark_status": benchmark_status,
        "leaderboard_eligible": bool(certificate.get("leaderboard_eligible", benchmark_status == "formal")),
        "case_id": pair.case_id,
        "event_type": "robot_pick_place",
        "interaction_mode": "robot_contact",
        "model_id": pair.wam,
        "evaluator_pipeline_prefix": evaluator_prefix,
        "reference_video": ref_name,
        "generated_video": gen_name,
        "view_contract": "matched_single_physical_view_v2",
        "reference_view_count": 1,
        "generated_view_count": 1,
        "reference_view_id": pair.reference_view,
        "generated_view_id": pair.generated_view,
        "benchmark_status": "formal",
        "evaluation_scope": "robotics_imagination_matched_view_v2",
        "leaderboard_eligible": False,
        "source_evaluation_pair": str(pair_path),
        "source_evaluation_pair_sha256": sha256_file(pair_path),
        "interaction_provenance": interaction_provenance,
        "tracks": {
            "physical": "required",
            "geometry": "required",
            "interaction": "required",
            "continuous_action": (
                "pending_symmetric_dream2flow_manifest"
                if continuous_action_applicable
                else "not_applicable_to_official_rigid_grasp_optimizer"
            ),
        },
    }
    plan_path = plan_case_evaluation(case).write_json(output / "evaluation_plan.json")
    case["evaluation_plan"] = plan_path.name
    return write_json(output / "worldolympiad_case.json", case)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair_manifest")
    parser.add_argument("output_root")
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--interaction-prompt")
    parser.add_argument("--interaction-chunks")
    parser.add_argument("--wam-output-manifest")
    parser.add_argument("--skip-continuous-action", action="store_true")
    args = parser.parse_args()
    print(
        prepare_robotics_case(
            args.pair_manifest,
            args.output_root,
            object_name=args.object_name,
            interaction_prompt=args.interaction_prompt,
            interaction_chunks=args.interaction_chunks,
            wam_output_manifest=args.wam_output_manifest,
            continuous_action_applicable=not args.skip_continuous_action,
        )
    )


if __name__ == "__main__":
    main()
