"""Strict LingBot-VA adapters for certified SceneCode robotics views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io import write_json
from .media import write_h264_rgb
from .robotics_contracts import WAMInputManifest, WAMOutputManifest
from .robotics_prompts import (
    copy_reference_prompt_snapshot,
    load_bound_robotics_prompt,
    render_reference_robotics_prompt,
)
from .robotics_quality import certify_wam_output
from .robotics_wam_bundle import (
    build_bundle_artifact_digests,
    discover_preflight_artifacts,
    write_raw_video_certificate,
)
from .robotics_views import (
    load_reference_manifest,
    read_reference_video_frame,
    resolve_artifact,
)
from .wam_adapter_contracts import (
    LINGBOT_MODEL_CONFIG,
    LINGBOT_SOURCE_MODEL,
    build_lingbot_adapter_contract,
    load_adapter_contract,
    sha256_file,
)


DEFAULT_SOURCE_MODEL = LINGBOT_SOURCE_MODEL
DEFAULT_MODEL_CONFIG = LINGBOT_MODEL_CONFIG
PROFILE_BY_SOURCE = {
    LINGBOT_SOURCE_MODEL: (
        LINGBOT_MODEL_CONFIG,
        "lingbot_va_demo_physical_multiview_v3",
        build_lingbot_adapter_contract,
    ),
}


def prepare_input(
    reference_manifest: str | Path,
    output_root: str | Path,
    *,
    source_model: str = DEFAULT_SOURCE_MODEL,
    model_config: str | None = None,
    start_frame_index: int | None = None,
) -> Path:
    if source_model not in PROFILE_BY_SOURCE:
        raise ValueError(f"unsupported LingBot source model: {source_model!r}")
    expected_config, protocol, builder = PROFILE_BY_SOURCE[source_model]
    model_config = expected_config if model_config is None else model_config
    if model_config != expected_config:
        raise ValueError(
            f"LingBot checkpoint/config mismatch: {source_model!r} requires "
            f"{expected_config!r}, got {model_config!r}"
        )
    if start_frame_index not in (None, 0):
        raise ValueError("formal LingBot adaptation must start at reference frame zero")
    reference_path, reference = load_reference_manifest(reference_manifest)
    prompt_contract = render_reference_robotics_prompt(
        reference_path, "lingbot_va_base"
    )
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite LingBot input bundle: {output}")
    images = output / "input_images"
    images.mkdir(parents=True)
    reference_snapshot, reference_task_snapshot, reference_task_sha256 = (
        copy_reference_prompt_snapshot(reference_path, output)
    )
    # Only frame-zero conditioning is bound to the certified reference. The
    # upstream LingBot config owns its independent rollout length.
    start_frame = 0 if start_frame_index is None else start_frame_index
    formal_view_id = "top"
    formal_view = reference.view(formal_view_id)
    reference_frame_count = formal_view.frame_count - start_frame
    if reference_frame_count < 2:
        raise ValueError("LingBot start frame leaves no formal reference rollout")
    adapter = builder(
        case_id=reference.case_id,
        reference_fps=formal_view.fps,
        reference_frame_count=reference_frame_count,
    )
    source_views = adapter.source_views
    camera_by_view = {camera.source_view: camera for camera in adapter.camera_inputs}
    for view_id in source_views.values():
        view = reference.view(view_id)  # type: ignore[arg-type]
        if view.frame_count != formal_view.frame_count or abs(view.fps - formal_view.fps) > 1e-6:
            raise ValueError("LingBot physical views must share one complete reference timeline")
        expected_size = camera_by_view[view_id].size_wh
        if (view.width, view.height) != expected_size:
            raise ValueError(
                "LingBot requires SceneCode to render each reference view at its "
                f"native size: {view_id}={(view.width, view.height)} expected={expected_size}"
            )
    adapter_path = adapter.write_json(output / "adapter_contract.json")
    prompt_path = prompt_contract.write_json(output / "prompt_manifest.json")
    input_images: dict[str, str] = {}
    for camera in adapter.camera_inputs:
        camera_key, view_id = camera.input_key, camera.source_view
        frame = read_reference_video_frame(
            reference_path, reference, view_id, start_frame
        )
        destination = images / f"{camera_key}.png"
        if camera.spatial_transform != "identity_exact":
            raise ValueError(
                "formal LingBot adaptation forbids reference crop/resize policies: "
                f"{camera.spatial_transform}"
            )
        if frame.shape[:2] != (camera.size_wh[1], camera.size_wh[0]):
            raise ValueError(
                "LingBot identity reference changed size after decoding: "
                f"{view_id}={frame.shape[1]}x{frame.shape[0]}"
            )
        transformed = np.ascontiguousarray(frame)
        if not cv2.imwrite(str(destination), transformed):
            raise RuntimeError(f"unable to write LingBot input image: {destination}")
        input_images[camera_key] = str(destination.relative_to(output))
    manifest = WAMInputManifest(
        case_id=reference.case_id,
        wam=adapter.wam,
        source_model=source_model,
        task_instruction=reference.task_instruction,
        model_conditioning_prompt=prompt_contract.rendered_prompt,
        prompt_manifest=prompt_path.name,
        prompt_manifest_sha256=sha256_file(prompt_path),
        reference_manifest=str(reference_path),
        reference_manifest_sha256=sha256_file(reference_path),
        reference_snapshot=reference_snapshot,
        reference_task_snapshot=reference_task_snapshot,
        reference_task_sha256=reference_task_sha256,
        adapter_contract=adapter_path.name,
        adapter_contract_sha256=sha256_file(adapter_path),
        input_images=input_images,
        source_views=source_views,  # type: ignore[arg-type]
        formal_reference_view=formal_view_id,
        reference_start_frame_index=start_frame,
        reference_frame_count=reference_frame_count,
        runtime_controller="upstream_model",
        formal_reference_transform=next(
            camera.spatial_transform
            for camera in adapter.camera_inputs
            if camera.source_view == adapter.output.formal_view
        ),
        model_input_size_wh=adapter.native_input_size_wh,
        simulator_state={
            "model_config": model_config,
            "protocol": protocol,
            "conditioning_prompt_policy": "typed_robotics_prompt_manifest_v2",
            "conditioning_resize_policy_by_camera": {
                camera.input_key: camera.spatial_transform
                for camera in adapter.camera_inputs
            },
            "conditioning_frame_selection": (
                "full_reference_rollout_from_frame_zero_v1"
                if start_frame_index is None
                else "explicit_simulator_candidate_v1"
            ),
        },
    )
    return manifest.write_json(output / "input_manifest.json")


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open WAM video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if fps <= 0 or len(frames) < 2:
        raise RuntimeError(f"WAM video has invalid timing: {path}")
    return frames, fps


def finalize_output(
    input_manifest: str | Path,
    raw_video: str | Path,
    output_video: str | Path,
    metadata_path: str | Path,
) -> Path:
    input_path = Path(input_manifest).resolve()
    spec = WAMInputManifest.model_validate_json(input_path.read_text(encoding="utf-8"))
    if spec.wam != "lingbot_va_base":
        raise ValueError("only the validated LingBot-VA base profile is executable")
    _prompt_path, prompt_contract = load_bound_robotics_prompt(
        input_path,
        schema_version=spec.schema_version,
        reference_manifest=spec.reference_manifest,
        reference_manifest_sha256=spec.reference_manifest_sha256,
        reference_snapshot=spec.reference_snapshot,
        reference_task_snapshot=spec.reference_task_snapshot,
        reference_task_sha256=spec.reference_task_sha256,
        case_id=spec.case_id,
        task_instruction=spec.task_instruction,
        wam=spec.wam,
        prompt_manifest=spec.prompt_manifest,
        prompt_manifest_sha256=spec.prompt_manifest_sha256,
        rendered_prompt=spec.model_conditioning_prompt,
    )
    _adapter_path, adapter = load_adapter_contract(
        input_path, spec.adapter_contract, spec.adapter_contract_sha256
    )
    adapter.assert_input_binding(
        wam=spec.wam,
        source_model=spec.source_model,
        source_views=spec.source_views,
        formal_reference_view=spec.formal_reference_view,
        reference_start_frame_index=spec.reference_start_frame_index,
        reference_frame_count=spec.reference_frame_count,
        model_input_size_wh=spec.model_input_size_wh,
    )
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    metadata_file = Path(metadata_path).resolve()
    runtime_value = metadata.get("runtime_identity")
    if not isinstance(runtime_value, str) or Path(runtime_value).is_absolute():
        raise ValueError("LingBot metadata lacks a bundle-relative runtime identity")
    runtime_path = (metadata_file.parent / runtime_value).resolve()
    try:
        runtime_path.relative_to(input_path.parent)
    except ValueError as exc:
        raise ValueError("LingBot runtime identity escapes its formal bundle") from exc
    runtime_identity = json.loads(runtime_path.read_text(encoding="utf-8"))
    identity = runtime_identity.get("identity")
    if (
        runtime_identity.get("schema_version") != "wam_runtime_identity.v1"
        or not isinstance(identity, dict)
        or identity.get("profile") != "lingbot_va_base"
        or identity.get("source_model") != spec.source_model
        or runtime_identity.get("identity_sha256")
        != metadata.get("runtime_identity_sha256")
        or identity.get("checkpoint", {}).get("resolved_revision")
        != metadata.get("checkpoint_revision")
    ):
        raise ValueError("LingBot metadata runtime identity changed")
    if metadata.get("source_model") != spec.source_model:
        raise ValueError("loaded LingBot model does not match the input manifest")
    if metadata.get("config_name") != adapter.upstream_config:
        raise ValueError("LingBot metadata config differs from the frozen adapter")
    if metadata.get("video_decode_mode") != "upstream_monolithic_causal_vae_aligned_prefix":
        raise ValueError("LingBot formal rollout must use one monolithic causal-VAE decode")
    if metadata.get("benchmark_target") != "imagination_video_only" or metadata.get("actions_scored") is not False:
        raise ValueError("LingBot metadata must preserve video-only benchmark semantics")
    if metadata.get("prompt") != prompt_contract.rendered_prompt:
        raise ValueError("LingBot inference prompt differs from its typed prompt contract")
    if metadata.get("rollout_length_source") != "upstream_model_config":
        raise ValueError("formal LingBot rollout length must come from its upstream config")
    native_fps = float(metadata.get("native_video_fps") or 0.0)
    if abs(native_fps - adapter.temporal.native_output_fps) > 1e-6:
        raise ValueError("LingBot metadata does not use the adapter's native 10 fps")
    termination_reason = metadata.get("termination_reason")
    task_success = metadata.get("task_success")
    if termination_reason != "upstream_stopped" or task_success is not False:
        raise ValueError(
            "LingBot base must preserve its upstream-config stop without inventing task success"
        )

    raw = Path(raw_video).resolve()
    frames, fps = _read_video(raw)
    height, width = frames[0].shape[:2]
    formal_width, formal_height = adapter.output.formal_size_wh
    if adapter.output.raw_layout == "top_then_wrist_horizontal":
        expected_raw_size = (formal_width * 2, formal_height)
        crop_xywh = (0, 0, formal_width, formal_height)
        selected = [frame[:, :formal_width] for frame in frames]
    elif adapter.output.raw_layout == "left_right_wrist_above_high":
        expected_raw_size = (formal_width, formal_height + formal_height // 2)
        crop_xywh = (0, formal_height // 2, formal_width, formal_height)
        selected = [frame[formal_height // 2 :, :] for frame in frames]
    else:
        raise ValueError(f"unsupported LingBot output layout: {adapter.output.raw_layout}")
    if (width, height) != expected_raw_size:
        raise ValueError(
            f"LingBot {adapter.upstream_config} output must be {expected_raw_size}, "
            f"got {(width, height)}"
        )
    if abs(fps - adapter.temporal.native_output_fps) > 1e-6:
        raise ValueError("LingBot output FPS differs from its frozen adapter profile")
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise ValueError("LingBot output changes spatial size")
    expected_frames = int(round(float(metadata.get("actual_duration_sec") or 0) * fps))
    if expected_frames < 2:
        raise ValueError("LingBot actual timeline is empty")
    if len(selected) != expected_frames:
        raise ValueError(
            "LingBot decoded video differs from its task-terminated timeline"
        )
    destination = Path(output_video).resolve()
    write_h264_rgb(
        destination,
        (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in selected),
        fps,
        crf=16,
    )
    formal_camera_key = next(
        camera.input_key
        for camera in adapter.camera_inputs
        if camera.source_view == adapter.output.formal_view
    )
    conditioning = cv2.imread(
        str(resolve_artifact(input_path, spec.input_images[formal_camera_key])),
        cv2.IMREAD_COLOR,
    )
    if conditioning is None:
        raise RuntimeError("LingBot top conditioning image cannot be decoded")
    if conditioning.shape[:2] != (formal_height, formal_width):
        raise ValueError("LingBot conditioning image changed after adapter preparation")
    mae = float(np.mean(np.abs(selected[0].astype(np.float32) - conditioning.astype(np.float32))))
    quality_path = certify_wam_output(
        destination,
        input_path.parent / "quality_certificate.json",
        expected_width=formal_width,
        expected_height=formal_height,
        expected_frame_count=len(selected),
        expected_fps=fps,
    )
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality.get("passed") is not True:
        raise ValueError(f"LingBot imagination quality gate failed: {quality_path}")
    raw_certificate = write_raw_video_certificate(
        bundle_root=input_path.parent,
        wam="lingbot_va_base",
        raw_video=raw,
        model_metadata=metadata_file,
        selected_view=adapter.output.formal_view,
        crop_xywh=crop_xywh,
        extraction_policy=adapter.output.extraction_policy,
    )
    result = WAMOutputManifest(
        case_id=spec.case_id,
        wam=spec.wam,
        source_model=spec.source_model,
        checkpoint_revision=str(metadata.get("checkpoint_revision") or "unknown"),
        input_manifest=input_path.name,
        adapter_contract_sha256=spec.adapter_contract_sha256,
        model_metadata=str(metadata_file.relative_to(input_path.parent)),
        raw_video=str(raw.relative_to(input_path.parent)),
        raw_video_certificate=raw_certificate.name,
        selected_view=adapter.output.formal_view,
        crop_xywh=crop_xywh,
        aligned_video=str(destination.relative_to(input_path.parent)),
        width=formal_width,
        height=formal_height,
        fps=fps,
        frame_count=len(selected),
        duration_sec=len(selected) / fps,
        termination_reason=termination_reason,
        task_success=bool(task_success),
        conditioning_frame_mae_8bit=mae,
        quality_certificate=quality_path.name,
        quality_passed=True,
        **discover_preflight_artifacts(input_path.parent),
        bundle_artifacts_sha256=build_bundle_artifact_digests(input_path.parent),
    )
    return result.write_json(input_path.parent / "output_manifest.json")


def finalize_run(run_root: str | Path) -> Path:
    root = Path(run_root).resolve()
    records: list[dict[str, Any]] = []
    wams: set[str] = set()
    source_models: set[str] = set()
    for path in sorted(root.glob("*/output_manifest.json")):
        manifest = WAMOutputManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.wam == "lingbot_va_base":
            records.append({"case_id": manifest.case_id, "manifest": str(path)})
            wams.add(manifest.wam)
            source_models.add(manifest.source_model)
    if not records:
        raise FileNotFoundError(f"no completed LingBot cases below {root}")
    if len(wams) != 1 or len(source_models) != 1:
        raise ValueError("one LingBot run manifest may not mix model profiles")
    return write_json(
        root / "wam_run_manifest.json",
        {
            "schema_version": "wam_run.v2",
            "wam": next(iter(wams)),
            "source_model": next(iter(source_models)),
            "case_count": len(records),
            "cases": records,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("reference_manifest")
    prepare.add_argument("output_root")
    prepare.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    prepare.add_argument("--model-config")
    prepare.add_argument("--start-frame-index", type=int)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("input_manifest")
    finalize.add_argument("raw_video")
    finalize.add_argument("output_video")
    finalize.add_argument("metadata")
    run = sub.add_parser("finalize-run")
    run.add_argument("run_root")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_input(
            args.reference_manifest,
            args.output_root,
            source_model=args.source_model,
            model_config=args.model_config,
            start_frame_index=args.start_frame_index,
        )
    elif args.command == "finalize":
        result = finalize_output(args.input_manifest, args.raw_video, args.output_video, args.metadata)
    else:
        result = finalize_run(args.run_root)
    print(result)


if __name__ == "__main__":
    main()
