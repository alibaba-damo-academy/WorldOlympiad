"""Content-address and validate one immutable formal WAM output bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

import cv2
import numpy as np

from .cosmos3_droid_window import restore_decoded_droid_panels
from .io import sha256_file, write_json
from .robotics_contracts import WAMInputManifest, WAMOutputManifest
from .robotics_prompts import load_bound_robotics_prompt
from .robotics_views import resize_with_pad
from .wam_adapter_contracts import (
    COSMOS_SOURCE_MODEL,
    LINGBOT_SOURCE_MODEL,
    load_adapter_contract,
)


WAM = Literal["cosmos3_nano_droid", "lingbot_va_base"]
EXPECTED_SOURCE_MODEL: dict[str, str] = {
    "cosmos3_nano_droid": COSMOS_SOURCE_MODEL,
    "lingbot_va_base": LINGBOT_SOURCE_MODEL,
}


def _bundle_files(root: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"formal WAM bundles may not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"formal WAM bundles may contain only regular files: {path}")
        relative = path.relative_to(root).as_posix()
        if relative == "output_manifest.json":
            continue
        records[relative] = sha256_file(path)
    if not records:
        raise ValueError("formal WAM bundle has no artifacts")
    return records


def build_bundle_artifact_digests(root: str | Path) -> dict[str, str]:
    """Hash every bundle file except the output manifest that embeds the map."""

    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return _bundle_files(resolved)


def discover_preflight_artifacts(root: str | Path) -> dict[str, str | None]:
    """Bind the complete local preflight set when finalizing a harness run."""

    bundle = Path(root).resolve()
    names = {
        "profile_report": "profile_compatibility.json",
        "observability_certificate": "conditioning_observability_certificate.json",
        "detection_report": "detection_report.json",
    }
    present = {key: (bundle / name).is_file() for key, name in names.items()}
    if any(present.values()) and not all(present.values()):
        raise ValueError("WAM finalization found an incomplete preflight evidence set")
    return {
        key: name if all(present.values()) else None for key, name in names.items()
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _video_evidence(path: Path) -> tuple[np.ndarray, int, int, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to decode formal WAM video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    first: np.ndarray | None = None
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if first is None:
            first = frame
        count += 1
    capture.release()
    if first is None or fps <= 0 or count < 2:
        raise RuntimeError(f"formal WAM video has invalid timing: {path}")
    height, width = first.shape[:2]
    return first, width, height, fps, count


def write_raw_video_certificate(
    *,
    bundle_root: str | Path,
    wam: WAM,
    raw_video: str | Path,
    model_metadata: str | Path,
    selected_view: str,
    crop_xywh: tuple[int, int, int, int],
    extraction_policy: str,
) -> Path:
    """Bind raw model media to the exact hashed metadata that produced it."""

    root = Path(bundle_root).resolve()
    raw = Path(raw_video).resolve()
    metadata = Path(model_metadata).resolve()
    try:
        raw_relative = raw.relative_to(root).as_posix()
        metadata_relative = metadata.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("raw WAM provenance must remain inside its bundle") from exc
    first, width, height, fps, frame_count = _video_evidence(raw)
    del first
    return write_json(
        root / "raw_video_certificate.json",
        {
            "schema_version": "wam_raw_video_provenance.v1",
            "wam": wam,
            "raw_video": raw_relative,
            "raw_video_sha256": sha256_file(raw),
            "model_metadata": metadata_relative,
            "model_metadata_sha256": sha256_file(metadata),
            "selected_view": selected_view,
            "crop_xywh": list(crop_xywh),
            "extraction_policy": extraction_policy,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
        },
    )


def _artifact(root: Path, value: str, *, label: str) -> Path:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or not value
        or value != relative.as_posix()
        or ".." in relative.parts
        or "\\" in value
    ):
        raise ValueError(f"{label} must be one normalized bundle-relative path")
    resolved = (root / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its WAM bundle: {value}") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return resolved


def _owned_artifact(owner: Path, root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be one bundle-relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or value != relative.as_posix()
        or "\\" in value
    ):
        raise ValueError(f"{label} must remain bundle-relative")
    resolved = (owner.parent / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its immutable WAM bundle") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return resolved


def _derive_formal_frame(
    *,
    root: Path,
    raw_video: Path,
    spec: WAMInputManifest,
    adapter: object,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], dict[str, float | int]]:
    first, width, height, fps, frame_count = _video_evidence(raw_video)
    temporal = getattr(adapter, "temporal")
    output = getattr(adapter, "output")
    if abs(fps - float(temporal.native_output_fps)) > 1e-6:
        raise ValueError("raw WAM video FPS differs from its adapter")
    if spec.wam == "lingbot_va_base":
        formal_width, formal_height = output.formal_size_wh
        if output.raw_layout != "top_then_wrist_horizontal" or (
            width,
            height,
        ) != (formal_width * 2, formal_height):
            raise ValueError("LingBot raw video layout differs from its adapter")
        crop = (0, 0, formal_width, formal_height)
        selected = first[:, :formal_width]
        formal_key = next(
            camera.input_key
            for camera in getattr(adapter, "camera_inputs")
            if camera.source_view == output.formal_view
        )
        conditioning = cv2.imread(
            str(_artifact(root, spec.input_images[formal_key], label="formal conditioning image")),
            cv2.IMREAD_COLOR,
        )
        if conditioning is None or conditioning.shape[:2] != (
            formal_height,
            formal_width,
        ):
            raise ValueError("LingBot formal conditioning image changed")
    else:
        if output.formal_view != "exterior_left":
            raise ValueError("Cosmos formal view differs from its adapter")
        if (width, height) == (320, 180):
            crop = (0, 0, 320, 180)
            selected = first
        elif (width, height) == (640, 360):
            crop = (0, 0, 640, 360)
            selected = first
        elif width == 640 and height in {528, 540}:
            _wrist, left, _right, _restoration = restore_decoded_droid_panels(
                first[None]
            )
            crop = (0, 360, 320, height - 360)
            selected = left[0]
        else:
            raise ValueError("Cosmos raw video layout differs from its adapter")
        formal_width, formal_height = output.formal_size_wh
        if selected.shape[:2] != (formal_height, formal_width):
            selected = resize_with_pad(selected, formal_width, formal_height)
        conditioning = cv2.imread(
            str(
                _artifact(
                    root,
                    spec.input_images[output.formal_view],
                    label="formal conditioning image",
                )
            ),
            cv2.IMREAD_COLOR,
        )
        if conditioning is None:
            raise ValueError("Cosmos formal conditioning image changed")
        conditioning = resize_with_pad(conditioning, formal_width, formal_height)
    return selected, conditioning, crop, {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
    }


def validate_wam_bundle(
    output_manifest: str | Path,
    *,
    expected_wam: WAM | None = None,
    certified_preflight: tuple[str | Path, str | Path, str | Path] | None = None,
) -> Path:
    """Validate all formal files, bindings, and digests without modifying them."""

    manifest_path = Path(output_manifest).resolve()
    root = manifest_path.parent
    output = WAMOutputManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if output.schema_version != "wam_output_bundle.v4":
        raise ValueError("formal WAM publication requires output bundle v4")
    if expected_wam is not None and output.wam != expected_wam:
        raise ValueError("published WAM output belongs to another adapter")
    if output.source_model != EXPECTED_SOURCE_MODEL[output.wam]:
        raise ValueError("published WAM source model is not the registered checkpoint")
    current_files = _bundle_files(root)
    if output.bundle_artifacts_sha256 != current_files:
        missing = sorted(set(output.bundle_artifacts_sha256) - set(current_files))
        extra = sorted(set(current_files) - set(output.bundle_artifacts_sha256))
        changed = sorted(
            path
            for path in set(output.bundle_artifacts_sha256) & set(current_files)
            if output.bundle_artifacts_sha256[path] != current_files[path]
        )
        raise ValueError(
            f"WAM bundle artifact map changed: missing={missing}, extra={extra}, "
            f"changed={changed}"
        )

    input_path = _artifact(root, output.input_manifest, label="input manifest")
    spec = WAMInputManifest.model_validate_json(input_path.read_text(encoding="utf-8"))
    if spec.schema_version != "wam_input_bundle.v5":
        raise ValueError("formal WAM publication requires typed input bundle v5")
    if (
        output.case_id != spec.case_id
        or output.wam != spec.wam
        or output.source_model != spec.source_model
        or output.adapter_contract_sha256 != spec.adapter_contract_sha256
    ):
        raise ValueError("WAM output identity differs from its certified input")
    prompt_path, _prompt = load_bound_robotics_prompt(
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
    adapter_path, adapter = load_adapter_contract(
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
    reference_snapshot = _artifact(
        root, str(spec.reference_snapshot), label="reference snapshot"
    )
    reference_task_snapshot = _artifact(
        root, str(spec.reference_task_snapshot), label="reference task snapshot"
    )
    for label, path in {
        "input manifest": input_path,
        "prompt manifest": prompt_path,
        "adapter contract": adapter_path,
        "reference snapshot": reference_snapshot,
        "reference task snapshot": reference_task_snapshot,
    }.items():
        relative = path.relative_to(root).as_posix()
        if output.bundle_artifacts_sha256.get(relative) != sha256_file(path):
            raise ValueError(f"{label} is absent from the immutable artifact map")

    metadata_path = _artifact(root, output.model_metadata, label="model metadata")
    raw_video = _artifact(root, output.raw_video, label="raw video")
    raw_certificate_path = _artifact(
        root,
        str(output.raw_video_certificate),
        label="raw video certificate",
    )
    aligned_video = _artifact(root, output.aligned_video, label="aligned video")
    quality_path = _artifact(root, output.quality_certificate, label="quality certificate")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("model metadata must be a JSON object")
    if metadata.get("source_model", metadata.get("model")) not in {
        output.source_model,
        "Cosmos3-Nano-Policy-DROID",
    }:
        raise ValueError("model metadata source identity differs from output")
    runtime_path = _owned_artifact(
        metadata_path,
        root,
        metadata.get("runtime_identity"),
        label="runtime identity",
    )
    runtime_identity = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_payload = runtime_identity.get("identity")
    if (
        runtime_identity.get("schema_version") != "wam_runtime_identity.v1"
        or not isinstance(runtime_payload, dict)
        or runtime_identity.get("identity_sha256")
        != _canonical_sha256(runtime_payload)
        or runtime_identity.get("identity_sha256")
        != metadata.get("runtime_identity_sha256")
    ):
        raise ValueError("model runtime identity digest changed")
    runtime_verification = runtime_identity.get("verification")
    if not isinstance(runtime_verification, dict):
        raise ValueError("runtime identity lacks verified asset roots")
    runtime_revision = (runtime_payload.get("checkpoint") or {}).get(
        "resolved_revision"
    )
    if (
        not isinstance(runtime_revision, str)
        or runtime_payload.get("profile") != output.wam
        or runtime_payload.get("source_model") != output.source_model
        or output.checkpoint_revision != metadata.get("checkpoint_revision")
        or output.checkpoint_revision != runtime_revision
    ):
        raise ValueError("WAM checkpoint revision differs from runtime identity")
    if output.wam == "lingbot_va_base":
        if (
            metadata.get("source_model") != output.source_model
            or metadata.get("config_name") != adapter.upstream_config
            or metadata.get("prompt") != spec.model_conditioning_prompt
            or metadata.get("rollout_length_source") != "upstream_model_config"
            or metadata.get("termination_reason") != "upstream_stopped"
            or metadata.get("task_success") is not False
            or metadata.get("benchmark_target") != "imagination_video_only"
            or metadata.get("actions_scored") is not False
            or metadata.get("video_decode_mode")
            != "upstream_monolithic_causal_vae_aligned_prefix"
            or abs(
                float(metadata.get("native_video_fps") or 0.0)
                - adapter.temporal.native_output_fps
            )
            > 1e-6
        ):
            raise ValueError("LingBot metadata violates its upstream-owned contract")
        for field in ("rollout_video_output", "latent_output", "output"):
            _owned_artifact(
                metadata_path,
                root,
                metadata.get(field),
                label=f"LingBot metadata {field}",
            )
        checkpoint_root = (runtime_verification.get("checkpoint") or {}).get("root")
        if metadata.get("model_path") != checkpoint_root:
            raise ValueError("LingBot metadata model path is not its verified checkpoint")
        if _owned_artifact(
            metadata_path,
            root,
            metadata.get("rollout_video_output"),
            label="LingBot raw rollout video",
        ) != raw_video:
            raise ValueError("LingBot raw video differs from model metadata")
    else:
        plan_path = _owned_artifact(
            metadata_path,
            root,
            metadata.get("worldir_plan"),
            label="Cosmos metadata worldir_plan",
        )
        _owned_artifact(
            metadata_path,
            root,
            metadata.get("trajectory"),
            label="Cosmos metadata trajectory",
        )
        actual_frames = int(metadata.get("frame_count") or 0)
        official_max_steps = int(metadata.get("official_task_max_steps") or 0)
        reason = metadata.get("termination_reason")
        success = metadata.get("task_success")
        world_hash = metadata.get("worldir_hash")
        try:
            valid_world_hash = isinstance(world_hash, str) and len(world_hash) == 64
            if valid_world_hash:
                int(world_hash, 16)
        except ValueError:
            valid_world_hash = False
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        graph_runtime = spec.runtime_controller == "task_graph_runtime"
        legacy_graph = graph_runtime and spec.official_runtime_task == "worldolympiad/task_graph_fixed_horizon"
        expected_controller = ("task_graph_fixed_horizon" if legacy_graph else
            "task_graph_native_episode" if graph_runtime else "official_robolab_task")
        from .cosmos_evaluation_window import validate_binding
        fixed_window = validate_binding(metadata, spec.simulator_state)
        allowed_reasons = ({"evaluation_window_complete"} if fixed_window else {"reference_horizon_complete"} if legacy_graph else
            {"task_completed", "native_task_timeout"} if graph_runtime else
            {"task_completed", "official_task_timeout"})
        if (
            metadata.get("schema_version")
            != "cosmos_worldir_isaac_closed_loop.v3"
            or metadata.get("status") != "complete"
            or metadata.get("terminated_early") is not False
            or metadata.get("physics_stability_pass") is not True
            or metadata.get("action_source") != "cosmos3_nano_policy_droid"
            or metadata.get("rollout_controller") != expected_controller
            or metadata.get("official_runtime_task") != spec.official_runtime_task
            or metadata.get("prompt") != spec.model_conditioning_prompt
            or metadata.get("benchmark_target") != "imagination_video_only"
            or metadata.get("actions_scored") is not False
            or metadata.get("observation_feedback") != "simulator_closed_loop"
            or metadata.get("closed_loop_horizon_steps") != 32
            or metadata.get("refresh_horizon_steps") != 32
            or metadata.get("video_stitching_policy")
            != "fresh_condition_once_per_32_step_window_v1"
            or actual_frames < 2
            or metadata.get("action_count") != actual_frames - 1
            or metadata.get("actual_future_steps") != actual_frames - 1
            or output.frame_count != actual_frames
            or official_max_steps < actual_frames - 1
            or abs(
                float(metadata.get("native_action_hz") or 0.0)
                - adapter.temporal.native_output_fps
            )
            > 1e-6
            or abs(
                float(metadata.get("actual_duration_sec") or 0.0)
                - actual_frames / adapter.temporal.native_output_fps
            )
            > 1e-6
            or reason not in allowed_reasons
            or (legacy_graph and success is not False)
            or (not fixed_window and not legacy_graph and success is not (reason == "task_completed"))
            or (legacy_graph and actual_frames != official_max_steps + 1)
            or not valid_world_hash
            or plan.get("world_hash") != world_hash
        ):
            raise ValueError("Cosmos metadata violates its bound runtime contract")
        if reason in {"official_task_timeout", "native_task_timeout"} and actual_frames != official_max_steps + 1:
            raise ValueError("Cosmos timeout does not match the official task episode")
        if reason == "task_completed":
            evidence = metadata.get("completion_evidence")
            if (
                metadata.get("completion_frame") != actual_frames - 1
                or not isinstance(evidence, dict)
                or evidence.get("official_predicate_passed") is not True
            ):
                raise ValueError("Cosmos success lacks official terminal evidence")
        windows = metadata.get("windows")
        expected_window_count = (actual_frames - 1 + 31) // 32
        if (
            not isinstance(windows, list)
            or len(windows) != expected_window_count
            or metadata.get("window_count") != expected_window_count
        ):
            raise ValueError("Cosmos metadata lacks its window provenance")
        capture_root = metadata_path.parent
        executed_total = 0
        for index, window in enumerate(windows):
            if not isinstance(window, dict) or window.get("window_index") != index:
                raise ValueError("Cosmos metadata window ordering changed")
            executed_actions = int(window.get("executed_action_count") or 0)
            expected_planned = min(32, official_max_steps - executed_total)
            planned_actions = int(window.get("planned_action_count") or 0)
            if (
                window.get("conditioning_frame") != executed_total
                or executed_actions < 1
                or expected_planned < 1
                or planned_actions != expected_planned
                or executed_actions > planned_actions
                or (index < len(windows) - 1 and executed_actions != planned_actions)
                or (reason in {"official_task_timeout", "evaluation_window_complete"} and executed_actions != planned_actions)
            ):
                raise ValueError("Cosmos window action interval changed")
            model_metadata = window.get("model_metadata")
            if (
                not isinstance(model_metadata, dict)
                or model_metadata.get("schema_version")
                != "cosmos3_droid_server_window.v1"
                or model_metadata.get("action_source")
                != "official_robolab_websocket_server"
                or model_metadata.get("request_index") != index
            ):
                raise ValueError("Cosmos window lacks model metadata")
            window_root = capture_root / "windows" / f"window_{index:06d}"
            conditioning = _owned_artifact(
                window_root / "metadata.json",
                root,
                model_metadata.get("conditioning_image"),
                label=f"Cosmos window {index} conditioning image",
            )
            if output.bundle_artifacts_sha256.get(
                conditioning.relative_to(root).as_posix()
            ) != sha256_file(conditioning):
                raise ValueError(f"Cosmos window {index} conditioning digest changed")
            actions_path = window_root / "executed_actions.npy"
            video_path = window_root / "model" / "droid_concat_rollout.npy"
            if not actions_path.is_file() or not video_path.is_file():
                raise FileNotFoundError(f"Cosmos window {index} is incomplete")
            actions = np.load(actions_path, allow_pickle=False)
            video = np.load(video_path, mmap_mode="r", allow_pickle=False)
            if (
                actions.shape != (executed_actions, 8)
                or not np.isfinite(actions).all()
                or video.ndim != 4
                or video.shape[0] < executed_actions
                or video.shape[-1] != 3
            ):
                raise ValueError(f"Cosmos window {index} executable evidence changed")
            executed_total += executed_actions
        if executed_total != actual_frames - 1:
            raise ValueError("Cosmos windows do not cover the official task timeline")
    if (
        output.termination_reason != metadata.get("termination_reason")
        or output.task_success is not metadata.get("task_success")
    ):
        raise ValueError("WAM completion claim differs from model metadata")

    raw_certificate = json.loads(raw_certificate_path.read_text(encoding="utf-8"))
    if raw_certificate.get("schema_version") != "wam_raw_video_provenance.v1":
        raise ValueError("unsupported raw WAM video provenance")
    if raw_certificate.get("wam") != output.wam:
        raise ValueError("raw WAM video provenance belongs to another adapter")
    certificate_raw = _artifact(
        root, raw_certificate.get("raw_video", ""), label="certified raw video"
    )
    certificate_metadata = _artifact(
        root,
        raw_certificate.get("model_metadata", ""),
        label="certified model metadata",
    )
    if (
        certificate_raw != raw_video
        or raw_certificate.get("raw_video_sha256") != sha256_file(raw_video)
        or certificate_metadata != metadata_path
        or raw_certificate.get("model_metadata_sha256") != sha256_file(metadata_path)
        or raw_certificate.get("extraction_policy") != adapter.output.extraction_policy
    ):
        raise ValueError("raw WAM video provenance changed")
    selected, conditioning, expected_crop, raw_evidence = _derive_formal_frame(
        root=root,
        raw_video=raw_video,
        spec=spec,
        adapter=adapter,
    )
    if (
        output.selected_view != adapter.output.formal_view
        or raw_certificate.get("selected_view") != adapter.output.formal_view
        or tuple(output.crop_xywh) != expected_crop
        or tuple(raw_certificate.get("crop_xywh") or ()) != expected_crop
        or (output.width, output.height) != adapter.output.formal_size_wh
    ):
        raise ValueError("WAM formal view extraction differs from its adapter")
    for field, expected_value in raw_evidence.items():
        actual_value = raw_certificate.get(field)
        if isinstance(expected_value, float):
            if abs(float(actual_value or 0.0) - expected_value) > 1e-6:
                raise ValueError("raw WAM video timeline provenance changed")
        elif actual_value != expected_value:
            raise ValueError("raw WAM video shape provenance changed")
    if (
        raw_evidence["frame_count"] != output.frame_count
        or abs(float(raw_evidence["fps"]) - output.fps) > 1e-6
    ):
        raise ValueError("raw WAM video timeline differs from formal output")
    expected_mae = float(
        np.mean(
            np.abs(
                selected.astype(np.float32) - conditioning.astype(np.float32)
            )
        )
    )
    if abs(output.conditioning_frame_mae_8bit - expected_mae) > 1e-6:
        raise ValueError("WAM conditioning-frame MAE was not derived from raw media")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality.get("schema_version") not in {"robotics_wam_video_quality.v3","robotics_wam_video_quality.v4"}:
        raise ValueError("formal WAM quality certificate must be content-addressed v3 or v4")
    if quality.get("passed") is not True:
        raise ValueError("formal WAM quality certificate is not passing")
    video_report = quality.get("video")
    if not isinstance(video_report, dict):
        raise ValueError("formal WAM quality certificate has no video report")
    if quality.get("schema_version")=="robotics_wam_video_quality.v4":
        checks=video_report.get("checks",{})
        if (not checks or not all(value is True for value in checks.values())
                or quality.get("policy",{}).get("expected_frame_count")!=output.frame_count
                or quality.get("policy",{}).get("expected_fps")!=output.fps):
            raise ValueError("WAM media integrity does not match the declared output clock")
    if video_report.get("path") != aligned_video.relative_to(root).as_posix():
        raise ValueError("quality certificate points to another formal video")
    if video_report.get("sha256") != sha256_file(aligned_video):
        raise ValueError("quality certificate formal video digest changed")
    if (
        int(video_report.get("width") or 0) != output.width
        or int(video_report.get("height") or 0) != output.height
        or int(video_report.get("frame_count") or 0) != output.frame_count
        or abs(float(video_report.get("fps") or 0.0) - output.fps) > 1e-6
        or abs(float(video_report.get("duration_sec") or 0.0) - output.duration_sec)
        > 1e-6
    ):
        raise ValueError("quality certificate timeline differs from output manifest")
    for label, path in {
        "model metadata": metadata_path,
        "raw video": raw_video,
        "raw video certificate": raw_certificate_path,
        "aligned video": aligned_video,
        "quality certificate": quality_path,
    }.items():
        relative = path.relative_to(root).as_posix()
        if output.bundle_artifacts_sha256.get(relative) != sha256_file(path):
            raise ValueError(f"{label} digest differs from the immutable artifact map")

    declared_preflight = (
        output.profile_report,
        output.observability_certificate,
        output.detection_report,
    )
    if all(value is not None for value in declared_preflight):
        # Imported lazily because input preparation imports the finalizers.
        from .robotics_wam_preflight import verify_preflight

        profile_path = _artifact(
            root, str(output.profile_report), label="profile report"
        )
        certificate_path = _artifact(
            root,
            str(output.observability_certificate),
            label="observability certificate",
        )
        detection_path = _artifact(
            root, str(output.detection_report), label="detection report"
        )
        published = (input_path, profile_path, certificate_path)
        verify_preflight(*published)
        detections = json.loads(detection_path.read_text(encoding="utf-8"))
        detection_schema = detections.get("schema_version")
        if detection_schema not in {
            "conditioning_detections.v1", "conditioning_detections.v2"
        }:
            raise ValueError("published conditioning detection schema changed")
        if detection_schema == "conditioning_detections.v2":
            # The current GroundingDINO report binds the exact typed input and
            # registered detector revision; keep the historical v1 path valid.
            import re
            if (detections.get("input_manifest_sha256") != sha256_file(input_path)
                    or not re.fullmatch(
                        r"[0-9a-f]{40}", str(detections.get("detector_model_revision", ""))
                    )):
                raise ValueError("published conditioning detection v2 binding changed")
        if _owned_artifact(
            detection_path,
            root,
            detections.get("input_manifest"),
            label="conditioning detection input",
        ) != input_path:
            raise ValueError("conditioning detections point to another WAM input")
        image_rows = detections.get("images")
        if not isinstance(image_rows, dict) or set(image_rows) != set(spec.input_images):
            raise ValueError("conditioning detections changed camera keys")
        for key, row in image_rows.items():
            if not isinstance(row, dict):
                raise ValueError(f"conditioning detection row is invalid: {key}")
            image_path = _owned_artifact(
                detection_path,
                root,
                row.get("image"),
                label=f"conditioning detection image {key}",
            )
            if row.get("image_sha256") != sha256_file(image_path):
                raise ValueError(f"conditioning detection image changed: {key}")
        if certified_preflight is not None:
            sources = tuple(Path(value).resolve() for value in certified_preflight)
            verify_preflight(*sources)
            for source, copied in zip(sources, published, strict=True):
                if sha256_file(source) != sha256_file(copied):
                    raise ValueError(
                        f"published preflight input changed: {source.name}"
                    )
    elif certified_preflight is not None:
        raise ValueError("published WAM bundle does not declare its preflight evidence")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("output_manifest")
    validate.add_argument(
        "--expected-wam",
        choices=("cosmos3_nano_droid", "lingbot_va_base"),
    )
    validate.add_argument("--input-manifest")
    validate.add_argument("--profile-report")
    validate.add_argument("--observability-certificate")
    args = parser.parse_args()
    preflight_values = (
        args.input_manifest,
        args.profile_report,
        args.observability_certificate,
    )
    if any(value is not None for value in preflight_values) and not all(
        value is not None for value in preflight_values
    ):
        parser.error("all three certified preflight paths must be supplied together")
    print(
        validate_wam_bundle(
            args.output_manifest,
            expected_wam=args.expected_wam,
            certified_preflight=(
                tuple(preflight_values) if all(preflight_values) else None
            ),
        )
    )


if __name__ == "__main__":
    main()
