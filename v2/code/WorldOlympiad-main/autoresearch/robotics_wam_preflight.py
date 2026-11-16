"""Typed WAM compatibility and conditioning-observability preflight.

The compatibility report distinguishes native upstream support from an
intentional out-of-distribution benchmark.  The observability certificate is
bound to exact file digests and must pass before an expensive WAM is launched.
It never examines generated model output and therefore cannot cherry-pick a
benchmark result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from .robotics_contracts import (
    ConditioningDetection,
    ConditioningObservabilityCertificate,
    ConditioningViewObservation,
    WAMInputManifest,
    WAMConditioningCandidate,
    WAMConditioningSelection,
    WAMName,
    WAMProfileCompatibilityReport,
)
from .robotics_views import (
    load_reference_manifest,
    resolve_artifact,
    sha256_file,
)
from .robotics_prompts import load_bound_robotics_prompt
from .wam_adapter_contracts import (
    load_adapter_contract,
    profile_summary,
    reference_requirements_for_wam,
)


def _resolve_declared(owner: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (owner.parent / path).resolve()


def _bundle_relative(owner: Path, artifact: Path) -> str:
    try:
        return artifact.resolve().relative_to(owner.resolve().parent).as_posix()
    except ValueError as exc:
        raise ValueError(f"preflight artifact escapes its candidate bundle: {artifact}") from exc
from .wam_cosmos import prepare_input as prepare_cosmos
from .wam_lingbot import prepare_input as prepare_lingbot


PROFILE_REGISTRY: dict[str, dict[str, Any]] = {
    wam: profile_summary(wam)
    for wam in (
        "lingbot_va_base",
        "lingbot_va_robotwin",
        "cosmos3_nano_droid",
    )
}

ROLE_SIZE_CONTRACTS_BY_WAM = {
    wam: {
        item.role: (item.minimum_linear_fraction, 1.0)
        for camera in reference_requirements_for_wam(wam).cameras
        for item in camera.required_roles
    }
    for wam in PROFILE_REGISTRY
}
SCORE_THRESHOLD = 0.40
MAX_BLACK_FRACTION = 0.02
MIN_LUMINANCE_STDDEV = 10.0


def assess_profile(
    reference_manifest: str | Path, wam: WAMName
) -> WAMProfileCompatibilityReport:
    reference_path, reference = load_reference_manifest(reference_manifest)
    profile = PROFILE_REGISTRY[wam]
    blocked = wam == "lingbot_va_robotwin"
    cosmos_native = (
        wam == "cosmos3_nano_droid"
        and reference.robot_embodiment == "franka_robotiq_2f85"
    )
    limitations = [] if cosmos_native else list(profile["limitations"])
    if wam == "cosmos3_nano_droid" and not cosmos_native:
        limitations.append(
            "This reference does not use the registered Franka+Robotiq DROID embodiment."
        )
    return WAMProfileCompatibilityReport(
        case_id=reference.case_id,
        wam=wam,
        source_model=profile["source_model"],
        adapter_profile=profile["adapter_profile"],
        reference_manifest=str(reference_path),
        reference_manifest_sha256=sha256_file(reference_path),
        reference_robot=reference.robot_embodiment,
        reference_end_effector=(
            "robotiq_2f_85"
            if reference.robot_embodiment == "franka_robotiq_2f85"
            else "panda_hand_parallel_fingers"
        ),
        published_embodiment=profile["published_embodiment"],
        required_camera_mapping=profile["camera_mapping"],
        state_conventions=profile["state_conventions"],
        compatibility_status=(
            "blocked" if blocked else "native_compatible" if cosmos_native else "ood_stress"
        ),
        inference_allowed=not blocked,
        benchmark_interpretation=(
            "Blocked: a single-Panda SceneCode reference cannot supply RoboTwin's "
            "native dual-arm high/left-wrist/right-wrist observations."
            if blocked
            else "Registered RoboLab/DROID embodiment and camera contract."
            if cosmos_native
            else "Explicit OOD imagination benchmark. A result may be scored, but must not "
            "be claimed as native embodiment compatibility or robot-control success."
        ),
        limitations=limitations,
        evidence=profile["evidence"],
    )


def prepare_conditioning(
    reference_manifest: str | Path,
    wam: WAMName,
    output_root: str | Path,
    *,
    start_frame_index: int | None = None,
) -> tuple[Path, Path]:
    output = Path(output_root).resolve()
    report = assess_profile(reference_manifest, wam)
    if not report.inference_allowed:
        output.mkdir(parents=True, exist_ok=False)
        profile_path = report.write_json(output / "profile_compatibility.json")
        raise ValueError(f"WAM profile is blocked: {profile_path}")
    input_path = (
        prepare_lingbot(
            reference_manifest,
            output,
            source_model=report.source_model,
            start_frame_index=start_frame_index,
        )
        if wam in {"lingbot_va_base", "lingbot_va_robotwin"}
        else prepare_cosmos(
            reference_manifest,
            output,
            source_model=report.source_model,
            start_frame_index=start_frame_index,
        )
    )
    profile_path = report.write_json(output / "profile_compatibility.json")
    return input_path, profile_path


def _validate_typed_prompt(
    input_path: Path, spec: WAMInputManifest
) -> None:
    load_bound_robotics_prompt(
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


def candidate_frame_indices(
    reference_manifest: str | Path, wam: WAMName
) -> list[int]:
    # Conditioning must use the certified initial observation. Rollout duration
    # remains wholly owned by the upstream WAM/runtime.
    load_reference_manifest(reference_manifest)
    return [0]


def select_conditioning(
    reference_manifest: str | Path,
    wam: WAMName,
    certificate_paths: list[str | Path],
    output_path: str | Path,
) -> Path:
    reference_path, reference = load_reference_manifest(reference_manifest)
    selection_path = Path(output_path).resolve()
    expected_frames = candidate_frame_indices(reference_path, wam)
    if len(certificate_paths) != len(expected_frames):
        raise ValueError("conditioning selection did not receive every candidate")
    candidates: list[WAMConditioningCandidate] = []
    for expected_frame, raw_path in zip(expected_frames, certificate_paths):
        path = Path(raw_path).resolve()
        certificate = ConditioningObservabilityCertificate.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        input_path = _resolve_declared(path, certificate.input_manifest)
        profile_path = _resolve_declared(path, certificate.profile_report)
        spec = WAMInputManifest.model_validate_json(
            input_path.read_text(encoding="utf-8")
        )
        _validate_typed_prompt(input_path, spec)
        if certificate.case_id != reference.case_id or certificate.wam != wam:
            raise ValueError("conditioning candidate belongs to a different case")
        if spec.reference_start_frame_index != expected_frame:
            raise ValueError("conditioning candidate frame ordering changed")
        candidates.append(
            WAMConditioningCandidate(
                frame_index=expected_frame,
                input_manifest=_bundle_relative(selection_path, input_path),
                profile_report=_bundle_relative(selection_path, profile_path),
                observability_certificate=_bundle_relative(selection_path, path),
                passed=certificate.passed,
                blockers=certificate.blockers,
            )
        )
    selected = next((item for item in candidates if item.passed), None)
    blockers = [] if selected else [
        f"frame {item.frame_index}: {blocker}"
        for item in candidates
        for blocker in item.blockers
    ]
    selection = WAMConditioningSelection(
        case_id=reference.case_id,
        wam=wam,
        reference_manifest=str(reference_path),
        reference_manifest_sha256=sha256_file(reference_path),
        candidate_frame_indices=expected_frames,
        candidates=candidates,
        selected_frame_index=selected.frame_index if selected else None,
        selected_input_manifest=selected.input_manifest if selected else None,
        selected_profile_report=selected.profile_report if selected else None,
        selected_observability_certificate=(
            selected.observability_certificate if selected else None
        ),
        blockers=blockers,
        passed=selected is not None,
    )
    return selection.write_json(selection_path)


def verify_conditioning_selection(
    reference_manifest: str | Path,
    wam: WAMName,
    selection_path: str | Path,
) -> None:
    """Revalidate a published selection before reusing an existing preflight.

    This is the idempotency boundary used by the agentic harness.  It binds the
    selection to the exact selected reference and then re-runs the selected
    input/profile/observability checks, including the typed prompt contract.
    """

    reference_path, reference = load_reference_manifest(reference_manifest)
    path = Path(selection_path).resolve()
    selection = WAMConditioningSelection.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if selection.case_id != reference.case_id or selection.wam != wam:
        raise ValueError("conditioning selection belongs to another case or WAM")
    if Path(selection.reference_manifest).resolve() != reference_path:
        raise ValueError("conditioning selection points to another reference")
    if selection.reference_manifest_sha256 != sha256_file(reference_path):
        raise ValueError("conditioning selection reference digest changed")
    expected_frames = candidate_frame_indices(reference_path, wam)
    if selection.candidate_frame_indices != expected_frames:
        raise ValueError("conditioning selection candidate frames changed")
    if not selection.passed:
        raise ValueError("conditioning selection does not allow inference")
    if selection.selected_frame_index not in expected_frames:
        raise ValueError("conditioning selection has an invalid selected frame")
    selected_values = (
        selection.selected_input_manifest,
        selection.selected_profile_report,
        selection.selected_observability_certificate,
    )
    if any(value is None for value in selected_values):
        raise ValueError("conditioning selection lacks selected preflight artifacts")
    matching = [
        candidate
        for candidate in selection.candidates
        if candidate.frame_index == selection.selected_frame_index
    ]
    if len(matching) != 1 or not matching[0].passed:
        raise ValueError("conditioning selection has no unique passing candidate")
    selected_candidate = matching[0]
    candidate_values = (
        selected_candidate.input_manifest,
        selected_candidate.profile_report,
        selected_candidate.observability_certificate,
    )
    if tuple(str(value) for value in selected_values) != candidate_values:
        raise ValueError("conditioning selection summary differs from its candidate")
    input_path, profile_path, certificate_path = (
        _resolve_declared(path, str(value)) for value in selected_values
    )
    spec = WAMInputManifest.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )
    if spec.schema_version != "wam_input_bundle.v5":
        raise ValueError("canonical conditioning selection requires WAM input v5")
    if spec.case_id != reference.case_id or spec.wam != wam:
        raise ValueError("selected WAM input belongs to another case or WAM")
    if Path(spec.reference_manifest).resolve() != reference_path:
        raise ValueError("selected WAM input points to another reference")
    if spec.reference_manifest_sha256 != sha256_file(reference_path):
        raise ValueError("selected WAM input reference digest changed")
    verify_preflight(input_path, profile_path, certificate_path)


def _load_detection_report(path: Path, input_path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {
        "conditioning_detections.v1",
        "conditioning_detections.v2",
    }:
        raise ValueError("unsupported conditioning detection report")
    if _resolve_declared(path, str(payload.get("input_manifest", ""))) != input_path:
        raise ValueError("detection report belongs to a different input manifest")
    if payload.get("input_manifest_sha256") != sha256_file(input_path):
        raise ValueError("detection report input digest changed")
    if not isinstance(payload.get("images"), dict):
        raise ValueError("detection report is missing image observations")
    return payload


def _load_reference_admission(
    reference_manifest: Path,
    *,
    wam: str,
) -> dict[str, Any] | None:
    reference_path, reference = load_reference_manifest(reference_manifest)
    certificate_path = reference_path.parent / "reference_admission_certificate.json"
    if not certificate_path.is_file():
        return None
    expected = (reference.bundle_artifacts_sha256 or {}).get(
        "reference_admission_certificate.json"
    )
    if expected != sha256_file(certificate_path):
        raise ValueError("reference admission certificate is not digest-bound")
    payload = json.loads(certificate_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "model_native_reference_admission.v1"
        or payload.get("case_id") != reference.case_id
        or payload.get("passed") is not True
        or wam not in payload.get("requirements", {})
    ):
        raise ValueError("reference admission certificate is invalid for this WAM")
    return payload


def certify_observability(
    input_manifest: str | Path,
    profile_report: str | Path,
    detection_report: str | Path,
    output_path: str | Path,
) -> Path:
    input_path = Path(input_manifest).resolve()
    profile_path = Path(profile_report).resolve()
    detections_path = Path(detection_report).resolve()
    certificate_path = Path(output_path).resolve()
    spec = WAMInputManifest.model_validate_json(input_path.read_text(encoding="utf-8"))
    _validate_typed_prompt(input_path, spec)
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
    profile = WAMProfileCompatibilityReport.model_validate_json(
        profile_path.read_text(encoding="utf-8")
    )
    if profile.case_id != spec.case_id or profile.wam != spec.wam:
        raise ValueError("WAM profile and input manifest identify different cases")
    if profile.source_model != spec.source_model:
        raise ValueError("WAM profile and input manifest identify different checkpoints")
    if profile.adapter_profile != adapter.profile_id:
        raise ValueError("WAM profile and adapter contract identify different interfaces")
    if Path(profile.reference_manifest).resolve() != Path(spec.reference_manifest).resolve():
        raise ValueError("WAM profile and input manifest use different references")
    report = _load_detection_report(detections_path, input_path)
    reference_admission = _load_reference_admission(
        Path(spec.reference_manifest), wam=spec.wam
    )
    expected_keys = set(spec.input_images)
    if set(report["images"]) != expected_keys:
        raise ValueError("detector did not inspect the exact WAM camera keys")

    formal_key = next(
        item.input_key
        for item in adapter.camera_inputs
        if item.source_view == adapter.output.formal_view
    )
    observations: list[ConditioningViewObservation] = []
    blockers: list[str] = []
    requirements = adapter.required_roles
    input_sizes = adapter.input_sizes
    reference_requirements = reference_requirements_for_wam(spec.wam)
    requirement_by_view = {
        camera.source_view: camera for camera in reference_requirements.cameras
    }
    formal_role_size_contracts = ROLE_SIZE_CONTRACTS_BY_WAM[spec.wam]
    if spec.reference_task_snapshot is not None:
        task_path = _resolve_declared(input_path, spec.reference_task_snapshot)
        task = json.loads(task_path.read_text(encoding="utf-8"))
        object_roles = {
            str(item["object_id"]): str(item["role"])
            for item in task.get("objects", [])
        }
    else:
        object_roles = {}
    detector_v2 = report["schema_version"] == "conditioning_detections.v2"
    for camera_key, image_artifact in spec.input_images.items():
        camera_requirement = requirement_by_view[spec.source_views[camera_key]]
        role_size_contracts = {
            item.role: (item.minimum_linear_fraction, 1.0)
            for item in camera_requirement.required_roles
        }
        # Cropping is only inadmissible for a role that has to be whole: a large
        # destination may extend past the frame edge as long as it is legible,
        # which is exactly the case for a self-standing host.
        role_whole_frame = {
            item.role: bool(item.require_complete_frame)
            for item in camera_requirement.required_roles
        }
        # Empty-role diagnostic views still record the profile-wide floors so
        # their evidence remains interpretable, but they never gate semantics.
        if not role_size_contracts:
            role_size_contracts = formal_role_size_contracts
        row = report["images"][camera_key]
        image_path = resolve_artifact(input_path, image_artifact)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"conditioning image cannot be decoded: {image_path}")
        height, width = image.shape[:2]
        if (width, height) != input_sizes[camera_key]:
            raise ValueError(
                f"{camera_key} size {(width, height)} violates adapter size {input_sizes[camera_key]}"
            )
        if _resolve_declared(detections_path, str(row.get("image", ""))) != image_path:
            raise ValueError(f"detector image path changed for {camera_key}")
        if row.get("image_sha256") != sha256_file(image_path):
            raise ValueError(f"detector image digest changed for {camera_key}")
        if tuple(row.get("size_wh", ())) != (width, height):
            raise ValueError(f"detector image size changed for {camera_key}")
        required = list(requirements[camera_key])
        candidates: list[ConditioningDetection] = []
        visible: set[str] = set()
        visible_object_ids: set[str] = set()
        for raw in row.get("detections", []):
            role = str(raw["role"])
            low, high = role_size_contracts.get(
                role, formal_role_size_contracts[role]
            )
            fraction = float(raw["linear_fraction"])
            x_min, y_min, x_max, y_max = (
                float(value) for value in raw["bbox_xyxy"]
            )
            fully_in_frame = (
                x_min > 0.5
                and y_min > 0.5
                and x_max < width - 0.5
                and y_max < height - 0.5
            )
            # There is deliberately no maximum object-size rejection.  A
            # correctly identified large pot or tray is valid if it is fully
            # visible; too-small observations stay inadmissible, and cropping
            # only disqualifies a role that must be whole.
            require_whole = role_whole_frame.get(role, True)
            inside = fraction >= low and (fully_in_frame or not require_whole)
            object_id = raw.get("object_id")
            candidate = ConditioningDetection(
                role=role,
                object_id=str(object_id) if object_id is not None else None,
                prompt=str(raw["prompt"]),
                score=float(raw["score"]),
                bbox_xyxy=(x_min, y_min, x_max, y_max),
                linear_fraction=fraction,
                fully_in_frame=fully_in_frame,
                inside_size_contract=inside,
            )
            candidates.append(candidate)
            if candidate.score >= SCORE_THRESHOLD and inside:
                visible.add(role)
                if candidate.object_id is not None:
                    visible_object_ids.add(candidate.object_id)
        deterministic_view = (
            reference_admission.get("views", {}).get(
                f"{spec.wam}:{spec.source_views[camera_key]}"
            )
            if reference_admission is not None
            else None
        )
        if deterministic_view is not None:
            if deterministic_view.get("passed") is not True:
                raise ValueError(
                    f"published deterministic reference evidence failed for {camera_key}"
                )
            visible.clear()
            visible_object_ids.clear()
            for object_id, measurement in deterministic_view.get(
                "measurements", {}
            ).items():
                if measurement.get("passed") is True:
                    visible.add(str(measurement["role"]))
                    visible_object_ids.add(str(object_id))
        required_object_ids = sorted(
            object_id
            for object_id, role in object_roles.items()
            if role in required
        )
        if "end_effector" in required:
            required_object_ids.append("__end_effector__")
        missing = [role for role in required if role not in visible]
        missing_objects = (
            [
                object_id
                for object_id in required_object_ids
                if object_id not in visible_object_ids
            ]
            if detector_v2 or deterministic_view is not None
            else []
        )
        black_fraction = float(row["black_fraction"])
        luminance_stddev = float(row["luminance_stddev"])
        image_quality = (
            black_fraction <= MAX_BLACK_FRACTION
            and luminance_stddev >= MIN_LUMINANCE_STDDEV
        )
        passed = not missing and not missing_objects and image_quality
        if black_fraction > MAX_BLACK_FRACTION:
            blockers.append(
                f"{camera_key}: black/padding fraction {black_fraction:.4f} exceeds {MAX_BLACK_FRACTION:.4f}"
            )
        if luminance_stddev < MIN_LUMINANCE_STDDEV:
            blockers.append(
                f"{camera_key}: luminance stddev {luminance_stddev:.3f} is below {MIN_LUMINANCE_STDDEV:.3f}"
            )
        for role in missing:
            blockers.append(
                f"{camera_key}: no {role} detection satisfies score>={SCORE_THRESHOLD:.2f} "
                f"with minimum linear coverage {role_size_contracts[role][0]:.3f} "
                "while remaining fully in frame"
            )
        for object_id in missing_objects:
            blockers.append(
                f"{camera_key}: required task object {object_id!r} is not independently "
                "detected above its role visibility floor and fully in frame"
            )
        observations.append(
            ConditioningViewObservation(
                camera_key=camera_key,
                source_view=spec.source_views[camera_key],
                image=_bundle_relative(certificate_path, image_path),
                image_sha256=sha256_file(image_path),
                size_wh=(width, height),
                black_fraction=black_fraction,
                luminance_stddev=luminance_stddev,
                detections=candidates,
                role_size_contracts=role_size_contracts,
                required_roles=required,
                visible_roles=sorted(visible),
                required_object_ids=required_object_ids,
                visible_object_ids=sorted(visible_object_ids),
                passed=passed,
            )
        )

    formal_observation = next(item for item in observations if item.camera_key == formal_key)
    checks = {
        "typed_prompt_contract_bound": True,
        "profile_allows_inference": profile.inference_allowed,
        "camera_mapping_exact": profile.required_camera_mapping == spec.source_views,
        "all_images_decodable_and_bound": True,
        "all_required_views_pass": all(item.passed for item in observations),
        "formal_view_contains_all_task_roles": all(
            role in formal_observation.visible_roles
            for role in ("manipuland", "destination", "end_effector")
        ),
    }
    if not profile.inference_allowed:
        blockers.append("profile compatibility report blocks inference")
    if profile.required_camera_mapping != spec.source_views:
        blockers.append("profile camera mapping differs from the input manifest")
    if not checks["formal_view_contains_all_task_roles"]:
        blockers.append(
            f"{formal_key}: formal scored view is not task-observable for all three roles"
        )
    passed = all(checks.values()) and all(item.passed for item in observations)
    certificate = ConditioningObservabilityCertificate(
        schema_version=(
            "conditioning_observability.v2"
            if detector_v2
            else "conditioning_observability.v1"
        ),
        case_id=spec.case_id,
        wam=spec.wam,
        input_manifest=_bundle_relative(certificate_path, input_path),
        input_manifest_sha256=sha256_file(input_path),
        profile_report=_bundle_relative(certificate_path, profile_path),
        profile_report_sha256=sha256_file(profile_path),
        detector=str(report["detector"]),
        detector_model_revision=str(report["detector_model_revision"]),
        visibility_authority=(
            "deterministic_renderer_labels_primary_semantic_detector_secondary"
            if reference_admission is not None
            else "semantic_detector_legacy"
        ),
        score_threshold=SCORE_THRESHOLD,
        role_size_contracts=formal_role_size_contracts,
        formal_camera_key=formal_key,
        views=observations,
        checks=checks,
        blockers=blockers,
        passed=passed,
    )
    return certificate.write_json(certificate_path)


def verify_preflight(
    input_manifest: str | Path,
    profile_report: str | Path,
    observability_certificate: str | Path,
) -> None:
    input_path = Path(input_manifest).resolve()
    profile_path = Path(profile_report).resolve()
    certificate_path = Path(observability_certificate).resolve()
    spec = WAMInputManifest.model_validate_json(input_path.read_text(encoding="utf-8"))
    _validate_typed_prompt(input_path, spec)
    _adapter_path, adapter = load_adapter_contract(
        input_path, spec.adapter_contract, spec.adapter_contract_sha256
    )
    profile = WAMProfileCompatibilityReport.model_validate_json(
        profile_path.read_text(encoding="utf-8")
    )
    certificate = ConditioningObservabilityCertificate.model_validate_json(
        certificate_path.read_text(encoding="utf-8")
    )
    if not profile.inference_allowed or not certificate.passed:
        raise ValueError("WAM preflight does not allow inference")
    if profile.case_id != spec.case_id or certificate.case_id != spec.case_id:
        raise ValueError("WAM preflight case binding changed")
    if profile.wam != spec.wam or certificate.wam != spec.wam:
        raise ValueError("WAM preflight model binding changed")
    if profile.adapter_profile != adapter.profile_id:
        raise ValueError("WAM preflight adapter profile changed")
    if _resolve_declared(certificate_path, certificate.input_manifest) != input_path:
        raise ValueError("observability certificate points to another WAM input")
    if _resolve_declared(certificate_path, certificate.profile_report) != profile_path:
        raise ValueError("observability certificate points to another profile report")
    if certificate.input_manifest_sha256 != sha256_file(input_path):
        raise ValueError("WAM input changed after observability certification")
    if certificate.profile_report_sha256 != sha256_file(profile_path):
        raise ValueError("WAM profile changed after observability certification")
    expected_images = {
        key: resolve_artifact(input_path, value)
        for key, value in spec.input_images.items()
    }
    observed_images = {
        item.camera_key: _resolve_declared(certificate_path, item.image)
        for item in certificate.views
    }
    if observed_images != expected_images:
        raise ValueError("observability certificate image bindings changed")
    for item in certificate.views:
        if item.image_sha256 != sha256_file(observed_images[item.camera_key]):
            raise ValueError(f"observability image digest changed: {item.camera_key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("reference_manifest")
    prepare.add_argument("wam", choices=sorted(PROFILE_REGISTRY))
    prepare.add_argument("output_root")
    prepare.add_argument("--start-frame-index", type=int)
    candidates = sub.add_parser("candidate-frames")
    candidates.add_argument("reference_manifest")
    candidates.add_argument("wam", choices=sorted(PROFILE_REGISTRY))
    certify = sub.add_parser("certify")
    certify.add_argument("input_manifest")
    certify.add_argument("profile_report")
    certify.add_argument("detection_report")
    certify.add_argument("output_path")
    verify = sub.add_parser("verify")
    verify.add_argument("input_manifest")
    verify.add_argument("profile_report")
    verify.add_argument("observability_certificate")
    verify_selection = sub.add_parser("verify-selection")
    verify_selection.add_argument("reference_manifest")
    verify_selection.add_argument("wam", choices=sorted(PROFILE_REGISTRY))
    verify_selection.add_argument("selection_path")
    select = sub.add_parser("select")
    select.add_argument("reference_manifest")
    select.add_argument("wam", choices=sorted(PROFILE_REGISTRY))
    select.add_argument("output_path")
    select.add_argument("certificates", nargs="+")
    args = parser.parse_args()
    if args.command == "prepare":
        input_path, profile_path = prepare_conditioning(
            args.reference_manifest,
            args.wam,
            args.output_root,
            start_frame_index=args.start_frame_index,
        )
        print(json.dumps({"input_manifest": str(input_path), "profile_report": str(profile_path)}))
    elif args.command == "candidate-frames":
        for frame_index in candidate_frame_indices(args.reference_manifest, args.wam):
            print(frame_index)
    elif args.command == "certify":
        path = certify_observability(
            args.input_manifest,
            args.profile_report,
            args.detection_report,
            args.output_path,
        )
        print(path)
        certificate = ConditioningObservabilityCertificate.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if not certificate.passed:
            raise SystemExit(3)
    elif args.command == "verify":
        verify_preflight(
            args.input_manifest, args.profile_report, args.observability_certificate
        )
        print("preflight verified")
    elif args.command == "verify-selection":
        verify_conditioning_selection(
            args.reference_manifest, args.wam, args.selection_path
        )
        print("conditioning selection verified")
    else:
        path = select_conditioning(
            args.reference_manifest, args.wam, args.certificates, args.output_path
        )
        print(path)
        selection = WAMConditioningSelection.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if not selection.passed:
            raise SystemExit(4)


if __name__ == "__main__":
    main()
