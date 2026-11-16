#!/usr/bin/env python3
"""Package one successful PyBullet rollout and SceneCode multiview replay."""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath

import cv2
import numpy as np

from autoresearch.io import write_json
from autoresearch.media import write_h264_rgb
from autoresearch.models import FreshGenerationCertificate
from autoresearch.robotics_contracts import (
    CameraViewArtifact,
    RoboticsReferenceManifest,
    TaskInstructionSpec,
    simulator_primitive_record,
)
from autoresearch.robotics_quality import (
    certify_reference_physical_render,
    certify_reference_views,
)
from autoresearch.robotics_visual_context import (
    certify_reference_visual_context,
)
from autoresearch.robotics_visual_revision_verifier import verify_visual_revision
from autoresearch.robotics_views import (
    build_reference_artifact_digests,
    load_reference_manifest,
    sha256_file,
)
from autoresearch.scenecode_visual_collision_alignment import (
    certify_reference_runtime_alignment,
)
from autoresearch.wam_adapter_contracts import reference_requirements_for_wam


def _operator_contact_render(
    trajectory: dict,
    render: dict,
    *,
    operator: str,
    trajectory_path: Path,
    render_path: Path,
    output: Path,
) -> dict:
    """Select real contact-phase masks without changing the source render.

    The renderer can emit masks during release while its articulated-contact
    predicate is still true.  Those frames are useful video, but are not
    evidence of the operator contact that the independent gate certifies.
    """
    if operator not in {"pull", "press", "close"}:
        return render
    fixed_views = ("top", "exterior_left", "exterior_right")
    views = render.get("views") or {}
    if any(view_id not in views for view_id in fixed_views):
        return render  # Let the protected gate report the missing view.
    phase_by_frame = {int(frame["frame"]): frame["phase"] for frame in trajectory["frames"]}
    original_ids = [
        {int(item["frame"]) for item in views[view_id].get("semantic_masks", [])}
        for view_id in fixed_views
    ]
    if any(ids != original_ids[0] for ids in original_ids[1:]):
        raise ValueError("fixed cameras disagree before contact-mask selection")
    if any(frame_id not in phase_by_frame for frame_id in original_ids[0]):
        raise ValueError("semantic evidence names a nonexistent trajectory frame")
    excluded = sorted(
        frame_id for frame_id in original_ids[0]
        if phase_by_frame[frame_id] not in {"contact", "actuate"}
    )
    if not excluded:
        return render
    selected = deepcopy(render)
    excluded_set = set(excluded)
    for view_id in fixed_views:
        selected["views"][view_id]["semantic_masks"] = [
            item for item in views[view_id]["semantic_masks"]
            if int(item["frame"]) not in excluded_set
        ]
    write_json(
        output / "semantic_evidence_selection.json",
        {
            "schema_version": "robotics_semantic_evidence_selection.v1",
            "operator": operator,
            "policy": "articulated_contact_and_actuate_phases_only",
            "trajectory_sha256": sha256_file(trajectory_path),
            "raw_render_manifest_sha256": sha256_file(render_path),
            "excluded_frames": [
                {"frame": frame_id, "phase": phase_by_frame[frame_id]}
                for frame_id in excluded
            ],
            "retained_frame_count": len(original_ids[0]) - len(excluded),
            "fixed_views": list(fixed_views),
        },
    )
    return selected


def _copy_bundle_directory(source: Path, destination: Path, *, label: str) -> None:
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{label} may not contain symlinks: {path}")
    shutil.copytree(source, destination)


def _mask_measurement(mask: np.ndarray, value: int) -> dict[str, object]:
    rows, columns = np.where(mask == value)
    height, width = mask.shape
    if rows.size == 0:
        return {
            "pixel_count": 0,
            "bbox_xyxy": None,
            "linear_fraction": 0.0,
            "fully_in_frame": False,
        }
    x_min = int(columns.min())
    x_max = int(columns.max())
    y_min = int(rows.min())
    y_max = int(rows.max())
    return {
        "pixel_count": int(rows.size),
        "bbox_xyxy": [x_min, y_min, x_max + 1, y_max + 1],
        "linear_fraction": float(
            max((x_max - x_min + 1) / width, (y_max - y_min + 1) / height)
        ),
        "fully_in_frame": bool(
            x_min > 0 and y_min > 0 and x_max < width - 1 and y_max < height - 1
        ),
    }


def _role_visible(measurement: dict[str, object], requirement: object) -> bool:
    """Apply the adapter's actual crop policy to a registered role mask."""
    return bool(
        float(measurement["linear_fraction"])
        >= float(requirement.minimum_linear_fraction)
        and (
            not requirement.require_complete_frame
            or measurement["fully_in_frame"]
        )
    )


def _certify_model_native_reference(
    *,
    task: TaskInstructionSpec,
    render: dict,
    output: Path,
) -> Path:
    """Certify frame-zero task visibility from renderer-owned object labels."""

    views: dict[str, dict] = {}
    blockers: list[str] = []
    requirements_payload: dict[str, dict] = {}
    task_objects = {
        item.object_id: item.role
        for item in task.objects
        if item.role in {"manipuland", "destination"}
    }
    for component in task.articulation_components:
        # Articulated components are task-visible manipulands even though the
        # SceneCode inventory stores their visual SDF on the host object.
        task_objects[component.component_id] = "manipuland"
        if component.host_object_id in {item.object_id for item in task.objects}:
            task_objects[component.host_object_id] = "destination"
    for wam in task.requested_wams:
        requirements = reference_requirements_for_wam(wam)
        requirements_payload[wam] = requirements.model_dump(
            mode="json", exclude_none=True
        )
        for camera in requirements.cameras:
            if not camera.required_roles:
                continue
            view_id = camera.source_view
            record = render.get("views", {}).get(view_id)
            if not isinstance(record, dict):
                blockers.append(f"{wam}/{view_id}: required native view is missing")
                continue
            semantic = record.get("conditioning_semantic")
            if not isinstance(semantic, dict):
                blockers.append(
                    f"{wam}/{view_id}: deterministic frame-zero semantic mask is missing"
                )
                continue
            mask_path = Path(str(semantic.get("image", ""))).resolve()
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2:
                blockers.append(
                    f"{wam}/{view_id}: deterministic frame-zero semantic mask is invalid"
                )
                continue
            if (mask.shape[1], mask.shape[0]) != camera.size_wh:
                blockers.append(
                    f"{wam}/{view_id}: mask size {(mask.shape[1], mask.shape[0])} "
                    f"does not equal native size {camera.size_wh}"
                )
                continue
            destination = output / "views" / view_id / "conditioning_semantic_t0.png"
            shutil.copy2(mask_path, destination)
            object_values = semantic.get("object_values") or {}
            end_effector_value = semantic.get("end_effector_value")
            role_requirements = {
                item.role: item for item in camera.required_roles
            }
            measurements: dict[str, dict[str, object]] = {}
            for object_id, role in task_objects.items():
                if role not in role_requirements:
                    continue
                value = object_values.get(object_id)
                if not isinstance(value, int):
                    blockers.append(
                        f"{wam}/{view_id}: renderer label is missing task object {object_id!r}"
                    )
                    continue
                measurement = _mask_measurement(mask, value)
                floor = role_requirements[role].minimum_linear_fraction
                passed = _role_visible(measurement, role_requirements[role])
                measurements[object_id] = {
                    "role": role,
                    "semantic_value": value,
                    "minimum_linear_fraction": floor,
                    **measurement,
                    "passed": passed,
                }
                if not passed:
                    blockers.append(
                        f"{wam}/{view_id}: {object_id!r} is too small, cropped, or occluded"
                    )
            if "end_effector" in role_requirements:
                if not isinstance(end_effector_value, int):
                    blockers.append(
                        f"{wam}/{view_id}: renderer end-effector label is missing"
                    )
                else:
                    measurement = _mask_measurement(mask, end_effector_value)
                    floor = role_requirements["end_effector"].minimum_linear_fraction
                    passed = _role_visible(measurement, role_requirements["end_effector"])
                    measurements["__end_effector__"] = {
                        "role": "end_effector",
                        "semantic_value": end_effector_value,
                        "minimum_linear_fraction": floor,
                        **measurement,
                        "passed": passed,
                    }
                    if not passed:
                        blockers.append(
                            f"{wam}/{view_id}: end effector is too small, cropped, or occluded"
                        )
            views[f"{wam}:{view_id}"] = {
                "wam": wam,
                "view_id": view_id,
                "size_wh": list(camera.size_wh),
                "mask": destination.relative_to(output).as_posix(),
                "mask_sha256": sha256_file(destination),
                "measurements": measurements,
                "passed": all(
                    bool(item.get("passed")) for item in measurements.values()
                ),
            }
    certificate = {
        "schema_version": "model_native_reference_admission.v1",
        "case_id": task.case_id,
        "task_sha256": task.content_hash(),
        "visibility_authority": (
            "deterministic_renderer_labels_primary_semantic_detector_secondary"
        ),
        "native_render_policy": "direct_native_size_no_rescale",
        "requirements": requirements_payload,
        "views": views,
        "blockers": blockers,
        "passed": not blockers and bool(views),
    }
    path = write_json(output / "reference_admission_certificate.json", certificate)
    if not certificate["passed"]:
        raise ValueError(f"model-native reference admission failed: {path}")
    return path


def _copy_agentic_provenance(
    *,
    task: TaskInstructionSpec,
    task_path: Path,
    task_graph: Path,
    generation_manifest: Path,
    identity_certificate: Path,
    output: Path,
) -> dict[str, str]:
    """Archive the immutable generic-harness evidence without a sample recipe."""

    for path in (task_graph, generation_manifest, identity_certificate):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    graph = json.loads(task_graph.read_text(encoding="utf-8"))
    generation = json.loads(generation_manifest.read_text(encoding="utf-8"))
    identity = json.loads(identity_certificate.read_text(encoding="utf-8"))
    if graph.get("case_id") != task.case_id:
        raise ValueError("task graph and task case ids differ")
    if graph.get("task_sha256") != task.content_hash():
        raise ValueError("task graph does not bind the frozen task content")
    task_file_sha256 = sha256_file(task_path)
    if (
        generation.get("case_id") != task.case_id
        or generation.get("task_sha256") != task_file_sha256
        or generation.get("passed") is not True
        or generation.get("semantic_identity_passed") is not True
    ):
        raise ValueError("SceneCode generation manifest is not admitted for this task")
    if identity.get("case_id") != task.case_id or identity.get("overall_passed") is not True:
        raise ValueError("semantic identity certificate is not passing for this task")
    declared_identity = (generation.get("asset_identity") or {}).get(
        "semantic_verdict_sha256"
    )
    if declared_identity != sha256_file(identity_certificate):
        raise ValueError("SceneCode generation and semantic identity digests differ")

    provenance = output / "agentic_provenance"
    provenance.mkdir()
    graph_out = provenance / "task_graph.json"
    generation_out = provenance / "scene_generation_manifest.json"
    shutil.copy2(task_graph, graph_out)
    shutil.copy2(generation_manifest, generation_out)
    identity_root = provenance / "asset_identity"
    _copy_bundle_directory(
        identity_certificate.parent,
        identity_root,
        label="semantic identity evidence",
    )
    identity_out = identity_root / identity_certificate.name
    return {
        "task_graph": graph_out.relative_to(output).as_posix(),
        "task_graph_sha256": sha256_file(graph_out),
        "scene_generation_manifest": generation_out.relative_to(output).as_posix(),
        "scene_generation_manifest_sha256": sha256_file(generation_out),
        "semantic_identity_certificate": identity_out.relative_to(output).as_posix(),
        "semantic_identity_certificate_sha256": sha256_file(identity_out),
    }


def _certified_same_asset_aliases(
    generation_manifest: Path | None, task: TaskInstructionSpec
) -> list[dict[str, object]]:
    """Bind task role aliases to the admitted SceneCode inventory, not names."""
    if generation_manifest is None:
        return []
    generation_manifest = generation_manifest.resolve()
    generation = json.loads(generation_manifest.read_text(encoding="utf-8"))
    inventory_path = Path(str(generation.get("task_inventory_certificate", ""))).resolve()
    if inventory_path.parent != generation_manifest.parent or not inventory_path.is_file():
        raise ValueError("generation inventory is outside its certified case")
    inventory_sha = sha256_file(inventory_path)
    if inventory_sha != generation.get("task_inventory_certificate_sha256"):
        raise ValueError("generation inventory digest changed")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("passed") is not True or inventory.get("case_id") != task.case_id:
        raise ValueError("generation inventory does not certify this task")
    task_ids = {item.object_id for item in task.objects}
    rows = {}
    for row in inventory.get("matches") or []:
        object_id = row.get("task_object_id")
        if object_id not in task_ids or object_id in rows:
            raise ValueError("generation inventory has an unknown or duplicate task object")
        sdf = Path(str(row.get("sdf_path", ""))).resolve()
        if not sdf.is_file() or sha256_file(sdf) != row.get("sdf_sha256"):
            raise ValueError("inventory asset changed before alias admission")
        rows[object_id] = row
    if set(rows) != task_ids:
        raise ValueError("generation inventory omits a task object")
    aliases = []
    for left_index, left in enumerate(sorted(rows)):
        for right in sorted(rows)[left_index + 1:]:
            a, b = rows[left], rows[right]
            if (a.get("scene_object_id") and
                    a.get("scene_object_id") == b.get("scene_object_id") and
                    a.get("sdf_sha256") == b.get("sdf_sha256")):
                aliases.append({
                    "object_ids": [left, right],
                    "scene_object_id": a["scene_object_id"],
                    "sdf_sha256": a["sdf_sha256"],
                    "inventory_sha256": inventory_sha,
                })
    return aliases


def _material_authority(identity_certificate: Path | None) -> dict[str, dict[str, object]]:
    """Return the per-object studio identity verdict as the material authority.

    The in-context close-ups are magnified crops of model-native planes, so the
    studio identity contact sheet stays the evidence of record for material
    authenticity while the in-context review owns context, separation and
    proportion.  A missing certificate yields an empty authority, which leaves
    every object fail-closed.
    """

    if identity_certificate is None or not identity_certificate.is_file():
        return {}
    certificate = json.loads(identity_certificate.read_text(encoding="utf-8"))
    authority: dict[str, dict[str, object]] = {}
    for entry in certificate.get("objects") or []:
        object_id = entry.get("object_id")
        if not isinstance(object_id, str):
            continue
        authority[object_id] = {
            "passed": entry.get("passed") is True,
            "evidence": str(entry.get("evidence", "")),
            "visual_description": entry.get("visual_description"),
            "contact_sheet": entry.get("contact_sheet"),
            "contact_sheet_sha256": entry.get("contact_sheet_sha256"),
        }
    return authority


def _replace_json_path_prefix(
    value: object,
    *,
    source: str,
    destination: str,
    rewritten: set[str],
) -> object:
    if isinstance(value, dict):
        result: dict[object, object] = {}
        for key, item in value.items():
            rebased_key = _replace_json_path_prefix(
                key,
                source=source,
                destination=destination,
                rewritten=rewritten,
            )
            if rebased_key in result:
                raise ValueError(
                    f"fresh generation path rebasing produced a duplicate key: {rebased_key}"
                )
            result[rebased_key] = _replace_json_path_prefix(
                item,
                source=source,
                destination=destination,
                rewritten=rewritten,
            )
        return result
    if isinstance(value, list):
        return [
            _replace_json_path_prefix(
                item,
                source=source,
                destination=destination,
                rewritten=rewritten,
            )
            for item in value
        ]
    if isinstance(value, str) and (
        value == source or value.startswith(f"{source}/")
    ):
        rebased = destination + value[len(source) :]
        rewritten.add(rebased)
        return rebased
    return value


def _json_strings(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_strings(item)
    elif isinstance(value, str):
        yield value


def _is_absolute_path(value: str) -> bool:
    windows = PureWindowsPath(value)
    return Path(value).is_absolute() or windows.is_absolute() or bool(windows.drive)


def _rebase_payload_paths(
    payload: object,
    replacements: list[tuple[str, str]],
    *,
    label: str,
) -> tuple[object, set[str]]:
    result = payload
    rewritten: set[str] = set()
    for source, destination in sorted(
        replacements,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        result = _replace_json_path_prefix(
            result,
            source=source,
            destination=destination,
            rewritten=rewritten,
        )
    absolute = sorted(
        value for value in _json_strings(result) if _is_absolute_path(value)
    )
    if absolute:
        raise ValueError(f"{label} retains external absolute paths: {absolute[:3]}")
    return result, rewritten


def _bundle_path(output_root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be one bundle-relative path")
    relative = PurePosixPath(value)
    if (
        not value
        or value == "."
        or relative.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or value != relative.as_posix()
        or ".." in relative.parts
        or "\\" in value
    ):
        raise ValueError(f"{label} must be one normalized bundle-relative path")
    resolved = (output_root / Path(relative.as_posix())).resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the robotics reference bundle") from exc
    if not resolved.exists() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return resolved


def _validate_prepared_assets(trajectory: dict, output_root: Path) -> None:
    prepared = trajectory.get("prepared_assets")
    if prepared is None:
        return
    if not isinstance(prepared, dict) or not prepared:
        raise ValueError("trajectory prepared_assets must be a non-empty object")
    for asset_id, record in prepared.items():
        if not isinstance(asset_id, str) or not isinstance(record, dict):
            raise ValueError("trajectory prepared_assets entries must be named objects")
        if simulator_primitive_record(record) is not None:
            # A simulator built-in is validated by its declaration alone: it
            # owns no bundle artifact and must not name one.
            continue
        for kind in ("source", "physics"):
            path = _bundle_path(
                output_root,
                record.get(f"{kind}_sdf"),
                label=f"prepared asset {asset_id} {kind} SDF",
            )
            digest = record.get(f"{kind}_sha256")
            if digest != sha256_file(path):
                raise ValueError(
                    f"prepared asset {asset_id} {kind} SDF digest changed"
                )


def _archived_episode_root(
    certificate: FreshGenerationCertificate,
    copied_source: Path,
) -> Path:
    generation_root = Path(certificate.generation_root)
    candidates: dict[Path, int] = {}
    for value in _json_strings(
        certificate.model_dump(mode="json", exclude_none=True)
    ):
        source = Path(value)
        if not source.is_absolute() or source == generation_root:
            continue
        try:
            source.relative_to(generation_root)
        except ValueError:
            continue
        matched: Path | None = None
        for ancestor in source.parents:
            try:
                relative = source.relative_to(ancestor)
                ancestor.relative_to(generation_root)
            except ValueError:
                continue
            if relative.parts and (copied_source / relative).exists():
                matched = ancestor
        if matched is not None:
            candidates[matched] = candidates.get(matched, 0) + 1
    if not candidates:
        raise ValueError(
            "unable to map the fresh generation certificate to archived source_scene"
        )
    maximum = max(candidates.values())
    winners = sorted(path for path, count in candidates.items() if count == maximum)
    if len(winners) != 1:
        raise ValueError(
            "fresh generation certificate has an ambiguous archived episode root: "
            f"{winners}"
        )
    return winners[0]


def _rebase_fresh_generation_certificate(
    scene_root: Path,
    output_root: Path,
) -> None:
    certificate_path = scene_root / "fresh_generation_certificate.json"
    if not certificate_path.is_file():
        return
    certificate = FreshGenerationCertificate.model_validate_json(
        certificate_path.read_text(encoding="utf-8")
    )
    if certificate.status not in {"pass", "repaired_pass"}:
        raise ValueError("SceneCode fresh generation certificate is not passing")
    generation_root = Path(certificate.generation_root)
    if not generation_root.is_absolute():
        raise ValueError(
            "source SceneCode fresh generation certificate must bind its original root"
        )
    copied_source = scene_root / "source_scene"
    if not copied_source.is_dir():
        raise FileNotFoundError(
            "fresh SceneCode certificate requires its archived source_scene directory"
        )
    destination = copied_source.relative_to(output_root).as_posix()
    episode_root = _archived_episode_root(certificate, copied_source)
    certificate_payload = certificate.model_dump(mode="json", exclude_none=True)
    diagnostics = certificate_payload.get("diagnostics")
    if isinstance(diagnostics, dict) and "generation_summary" in diagnostics:
        diagnostics.pop("generation_summary")
        diagnostics["generation_summary_archival_status"] = (
            "not_present_in_frozen_scene_release"
        )
    payload, rewritten = _rebase_payload_paths(
        certificate_payload,
        [
            (str(episode_root), destination),
            (str(generation_root), destination),
        ],
        label="fresh SceneCode certificate",
    )
    assert isinstance(payload, dict)
    if payload.get("generation_root") != destination:
        raise ValueError(
            "fresh SceneCode certificate root does not map to the archived source_scene"
        )
    for relative in rewritten:
        resolved = (output_root / relative).resolve()
        try:
            resolved.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(
                "fresh SceneCode certificate path escapes the reference bundle"
            ) from exc
        if not resolved.exists() or resolved.is_symlink():
            raise FileNotFoundError(
                f"fresh SceneCode certificate points to a missing artifact: {relative}"
            )
    FreshGenerationCertificate.model_validate(payload)
    write_json(certificate_path, payload)


def _rewrite_quality_paths(
    quality_path: Path,
    output_root: Path,
    video_paths: dict[str, Path],
) -> dict:
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    reports = quality.get("views")
    if not isinstance(reports, dict) or set(reports) != set(video_paths):
        raise ValueError("reference quality certificate does not cover exact views")
    for view_id, video in video_paths.items():
        report = reports.get(view_id)
        if not isinstance(report, dict):
            raise ValueError(f"reference quality report is invalid: {view_id}")
        report["path"] = video.relative_to(output_root).as_posix()
        report["sha256"] = sha256_file(video)
    write_json(quality_path, quality)
    return quality


def _validate_visual_variant_manifest(
    manifest_path: Path,
    output_root: Path,
) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "scenecode_visual_variant.v1"
        or payload.get("physics_geometry_unchanged") is not True
    ):
        raise ValueError("visual variant manifest is not a physical-preserving v1 artifact")
    output_hashes = payload.get("output_file_sha256")
    source_hashes = payload.get("source_file_sha256")
    if not isinstance(output_hashes, dict) or not isinstance(source_hashes, dict):
        raise ValueError("visual variant manifest lacks complete file digest maps")
    variant_root = manifest_path.parent
    actual_output = {
        path.relative_to(variant_root).as_posix(): sha256_file(path)
        for path in sorted(variant_root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    if output_hashes != actual_output:
        raise ValueError("visual variant output file digest map changed")
    source_sdf = _bundle_path(
        output_root,
        payload.get("source_sdf"),
        label="visual variant source SDF",
    )
    if not source_sdf.is_file():
        raise FileNotFoundError(source_sdf)
    source_root = source_sdf.parent
    actual_source: dict[str, str] = {}
    for relative, digest in source_hashes.items():
        path = (source_root / relative).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(
                f"visual variant source artifact escapes its asset: {relative!r}"
            ) from exc
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise ValueError(
                f"visual variant source artifact digest changed: {relative!r}"
            )
        actual_source[relative] = digest
    source_files = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file() and path.name != "visual_variant_manifest.json"
    }
    if set(actual_source) != source_files:
        raise ValueError("visual variant source file digest map is incomplete")
    # Re-run the independent structural verifier at the package boundary.  The
    # compact manifest checks above bind bytes, but alone would accept a changed
    # collision resource or glTF graph under a visual-only claim.
    verify_visual_revision(
        manifest_path=manifest_path,
        output=variant_root / ".visual_revision_verdict.json",
        source_sdf_override=source_sdf,
        expected_source_sdf=source_sdf,
        write_output=False,
    )
    return payload


def _validate_physical_variant_manifest(
    manifest_path: Path,
    output_root: Path,
) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("physical variant manifest is not an object")
    scale_v1 = (
        payload.get("schema_version") == "scenecode_physical_scale_variant.v1"
        and payload.get("visual_collision_scale_coherent") is True
    )
    planar_v2 = (
        payload.get("schema_version") == "scenecode_physical_variant.v2"
        and payload.get("visual_geometry_unchanged") is True
        and payload.get("collision_geometry_matches_visual_bounds") is True
        and (payload.get("transformation") or {}).get("kind")
        == "add_planar_box_collision"
    )
    if not (scale_v1 or planar_v2):
        raise ValueError("physical variant manifest is not a coherent supported artifact")
    output_hashes = payload.get("output_file_sha256")
    source_hashes = payload.get("source_file_sha256")
    if not isinstance(output_hashes, dict) or not isinstance(source_hashes, dict):
        raise ValueError("physical variant manifest lacks complete file digest maps")
    variant_root = manifest_path.parent
    actual_output = {
        path.relative_to(variant_root).as_posix(): sha256_file(path)
        for path in sorted(variant_root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    if output_hashes != actual_output:
        raise ValueError("physical variant output file digest map changed")
    source_sdf = _bundle_path(
        output_root,
        payload.get("source_sdf"),
        label="physical variant source SDF",
    )
    if not source_sdf.is_file():
        raise FileNotFoundError(source_sdf)
    source_root = source_sdf.parent
    actual_source = {
        path.relative_to(source_root).as_posix(): sha256_file(path)
        for path in sorted(source_root.rglob("*"))
        if path.is_file() and path.name != "physical_variant_manifest.json"
    }
    if source_hashes != actual_source:
        raise ValueError("physical variant source file digest map changed")
    changed = payload.get("changed_files")
    if (
        not isinstance(changed, list)
        or not changed
        or any(not isinstance(value, str) or value not in output_hashes for value in changed)
        or any(output_hashes[value] == source_hashes.get(value) for value in changed)
    ):
        raise ValueError("physical variant changed_files does not bind changed outputs")
    return payload


def _trajectory_npz(trajectory: dict, destination: Path) -> None:
    frames = trajectory["frames"]
    names = frames[0]["robot"]["joint_names"]
    if any(frame["robot"]["joint_names"] != names for frame in frames):
        raise ValueError("robot joint ordering changes in the trajectory")
    subject_positions = []
    target_positions = []
    for frame in frames:
        if "cup" in frame and "tray" in frame:
            subject_positions.append(frame["cup"]["position"])
            target_positions.append(frame["tray"]["position"])
            continue
        articulated = frame.get("articulated") or {}
        component_id = str(frame.get("active_subject_id", articulated.get("component_id", "")))
        host_id = str(frame.get("active_target_id", articulated.get("host_object_id", "")))
        host_state = (frame.get("objects") or {}).get(host_id)
        child_state = ((host_state or {}).get("link_states") or {}).get(
            str(articulated.get("child_link", "")),
            {},
        )
        if not child_state or component_id != str(articulated.get("component_id", component_id)):
            raise ValueError("articulated trajectory lacks subject link state")
        subject_positions.append(child_state["position"])
        target_positions.append((host_state or {}).get("position"))
    if any(value is None for value in target_positions):
        raise ValueError("trajectory target state is incomplete")
    payload = {
        "time_s": np.asarray([frame["time_s"] for frame in frames], dtype=np.float64),
        "robot_joint_names": np.asarray(names),
        "robot_joint_positions": np.asarray(
            [frame["robot"]["joint_positions"] for frame in frames], dtype=np.float64
        ),
        "end_effector_position": np.asarray(
            [frame["robot"]["end_effector_position"] for frame in frames], dtype=np.float64
        ),
        "subject_position": np.asarray(subject_positions, dtype=np.float64),
        "target_position": np.asarray(target_positions, dtype=np.float64),
        "operator_step_index": np.asarray(
            [int(frame.get("operator_step_index", 0)) for frame in frames],
            dtype=np.int64,
        ),
        "active_subject_id": np.asarray(
            [str(frame.get("active_subject_id", "cup")) for frame in frames]
        ),
        "active_target_id": np.asarray(
            [str(frame.get("active_target_id", "tray")) for frame in frames]
        ),
    }
    object_states = [frame.get("objects") for frame in frames]
    if all(isinstance(value, dict) and value for value in object_states):
        object_ids = sorted(object_states[0])
        if any(set(value) != set(object_ids) for value in object_states):
            raise ValueError("trajectory object table changes across frames")
        payload["object_ids"] = np.asarray(object_ids)
        payload["object_positions"] = np.asarray(
            [
                [value[object_id]["position"] for object_id in object_ids]
                for value in object_states
            ],
            dtype=np.float64,
        )
        payload["object_orientations_xyzw"] = np.asarray(
            [
                [value[object_id]["orientation_xyzw"] for object_id in object_ids]
                for value in object_states
            ],
            dtype=np.float64,
        )
    articulated_states = [frame.get("articulated") for frame in frames]
    if all(isinstance(value, dict) for value in articulated_states):
        payload["articulated_joint_name"] = np.asarray(
            [str(value["joint_name"]) for value in articulated_states]
        )
        payload["articulated_joint_position"] = np.asarray(
            [float(value["position"]) for value in articulated_states],
            dtype=np.float64,
        )
        payload["articulated_joint_velocity"] = np.asarray(
            [float(value["velocity"]) for value in articulated_states],
            dtype=np.float64,
        )
    droid_gripper = [
        frame["robot"].get("droid_gripper_position") for frame in frames
    ]
    if all(value is not None for value in droid_gripper):
        payload["droid_gripper_position"] = np.asarray(
            droid_gripper, dtype=np.float64
        )
    np.savez_compressed(destination, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--simulation-result", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--scene-prompt-bundle", type=Path, required=True)
    parser.add_argument("--scenecode-release", required=True)
    parser.add_argument("--recipe-id")
    parser.add_argument("--harness-id")
    parser.add_argument("--task-graph", type=Path)
    parser.add_argument("--generation-manifest", type=Path)
    parser.add_argument("--identity-certificate", type=Path)
    parser.add_argument("--visual-variant-manifest", type=Path)
    parser.add_argument("--physical-variant-manifest", type=Path)
    parser.add_argument(
        "--visual-context-vision",
        choices=("required", "disabled"),
        default="required",
        help=(
            "rendered in-context appearance/proportion review; 'required' fails "
            "closed without the configured image-capable route"
        ),
    )
    parser.add_argument(
        "--visual-context-vision-model", default="gpt-5.6-sol"
    )
    parser.add_argument(
        "--grasp-band-evidence",
        type=Path,
        help="directory of measured grasp-band profiles/plans to archive with the reference",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    agentic_values = (
        args.harness_id,
        args.task_graph,
        args.generation_manifest,
        args.identity_certificate,
    )
    if args.recipe_id is not None and any(value is not None for value in agentic_values):
        raise ValueError("reference packaging accepts either a legacy recipe or agentic provenance")
    if args.recipe_id is None and not all(value is not None for value in agentic_values):
        raise ValueError(
            "recipe-free reference packaging requires harness, task graph, generation, and identity"
        )

    task = TaskInstructionSpec.model_validate_json(args.task.read_text(encoding="utf-8"))
    trajectory = json.loads(args.trajectory_json.read_text(encoding="utf-8"))
    result = json.loads(args.simulation_result.read_text(encoding="utf-8"))
    render = json.loads(args.render_manifest.read_text(encoding="utf-8"))
    if result.get("success") is not True:
        raise ValueError("PyBullet result is not successful")
    if render.get("schema_version") not in {
        "scenecode_robotics_multiview_render.v3",
        "scenecode_robotics_multiview_render.v4",
    }:
        raise ValueError("unsupported multiview render manifest")
    required = {"top", "wrist", "exterior_left", "exterior_right"}
    if set(render.get("views", {})) != required:
        raise ValueError("multiview render does not contain the exact camera roles")

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reference bundle: {output}")
    output.mkdir(parents=True)
    scene_prompt_name = "scene_prompt_bundle"
    _copy_bundle_directory(
        args.scene_prompt_bundle,
        output / scene_prompt_name,
        label="scene prompt bundle",
    )
    if args.grasp_band_evidence is not None:
        if not args.grasp_band_evidence.is_dir():
            raise FileNotFoundError(args.grasp_band_evidence)
        _copy_bundle_directory(
            args.grasp_band_evidence,
            output / "grasp_band_evidence",
            label="grasp-band evidence",
        )
    _rebase_fresh_generation_certificate(
        output / scene_prompt_name,
        output,
    )
    agentic_provenance: dict[str, str] = {}
    if args.harness_id is not None:
        assert args.task_graph is not None
        assert args.generation_manifest is not None
        assert args.identity_certificate is not None
        agentic_provenance = _copy_agentic_provenance(
            task=task,
            task_path=args.task,
            task_graph=args.task_graph,
            generation_manifest=args.generation_manifest,
            identity_certificate=args.identity_certificate,
            output=output,
        )
    replacements: list[tuple[str, str]] = []
    source_scene = args.scene_prompt_bundle.resolve() / "source_scene"
    if source_scene.is_dir():
        replacements.append(
            (
                str(source_scene),
                f"{scene_prompt_name}/source_scene",
            )
        )
    visual_variant_name = None
    visual_variant_sha256 = None
    if args.visual_variant_manifest is not None:
        if not args.visual_variant_manifest.is_file():
            raise FileNotFoundError(args.visual_variant_manifest)
        visual_source_payload = json.loads(
            args.visual_variant_manifest.read_text(encoding="utf-8")
        )
        source_sdf_value = visual_source_payload.get("source_sdf")
        if not isinstance(source_sdf_value, str):
            raise ValueError("visual variant manifest lacks source_sdf")
        source_sdf = Path(source_sdf_value).resolve()
        if not source_sdf.is_file():
            raise FileNotFoundError(source_sdf)
        source_asset_root = source_sdf.parent
        _copy_bundle_directory(
            source_asset_root,
            output / "visual_variant_source_asset",
            label="visual variant source asset",
        )
        replacements.append((str(source_asset_root), "visual_variant_source_asset"))
        source_variant_root = args.visual_variant_manifest.resolve().parent
        variant_root = output / "visual_variant"
        _copy_bundle_directory(
            source_variant_root,
            variant_root,
            label="visual variant asset",
        )
        replacements.append((str(source_variant_root), "visual_variant"))
        visual_variant_name = (
            variant_root / args.visual_variant_manifest.name
        ).relative_to(output).as_posix()
        visual_manifest_path = output / visual_variant_name
        visual_payload = json.loads(
            visual_manifest_path.read_text(encoding="utf-8")
        )
        visual_payload, _ = _rebase_payload_paths(
            visual_payload,
            replacements,
            label="visual variant manifest",
        )
        write_json(visual_manifest_path, visual_payload)
        _validate_visual_variant_manifest(visual_manifest_path, output)
        visual_variant_sha256 = sha256_file(visual_manifest_path)

    physical_variant_name = None
    physical_variant_sha256 = None
    if args.physical_variant_manifest is not None:
        if not args.physical_variant_manifest.is_file():
            raise FileNotFoundError(args.physical_variant_manifest)
        physical_payload = json.loads(
            args.physical_variant_manifest.read_text(encoding="utf-8")
        )
        source_sdf_value = physical_payload.get("source_sdf")
        if not isinstance(source_sdf_value, str):
            raise ValueError("physical variant manifest lacks source_sdf")
        source_sdf = Path(source_sdf_value).resolve()
        if not source_sdf.is_file():
            raise FileNotFoundError(source_sdf)
        source_variant_root = args.physical_variant_manifest.resolve().parent
        physical_root = output / "physical_variant"
        output_variant_root = physical_root / "output_asset"
        _copy_bundle_directory(
            source_variant_root,
            output_variant_root,
            label="physical variant output asset",
        )
        source_asset_root = source_sdf.parent
        _copy_bundle_directory(
            source_asset_root,
            physical_root / "source_asset",
            label="physical variant source asset",
        )
        replacements.append(
            (str(source_asset_root), "physical_variant/source_asset")
        )
        replacements.append(
            (str(source_variant_root), "physical_variant/output_asset")
        )
        physical_variant_name = (
            output_variant_root / args.physical_variant_manifest.name
        ).relative_to(output).as_posix()
        physical_manifest_path = output / physical_variant_name
        physical_payload, _ = _rebase_payload_paths(
            physical_payload,
            [
                (str(source_variant_root), "physical_variant/output_asset"),
                (str(source_asset_root), "physical_variant/source_asset"),
                *replacements,
            ],
            label="physical variant manifest",
        )
        write_json(physical_manifest_path, physical_payload)
        _validate_physical_variant_manifest(physical_manifest_path, output)
        physical_variant_sha256 = sha256_file(physical_manifest_path)

    camera_rig_name = None
    camera_rig_sha256 = None
    if render.get("camera_rig") is not None:
        source_rig = Path(render["camera_rig"]).resolve()
        if (not source_rig.is_file()
                or sha256_file(source_rig) != render.get("camera_rig_sha256")):
            raise ValueError("rendered camera rig changed before reference packaging")
        camera_rig_name = "camera_rig.json"
        shutil.copy2(source_rig, output / camera_rig_name)
        camera_rig_sha256 = sha256_file(output / camera_rig_name)

    prepared = trajectory.get("prepared_assets")
    if prepared is not None:
        if not isinstance(prepared, dict) or not prepared:
            raise ValueError("trajectory prepared_assets must be a non-empty object")
        source_root_destinations: dict[Path, str] = {}
        source_names: set[str] = set()
        prepared = {
            asset_id: record
            for asset_id, record in prepared.items()
            if record is not None
        }
        trajectory["prepared_assets"] = prepared
        for asset_id, record in sorted(prepared.items()):
            if not isinstance(asset_id, str):
                raise ValueError("every prepared asset must be one named object")
            primitive = simulator_primitive_record(record)
            if primitive is not None:
                # The physics engine loads this built-in itself, so the bundle
                # archives nothing for it.  Normalize the declaration so every
                # published trajectory carries the same shape.
                prepared[asset_id] = primitive
                continue
            if not isinstance(record, dict):
                raise ValueError("every prepared asset must be one named object")
            source_value = record.get("source_sdf")
            if not isinstance(source_value, str):
                raise ValueError("every prepared asset must declare one source SDF")
            source_sdf = Path(source_value).resolve()
            if not source_sdf.is_file() or source_sdf.is_symlink():
                raise FileNotFoundError(source_sdf)
            # A physical/visual variant may already have archived this source
            # root.  Otherwise copy the complete SDF asset directory so the v6
            # trajectory never retains an executable path back into a
            # SceneCode generation or transient staging tree.
            already_archived = any(
                str(source_sdf) == source
                or str(source_sdf).startswith(f"{source}/")
                for source, _ in replacements
            )
            if already_archived:
                continue
            source_root = source_sdf.parent
            destination_name = "".join(
                character if character.isalnum() or character in "_-" else "_"
                for character in asset_id
            ).strip("_")
            if not destination_name or destination_name in source_names:
                raise ValueError(f"prepared asset id is not bundle-path safe: {asset_id!r}")
            source_names.add(destination_name)
            if source_root in source_root_destinations:
                replacements.append(
                    (str(source_root), source_root_destinations[source_root])
                )
                continue
            archived_source = (
                output / "simulation_assets" / "source_sdf" / destination_name
            )
            _copy_bundle_directory(
                source_root,
                archived_source,
                label=f"prepared source SDF asset {asset_id}",
            )
            relative_source = archived_source.relative_to(output).as_posix()
            source_root_destinations[source_root] = relative_source
            replacements.append((str(source_root), relative_source))
        archived_assets = [
            record
            for record in prepared.values()
            if simulator_primitive_record(record) is None
        ]
        physics_paths = {
            Path(record["physics_sdf"]).resolve()
            for record in archived_assets
            if isinstance(record.get("physics_sdf"), str)
        }
        if len(physics_paths) != len(archived_assets):
            raise ValueError("every prepared asset must declare one physics SDF")
        physics_roots = {path.parent for path in physics_paths}
        if len(physics_roots) != 1:
            raise ValueError("prepared physics SDFs must share one immutable directory")
        physics_root = next(iter(physics_roots))
        archived_physics = output / "simulation_assets" / "physics_sdf"
        _copy_bundle_directory(
            physics_root,
            archived_physics,
            label="prepared physics SDF directory",
        )
        replacements.append(
            (
                str(physics_root),
                archived_physics.relative_to(output).as_posix(),
            )
        )

    trajectory, _ = _rebase_payload_paths(
        trajectory,
        replacements,
        label="reference JSON trajectory",
    )
    assert isinstance(trajectory, dict)
    _validate_prepared_assets(trajectory, output)
    trajectory_out = output / "trajectory.npz"
    _trajectory_npz(trajectory, trajectory_out)
    trajectory_json_out = output / "trajectory.json"
    write_json(trajectory_json_out, trajectory)
    shutil.copy2(args.task, output / "task_instruction.json")
    visual_alignment_path = None
    if render.get("articulation_binding"):
        articulation = json.loads(
            Path(render["articulation_binding"]).read_text(encoding="utf-8"))
        host_id = articulation["host_object_id"]
        host_asset = trajectory["prepared_assets"]["host"]
        physics_sdf = _bundle_path(
            output, host_asset["physics_sdf"], label="articulated physics SDF")
        if visual_variant_name is not None:
            visual_sdf = output / "visual_variant" / Path(
                articulation["host_sdf"]).name
        else:
            visual_sdf = _bundle_path(
                output, host_asset["source_sdf"], label="articulated visual SDF")
        if sha256_file(visual_sdf) != articulation["host_sdf_sha256"]:
            raise ValueError("articulated visual SDF differs from rendered binding")
        visual_alignment_path = certify_reference_runtime_alignment(
            visual_sdf=visual_sdf,
            physics_sdf=physics_sdf,
            trajectory=trajectory,
            host_object_id=host_id,
            base_link=articulation["parent_link"],
            output=output / "visual_physics_alignment_certificate.json",
        )
    if result.get("operator") == "pour":
        verdict_value = result.get("independent_verdict_path")
        if not isinstance(verdict_value, str):
            raise ValueError("pour result lacks an independent verdict file")
        verdict_source = Path(verdict_value).resolve()
        if (not verdict_source.is_file()
                or sha256_file(verdict_source) != result.get("independent_verdict_sha256")
                or json.loads(verdict_source.read_text()) != result.get("independent_verdict")):
            raise ValueError("pour independent verdict changed before packaging")
        verdict_destination = output / "independent_pour_verdict.json"
        shutil.copy2(verdict_source, verdict_destination)
        replacements.append((str(verdict_source), verdict_destination.name))
    result_replacements = [
        *replacements,
        (str(args.trajectory_json.resolve()), trajectory_json_out.name),
    ]
    result, _ = _rebase_payload_paths(
        result,
        result_replacements,
        label="simulation result",
    )
    assert isinstance(result, dict)
    simulation_result_out = output / "simulation_result.json"
    write_json(simulation_result_out, result)
    simulation_certificate = output / "simulation_certificate.json"
    write_json(
        simulation_certificate,
        {
            "schema_version": "robotics_simulation_certificate.v2",
            "passed": True,
            "authority": "PyBullet",
            "source_result": simulation_result_out.relative_to(output).as_posix(),
            "measurements": result,
            "trajectory_json_sha256": sha256_file(trajectory_json_out),
        },
    )

    fps = float(trajectory.get("sample_hz") or task.fps)
    view_artifacts: list[CameraViewArtifact] = []
    video_paths: dict[str, Path] = {}
    view_sizes: dict[str, tuple[int, int]] = {}
    for view_id in sorted(required):
        record = render["views"][view_id]
        images = [Path(item["image"]).resolve() for item in record["frames"]]
        if [item["frame"] for item in record["frames"]] != [item["frame"] for item in trajectory["frames"]]:
            raise ValueError(f"{view_id} source frame order differs from the measured trajectory")
        if len(images) != len(trajectory["frames"]):
            raise ValueError(f"{view_id} frame count differs from the trajectory")
        frames = []
        for image in images:
            frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"unable to decode rendered frame: {image}")
            frames.append(frame)
        view_root = output / "views" / view_id
        view_root.mkdir(parents=True)
        source_base = Path(record["cae_base_scene"]).resolve()
        for name in ("camera_calibration_info.json", "initial_depth.pt", "scene_data.yaml"):
            if not (source_base / name).is_file():
                raise FileNotFoundError(f"{view_id} CAE base scene is missing {name}")
        base = view_root / "cae_base_scene"
        _copy_bundle_directory(
            source_base,
            base,
            label=f"{view_id} CAE base scene",
        )
        conditioning = view_root / "conditioning_t0.png"
        shutil.copy2(images[0], conditioning)
        video = view_root / "reference.mp4"
        write_h264_rgb(
            video,
            (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames),
            fps,
            crf=14,
            lossless=True,
        )
        height, width = frames[0].shape[:2]
        declared_size = record.get("size_wh")
        if declared_size is not None and declared_size != [width, height]:
            raise ValueError(
                f"{view_id} rendered size differs from its SceneCode manifest"
            )
        if render.get("schema_version") == "scenecode_robotics_multiview_render.v4":
            expected_size = (render.get("render_size_wh_by_view") or {}).get(view_id)
            if expected_size != [width, height]:
                raise ValueError(
                    f"{view_id} does not match its model-native reference size"
                )
        video_paths[view_id] = video
        view_sizes[view_id] = (width, height)
        view_artifacts.append(
            CameraViewArtifact(
                view_id=view_id,
                camera_type=record["camera_type"],
                camera_signature=sha256_file(conditioning)
                + ":"
                + str(record["camera_signature"]),
                video=str(video.relative_to(output)),
                conditioning_image=str(conditioning.relative_to(output)),
                cae_base_scene=str(base.relative_to(output)),
                width=width,
                height=height,
                fps=fps,
                frame_count=len(frames),
                sha256=sha256_file(video),
            )
        )

    if args.harness_id is not None and render.get("schema_version") == (
        "scenecode_robotics_multiview_render.v4"
    ):
        _certify_model_native_reference(task=task, render=render, output=output)

    quality_path = certify_reference_views(
        video_paths,
        output / "quality_certificate.json",
        expected_sizes=view_sizes,
        source_frames={name:[Path(item["image"]) for item in record["frames"]]
                       for name,record in render["views"].items()},
        expected_fps=fps,
    )
    quality = _rewrite_quality_paths(quality_path, output, video_paths)
    if quality.get("passed") is not True:
        failures = {name: {key: value for key, value in view.get("checks", {}).items() if not value}
                    for name, view in quality.get("views", {}).items() if not view.get("passed")}
        raise ValueError(f"reference quality gate failed: {quality_path}; checks={failures}; "
                         f"episode={quality.get('episode_checks')}")
    contact_render = _operator_contact_render(
        trajectory,
        render,
        operator=str(result.get("operator", "")),
        trajectory_path=args.trajectory_json,
        render_path=args.render_manifest,
        output=output,
    )
    physical_render_path = certify_reference_physical_render(
        trajectory,
        result,
        contact_render,
        output / "physical_render_certificate.json",
    )
    physical_render = json.loads(physical_render_path.read_text(encoding="utf-8"))
    if physical_render.get("passed") is not True:
        raise ValueError(
            f"reference visual grasp registration failed: {physical_render_path}"
        )
    physical_variant = None
    if args.physical_variant_manifest is not None:
        physical_variant = json.loads(
            args.physical_variant_manifest.read_text(encoding="utf-8")
        )
    grasp_band_plan = None
    grasp_band_dir = output / "grasp_band_evidence"
    if grasp_band_dir.is_dir():
        plans = []
        for candidate in sorted(grasp_band_dir.glob("grasp_band_*.json")):
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            plan = payload.get("plan") if isinstance(payload, dict) else None
            if isinstance(plan, dict):
                plans.append(plan)
        if plans:
            grasp_band_plan = plans[0]
    visual_context_path = certify_reference_visual_context(
        task=task,
        trajectory=trajectory,
        render=render,
        output=output,
        physical_variant_manifest=physical_variant,
        grasp_band_plan=grasp_band_plan,
        material_authority=_material_authority(args.identity_certificate),
        same_asset_aliases=_certified_same_asset_aliases(
            args.generation_manifest, task
        ),
        vision=args.visual_context_vision,
        vision_model=args.visual_context_vision_model,
    )
    manifest = RoboticsReferenceManifest(
        schema_version=(
            "robotics_reference_bundle.v6"
            if args.harness_id is not None
            else "robotics_reference_bundle.v5"
        ),
        case_id=task.case_id,
        recipe_id=args.recipe_id,
        harness_id=args.harness_id,
        **agentic_provenance,
        task_instruction=task.instruction,
        scene_prompt_bundle=scene_prompt_name,
        source_scenecode_release=args.scenecode_release,
        robot_embodiment=task.robot,
        visual_variant_manifest=visual_variant_name,
        visual_variant_manifest_sha256=visual_variant_sha256,
        visual_physics_alignment_certificate=(
            visual_alignment_path.name if visual_alignment_path is not None else None),
        visual_physics_alignment_certificate_sha256=(
            sha256_file(visual_alignment_path)
            if visual_alignment_path is not None else None),
        physical_variant_manifest=physical_variant_name,
        physical_variant_manifest_sha256=physical_variant_sha256,
        camera_rig=camera_rig_name,
        camera_rig_sha256=camera_rig_sha256,
        trajectory=trajectory_out.name,
        trajectory_sha256=sha256_file(trajectory_out),
        trajectory_json=trajectory_json_out.name,
        trajectory_json_sha256=sha256_file(trajectory_json_out),
        simulation_certificate=simulation_certificate.name,
        simulation_passed=True,
        physical_render_certificate=physical_render_path.name,
        physical_render_passed=True,
        views=view_artifacts,
        quality_certificate=quality_path.name,
        quality_passed=True,
        visual_context_certificate=visual_context_path.name,
        visual_context_certificate_sha256=sha256_file(visual_context_path),
        visual_context_passed=True,
        bundle_artifacts_sha256=build_reference_artifact_digests(output),
    )
    path = manifest.write_json(output / "robotics_reference_manifest.json")
    load_reference_manifest(path)
    print(path)


if __name__ == "__main__":
    main()
