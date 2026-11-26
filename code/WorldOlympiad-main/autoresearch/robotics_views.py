"""Deterministic camera-view preparation for LingBot-VA and Cosmos3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import cv2
import numpy as np
from PIL import Image

from .models import FreshGenerationCertificate
from .robotics_contracts import (
    RoboticsReferenceManifest,
    TaskInstructionSpec,
    simulator_primitive_record,
)


REFERENCE_MANIFEST_NAME = "robotics_reference_manifest.json"
REQUIRED_CAE_BASE_FILES = (
    "camera_calibration_info.json",
    "initial_depth.pt",
    "scene_data.yaml",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_bundle_files(root: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"robotics reference bundles may not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(
                f"robotics reference bundles may contain only files and directories: {path}"
            )
        relative = path.relative_to(root).as_posix()
        if relative == REFERENCE_MANIFEST_NAME:
            continue
        records[relative] = sha256_file(path)
    if not records:
        raise ValueError("robotics reference bundle has no artifacts")
    return records


def build_reference_artifact_digests(root: str | Path) -> dict[str, str]:
    """Hash every reference file except the manifest that embeds the digest map."""

    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return _reference_bundle_files(resolved)


def _normalized_reference_path(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be one normalized bundle-relative path")
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
    return relative


def _reference_file(
    root: Path,
    value: str,
    artifact_map: dict[str, str],
    *,
    label: str,
) -> Path:
    relative = _normalized_reference_path(value, label=label)
    resolved = (root / Path(relative.as_posix())).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its robotics reference bundle") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    if artifact_map.get(relative.as_posix()) != sha256_file(resolved):
        raise ValueError(f"{label} is absent from the immutable artifact map")
    return resolved


def _reference_directory(
    root: Path,
    value: str,
    artifact_map: dict[str, str],
    *,
    label: str,
) -> Path:
    relative = _normalized_reference_path(value, label=label)
    resolved = (root / Path(relative.as_posix())).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its robotics reference bundle") from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    prefix = f"{relative.as_posix()}/"
    if not any(path.startswith(prefix) for path in artifact_map):
        raise ValueError(f"{label} has no files in the immutable artifact map")
    return resolved


def _owned_reference_file(
    owner: Path,
    owner_root: Path,
    bundle_root: Path,
    value: object,
    artifact_map: dict[str, str],
    *,
    label: str,
) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be one normalized bundle-relative path")
    relative = _normalized_reference_path(value, label=label)
    resolved = (owner.parent / Path(relative.as_posix())).resolve()
    try:
        resolved.relative_to(owner_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its owning reference artifact") from exc
    bundle_relative = resolved.relative_to(bundle_root).as_posix()
    return _reference_file(
        bundle_root,
        bundle_relative,
        artifact_map,
        label=label,
    )


def _json_object(path: Path, *, label: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


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


def _reference_entry(
    root: Path,
    value: str,
    artifact_map: dict[str, str],
    *,
    label: str,
) -> Path:
    relative = _normalized_reference_path(value, label=label)
    normalized = relative.as_posix()
    if normalized in artifact_map:
        return _reference_file(root, normalized, artifact_map, label=label)
    return _reference_directory(root, normalized, artifact_map, label=label)


def _tree_sha256(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"reference artifact directory is empty: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _digest_records(value: object):
    if isinstance(value, dict):
        if "path" in value or "sha256" in value:
            if not isinstance(value.get("path"), str) or not isinstance(
                value.get("sha256"), str
            ):
                raise ValueError("fresh SceneCode artifact record is incomplete")
            yield value["path"], value["sha256"]
        for item in value.values():
            yield from _digest_records(item)
    elif isinstance(value, list):
        for item in value:
            yield from _digest_records(item)


def _validate_fresh_generation(
    scene_root: Path,
    root: Path,
    artifact_map: dict[str, str],
) -> None:
    relative = (
        scene_root / "fresh_generation_certificate.json"
    ).relative_to(root).as_posix()
    if relative not in artifact_map:
        return
    path = _reference_file(
        root,
        relative,
        artifact_map,
        label="fresh SceneCode generation certificate",
    )
    certificate = FreshGenerationCertificate.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if certificate.status not in {"pass", "repaired_pass"}:
        raise ValueError("fresh SceneCode generation certificate is not passing")
    payload = certificate.model_dump(mode="json", exclude_none=True)
    absolute = sorted(
        value for value in _json_strings(payload) if _is_absolute_path(value)
    )
    if absolute:
        raise ValueError(
            "fresh SceneCode generation certificate retains external paths: "
            f"{absolute[:3]}"
        )
    generation_root = _reference_directory(
        root,
        certificate.generation_root,
        artifact_map,
        label="fresh SceneCode generation root",
    )
    try:
        generation_root.relative_to(scene_root)
    except ValueError as exc:
        raise ValueError(
            "fresh SceneCode generation root escapes the archived scene"
        ) from exc
    prefix = f"{certificate.generation_root}/"
    for value in _json_strings(payload):
        if value == certificate.generation_root or value.startswith(prefix):
            _reference_entry(
                root,
                value,
                artifact_map,
                label="fresh SceneCode referenced artifact",
            )
    house_probes = [
        probe
        for probe in certificate.probes
        if probe.name == "house_state_inside_current_generation_root"
    ]
    if len(house_probes) != 1 or not isinstance(house_probes[0].measured, str):
        raise ValueError("fresh SceneCode certificate lacks its house-state binding")
    house_state = _reference_file(
        root,
        house_probes[0].measured,
        artifact_map,
        label="fresh SceneCode house state",
    )
    if sha256_file(house_state) != certificate.house_state_sha256:
        raise ValueError("fresh SceneCode house-state digest changed")

    for relative, claimed_digest in _digest_records(certificate.assets):
        artifact = _reference_entry(
            root,
            relative,
            artifact_map,
            label="fresh SceneCode certified asset",
        )
        try:
            artifact.relative_to(generation_root)
        except ValueError as exc:
            raise ValueError(
                "fresh SceneCode certified asset escapes its generation root"
            ) from exc
        if _tree_sha256(artifact) != claimed_digest:
            raise ValueError(
                f"fresh SceneCode certified asset digest changed: {relative}"
            )


def _validate_visual_variant(
    manifest_path: Path,
    root: Path,
    artifact_map: dict[str, str],
) -> None:
    payload = _json_object(manifest_path, label="visual variant manifest")
    if (
        payload.get("schema_version") != "scenecode_visual_variant.v1"
        or payload.get("physics_geometry_unchanged") is not True
    ):
        raise ValueError("visual variant is not a physical-preserving v1 artifact")
    absolute = sorted(
        value for value in _json_strings(payload) if _is_absolute_path(value)
    )
    if absolute:
        raise ValueError(
            f"visual variant manifest retains external paths: {absolute[:3]}"
        )
    source_sdf = _reference_file(
        root,
        payload.get("source_sdf"),
        artifact_map,
        label="visual variant source SDF",
    )
    source_hashes = payload.get("source_file_sha256")
    output_hashes = payload.get("output_file_sha256")
    if not isinstance(source_hashes, dict) or not isinstance(output_hashes, dict):
        raise ValueError("visual variant manifest lacks complete file digest maps")
    source_root = source_sdf.parent
    for relative, digest in source_hashes.items():
        source = _owned_reference_file(
            source_sdf,
            source_root,
            root,
            relative,
            artifact_map,
            label=f"visual variant source artifact {relative}",
        )
        if digest != sha256_file(source):
            raise ValueError(
                f"visual variant source artifact digest changed: {relative!r}"
            )
    actual_source = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file() and path.name != "visual_variant_manifest.json"
    }
    if set(source_hashes) != actual_source:
        raise ValueError("visual variant source file digest map is incomplete")

    variant_root = manifest_path.parent
    for relative, digest in output_hashes.items():
        output = _owned_reference_file(
            manifest_path,
            variant_root,
            root,
            relative,
            artifact_map,
            label=f"visual variant output artifact {relative}",
        )
        if digest != sha256_file(output):
            raise ValueError(
                f"visual variant output artifact digest changed: {relative!r}"
            )
    actual_output = {
        path.relative_to(variant_root).as_posix()
        for path in variant_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(output_hashes) != actual_output:
        raise ValueError("visual variant output file digest map is incomplete")


def _validate_physical_variant(
    manifest_path: Path,
    root: Path,
    artifact_map: dict[str, str],
) -> None:
    payload = _json_object(manifest_path, label="physical variant manifest")
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
        raise ValueError("physical variant is not a coherent supported artifact")
    absolute = sorted(
        value for value in _json_strings(payload) if _is_absolute_path(value)
    )
    if absolute:
        raise ValueError(
            f"physical variant manifest retains external paths: {absolute[:3]}"
        )
    source_sdf = _reference_file(
        root,
        payload.get("source_sdf"),
        artifact_map,
        label="physical variant source SDF",
    )
    source_hashes = payload.get("source_file_sha256")
    output_hashes = payload.get("output_file_sha256")
    if not isinstance(source_hashes, dict) or not isinstance(output_hashes, dict):
        raise ValueError("physical variant manifest lacks complete file digest maps")
    source_root = source_sdf.parent
    for relative, digest in source_hashes.items():
        source = _owned_reference_file(
            source_sdf,
            source_root,
            root,
            relative,
            artifact_map,
            label=f"physical variant source artifact {relative}",
        )
        if digest != sha256_file(source):
            raise ValueError(
                f"physical variant source artifact digest changed: {relative!r}"
            )
    actual_source = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file() and path.name != "physical_variant_manifest.json"
    }
    if set(source_hashes) != actual_source:
        raise ValueError("physical variant source file digest map is incomplete")

    variant_root = manifest_path.parent
    for relative, digest in output_hashes.items():
        output = _owned_reference_file(
            manifest_path,
            variant_root,
            root,
            relative,
            artifact_map,
            label=f"physical variant output artifact {relative}",
        )
        if digest != sha256_file(output):
            raise ValueError(
                f"physical variant output artifact digest changed: {relative!r}"
            )
    actual_output = {
        path.relative_to(variant_root).as_posix()
        for path in variant_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(output_hashes) != actual_output:
        raise ValueError("physical variant output file digest map is incomplete")


def _validate_v5_reference_bundle(
    manifest_path: Path,
    manifest: RoboticsReferenceManifest,
) -> dict[str, str]:
    if manifest_path.name != REFERENCE_MANIFEST_NAME:
        raise ValueError(
            f"robotics reference v5 manifest must be named {REFERENCE_MANIFEST_NAME}"
        )
    root = manifest_path.parent
    expected = manifest.bundle_artifacts_sha256
    assert expected is not None
    actual = _reference_bundle_files(root)
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            path
            for path in set(expected) & set(actual)
            if expected[path] != actual[path]
        )
        raise ValueError(
            "robotics reference bundle artifact map changed: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )

    task_path = _reference_file(
        root,
        "task_instruction.json",
        expected,
        label="typed task instruction",
    )
    task = TaskInstructionSpec.model_validate_json(
        task_path.read_text(encoding="utf-8")
    )
    if (
        task.case_id != manifest.case_id
        or task.instruction != manifest.task_instruction
        or task.robot != manifest.robot_embodiment
        or task.simulator != manifest.simulator
    ):
        raise ValueError("typed task identity differs from its reference manifest")

    trajectory_json_path = _reference_file(
        root,
        manifest.trajectory_json,
        expected,
        label="reference JSON trajectory",
    )
    trajectory = _json_object(
        trajectory_json_path,
        label="reference JSON trajectory",
    )
    trajectory_absolute = sorted(
        value for value in _json_strings(trajectory) if _is_absolute_path(value)
    )
    if trajectory_absolute:
        raise ValueError(
            "reference JSON trajectory retains external paths: "
            f"{trajectory_absolute[:3]}"
        )
    prepared = trajectory.get("prepared_assets")
    if prepared is not None:
        if not isinstance(prepared, dict) or not prepared:
            raise ValueError("trajectory prepared_assets must be a non-empty object")
        for asset_id, record in prepared.items():
            if not isinstance(asset_id, str) or not isinstance(record, dict):
                raise ValueError(
                    "trajectory prepared_assets entries must be named objects"
                )
            if simulator_primitive_record(record) is not None:
                # Simulator built-ins are loaded by the engine itself; the
                # bundle archives nothing for them and they name no SDF.
                continue
            for kind in ("source", "physics"):
                artifact = _reference_file(
                    root,
                    record.get(f"{kind}_sdf"),
                    expected,
                    label=f"prepared asset {asset_id} {kind} SDF",
                )
                if record.get(f"{kind}_sha256") != sha256_file(artifact):
                    raise ValueError(
                        f"prepared asset {asset_id} {kind} SDF digest changed"
                    )
    trajectory_frames = trajectory.get("frames")
    sample_hz = trajectory.get("sample_hz") or task.fps
    view_frame_count = manifest.views[0].frame_count
    view_fps = manifest.views[0].fps
    if (
        not isinstance(trajectory_frames, list)
        or len(trajectory_frames) != view_frame_count
        or not isinstance(sample_hz, (int, float))
        or abs(float(sample_hz) - view_fps) > 1e-6
    ):
        raise ValueError("reference trajectory timeline differs from its physical views")

    scene_root = _reference_directory(
        root,
        manifest.scene_prompt_bundle,
        expected,
        label="scene prompt bundle",
    )
    # Historical recipes use this field for a frozen SceneCode generation
    # directory, while smaller prompt-only recipes use scene_prompt_bundle.v1.
    # The exact artifact map closes either directory layout.  When the typed
    # prompt manifest is present, additionally validate its own relative links.
    scene_manifest_relative = (
        scene_root / "scene_prompt_bundle.json"
    ).relative_to(root).as_posix()
    if scene_manifest_relative in expected:
        scene_manifest = _reference_file(
            root,
            scene_manifest_relative,
            expected,
            label="scene prompt bundle manifest",
        )
        scene = _json_object(scene_manifest, label="scene prompt bundle manifest")
        if (
            scene.get("schema_version") != "scene_prompt_bundle.v1"
            or scene.get("passed") is not True
        ):
            raise ValueError("scene prompt bundle is not a passing v1 artifact")
        scene_files: dict[str, Path] = {}
        for field in ("prompt_spec", "prompt_quality", "prompt_text"):
            scene_files[field] = _owned_reference_file(
                scene_manifest,
                scene_root,
                root,
                scene.get(field),
                expected,
                label=f"scene prompt {field}",
            )
        prompt_quality = _json_object(
            scene_files["prompt_quality"],
            label="scene prompt quality",
        )
        if prompt_quality.get("passed") is not True:
            raise ValueError("scene prompt quality artifact is not passing")
    _validate_fresh_generation(scene_root, root, expected)

    simulation_path = _reference_file(
        root,
        manifest.simulation_certificate,
        expected,
        label="simulation certificate",
    )
    simulation = _json_object(simulation_path, label="simulation certificate")
    if (
        simulation.get("schema_version") != "robotics_simulation_certificate.v2"
        or simulation.get("passed") is not True
        or simulation.get("trajectory_json_sha256")
        != manifest.trajectory_json_sha256
    ):
        raise ValueError("simulation certificate does not bind this successful reference")
    source_result = _owned_reference_file(
        simulation_path,
        root,
        root,
        simulation.get("source_result"),
        expected,
        label="simulation source result",
    )
    result = _json_object(source_result, label="simulation source result")
    result_absolute = sorted(
        value for value in _json_strings(result) if _is_absolute_path(value)
    )
    if result_absolute:
        raise ValueError(
            f"simulation source result retains external paths: {result_absolute[:3]}"
        )
    result_trajectory = result.get("trajectory_path")
    if result_trajectory is not None:
        bound_trajectory = _reference_file(
            root,
            result_trajectory,
            expected,
            label="simulation result trajectory",
        )
        if bound_trajectory != trajectory_json_path:
            raise ValueError("simulation result points to another trajectory")
    if (
        result.get("success") is not True
        or simulation.get("measurements") != result
    ):
        raise ValueError(
            "simulation source result is not successful or differs from its certificate"
        )

    physical_path = _reference_file(
        root,
        manifest.physical_render_certificate,
        expected,
        label="physical render certificate",
    )
    physical = _json_object(physical_path, label="physical render certificate")
    if (
        physical.get("schema_version")
        not in {
            "robotics_reference_physical_render.v3",
            "robotics_reference_physical_render.v4",
            "robotics_reference_physical_render.v5",
            "robotics_reference_physical_render.v6",
        }
        or physical.get("passed") is not True
    ):
        raise ValueError("physical render certificate is not a passing artifact")

    if manifest.visual_variant_manifest is not None:
        visual = _reference_file(
            root,
            manifest.visual_variant_manifest,
            expected,
            label="visual variant manifest",
        )
        if sha256_file(visual) != manifest.visual_variant_manifest_sha256:
            raise ValueError("visual variant manifest digest changed")
        _validate_visual_variant(visual, root, expected)

    if manifest.physical_variant_manifest is not None:
        physical_variant = _reference_file(
            root,
            manifest.physical_variant_manifest,
            expected,
            label="physical variant manifest",
        )
        if sha256_file(physical_variant) != manifest.physical_variant_manifest_sha256:
            raise ValueError("physical variant manifest digest changed")
        _validate_physical_variant(physical_variant, root, expected)

    quality_path = _reference_file(
        root,
        manifest.quality_certificate,
        expected,
        label="reference quality certificate",
    )
    quality = _json_object(quality_path, label="reference quality certificate")
    quality_views = quality.get("views")
    if (
        quality.get("schema_version") not in {"robotics_reference_quality.v2","robotics_reference_quality.v3"}
        or quality.get("passed") is not True
        or not isinstance(quality_views, dict)
        or set(quality_views) != {view.view_id for view in manifest.views}
    ):
        raise ValueError("reference quality certificate is not a complete passing artifact")
    if quality.get("schema_version")=="robotics_reference_quality.v3":
        required_fidelity_checks={"source_clock_matches","exact_source_frame_count",
                                  "source_resolution_matches","decoded_frames_match_renderer"}
        if (quality.get("episode_checks",{}).get("fixed_view_motion_evidence") is not True
                or any(row.get("source_fidelity",{}).get("passed") is not True
                       or not all(value is True for value in row.get("source_fidelity",{}).get("checks",{}).values())
                       or set(row.get("source_fidelity",{}).get("checks",{}))!=required_fidelity_checks
                       for row in quality_views.values())):
            raise ValueError("source-aware reference quality lacks motion or frame-fidelity evidence")
        if not any(quality_views[name]["source_fidelity"].get(
                "maximum_local_source_motion_from_first_8bit",0)>=4.
                for name in ("top","exterior_left","exterior_right")):
            raise ValueError("source-aware reference quality has no measured fixed-view motion")
    for view in manifest.views:
        report = quality_views[view.view_id]
        if quality.get("schema_version")=="robotics_reference_quality.v3":
            fidelity=report["source_fidelity"]
            if (fidelity.get("expected_frame_count")!=view.frame_count
                    or fidelity.get("decoded_frame_count")!=view.frame_count
                    or len(fidelity.get("source_frame_sha256",[]))!=view.frame_count
                    or not 0<=fidelity.get("maximum_frame_mse_8bit",float("inf"))<=65.025
                    or not 0<=fidelity.get("maximum_local_codec_mae_8bit",float("inf"))<=8.):
                raise ValueError("source-aware quality measurement differs from reference frames")
        if not isinstance(report, dict) or report.get("passed") is not True:
            raise ValueError(f"reference quality view is not passing: {view.view_id}")
        report_video = _owned_reference_file(
            quality_path,
            root,
            root,
            report.get("path"),
            expected,
            label=f"reference quality video {view.view_id}",
        )
        declared_video = _reference_file(
            root,
            view.video,
            expected,
            label=f"reference video {view.view_id}",
        )
        if report_video != declared_video or report.get("sha256") != view.sha256:
            raise ValueError(
                f"reference quality report points to another video: {view.view_id}"
            )
        if (
            int(report.get("width") or 0) != view.width
            or int(report.get("height") or 0) != view.height
            or int(report.get("frame_count") or 0) != view.frame_count
            or abs(float(report.get("fps") or 0.0) - view.fps) > 1e-6
        ):
            raise ValueError(
                f"reference quality report timeline differs from view: {view.view_id}"
            )
    return expected


def load_reference_manifest(path: str | Path) -> tuple[Path, RoboticsReferenceManifest]:
    source_path = Path(path)
    manifest_path = source_path.resolve()
    manifest = RoboticsReferenceManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    self_contained = manifest.schema_version in {
        "robotics_reference_bundle.v5",
        "robotics_reference_bundle.v6",
    }
    if self_contained and source_path.is_symlink():
        raise ValueError("self-contained robotics reference manifest may not be a symlink")
    artifact_map = (
        _validate_v5_reference_bundle(manifest_path, manifest)
        if self_contained
        else None
    )

    def artifact(value: str, *, label: str, directory: bool = False) -> Path:
        if artifact_map is None:
            return resolve_artifact(manifest_path, value)
        if directory:
            return _reference_directory(
                manifest_path.parent,
                value,
                artifact_map,
                label=label,
            )
        return _reference_file(
            manifest_path.parent,
            value,
            artifact_map,
            label=label,
        )

    trajectory = artifact(manifest.trajectory, label="reference trajectory")
    if not trajectory.is_file() or sha256_file(trajectory) != manifest.trajectory_sha256:
        raise ValueError("reference trajectory is missing or its digest changed")
    trajectory_json = artifact(
        manifest.trajectory_json,
        label="reference JSON trajectory",
    )
    if (
        not trajectory_json.is_file()
        or sha256_file(trajectory_json) != manifest.trajectory_json_sha256
    ):
        raise ValueError("reference JSON trajectory is missing or its digest changed")
    for certificate in (
        manifest.simulation_certificate,
        manifest.physical_render_certificate,
        manifest.quality_certificate,
    ):
        if not artifact(certificate, label="reference certificate").is_file():
            raise FileNotFoundError(f"reference certificate is missing: {certificate}")
    if manifest.schema_version == "robotics_reference_bundle.v6":
        for label, value, expected_digest in (
            ("task graph", manifest.task_graph, manifest.task_graph_sha256),
            (
                "scene generation manifest",
                manifest.scene_generation_manifest,
                manifest.scene_generation_manifest_sha256,
            ),
            (
                "semantic identity certificate",
                manifest.semantic_identity_certificate,
                manifest.semantic_identity_certificate_sha256,
            ),
        ):
            assert value is not None and expected_digest is not None
            provenance = artifact(value, label=label)
            if sha256_file(provenance) != expected_digest:
                raise ValueError(f"reference {label} digest changed")
    for view in manifest.views:
        video = artifact(view.video, label=f"reference video {view.view_id}")
        image = artifact(
            view.conditioning_image,
            label=f"reference conditioning image {view.view_id}",
        )
        if not video.is_file() or not image.is_file():
            raise FileNotFoundError(f"reference view artifact is missing: {view.view_id}")
        if sha256_file(video) != view.sha256:
            raise ValueError(f"reference video digest changed: {view.view_id}")
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"unable to decode reference view: {view.view_id}")
        actual_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if (
            actual_frame_count != view.frame_count
            or abs(actual_fps - view.fps) > 1e-6
            or actual_width != view.width
            or actual_height != view.height
        ):
            raise ValueError(
                f"reference video contract changed: {view.view_id} has "
                f"{actual_frame_count} frames at {actual_fps:g} FPS and "
                f"{actual_width}x{actual_height}, expected {view.frame_count} frames "
                f"at {view.fps:g} FPS and {view.width}x{view.height}"
            )
        base = artifact(
            view.cae_base_scene,
            label=f"reference CAE base {view.view_id}",
            directory=True,
        )
        for name in REQUIRED_CAE_BASE_FILES:
            cae_file = base / name
            if not cae_file.is_file():
                raise FileNotFoundError(f"reference {view.view_id} CAE base is missing {name}")
            if artifact_map is not None:
                relative = cae_file.relative_to(manifest_path.parent).as_posix()
                if artifact_map.get(relative) != sha256_file(cae_file):
                    raise ValueError(
                        f"reference {view.view_id} CAE base digest changed: {name}"
                    )
    return manifest_path, manifest


def resolve_artifact(manifest_path: Path, artifact: str) -> Path:
    path = Path(artifact)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def read_conditioning_view(
    manifest_path: Path,
    manifest: RoboticsReferenceManifest,
    view_id: str,
) -> tuple[np.ndarray, Path]:
    view = manifest.view(view_id)  # type: ignore[arg-type]
    path = resolve_artifact(manifest_path, view.conditioning_image)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to decode conditioning view {view_id}: {path}")
    return image, path


def select_pregrasp_start_frame(
    manifest_path: Path,
    manifest: RoboticsReferenceManifest,
    *,
    max_end_effector_distance_m: float = 0.16,
    min_remaining_seconds: float = 6.0,
) -> int:
    """Select the first operator-ready frame from simulator state, or frame zero.

    Reference rollouts often contain several seconds of an idle/home pose.
    Conditioning a WAM on that pose is both visually weak and unlike the
    task-ready observations used by released robotics models.  Selection is
    based only on immutable simulator state and leaves a substantial future
    suffix; it never inspects generated model output.
    """

    trajectory = resolve_artifact(manifest_path, manifest.trajectory)
    with np.load(trajectory, allow_pickle=False) as payload:
        required = {"end_effector_position", "time_s"}
        subject_key = (
            "subject_position"
            if "subject_position" in payload.files
            else "cup_position"
        )
        if not required.issubset(payload.files) or subject_key not in payload.files:
            return 0
        end_effector = np.asarray(payload["end_effector_position"], dtype=np.float64)
        target = np.asarray(payload[subject_key], dtype=np.float64)
        time_s = np.asarray(payload["time_s"], dtype=np.float64)
    if (
        end_effector.shape != target.shape
        or end_effector.ndim != 2
        or end_effector.shape[1] != 3
        or time_s.shape != (len(end_effector),)
        or len(time_s) < 2
    ):
        raise ValueError("reference task-ready state arrays are inconsistent")
    distance = np.linalg.norm(end_effector - target, axis=1)
    remaining = time_s[-1] - time_s
    candidates = np.flatnonzero(
        (distance <= max_end_effector_distance_m)
        & (remaining >= min_remaining_seconds)
    )
    return int(candidates[0]) if candidates.size else 0


def read_reference_video_frame(
    manifest_path: Path,
    manifest: RoboticsReferenceManifest,
    view_id: str,
    frame_index: int,
) -> np.ndarray:
    view = manifest.view(view_id)  # type: ignore[arg-type]
    if frame_index < 0 or frame_index >= view.frame_count:
        raise ValueError(f"conditioning frame {frame_index} is outside {view_id}")
    video = resolve_artifact(manifest_path, view.video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"unable to decode reference view {view_id}: {video}")
    frame: np.ndarray | None = None
    for _ in range(frame_index + 1):
        ok, value = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"unable to read frame {frame_index} from {view_id}")
        frame = value
    capture.release()
    assert frame is not None
    return frame


def resize_with_pad(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    output[y : y + resized_h, x : x + resized_w] = resized
    return output


def resize_with_pad_openpi(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Match ``openpi_client.image_tools.resize_with_pad`` for one uint8 image.

    Cosmos3-Nano-Policy-DROID is sensitive to the exact released client
    preprocessor.  OpenPI uses PIL bilinear interpolation, floor-sized aspect
    preservation, zero padding, and an early return at the requested size.
    Keep this model-specific path separate from the generic OpenCV letterbox
    used by evaluation and other adapters.
    """

    values = np.asarray(frame)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
        raise ValueError(
            "OpenPI camera input must be one HxWx3 uint8 image, "
            f"got {values.shape} {values.dtype}"
        )
    if width <= 0 or height <= 0:
        raise ValueError("OpenPI output dimensions must be positive")
    source_h, source_w = values.shape[:2]
    if (source_w, source_h) == (width, height):
        return values

    ratio = max(source_w / width, source_h / height)
    resized_h = int(source_h / ratio)
    resized_w = int(source_w / ratio)
    if resized_h < 1 or resized_w < 1:
        raise ValueError("OpenPI resize produced an empty image")
    resized = Image.fromarray(values).resize(
        (resized_w, resized_h), resample=Image.Resampling.BILINEAR
    )
    padded = Image.new(resized.mode, (width, height), 0)
    pad_h = max(0, int((height - resized_h) / 2))
    pad_w = max(0, int((width - resized_w) / 2))
    padded.paste(resized, (pad_w, pad_h))
    return np.asarray(padded, dtype=np.uint8)


def _torch_bilinear_half_uint8(frame: np.ndarray) -> np.ndarray:
    """Match Torch bilinear 2x downsampling with ``align_corners=False``.

    The released client always downsamples a 640x360 exterior view to 320x180.
    At this exact factor, each result pixel is the float32 mean of one 2x2
    source block and the final uint8 cast truncates it.
    """

    values = np.asarray(frame)
    if values.shape != (360, 640, 3) or values.dtype != np.uint8:
        raise ValueError(
            "Cosmos exterior input must be 360x640x3 uint8 before half resize"
        )
    sums = (
        values[0::2, 0::2].astype(np.uint16)
        + values[1::2, 0::2].astype(np.uint16)
        + values[0::2, 1::2].astype(np.uint16)
        + values[1::2, 1::2].astype(np.uint16)
    )
    return (sums // 4).astype(np.uint8)


def prepare_cosmos_droid_camera_view(
    frame: np.ndarray, *, exterior: bool
) -> np.ndarray:
    """Apply the released RoboLab Cosmos3 client camera transform exactly."""

    resized = resize_with_pad_openpi(frame, 640, 360)
    return _torch_bilinear_half_uint8(resized) if exterior else resized


def resize_cover_anchor(
    frame: np.ndarray,
    width: int,
    height: int,
    *,
    zoom: float = 1.0,
    anchor_x: float = 0.5,
    anchor_y: float = 0.5,
) -> np.ndarray:
    """Fill the canvas with a deterministic anchored cover crop and no padding."""

    if zoom < 1.0:
        raise ValueError("cover crop zoom must be at least 1")
    if not 0.0 <= anchor_x <= 1.0 or not 0.0 <= anchor_y <= 1.0:
        raise ValueError("cover crop anchors must be within [0, 1]")
    source_h, source_w = frame.shape[:2]
    scale = max(width / source_w, height / source_h) * zoom
    resized_w = max(width, int(round(source_w * scale)))
    resized_h = max(height, int(round(source_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)
    x = int(round(anchor_x * resized_w - width / 2.0))
    y = int(round(anchor_y * resized_h - height / 2.0))
    x = min(max(0, x), resized_w - width)
    y = min(max(0, y), resized_h - height)
    return np.ascontiguousarray(resized[y : y + height, x : x + width])


def resize_cover_center(
    frame: np.ndarray, width: int, height: int, *, zoom: float = 1.0
) -> np.ndarray:
    """Fill the target canvas with a deterministic centered crop and no padding."""

    return resize_cover_anchor(frame, width, height, zoom=zoom)


def apply_spatial_transform(
    frame: np.ndarray, width: int, height: int, policy: str
) -> np.ndarray:
    if policy == "identity_exact":
        if frame.shape[:2] != (height, width):
            raise ValueError(
                "identity_exact forbids reference resizing: "
                f"source={frame.shape[1]}x{frame.shape[0]} target={width}x{height}"
            )
        return np.ascontiguousarray(frame)
    if policy == "center_crop_cover_zoom_1p0":
        return resize_cover_center(frame, width, height, zoom=1.0)
    if policy == "center_crop_cover_zoom_1p2":
        return resize_cover_center(frame, width, height, zoom=1.2)
    if policy == "center_crop_cover_zoom_1p35":
        return resize_cover_center(frame, width, height, zoom=1.35)
    if policy == "aspect_preserving_letterbox":
        return resize_with_pad(frame, width, height)
    raise ValueError(f"unknown spatial transform policy: {policy!r}")


def compose_cosmos_droid_view(
    wrist: np.ndarray,
    exterior_left: np.ndarray,
    exterior_right: np.ndarray,
) -> np.ndarray:
    """Compose the released Cosmos3 DROID layout.

    Model-native SceneCode panels are copied exactly.  The raw-camera branch is
    retained only as a utility for reproducing the released OpenPI client on
    non-native external inputs; formal harness input is rejected before it can
    enter that branch.
    """

    canvas = np.zeros((540, 640, 3), dtype=np.uint8)
    if (
        wrist.shape == (360, 640, 3)
        and exterior_left.shape == (180, 320, 3)
        and exterior_right.shape == (180, 320, 3)
    ):
        canvas[:360] = wrist
        canvas[360:, :320] = exterior_left
        canvas[360:, 320:] = exterior_right
        return canvas
    canvas[:360] = prepare_cosmos_droid_camera_view(wrist, exterior=False)
    canvas[360:, :320] = prepare_cosmos_droid_camera_view(
        exterior_left, exterior=True
    )
    canvas[360:, 320:] = prepare_cosmos_droid_camera_view(
        exterior_right, exterior=True
    )
    return canvas


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
