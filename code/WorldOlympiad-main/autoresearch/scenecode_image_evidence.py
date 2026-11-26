"""Bind SceneCode asset generation to the image it actually consumes."""
from __future__ import annotations

import hashlib
import json
import copy
import dataclasses
import os
from pathlib import Path


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _bound_file(root, record, path_key, hash_key):
    path = root / record[path_key]
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    if path.is_symlink() or not resolved.is_file() or hashlib.sha256(resolved.read_bytes()).hexdigest() != record.get(hash_key):
        raise ValueError("provided object reference file is missing or changed: " + path_key)
    return resolved


def select_provided_object_reference(item):
    """Select an explicit object/geometry-bound image; legacy prompt profiles stay unchanged."""
    name = os.environ.get("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST")
    if not name:
        return None
    manifest = Path(name).resolve()
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != os.environ.get("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST_SHA256"):
        raise ValueError("provided object reference manifest changed")
    payload = json.loads(manifest.read_text())
    if payload.get("schema_version") != "cosmos_seven_object_reference_images.v1":
        raise ValueError("unsupported object reference manifest")
    records = payload.get("items", [])
    if not any(r.get("mode") == "provided_image" for r in records):
        return None
    authored = json.loads(os.environ.get("SCENECODE_AUTHORED_TASK_REQUIREMENTS", "{}"))
    if not payload.get("case_id") or authored.get("case_id") != payload["case_id"]:
        raise ValueError("provided object reference case mismatch")
    matches = [r for r in records if str(item.short_name).casefold() in
               [str(n).casefold() for n in [r["object_id"], *r.get("aliases", [])]]]
    if len(matches) != 1:
        raise ValueError("provided reference requires one exact canonical name/alias: " + str(item.short_name))
    record = matches[0]
    if record.get("mode") in (None, "native_profile"):
        return None
    if record.get("mode") != "provided_image":
        raise ValueError("unknown object reference mode")
    objects = [o for o in authored.get("objects", []) if o.get("object_id") == record["object_id"]]
    if len(objects) != 1:
        raise ValueError("provided object is absent or ambiguous in authored task")
    geometry = objects[0].get("geometry_constraints", {})
    if _canonical_sha(geometry) != record.get("geometry_contract_sha256"):
        raise ValueError("provided object geometry contract changed")
    dimensions = list(geometry.get("overall_dimensions_m", []))
    if not dimensions or dimensions != record.get("desired_dimensions_m") or dimensions != list(item.dimensions):
        raise ValueError("provided object dimensions disagree with authored geometry")
    if not isinstance(record.get("producer"), str) or not record["producer"].strip():
        raise ValueError("provided reference producer is required")
    image = _bound_file(manifest.parent, record, "image_path", "image_sha256")
    prompt = _bound_file(manifest.parent, record, "producer_prompt_path", "producer_prompt_sha256")
    if not prompt.read_text().strip():
        raise ValueError("provided reference producer prompt is empty")
    from PIL import Image
    with Image.open(image) as pixels:
        pixels.verify()
    return dict(record, image_path=str(image), producer_prompt_path=str(prompt),
                reference_manifest=str(manifest), reference_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                task_sha256=payload.get("task_sha256"))


def record_provided_object_reference(output_dir, image_path, item, selected):
    verified = select_provided_object_reference(item)
    if verified != selected or hashlib.sha256(Path(image_path).read_bytes()).hexdigest() != selected["image_sha256"]:
        raise ValueError("provided reference copy or binding changed")
    from PIL import Image
    with Image.open(image_path) as pixels:
        pixels.verify()
    receipt = dict(selected, schema_version="scenecode_external_object_image_generation.v1",
                   object_short_name=str(item.short_name), object_description=str(item.description),
                   generated_image_path=str(Path(image_path).resolve()),
                   generated_image_sha256=selected["image_sha256"],
                   original_server_request_sha256=None,
                   scope="external semantic product reference; no native-server generation or physical success claim")
    (Path(output_dir) / "object_image_generation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def record_image_input(output_dir, image_path, *, method, generator=None):
    output = Path(output_dir)
    image = Path(image_path).resolve()
    if not image.is_file() or not image.stat().st_size:
        raise ValueError("SceneCode reference image is missing or empty")
    from PIL import Image
    with Image.open(image) as pixels:
        pixels.verify()
    generated = output / "object_image_generation_receipt.json"
    provenance = json.loads(generated.read_text()) if generated.is_file() else None
    external = image.parent / "image_receipt.json"
    if provenance is None and external.is_file():
        provenance = json.loads(external.read_text())
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    if provenance and provenance.get("generated_image_sha256", provenance.get("image_sha256")) != digest:
        raise ValueError("SceneCode image differs from its generation receipt")
    receipt = {
        "schema_version": "scenecode_asset_image_input.v1",
        "image_path": str(image), "image_sha256": digest,
        "pipeline_method": method, "status": "submitted",
        "generation_receipt": provenance,
        "generator_class": None if provenance and provenance.get("mode") == "provided_image" else type(generator).__name__ if generator else None,
        "generator_model": None if provenance and provenance.get("mode") == "provided_image" else getattr(generator, "model", None),
        "actual_producer": provenance.get("producer") if provenance else None,
        "configured_generator_class": type(generator).__name__ if generator else None,
        "configured_generator_model": getattr(generator, "model", None),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "asset_image_input.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def record_image_result(output_dir, result):
    output = Path(output_dir)
    receipt_path = output / "asset_image_input.json"
    receipt = json.loads(receipt_path.read_text())
    image = Path(receipt["image_path"])
    if hashlib.sha256(image.read_bytes()).hexdigest() != receipt["image_sha256"]:
        raise ValueError("SceneCode input image changed during asset generation")
    mesh = Path(result.mesh_path).resolve()
    if not mesh.is_file():
        raise ValueError("image-conditioned SceneCode pipeline produced no mesh")
    receipt.update(status="generated", mesh_path=str(mesh),
                   mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
                   asset_status=result.status)
    if result.urdf_path is not None:
        urdf = Path(result.urdf_path).resolve()
        receipt.update(urdf_path=str(urdf), urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest())
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def prepare_articulated_image_result(output_dir, result):
    """Normalize a CodeObject export boundary while preserving its raw URDF."""
    if result.urdf_path is None:
        return result
    from .scenecode_code_object_urdf_frames import align_world_baked_mesh_frames
    output=Path(output_dir).resolve();source=Path(result.urdf_path).resolve()
    aligned=source.with_name(source.stem+'_verified_link_frames.urdf')
    receipt_path=aligned.with_suffix('.frame_alignment.json')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if aligned.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('source_urdf_sha256')!=digest(source) or receipt.get('output_urdf_sha256')!=digest(aligned):
            raise ValueError('cached CodeObject URDF frame derivative changed')
    else:
        receipt=align_world_baked_mesh_frames(source,output/'blender_output/parts',aligned)
    if dataclasses.is_dataclass(result):
        normalized=dataclasses.replace(result,urdf_path=aligned)
    else:
        normalized=copy.copy(result);normalized.urdf_path=aligned
    image_receipt=output/'asset_image_input.json'
    if image_receipt.exists():
        metadata=json.loads(image_receipt.read_text())
        metadata.update(raw_generated_urdf_path=str(source),raw_generated_urdf_sha256=digest(source),
            urdf_frame_alignment=str(receipt_path),urdf_frame_alignment_sha256=digest(receipt_path))
        image_receipt.write_text(json.dumps(metadata,indent=2)+'\n')
    return normalized


def validate_transport_resume(output_dir, image_path, receipt_path):
    """Keep checked cached geometry immutable while changing API transport."""
    output=Path(output_dir)
    receipt=json.loads(Path(receipt_path).read_text())
    if receipt.get('schema_version')!='scenecode_transport_resume.v1':
        raise ValueError('unsupported SceneCode transport resume receipt')
    files={
        'image_sha256':Path(image_path),
        'object_plan_sha256':output/'ObjectPlan.json',
        'plan_check_history_sha256':output/'plan_check_history.json',
        'responses_helper_sha256':Path(__file__).with_name('scenecode_openai_stream.py'),
    }
    for key,file in files.items():
        if hashlib.sha256(file.read_bytes()).hexdigest()!=receipt.get(key):
            raise ValueError('SceneCode transport resume input changed: '+key)
    for relative,digest in receipt.get('cached_part_sources',{}).items():
        file=(output/relative).resolve();file.relative_to(output.resolve())
        if hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            raise ValueError('cached SceneCode part changed before transport resume')
    return receipt
