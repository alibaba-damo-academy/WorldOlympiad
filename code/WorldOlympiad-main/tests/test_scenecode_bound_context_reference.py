import ast
import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

RUNTIME = Path(os.environ.get("SCENECODE_RUNTIME_ROOT", "/tmp/SceneCode-main-worldolympiad"))
SOURCE = RUNTIME / "scenecode/agent_utils/image_generation.py"
if not SOURCE.is_file():
    pytest.skip("requires staged SceneCode runtime", allow_module_level=True)


@pytest.fixture
def context_api(tmp_path, monkeypatch):
    import shutil
    tree = ast.parse(SOURCE.read_text())
    names = {"_context_sha", "_checked_context_reference", "_write_context_receipt",
             "_reuse_bound_context_image", "_extract_and_save_openai_image"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OpenAIImageGenerator")
    edit = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_edit_image")
    namespace = dict(Path=Path, Image=Image, hashlib=hashlib, json=json, os=os,
                     shutil=shutil, base64=base64, time=__import__("time"),
                     console_logger=logging.getLogger("context-reference-test"))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *functions, edit], type_ignores=[])), str(SOURCE), "exec"), namespace)
    reference, generated = tmp_path / "room.png", tmp_path / "curated.png"
    Image.new("RGB", (32, 32), "white").save(reference)
    Image.new("RGB", (32, 32), "blue").save(generated)
    prompt = "The exact original room edit request."
    producer_prompt = tmp_path / "producer.txt"
    producer_prompt.write_text("Modified external producer instructions, separately recorded.")
    sha = namespace["_context_sha"]
    item = {"kind": "furniture_context", "input_image_sha256": sha(reference),
            "request_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "output_image": generated.name, "output_image_sha256": sha(generated),
            "producer": "external image_gen semantic design reference",
            "producer_prompt_path": producer_prompt.name, "producer_prompt_sha256": sha(producer_prompt)}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "scenecode_context_reference.v1", "items": [item]}))
    monkeypatch.setenv("SCENECODE_CONTEXT_REFERENCE_MANIFEST", str(manifest))
    monkeypatch.setenv("SCENECODE_CONTEXT_REFERENCE_MANIFEST_SHA256", sha(manifest))
    api = SimpleNamespace(model="configured-image-model", image_api=SimpleNamespace(edit=Mock()))
    output = tmp_path / "stage" / "context.png"
    return SimpleNamespace(ns=namespace, reference=reference, generated=generated, prompt=prompt,
                           producer_prompt=producer_prompt, manifest=manifest, api=api,
                           output=output, item=item, sha=sha)


def test_exact_external_request_uses_real_decodable_image_and_preserves_producer(context_api):
    c = context_api
    result = c.ns["_edit_image"](c.api, c.prompt, c.reference, c.output, context_kind="furniture_context")
    assert result == c.output and c.output.read_bytes() == c.generated.read_bytes()
    c.api.image_api.edit.assert_not_called()
    receipt = json.loads(c.output.with_suffix(".png.context.json").read_text())
    assert receipt["request_prompt_sha256"] != receipt["producer_prompt_sha256"]
    assert receipt["producer"] == c.item["producer"]
    assert receipt["external_manifest_sha256"] == c.sha(c.manifest)


@pytest.mark.parametrize("changed", ["request", "input", "image", "producer_prompt", "manifest", "undecodable"])
def test_tampering_or_request_mismatch_fails_before_api(context_api, changed, monkeypatch):
    c = context_api
    prompt = c.prompt
    if changed == "request":
        prompt += " changed"
    elif changed == "input":
        Image.new("RGB", (32, 32), "black").save(c.reference)
    elif changed == "image":
        c.generated.write_bytes(b"changed")
    elif changed == "producer_prompt":
        c.producer_prompt.write_text("changed")
    elif changed == "manifest":
        c.manifest.write_text("{}")
    else:
        c.generated.write_bytes(b"not an image")
        payload = json.loads(c.manifest.read_text())
        payload["items"][0]["output_image_sha256"] = c.sha(c.generated)
        c.manifest.write_text(json.dumps(payload))
        monkeypatch.setenv("SCENECODE_CONTEXT_REFERENCE_MANIFEST_SHA256", c.sha(c.manifest))
    with pytest.raises((ValueError, OSError)):
        c.ns["_edit_image"](c.api, prompt, c.reference, c.output, context_kind="furniture_context")
    c.api.image_api.edit.assert_not_called()


def test_verified_same_request_cache_survives_service_unavailability(context_api, monkeypatch):
    c = context_api
    c.ns["_edit_image"](c.api, c.prompt, c.reference, c.output, context_kind="furniture_context")
    monkeypatch.delenv("SCENECODE_CONTEXT_REFERENCE_MANIFEST")
    monkeypatch.delenv("SCENECODE_CONTEXT_REFERENCE_MANIFEST_SHA256")
    c.api.image_api.edit.side_effect = RuntimeError("image service unavailable")
    assert c.ns["_edit_image"](c.api, c.prompt, c.reference, c.output, context_kind="furniture_context") == c.output
    c.api.image_api.edit.assert_not_called()


def test_no_manifest_preserves_api_then_binds_api_cache(context_api, monkeypatch):
    c = context_api
    monkeypatch.delenv("SCENECODE_CONTEXT_REFERENCE_MANIFEST")
    monkeypatch.delenv("SCENECODE_CONTEXT_REFERENCE_MANIFEST_SHA256")
    c.output.parent.mkdir()
    data = base64.b64encode(c.generated.read_bytes()).decode()
    c.api.image_api.edit.return_value = SimpleNamespace(data=[SimpleNamespace(b64_json=data, url=None)])
    c.ns["_edit_image"](c.api, c.prompt, c.reference, c.output, context_kind="furniture_context")
    assert c.api.image_api.edit.call_count == 1
    c.api.image_api.edit.side_effect = RuntimeError("image service unavailable")
    c.ns["_edit_image"](c.api, c.prompt, c.reference, c.output, context_kind="furniture_context")
    assert c.api.image_api.edit.call_count == 1


def test_case_sidecar_is_scoped_and_absence_is_unchanged(tmp_path):
    from wo.tasks.scene import context_reference_env
    task = tmp_path / "tasks" / "case_a.json"
    task.parent.mkdir()
    task.write_text("{}")
    assert context_reference_env(task) == {}
    directory = tmp_path / "context_references"
    directory.mkdir()
    (directory / "case_b.json").write_text("{}")
    assert context_reference_env(task) == {}
    manifest = directory / "case_a.json"
    manifest.write_text(json.dumps({"schema_version": "scenecode_context_reference.v1", "case_id": "case_a", "items": []}))
    result = context_reference_env(task)
    assert Path(result["SCENECODE_CONTEXT_REFERENCE_MANIFEST"]) == manifest
    assert result["SCENECODE_CONTEXT_REFERENCE_MANIFEST_SHA256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({"schema_version": "scenecode_context_reference.v1", "case_id": "case_b", "items": []}))
    with pytest.raises(ValueError, match="case"):
        context_reference_env(task)


@pytest.mark.parametrize("kind", [None, "manipuland_context", "product_edit"])
def test_unbound_later_operation_still_uses_original_api(context_api, kind):
    c = context_api
    c.output.parent.mkdir()
    data = base64.b64encode(c.generated.read_bytes()).decode()
    c.api.image_api.edit.return_value = SimpleNamespace(data=[SimpleNamespace(b64_json=data, url=None)])
    c.ns["_edit_image"](c.api, c.prompt + " for manipulands", c.reference, c.output,
                       context_kind=kind)
    c.api.image_api.edit.assert_called_once()
    assert json.loads(c.output.with_suffix(".png.context.json").read_text())["producer"] == "configured SceneCode image API"


def test_cache_receipt_cannot_point_at_another_legitimate_image(context_api, monkeypatch):
    c = context_api
    c.ns["_edit_image"](c.api, c.prompt, c.reference, c.output, context_kind="furniture_context")
    monkeypatch.delenv("SCENECODE_CONTEXT_REFERENCE_MANIFEST")
    monkeypatch.delenv("SCENECODE_CONTEXT_REFERENCE_MANIFEST_SHA256")
    receipt = c.output.with_suffix(".png.context.json")
    payload = json.loads(receipt.read_text())
    payload["output_image"] = str(c.generated)
    receipt.write_text(json.dumps(payload))
    Image.new("RGB", (32, 32), "green").save(c.output)
    with pytest.raises(ValueError, match="this output path"):
        c.ns["_edit_image"](c.api, c.prompt, c.reference, c.output, context_kind="furniture_context")
    c.api.image_api.edit.assert_not_called()


def test_external_copy_is_rechecked_before_receipt(context_api, monkeypatch):
    c = context_api
    def corrupt_copy(source, destination):
        Image.new("RGB", (32, 32), "green").save(destination)
    monkeypatch.setattr(c.ns["shutil"], "copy2", corrupt_copy)
    with pytest.raises(ValueError, match="expected output bytes"):
        c.ns["_edit_image"](c.api, c.prompt, c.reference, c.output, context_kind="furniture_context")
    assert not c.output.with_suffix(".png.context.json").exists()
    c.api.image_api.edit.assert_not_called()
