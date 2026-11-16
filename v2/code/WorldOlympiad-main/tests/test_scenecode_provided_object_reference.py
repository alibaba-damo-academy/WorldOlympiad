import ast
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from autoresearch.scenecode_image_evidence import (
    _canonical_sha, select_provided_object_reference, record_image_input,
)
from wo.tasks.scene import object_reference_env, context_reference_env

RUNTIME = Path(os.environ.get("SCENECODE_RUNTIME_ROOT", "/tmp/SceneCode-main-worldolympiad"))


@pytest.fixture
def bound(tmp_path, monkeypatch):
    task = tmp_path / "tasks/case_a.json"
    task.parent.mkdir()
    geometry = {"overall_dimensions_m": [1.4, .75, .9], "cavity": {"depth_m": .16}}
    authored = {"case_id": "case_a", "objects": [{"object_id": "counter", "geometry_constraints": geometry}]}
    task.write_text(json.dumps(authored))
    folder = tmp_path / "object_references"
    folder.mkdir()
    Image.new("RGB", (32, 32), "blue").save(folder / "counter.png")
    (folder / "producer.txt").write_text("Actual external image producer prompt")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    record = {"mode": "provided_image", "object_id": "counter", "aliases": ["kitchen_counter"],
              "geometry_contract_sha256": _canonical_sha(geometry), "desired_dimensions_m": [1.4, .75, .9],
              "image_path": "counter.png", "image_sha256": sha(folder / "counter.png"),
              "producer": "external image_gen", "producer_prompt_path": "producer.txt",
              "producer_prompt_sha256": sha(folder / "producer.txt")}
    payload = {"schema_version": "cosmos_seven_object_reference_images.v1", "case_id": "case_a",
               "task_sha256": sha(task), "items": [record]}
    manifest = folder / "case_a.json"
    def save():
        manifest.write_text(json.dumps(payload))
        monkeypatch.setenv("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST_SHA256", sha(manifest))
    save()
    monkeypatch.setenv("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST", str(manifest))
    monkeypatch.setenv("SCENECODE_AUTHORED_TASK_REQUIREMENTS", json.dumps(authored))
    monkeypatch.delenv("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE", raising=False)
    item = SimpleNamespace(short_name="kitchen_counter", description="real kitchen counter", dimensions=[1.4, .75, .9], code_object_profile="manipuland")
    return SimpleNamespace(**locals())


def actual_prepare():
    source = RUNTIME / "scenecode/agent_utils/asset_router/router.py"
    if not source.is_file():
        pytest.skip("requires staged SceneCode runtime")
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and any(getattr(m, "name", "") == "_prepare_code_object_inputs" for m in n.body))
    method = next(n for n in owner.body if getattr(n, "name", "") == "_prepare_code_object_inputs")
    namespace = dict(Path=Path, os=os, json=json, hashlib=hashlib, shutil=shutil, time=time,
                     re=__import__("re"), console_logger=logging.getLogger("provided-test"))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), str(source), "exec"), namespace)
    return namespace["_prepare_code_object_inputs"]


def run_prepare(c):
    images = c.tmp_path / "images"
    images.mkdir(exist_ok=True)
    def generate_images(**kw):
        Image.new("RGB", (32, 32), "red").save(kw["output_paths"][0])
    api = SimpleNamespace(generate_images=Mock(side_effect=generate_images), model="configured-server-model")
    owner = SimpleNamespace(cfg=SimpleNamespace(asset_manager={"code_object": {"manipuland_config_path": "original-profile.yml"}}))
    prepared = actual_prepare()(owner, c.item, api, images, c.tmp_path / "code")
    return prepared, api


def test_actual_prepare_preserves_profile_and_binds_external_producer(bound):
    c = bound
    assert object_reference_env(c.task)["SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST"] == str(c.manifest)
    (image, output, profile), api = run_prepare(c)
    api.generate_images.assert_not_called()
    assert profile == "original-profile.yml"
    assert image.read_bytes() == (c.folder / "counter.png").read_bytes()
    receipt = json.loads((output / "object_image_generation_receipt.json").read_text())
    assert receipt["schema_version"] == "scenecode_external_object_image_generation.v1"
    assert receipt["original_server_request_sha256"] is None
    evidence = record_image_input(output, image, method="generate_from_image", generator=api)
    assert evidence["actual_producer"] == "external image_gen"
    assert evidence["generator_model"] is None
    assert evidence["configured_generator_model"] == "configured-server-model"


@pytest.mark.parametrize("failure", ["wrong_name", "substring", "ambiguous", "geometry", "dimensions", "case", "image", "prompt", "producer", "producer_null", "producer_nonstring", "manifest"])
def test_binding_failures_reject_before_api(bound, failure):
    c = bound
    if failure == "wrong_name": c.item.short_name = "ceramic_pot"
    elif failure == "substring": c.item.short_name = "large_kitchen_counter"
    elif failure == "ambiguous": c.payload["items"].append(dict(c.record)); c.save()
    elif failure == "geometry": c.record["geometry_contract_sha256"] = "wrong"; c.save()
    elif failure == "dimensions": c.item.dimensions = [.7, .375, .45]
    elif failure == "case": c.payload["case_id"] = "other_case"; c.save()
    elif failure == "image": (c.folder / "counter.png").write_bytes(b"changed")
    elif failure == "prompt": (c.folder / "producer.txt").write_text("changed")
    elif failure == "producer": c.record["producer"] = ""; c.save()
    elif failure == "producer_null": c.record["producer"] = None; c.save()
    elif failure == "producer_nonstring": c.record["producer"] = {}; c.save()
    else: c.manifest.write_text("{}")
    with pytest.raises(ValueError): run_prepare(c)


def test_copy_is_revalidated_in_actual_prepare(bound, monkeypatch):
    c = bound
    def corrupt(source, destination):
        Image.new("RGB", (32, 32), "green").save(destination)
    monkeypatch.setattr(shutil, "copy2", corrupt)
    with pytest.raises(ValueError, match="copy or binding changed"): run_prepare(c)
    assert not list(c.tmp_path.glob("code/**/object_image_generation_receipt.json"))


def test_task_hash_rejects_changed_spec(bound):
    bound.task.write_text("{}")
    with pytest.raises(ValueError, match="TaskSpec"): object_reference_env(bound.task)


def test_symlink_task_uses_callers_run_sidecars(bound):
    c = bound
    current = c.tmp_path / "new_run/tasks/case_a.json"
    current.parent.mkdir(parents=True)
    current.symlink_to(c.task)
    assert object_reference_env(current) == {}
    assert context_reference_env(current) == {}
    new_context = current.parent.parent / "context_references/case_a.json"
    new_context.parent.mkdir()
    new_context.write_text(json.dumps({"schema_version": "scenecode_context_reference.v1", "case_id": "case_a", "items": []}))
    assert context_reference_env(current)["SCENECODE_CONTEXT_REFERENCE_MANIFEST"] == str(new_context)
    own = current.parent.parent / "object_references/case_a.json"
    own.parent.mkdir()
    own.write_bytes(c.manifest.read_bytes())
    assert object_reference_env(current)["SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST"] == str(own)


def test_unprovided_legacy_profile_manifest_unchanged(bound):
    c = bound
    c.record.pop("mode")
    c.save()
    assert select_provided_object_reference(c.item) is None


def test_mixed_manifest_explicit_native_profile_still_calls_original_api(bound):
    c = bound
    c.payload["items"].append({"mode": "native_profile", "object_id": "native_lamp", "aliases": ["lamp"]})
    c.save()
    c.item.short_name = "lamp"
    c.item.dimensions = [.3, .3, .5]
    (image, output, _), api = run_prepare(c)
    api.generate_images.assert_called_once()
    assert image.is_file()
    assert json.loads((output / "object_image_generation_receipt.json").read_text())["schema_version"] == "scenecode_native_object_image_generation.v2"


def test_mixed_manifest_cross_mode_alias_ambiguity_is_rejected(bound):
    c = bound
    c.payload["items"].append({"mode": "native_profile", "object_id": "lamp", "aliases": ["kitchen_counter"]})
    c.save()
    with pytest.raises(ValueError, match="one exact canonical"):
        run_prepare(c)


def test_no_manifest_preserves_original_native_generation(bound, monkeypatch):
    c = bound
    monkeypatch.delenv("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST")
    monkeypatch.delenv("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST_SHA256")
    # The actual original generation branch is called; simulate only its API result.
    def generate_images(**kw):
        Image.new("RGB", (32, 32), "red").save(kw["output_paths"][0])
    images = c.tmp_path / "native_images"
    images.mkdir()
    api = SimpleNamespace(generate_images=Mock(side_effect=generate_images), model="server")
    owner = SimpleNamespace(cfg=SimpleNamespace(asset_manager={}))
    image, output, profile = actual_prepare()(owner, c.item, api, images, c.tmp_path / "native_code")
    api.generate_images.assert_called_once()
    receipt = output / "object_image_generation_receipt.json"
    if receipt.exists():
        assert json.loads(receipt.read_text())["schema_version"] == "scenecode_native_object_image_generation.v2"
    assert image.is_file()
