import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNTIME = Path(os.environ.get("SCENECODE_RUNTIME_ROOT", "/tmp/SceneCode-main-worldolympiad"))
if not (RUNTIME / "scenecode/agent_utils/asset_router/router.py").is_file():
    pytest.skip("requires the staged SceneCode runtime", allow_module_level=True)
pytest.importorskip("bpy")
sys.path.insert(0, str(RUNTIME))
from scenecode.agent_utils.asset_router import router as module
from autoresearch import scenecode_image_evidence as evidence


@pytest.fixture
def harness(tmp_path, monkeypatch):
    for name in ["SCENECODE_CODE_OBJECT_REFERENCE_IMAGE", "SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST",
                 "SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST_SHA256", "SCENECODE_CODE_OBJECT_TEXT_ONLY",
                 "SCENECODE_CODE_OBJECT_REQUIRE_NATIVE_IMAGE", "SCENECODE_REQUIRE_REFERENCE_IMAGE"]:
        monkeypatch.delenv(name, raising=False)
    router = module.AssetRouter.__new__(module.AssetRouter)
    router.cfg = SimpleNamespace(asset_manager={"code_object": {}})
    item = SimpleNamespace(short_name="cup", description="ceramic cup with a blue handle",
                           dimensions=[.12, .08, .1], code_object_profile=None)
    images, calls, validations = [], [], []
    def image(**kwargs):
        from PIL import Image
        images.append(kwargs)
        for path in kwargs["output_paths"]:
            Image.new("RGB", (8, 8), "white").save(path)
    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(mesh_path=kwargs["output_dir"] / f"mesh_{len(calls)}.glb")
    runner = SimpleNamespace(generate_from_image=generate, generate_from_text=generate)
    image_generator = SimpleNamespace(generate_images=image)
    monkeypatch.setattr(evidence, "record_image_input", lambda *a, **kw: None)
    monkeypatch.setattr(evidence, "record_image_result", lambda *a, **kw: None)
    router._build_code_generated_geometry = lambda **kw: SimpleNamespace(
        geometry_path=kw["result"].mesh_path, item=kw["item"])
    monkeypatch.setattr(module, "create_solid_primitive_glb", lambda **kwargs: None)
    dirs = {name: tmp_path / name for name in ["images_dir", "code_object_dir", "debug_dir"]}
    for path in dirs.values():
        path.mkdir()
    def run(verdicts, attempts=3):
        outcomes = iter(verdicts)
        def validate(**kwargs):
            validations.append(kwargs)
            return next(outcomes)
        router.validate_asset = validate
        return router._try_code_generated_strategy(item=item, max_retries=attempts,
            code_object_runner=runner, image_generator=image_generator, **dirs)
    good = SimpleNamespace(is_acceptable=True, reason="valid", suggestions=[])
    bad = SimpleNamespace(is_acceptable=False, reason="The declared handle is missing",
                          suggestions=["Construct the blue handle and attach it to the cup"])
    return SimpleNamespace(run=run, good=good, bad=bad, item=item, images=images,
                           calls=calls, validations=validations)


@pytest.mark.parametrize("text_only", [False, True])
def test_validation_feedback_reaches_next_generation_without_changing_target(harness, monkeypatch, text_only):
    h = harness
    if text_only:
        monkeypatch.setenv("SCENECODE_CODE_OBJECT_TEXT_ONLY", "1")
    assert h.run([h.bad, h.good]) is not None
    key = "text_input" if text_only else "object_description"
    assert h.calls[0][key] == h.item.description
    for guidance in [h.bad.reason, *h.bad.suggestions]:
        assert guidance in h.calls[1][key]
        if not text_only:
            assert guidance in h.images[1]["object_descriptions"][0]
    assert h.item.description == "ceramic cup with a blue handle"
    assert h.item.dimensions == [.12, .08, .1]
    assert all(v["description"] == h.item.description and not v["use_lenient"] for v in h.validations)
    if not text_only:
        assert h.images[0]["object_descriptions"] == [h.item.description]
        assert h.calls[1]["desired_dimensions"] == h.item.dimensions


def test_successful_first_attempt_keeps_original_inputs_and_no_retry(harness):
    h = harness
    assert h.run([h.good]) is not None
    assert len(h.calls) == len(h.images) == len(h.validations) == 1
    assert h.calls[0]["object_description"] == h.item.description
    assert h.images[0]["object_descriptions"] == [h.item.description]


def test_feedback_cannot_accept_failed_asset_or_increase_attempt_budget(harness):
    h = harness
    assert h.run([h.bad, h.bad], attempts=2) is None
    assert len(h.calls) == len(h.validations) == 2
    assert h.bad.reason in h.calls[1]["object_description"]
    assert all(v["description"] == h.item.description and not v["use_lenient"] for v in h.validations)


def test_external_reference_preserved_while_planner_gets_feedback(harness, tmp_path, monkeypatch):
    from PIL import Image
    h = harness
    reference = tmp_path / "fixed_reference.png"
    Image.new("RGB", (8, 8), "blue").save(reference)
    original = reference.read_bytes()
    monkeypatch.setenv("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE", str(reference))
    assert h.run([h.bad, h.good]) is not None
    assert not h.images
    assert reference.read_bytes() == original
    assert all(c["image_path"].read_bytes() == original for c in h.calls)
    assert h.bad.reason in h.calls[1]["object_description"]


def test_existing_no_validation_mode_is_unchanged(harness):
    h = harness
    assert h.run([], attempts=0) is not None
    assert len(h.calls) == 1 and not h.validations
    assert h.calls[0]["object_description"] == h.item.description
