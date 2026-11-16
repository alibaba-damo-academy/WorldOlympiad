import json
import os
import asyncio
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from autoresearch.scenecode_object_requirements import (
    contextual_messages, object_requirements, prepare_request_context,
)


def test_metric_requirements_reach_context_without_guessing_image_dimensions(monkeypatch):
    monkeypatch.delenv("SCENECODE_AUTHORED_TASK_REQUIREMENTS", raising=False)
    text = object_requirements("Bore diameter 0.055 m and depth 0.04 m.", [.08, .08, .05])
    assert "Bore diameter 0.055 m and depth 0.04 m." in text
    assert "[0.08, 0.08, 0.05]" in text
    assert text.splitlines()[-1] == "One model unit is 0.08 m; divide metre lengths by 0.08."
    assert text.count("One model unit") == 1
    assert object_requirements() == ""


@pytest.mark.parametrize("dims", [[0, 1, 1], [1, 2], [1, 2, float("nan")]])
def test_invalid_dimensions_are_not_silently_used(dims):
    with pytest.raises(ValueError):
        object_requirements("object", dims)


def test_context_preserves_image_messages_and_does_not_leak_between_requests():
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "image"}}]}]
    a = contextual_messages(messages, "bore 55 mm")
    b = contextual_messages(messages, "plain cube")
    assert a[0] == b[0] == messages[0]
    assert len(messages) == 1
    assert a[-1]["content"] == "bore 55 mm"
    assert b[-1]["content"] == "plain cube"


def test_old_image_only_plan_is_archived_but_matching_context_is_resumed(tmp_path):
    (tmp_path / "ObjectPlan.json").write_text("old image estimate")
    (tmp_path / "code").mkdir()
    (tmp_path / "code/part.py").write_text("old part")
    prepare_request_context(tmp_path, "bore 55 mm")
    history = list((tmp_path / ".context_history").iterdir())
    assert len(history) == 1
    assert (history[0] / "ObjectPlan.json").read_text() == "old image estimate"
    assert (history[0] / "code/part.py").read_text() == "old part"
    assert not (tmp_path / "ObjectPlan.json").exists()
    (tmp_path / "ObjectPlan.json").write_text("new correct plan")
    prepare_request_context(tmp_path, "bore 55 mm")
    assert (tmp_path / "ObjectPlan.json").read_text() == "new correct plan"
    assert len(list((tmp_path / ".context_history").iterdir())) == 1


def _runtime():
    import os
    path = Path(os.environ.get("SCENECODE_TEST_RUNTIME", "/nonexistent"))
    if not path.is_dir():
        pytest.skip("native SceneCode runtime integration is checked on deployment host")
    return path


@pytest.mark.parametrize("method", ["generate_from_image", "generate_articulated_from_image"])
def test_native_image_runner_forwards_context_to_pipeline(tmp_path, method, monkeypatch):
    import sys
    sys.path.insert(0, str(_runtime()))
    # Conversion is outside this interface test; no Blender module is required.
    monkeypatch.setitem(sys.modules, "scenecode.agent_utils.mesh_utils",
                        SimpleNamespace(convert_gltf_to_glb=lambda *a, **k: None))
    from scenecode.agent_utils.code_object_generation import CodeObjectRunner
    original = "a block with a bore and a raised U-pull"
    geometry = {"overall_dimensions_m": [.16, .10, .12], "bore_diameter_m": .055, "bore_depth_m": .04, "handle_crossbar_length_m": .12, "handle_clearance_m": .04}
    authored = json.dumps({"case_id": "runner_fixture", "objects": [
        {"object_id": "block", "visual_description": original, "geometry_constraints": geometry},
        {"object_id": "unrelated", "visual_description": "UNRELATED_ASSET_TEXT"}],
        "action_program": [{"operation": "transport"}], "fps": 15})
    monkeypatch.setenv("SCENECODE_AUTHORED_TASK_REQUIREMENTS", authored)
    monkeypatch.delenv("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST", raising=False)
    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    calls = []
    class FakePipeline:
        def __init__(self, **kwargs): pass
        async def run(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(success=True, status="no_movable_parts")
    runner = object.__new__(CodeObjectRunner)
    runner._config_path = tmp_path / "config.yaml"
    runner._max_attempts = 1
    runner._reserve_port = lambda: 9990
    runner._release_port = lambda port: None
    runner._load_pipeline_class = lambda *args: FakePipeline
    runner._resolve_mesh_path = lambda **kwargs: image
    runner._build_generation_result = lambda **kwargs: kwargs
    out = tmp_path / "output"
    getattr(runner, method)(image_path=image, output_dir=out,
        object_description="shortened router description", desired_dimensions=[.16, .10, .12], asset_name="block")
    assert calls[0]["image_path"] == str(image)
    assert original in calls[0]["request_context"]
    selected = json.loads(next(line.split("=", 1)[1] for line in calls[0]["request_context"].splitlines() if line.startswith("geometry_constraints=")))
    assert selected == geometry
    context = calls[0]["request_context"]
    assert "Selected asset: block" in context
    assert context.splitlines()[-1] == "One model unit is 0.16 m; divide metre lengths by 0.16."
    assert context.count("One model unit") == 1
    for omitted in ("UNRELATED_ASSET_TEXT", "action_program", "fps", "shortened router description"):
        assert omitted not in context
    assert os.environ["SCENECODE_AUTHORED_TASK_REQUIREMENTS"] == authored
    assert (out / "request_context.txt").read_text() == calls[0]["request_context"]


def test_native_llm_boundary_sees_context_for_every_agent():
    source = _runtime() / "scenecode/code_object/agent_framework/agents/base_agent.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseAgent")
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_call_llm")
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])), str(source), "exec"), namespace)
    captured = []
    async def chat(messages, **kwargs):
        captured.append(messages)
        return "ok"
    agent = SimpleNamespace(llm_client=SimpleNamespace(chat=chat), request_context="exact dimensions")
    assert asyncio.run(namespace["_call_llm"](agent, [{"role": "user", "content": "construct"}])) == "ok"
    assert captured[0][-1]["content"] == "exact dimensions"
