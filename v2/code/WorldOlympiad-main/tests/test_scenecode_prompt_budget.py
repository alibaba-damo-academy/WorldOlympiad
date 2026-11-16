import ast
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from autoresearch.scenecode_object_requirements import (
    CONTEXT_MAX_CHARS, PROMPT_MAX_CHARS, contextual_messages, measure_messages,
    object_requirements, prepare_request_context,
)


@pytest.fixture
def task(monkeypatch):
    payload = {"case_id": "typed_fixture", "objects": [
        {"object_id": "counter", "visual_description": "a light-oak kitchen counter",
         "geometry_constraints": {"overall_dimensions_m": [1.4, .75, .9],
                                  "cavity": {"depth_m": .16, "clear_inner_opening_m": [.32, .3]}}},
        {"object_id": "same_size_other", "visual_description": "a metal table",
         "geometry_constraints": {"overall_dimensions_m": [1.4, .75, .9]}},
        {"object_id": "ceramic_pot", "visual_description": "an empty ceramic crock",
         "geometry_constraints": {"overall_dimensions_m": [.18, .18, .11]}},
        {"object_id": "ceramic_pot_lid", "visual_description": "a ceramic knobbed lid",
         "geometry_constraints": {"overall_dimensions_m": [.184, .184, .048]}}],
        "action_program": [{"operation": "cover"}],
        "success_predicates": [{"kind": "covering", "subject": "ceramic_pot_lid", "object": "ceramic_pot"}],
        "fps": 15, "requested_wams": ["test_model"], "initial_state": "not copied into asset requirements"}
    monkeypatch.setenv("SCENECODE_AUTHORED_TASK_REQUIREMENTS", json.dumps(payload))
    monkeypatch.delenv("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST", raising=False)
    monkeypatch.delenv("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST_SHA256", raising=False)
    return payload


def test_identified_asset_keeps_complete_own_geometry_without_task_json(task):
    before = os.environ["SCENECODE_AUTHORED_TASK_REQUIREMENTS"]
    text = object_requirements("router paraphrase", [1.4, .75, .9], asset_name="counter")
    geometry = json.loads(next(line.split("=", 1)[1] for line in text.splitlines() if line.startswith("geometry_constraints=")))
    assert geometry == task["objects"][0]["geometry_constraints"]
    assert text.count("a light-oak kitchen counter") == 1
    for forbidden in ["ceramic_pot", "action_program", "success_predicates", "requested_wams", "initial_state", "fps", "router paraphrase"]:
        assert forbidden not in text
    assert os.environ["SCENECODE_AUTHORED_TASK_REQUIREMENTS"] == before
    assert len(text) <= CONTEXT_MAX_CHARS


def test_equal_dimensions_never_select_identity(task):
    with pytest.raises(ValueError, match="canonical"):
        object_requirements("counter", [1.4, .75, .9])
    selected = object_requirements("", [1.4, .75, .9], asset_name="same_size_other")
    assert "a metal table" in selected and "light-oak" not in selected
    with pytest.raises(ValueError, match="identity"):
        object_requirements("a light-oak kitchen counter", [1.4, .75, .9], asset_name="unknown")


def test_explicit_alias_and_cross_owner_ambiguity(task, tmp_path, monkeypatch):
    manifest = tmp_path / "objects.json"
    payload = {"case_id": task["case_id"], "items": [{"object_id": "counter", "aliases": ["counter_unit"]}]}
    def save():
        manifest.write_text(json.dumps(payload))
        monkeypatch.setenv("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST", str(manifest))
        monkeypatch.setenv("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    save()
    assert "Selected asset: counter" in object_requirements("", [1.4, .75, .9], asset_name="counter_unit")
    payload["items"].append({"object_id": "same_size_other", "aliases": ["counter_unit"]})
    save()
    with pytest.raises(ValueError, match="ambiguous"):
        object_requirements("", [1.4, .75, .9], asset_name="counter_unit")


def test_host_components_keep_semantics_and_appearance(task, monkeypatch):
    component = {"component_id": "drawer", "host_object_id": "counter", "joint_type": "prismatic", "semantic_kind": "drawer", "visual_description": "a wood drawer with a metal handle", "geometry_constraints": {"handle_clearance_m": .04}}
    task["articulation_components"] = [component, dict(component, component_id="other", host_object_id="same_size_other")]
    monkeypatch.setenv("SCENECODE_AUTHORED_TASK_REQUIREMENTS", json.dumps(task))
    text = object_requirements("", [1.4, .75, .9], asset_name="counter")
    data = json.loads(next(line.split("=", 1)[1] for line in text.splitlines() if line.startswith("components=")))
    assert data == [{key: value for key, value in component.items() if key != "host_object_id"}]


def test_context_budget_precedes_cache_archive(tmp_path):
    plan = tmp_path / "ObjectPlan.json";plan.write_text('{"parts": []}')
    with pytest.raises(ValueError, match="own asset context"):
        prepare_request_context(tmp_path, "x" * 1601)
    assert plan.exists() and not (tmp_path / ".context_history").exists()
    (tmp_path / "request_context.txt").write_text("old")
    prepare_request_context(tmp_path, "new")
    assert not plan.exists()
    assert next((tmp_path / ".context_history").glob("*/ObjectPlan.json")).read_text() == '{"parts": []}'


def test_final_exact_character_boundary_and_no_context_bypass():
    messages = [{"role": "user", "content": ""}]
    overhead = measure_messages(messages)["characters"]
    messages[0]["content"] = "x" * (PROMPT_MAX_CHARS - overhead)
    assert measure_messages(messages)["characters"] == PROMPT_MAX_CHARS
    contextual_messages(messages, "")
    messages[0]["content"] += "x"
    with pytest.raises(ValueError, match="assembled prompt"):
        contextual_messages(messages, "")


def test_utf8_budget_rejects_chinese_without_truncation():
    messages = [{"role": "user", "content": "中" * 27000}]
    assert measure_messages(messages)["characters"] < PROMPT_MAX_CHARS
    assert measure_messages(messages)["utf8_bytes"] > 80000
    with pytest.raises(ValueError, match="UTF-8"):
        contextual_messages(messages, "")
    assert len(messages[0]["content"]) == 27000


@pytest.mark.parametrize("where", ["tool_history", "call_arguments", "tool_schema"])
def test_all_tool_text_is_budgeted(where):
    messages = [{"role": "user", "content": "short"}];tools = None
    if where == "tool_history": messages.append({"role": "tool", "tool_call_id": "call_1", "content": "x" * 40000})
    elif where == "call_arguments": messages.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "execute_code", "arguments": "x" * 40000}}]})
    else: tools = [{"type": "function", "function": {"name": "build", "description": "x" * 40000}}]
    with pytest.raises(ValueError, match="assembled prompt"):
        contextual_messages(messages, "", tools=tools)


def test_images_are_excluded_and_original_messages_remain_unchanged():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 100000}}
    messages = [{"role": "user", "content": [image, {"type": "text", "text": "short"}]}]
    before = copy.deepcopy(messages)
    result = contextual_messages(messages, "own context")
    assert messages == before and result[0] == before[0]
    assert measure_messages(result)["characters"] < 200
    assert measure_messages(result)["images"][0]["reference_characters"] > 100000


def test_compiled_scene_write_rejects_before_creating_output(tmp_path):
    from autoresearch.scene_prompting import write_prompt_bundle
    target = tmp_path / "rejected"
    with pytest.raises(ValueError, match="compiled scene prompt"):
        write_prompt_bundle(target, SimpleNamespace(domain="robotics_manipulation"),
                            "x" * 1601, None, source="test")
    assert not target.exists()


@pytest.mark.parametrize("oversized", ["whole_task_context", "tools"])
def test_actual_call_boundary_rejects_locally_with_zero_api_calls(oversized):
    runtime = Path(os.environ.get("SCENECODE_RUNTIME_ROOT", "/root/autodl-tmp/worldolympiad2/runtime/scenecode-worldolympiad"))
    source = runtime / "scenecode/code_object/agent_framework/agents/base_agent.py"
    tree = ast.parse(source.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseAgent")
    method = next(node for node in owner.body if getattr(node, "name", None) == "_call_llm")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), str(source), "exec"), namespace)
    api = SimpleNamespace(chat=AsyncMock())
    context = json.dumps({"objects": ["old task JSON" * 1000], "action_program": [], "success_predicates": []}) if oversized == "whole_task_context" else ""
    instance = SimpleNamespace(llm_client=api, request_context=context)
    kwargs = {"tools": [{"description": "x" * 40000}]} if oversized == "tools" else {}
    with pytest.raises(ValueError):
        asyncio.run(namespace["_call_llm"](instance, [{"role": "user", "content": "short"}], **kwargs))
    api.chat.assert_not_called()
