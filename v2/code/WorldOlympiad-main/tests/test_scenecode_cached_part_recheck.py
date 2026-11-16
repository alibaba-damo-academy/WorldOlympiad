"""Exercise the actual staged constructor with measured execution outcomes."""
import asyncio
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

RUNTIME = Path(os.getenv("SCENECODE_RUNTIME_ROOT", "/tmp/SceneCode-main-worldolympiad"))
if not (RUNTIME / "scenecode/code_object/agent_framework").is_dir():
    pytest.skip("requires staged SceneCode runtime", allow_module_level=True)
sys.path.insert(0, str(RUNTIME / "scenecode/code_object"))
source = Path(os.getenv("SCENECODE_TEST_CONSTRUCTOR", str(RUNTIME / "scenecode/code_object/agent_framework/agents/part_constructor.py")))
spec = importlib.util.spec_from_file_location("agent_framework.agents.constructor_recheck_test", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def constructor():
    agent = module.PartConstructorAgent(llm_client=object(), blender_client=object())
    agent._generate_code_with_llm = AsyncMock(side_effect=AssertionError("valid cached source must not be regenerated"))
    agent._generate_code_fix_from_error = AsyncMock(return_value="# repaired cached part\npass")
    return agent


def part():
    data = {"name": "part", "shape": {"dimensions": {"width": .2, "depth": .3, "height": .4}},
            "position": {"x": 0, "y": 0, "z": .2}, "rotation": {}}
    return SimpleNamespace(name="part", instances=[], to_dict=lambda: data)


@pytest.mark.parametrize("first_passes", [True, False])
def test_existing_part_is_reexecuted_and_only_failure_invokes_repair(tmp_path, first_passes):
    agent = constructor()
    outcomes = [{"success": True}] if first_passes else [
        {"success": False, "error": "PART_BOUNDS_CONTRACT: center offset .2"}, {"success": True}]
    agent._execute_code = AsyncMock(side_effect=outcomes)
    result = asyncio.run(agent.construct_part(part(), None, str(tmp_path), text_input="part",
                                             initial_code="# retained geometry\npass"))
    assert result["execution_result"]["success"]
    assert agent._execute_code.await_count == len(outcomes)
    assert agent._generate_code_fix_from_error.await_count == (0 if first_passes else 1)
    for call in agent._execute_code.await_args_list:
        assert "WORLDOLYMPIAD_PART_BOUNDS_CONTRACT" in call.args[0]
    agent._generate_code_with_llm.assert_not_awaited()


def test_run_routes_cached_code_through_constructor(tmp_path, monkeypatch):
    agent = constructor()
    agent._generate_main_file = lambda *_args: "pass"
    agent._execute_code = AsyncMock(return_value={"success": True})
    item = part()
    plan = SimpleNamespace(name="test", description="test", get_parts_sorted_by_priority=lambda: [item])
    parts_dir = tmp_path / module.get_parts_package_name("test")
    parts_dir.mkdir()
    (parts_dir / "part.py").write_text("# old cached source\npass")
    # Main-file assembly is not the subject of this execution-path regression.
    agent._generate_main_code = lambda *_args: "pass"
    asyncio.run(agent.run(plan, None, str(tmp_path), text_input="part"))
    assert agent._execute_code.await_count == 1
    assert "# old cached source" in agent._execute_code.await_args.args[0]


def test_bad_cached_part_never_becomes_success_when_budget_is_exhausted(tmp_path):
    agent = constructor()
    agent.max_retry = 2
    agent._execute_code = AsyncMock(return_value={"success": False, "error": "PART_BOUNDS_CONTRACT: unchanged"})
    with pytest.raises(RuntimeError, match="Failed to create part"):
        asyncio.run(agent.construct_part(part(), None, str(tmp_path), text_input="part", initial_code="pass"))
    assert agent._execute_code.await_count == 2
    assert agent._generate_code_fix_from_error.await_count == 1
