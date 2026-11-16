import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from jinja2 import Template

RUNTIME = Path(os.environ.get("SCENECODE_RUNTIME_ROOT", "/tmp/SceneCode-main-worldolympiad"))
if not (RUNTIME / "scenecode/agent_utils/base_stateful_agent.py").is_file():
    pytest.skip("requires staged SceneCode runtime", allow_module_level=True)

from agents import Agent, ModelSettings
from agents.agent_output import AgentOutputSchema
from agents.exceptions import ModelBehaviorError
import sys
sys.path.insert(0, str(RUNTIME))
from scenecode.agent_utils.scoring import FloorPlanCritiqueWithScores


@pytest.fixture
def critic_and_prompt():
    path = RUNTIME / "scenecode/agent_utils/base_stateful_agent.py"
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseStatefulAgent")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_create_critic_agent")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"Agent": Agent, "AgentOutputSchema": AgentOutputSchema, "json": json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), str(path), "exec"), namespace)
    prompt = yaml.safe_load((RUNTIME / "scenecode/prompts/data/floor_plan_agent/critic_agent.yaml").read_text())["prompt"]
    rendered = Template(prompt).render(mode="room", house_prompt="One real kitchen with the authored sink counter.")
    settings_calls = []
    def settings(**kwargs):
        settings_calls.append(kwargs)
        return ModelSettings(tool_choice=kwargs["tool_choice"])
    owner = SimpleNamespace(cfg=SimpleNamespace(agents=SimpleNamespace(critic_agent=SimpleNamespace(name="floor_critic")), openai=SimpleNamespace(model="gpt-5.6-sol")), prompt_registry=SimpleNamespace(get_prompt=lambda **_: rendered), _get_model_settings=settings)
    agent = namespace["_create_critic_agent"](owner, tools=[], prompt_enum="floor", output_type=FloorPlanCritiqueWithScores, mode="room", house_prompt="kitchen")
    return agent, rendered, settings_calls


def test_deployed_critic_exposes_exact_sdk_schema_and_keeps_evaluation(critic_and_prompt):
    agent, rendered, settings = critic_and_prompt
    assert agent.instructions.startswith(rendered)
    assert agent.output_type is FloorPlanCritiqueWithScores
    assert settings == [{"settings_key": "critic", "tool_choice": "observe_scene"}]
    emitted = json.loads(agent.instructions.rsplit("JSON schema:\n", 1)[1])
    assert emitted == AgentOutputSchema(agent.output_type).json_schema()
    assert "no Markdown fences or prose outside" in agent.instructions
    assert "Keep all requested narrative" in agent.instructions


def test_actual_sdk_accepts_only_its_structured_wrapper(critic_and_prompt):
    agent, _, _ = critic_and_prompt
    schema = AgentOutputSchema(agent.output_type)
    response = {"critique": "## Evaluation Checklist\n- Validate the observed room.\n## Summary\nKeep the useful narrative.", **{name: {"name": name, "grade": 8, "comment": "Checked against original criteria."} for name in ("room_proportions", "spatial_flow", "natural_lighting", "material_consistency", "prompt_following")}}
    with pytest.raises(ModelBehaviorError):
        schema.validate_json(response["critique"])
    with pytest.raises(ModelBehaviorError):
        schema.validate_json(json.dumps(response))
    result = schema.validate_json(json.dumps({"response": response}))
    assert isinstance(result, FloorPlanCritiqueWithScores)
    assert result.critique == response["critique"]
    assert [score.grade for score in result.get_scores()] == [8] * 5
