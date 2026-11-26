from __future__ import annotations

import json

import pytest

from autoresearch.scene_prompting import (
    DEFAULT_PROFILES_PATH,
    ScenePromptSpec,
    architect_request,
    assess_scene_prompt,
    compile_scene_prompt,
    load_profile,
)


@pytest.mark.parametrize(
    "profile_id",
    [
        "robotics_tabletop_physics_v1",
        "robotics_articulated_storage_v1",
        "world_living_room_dynamics_v1",
    ],
)
def test_profiles_compile_and_pass(profile_id: str) -> None:
    spec = load_profile(DEFAULT_PROFILES_PATH, profile_id)
    prompt = compile_scene_prompt(spec)
    report = assess_scene_prompt(spec, prompt)
    assert report.passed, report.problems
    assert len(prompt) > 700
    assert "physics showcase" not in prompt.lower()
    assert "avoid no " not in prompt.lower()
    assert "place 3 " not in prompt.lower()


def test_robotics_profile_must_be_unsolved() -> None:
    raw = load_profile(DEFAULT_PROFILES_PATH, "robotics_tabletop_physics_v1").model_dump()
    raw["task_setup"] = "The task is already finished"
    with pytest.raises(ValueError, match="unsolved"):
        ScenePromptSpec.model_validate(raw)


def test_robotics_profile_accepts_plain_unsolved_wording() -> None:
    raw = load_profile(DEFAULT_PROFILES_PATH, "robotics_tabletop_physics_v1").model_dump()
    raw["task_setup"] = "The objects start apart and the task is unsolved"
    assert ScenePromptSpec.model_validate(raw).task_setup.endswith("unsolved")


def test_architect_request_contains_structured_official_patterns() -> None:
    messages = architect_request("a robot opens a drawer", DEFAULT_PROFILES_PATH)
    payload = json.loads(messages[1]["content"])
    assert payload["brief"] == "a robot opens a drawer"
    assert len(payload["reference_patterns"]) == 2
    assert "cabinet" in payload["reference_patterns"][0]["brief"].lower()
    assert "not a toy showcase" in messages[0]["content"]


def test_robotics_prompt_retains_declared_definite_article_without_relaxing_coverage():
    raw=load_profile(DEFAULT_PROFILES_PATH,"robotics_tabletop_physics_v1").model_dump()
    raw['objects'][0]['name']='the broad matte oak table'
    spec=ScenePromptSpec.model_validate(raw)
    prompt=compile_scene_prompt(spec)
    assert assess_scene_prompt(spec,prompt).checks['declared_object_coverage']
    altered=prompt.replace('the broad matte oak table','a narrow plastic table')
    assert not assess_scene_prompt(spec,altered).checks['declared_object_coverage']
