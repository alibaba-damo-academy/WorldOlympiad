from wo.spec import TaskSpec
from wo.tasks.scene import legacy_task


def task(kind, peg_constraints=None):
    return TaskSpec.from_dict({
        "schema_version": "wo_task_spec.v1", "case_id": "mating", "family": "insertion",
        "instruction": "Insert peg into socket", "initial_state": "Peg beside empty socket.",
        "objects": [
            {"object_id": "peg", "category": "peg", "role": "manipuland",
             "description": "red peg, 5 cm diameter, 9 cm tall",
             "geometry_constraints": peg_constraints or {}}, 
            {"object_id": "socket", "category": "socket", "role": "destination",
             "description": "blue socket 8 cm x 8 cm x 5 cm, 5.5 cm opening"},
            {"object_id": "table", "category": "table", "role": "support",
             "description": "table 80 cm x 60 cm x 3 cm"}],
        "steps": [{"primitive": "transport", "subject": "peg", "target": "socket",
                   "goal": {"kind": kind, "subject": "peg", "target": "socket"}}]})


def test_insertion_requires_collision_cavity_without_changing_dimensions():
    spec = task("inserted")
    result = legacy_task(spec)
    brief = result["scene_brief"]
    assert brief.startswith(spec.initial_state)
    assert "peg into socket" in brief
    assert "exported convex collision decomposition" in brief
    assert "collision margins" in brief
    assert "Do not shrink" in brief
    assert "enlarge the declared opening" in brief
    assert result["instruction"] == spec.instruction
    assert [o["visual_description"] for o in result["objects"]] == [o.description for o in spec.objects]
    assert result["success_predicates"] == [{"kind": "inserted", "subject": "peg", "object": "socket"}]


def test_noninsertion_scene_brief_includes_authored_geometry_contract():
    spec = task("on", {
        "overall_dimensions_m": [0.05, 0.05, 0.09],
        "approach_clearance_m": 0.02,
    })
    brief = legacy_task(spec)["scene_brief"]
    assert brief.startswith(spec.initial_state)
    assert "peg authoritative geometry contract" in brief
    assert "overall_dimensions_m" in brief
    assert "0.09" in brief
    assert "do not substitute a guessed or normalized size" in brief


def test_insertion_requirements_reach_native_scenecode_prompt():
    from autoresearch.robotics_scene_prompt_compiler import compile_task_prompt
    from autoresearch.robotics_contracts import TaskInstructionSpec
    from autoresearch.scene_prompting import compile_scene_prompt, assess_scene_prompt

    payload = legacy_task(task("inserted"))
    typed = TaskInstructionSpec.model_validate(payload)
    prompt_spec = compile_task_prompt(typed)
    prompt = compile_scene_prompt(prompt_spec)
    assert payload["scene_brief"].rstrip(".") in prompt
    assert "exported convex collision decomposition" in prompt
    assert "5.5 cm opening" in prompt
    assert assess_scene_prompt(prompt_spec, prompt).checks["scene_brief_preserved"]
    stripped = prompt.replace(payload["scene_brief"].rstrip(".") + ".", "")
    assert not assess_scene_prompt(prompt_spec, stripped).checks["scene_brief_preserved"]


def test_explicit_geometry_contract_roundtrip_and_constructor_ratios():
    from autoresearch.robotics_scene_prompt_compiler import _geometry_constraint_attributes
    payload = task("inserted").to_dict()
    constraints = {"overall_dimensions_m": [.08, .08, .05],
                   "feature_dimensions_m": {"bore_diameter": .055, "bore_depth": .04}}
    payload['objects'][1]['geometry_constraints'] = constraints
    spec = TaskSpec.from_dict(payload)
    assert TaskSpec.from_dict(spec.to_dict()).objects[1].geometry_constraints == constraints
    assert legacy_task(spec)['objects'][1]['geometry_constraints'] == constraints
    clauses = ' '.join(_geometry_constraint_attributes(constraints))
    assert '1/1/0.625' in clauses
    assert 'bore diameter in the same normalized constructor units: 0.6875' in clauses
    assert 'bore depth in the same normalized constructor units: 0.5' in clauses
