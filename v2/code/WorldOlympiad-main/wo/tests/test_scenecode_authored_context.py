import json
import os
import pytest
from autoresearch.scenecode_object_requirements import object_requirements, prepare_request_context


def test_original_requirements_survive_a_shortened_asset_description(monkeypatch, tmp_path):
    original = "an oven with a raised U-pull and no tray"
    geometry = {"overall_dimensions_m": [.35, .25, .23], "handle_crossbar_length_m": .12, "handle_clearance_m": .04, "initial_joint_position": .1, "cavity": {"clear_inner_opening_m": [.22, .16], "depth_m": .18}}
    task = {"case_id": "context_fixture", "objects": [{"object_id": "oven", "visual_description": original, "geometry_constraints": geometry}],
            "action_program": [{"operation": "articulate"}], "initial_state": "FULL_TASK_INITIAL_STATE", "fps": 15}
    authored = json.dumps(task)
    monkeypatch.setenv("SCENECODE_AUTHORED_TASK_REQUIREMENTS", authored)
    monkeypatch.delenv("SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST", raising=False)
    context = object_requirements("oven with simple handle", [.35, .25, .23], asset_name="oven")
    assert original in context
    projected = json.loads(next(line.split("=", 1)[1] for line in context.splitlines() if line.startswith("geometry_constraints=")))
    assert projected == geometry
    assert "handle_clearance_m" in context and "action_program" not in context
    assert "FULL_TASK_INITIAL_STATE" not in context and "fps" not in context
    assert "oven with simple handle" not in context
    assert context.count(original) == 1
    assert context.splitlines()[-1] == "One model unit is 0.35 m; divide metre lengths by 0.35."
    assert context.count("One model unit") == 1
    assert os.environ["SCENECODE_AUTHORED_TASK_REQUIREMENTS"] == authored
    prepare_request_context(tmp_path, context)
    (tmp_path / "ObjectPlan.json").write_text("preserved old plan")
    geometry["initial_joint_position"] = 0.0
    updated_authored = json.dumps(task)
    monkeypatch.setenv("SCENECODE_AUTHORED_TASK_REQUIREMENTS", updated_authored)
    updated = object_requirements("oven with simple handle", [.35, .25, .23], asset_name="oven")
    assert updated != context
    updated_geometry = json.loads(next(line.split("=", 1)[1] for line in updated.splitlines() if line.startswith("geometry_constraints=")))
    assert updated_geometry == geometry and updated_geometry["initial_joint_position"] == 0.0
    assert os.environ["SCENECODE_AUTHORED_TASK_REQUIREMENTS"] == updated_authored
    prepare_request_context(tmp_path, updated)
    assert (tmp_path / "request_context.txt").read_text() == updated
    assert os.environ["SCENECODE_AUTHORED_TASK_REQUIREMENTS"] == updated_authored
    archived = list((tmp_path / ".context_history").glob("*/ObjectPlan.json"))
    assert len(archived) == 1 and archived[0].read_text() == "preserved old plan"
    assert (archived[0].parent / "request_context.txt").read_text() == context
    assert not (tmp_path / "ObjectPlan.json").exists()
    monkeypatch.delenv("SCENECODE_AUTHORED_TASK_REQUIREMENTS")
    assert original not in object_requirements("cup", [.1, .1, .1])


def test_registered_scene_launcher_forwards_exact_authored_task(monkeypatch, tmp_path):
    from wo.spec import TaskSpec, TaskObject, Step, Predicate
    from wo.tasks import scene
    spec = TaskSpec('context_probe', 'pick_place', 'Place the cup on the mat.',
        (TaskObject('table', 'table', 'support', 'A stone worktop.'),
         TaskObject('cup', 'cup', 'manipuland', 'Loop handle with 0.04 m finger clearance.'),
         TaskObject('mat', 'mat', 'destination', 'A dry cloth mat.')),
        (Step('transport', 'cup', Predicate('on', 'cup', 'mat'), target='mat'),))
    source = tmp_path / 'task.json'
    source.write_text(json.dumps(spec.to_dict()))
    before = source.read_bytes()
    monkeypatch.setattr(scene, 'OUT', tmp_path / 'outputs')
    monkeypatch.setattr(scene.subprocess, 'run', lambda *a, **k: None)
    monkeypatch.setattr(scene, 'archive_stale_task_recoveries', lambda *a: None)
    monkeypatch.setattr(scene, 'archive_transient_incomplete_recoveries', lambda *a: None)
    captured = {}
    class StopBeforeGeneration(Exception): pass
    def intercept(command, **kwargs):
        captured.update(kwargs['env'])
        raise StopBeforeGeneration()
    monkeypatch.setattr(scene, 'run_with_service_retries', intercept)
    with pytest.raises(StopBeforeGeneration):
        scene.generate(source)
    task_path = tmp_path / 'outputs/robotics_task_batches/wo_context_probe/tasks/context_probe.json'
    assert captured['SCENECODE_AUTHORED_TASK_REQUIREMENTS'] == task_path.read_text()
    assert json.loads(captured['SCENECODE_AUTHORED_TASK_REQUIREMENTS']) == scene.legacy_task(spec)
    assert source.read_bytes() == before
    monkeypatch.setenv("SCENECODE_AUTHORED_TASK_REQUIREMENTS", captured["SCENECODE_AUTHORED_TASK_REQUIREMENTS"])
    context = object_requirements("a simple cup", asset_name="cup")
    assert "Loop handle with 0.04 m finger clearance." in context
    assert "A stone worktop." not in context and "A dry cloth mat." not in context
    assert os.environ["SCENECODE_AUTHORED_TASK_REQUIREMENTS"] == captured["SCENECODE_AUTHORED_TASK_REQUIREMENTS"]
    assert source.read_bytes() == before
