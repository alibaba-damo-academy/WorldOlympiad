import json
from autoresearch.io import write_json as atomic_json
from autoresearch.research_failures import capture_compiled_gap, capture_failure
from autoresearch.robotics_task_compiler import compile_task
from autoresearch.robotics_contracts import TaskInstructionSpec


def test_real_failure_is_retained_once_for_shared_research(tmp_path):
    report = tmp_path / "robotics_scenecode_failures" / "case.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"case_id":"case","batch_id":"batch",
        "classification":"native_capability","failed_stage":"capability_preflight",
        "reason":"missing hang"}))
    item = capture_failure(tmp_path, report)
    assert capture_failure(tmp_path, report) == item
    report.write_text("{}")
    stored = json.loads(item.read_text())
    assert json.loads(open(stored["failure_evidence"]).read())["reason"] == "missing hang"


def test_asset_failure_enters_the_same_shared_research_intake(tmp_path):
    report = tmp_path / "robotics_scenecode_failures" / "batch" / "case.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"case_id": "case", "batch_id": "batch",
        "classification": "asset_geometry", "failed_stage": "task_asset_inventory",
        "reason": "missing SDF"}))
    assert capture_failure(tmp_path, report).is_file()
    assert list((tmp_path / "robotics_rsi/queue").glob("*.json")) == []
    assert len(list((tmp_path / "robotics_rsi/traces").glob("*.json"))) == 1


def test_compiled_gap_is_durable_and_task_bound(tmp_path):
    task = TaskInstructionSpec.model_validate({
        "case_id": "case", "instruction": "Put the object in the tray.",
        "scene_brief": "An object and a tray on a table.",
        "initial_state": "The object is outside the tray.",
        "objects": [
            {"object_id": "item", "category": "block", "role": "manipuland", "visual_description": "block"},
            {"object_id": "tray", "category": "tray", "role": "destination", "visual_description": "tray"},
            {"object_id": "table", "category": "table", "role": "support", "visual_description": "table"},
        ],
        "action_program": [{"operation": "new_operator", "subjects": ["item"], "target": "tray"}],
        "success_predicates": [{"kind": "in_container", "subject": "item", "object": "tray"}],
        "requested_wams": ["cosmos3_nano_droid"],
    })
    root = tmp_path / "outputs"
    task_path = root / "robotics_task_batches/batch/tasks/case.json"
    plan_path = root / "robotics_compilation_plans/batch/case.json"
    atomic_json(task_path, task.model_dump(mode="json"))
    atomic_json(plan_path, compile_task(task))
    item = capture_compiled_gap(tmp_path, "batch", "case")
    assert item is not None
    assert capture_compiled_gap(tmp_path, "batch", "case") == item
    stored = json.loads(item.read_text())
    gap = json.loads(open(stored["failure_evidence"]).read())
    assert gap["unsupported_operators"] == ["new_operator"]
    assert json.loads(open(gap["compilation_plan"]).read())["case_id"] == "case"


def test_old_plan_cannot_advertise_unverified_pour_as_executable(tmp_path):
    task = TaskInstructionSpec.model_validate({
        "case_id": "pour_case", "instruction": "Move the contents into the bin.",
        "scene_brief": "A source and bin on a table.",
        "initial_state": "The contents are in the source.",
        "objects": [
            {"object_id": "source", "category": "container", "role": "manipuland", "visual_description": "source"},
            {"object_id": "bin", "category": "bin", "role": "destination", "visual_description": "bin"},
            {"object_id": "table", "category": "table", "role": "support", "visual_description": "table"},
        ],
        "action_program": [
            {"operation": "pick", "subjects": ["source"]},
            {"operation": "pour", "subjects": ["source"], "target": "bin"},
        ],
        "success_predicates": [{"kind": "in_container", "subject": "source", "object": "bin"}],
        "requested_wams": ["cosmos3_nano_droid"],
    })
    root = tmp_path / "outputs"
    atomic_json(root / "robotics_task_batches/batch/tasks/pour_case.json", task.model_dump(mode="json"))
    plan = compile_task(task)
    plan["task_sha256"] = "legacy-compiler-digest"
    for step in plan["controller_program"]:
        step.pop("capability_status", None)
    for predicate in plan["success_program"]:
        predicate.pop("evaluator_status", None)
    plan["native_capabilities"]["cosmos3_nano_droid"].update(
        status="native_supported", unsupported_operators=[])
    plan["execution_capability_preflight"].update(passed=True)
    atomic_json(root / "robotics_compilation_plans/batch/pour_case.json", plan)
    item = capture_compiled_gap(tmp_path, "batch", "pour_case")
    assert item is not None
    evidence = json.loads(item.read_text())["failure_evidence"]
    gap = json.loads(open(evidence).read())
    assert gap["unsupported_operators"] == ["pour"]
    assert json.loads(open(gap["compilation_plan"]).read())["task_sha256"] == task.content_hash()
