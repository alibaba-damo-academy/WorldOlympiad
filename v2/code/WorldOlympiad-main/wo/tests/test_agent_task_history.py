import json
import pytest
from wo.agent import tools as T, stages


def task():
    return {"schema_version": "wo_task_spec.v1", "case_id": "history_test", "family": "pick_place",
            "instruction": "Put the apple on the mat.",
            "objects": [{"object_id": "apple", "role": "manipuland", "category": "fruit", "description": "red apple"},
                        {"object_id": "mat", "role": "destination", "category": "mat", "description": "cork mat"},
                        {"object_id": "table", "role": "support", "category": "table", "description": "wood table"}],
            "steps": [{"primitive": "transport", "subject": "apple", "target": "mat",
                       "goal": {"kind": "on_region", "subject": "apple", "target": "mat"}}]}


def test_author_revision_preserves_exact_prior_task_and_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "RUNS", tmp_path)
    monkeypatch.setattr(T, "load_profile", lambda: None)
    monkeypatch.setattr(T, "registered_models", lambda: ["fastwam", "longwam"])
    original = task()
    original["generation"] = "/assets/prior_generation"
    T.author_task("dev", json.dumps(original))
    path = tmp_path / "dev/tasks/history_test.json"
    previous_bytes = path.read_bytes()
    revised = task()
    revised["objects"][0]["description"] = "red apple with a short stem"
    T.author_task("dev", json.dumps(revised))
    snapshots = list((tmp_path / "dev/task_history/history_test").glob("*.json"))
    assert len(snapshots) == 1 and snapshots[0].read_bytes() == previous_bytes
    current = json.loads(path.read_text())
    assert current["objects"][0]["description"] == revised["objects"][0]["description"]
    assert current["generation"] == original["generation"]
    T.author_task("dev", json.dumps(revised))
    assert len(list(snapshots[0].parent.glob("*.json"))) == 1
    bad = task()
    bad["steps"][0]["primitive"] = "unknown_primitive"
    with pytest.raises(ValueError):
        T.author_task("dev", json.dumps(bad))
    assert json.loads(path.read_text()) == current
    assert len(list(snapshots[0].parent.glob("*.json"))) == 1


def test_run_declares_models_once_and_later_registry_changes_do_not_rewrite_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "RUNS", tmp_path)
    monkeypatch.setattr(T, "load_profile", lambda: None)
    models = ["fastwam"]
    monkeypatch.setattr(T, "registered_models", lambda: list(models))
    T.author_task("scope", json.dumps(task()))
    manifest = tmp_path / "scope/manifest.json"
    assert json.loads(manifest.read_text()) == {
        "schema_version": "wo_run_manifest.v1", "cases": ["history_test"],
        "models": ["fastwam"], "seeds": [0]}
    original = manifest.read_bytes()
    models.append("longwam")
    revised = task()
    revised["objects"][0]["description"] = "a small ripe apple"
    T.author_task("scope", json.dumps(revised))
    assert manifest.read_bytes() == original


def test_unscoped_default_manifest_uses_one_configured_model(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "RUNS", tmp_path)
    models = ["cosmos3", "fastwam", "gr00t", "lingbot_va", "longwam"]
    monkeypatch.setattr(T, "registered_models", lambda: list(models))
    monkeypatch.setattr(T, "load_profile", lambda: {"tested_model": {"adapter": "lingbot_va"}})
    legacy = tmp_path / "single/manifest.json"
    legacy.parent.mkdir()
    legacy.write_text(json.dumps({
        "schema_version": "wo_run_manifest.v1", "models": models, "seeds": [0]}))
    T.author_task("single", json.dumps(task()))
    manifest = json.loads((tmp_path / "single/manifest.json").read_text())
    assert manifest == {
        "schema_version": "wo_run_manifest.v1", "cases": ["history_test"],
        "models": ["lingbot_va"], "seeds": [0]}
