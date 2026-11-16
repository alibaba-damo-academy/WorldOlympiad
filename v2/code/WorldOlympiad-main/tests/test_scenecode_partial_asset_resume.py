import json
from types import SimpleNamespace
from autoresearch.scenecode_partial_asset_resume import seed_plan, seed_part


def fixture(tmp_path, monkeypatch):
    src = tmp_path / "src"
    part = {"name": "body", "shape": {"radius": 1}}
    (src / "code/parts_test").mkdir(parents=True)
    (src / "ObjectPlan.json").write_text(json.dumps({"parts": [part]}))
    (src / "input_text.txt").write_text("same kettle")
    (src / "code/parts_test/body.py").write_text("preserved_body_code")
    (src / "code/parts_test/body_trace.json").write_text(json.dumps({"actions": [{"action_type": "execute_code", "result": "success"}]}))
    monkeypatch.setenv("SCENECODE_CODE_OBJECT_RECOVERY_DIR", str(src))
    return src, part


def test_reuse_exact_input_part_and_successful_execution(tmp_path, monkeypatch):
    src, part = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    seed_plan(out, "same kettle")
    assert (out / "ObjectPlan.json").read_bytes() == (src / "ObjectPlan.json").read_bytes()
    dest = out / "code/parts_test/body.py"
    seed_part(SimpleNamespace(name="body", to_dict=lambda: part), "same kettle", dest)
    assert dest.read_text() == "preserved_body_code"


def test_changed_input_or_part_or_failed_execution_is_not_reused(tmp_path, monkeypatch):
    src, part = fixture(tmp_path, monkeypatch)
    out = tmp_path / "out"
    seed_plan(out, "different kettle")
    assert not (out / "ObjectPlan.json").exists()
    dest = out / "body.py"
    changed = {**part, "shape": {"radius": 2}}
    seed_part(SimpleNamespace(name="body", to_dict=lambda: changed), "same kettle", dest)
    assert not dest.exists()
    (src / "code/parts_test/body_trace.json").write_text(json.dumps({"actions": [{"action_type": "execute_code", "result": "failed"}]}))
    seed_part(SimpleNamespace(name="body", to_dict=lambda: part), "same kettle", dest)
    assert not dest.exists()
