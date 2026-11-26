import json

from wo.expert import regress


def test_compare_reports_fixed_broken_and_new():
    a = {"x": {"success": False}, "y": {"success": True}, "z": {"success": True}}
    b = {"x": {"success": True}, "y": {"success": False}, "z": {"success": True}, "w": {"success": True}}
    assert regress.compare(a, b) == {"fixed": ["x"], "broken": ["y"], "new": ["w"]}


def test_summarize_reads_the_failing_step_and_survives_a_crash(tmp_path):
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"success": True, "steps": []}))
    assert regress.summarize(ok) == (True, "")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"success": False, "steps": [
        {"stage": "planning", "detail": "no antipodal grasp"}]}))
    assert regress.summarize(bad) == (False, "planning: no antipodal grasp")
    assert regress.summarize(tmp_path / "missing.json") == (False, "no certificate (crash)")


def test_scene_ready_cases_needs_a_generation_with_a_certificate(tmp_path):
    (tmp_path / "tasks").mkdir()
    gen = tmp_path / "gen"
    gen.mkdir()
    (gen / "task_inventory_certificate.json").write_text("{}")
    (tmp_path / "tasks" / "a.json").write_text(json.dumps({"case_id": "a", "generation": str(gen)}))
    (tmp_path / "tasks" / "b.json").write_text(json.dumps({"case_id": "b", "generation": str(tmp_path / "none")}))
    (tmp_path / "tasks" / "c.json").write_text(json.dumps({"case_id": "c"}))
    assert regress.scene_ready_cases(tmp_path) == ["a"]


def test_run_one_logs_crash_and_rejects_stale_success(tmp_path, monkeypatch):
    from types import SimpleNamespace
    out = tmp_path / "tag" / "case"
    out.mkdir(parents=True)
    (out / "certificate.json").write_text('{"success": true}')

    def crash(*args, **kwargs):
        kwargs["stdout"].write("missing runtime dependency\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(regress.subprocess, "run", crash)
    case, ok, why, _ = regress.run_one("case", out.parent, tmp_path, tmp_path)
    assert case == "case" and not ok
    assert "exited 1" in why
    assert "missing runtime dependency" in (out / "run.log").read_text()
