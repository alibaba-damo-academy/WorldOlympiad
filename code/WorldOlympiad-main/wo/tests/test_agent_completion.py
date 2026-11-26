import hashlib
import json
import sqlite3
import pytest
from wo.agent import run as agent


@pytest.fixture
def complete_run(tmp_path, monkeypatch):
    task = tmp_path / "tasks/a.json";task.parent.mkdir();task.write_text('{"case_id":"a","family":"transfer"}')
    ref = tmp_path / "references/a";(ref / "replay").mkdir(parents=True)
    (ref / "certificate.json").write_text(json.dumps({"success":True,"task":str(task),
        "task_sha256":hashlib.sha256(task.read_bytes()).hexdigest()}))
    (ref / "replay/exterior_left.mp4").write_bytes(b"existing reference evidence")
    (tmp_path / "manifest.json").write_text(json.dumps({"models":["model"],"seeds":[0]}))
    document = {"coverage":{"matrix_complete":True},"episodes":[{"case_id":"a","model":"model","seed":0,"status":"complete"}]}
    (tmp_path / "results.json").write_text(json.dumps(document))
    db = sqlite3.connect(":memory:")
    db.execute("create table jobs(id integer primary key,kind text,args text,status text)")
    db.execute("insert into jobs values(1,'report',?, 'done')",(json.dumps({"run":"probe"}),))
    monkeypatch.setattr(agent.J,"connect",lambda:db)
    monkeypatch.setattr(agent.T,"run_dir",lambda _:tmp_path)
    return tmp_path, db, document


@pytest.mark.parametrize("text", [
    "GOAL COMPLETE is not yet established: the matrix remains incomplete.",
    "**GOAL COMPLETE is not yet established:** the audit remains incomplete.",
    "Not GOAL COMPLETE", "Do not say GOAL COMPLETE", "GOAL COMPLETE\nStill waiting for scores.",
])
def test_negative_or_embedded_completion_text_never_stops_agent(complete_run,text):
    assert not agent.completion_verified(text,"probe")


@pytest.mark.parametrize("text", ["GOAL COMPLETE", "All evidence audited.\n**GOAL COMPLETE**", "GOAL COMPLETE."])
def test_explicit_terminal_marker_with_current_complete_report_can_stop(complete_run,text):
    assert agent.completion_verified(text,"probe")


def test_marker_cannot_stop_while_latest_report_is_pending(complete_run):
    _, db, _ = complete_run
    db.execute("insert into jobs values(2,'report',?, 'queued')",(json.dumps({"run":"probe"}),))
    assert not agent.completion_verified("GOAL COMPLETE","probe")


def test_report_flag_does_not_override_new_missing_manifest_rows(complete_run):
    root, _, _ = complete_run
    (root / "manifest.json").write_text(json.dumps({"models":["model","new_model"],"seeds":[0]}))
    assert not agent.completion_verified("GOAL COMPLETE","probe")


def test_report_flag_does_not_override_changed_reference_task(complete_run):
    root, _, _ = complete_run
    (root / "tasks/a.json").write_text('{"case_id":"a","family":"transfer","revision":2}')
    assert not agent.completion_verified("GOAL COMPLETE","probe")


def test_missing_or_partial_report_keeps_loop_alive(complete_run):
    root, _, document = complete_run
    (root / "results.json").write_text('{')
    assert not agent.completion_verified("GOAL COMPLETE","probe")
    document["coverage"]["matrix_complete"] = False
    (root / "results.json").write_text(json.dumps(document))
    assert not agent.completion_verified("GOAL COMPLETE","probe")


def test_database_or_report_errors_do_not_terminate_controller(complete_run, monkeypatch):
    root, _, _ = complete_run
    (root / "results.json").write_text("null")
    assert not agent.completion_verified("GOAL COMPLETE", "probe")
    def locked():raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(agent.J, "connect", locked)
    assert not agent.completion_verified("GOAL COMPLETE", "probe")


def test_unbounded_rounds_resume_beyond_previous_ceiling_and_explicit_caps_remain():
    from itertools import islice
    assert list(islice(agent.round_numbers(676,0),4))==[676,677,678,679]
    assert list(agent.round_numbers(676,2))==[676,677]
    with pytest.raises(ValueError):agent.round_numbers(0,-1)
