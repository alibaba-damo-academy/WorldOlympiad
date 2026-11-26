import json
import time
from pathlib import Path

import pytest

from wo.agent import tools
from wo.eval import reference_binding


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(tools.J, "DB", tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(tools, "run_dir", lambda run: tmp_path / run)
    return tmp_path


def ready_reference(sandbox):
    ref = sandbox / "run/references/case"
    ref.mkdir(parents=True)
    (ref / "certificate.json").write_text('{"success":true}')
    for name in ["trajectory.json", "scene_inputs.json", "replay_plan.json", "replay/exterior_left.mp4"]:
        p = ref / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"existing proof")
    return ref


def test_complete_reference_is_verified_and_reused_without_queue_write(sandbox, monkeypatch):
    ref = ready_reference(sandbox)
    calls = []
    monkeypatch.setattr(reference_binding, "verify_task", lambda r, t: calls.append((r, t)))
    monkeypatch.setattr(reference_binding, "capture", lambda r: calls.append(r))
    monkeypatch.setattr(tools.J, "submit", lambda *a, **kw: pytest.fail("must not queue a ready reference"))
    monkeypatch.setattr(tools.J, "connect", lambda: pytest.fail("ready reference should not mutate the queue"))
    before = {str(p): p.read_bytes() for p in ref.rglob("*") if p.is_file()}
    result = tools.submit("reference", '{"run":"run","case":"case"}')
    assert result.startswith("reused complete reference")
    assert calls == [(ref, sandbox / "run/tasks/case.json"), ref]
    assert {str(p): p.read_bytes() for p in ref.rglob("*") if p.is_file()} == before


def test_complete_reference_with_changed_task_is_not_silently_reused(sandbox, monkeypatch):
    ready_reference(sandbox)
    def reject(*args):
        raise ValueError("different task version")
    monkeypatch.setattr(reference_binding, "verify_task", reject)
    monkeypatch.setattr(tools.J, "submit", lambda *a, **kw: pytest.fail("must not overwrite old success"))
    with pytest.raises(ValueError, match="different task version"):
        tools.submit("reference", '{"run":"run","case":"case"}')


@pytest.mark.parametrize("status", ["queued", "running"])
def test_identical_active_job_is_reused(sandbox, monkeypatch, status):
    args = {"run": "run", "case": "case"}
    con = tools.J.connect()
    cursor = con.execute("insert into jobs(kind,args,status,timeout_s,created) values(?,?,?,?,?)",
                         ("scene", json.dumps(args), status, 100, time.time()))
    jid = cursor.lastrowid
    monkeypatch.setattr(tools.J, "submit", lambda *a, **kw: pytest.fail("duplicate active job"))
    assert tools.submit("scene", '{"case":"case","run":"run"}') == f"reused {status} job #{jid} (scene)"
    assert con.execute("select count(*) from jobs").fetchone()[0] == 1


def test_failed_reference_still_queues_a_real_retry(sandbox, monkeypatch):
    ref = sandbox / "run/references/case"; ref.mkdir(parents=True)
    (ref / "certificate.json").write_text('{"success":false}')
    calls = []
    def queue(*a, **kw):
        calls.append((a, kw)); return 77
    monkeypatch.setattr(tools.J, "submit", queue)
    assert tools.submit("reference", '{"run":"run","case":"case"}', "9,10", 2) == "queued job #77 (reference) priority 2"
    assert calls[0][1]["after"] == [9, 10]


@pytest.mark.parametrize("status", ["queued", "running"])
def test_scene_check_waits_for_same_case_reference(sandbox, monkeypatch, status):
    con = tools.J.connect()
    cursor = con.execute("insert into jobs(kind,args,status,timeout_s,created) values(?,?,?,?,?)",
        ("reference", json.dumps({"run":"run", "case":"case", "layout":"droid"}), status, 100, time.time()))
    parent = cursor.lastrowid
    con.execute("insert into jobs(kind,args,status,timeout_s,created) values(?,?,?,?,?)",
        ("reference", json.dumps({"run":"other", "case":"case"}), status, 100, time.time()))
    calls = []
    def queue(*a, **kw):
        calls.append(kw); return 99
    monkeypatch.setattr(tools.J, "submit", queue)
    tools.submit("scene_check", '{"run":"run","case":"case"}', "9")
    assert calls[0]["after"] == sorted({parent, 9})
