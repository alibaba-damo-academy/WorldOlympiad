import json
import sqlite3

import pytest

from wo.agent import jobs


def queue():
    con = sqlite3.connect(":memory:", isolation_level=None)
    con.row_factory = sqlite3.Row
    con.executescript(jobs.SCHEMA)
    return con


def insert(con, kind, status="queued", run="run", case="case", after=()):
    cur = con.execute("insert into jobs(kind,args,status,after,created,timeout_s) values(?,?,?,?,0,100)",
                      (kind, json.dumps({"run":run,"case":case}), status, json.dumps(list(after))))
    return cur.lastrowid


@pytest.mark.parametrize("status", ["queued", "running"])
def test_admission_finds_reference_inserted_after_scene_check(status):
    con = queue()
    child = insert(con, "scene_check")
    parent = insert(con, "reference", status)
    insert(con, "reference", run="other")
    insert(con, "reference", case="other")
    row = con.execute("select * from jobs where id=?", (child,)).fetchone()
    assert [d["id"] for d in jobs._dependencies(con, row)] == [parent]
    assert json.loads(con.execute("select after from jobs where id=?", (child,)).fetchone()[0]) == [parent]
    con.execute("update jobs set status='failed' where id=?", (parent,))
    row = con.execute("select * from jobs where id=?", (child,)).fetchone()
    assert jobs._dependencies(con, row)[0]["status"] == "failed"


def test_preserves_explicit_dependencies_and_does_not_add_completed_old_attempts():
    con = queue()
    explicit = insert(con, "scene", "done")
    insert(con, "reference", "failed")
    insert(con, "reference", "done")
    parent = insert(con, "reference")
    child = insert(con, "scene_check", after=[explicit,parent])
    row = con.execute("select * from jobs where id=?", (child,)).fetchone()
    assert [d["id"] for d in jobs._dependencies(con, row)] == [explicit,parent]


def test_other_stages_keep_their_declared_dependencies():
    con = queue()
    insert(con, "reference")
    child = insert(con, "episode")
    row = con.execute("select * from jobs where id=?", (child,)).fetchone()
    assert jobs._dependencies(con, row) == []


def test_scene_check_without_active_reference_keeps_completed_reference_path():
    con = queue()
    insert(con, "reference", "done")
    child = insert(con, "scene_check")
    row = con.execute("select * from jobs where id=?", (child,)).fetchone()
    assert jobs._dependencies(con, row) == []


def test_worker_does_not_launch_check_while_later_reference_is_queued(tmp_path, monkeypatch):
    con = queue()
    child = insert(con, "scene_check")
    parent = insert(con, "reference")
    con.execute("update jobs set gpu_gb=1000 where id=?", (parent,))
    monkeypatch.setattr(jobs, "connect", lambda: con)
    monkeypatch.setattr(jobs, "LOGS", tmp_path)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a,**kw: pytest.fail("scene check launched before reference"))
    class EndPoll(Exception):
        pass
    def finish_poll(_):
        raise EndPoll()
    monkeypatch.setattr(jobs.time, "sleep", finish_poll)
    with pytest.raises(EndPoll):
        jobs.worker()
    row = con.execute("select * from jobs where id=?", (child,)).fetchone()
    assert row["status"] == "queued"
    assert json.loads(row["after"]) == [parent]


@pytest.mark.parametrize("kind,expected", [("scene", True), ("episode", False)])
def test_finished_scene_releases_owned_process_group_only(tmp_path, monkeypatch, kind, expected):
    con = queue()
    jid = insert(con, kind, "running")
    con.execute("update jobs set pid=4321,started=0 where id=?", (jid,))
    (tmp_path / f"{jid:06d}.rc").write_text("0")
    monkeypatch.setattr(jobs, "connect", lambda: con)
    monkeypatch.setattr(jobs, "LOGS", tmp_path)
    monkeypatch.setattr(jobs, "_alive", lambda pid: False)
    signals = []
    monkeypatch.setattr(jobs.os, "killpg", lambda pid,sig: signals.append((pid,sig)))
    class EndPoll(Exception):
        pass
    def finish_poll(_):
        raise EndPoll()
    monkeypatch.setattr(jobs.time, "sleep", finish_poll)
    with pytest.raises(EndPoll):
        jobs.worker()
    assert signals == ([(4321, jobs.signal.SIGTERM)] if expected else [])
