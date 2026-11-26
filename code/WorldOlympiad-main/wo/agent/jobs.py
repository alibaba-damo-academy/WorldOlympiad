"""Durable job queue + worker for pipeline stages.

One SQLite table; one long-running worker (``python -m wo.agent.jobs worker``)
admits jobs while their declared GPU memory fits the budget, runs each as a
subprocess with a timeout, and records rc/log/result.  Jobs may depend on other
jobs (``after``); a failed dependency fails the dependent job without running it.
Only registered stage kinds can be queued (see stages.py) - never free shell.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from ..scene import WORKSPACE
from .config import apply_environment, job_defaults, load_private_environment, load_profile, resource_limits

DB = WORKSPACE / "outputs/wo_agent/jobs.sqlite3"
LOGS = WORKSPACE / "outputs/wo_agent/job_logs"
_PROFILE = load_profile()
# Importing the agent must not probe hardware; actual resource admission is done
# by the user-loop preflight with its explicit GPU allowlist.
_RESOURCES = resource_limits(_PROFILE, inventory=[])
GPU_BUDGET_GB = _RESOURCES["budget"]
MAX_PARALLEL = 1
# long LLM-bound stages must not occupy every slot and starve GPU stages
MAX_PER_KIND = {"scene": 4, "reference": 6, "expert_check": 4}  # >4 parallel SceneCode runs time out local servers

SCHEMA = """
create table if not exists jobs (
  id integer primary key autoincrement,
  kind text not null, args text not null, after text not null default '[]',
  gpu_gb real not null default 0, timeout_s integer not null,
  status text not null default 'queued',  -- queued|running|done|failed|cancelled
  created real not null, started real, finished real,
  pid integer, rc integer, log text, result text, error text,
  submitted_by text,
  priority integer not null default 0
);
"""


def connect() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("pragma journal_mode=wal")
    con.executescript(SCHEMA)
    # existing databases predate the priority column
    if "priority" not in {r[1] for r in con.execute("pragma table_info(jobs)")}:
        con.execute("alter table jobs add column priority integer not null default 0")
    return con


def submit(kind: str, args: dict, *, after: list[int] | None = None, submitted_by: str = "",
           priority: int = 0) -> int:
    """Queue a stage.  Higher priority is admitted first, so a diagnostic or a
    positive control is not stuck behind a long serial chain of bulk work."""
    # reload: a long-running agent process must see stages/models promoted after it started
    import importlib

    from . import stages as stages_mod

    STAGES = importlib.reload(stages_mod).STAGES

    args = job_defaults(kind, args)
    if kind not in STAGES:
        raise ValueError(f"unknown stage kind {kind!r}; known: {sorted(STAGES)}")
    stage = STAGES[kind]
    stage.validate(args)
    con = connect()
    cur = con.execute(
        "insert into jobs(kind,args,after,gpu_gb,timeout_s,created,submitted_by,priority) "
        "values(?,?,?,?,?,?,?,?)",
        (kind, json.dumps(args, sort_keys=True), json.dumps(after or []), stage.gpu_gb,
         stage.timeout_s, time.time(), submitted_by, int(priority)))
    return int(cur.lastrowid)


def get(job_id: int) -> dict:
    row = connect().execute("select * from jobs where id=?", (job_id,)).fetchone()
    if row is None:
        raise KeyError(job_id)
    return dict(row)


def listing(status: str | None = None, limit: int = 50) -> list[dict]:
    con = connect()
    if status:
        rows = con.execute("select * from jobs where status=? order by id desc limit ?", (status, limit))
    else:
        rows = con.execute("select * from jobs order by id desc limit ?", (limit,))
    return [dict(r) for r in rows]


def cancel(job_id: int) -> None:
    con = connect()
    row = con.execute("select status,pid from jobs where id=?", (job_id,)).fetchone()
    if row and row["status"] == "running" and row["pid"]:
        try:
            os.killpg(row["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    con.execute("update jobs set status='cancelled', finished=? where id=? and status in ('queued','running')",
                (time.time(), job_id))


# ------------------------------------------------------------------ worker
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # zombies count as dead
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def _rc_file(job_id: int) -> Path:
    return LOGS / f"{job_id:06d}.rc"


def _dependencies(con: sqlite3.Connection, job) -> list[dict]:
    ids = set(json.loads(job["after"]))
    if job["kind"] == "scene_check":
        args = json.loads(job["args"])
        for row in con.execute(
                "select id,args from jobs where kind='reference' and status in ('queued','running')"):
            parent = json.loads(row["args"])
            if all(parent.get(key) == args.get(key) for key in ("run", "case")):
                ids.add(row["id"])
        # Concurrent tool submissions can miss a peer reference before its insert.
        # Persist admission-time dependencies so a later failed reference still blocks.
        con.execute('update jobs set after=? where id=?', (json.dumps(sorted(ids)), job["id"]))
    return [dict(con.execute('select * from jobs where id=?', (jid,)).fetchone()) for jid in sorted(ids)]


def worker(poll_s: float = 3.0) -> None:
    """Restart-safe: jobs are tracked by pid + an rc file written by a bash wrapper,
    so a restarted worker adopts still-running jobs.  The stage registry is reloaded
    every cycle, so promoted stage/adapter changes apply without a restart."""
    import importlib

    _prepare_worker_environment()

    from . import stages as stages_mod

    LOGS.mkdir(parents=True, exist_ok=True)
    con = connect()
    while True:
        stages_mod = importlib.reload(stages_mod)
        STAGES = stages_mod.STAGES
        running = [dict(r) for r in con.execute("select * from jobs where status='running'")]
        for job in running:
            rc_path = _rc_file(job["id"])
            alive = bool(job["pid"]) and _alive(job["pid"])
            timed_out = alive and time.time() - job["started"] > job["timeout_s"]
            if timed_out:
                try:
                    os.killpg(job["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
                alive = False
            if alive:
                continue
            if job["kind"] == "scene" and job["pid"]:
                # Scene-local render/CoACD servers share the completed job's group.
                try:
                    os.killpg(job["pid"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
            rc = int(rc_path.read_text().strip()) if rc_path.is_file() else None
            stage = STAGES.get(job["kind"])
            result, error = None, None
            if timed_out:
                error = "timeout"
            elif rc not in (0, None):
                error = f"rc={rc}"
            if error is None:  # rc 0, or an adopted job without rc file: artifacts decide
                try:
                    result = stage.collect(json.loads(job["args"]))
                except Exception as exc:
                    error = f"stage finished but its artifacts are invalid: {exc}"
            if error and job.get("log") and Path(job["log"]).is_file():
                tail = [l for l in Path(job["log"]).read_text(errors="replace").splitlines()
                        if "b3Warning" not in l and "world element" not in l][-25:]
                error += "\n" + "\n".join(tail)
            con.execute("update jobs set status=?, rc=?, finished=?, result=?, error=? where id=?",
                        ("done" if error is None else "failed", rc, time.time(),
                         json.dumps(result) if result is not None else None, error, job["id"]))
        live = [j for j in running if j["pid"] and _alive(j["pid"])]
        used = sum(j["gpu_gb"] for j in live)
        n_live = len(live)
        per_kind: dict[str, int] = {}
        for j in live:
            per_kind[j["kind"]] = per_kind.get(j["kind"], 0) + 1
        for row in con.execute(
                "select * from jobs where status='queued' order by priority desc, id").fetchall():
            if n_live >= MAX_PARALLEL:
                break
            if per_kind.get(row["kind"], 0) >= MAX_PER_KIND.get(row["kind"], MAX_PARALLEL):
                continue
            deps = _dependencies(con, row)
            if any(d["status"] in ("failed", "cancelled") for d in deps):
                con.execute("update jobs set status='failed', error='dependency failed', finished=? "
                            "where id=?", (time.time(), row["id"]))
                continue
            if any(d["status"] != "done" for d in deps) or used + row["gpu_gb"] > GPU_BUDGET_GB:
                continue
            stage = STAGES[row["kind"]]
            args = json.loads(row["args"])
            log = LOGS / f"{row['id']:06d}_{row['kind']}.log"
            try:
                stage_env = stage.env(args)
                cmd = " ".join(shlex.quote(c) for c in stage.command(args, row["id"]))
            except Exception as exc:
                diagnostic = f"RESOURCE PREFLIGHT BLOCKED: {type(exc).__name__}: {exc}\n"
                log.write_text(diagnostic, encoding="utf-8")
                con.execute("update jobs set status='failed', finished=?, log=?, error=? where id=?",
                            (time.time(), str(log), diagnostic.strip(), row["id"]))
                continue
            wrapped = f"{cmd}; echo $? > {shlex.quote(str(_rc_file(row['id'])))}"
            with log.open("w") as fh:
                proc = subprocess.Popen(["bash", "-c", wrapped], cwd=stage.cwd, env=stage_env,
                                        stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            n_live += 1
            per_kind[row["kind"]] = per_kind.get(row["kind"], 0) + 1
            used += row["gpu_gb"]
            con.execute("update jobs set status='running', started=?, pid=?, log=? where id=?",
                        (time.time(), proc.pid, str(log), row["id"]))
        time.sleep(poll_s)


def _prepare_worker_environment() -> None:
    """Load credentials/configuration only; acquire hardware at stage admission."""
    requested_phase = os.environ.get("WO_RESOURCE_PHASE")
    load_private_environment(os.environ)
    apply_environment(os.environ, phase="worker")
    if requested_phase in {"preparation", "episode", "scoring", "evaluation"}:
        # Worker setup must not erase the launcher's phase gate or query GPUs.
        os.environ["WO_RESOURCE_PHASE"] = requested_phase


if __name__ == "__main__":
    if sys.argv[1:] == ["worker"]:
        worker()
    else:
        print("usage: python -m wo.agent.jobs worker")
