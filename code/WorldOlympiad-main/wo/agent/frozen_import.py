"""Audited import of a completed native episode and its frozen reference.

This is a scoped continuation aid: it reuses immutable Scene/reference artifacts
and a completed native trajectory while keeping the new scoring run isolated.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

RECEIPT = "phase1_import_receipt.json"
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _root(run: Path) -> Path:
    return Path(run).resolve().parents[2]


def _candidate_revision(root: Path) -> str | None:
    proc = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          check=False)
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value) else None


def _job(database: Path, *, kind: str, run: str, case: str,
         model: str | None = None, seed: int | None = None) -> dict:
    matches = []
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        for row in con.execute("select id,kind,args,status,rc from jobs where kind=? order by id", (kind,)):
            args = json.loads(row["args"])
            if args.get("run") != run or args.get("case") != case:
                continue
            if model is not None and args.get("model") != model:
                continue
            if seed is not None and int(args.get("seed", -1)) != int(seed):
                continue
            matches.append((dict(row), args))
    if not matches:
        raise ValueError(f"source run has no {kind} job for the requested task/model")
    row, args = matches[-1]
    if row["status"] != "done" or row["rc"] != 0:
        raise ValueError(f"latest source {kind} job #{row['id']} did not complete successfully")
    return {"id": row["id"], "kind": kind, "run": run, "case": case,
            "model": model, "seed": seed, "status": row["status"], "rc": row["rc"],
            "args": args}


def _source_artifacts(source_run: Path, case: str, model: str, seed: int) -> tuple[Path, Path, Path, dict]:
    task = source_run / "tasks" / f"{case}.json"
    ref = source_run / "references" / case
    episode = source_run / case / f"{model}_s{seed}"
    required = {
        "task": task,
        "reference_certificate": ref / "certificate.json",
        "reference_trajectory": ref / "trajectory.json",
        "reference_exterior_left": ref / "replay/exterior_left.mp4",
        "episode_metadata": episode / "episode.json",
        "episode_binding": episode / "episode_reference_binding.json",
        "episode_task_graph": episode / "task_graph.json",
        "native_trajectory": episode / "native_run/simulation/trajectory.json",
        "capture_metadata": episode / "capture/capture_metadata.json",
    }
    missing = [name for name, path in required.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("frozen source is missing required artifacts: " + ", ".join(missing))
    return task, ref, episode, {name: sha256(path) for name, path in required.items()}


def verify_frozen_import(run: Path, case: str) -> dict:
    run = Path(run).resolve()
    receipt_path = run / RECEIPT
    data = json.loads(receipt_path.read_text())
    if data.get("schema_version") != "wo_phase1_frozen_import.v1" or data.get("case_id") != case:
        raise ValueError("invalid frozen-import receipt or case binding")
    source_run_id = data.get("source_run")
    if not isinstance(source_run_id, str) or not RUN_ID.fullmatch(source_run_id):
        raise ValueError("frozen-import receipt has an invalid source run id")
    model, seed = data.get("model"), data.get("seed")
    if not isinstance(model, str) or not isinstance(seed, int):
        raise ValueError("frozen-import receipt has an invalid model/seed")
    root = _root(run)
    source_run = root / "outputs/wo_runs" / source_run_id
    database = root / "outputs/wo_agent/jobs.sqlite3"
    source_jobs = data.get("source_jobs") or {}
    for kind in ("scene", "reference", "episode"):
        jid = source_jobs.get(kind)
        if not isinstance(jid, int):
            raise ValueError(f"frozen-import receipt lacks source {kind} job identity")
        with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as con:
            con.row_factory = sqlite3.Row
            row = con.execute("select id,kind,args,status,rc from jobs where id=?", (jid,)).fetchone()
        if row is None or row["kind"] != kind or row["status"] != "done" or row["rc"] != 0:
            raise ValueError(f"frozen-import source {kind} job is no longer verified")
        args = json.loads(row["args"])
        if args.get("run") != source_run_id or args.get("case") != case:
            raise ValueError(f"frozen-import source {kind} job belongs to another task")
        if kind == "episode" and (args.get("model") != model or int(args.get("seed", -1)) != seed):
            raise ValueError("frozen-import episode job belongs to another model/seed")
    task, ref, source_episode, artifacts = _source_artifacts(source_run, case, model, seed)
    if artifacts != data.get("source_artifacts"):
        raise ValueError("frozen source task/reference/episode inputs changed after import")
    imported_task = run / "tasks" / f"{case}.json"
    imported_ref = run / "references" / case
    if not imported_task.is_file() or imported_task.resolve() != task.resolve():
        raise ValueError("new run TaskSpec is not linked to the recorded frozen source")
    if not imported_ref.is_dir() or imported_ref.resolve() != ref.resolve():
        raise ValueError("new run reference is not linked to the recorded frozen source")
    if sha256(run / case / f"{model}_s{seed}/episode_original.json") != data["source_artifacts"]["episode_metadata"]:
        raise ValueError("preserved original episode metadata differs from the frozen source")
    imported_episode = run / case / f"{model}_s{seed}"
    if sha256(imported_episode / "native_run/simulation/trajectory.json") != data["source_artifacts"]["native_trajectory"]:
        raise ValueError("imported native trajectory differs from the frozen source")
    if sha256(imported_episode / "task_graph.json") != data["source_artifacts"]["episode_task_graph"]:
        raise ValueError("imported task graph differs from the frozen source")
    if sha256(imported_episode / "episode_reference_binding.json") != data["source_artifacts"]["episode_binding"]:
        raise ValueError("imported episode binding receipt differs from the frozen source")
    if sha256(imported_episode / "capture/capture_metadata.json") != data["source_artifacts"]["capture_metadata"]:
        raise ValueError("imported capture metadata differs from the frozen source")
    from ..eval.reference_binding import verify_episode_binding, verify_task
    verify_task(imported_ref, imported_task)
    episode_dir = run / case / f"{model}_s{seed}"
    episode = json.loads((episode_dir / "episode.json").read_text())
    if episode.get("model") != model or int(episode.get("seed", -1)) != seed:
        raise ValueError("imported episode metadata disagrees with the declared model/seed")
    if Path(episode["reference"]).resolve() != imported_ref.resolve():
        raise ValueError("imported episode points at a different expert reference")
    if verify_episode_binding(episode_dir, imported_ref).get("status") != "verified":
        raise ValueError("imported native episode is not reference-bound")
    return data


def create_phase1_import(root: Path, run_id: str, source_run_id: str,
                         case: str, model: str, seed: int) -> dict:
    root = Path(root).resolve()
    for value, label in ((run_id, "run id"), (source_run_id, "source run id"), (case, "case"), (model, "model")):
        if not isinstance(value, str) or not RUN_ID.fullmatch(value):
            raise ValueError(f"invalid {label}")
    seed = int(seed)
    source_run = root / "outputs/wo_runs" / source_run_id
    target = root / "outputs/wo_runs" / run_id
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing run: {target}")
    database = root / "outputs/wo_agent/jobs.sqlite3"
    if not database.is_file():
        raise FileNotFoundError("source job database is missing")
    source_jobs = {
        "scene": _job(database, kind="scene", run=source_run_id, case=case),
        "reference": _job(database, kind="reference", run=source_run_id, case=case),
        "episode": _job(database, kind="episode", run=source_run_id, case=case, model=model, seed=seed),
    }
    task, ref, episode, artifacts = _source_artifacts(source_run, case, model, seed)
    from ..eval.reference_binding import verify_episode_binding, verify_task
    cert = verify_task(ref, task)
    if cert.get("case_id") != case:
        raise ValueError("expert certificate does not match the requested case")
    source_episode = json.loads((episode / "episode.json").read_text())
    if source_episode.get("model") != model or int(source_episode.get("seed", -1)) != seed:
        raise ValueError("native episode metadata does not match the requested model/seed")
    if Path(source_episode["reference"]).resolve() != ref.resolve():
        raise ValueError("native episode points at another reference")
    if verify_episode_binding(episode, ref).get("status") != "verified":
        raise ValueError("native episode has no verified reference binding")
    manifest = {"schema_version": "wo_run_manifest.v1", "models": [model],
                "seeds": [seed], "cases": [case],
                "phase1_mode": "frozen_reference_and_native_episode_reuse",
                "source_run": source_run_id}
    target.mkdir(parents=True)
    (target / "tasks").mkdir()
    (target / "references").mkdir()
    (target / "tasks" / f"{case}.json").symlink_to(task.resolve())
    (target / "references" / case).symlink_to(ref.resolve(), target_is_directory=True)
    episode_target = target / case / f"{model}_s{seed}"
    episode_target.mkdir(parents=True)
    for name in ("episode.json", "episode_reference_binding.json", "task_graph.json"):
        shutil.copy2(episode / name, episode_target / name)
    shutil.copy2(episode / "episode.json", episode_target / "episode_original.json")
    for name in ("capture", "native_run", "server_output", "lingbot"):
        path = episode / name
        if path.exists():
            (episode_target / name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    receipt = {"schema_version": "wo_phase1_frozen_import.v1", "run_id": run_id,
               "candidate_commit": _candidate_revision(root),
               "source_run": source_run_id, "case_id": case, "model": model, "seed": seed,
               "source_jobs": {name: item["id"] for name, item in source_jobs.items()},
               "source_artifacts": artifacts,
               "source_paths": {"task": str(task.resolve()), "reference": str(ref.resolve()),
                                "episode": str(episode.resolve())},
               "reuse_scope": ["task", "scene_generation", "expert_reference", "native_episode", "native_trajectory"],
               "note": "Reference and task are read-only symlinks; original episode metadata is preserved separately."}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (target / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
    verify_frozen_import(target, case)
    return receipt
