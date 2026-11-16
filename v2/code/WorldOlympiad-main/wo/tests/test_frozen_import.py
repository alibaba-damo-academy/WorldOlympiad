import json
import sqlite3

from wo.agent import frozen_import


def test_phase1_import_reuses_only_verified_completed_sources(tmp_path, monkeypatch):
    source_run_id, case, model, seed = "source_run", "case_a", "lingbot_va", 0
    source_run = tmp_path / "outputs/wo_runs" / source_run_id
    task = source_run / "tasks" / f"{case}.json"
    task.parent.mkdir(parents=True)
    task.write_text('{"case_id":"case_a","generation":"/scene"}')
    ref = source_run / "references" / case
    (ref / "replay").mkdir(parents=True)
    (ref / "certificate.json").write_text(json.dumps({"case_id": case, "success": True,
        "task": str(task), "task_sha256": frozen_import.sha256(task)}))
    (ref / "trajectory.json").write_text('{"frames": []}')
    (ref / "replay/exterior_left.mp4").write_bytes(b"reference")
    episode = source_run / case / f"{model}_s{seed}"
    (episode / "native_run/simulation").mkdir(parents=True)
    (episode / "capture").mkdir()
    (episode / "episode.json").write_text(json.dumps({"model": model, "seed": seed,
        "reference": str(ref), "task_success": False}))
    (episode / "episode_reference_binding.json").write_text('{}')
    (episode / "task_graph.json").write_text('{}')
    (episode / "native_run/simulation/trajectory.json").write_text('{"frames": []}')
    (episode / "capture/capture_metadata.json").write_text('{}')
    db = tmp_path / "outputs/wo_agent/jobs.sqlite3"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as con:
        con.execute("create table jobs(id integer primary key,kind text,args text,status text,rc integer)")
        rows = [
            (1, "scene", {"run": source_run_id, "case": case}),
            (2, "reference", {"run": source_run_id, "case": case}),
            (3, "episode", {"run": source_run_id, "case": case, "model": model, "seed": seed}),
        ]
        for jid, kind, args in rows:
            con.execute("insert into jobs values(?,?,?,?,?)", (jid, kind, json.dumps(args), "done", 0))
    monkeypatch.setattr("wo.eval.reference_binding.verify_task", lambda *_: {"case_id": case})
    monkeypatch.setattr("wo.eval.reference_binding.verify_episode_binding", lambda *_: {"status": "verified"})

    receipt = frozen_import.create_phase1_import(tmp_path, "phase1_run", source_run_id, case, model, seed)

    target = tmp_path / "outputs/wo_runs/phase1_run"
    assert receipt["source_jobs"] == {"scene": 1, "reference": 2, "episode": 3}
    assert (target / "tasks" / f"{case}.json").resolve() == task.resolve()
    assert (target / "references" / case).resolve() == ref.resolve()
    assert (target / case / f"{model}_s0/episode_original.json").is_file()
    assert frozen_import.verify_frozen_import(target, case)["source_run"] == source_run_id


def test_import_refuses_existing_run_directory(tmp_path):
    target = tmp_path / "outputs/wo_runs/phase1_run"
    target.mkdir(parents=True)
    try:
        frozen_import.create_phase1_import(tmp_path, "phase1_run", "source_run", "case_a", "lingbot_va", 0)
    except FileExistsError:
        pass
    else:
        raise AssertionError("refused to overwrite an existing run")
