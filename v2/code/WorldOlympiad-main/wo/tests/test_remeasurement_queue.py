import json
from wo.agent import jobs, remeasure


def test_invalidated_attempts_requeue_once_and_real_failures_are_untouched(tmp_path,monkeypatch):
    monkeypatch.setattr(jobs,"DB",tmp_path/"jobs.sqlite3")
    run=tmp_path/"run";invalid=run/"sample"/"fastwam_s0";invalid.mkdir(parents=True)
    (invalid/"evaluation_invalidated.json").write_text(json.dumps({"reason":"known repaired adapter bug"}))
    failed=run/"real_failure"/"fastwam_s0";failed.mkdir(parents=True)
    (failed/"episode.json").write_text(json.dumps({"task_success":False}))
    monkeypatch.setattr(remeasure,"run_dir",lambda _:run)
    first=remeasure.queue_invalidated("audit")
    second=remeasure.queue_invalidated("audit")
    assert len(first)==len(second)==1
    assert first[0]["episode"]==second[0]["episode"] and first[0]["score"]==second[0]["score"]
    assert second[0]["episode_reused"] and second[0]["score_reused"]
    assert len(jobs.listing(limit=10))==2
    score=jobs.get(first[0]["score"])
    assert json.loads(score["after"])==[first[0]["episode"]]
    assert not (failed/"evaluation_invalidated.json").exists()
