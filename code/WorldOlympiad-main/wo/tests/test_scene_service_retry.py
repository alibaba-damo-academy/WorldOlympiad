import time

from wo.tasks.scene import _scenecode_logs_mention, run_with_service_retries


def test_empty_renderer_response_is_bounded_availability_retry(monkeypatch, tmp_path):
    import os
    import sys
    count = tmp_path / 'calls'
    code = ("import sys; from pathlib import Path; "
            f"p=Path({str(count)!r}); n=int(p.read_text())+1 if p.exists() else 1; "
            "p.write_text(str(n)); print('RenderClient: Server returned nothing (no headers, no data)',flush=True); "
            "sys.exit(1 if n==1 else 0)")
    monkeypatch.setattr('wo.tasks.scene.time.sleep', lambda _: None)
    run_with_service_retries([sys.executable, '-c', code], cwd=tmp_path, env=dict(os.environ), attempts=2)
    assert count.read_text() == '2'


def test_detects_upstream_error_in_scenecode_internal_log(tmp_path):
    run = tmp_path / "case" / "scene_run"
    run.mkdir(parents=True)
    (run / "experiment.log").write_text("VLM analysis failed: Error code: 503 - Service temporarily unavailable")
    assert _scenecode_logs_mention(("Error code: 503",), time.time() - 60, [tmp_path / "case"])


def test_ignores_old_logs_and_clean_logs(tmp_path):
    run = tmp_path / "case" / "scene_run"
    run.mkdir(parents=True)
    log = run / "experiment.log"
    log.write_text("all good")
    assert not _scenecode_logs_mention(("Error code: 503",), 0, [tmp_path / "case"])
    log.write_text("Error code: 503")
    assert not _scenecode_logs_mention(("Error code: 503",), time.time() + 60, [tmp_path / "case"])
