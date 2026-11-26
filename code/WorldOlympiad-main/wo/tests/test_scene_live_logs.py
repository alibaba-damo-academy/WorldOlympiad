import errno
import os
import sys
import threading

from wo.tasks import scene


def test_relays_progress_and_split_utf8_before_process_exits(tmp_path, monkeypatch):
    release = tmp_path / "release"
    seen = threading.Event()
    messages, errors = [], []
    def emit(message, **kwargs):
        messages.append(message)
        if "ready" in message:
            seen.set()
    monkeypatch.setattr(scene, "print", emit, raising=False)
    code = (
        "import os,time; from pathlib import Path; "
        "os.write(1,b'ready\\n'+bytes([228,189])); "
        f"release=Path({str(release)!r})\n"
        "while not release.exists(): time.sleep(.01)\n"
        "os.write(1,bytes([160])+b'\\nfinished\\n'); os.write(2,b'stderr warning\\n')"
    )
    def run():
        try:
            scene.run_with_service_retries([sys.executable, "-c", code],
                cwd=tmp_path, env=dict(os.environ), attempts=1)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        observed_before_exit = seen.wait(3)
    finally:
        release.touch()
        thread.join(5)
    assert not thread.is_alive()
    assert not errors
    assert observed_before_exit, "progress remained buffered until the child exited"
    output = "".join(messages)
    assert "你" in output
    assert "finished" in output and "stderr warning" in output


def test_does_not_wait_for_descendant_to_close_stdout(tmp_path, monkeypatch):
    release = tmp_path / "release_descendant"
    errors = []
    child = f"import time; from pathlib import Path; release=Path({str(release)!r})\nwhile not release.exists(): time.sleep(.01)"
    parent = "import subprocess,sys; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); print('parent done',flush=True)"
    def run():
        try:
            scene.run_with_service_retries([sys.executable, "-c", parent],
                cwd=tmp_path, env=dict(os.environ), attempts=1)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        thread.join(3)
        parent_finished = not thread.is_alive()
    finally:
        release.touch()
        thread.join(5)
    assert parent_finished, "relay waited for a background descendant's inherited stdout"
    assert not errors


def test_scene_log_spool_uses_node_local_storage_and_retries_cleanup(tmp_path, monkeypatch):
    real_mkdtemp = scene.tempfile.mkdtemp
    created = []

    def local_mkdtemp(*, prefix, dir):
        assert dir == "/tmp"
        result = real_mkdtemp(prefix=prefix, dir=dir)
        created.append(result)
        return result

    real_rmtree = scene.shutil.rmtree
    calls = 0

    def fail_once(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.ENOTEMPTY, "directory not empty", str(path))
        real_rmtree(path)

    monkeypatch.setattr(scene.tempfile, "mkdtemp", local_mkdtemp)
    monkeypatch.setattr(scene.shutil, "rmtree", fail_once)
    monkeypatch.setattr(scene.time, "sleep", lambda _: None)

    scene.run_with_service_retries([sys.executable, "-c", "print('done')"],
                                   cwd=tmp_path, env=dict(os.environ), attempts=1)

    assert calls == 2
    assert len(created) == 1
    assert not os.path.exists(created[0])


def test_split_availability_marker_and_attempt_budget(tmp_path, monkeypatch, capsys):
    import subprocess
    import pytest
    count = tmp_path / "attempts"
    code = ("import os,time; from pathlib import Path; "
            f"p=Path({str(count)!r}); p.write_text(str(int(p.read_text())+1 if p.exists() else 1)); "
            "os.write(1,b'HTTP 5'); time.sleep(.3); os.write(2,b'03 upstream unavailable'); raise SystemExit(7)")
    sleep = scene.time.sleep
    monkeypatch.setattr(scene.time, "sleep", lambda seconds: sleep(seconds) if seconds < 30 else None)
    with pytest.raises(subprocess.CalledProcessError) as error:
        scene.run_with_service_retries([sys.executable, "-c", code],
            cwd=tmp_path, env=dict(os.environ), attempts=2)
    assert count.read_text() == "2"
    assert error.value.returncode == 7
    assert capsys.readouterr().out.count("HTTP 503 upstream unavailable") == 2
