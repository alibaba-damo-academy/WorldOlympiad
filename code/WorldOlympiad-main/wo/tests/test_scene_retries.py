import os
import subprocess
import sys

import pytest
from wo.tasks import scene


@pytest.mark.parametrize('message,retries', [('HTTP 503 upstream', True),
                                            ('invalid geometry', False)])
def test_retry_only_availability(monkeypatch, tmp_path, message, retries):
    count = tmp_path / 'calls'
    code = ("import sys; from pathlib import Path; "
            f"p=Path({str(count)!r}); n=int(p.read_text())+1 if p.exists() else 1; "
            "p.write_text(str(n)); " + f"print({message!r},flush=True); "
            "sys.exit(1 if n==1 else 0)")
    command = [sys.executable, '-c', code]
    monkeypatch.setattr(scene.time, 'sleep', lambda _: None)
    if retries:
        scene.run_with_service_retries(command, cwd=tmp_path, env=dict(os.environ))
        assert count.read_text() == '2'
    else:
        with pytest.raises(subprocess.CalledProcessError) as failure:
            scene.run_with_service_retries(command, cwd=tmp_path, env=dict(os.environ))
        assert failure.value.returncode == 1
        assert failure.value.cmd == command
        assert count.read_text() == '1'
