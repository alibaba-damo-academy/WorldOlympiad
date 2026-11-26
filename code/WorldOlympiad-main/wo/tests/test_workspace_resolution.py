from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_default_workspace_tracks_the_enclosing_wov2_checkout():
    source_root = Path(__file__).resolve().parents[2]
    expected = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env.pop("WO_WORKSPACE", None)
    env["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [sys.executable, "-c", "from wo.paths import WORKSPACE; print(WORKSPACE)"],
        cwd=source_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(result.stdout.strip()) == expected
