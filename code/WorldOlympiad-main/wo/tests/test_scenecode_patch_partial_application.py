from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "deployment/ziplab5090/scenecode_patch_utils.sh"


def test_partial_patch_dry_run_is_rejected_even_if_patch_exits_zero(tmp_path):
    target = tmp_path / "router.py"
    target.write_text("alpha\nbeta\ngamma\n")
    patch_file = tmp_path / "partial.patch"
    patch_file.write_text(
        "--- a/router.py\n"
        "+++ b/router.py\n"
        "@@ -1,2 +1,2 @@\n"
        " alpha\n"
        "-beta\n"
        "+BETA\n"
        "@@ -3,1 +3,1 @@\n"
        "-not-the-source\n"
        "+GAMMA\n"
    )
    dry_run = subprocess.run(
        ["patch", "--dry-run", "--forward", "--batch", "-p1", "-i", str(patch_file)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    output = dry_run.stdout + dry_run.stderr
    # GNU patch can report success after at least one hunk applied; rejected
    # hunks must independently fail the staging gate.
    checked = subprocess.run(
        ["bash", "-c", 'source "$1"; scenecode_patch_output_has_rejects "$2"',
         "test", str(HELPER), output],
        text=True,
        capture_output=True,
        check=False,
    )
    assert "FAILED" in output
    assert checked.returncode == 0
    assert target.read_text() == "alpha\nbeta\ngamma\n"
