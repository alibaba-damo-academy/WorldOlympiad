from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "deployment/ziplab5090/patches/scenecode_solid_task_marker_fallback.patch"


def test_task_marker_patch_import_hunk_tolerates_extra_upstream_import(tmp_path):
    """The staged SceneCode router may import shutil between os and tempfile."""
    source = tmp_path / "scenecode/agent_utils/asset_router/router.py"
    source.parent.mkdir(parents=True)
    source.write_text("import json\nimport logging\nimport os\nimport shutil\nimport tempfile\nimport time\n")

    patch_lines = PATCH.read_text().splitlines(keepends=True)
    hunk_start = next(i for i, line in enumerate(patch_lines) if line.startswith("@@"))
    hunk_end = next(
        i for i in range(hunk_start + 1, len(patch_lines))
        if patch_lines[i].startswith("@@")
    )
    minimal_patch = (
        "--- a/scenecode/agent_utils/asset_router/router.py\n"
        "+++ b/scenecode/agent_utils/asset_router/router.py\n"
        + "".join(patch_lines[hunk_start:hunk_end])
    )
    patch_file = tmp_path / "import.patch"
    patch_file.write_text(minimal_patch)
    completed = subprocess.run(
        ["patch", "--forward", "--batch", "-p1", "-i", str(patch_file)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "import re\n" in source.read_text()
