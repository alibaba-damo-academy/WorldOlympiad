from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "deployment/ziplab5090/scenecode_runtime_utils.sh"


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True,
                   capture_output=True, text=True)


def _recognized(source: Path, runtime: Path) -> bool:
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; scenecode_runtime_is_interrupted_source_copy "$2" "$3"',
         "test", str(HELPER), str(source), str(runtime)],
        check=False, capture_output=True, text=True,
    )
    return result.returncode == 0


def test_interrupted_staging_copy_is_recovered_only_for_same_source_revision(tmp_path):
    source = tmp_path / "source"
    runtime = tmp_path / "runtime"
    (source / "scenecode").mkdir(parents=True)
    (source / "main.py").write_text("source entrypoint\n")
    (source / "scenecode" / "module.py").write_text("source module\n")
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "test@example.invalid")
    _git(source, "config", "user.name", "Test")
    _git(source, "remote", "add", "origin", "https://example.invalid/SceneCode.git")
    _git(source, "add", ".")
    _git(source, "commit", "-qm", "source")

    subprocess.run(["cp", "-R", str(source), str(runtime)], check=True)
    # A prior patch may have changed files when staging was interrupted.
    (runtime / "scenecode" / "module.py").write_text("patched module\n")
    assert _recognized(source, runtime)

    _git(runtime, "config", "remote.origin.url", "https://unrelated.invalid/repo.git")
    assert not _recognized(source, runtime)
    _git(runtime, "config", "remote.origin.url", "https://example.invalid/SceneCode.git")
    _git(runtime, "config", "user.email", "test@example.invalid")
    _git(runtime, "config", "user.name", "Test")
    (runtime / "scenecode" / "module.py").write_text("different committed source\n")
    _git(runtime, "add", ".")
    _git(runtime, "commit", "-qm", "different revision")
    assert not _recognized(source, runtime)
