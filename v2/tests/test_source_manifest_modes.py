from __future__ import annotations

import importlib.util
import os
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/setup_sources.py"
spec = importlib.util.spec_from_file_location("setup_sources", SCRIPT)
setup_sources = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(setup_sources)


def test_manifest_modes_match_git_checkout_semantics(tmp_path: Path):
    regular = tmp_path / "shared.txt"
    executable = tmp_path / "run.sh"
    regular.write_text("data")
    executable.write_text("#!/bin/sh\necho ok\n")
    regular.chmod(0o664)
    executable.chmod(0o775)
    (tmp_path / "alias.txt").symlink_to("shared.txt")

    entries = setup_sources.bundle_entries(tmp_path, set())
    by_path = {row["path"]: row for row in entries}
    assert by_path["shared.txt"]["mode"] == 0o644
    assert by_path["run.sh"]["mode"] == 0o755
    assert by_path["alias.txt"]["mode"] == stat.S_IMODE(os.lstat(tmp_path / "alias.txt").st_mode)


def test_only_declared_elf_runtime_artifacts_are_excluded(tmp_path: Path):
    generated = "src/curobo/curobolib/geom_cu.cpython-310-x86_64-linux-gnu.so"
    artifact = tmp_path / generated
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"\x7fELFgenerated")
    unexpected = tmp_path / "src/curobo/curobolib/unexpected.so"
    unexpected.write_bytes(b"\x7fELFother")

    excluded = setup_sources.runtime_generated_artifacts(
        tmp_path, {"name": "cuRobo", "runtime_generated_artifacts": [generated]}
    )
    assert excluded == {generated}
    entries = setup_sources.bundle_entries(tmp_path, excluded)
    assert generated not in {row["path"] for row in entries}
    assert "src/curobo/curobolib/unexpected.so" in {row["path"] for row in entries}


def test_declared_runtime_artifact_must_be_elf(tmp_path: Path):
    generated = "src/curobo/curobolib/geom_cu.cpython-310-x86_64-linux-gnu.so"
    artifact = tmp_path / generated
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"not an ELF")

    try:
        setup_sources.runtime_generated_artifacts(
            tmp_path, {"name": "cuRobo", "runtime_generated_artifacts": [generated]}
        )
    except SystemExit as exc:
        assert "not an ELF binary" in str(exc)
    else:
        raise AssertionError("non-ELF file was accepted as a runtime build artifact")
