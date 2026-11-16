from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_rsi_base.py"
spec = importlib.util.spec_from_file_location("prepare_rsi_base", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(prepare)


def test_output_link_setup_is_idempotent_when_target_starts_absent(tmp_path: Path):
    worktree = tmp_path / "source-worktree"
    target = tmp_path / "candidate" / "outputs"
    worktree.mkdir()

    prepare.ensure_output_link(worktree, target)
    link = worktree / "outputs"
    assert target.is_dir()
    assert link.is_symlink()
    assert link.resolve(strict=True) == target.resolve(strict=True)

    prepare.ensure_output_link(worktree, target)
    assert link.is_symlink()
    assert link.resolve(strict=True) == target.resolve(strict=True)


def test_output_link_setup_preserves_an_occupied_path(tmp_path: Path):
    worktree = tmp_path / "source-worktree"
    target = tmp_path / "candidate" / "outputs"
    worktree.mkdir()
    occupied = worktree / "outputs"
    occupied.mkdir()
    marker = occupied / "keep.txt"
    marker.write_text("keep")

    with pytest.raises(RuntimeError, match="left untouched"):
        prepare.ensure_output_link(worktree, target)

    assert marker.read_text() == "keep"
    assert not target.exists()
