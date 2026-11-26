from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.verify_wam_runtime_identity import (
    _canonical_sha256,
    _git_blob_sha1,
    verify_runtime_identity,
)


def _git(args: list[str], root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_runtime_identity_binds_git_source_and_checkpoint_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(["init", "-q"], source)
    _git(["config", "user.email", "test@example.com"], source)
    _git(["config", "user.name", "Runtime Test"], source)
    source_file = source / "runner.py"
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    _git(["add", "runner.py"], source)
    _git(["commit", "-qm", "frozen"], source)
    commit = _git(["rev-parse", "HEAD"], source)
    source_sha = hashlib.sha256(source_file.read_bytes()).hexdigest()

    checkpoint = tmp_path / "checkpoint"
    revision = "a" * 40
    tree_root = checkpoint / ".cache" / "huggingface" / "trees"
    tree_root.mkdir(parents=True)
    config = checkpoint / "config.json"
    weights = checkpoint / "weights.bin"
    config.write_text('{"model":"fixture"}\n', encoding="utf-8")
    weights.write_bytes(b"frozen-weights")
    tree_files = {
        "config.json": {
            "size": config.stat().st_size,
            "blob_id": _git_blob_sha1(config),
        },
        "weights.bin": {
            "size": weights.stat().st_size,
            "blob_id": "b" * 40,
            "lfs_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "lfs_size": weights.stat().st_size,
        },
    }
    (tree_root / f"{revision}.json").write_text(
        json.dumps({"format_version": 1, "files": tree_files}),
        encoding="utf-8",
    )
    tree_identity = _canonical_sha256(
        {"resolved_revision": revision, "files": tree_files}
    )
    lock = {
        "schema_version": 2,
        "runtime_profiles": {
            "lingbot_va_base": {
                "source_model": "fixture/model",
                "checkpoint": "fixture/model",
                "source_component": "fixture_source",
                "external_artifacts": [],
            }
        },
        "checkpoints": {
            "fixture/model": {
                "resolved_revision": revision,
                "tree_identity_sha256": tree_identity,
                "include_paths": None,
                "overlays": {},
                "always_hash_lfs": ["weights.bin"],
            }
        },
        "components": {
            "fixture_source": {
                "commit": commit,
                "files": {"runner.py": source_sha},
                "allowed_worktree_files": {},
            }
        },
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    output = tmp_path / "identity.json"
    verify_runtime_identity(
        profile="lingbot_va_base",
        lock_path=lock_path,
        checkpoint_root=checkpoint,
        source_root=source,
        cache_root=tmp_path / "cache",
        output=output,
    )
    certificate = json.loads(output.read_text(encoding="utf-8"))
    assert certificate["schema_version"] == "wam_runtime_identity.v1"
    assert len(certificate["identity_sha256"]) == 64
    assert certificate["identity"]["checkpoint"]["resolved_revision"] == revision
    first_certificate = output.read_bytes()
    verify_runtime_identity(
        profile="lingbot_va_base",
        lock_path=lock_path,
        checkpoint_root=checkpoint,
        source_root=source,
        cache_root=tmp_path / "cache",
        output=output,
    )
    assert output.read_bytes() == first_certificate

    weights.write_bytes(b"tamper-weights")
    with pytest.raises(ValueError, match="LFS digest changed"):
        verify_runtime_identity(
            profile="lingbot_va_base",
            lock_path=lock_path,
            checkpoint_root=checkpoint,
            source_root=source,
            cache_root=tmp_path / "cache",
            output=output,
        )


def test_runtime_identity_rejects_unreviewed_source_changes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(["init", "-q"], source)
    _git(["config", "user.email", "test@example.com"], source)
    _git(["config", "user.name", "Runtime Test"], source)
    file = source / "runner.py"
    file.write_text("VALUE = 1\n", encoding="utf-8")
    _git(["add", "runner.py"], source)
    _git(["commit", "-qm", "frozen"], source)
    file.write_text("VALUE = 2\n", encoding="utf-8")

    from scripts.verify_wam_runtime_identity import _verify_git_component

    with pytest.raises(ValueError, match="worktree differs"):
        _verify_git_component(
            component="fixture",
            root=source,
            expected={
                "commit": _git(["rev-parse", "HEAD"], source),
                "files": {},
                "allowed_worktree_files": {},
            },
        )
