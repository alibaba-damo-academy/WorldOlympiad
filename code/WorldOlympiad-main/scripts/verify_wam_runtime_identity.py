#!/usr/bin/env python3
"""Verify the exact source, checkpoint, and runtime assets used by one WAM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from autoresearch.io import sha256_file, write_json


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_record(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
    }


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": "wam_runtime_hash_cache.v1", "files": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "wam_runtime_hash_cache.v1"
        or not isinstance(payload.get("files"), dict)
    ):
        raise ValueError(f"invalid runtime hash cache: {path}")
    return payload


def _verify_checkpoint(
    *,
    key: str,
    root: Path,
    expected: dict[str, Any],
    cache_root: Path,
    strict_full_hash: bool,
) -> dict[str, Any]:
    revision = str(expected["resolved_revision"])
    tree_path = root / ".cache" / "huggingface" / "trees" / f"{revision}.json"
    tree_payload = json.loads(tree_path.read_text(encoding="utf-8"))
    tree_files = tree_payload.get("files")
    if tree_payload.get("format_version") != 1 or not isinstance(tree_files, dict):
        raise ValueError(f"unsupported Hugging Face local-dir tree: {tree_path}")
    include = expected.get("include_paths")
    include_set = set(include) if include is not None else None
    selected = {
        path: {
            field: record[field]
            for field in ("size", "blob_id", "lfs_sha256", "lfs_size")
            if field in record
        }
        for path, record in tree_files.items()
        if include_set is None or path in include_set
    }
    if include_set is not None and set(selected) != include_set:
        raise ValueError(f"checkpoint tree is missing its frozen include set: {key}")
    tree_identity = _canonical_sha256(
        {"resolved_revision": revision, "files": selected}
    )
    if tree_identity != expected["tree_identity_sha256"]:
        raise ValueError(f"checkpoint tree identity changed: {key}")

    overlays = expected.get("overlays") or {}
    always_hash = set(expected.get("always_hash_lfs") or [])
    cache_path = cache_root / f"{tree_identity}.json"
    cache = _load_cache(cache_path)
    cached_files = cache["files"]
    next_cache: dict[str, Any] = {}
    reused_large_hashes = 0
    computed_large_hashes = 0
    verified: dict[str, dict[str, Any]] = {}
    for relative, record in selected.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if relative in overlays:
            overlay = overlays[relative]
            if path.stat().st_size != int(overlay["size"]):
                raise ValueError(f"runtime overlay size changed: {key}:{relative}")
            actual = sha256_file(path)
            if actual != overlay["sha256"]:
                raise ValueError(f"runtime overlay digest changed: {key}:{relative}")
            verified[relative] = {"kind": "reviewed_overlay_sha256", "digest": actual}
            continue
        expected_size = int(record.get("lfs_size", record["size"]))
        if path.stat().st_size != expected_size:
            raise ValueError(f"checkpoint file size changed: {key}:{relative}")
        if "lfs_sha256" in record:
            wanted = str(record["lfs_sha256"])
            stat = _stat_record(path)
            cached = cached_files.get(relative)
            can_reuse = (
                not strict_full_hash
                and relative not in always_hash
                and expected_size >= 64 * 1024 * 1024
                and isinstance(cached, dict)
                and cached.get("expected_sha256") == wanted
                and cached.get("stat") == stat
            )
            if can_reuse:
                actual = wanted
                reused_large_hashes += 1
            else:
                actual = sha256_file(path)
                if expected_size >= 64 * 1024 * 1024:
                    computed_large_hashes += 1
            if actual != wanted:
                raise ValueError(f"checkpoint LFS digest changed: {key}:{relative}")
            next_cache[relative] = {"expected_sha256": wanted, "stat": stat}
            verified[relative] = {"kind": "sha256", "digest": wanted}
        else:
            actual = _git_blob_sha1(path)
            if actual != record["blob_id"]:
                raise ValueError(f"checkpoint Git blob changed: {key}:{relative}")
            verified[relative] = {"kind": "git_blob_sha1", "digest": actual}

    for relative, record in (expected.get("extra_files") or {}).items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["size"]) or sha256_file(path) != record["sha256"]:
            raise ValueError(f"checkpoint runtime marker changed: {key}:{relative}")
        verified[relative] = {"kind": "runtime_marker_sha256", "digest": record["sha256"]}
    cache_root.mkdir(parents=True, exist_ok=True)
    write_json(
        cache_path,
        {
            "schema_version": "wam_runtime_hash_cache.v1",
            "tree_identity_sha256": tree_identity,
            "files": next_cache,
        },
    )
    return {
        "key": key,
        "root": str(root),
        "resolved_revision": revision,
        "tree_identity_sha256": tree_identity,
        "verified_file_count": len(verified),
        "verified_files_sha256": _canonical_sha256(verified),
        "reused_large_hash_count": reused_large_hashes,
        "computed_large_hash_count": computed_large_hashes,
    }


def _verify_git_component(
    *, component: str, root: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected["commit"]:
        raise ValueError(f"{component} commit changed: {commit}")
    status_lines = [
        line
        for line in subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if line
    ]
    changed: set[str] = set()
    for line in status_lines:
        if len(line) < 4 or " -> " in line:
            raise ValueError(f"unsupported {component} worktree change: {line}")
        changed.add(line[3:])
    allowed = expected.get("allowed_worktree_files") or {}
    if changed != set(allowed):
        raise ValueError(
            f"{component} worktree differs from its reviewed overlay: "
            f"actual={sorted(changed)}, expected={sorted(allowed)}"
        )
    for relative, digest in allowed.items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"{component} reviewed overlay changed: {relative}")
    files: dict[str, str] = {}
    for relative, wanted in expected["files"].items():
        path = root / relative
        actual = sha256_file(path) if path.is_file() else "<missing>"
        if actual != wanted:
            raise ValueError(f"{component} runtime source changed: {relative}")
        files[relative] = actual
    return {
        "component": component,
        "root": str(root),
        "commit": commit,
        "worktree_overlays": sorted(changed),
        "files_sha256": _canonical_sha256(files),
    }


def _verify_runtime_tree(
    *, component: str, root: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    tree_root = root / expected["tree_relative_root"]
    files = {
        path.relative_to(tree_root).as_posix(): sha256_file(path)
        for path in sorted(tree_root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    digest = _canonical_sha256(files)
    if len(files) != int(expected["file_count"]) or digest != expected["tree_sha256"]:
        raise ValueError(f"{component} runtime tree changed")
    return {
        "component": component,
        "root": str(tree_root),
        "file_count": len(files),
        "tree_sha256": digest,
    }


def verify_runtime_identity(
    *,
    profile: str,
    lock_path: Path,
    checkpoint_root: Path,
    source_root: Path,
    cache_root: Path,
    output: Path,
    external_roots: dict[str, Path] | None = None,
    runtime_tree_root: Path | None = None,
    strict_full_hash: bool = False,
) -> Path:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 2:
        raise ValueError("WAM runtime identity requires upstream lock v2")
    profile_contract = lock["runtime_profiles"][profile]
    component_name = profile_contract["source_component"]
    source = _verify_git_component(
        component=component_name,
        root=source_root.resolve(),
        expected=lock["components"][component_name],
    )
    checkpoint_key = profile_contract["checkpoint"]
    checkpoint = _verify_checkpoint(
        key=checkpoint_key,
        root=checkpoint_root.resolve(),
        expected=lock["checkpoints"][checkpoint_key],
        cache_root=cache_root.resolve(),
        strict_full_hash=strict_full_hash,
    )
    external_roots = external_roots or {}
    external = []
    for key in profile_contract.get("external_artifacts", []):
        if key not in external_roots:
            raise ValueError(f"missing external checkpoint root for {key}")
        external.append(
            _verify_checkpoint(
                key=key,
                root=external_roots[key].resolve(),
                expected=lock["checkpoints"][key],
                cache_root=cache_root.resolve(),
                strict_full_hash=strict_full_hash,
            )
        )
    runtime_tree = None
    tree_component = profile_contract.get("runtime_tree_component")
    if tree_component is not None:
        if runtime_tree_root is None:
            raise ValueError(f"profile {profile} requires its native runtime tree")
        runtime_tree = _verify_runtime_tree(
            component=tree_component,
            root=runtime_tree_root.resolve(),
            expected=lock["components"][tree_component],
        )
    relevant_lock = {
        "profile": profile_contract,
        "source_component": lock["components"][component_name],
        "checkpoint": lock["checkpoints"][checkpoint_key],
        "external": {
            key: lock["checkpoints"][key]
            for key in profile_contract.get("external_artifacts", [])
        },
        "runtime_tree": (
            lock["components"][tree_component] if tree_component is not None else None
        ),
    }
    identity = {
        "profile": profile,
        "source_model": profile_contract["source_model"],
        "relevant_lock_sha256": _canonical_sha256(relevant_lock),
        "source": {key: value for key, value in source.items() if key != "root"},
        "checkpoint": {
            key: value
            for key, value in checkpoint.items()
            if not key.endswith("_count") and key != "root"
        },
        "external_artifacts": [
            {key: value for key, value in item.items() if not key.endswith("_count") and key != "root"}
            for item in external
        ],
        "runtime_tree": (
            {key: value for key, value in runtime_tree.items() if key != "root"}
            if runtime_tree is not None
            else None
        ),
    }
    stable_checkpoint_verification = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"reused_large_hash_count", "computed_large_hash_count"}
    }
    stable_external_verification = [
        {
            key: value
            for key, value in item.items()
            if key not in {"reused_large_hash_count", "computed_large_hash_count"}
        }
        for item in external
    ]
    certificate = {
        "schema_version": "wam_runtime_identity.v1",
        "identity_sha256": _canonical_sha256(identity),
        "identity": identity,
        "verification": {
            "source": source,
            "checkpoint": stable_checkpoint_verification,
            "external_artifacts": stable_external_verification,
            "runtime_tree": runtime_tree,
        },
    }
    return write_json(output.resolve(), certificate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=("cosmos3_nano_droid", "lingbot_va_base"))
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-tree-root", type=Path)
    parser.add_argument("--external", action="append", default=[], metavar="LOCK_KEY=PATH")
    parser.add_argument("--strict-full-hash", action="store_true")
    args = parser.parse_args()
    external: dict[str, Path] = {}
    for value in args.external:
        key, separator, path = value.partition("=")
        if not separator or not key or not path:
            parser.error(f"invalid --external value: {value!r}")
        external[key] = Path(path)
    print(
        verify_runtime_identity(
            profile=args.profile,
            lock_path=args.lock.resolve(),
            checkpoint_root=args.checkpoint_root,
            source_root=args.source_root,
            cache_root=args.cache_root,
            output=args.output,
            external_roots=external,
            runtime_tree_root=args.runtime_tree_root,
            strict_full_hash=args.strict_full_hash,
        )
    )


if __name__ == "__main__":
    main()
