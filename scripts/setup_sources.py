#!/usr/bin/env python3
"""Materialize and verify the pinned WOV2 source set without replacing data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "sources.lock.json"
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED = {
    "WorldOlympiad-main", "scenecode_pybullet_minimal", "SceneCode",
    "RoboTwin", "LingBot-VA", "cuRobo", "Depth-Anything-3",
}


def stop(message: str) -> None:
    raise SystemExit(f"source setup blocked: {message}")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def git(args: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if check and proc.returncode:
        stop(proc.stderr.strip() or f"command failed: {args[0]}")
    return proc.stdout.strip()


def relpath(raw: object, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        stop(f"{label}: path is missing")
    value = Path(raw)
    if value.is_absolute() or ".." in value.parts or not value.parts or value.parts[0] != "code":
        stop(f"{label}: path must stay under code/")
    return value


def read_lock() -> list[dict]:
    try:
        data = json.loads(LOCK.read_text(encoding="utf-8"))
    except Exception as exc:
        stop(f"cannot parse {LOCK.name} ({type(exc).__name__})")
    if not isinstance(data, dict) or data.get("schema_version") != 3 or not isinstance(data.get("sources"), list):
        stop("unsupported source-lock schema; expected schema_version 3")
    items = data["sources"]
    names = [item.get("name") if isinstance(item, dict) else None for item in items]
    if len(names) != len(set(names)) or set(names) != EXPECTED:
        stop("lock must contain the seven expected source entries exactly once")
    seen: set[Path] = set()
    for item in items:
        name = item["name"]
        kind = item.get("kind")
        if kind not in {"bundle", "git"}:
            stop(f"{name}: kind must be bundle or git")
        path = relpath(item.get("path"), name)
        resolved = (ROOT / path).resolve(strict=False)
        if not resolved.is_relative_to(ROOT.resolve()):
            stop(f"{name}: target escapes the evaluator root")
        if path in seen:
            stop(f"{name}: duplicate source destination")
        seen.add(path)
        if kind == "bundle":
            if item.get("source_commit") is not None and not SHA1.fullmatch(str(item["source_commit"])):
                stop(f"{name}: invalid source commit")
            if not SHA256.fullmatch(str(item.get("manifest_sha256", ""))) or not SHA256.fullmatch(str(item.get("tree_sha256", ""))):
                stop(f"{name}: manifest and tree SHA-256 values are required")
            baseline = item.get("rsi_git_baseline")
            if baseline is not None and not SHA1.fullmatch(str(baseline)):
                stop(f"{name}: invalid deterministic RSI Git baseline")
            manifest = Path(str(item.get("manifest", "")))
            if manifest.is_absolute() or ".." in manifest.parts or not manifest.parts or manifest.parts[0] != "locks":
                stop(f"{name}: manifest must stay under locks/")
        else:
            url = item.get("url")
            if not isinstance(url, str) or not url.startswith("https://"):
                stop(f"{name}: a canonical HTTPS Git URL is required")
            parsed = urlsplit(url)
            if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                stop(f"{name}: Git URL must not contain credentials, query, or fragment")
            if not SHA1.fullmatch(str(item.get("commit", ""))):
                stop(f"{name}: immutable 40-character commit is required")
            if not isinstance(item.get("submodules"), bool):
                stop(f"{name}: submodules must be explicitly true or false")
        generated = item.get("runtime_generated_artifacts", [])
        if not isinstance(generated, list) or len(generated) != len(set(generated)):
            stop(f"{name}: runtime_generated_artifacts must be a unique path list")
        if generated and kind != "bundle":
            stop(f"{name}: runtime-generated artifacts are only valid for bundles")
        for rel in generated:
            p = Path(rel) if isinstance(rel, str) else Path("/")
            if p.is_absolute() or ".." in p.parts or not p.parts:
                stop(f"{name}: invalid runtime-generated artifact path")
        required = item.get("required_paths", [])
        if not isinstance(required, list):
            stop(f"{name}: required_paths must be a list")
        for rel in required:
            p = Path(rel) if isinstance(rel, str) else Path("/")
            if p.is_absolute() or ".." in p.parts or not p.parts:
                stop(f"{name}: invalid required path")
        for rec in item.get("required_files", []):
            p = Path(rec.get("path", ""))
            if p.is_absolute() or ".." in p.parts or not p.parts or not SHA256.fullmatch(str(rec.get("sha256", ""))):
                stop(f"{name}: malformed required-file checksum")
    return items


def bundle_entries(base: Path, excluded: set[str]) -> list[dict]:
    entries: list[dict] = []

    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda p: p.name):
            rel = child.relative_to(base).as_posix()
            if (rel in excluded or ".git" in Path(rel).parts
                    or "__pycache__" in Path(rel).parts
                    or Path(rel).name == ".pytest_cache" or Path(rel).suffix == ".pyc"):
                continue
            st = child.lstat()
            mode = stat.S_IMODE(st.st_mode)
            if stat.S_ISDIR(st.st_mode):
                visit(child)
            elif stat.S_ISREG(st.st_mode):
                content = child.read_bytes()
                # Git stores only the executable bit for regular files. Normalize
                # group/other write bits so a clean clone matches this manifest.
                git_mode = 0o755 if st.st_mode & 0o111 else 0o644
                entries.append({"path": rel, "type": "file", "mode": git_mode,
                                "size": len(content), "sha256": sha(content)})
            elif stat.S_ISLNK(st.st_mode):
                entries.append({"path": rel, "type": "symlink", "mode": mode,
                                "target": os.readlink(child)})
            else:
                stop(f"unsupported special file in bundle: {child}")

    visit(base)
    return sorted(entries, key=lambda row: row["path"])


def verify_managed_bundle(base: Path, item: dict, expected: list[dict]) -> None:
    """Verify the locked baseline commit while allowing clean RSI promotion descendants."""
    baseline = item.get("rsi_git_baseline")
    if not baseline:
        stop(f"{item['name']}: managed Git checkout has no locked RSI baseline")
    top = git(["git", "-C", str(base), "rev-parse", "--show-toplevel"])
    if Path(top).resolve() != base.resolve():
        stop(f"{item['name']}: RSI Git repository root differs from the source bundle")
    head = git(["git", "-C", str(base), "rev-parse", "HEAD"])
    if git(["git", "-C", str(base), "cat-file", "-t", baseline]) != "commit":
        stop(f"{item['name']}: locked RSI baseline commit is missing")
    ancestor = subprocess.run(["git", "-C", str(base), "merge-base", "--is-ancestor", baseline, head],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if ancestor.returncode != 0:
        stop(f"{item['name']}: HEAD is not a descendant of its locked RSI baseline")
    status = git(["git", "-C", str(base), "status", "--porcelain=v1", "--untracked-files=all"])
    if status:
        stop(f"{item['name']}: managed RSI source checkout has uncommitted or untracked changes")

    listing = subprocess.run(["git", "-C", str(base), "ls-tree", "-r", "-z", "--full-tree", baseline],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if listing.returncode:
        stop(f"{item['name']}: could not inspect locked RSI baseline tree")
    rows = []
    oids = set()
    for raw in listing.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            header, raw_path = raw.split(b"\t", 1)
            mode, kind, oid = header.decode("ascii").split(" ")
            rel = raw_path.decode("utf-8")
        except Exception:
            stop(f"{item['name']}: malformed Git baseline tree entry")
        if kind != "blob":
            stop(f"{item['name']}: unsupported Git baseline object type")
        rows.append((mode, oid, rel))
        oids.add(oid)
    cat = subprocess.run(["git", "-C", str(base), "cat-file", "--batch"],
                         input=("\n".join(sorted(oids)) + "\n").encode(),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if cat.returncode:
        stop(f"{item['name']}: could not read locked RSI baseline blobs")
    blobs = {}
    offset = 0
    for _ in sorted(oids):
        newline = cat.stdout.find(b"\n", offset)
        if newline < 0:
            stop(f"{item['name']}: malformed Git blob batch header")
        parts = cat.stdout[offset:newline].decode("ascii").split()
        if len(parts) != 3 or parts[1] != "blob":
            stop(f"{item['name']}: unexpected object in Git baseline blob batch")
        oid, _, size_text = parts
        size = int(size_text)
        start = newline + 1
        end = start + size
        if end >= len(cat.stdout) or cat.stdout[end:end + 1] != b"\n":
            stop(f"{item['name']}: malformed Git baseline blob body")
        blobs[oid] = cat.stdout[start:end]
        offset = end + 1
    actual = []
    for mode, oid, rel in rows:
        content = blobs[oid]
        if mode == "120000":
            actual.append({"path": rel, "type": "symlink", "mode": 0o777,
                           "target": content.decode("utf-8")})
        elif mode in {"100644", "100755"}:
            actual.append({"path": rel, "type": "file", "mode": int(mode[-3:], 8),
                           "size": len(content), "sha256": sha(content)})
        else:
            stop(f"{item['name']}: unsupported file mode in Git baseline")
    actual.sort(key=lambda row: row["path"])
    # Git records executable/non-executable semantics, not group-write bits.
    # Exact mode bits were checked against the bundle manifest before RSI init.
    normalized_expected = []
    for row in expected:
        item_row = dict(row)
        if item_row.get("type") == "file":
            item_row["mode"] = 0o755 if item_row["mode"] & 0o111 else 0o644
        normalized_expected.append(item_row)
    if actual != normalized_expected:
        expected_map = {row.get("path"): row for row in normalized_expected}
        actual_map = {row.get("path"): row for row in actual}
        missing = sorted(set(expected_map) - set(actual_map))
        extra = sorted(set(actual_map) - set(expected_map))
        changed = sorted(p for p in set(expected_map) & set(actual_map) if expected_map[p] != actual_map[p])
        stop(f"{item['name']}: locked RSI Git baseline differs from source manifest "
             f"(missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]})")
    print(f"verified RSI source {item['name']}: baseline {baseline}; HEAD {head}; clean descendant")


def runtime_generated_artifacts(base: Path, item: dict) -> set[str]:
    """Exclude only declared compiler outputs, never arbitrary untracked files."""
    excluded: set[str] = set()
    source_name = item.get("name", "source")
    for rel in item.get("runtime_generated_artifacts", []):
        path = base / rel
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            stop(f"{source_name}: runtime-generated artifact is not a regular file: {rel}")
        try:
            with path.open("rb") as stream:
                magic = stream.read(4)
        except OSError as exc:
            stop(f"{source_name}: runtime-generated artifact is unreadable ({type(exc).__name__}): {rel}")
        if magic != b"\x7fELF":
            stop(f"{source_name}: runtime-generated artifact is not an ELF binary: {rel}")
        excluded.add(rel)
    return excluded


def verify_bundle(item: dict) -> None:
    name = item["name"]
    base = ROOT / item["path"]
    if base.is_symlink() or not base.is_dir():
        stop(f"bundled source is missing or redirected: {item['path']}")
    manifest_path = ROOT / item["manifest"]
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except Exception as exc:
        stop(f"{name}: bundle manifest is unreadable ({type(exc).__name__})")
    if sha(raw) != item["manifest_sha256"]:
        stop(f"{name}: manifest checksum differs from source lock")
    if (manifest.get("schema_version") != 1 or manifest.get("name") != name
            or manifest.get("source_commit") != item.get("source_commit")):
        stop(f"{name}: manifest identity or source commit differs from lock")
    expected = manifest.get("entries")
    if not isinstance(expected, list) or manifest.get("tree_sha256") != item["tree_sha256"]:
        stop(f"{name}: malformed manifest or locked tree digest mismatch")
    if sha(canonical(expected)) != manifest["tree_sha256"]:
        stop(f"{name}: manifest entries do not produce the declared tree digest")
    allowed = item.get("allowed_runtime_symlinks", {})
    excluded: set[str] = set()
    for link_rel, target_rel in allowed.items():
        link = base / link_rel
        target = ROOT / target_rel
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or not target.exists() or link.resolve() != target.resolve():
                stop(f"{name}: runtime path is occupied by an unapproved item and was left untouched: {link_rel}")
            excluded.add(link_rel)
    excluded.update(runtime_generated_artifacts(base, item))
    managed_git = (base / ".git").exists() and item.get("rsi_git_baseline")
    if managed_git:
        verify_managed_bundle(base, item, expected)
        actual = expected
    else:
        actual = bundle_entries(base, excluded)
    if actual != expected:
        expected_map = {row.get("path"): row for row in expected}
        actual_map = {row.get("path"): row for row in actual}
        missing = sorted(set(expected_map) - set(actual_map))
        extra = sorted(set(actual_map) - set(expected_map))
        changed = sorted(p for p in set(expected_map) & set(actual_map) if expected_map[p] != actual_map[p])
        stop(f"{name}: bundle differs from manifest (missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]})")
    for rel in item.get("required_paths", []):
        if not (base / rel).exists():
            stop(f"{name}: required file or directory is missing: {rel}")
    for rec in item.get("required_files", []):
        path = base / rec["path"]
        if not path.is_file() or sha(path.read_bytes()) != rec["sha256"]:
            stop(f"{name}: required-file checksum mismatch: {rec['path']}")
    print(f"verified bundle {name}: {len(actual)} entries; tree {manifest['tree_sha256']}")


def verify_git(item: dict, path: Path) -> None:
    if path.is_symlink() or not path.is_dir() or not ((path / ".git").is_dir() or (path / ".git").is_file()):
        stop(f"{item['name']}: existing path is not a Git checkout; it was left untouched: {path}")
    origin = git(["git", "-C", str(path), "remote", "get-url", "origin"])
    head = git(["git", "-C", str(path), "rev-parse", "HEAD"])
    if origin != item["url"] or head != item["commit"]:
        stop(f"{item['name']}: origin or commit differs from the lock; it was left untouched")
    status = git(["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"])
    if status:
        stop(f"{item['name']}: checkout has local changes; it was left untouched")
    sub_status = git(["git", "-C", str(path), "submodule", "status", "--recursive"])
    if item["submodules"]:
        bad = [line for line in sub_status.splitlines() if not line or line[0] in "-+U" or "-dirty" in line]
        if bad:
            stop(f"{item['name']}: submodule checkout is missing, modified, or mismatched")
    elif sub_status:
        stop(f"{item['name']}: lock says no submodules but checkout contains submodules")
    for rel in item.get("required_paths", []):
        if not (path / rel).exists():
            stop(f"{item['name']}: required file or directory is missing: {rel}")


def validate_present(items: list[dict], *, require_all: bool) -> list[dict]:
    missing = []
    for item in items:
        if item["kind"] == "bundle":
            verify_bundle(item)
            continue
        path = ROOT / item["path"]
        if path.exists() or path.is_symlink():
            verify_git(item, path)
            print(f"verified Git source {item['name']} at {item['commit']}")
        else:
            missing.append(item)
    if require_all and missing:
        stop("pinned Git sources are missing: " + ", ".join(row["name"] for row in missing))
    return missing


def materialize(item: dict) -> None:
    target = ROOT / item["path"]
    if target.exists() or target.is_symlink():
        verify_git(item, target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.partial"
    if staging.exists() or staging.is_symlink():
        stop(f"partial checkout exists and was preserved for inspection: {staging}")
    args = ["git", "clone", "--filter=blob:none", "--no-checkout", "--no-tags",
            item["url"], str(staging)]
    git(args)
    git(["git", "-C", str(staging), "fetch", "--filter=blob:none", "--depth=1",
         "origin", item["commit"]])
    git(["git", "-C", str(staging), "checkout", "--detach", item["commit"]])
    if item["submodules"]:
        git(["git", "-C", str(staging), "submodule", "update", "--init", "--recursive"])
    verify_git(item, staging)
    if target.exists() or target.is_symlink():
        stop(f"target appeared during clone; partial checkout preserved at {staging}")
    os.rename(staging, target)
    print(f"cloned {item['name']} at {item['commit']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="validate lock, bundle manifests and any existing Git checkouts")
    parser.add_argument("--verify-only", action="store_true",
                        help="require and verify every pinned source without mutation")
    parser.add_argument("--list-unresolved", action="store_true",
                        help="report bundles whose upstream Git revision is unknown")
    parser.add_argument("--only", action="append", choices=sorted(EXPECTED),
                        help="materialize/verify selected Git sources; bundle integrity is always checked")
    args = parser.parse_args()
    if sum((args.check_only, args.verify_only, args.list_unresolved)) > 1:
        parser.error("choose at most one verification mode")
    items = read_lock()
    if args.list_unresolved:
        unresolved = [item["name"] for item in items if item["kind"] == "bundle" and not item.get("source_commit")]
        if unresolved:
            print("Bundle source revision is unknown for: " + ", ".join(unresolved))
        else:
            print("All source revisions are pinned.")
        return 0
    selected_names = set(args.only) if args.only else EXPECTED
    selected = [item for item in items
                if item["kind"] == "bundle" or item["name"] in selected_names]
    missing = validate_present(selected, require_all=args.verify_only)
    if args.check_only:
        print("source lock valid; absent external Git sources are not yet materialized")
        return 0
    for item in missing:
        materialize(item)
    validate_present(selected, require_all=True)
    print("all pinned sources verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
