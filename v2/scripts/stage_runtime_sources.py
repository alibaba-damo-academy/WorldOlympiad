#!/usr/bin/env python3
"""Stage or verify pinned, patchable runtime copies without changing source checkouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "code/WorldOlympiad-main"
LOCK = json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))
BY_NAME = {item["name"]: item for item in LOCK["sources"]}
RECEIPT = ".wov2-evaluator-runtime.json"


def stop(message: str) -> None:
    raise SystemExit(f"runtime staging blocked: {message}")


def call(args: list[str], *, cwd: Path | None = None) -> str:
    p = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, check=False)
    if p.returncode:
        stop(p.stderr.strip() or "staging command failed")
    return p.stdout


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def world_patch_path(rel: str) -> Path:
    """Resolve locked patch paths against the source bundle's real layout."""
    candidates = [WORLD / "deployment/ziplab5090" / rel,
                  WORLD / "deployment" / rel]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    stop(f"locked patch is missing from the source bundle: {rel}")


def git_status(repo: Path) -> list[str]:
    return call(["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"]).splitlines()


def diff_bytes(repo: Path) -> bytes:
    return call(["git", "-C", str(repo), "diff", "--binary", "HEAD"]).encode()


def diff_hash(repo: Path) -> str:
    return sha(diff_bytes(repo))


def content_entries(base: Path, excluded: set[str] | None = None) -> list[dict]:
    excluded = excluded or set()
    entries: list[dict] = []
    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda p: p.name):
            rel = child.relative_to(base).as_posix()
            if (rel in excluded or "__pycache__" in Path(rel).parts
                    or Path(rel).name == ".pytest_cache" or Path(rel).suffix == ".pyc"):
                continue
            st = child.lstat()
            mode = stat.S_IMODE(st.st_mode)
            if stat.S_ISDIR(st.st_mode):
                visit(child)
            elif stat.S_ISREG(st.st_mode):
                payload = child.read_bytes()
                entries.append({"path": rel, "type": "file", "mode": mode,
                                "size": len(payload), "sha256": sha(payload)})
            elif stat.S_ISLNK(st.st_mode):
                entries.append({"path": rel, "type": "symlink", "mode": mode,
                                "target": os.readlink(child)})
            else:
                stop(f"unsupported special file in runtime overlay: {child}")
    visit(base)
    return sorted(entries, key=lambda row: row["path"])


def tree_hash(base: Path, excluded: set[str] | None = None) -> str:
    payload = json.dumps(content_entries(base, excluded), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha(payload)


def validate_bundle_runtime(name: str, target: Path, patches: list[str]) -> None:
    item = BY_NAME[name]
    if target.is_symlink() or not target.is_dir():
        stop(f"existing bundle runtime is not a directory: {target.relative_to(ROOT)}")
    receipt_path = target / RECEIPT
    if not receipt_path.is_file():
        stop(f"bundle runtime has no provenance receipt: {target.relative_to(ROOT)}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        stop(f"bundle runtime receipt is unreadable ({type(exc).__name__}); left untouched")
    patch_hashes = {rel: sha((world_patch_path(rel)).read_bytes()) for rel in patches}
    source_manifest = ROOT / item["manifest"]
    manifest_hash = sha(source_manifest.read_bytes())
    actual_tree = tree_hash(target, {RECEIPT})
    if (receipt.get("schema_version") != 1 or receipt.get("source") != name
            or receipt.get("source_manifest_sha256") != manifest_hash
            or receipt.get("source_tree_sha256") != item["tree_sha256"]
            or receipt.get("patches") != patch_hashes
            or receipt.get("runtime_tree_sha256") != actual_tree):
        stop(f"bundle runtime provenance or content differs at {target.relative_to(ROOT)}; left untouched")
    for rel in item.get("required_paths", []):
        if not (target / rel).exists():
            stop(f"bundle runtime required path is missing: {rel}")
    for rel in item.get("runtime_required_paths", []):
        if not (target / rel).exists():
            stop(f"bundle runtime required asset path is missing: {rel}")
    for record in item.get("required_files", []):
        path = target / record["path"]
        if not path.is_file() or sha(path.read_bytes()) != record["sha256"]:
            stop(f"bundle runtime required file is missing or changed: {record['path']}")


def stage_bundle(name: str, dest_rel: str, patches: list[str], verify_only: bool) -> None:
    item = BY_NAME[name]
    source = ROOT / item["path"]
    target = ROOT / dest_rel
    if target.exists() or target.is_symlink():
        validate_bundle_runtime(name, target, patches)
        print(f"verified bundled runtime overlay {dest_rel}")
        return
    if verify_only:
        stop(f"required bundled runtime overlay is missing: {dest_rel}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_manifest_hash = sha((ROOT / item["manifest"]).read_bytes())
    staging = target.parent / f".{target.name}.partial.{source_manifest_hash[:12]}.{time.time_ns()}"
    if staging.exists() or staging.is_symlink():
        stop(f"partial bundle runtime exists and was preserved: {staging.relative_to(ROOT)}")
    if name == "RoboTwin":
        data_root = ROOT / "runtime/robotwin_assets"
        required_data = ["embodiments/aloha-agilex/config.yml", "objects/objaverse/list.json"]
        missing = [rel for rel in required_data if not (data_root / rel).is_file()]
        if missing:
            stop("reused RoboTwin data assets are missing under runtime/robotwin_assets: " + ", ".join(missing))
    shutil.copytree(source, staging, symlinks=True, copy_function=shutil.copy2)
    if name == "RoboTwin":
        asset_root = staging / "assets"
        for part in ("embodiments", "objects"):
            link = asset_root / part
            if link.exists() or link.is_symlink():
                stop(f"RoboTwin asset target already exists in staged source: {link.relative_to(ROOT)}")
            os.symlink(f"../../robotwin_assets/{part}", link)
    patch_hashes = {}
    for rel in patches:
        patch_path = world_patch_path(rel)
        if not patch_path.is_file():
            stop(f"locked patch is missing: {rel}")
        check = subprocess.run(["patch", "--dry-run", "--forward", "--batch", "-p1", "-d", str(staging)],
                               input=patch_path.read_bytes(), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
        if check.returncode:
            stop(f"bundle patch is not applicable to pristine source {name}: {rel}: "
                 + check.stdout.decode(errors="replace")[-1200:])
        apply = subprocess.run(["patch", "--forward", "--batch", "-p1", "-d", str(staging)],
                               input=patch_path.read_bytes(), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
        if apply.returncode:
            stop(f"bundle patch application failed for {name}: {rel}: "
                 + apply.stdout.decode(errors="replace")[-1200:])
        patch_hashes[rel] = sha(patch_path.read_bytes())
    receipt = {"schema_version": 1, "source": name,
               "source_manifest_sha256": sha((ROOT / item["manifest"]).read_bytes()),
               "source_tree_sha256": item["tree_sha256"], "patches": patch_hashes,
               "runtime_tree_sha256": tree_hash(staging)}
    (staging / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.chmod(staging / RECEIPT, 0o600)
    validate_bundle_runtime(name, staging, patches)
    if target.exists() or target.is_symlink():
        stop(f"target appeared during bundle staging; partial preserved at {staging.relative_to(ROOT)}")
    os.rename(staging, target)
    print(f"staged bundled runtime overlay {dest_rel}")


def validate_status(repo: Path) -> None:
    for line in git_status(repo):
        if line == f"?? {RECEIPT}":
            continue
        if line.startswith("?? "):
            stop(f"unexpected untracked runtime file: {line[3:]}")
        if line[0] != " " or line[1] not in {"M", "D", "A"}:
            stop(f"unexpected staged or conflicted runtime change: {line}")
    submodules = call(["git", "-C", str(repo), "submodule", "status", "--recursive"]).splitlines()
    if any(line.startswith(("-", "+", "U")) or "-dirty" in line for line in submodules):
        stop("runtime overlay has an uninitialized, changed, or dirty submodule")


def patch_one(repo: Path, rel: str) -> str:
    patch = world_patch_path(rel)
    if not patch.is_file():
        stop(f"locked patch is missing: {rel}")
    check = subprocess.run(["git", "-C", str(repo), "apply", "--check", "--whitespace=error", str(patch)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check.returncode == 0:
        call(["git", "-C", str(repo), "apply", "--whitespace=error", str(patch)])
    else:
        reverse = subprocess.run(["git", "-C", str(repo), "apply", "--reverse", "--check", "--whitespace=error", str(patch)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if reverse.returncode:
            stop(f"patch {rel} matches neither pristine nor already-patched source")
    return sha(patch.read_bytes())


def validate_existing(name: str, target: Path, patches: list[str]) -> None:
    item = BY_NAME[name]
    if target.is_symlink() or not target.is_dir() or not (target / ".git").exists():
        stop(f"existing runtime path is not an owned Git checkout: {target.relative_to(ROOT)}")
    head = call(["git", "-C", str(target), "rev-parse", "HEAD"]).strip()
    if head != item["commit"]:
        stop(f"runtime revision differs at {target.relative_to(ROOT)}; left untouched")
    receipt_path = target / RECEIPT
    if not receipt_path.is_file():
        stop(f"runtime checkout has no provenance receipt: {target.relative_to(ROOT)}")
    try:
        saved = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        stop(f"runtime receipt is unreadable ({type(exc).__name__}); checkout left untouched")
    patch_hashes = {rel: sha((world_patch_path(rel)).read_bytes()) for rel in patches}
    current_diff = diff_hash(target)
    if (saved.get("schema_version") != 1 or saved.get("source") != name
            or saved.get("commit") != head or saved.get("url") != item["url"]
            or saved.get("patches") != patch_hashes
            or saved.get("runtime_diff_sha256") != current_diff):
        stop(f"runtime provenance or patched diff differs at {target.relative_to(ROOT)}; left untouched")
    validate_status(target)


def stage(name: str, dest_rel: str, patches: list[str], verify_only: bool) -> None:
    item = BY_NAME[name]
    if item.get("kind") == "bundle":
        stage_bundle(name, dest_rel, patches, verify_only)
        return
    source = ROOT / item["path"]
    target = ROOT / dest_rel
    if target.exists() or target.is_symlink():
        validate_existing(name, target, patches)
        print(f"verified runtime overlay {dest_rel} at {item['commit']}")
        return
    if verify_only:
        stop(f"required runtime overlay is missing: {dest_rel}")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.partial"
    if staging.exists() or staging.is_symlink():
        stop(f"partial runtime path exists and was preserved: {staging.relative_to(ROOT)}")
    clone = subprocess.run(["git", "clone", "--shared", "--no-checkout", str(source), str(staging)],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if clone.returncode:
        stop(clone.stderr.strip() or f"cannot stage {name}")
    call(["git", "-C", str(staging), "checkout", "--detach", item["commit"]])
    if item.get("submodules"):
        call(["git", "-C", str(staging), "submodule", "update", "--init", "--recursive"])
    patch_hashes = {rel: patch_one(staging, rel) for rel in patches}
    receipt = {"schema_version": 1, "source": name, "url": item["url"],
               "commit": item["commit"], "patches": patch_hashes,
               "runtime_diff_sha256": diff_hash(staging)}
    (staging / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    validate_existing(name, staging, patches)
    if target.exists() or target.is_symlink():
        stop(f"target appeared during staging; partial checkout preserved: {staging.relative_to(ROOT)}")
    os.rename(staging, target)
    print(f"staged runtime overlay {dest_rel} at {item['commit']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="verify overlays without creating them")
    args = parser.parse_args()
    stage("RoboTwin", "runtime/RoboTwin", [
        "patches/robotwin_lingbot_native_camera.patch",
        "patches/robotwin_lingbot_upstream_intrinsics.patch",
    ], args.verify_only)
    stage("LingBot-VA", "runtime/lingbot-va", ["lingbot_torch_attention.patch"], args.verify_only)
    stage("scenecode_pybullet_minimal", "runtime/scenecode_pybullet_minimal", [], args.verify_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
