#!/usr/bin/env python3
"""Create and verify the deterministic local Git base required by RSI worktrees."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "sources.lock.json"
NAME = "WorldOlympiad-main"


def run(args: list[str], *, cwd: Path | None = None, env: dict | None = None,
        input: bytes | None = None) -> bytes:
    result = subprocess.run(args, cwd=cwd, env=env, input=input,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(detail or f"command failed: {args[0]}")
    return result.stdout


def load_world() -> tuple[dict, Path, dict]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    item = next(row for row in lock["sources"] if row["name"] == NAME)
    base = ROOT / item["path"]
    manifest = json.loads((ROOT / item["manifest"]).read_text(encoding="utf-8"))
    return lock, base, manifest


def install_excludes(base: Path, links: dict[str, str]) -> None:
    exclude = base / ".git/info/exclude"
    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    desired = ["/outputs", "/weights", "/Depth-Anything-3"]
    lines = existing.splitlines()
    changed = False
    for line in desired:
        if line not in lines:
            lines.append(line)
            changed = True
    if changed:
        exclude.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_output_link(worktree: Path, output_target: Path) -> None:
    output_link = worktree / "outputs"
    if output_link.exists() or output_link.is_symlink():
        if not output_link.is_symlink() or output_link.resolve(strict=False) != output_target.resolve(strict=False):
            raise RuntimeError(f"RSI output path is occupied; left untouched: {output_link}")
        output_target.mkdir(parents=True, exist_ok=True)
        if output_link.resolve(strict=True) != output_target.resolve(strict=True):
            raise RuntimeError(f"RSI output link resolves unexpectedly: {output_link}")
    else:
        output_target.mkdir(parents=True, exist_ok=True)
        output_link.symlink_to(output_target, target_is_directory=True)


def initialize(lock: dict, item: dict, base: Path, manifest: dict) -> str:
    if (base / ".git").exists():
        raise RuntimeError("WorldOlympiad source already has Git metadata but no matching locked RSI baseline; left untouched")
    check = run([sys.executable, str(ROOT / "scripts/setup_sources.py"), "--check-only"])
    if check:
        print(check.decode(errors="replace"), end="")
    run(["git", "init", "--quiet", "-b", "main"], cwd=base)
    install_excludes(base, item.get("allowed_runtime_symlinks", {}))
    entries = manifest["entries"]
    pathspecs = b"".join((b":(literal)" + row["path"].encode("utf-8") + b"\0") for row in entries)
    run(["git", "add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul"],
        cwd=base, input=pathspecs)
    tree = run(["git", "write-tree"], cwd=base).decode().strip()
    env = os.environ.copy()
    env.update({
        "GIT_AUTHOR_NAME": "WOV2 locked source snapshot",
        "GIT_AUTHOR_EMAIL": "wov2-source@localhost",
        "GIT_COMMITTER_NAME": "WOV2 locked source snapshot",
        "GIT_COMMITTER_EMAIL": "wov2-source@localhost",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
    })
    baseline_label = item.get("source_commit") or "unattributed-recovered-bundle"
    message = (f"WOV2 deterministic RSI baseline: {baseline_label}\n\n"
               f"Locked source tree SHA-256: {item['tree_sha256']}\n")
    commit = run(["git", "commit-tree", tree, "-m", message], cwd=base, env=env).decode().strip()
    run(["git", "update-ref", "refs/heads/main", commit], cwd=base)
    run(["git", "symbolic-ref", "HEAD", "refs/heads/main"], cwd=base)
    item["rsi_git_baseline"] = commit
    LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return commit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="verify an existing baseline without creating Git metadata")
    args = parser.parse_args()
    lock, base, manifest = load_world()
    item = next(row for row in lock["sources"] if row["name"] == NAME)
    if not (base / ".git").exists():
        if args.check_only:
            raise RuntimeError("RSI Git metadata is not prepared")
        prior = item.get("rsi_git_baseline")
        commit = initialize(lock, item, base, manifest)
        if prior:
            print("nested RSI Git metadata was absent; recreated a deterministic local baseline from the verified source manifest")
        print(f"created deterministic RSI Git baseline {commit}")
    else:
        if not item.get("rsi_git_baseline"):
            raise RuntimeError("unlocked Git metadata exists; refusing to adopt or modify it")
        commit = item["rsi_git_baseline"]
    check = run([sys.executable, str(ROOT / "scripts/setup_sources.py"), "--check-only"])
    print(check.decode(errors="replace"), end="")
    if not args.check_only:
        # Create/reuse RSI's worktree now, before the API-driven task starts.
        worktree = ROOT / "code/wo-agent-worktree"
        install_excludes(base, item.get("allowed_runtime_symlinks", {}))
        if not (worktree.exists() or worktree.is_symlink()):
            run(["git", "-c", "user.name=wo-agent", "-c", "user.email=agent@localhost",
                 "worktree", "add", "-B", "agent/dev", str(worktree), "HEAD"], cwd=base)
        elif worktree.is_symlink() or not worktree.is_dir():
            raise RuntimeError(f"RSI worktree path is occupied; left untouched: {worktree}")
        worktree_top = Path(run(["git", "-C", str(worktree), "rev-parse", "--show-toplevel"]).decode().strip()).resolve()
        main_common = Path(run(["git", "-C", str(base), "rev-parse", "--git-common-dir"]).decode().strip())
        work_common = Path(run(["git", "-C", str(worktree), "rev-parse", "--git-common-dir"]).decode().strip())
        if not main_common.is_absolute(): main_common = (base / main_common).resolve()
        if not work_common.is_absolute(): work_common = (worktree / work_common).resolve()
        if worktree_top != worktree.resolve() or work_common != main_common:
            raise RuntimeError("existing RSI worktree is not attached to this locked source repository")
        branch = run(["git", "-C", str(worktree), "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        if branch != "agent/dev":
            raise RuntimeError(f"existing RSI worktree is on unexpected branch {branch!r}; left untouched")
        head = run(["git", "-C", str(worktree), "rev-parse", "HEAD"]).decode().strip()
        ancestry = subprocess.run(["git", "-C", str(base), "merge-base", "--is-ancestor", commit, head],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if ancestry.returncode:
            raise RuntimeError("existing RSI worktree is not based on the locked source baseline")
        for name in item.get("allowed_runtime_symlinks", {}):
            source = base / name
            if not (source.exists() or source.is_symlink()):
                continue
            target = source.resolve(strict=True)
            destination = worktree / name
            if destination.exists() or destination.is_symlink():
                if not destination.is_symlink() or destination.resolve(strict=True) != target:
                    raise RuntimeError(f"RSI runtime link path is occupied; left untouched: {destination}")
            else:
                destination.symlink_to(target, target_is_directory=True)
        output_target = ROOT / "outputs"
        ensure_output_link(worktree, output_target)
        status = run(["git", "-C", str(worktree), "status", "--porcelain=v1", "--untracked-files=all"])
        if status.strip():
            raise RuntimeError("RSI worktree has uncommitted/untracked changes; preserved without cleanup")
        print(f"prepared RSI worktree {worktree}; HEAD {head}; source baseline {commit}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        raise SystemExit(f"RSI source preparation blocked: {type(exc).__name__}: {exc}")
