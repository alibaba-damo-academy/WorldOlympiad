#!/usr/bin/env python3
"""Resumably download and SHA-256 verify a public ModelScope snapshot."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _list_directory(repo_id: str, revision: str, root: str) -> list[dict[str, Any]]:
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    query = urllib.parse.urlencode({"Revision": revision, "Root": root})
    url = f"https://www.modelscope.cn/api/v1/models/{encoded_repo}/repo/files?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "worldolympiad-modelscope-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if payload.get("Code") != 200:
        raise RuntimeError(f"ModelScope API error for {root!r}: {payload}")
    return list(payload["Data"]["Files"])


def _manifest(repo_id: str, revision: str) -> tuple[list[dict[str, Any]], str]:
    pending = [""]
    files: list[dict[str, Any]] = []
    resolved_revisions: set[str] = set()
    while pending:
        root = pending.pop()
        for item in _list_directory(repo_id, revision, root):
            if item.get("Revision"):
                resolved_revisions.add(str(item["Revision"]))
            if item.get("Type") == "tree":
                pending.append(str(item["Path"]))
            else:
                files.append(
                    {
                        "path": str(item["Path"]),
                        "size": int(item["Size"]),
                        "sha256": str(item.get("Sha256") or ""),
                    }
                )
    resolved_revision = sorted(resolved_revisions)[0] if len(resolved_revisions) == 1 else revision
    return sorted(files, key=lambda item: item["path"]), resolved_revision


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, size: int, sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    return not sha256 or _sha256(path) == sha256


def _download_one(
    item: dict[str, Any], *, repo_id: str, revision: str, local_dir: Path
) -> tuple[str, int, str]:
    relative = str(item["path"])
    size = int(item["size"])
    sha256 = str(item["sha256"])
    destination = local_dir / relative
    partial = destination.with_name(destination.name + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _verified(destination, size, sha256):
        return relative, size, "cached"
    if destination.exists():
        destination.replace(destination.with_name(destination.name + ".invalid"))
    if partial.exists() and partial.stat().st_size > size:
        partial.replace(partial.with_name(partial.name + ".oversize"))
    if _verified(partial, size, sha256):
        os.replace(partial, destination)
        return relative, size, "resumed-and-verified"

    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    encoded_path = urllib.parse.quote(relative, safe="/")
    url = f"https://www.modelscope.cn/models/{encoded_repo}/resolve/{encoded_revision}/{encoded_path}"
    subprocess.run(
        [
            "wget",
            "--no-proxy",
            "--continue",
            "--timeout=30",
            "--read-timeout=120",
            "--tries=0",
            "--retry-connrefused",
            "--no-verbose",
            "--output-document",
            str(partial),
            url,
        ],
        check=True,
    )
    if partial.stat().st_size != size:
        raise RuntimeError(f"size mismatch for {relative}: expected {size}, got {partial.stat().st_size}")
    actual_sha256 = _sha256(partial)
    if sha256 and actual_sha256 != sha256:
        mismatch = partial.with_name(partial.name + ".checksum_mismatch")
        partial.replace(mismatch)
        raise RuntimeError(f"SHA-256 mismatch for {relative}; preserved as {mismatch}")
    os.replace(partial, destination)
    return relative, size, "downloaded-and-verified"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")

    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    files, resolved_revision = _manifest(args.repo_id, args.revision)
    total_bytes = sum(int(item["size"]) for item in files)
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "resolved_revision": resolved_revision,
                "files": len(files),
                "total_bytes": total_bytes,
                "local_dir": str(local_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    completed_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _download_one,
                item,
                repo_id=args.repo_id,
                revision=args.revision,
                local_dir=local_dir,
            ): item
            for item in files
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            relative, size, status = future.result()
            completed_bytes += size
            print(
                f"[{index}/{len(files)}] {status}: {relative} "
                f"({completed_bytes / max(total_bytes, 1):.1%} by bytes)",
                flush=True,
            )
    snapshot_metadata = {
        "provider": "modelscope",
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "files": len(files),
        "total_bytes": total_bytes,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation": "size_and_sha256",
    }
    (local_dir / ".worldolympiad_snapshot.json").write_text(
        json.dumps(snapshot_metadata, indent=2), encoding="utf-8"
    )
    print(f"snapshot complete: {args.repo_id}@{args.revision}", flush=True)


if __name__ == "__main__":
    main()
