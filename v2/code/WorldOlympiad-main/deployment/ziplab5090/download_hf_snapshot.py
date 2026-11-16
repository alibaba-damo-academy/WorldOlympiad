#!/usr/bin/env python3
"""Download a public Hugging Face snapshot without storing it in Git.

The standard Hub/Xet downloader can stall behind some datacenter proxies. This
small resumable downloader reads the official repository file manifest, then
fetches independent files in parallel from a configurable Hub-compatible
endpoint. Every file is size-checked before it is atomically installed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TRANSFER_START_LOCK = threading.Lock()
_TRANSFER_START_LAST = 0.0


def _pace_transfer_start(spacing_seconds: float = 4.0) -> None:
    """Avoid overwhelming a single local HTTP proxy with simultaneous TLS handshakes."""
    global _TRANSFER_START_LAST
    with _TRANSFER_START_LOCK:
        wait_seconds = spacing_seconds - (time.monotonic() - _TRANSFER_START_LAST)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _TRANSFER_START_LAST = time.monotonic()


def _api_url(endpoint: str, repo_id: str, repo_type: str, revision: str) -> str:
    namespace = "models" if repo_type == "model" else "datasets"
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    query = urllib.parse.urlencode({"revision": revision, "blobs": "true"})
    return f"{endpoint.rstrip('/')}/api/{namespace}/{encoded_repo}?{query}"


def _parse_manifest(
    payload: dict[str, Any], repo_id: str, revision: str
) -> tuple[list[dict[str, Any]], str]:
    siblings = payload.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise RuntimeError(f"Hub API returned no files for {repo_id}@{revision}")
    files: list[dict[str, Any]] = []
    for item in siblings:
        relative = item.get("rfilename")
        size = item.get("size")
        if not isinstance(relative, str) or not relative:
            continue
        if not isinstance(size, int) or size < 0:
            raise RuntimeError(
                f"missing blob size for {relative!r}; use a Hub API manifest with blobs=true"
            )
        lfs = item.get("lfs")
        sha256 = lfs.get("sha256") if isinstance(lfs, dict) else None
        if sha256 is not None and (not isinstance(sha256, str) or len(sha256) != 64):
            raise RuntimeError(f"invalid LFS SHA-256 for {relative!r}")
        files.append({"path": relative, "size": size, "sha256": sha256})
    return files, str(payload.get("sha") or revision)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(
    endpoint: str, repo_id: str, repo_type: str, revision: str
) -> tuple[list[dict[str, Any]], str]:
    request = urllib.request.Request(
        _api_url(endpoint, repo_id, repo_type, revision),
        headers={"User-Agent": "worldolympiad-snapshot-downloader/1.0"},
    )
    last_error: BaseException | None = None
    for attempt in range(1, 21):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
            break
        except BaseException as error:
            last_error = error
            if attempt == 20:
                raise RuntimeError("failed to fetch Hub manifest after 20 attempts") from error
            print(f"retry Hub manifest: attempt={attempt}/20, error={error}", flush=True)
            time.sleep(min(30, attempt * 2))
    else:  # pragma: no cover - defensive; loop either breaks or raises.
        raise RuntimeError("failed to fetch Hub manifest") from last_error
    return _parse_manifest(payload, repo_id, revision)


def _download_one(
    item: dict[str, Any],
    *,
    repo_id: str,
    revision: str,
    local_dir: Path,
    download_endpoint: str,
    max_attempts: int,
    verify_sha256: bool,
) -> tuple[str, int, str]:
    relative = str(item["path"])
    expected_size = int(item["size"])
    expected_sha256 = item.get("sha256")
    destination = local_dir / relative
    partial = destination.with_name(destination.name + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size == expected_size:
        if not verify_sha256 or expected_sha256 is None or _sha256(destination) == expected_sha256:
            status = "cached_sha256_verified" if verify_sha256 and expected_sha256 else "cached_size_verified"
            return relative, expected_size, status
        destination.replace(destination.with_name(destination.name + ".sha256_mismatch"))
    if destination.exists():
        destination.replace(destination.with_name(destination.name + ".invalid"))
    if partial.exists() and partial.stat().st_size > expected_size:
        partial.replace(partial.with_name(partial.name + ".oversize"))

    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    encoded_path = urllib.parse.quote(relative, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    url = f"{download_endpoint.rstrip('/')}/{encoded_repo}/resolve/{encoded_revision}/{encoded_path}"
    # Hub mirrors redirect large Xet blobs to short-lived signed URLs.  Some
    # datacenter routes return HTTP 408 part-way through a multi-GB range.  A
    # fresh request to the stable resolve URL yields a new signed range URL,
    # and curl --continue-at resumes from the already verified byte count.
    last_error: int | None = None
    for attempt in range(1, max_attempts + 1):
        before = partial.stat().st_size if partial.exists() else 0
        command = [
            "curl",
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--connect-timeout",
            "30",
            "--speed-time",
            "180",
            "--speed-limit",
            "1024",
            "--max-time",
            "600",
            "--silent",
            "--show-error",
            "--output",
            str(partial),
            url,
        ]
        _pace_transfer_start()
        completed = subprocess.run(command, check=False)
        last_error = completed.returncode
        after = partial.stat().st_size if partial.exists() else 0
        if after == expected_size:
            break
        if after > expected_size:
            partial.replace(partial.with_name(partial.name + ".oversize"))
            raise RuntimeError(f"download exceeded manifest size for {relative}")
        print(
            f"retry {relative}: attempt={attempt}/{max_attempts}, "
            f"bytes={after}/{expected_size}, curl_rc={last_error}",
            flush=True,
        )
        time.sleep(1 if after > before else min(30, attempt * 2))
    else:
        raise RuntimeError(
            f"download retries exhausted for {relative}: "
            f"bytes={partial.stat().st_size if partial.exists() else 0}/{expected_size}, "
            f"last curl rc={last_error}"
        )
    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"size mismatch for {relative}: expected {expected_size}, got {actual_size}")
    if verify_sha256 and expected_sha256 is not None:
        actual_sha256 = _sha256(partial)
        if actual_sha256 != expected_sha256:
            partial.replace(partial.with_name(partial.name + ".sha256_mismatch"))
            raise RuntimeError(
                f"SHA-256 mismatch for {relative}: expected {expected_sha256}, got {actual_sha256}"
            )
    os.replace(partial, destination)
    return relative, expected_size, "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--api-endpoint", default="https://huggingface.co")
    parser.add_argument(
        "--manifest-path",
        help="Optional cached Hub API JSON (from an endpoint queried with blobs=true)",
    )
    parser.add_argument("--download-endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-file-attempts", type=int, default=128)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Optional Hub path glob to download; repeat for multiple patterns",
    )
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Verify every LFS file against the SHA-256 published by the official Hub API",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_file_attempts < 1:
        parser.error("--max-file-attempts must be at least 1")

    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest_path:
        payload = json.loads(Path(args.manifest_path).read_text(encoding="utf-8"))
        files, resolved_revision = _parse_manifest(payload, args.repo_id, args.revision)
    else:
        files, resolved_revision = _load_manifest(
            args.api_endpoint, args.repo_id, args.repo_type, args.revision
        )
    if args.include:
        files = [
            item
            for item in files
            if any(fnmatch.fnmatchcase(str(item["path"]), pattern) for pattern in args.include)
        ]
        if not files:
            parser.error(f"--include patterns matched no Hub files: {args.include}")
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
                "download_endpoint": args.download_endpoint,
                "verify_lfs_sha256": bool(args.verify_sha256),
                "include": list(args.include),
            },
            indent=2,
        ),
        flush=True,
    )

    completed_bytes = 0
    completed_files = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _download_one,
                item,
                repo_id=args.repo_id,
                revision=args.revision,
                local_dir=local_dir,
                download_endpoint=args.download_endpoint,
                max_attempts=args.max_file_attempts,
                verify_sha256=args.verify_sha256,
            ): item
            for item in files
        }
        try:
            for future in concurrent.futures.as_completed(futures):
                relative, size, status = future.result()
                completed_files += 1
                completed_bytes += size
                print(
                    f"[{completed_files}/{len(files)}] {status}: {relative} "
                    f"({completed_bytes / max(total_bytes, 1):.1%} by bytes)",
                    flush=True,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    for item in files:
        output = local_dir / str(item["path"])
        if not output.is_file() or output.stat().st_size != int(item["size"]):
            raise RuntimeError(f"final snapshot validation failed for {output}")
    snapshot_metadata = {
        "provider": "huggingface",
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "files": len(files),
        "total_bytes": total_bytes,
        "lfs_sha256_verified": bool(args.verify_sha256),
        "include": list(args.include),
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (local_dir / ".worldolympiad_snapshot.json").write_text(
        json.dumps(snapshot_metadata, indent=2), encoding="utf-8"
    )
    print(f"snapshot complete: {args.repo_id}@{args.revision}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
