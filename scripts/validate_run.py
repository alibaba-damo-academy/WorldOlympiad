#!/usr/bin/env python3
"""Validate fresh-run or reference-continuation state without changing artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "code/WorldOlympiad-main"
sys.path.insert(0, str(WORLD))
RUNS = ROOT / "outputs/wo_runs"
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def fail(message: str) -> None:
    raise SystemExit(f"run gate blocked: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fresh", "continue"))
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    if not SAFE.fullmatch(args.run): fail("unsafe run id")
    root = RUNS / args.run
    if args.phase == "fresh":
        if root.exists(): fail(f"run directory already exists and was preserved: {root}")
        print("PASS: run id is fresh")
        return 0

    task_files = sorted((root / "tasks").glob("*.json")) if (root / "tasks").is_dir() else []
    if not task_files:
        fail("continuation requires an authored task with a verified scene and expert reference")

    ready = []
    for task_path in task_files:
        case = task_path.stem
        try:
            spec = json.loads(task_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        spec_case = spec.get("case_id", spec.get("task_id", spec.get("id")))
        if spec_case and spec_case != case:
            continue

        generation_value = spec.get("generation")
        if not isinstance(generation_value, str) or not generation_value:
            continue
        generation = Path(generation_value)
        if not generation.is_absolute():
            generation = (ROOT / generation).resolve()
        inventory = generation / "task_inventory_certificate.json"
        manifest_path = generation / "generation_manifest.json"
        if not inventory.is_file() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("passed") is not True or manifest.get("case_id") != case:
            continue

        reference = root / "references" / case
        cert_path = reference / "certificate.json"
        try:
            cert = json.loads(cert_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        task_hash = hashlib.sha256(task_path.read_bytes()).hexdigest()
        trajectory = reference / "trajectory.json"
        required_video = reference / "replay/exterior_left.mp4"
        if (cert.get("success") is not True
                or cert.get("case_id") != case
                or cert.get("task_sha256") != task_hash
                or not trajectory.is_file() or trajectory.stat().st_size == 0
                or not required_video.is_file() or required_video.stat().st_size == 0):
            continue
        ready.append(case)

    if len(ready) != 1:
        fail("continuation requires exactly one task bound to a passed scene and successful expert reference; "
             f"found {len(ready)} reference-ready tasks among {len(task_files)} authored tasks")
    case = ready[0]
    db = ROOT / "outputs/wo_agent/jobs.sqlite3"
    if not db.is_file(): fail("job database is missing; cannot prove the reference stage completed")
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=3)
        rows = con.execute("select kind,status,args from jobs order by id").fetchall()
    except sqlite3.Error as exc: fail(f"cannot inspect stage receipts ({type(exc).__name__})")
    finally:
        try: con.close()
        except Exception: pass
    matched = []
    for kind,status,raw in rows:
        try: item=json.loads(raw)
        except Exception: continue
        if item.get("run")==args.run and item.get("case")==case:
            matched.append((kind,status))
    if ("reference","done") not in matched:
        try:
            from wo.agent.frozen_import import verify_frozen_import
            verify_frozen_import(root, case)
        except Exception:
            fail("job database has no completed in-run reference stage and the frozen-reference import receipt is not valid")
    print(json.dumps({"run": args.run, "case": case, "reference": "verified",
                      "existing_episode_jobs": sum(1 for k,_ in matched if k=="episode"),
                      "existing_score_jobs": sum(1 for k,_ in matched if k=="score")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
