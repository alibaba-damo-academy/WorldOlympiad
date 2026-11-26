"""Run the expert over every case that has a scene and tabulate success.

    python -m wo.expert.regress TAG [CASE ...]      run (default: every case with a scene)
    python -m wo.expert.regress --compare A B       what changed between two tagged runs

It is the reference stage's first command (`wo.expert.run --layout droid`) run in
parallel into $WORKSPACE/tmp/ref_dev/TAG/<case>.  Use it as the regression gate for
any change under wo/expert or wo/sim: every case that passed before must still pass.
Certificates (with the planner's rejection histograms) stay in the output folders.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..scene import WORKSPACE

RUN = WORKSPACE / "outputs/wo_runs/bench_v1"
OUT = WORKSPACE / "tmp/ref_dev"


def scene_ready_cases(run_dir: Path) -> list[str]:
    cases = []
    for task in sorted((run_dir / "tasks").glob("*.json")):
        spec = json.loads(task.read_text())
        gen = spec.get("generation")
        if gen and (Path(gen) / "task_inventory_certificate.json").is_file():
            cases.append(spec["case_id"])
    return cases


def summarize(cert_path: Path) -> tuple[bool, str]:
    if not cert_path.is_file():
        return False, "no certificate (crash)"
    cert = json.loads(cert_path.read_text())
    if cert.get("success"):
        return True, ""
    last = cert["steps"][-1] if cert.get("steps") else {}
    return False, f"{last.get('stage')}: {last.get('detail')}"


def compare(a: dict, b: dict) -> dict:
    """Cases whose outcome differs between two summaries ({case: {"success": bool, ...}})."""
    fixed = sorted(c for c in b if b[c]["success"] and c in a and not a[c]["success"])
    broken = sorted(c for c in b if not b[c]["success"] and c in a and a[c]["success"])
    new = sorted(c for c in b if c not in a)
    return {"fixed": fixed, "broken": broken, "new": new}


def run_one(case: str, out_root: Path, run_dir: Path, repo: Path) -> tuple[str, bool, str, float]:
    out = out_root / case
    env = dict(os.environ, PYTHONPATH=str(repo), TMPDIR=str(WORKSPACE / "tmp/jobs"),
               __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.log").open("w") as log:
        proc = subprocess.run([sys.executable, "-m", "wo.expert.run", "--task", str(run_dir / "tasks" / f"{case}.json"),
                               "--out", str(out), "--layout", "droid"], cwd=repo, env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    # A previous certificate must never conceal a new process crash.
    if proc.returncode:
        return case, False, f"expert process exited {proc.returncode}; see {out / 'run.log'}", time.time() - t0
    ok, why = summarize(out / "certificate.json")
    return case, ok, why, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?")
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()

    if a.compare:
        sums = [json.loads((OUT / t / "summary.json").read_text()) for t in a.compare]
        diff = compare(*sums)
        for k, v in diff.items():
            print(f"{k}: {v}")
        sys.exit(1 if diff["broken"] else 0)
    if not a.tag:
        ap.error("a TAG is required")

    repo = Path(__file__).resolve().parents[2]
    cases = scene_ready_cases(RUN)
    if a.cases:
        cases = [c for c in cases if c in a.cases]
    out_root = OUT / a.tag
    out_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        results = list(pool.map(lambda c: run_one(c, out_root, RUN, repo), cases))
    print(f"{a.tag}: {sum(r[1] for r in results)}/{len(results)} succeed")
    for case, ok, why, dt in results:
        print(f"  {'OK  ' if ok else 'FAIL'} {case:48s} {dt:5.0f}s  {why}")
    (out_root / "summary.json").write_text(json.dumps(
        {c: {"success": ok, "why": why} for c, ok, why, _ in results}, indent=1))


if __name__ == "__main__":
    main()
