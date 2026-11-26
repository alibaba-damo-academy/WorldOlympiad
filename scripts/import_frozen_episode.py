#!/usr/bin/env python3
"""Create a new report/scoring run around a completed frozen native episode."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/WorldOlympiad-main"))
from wo.agent.frozen_import import create_phase1_import


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    receipt = create_phase1_import(ROOT, args.run, args.source_run, args.case, args.model, args.seed)
    print(json.dumps({"run": args.run, "case": args.case, "model": args.model,
                      "seed": args.seed, "source_jobs": receipt["source_jobs"],
                      "source_artifact_sha256": receipt["source_artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
