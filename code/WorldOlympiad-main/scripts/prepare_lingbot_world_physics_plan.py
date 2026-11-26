#!/usr/bin/env python3
"""Prepare native LingBot-World v2 physics input bundles and a multi-case plan.

Reference cases produced by `autoresearch.physical_reference` are converted one
by one into the `lingbot_world_v2_native_render_physics.v2` contract, then
listed in a plan file for `scripts/run_lingbot_world_physics_v2.py`.
"""
import argparse, json, sys
from pathlib import Path

TASKS = ("free_fall", "projectile", "rebound", "sliding", "collision", "support_tip",
         "ramp_roll", "domino_chain", "collision_chain", "shallow_containment")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--inputs-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    sys.path.insert(0, str(workspace / "code/WorldOlympiad-main"))
    from autoresearch.lingbot_world_physics import prepare_case
    reference = Path(args.reference_root).resolve()
    inputs = Path(args.inputs_root).resolve()
    cases = []
    skipped = []
    for task in args.tasks:
        source = reference / "tasks" / task
        if not (source / "simulation.json").is_file() or not (source / "render.json").is_file():
            skipped.append(task)
            continue
        destination = inputs / task
        if destination.exists() and any(destination.iterdir()):
            receipt = json.loads((destination / "input_contract.json").read_text())
            if receipt["source_case"] != str(source):
                raise SystemExit("existing input bundle came from a different source case: " + task)
        else:
            prepare_case(source, destination, model_id=args.model_id)
        receipt = json.loads((destination / "input_contract.json").read_text())
        frames = receipt["sampling"]["frame_num"]
        cases.append({"case_id": task, "input_dir": str(destination), "frames": frames,
                      "prompt": receipt["conditioning"]["prompt"]})
    plan = {"schema_version": "lingbot_world_v2_physics_plan.v1", "model_id": args.model_id,
            "reference_root": str(reference), "base_seed": args.base_seed,
            "skipped_cases": skipped, "cases": cases}
    Path(args.plan).parent.mkdir(parents=True, exist_ok=True)
    Path(args.plan).write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "skipped": skipped}))


if __name__ == "__main__":
    main()
