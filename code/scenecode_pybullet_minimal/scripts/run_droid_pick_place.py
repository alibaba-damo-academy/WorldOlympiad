#!/usr/bin/env python3
"""Run a RoboLab-DROID cup-to-tray expert reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenecode_pybullet_bridge import DroidPickPlaceInputs, run_droid_pick_place
from scenecode_pybullet_bridge.droid_pick_place import SUPPORTED_SEMANTIC_OPERATORS


# Task-level terminal predicates map onto the physical bridge's restricted
# success-check vocabulary. Synonyms select the exact same physical check, so
# this never weakens the declared success condition.
_SUCCESS_PREDICATE_ALIASES = {
    "in_container": "in_container",
    "inside": "in_container",
    "on_region": "on_region",
    "stacked": "stacked",
    "inserted": "inserted",
    "covering": "covering",
    "contact": "contact",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-sdf", type=Path, required=True)
    parser.add_argument("--cup-sdf", type=Path, required=True)
    parser.add_argument("--tray-sdf", type=Path, required=True)
    parser.add_argument("--panda-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grasp-base-offset-z-m", type=float, default=0.19)
    parser.add_argument("--hover-base-offset-z-m", type=float, default=0.38)
    parser.add_argument("--lift-base-offset-z-m", type=float, default=0.43)
    parser.add_argument("--grasp-yaw-deg", type=float, default=90.0)
    parser.add_argument("--place-target-offset-xy-m", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--place-target-offset-z-m", type=float, default=0.0)
    parser.add_argument(
        "--grasp-offset-xy-m",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        help="Base-frame XY offset of the intended grasp site from the subject root link.",
    )
    parser.add_argument("--cup-center-xy", type=float, nargs=2, default=(0.24, -0.08))
    parser.add_argument("--tray-center-xy", type=float, nargs=2, default=(-0.18, 0.10))
    parser.add_argument(
        "--cup-orientation-xyzw",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 1.0),
    )
    parser.add_argument(
        "--tray-orientation-xyzw",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 1.0),
    )
    parser.add_argument(
        "--terminal-alignment-axis-local", type=float, nargs=3
    )
    parser.add_argument(
        "--terminal-alignment-axis-world", type=float, nargs=3
    )
    parser.add_argument(
        "--terminal-upright-min-cosine",
        type=float,
        default=None,
        help=(
            "require the measured final subject alignment cosine to reach this "
            "value against the declared terminal axes"
        ),
    )
    parser.add_argument(
        "--target-collision-enabled",
        type=int,
        choices=(0, 1),
        default=1,
        help="load the target SDF collision geometry (0 for a support-region alias)",
    )
    # Every terminal operator the compiler can emit must reach the shared
    # expert; the place family uses the default motion envelope.
    parser.add_argument(
        "--semantic-operator",
        choices=sorted(SUPPORTED_SEMANTIC_OPERATORS),
        default="pick_place",
    )
    parser.add_argument(
        "--success-predicate",
        default="in_container",
        help="task-level terminal predicate or its canonical physical evaluator",
    )
    parser.add_argument("--target-frame-count", type=int)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        success_predicate = _SUCCESS_PREDICATE_ALIASES[args.success_predicate]
    except KeyError:
        parser.error(
            "unsupported pick-place success predicate; expected a known task "
            "predicate or its canonical physical evaluator"
        )
    inputs = vars(args)
    inputs["success_predicate"] = success_predicate
    inputs["target_collision_enabled"] = bool(inputs.pop("target_collision_enabled"))
    result = run_droid_pick_place(DroidPickPlaceInputs(**inputs))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if not result.success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
