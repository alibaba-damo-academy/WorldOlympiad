#!/usr/bin/env python3
"""Run the reusable RoboLab-DROID push-to-region reference operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenecode_pybullet_bridge import DroidPushInputs, run_droid_push


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-sdf", type=Path, required=True)
    parser.add_argument("--subject-sdf", type=Path, required=True)
    parser.add_argument("--target-sdf", type=Path, required=True)
    parser.add_argument("--panda-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject-center-xy", type=float, nargs=2, required=True)
    parser.add_argument("--target-center-xy", type=float, nargs=2, required=True)
    parser.add_argument(
        "--subject-orientation-xyzw",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 1.0),
    )
    parser.add_argument(
        "--target-orientation-xyzw",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 1.0),
    )
    parser.add_argument("--subject-lateral-friction", type=float, default=0.35)
    args = parser.parse_args()
    result = run_droid_push(DroidPushInputs(**vars(args)))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if not result.success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
