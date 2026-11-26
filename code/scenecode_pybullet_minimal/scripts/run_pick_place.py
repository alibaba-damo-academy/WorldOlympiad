#!/usr/bin/env python3
"""Run the minimal SceneCode cup-to-tray PyBullet sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenecode_pybullet_bridge import PickPlaceInputs, run_pick_place


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-sdf", type=Path, required=True)
    parser.add_argument("--cup-sdf", type=Path, required=True)
    parser.add_argument("--tray-sdf", type=Path, required=True)
    parser.add_argument("--panda-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_pick_place(PickPlaceInputs(**vars(args)))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if not result.success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
