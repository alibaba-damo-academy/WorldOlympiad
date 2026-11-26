#!/usr/bin/env python3
"""Materialize a prepared WorldIR plan as USD inside Isaac Sim."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import cv2  # noqa: F401  # RoboLab/Isaac requires OpenCV to load first.
from isaaclab.app import AppLauncher


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument('--fixed-articulation-body', action='append', default=[],
                        help='Author this static jointed body as a native fixed-base articulation')
    parser.add_argument(
        "--visual-asset-manifest",
        help="Manifest produced by convert_scenecode_visual_assets_to_usd.py",
    )
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    args.headless = True
    args.enable_cameras = False
    args.device = "cpu"
    return args


ARGS = _args()
APP = AppLauncher(ARGS).app

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch.isaac_compiler import materialize_isaac_usd  # noqa: E402


def main() -> None:
    print(f"[WorldIR] materializing {ARGS.plan}", flush=True)
    try:
        result = materialize_isaac_usd(
            ARGS.plan,
            ARGS.output,
            visual_asset_manifest_path=ARGS.visual_asset_manifest,
            fixed_articulation_bodies=tuple(ARGS.fixed_articulation_body),
        )
    except BaseException:  # Isaac's exception hook can otherwise hide Python failures.
        traceback.print_exc()
        raise
    print(f"[WorldIR] bundle={result}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        APP.close()
