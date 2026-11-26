#!/usr/bin/env python
"""Validate task completion after robot execution.

This validates whether a robot successfully completed a task by evaluating the final
scene state.

Usage:
    python scripts/robot_eval/validate.py \
        --scene-state outputs/.../scene_002/combined_house/house_state.json \
        --dmd outputs/.../scene_002/combined_house/house.dmd.yaml \
        --scene-dir outputs/.../scene_002 \
        --task "Find a speaker and place it on the bed"

Inputs:
    --scene-state: Path to scene_state.json (per-room) or house_state.json (combined house)
    --dmd: Modified scene.dmd.yaml or house.dmd.yaml from robot (new poses after task)
    --scene-dir: Scene root directory for package:// URI resolution (default: parent of DMD)
    --task: Human task description to validate

Output:
    Validation result with per-requirement scores and overall success.
"""

import argparse
import asyncio
import logging

from pathlib import Path

from scenecode.agent_utils.blender import BlenderServer
from scenecode.robot_eval import create_robot_eval_config
from scenecode.robot_eval.success_validation.validator_agent import validate_task

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(
        description="Validate task completion after robot execution"
    )
    parser.add_argument(
        "--scene-state",
        required=True,
        type=Path,
        help="Path to scene_state.json (object metadata from scene generation)",
    )
    parser.add_argument(
        "--dmd",
        required=True,
        type=Path,
        help="Path to scene.dmd.yaml (final scene state after robot execution)",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Human task description to validate",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision tools (skip Blender server startup)",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        help="Scene root directory for package:// URI resolution (default: parent of DMD)",
    )
    args = parser.parse_args()

    cfg = create_robot_eval_config()

    logger.info(f"Task: {args.task}")
    logger.info(f"Scene state: {args.scene_state}")
    logger.info(f"DMD: {args.dmd}")

    # Start Blender server for vision tools (unless disabled).
    blender_server = None
    if not args.no_vision:
        logger.info("Starting Blender server for vision tools...")
        blender_server = BlenderServer()
        blender_server.start()

    try:
        result = await validate_task(
            task_description=args.task,
            cfg=cfg,
            scene_state_path=args.scene_state,
            dmd_path=args.dmd,
            blender_server=blender_server,
            scene_dir=args.scene_dir,
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("VALIDATION RESULT")
        logger.info("=" * 60)
        logger.info(f"Overall Success: {result.overall_success}")
        logger.info(f"Overall Score: {result.overall_score:.2f}")
        logger.info("")
        logger.info("Requirements:")
        for req in result.requirements:
            score_float = req.score.to_float()
            status = "✓" if score_float >= 1.0 else "◐" if score_float >= 0.5 else "✗"
            logger.info(f"  [{status}] {req.description} (score: {score_float})")
            logger.info(f"      {req.reasoning}")
        logger.info("")
        logger.info(f"Reasoning: {result.overall_reasoning}")
    finally:
        if blender_server is not None:
            logger.info("Stopping Blender server...")
            blender_server.stop()


if __name__ == "__main__":
    asyncio.run(main())
