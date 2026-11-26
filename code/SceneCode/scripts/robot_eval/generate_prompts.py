#!/usr/bin/env python
"""Generate scene prompts from a human task description.

This is Stage 1 of the robot evaluation pipeline. It uses an LLM to convert
a natural language task (e.g., "Find a fruit and place it on the table") into
diverse scene prompts that can be fed into main.py for scene generation.

Usage:
    python scripts/robot_eval/generate_prompts.py \
        --task "Find a fruit and place it on the kitchen table" \
        --output-dir outputs/eval_run \
        --num-prompts 5

Output:
    outputs/eval_run/prompts.csv      - Scene prompts for main.py
    outputs/eval_run/task_metadata.yaml - Task info for validation
"""

import argparse
import asyncio
import logging

from pathlib import Path

from scenecode.robot_eval import create_robot_eval_config
from scenecode.robot_eval.task_generation.scene_prompt_generator import (
    generate_scene_prompts,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(
        description="Generate scene prompts from a human task description"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Human task description (e.g., 'Find a fruit and place it on the table')",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for prompts.csv and task_metadata.yaml",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of diverse scene prompts to generate (default: 5)",
    )
    args = parser.parse_args()

    # Create config for model settings.
    cfg = create_robot_eval_config()

    logger.info(f"Generating {args.num_prompts} scene prompts for task: {args.task}")

    output = await generate_scene_prompts(
        task_description=args.task,
        cfg=cfg,
        output_dir=args.output_dir,
        num_prompts=args.num_prompts,
    )

    logger.info(f"Generated {len(output.scene_prompts)} prompts")
    logger.info(f"  Room requirement: {output.analysis.room_requirement}")
    logger.info(f"  Required objects: {output.analysis.required_objects}")
    logger.info("")
    logger.info(f"Output files:")
    logger.info(f"  {args.output_dir}/prompts.csv")
    logger.info(f"  {args.output_dir}/task_metadata.yaml")
    logger.info("")
    logger.info("Next step: Run scene generation with:")
    logger.info(
        f"  python main.py +name=eval experiment.csv_path={args.output_dir}/prompts.csv"
    )


if __name__ == "__main__":
    asyncio.run(main())
