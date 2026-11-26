#!/usr/bin/env python3
"""Convert a generated articulated URDF file into a packaged SDF asset."""

from __future__ import annotations

import argparse
import logging
import sys

from pathlib import Path

from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scenecode.agent_utils.code_articulated_conversion import (
    convert_generated_articulated_urdf,
)
from scenecode.agent_utils.blender import BlenderServer
from scenecode.agent_utils.convex_decomposition_server import (
    ConvexDecompositionServer,
)
from scenecode.agent_utils.vlm_service import VLMService

DEFAULT_URDF_PATH = (
    REPO_ROOT / 'examples/urdf_examples/Box_01/box_01.urdf'
)
DEFAULT_CFG_PATH = (
    REPO_ROOT / 'configs/furniture_agent/base_furniture_agent.yaml'
)
console_logger = logging.getLogger(__name__)


def _parse_dimensions(value: str | None) -> list[float] | None:
    if not value:
        return None
    parts = [float(part.strip()) for part in value.split(',') if part.strip()]
    if len(parts) != 3:
        raise ValueError('--dimensions must contain exactly 3 comma-separated floats')
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert a generated articulated URDF file to packaged SDF.'
    )
    parser.add_argument(
        '--input',
        '-i',
        type=Path,
        default=DEFAULT_URDF_PATH,
        help='Path to input URDF file.',
    )
    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        default=None,
        help='Path to output SDF file. Defaults to input path with .sdf suffix.',
    )
    parser.add_argument(
        '--dimensions',
        type=str,
        default=None,
        help='Desired dimensions as width,depth,height in meters.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_CFG_PATH,
        help='Path to SceneCode config used for articulated VLM analysis.',
    )
    parser.add_argument(
        '--api-base',
        type=str,
        default=None,
        help='Optional OpenAI-compatible API base for VLM requests.',
    )
    parser.add_argument(
        '--collision-threshold',
        type=float,
        default=0.05,
        help='CoACD approximation threshold for collision generation.',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging.',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(message)s',
    )

    urdf_path = args.input.resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f'URDF file not found: {urdf_path}')
    output_path = args.output.resolve() if args.output else urdf_path.with_suffix('.sdf')
    desired_dimensions = _parse_dimensions(args.dimensions)

    cfg = OmegaConf.load(args.config)
    vlm_service = VLMService(api_base=args.api_base)

    console_logger.info('Input URDF: %s', urdf_path)
    console_logger.info('Output SDF: %s', output_path)
    console_logger.info('Desired dimensions: %s', desired_dimensions)

    server = ConvexDecompositionServer()
    blender_server = BlenderServer()
    server.start()
    server.wait_until_ready()
    blender_server.start()
    blender_server.wait_until_ready()
    client = server.get_client()
    try:
        result = convert_generated_articulated_urdf(
            urdf_path=urdf_path,
            output_path=output_path,
            desired_dimensions=desired_dimensions,
            collision_client=client,
            vlm_service=vlm_service,
            cfg=cfg,
            blender_server=blender_server,
            collision_threshold=args.collision_threshold,
            model_name=urdf_path.stem,
        )
    finally:
        blender_server.stop()
        server.stop()

    console_logger.info('Successfully generated SDF: %s', result.sdf_path)
    console_logger.info('Analysis written to: %s', result.analysis_path)
    console_logger.info('Scale factor: %.6f', result.scale_factor)


if __name__ == '__main__':
    main()
