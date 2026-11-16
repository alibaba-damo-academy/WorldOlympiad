#!/usr/bin/env python3
"""
Geometry generation server entry point.

This module provides the main entry point for running the geometry generation server.
It uses a class-based architecture with proper lifecycle management and no global
variables or import side effects.
"""

import argparse
import logging
import signal
import sys

from .server_manager import GeometryGenerationServer

console_logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description="Geometry generation server for converting 2D images to 3D geometry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to, default: %(default)s.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7000,
        help="Port to bind to, default: %(default)s.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        default=True,
        help="Preload the Hunyuan3D pipeline on server start for consistent "
        "performance (default: True).",
    )
    parser.add_argument(
        "--no-preload",
        action="store_false",
        dest="preload",
        help="Disable pipeline preloading. The pipeline will be loaded lazily on "
        "first request, which can be useful for development/testing.",
    )
    parser.add_argument(
        "--use-mini",
        action="store_true",
        default=False,
        help="Use the mini model variant (0.6B parameters) instead of the full "
        "model. The mini model is faster with lower memory usage but may have "
        "reduced quality (default: False). Only applies to Hunyuan3D backend.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hunyuan3d", "sam3d"],
        default="hunyuan3d",
        help="Which 3D generation backend to use (default: %(default)s).",
    )
    parser.add_argument(
        "--sam3-checkpoint",
        type=str,
        default="external/checkpoints/sam3.pt",
        help="Path to SAM3 segmentation model checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--sam3d-checkpoint",
        type=str,
        default="external/checkpoints/pipeline.yaml",
        help="Path to SAM 3D Objects pipeline config (default: %(default)s).",
    )
    parser.add_argument(
        "--sam3d-mode",
        type=str,
        choices=["foreground", "text"],
        default="foreground",
        help="SAM3D segmentation mode (default: %(default)s). "
        "'foreground' auto-detects objects on uniform backgrounds, "
        "'text' uses text prompts for segmentation.",
    )
    parser.add_argument(
        "--sam3d-threshold",
        type=float,
        default=0.5,
        help="SAM3D confidence threshold for mask generation (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable debug level logging if True.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> int:
    """Main entry point for the asset generation server.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_arguments()
    setup_logging(args.verbose)

    server = None

    def signal_handler(signum, _):
        """Handle shutdown signals gracefully."""
        console_logger.info(f"Received signal {signum}, shutting down...")
        if server:
            server.stop()
        sys.exit(0)

    # Register signal handlers for graceful shutdown.
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Build SAM3D config if using SAM3D backend.
        sam3d_config = None
        if args.backend == "sam3d":
            sam3d_config = {
                "sam3_checkpoint": args.sam3_checkpoint,
                "sam3d_checkpoint": args.sam3d_checkpoint,
                "mode": args.sam3d_mode,
                "text_prompt": None,
                "threshold": args.sam3d_threshold,
            }

        # Create and start the server.
        server = GeometryGenerationServer(
            host=args.host,
            port=args.port,
            preload_pipeline=args.preload,
            use_mini=args.use_mini,
            backend=args.backend,
            sam3d_config=sam3d_config,
        )
        server.start()

        console_logger.info(
            f"Geometry generation server ({args.backend}) running on "
            f"{args.host}:{args.port}"
        )
        console_logger.info("Press Ctrl+C to stop the server")

        # Keep the main thread alive.
        # The server runs in its own thread, so we just wait.
        try:
            while server.is_running():
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            console_logger.info("Keyboard interrupt received")

    except Exception as e:
        console_logger.error(f"Failed to start server: {e}")
        return 1

    finally:
        if server:
            server.stop()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
