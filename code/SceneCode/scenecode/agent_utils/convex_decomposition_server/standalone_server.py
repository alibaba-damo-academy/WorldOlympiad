#!/usr/bin/env python3
"""Standalone convex decomposition server for collision geometry generation.

This script starts a Flask server that handles convex decomposition requests
(CoACD and V-HACD). It must be run as a subprocess to isolate CoACD's OpenMP
from the main worker's ThreadPoolExecutor.

Usage:
    python -m scenecode.agent_utils.convex_decomposition_server.standalone_server \
        --host 127.0.0.1 --port 7100 --omp-threads 4
"""

import argparse
import logging
import os
import sys


def main() -> None:
    """Run the convex decomposition server."""
    parser = argparse.ArgumentParser(
        description="Convex decomposition collision geometry generation server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port number to bind to",
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=4,
        help="Number of OpenMP threads for CoACD (default: 4)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file for persistent logging",
    )
    args = parser.parse_args()

    # CRITICAL: Set OMP_NUM_THREADS BEFORE importing coacd.
    # OpenMP reads this environment variable at first import, so it must be
    # set before any module imports coacd.
    os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)

    # Configure logging.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[ConvexDecomp Server] %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger(__name__)

    # Add file handler if log file specified.
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - [ConvexDecomp] %(levelname)s - %(message)s"
            )
        )
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {args.log_file}")

    logger.info(f"OMP_NUM_THREADS set to {args.omp_threads}")

    # Import server app AFTER setting environment variable.
    from scenecode.agent_utils.convex_decomposition_server.server_app import (
        ConvexDecompositionServerApp,
    )

    app = ConvexDecompositionServerApp()

    logger.info(f"Starting convex decomposition server on {args.host}:{args.port}")

    # Run with threaded=False to ensure CoACD runs on main thread.
    # This is important because OpenMP should not be used from multiple threads.
    app.run(host=args.host, port=args.port, threaded=False, debug=False)


if __name__ == "__main__":
    main()
