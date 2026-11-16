#!/usr/bin/env python3
"""
Standalone Blender server script.

This script runs the Blender server in the main thread, which is required
because bpy (Blender Python API) is not thread-safe.
"""

import argparse
import logging
import tempfile

from pathlib import Path

from scenecode.agent_utils.blender.server_app import BlenderRenderApp
from scenecode.agent_utils.blender.server_manager import (
    find_available_port,
    is_port_available,
)

console_logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
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
        help="Specific port to bind to. If not specified, will find available port.",
    )
    parser.add_argument(
        "--port-range",
        type=str,
        default="8000-8050",
        help="Port range to search for available port (format: start-end), "
        "default: %(default)s.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="Temporary directory for server files.",
    )
    parser.add_argument(
        "--blend-file",
        type=Path,
        help="Path to a .blend file to use as base scene.",
    )
    parser.add_argument(
        "--bpy-settings-file",
        type=Path,
        help="Path to a .py file with Blender settings.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file for persistent logging.",
    )

    return parser.parse_args()


def validate_files(args: argparse.Namespace) -> None:
    """Validate that blend and settings files exist if provided.

    Args:
        args: Parsed command line arguments.

    Raises:
        FileNotFoundError: If specified files don't exist.
    """
    if args.blend_file and not args.blend_file.exists():
        raise FileNotFoundError(f"Blend file not found: {args.blend_file}")

    if args.bpy_settings_file and not args.bpy_settings_file.exists():
        raise FileNotFoundError(
            f"Bpy settings file not found: {args.bpy_settings_file}"
        )


def resolve_port(host: str, port: int | None, port_range: str) -> int:
    """Resolve which port to use for the server.

    Args:
        host: Host address to check port availability on.
        port: Specific port to use, or None to find available port.
        port_range: Port range string in format "start-end".

    Returns:
        The port number to use.

    Raises:
        RuntimeError: If specific port is unavailable or no ports in range.
        ValueError: If port range format is invalid.
    """
    if port is not None:
        # Use specific port.
        if not is_port_available(host=host, port=port):
            raise RuntimeError(f"Port {port} is not available on {host}")
        return port

    # Parse port range and find available port.
    try:
        start_port, end_port = map(int, port_range.split("-"))
        if start_port > end_port:
            raise ValueError(f"Invalid port range: {port_range} (start > end)")
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(
                f"Invalid port range format: {port_range} "
                f"(expected format: start-end)"
            ) from e
        raise

    target_port = find_available_port(host=host, port_range=(start_port, end_port))
    if target_port is None:
        raise RuntimeError(
            f"No available ports found in range {start_port}-{end_port} on {host}"
        )

    console_logger.info(
        f"Found available port {target_port} in range {start_port}-{end_port}"
    )
    return target_port


def setup_and_run_server(args: argparse.Namespace, port: int) -> None:
    """Set up temporary directory and run the Blender server.

    Args:
        args: Parsed command line arguments.
        port: Port number to bind to.
    """
    # Use provided temp dir or create one.
    if args.temp_dir:
        temp_dir = str(args.temp_dir)
        args.temp_dir.mkdir(parents=True, exist_ok=True)
        console_logger.info(f"Using provided temp directory: {temp_dir}")
    else:
        temp_dir = tempfile.mkdtemp(prefix="blender_server_")
        console_logger.info(f"Created temp directory: {temp_dir}")

    # Log configuration.
    console_logger.info(f"Starting Blender server at {args.host}:{port}")
    if args.blend_file:
        console_logger.info(f"Using blend file: {args.blend_file}")
    if args.bpy_settings_file:
        console_logger.info(f"Using bpy settings file: {args.bpy_settings_file}")

    # Create and run the Flask app.
    console_logger.info("Initializing Blender render app...")
    app = BlenderRenderApp(
        temp_dir=temp_dir,
        blend_file=args.blend_file,
        bpy_settings_file=args.bpy_settings_file,
    )

    console_logger.info("Starting Flask server (bpy requires main thread execution)...")
    # Run in main thread - required for bpy.
    app.run(
        host=args.host,
        port=port,
        debug=False,
        use_reloader=False,
        threaded=False,
    )


def main() -> int:
    """Main entry point for the standalone Blender server.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # Parse arguments first to check for log file.
    args = parse_arguments()

    # Set up logging handlers based on --log-file argument.
    handlers: list[logging.Handler] = []
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(args.log_file, mode="a")
        handlers.append(file_handler)
    else:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    try:
        # Adjust logging level if verbose requested.
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            console_logger.debug("Verbose logging enabled")

        # Validate file arguments.
        validate_files(args)

        # Determine which port to use.
        target_port = resolve_port(
            host=args.host, port=args.port, port_range=args.port_range
        )

        # Set up and run server.
        setup_and_run_server(args=args, port=target_port)

        console_logger.info("Server stopped gracefully")
        return 0

    except SystemExit as e:
        return e.code if e.code else 1
    except KeyboardInterrupt:
        console_logger.info("Server interrupted by user")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        console_logger.error(str(e))
        return 1
    except OSError as e:
        if "Address already in use" in str(e):
            console_logger.error(f"Port is already in use: {e}")
        else:
            console_logger.error(f"Network error: {e}")
        return 1
    except Exception as e:
        console_logger.error(f"Server failed with error: {e}")
        if hasattr(args, "verbose") and args.verbose:
            console_logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    main()
