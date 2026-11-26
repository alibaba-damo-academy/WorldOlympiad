"""Subprocess manager for convex decomposition server lifecycle."""

import logging
import os
import subprocess
import sys
import time

from pathlib import Path

import requests

from scenecode.agent_utils.convex_decomposition_server.client import (
    ConvexDecompositionClient,
)
from scenecode.utils.network_utils import find_available_port, is_port_available

console_logger = logging.getLogger(__name__)


class ConvexDecompositionServer:
    """Manages a convex decomposition server running in a separate subprocess.

    This server isolates CoACD's OpenMP operations from the main worker process
    to prevent deadlocks when using ThreadPoolExecutor. Each worker spawns its
    own server subprocess. Supports both CoACD and V-HACD decomposition methods.

    The server provides a /generate_collision endpoint for convex decomposition.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        port_range: tuple[int, int] | None = None,
        omp_threads: int = os.cpu_count() or 1,
        server_startup_delay: float = 1.0,
        port_cleanup_delay: float = 1.0,
        log_file: Path | None = None,
    ) -> None:
        """Initialize the convex decomposition server manager.

        Args:
            host: The host address to bind the server to.
            port: The specific port number to bind to. If None, will use port_range.
            port_range: Tuple of (start_port, end_port) to search for available
                port. Defaults to (7100, 7150) if neither port nor port_range
                specified.
            omp_threads: Number of OpenMP threads for CoACD operations.
            server_startup_delay: Seconds to wait after starting server subprocess
                to allow initialization.
            port_cleanup_delay: Seconds to wait after stopping server to allow
                OS port cleanup.
            log_file: Optional path to log file for persistent logging.

        Raises:
            ValueError: If both port and port_range are specified.
        """
        if port is not None and port_range is not None:
            raise ValueError("Cannot specify both port and port_range")

        if port is None and port_range is None:
            port_range = (7100, 7150)

        self._host = host
        self._port = port
        self._port_range = port_range
        self._omp_threads = omp_threads
        self._actual_port: int | None = None
        self._server_startup_delay = server_startup_delay
        self._port_cleanup_delay = port_cleanup_delay
        self._log_file = log_file
        self._server_process: subprocess.Popen | None = None
        self._running = False

        console_logger.debug(
            f"Initialized ConvexDecompositionServer(host={host}, port={port}, "
            f"port_range={port_range}, omp_threads={omp_threads})"
        )

    def start(self) -> None:
        """Start the convex decomposition server in a separate process.

        Raises:
            RuntimeError: If server is already running, or if no port is available.
            FileNotFoundError: If standalone server script not found.
            subprocess.SubprocessError: If server process fails to start.
        """
        if self._running:
            raise RuntimeError("Server is already running")

        target_port = self._determine_port()
        self._actual_port = target_port
        console_logger.info(
            f"Starting convex decomposition server on {self._host}:{target_port}"
        )

        try:
            standalone_script = Path(__file__).parent / "standalone_server.py"
            if not standalone_script.exists():
                raise FileNotFoundError(
                    f"Standalone server script not found: {standalone_script}"
                )

            cmd = [
                sys.executable,
                str(standalone_script),
                "--host",
                self._host,
                "--port",
                str(target_port),
                "--omp-threads",
                str(self._omp_threads),
            ]

            if self._log_file:
                cmd.extend(["--log-file", str(self._log_file)])

            console_logger.debug(f"Server command: {' '.join(cmd)}")

            # Set PYTHONPATH so subprocess can find scenecode module.
            env = os.environ.copy()
            project_root = Path(__file__).parent.parent.parent.parent
            env["PYTHONPATH"] = str(project_root) + ":" + env.get("PYTHONPATH", "")

            # Start the server process.
            self._server_process = subprocess.Popen(cmd, text=True, env=env)

            # Wait for server to initialize.
            time.sleep(self._server_startup_delay)

            self._running = True
            console_logger.info(
                f"Convex decomposition server started with PID "
                f"{self._server_process.pid}"
            )

        except Exception as e:
            self._running = False
            self._actual_port = None
            console_logger.error(f"Failed to start convex decomposition server: {e}")
            raise

    def _determine_port(self) -> int:
        """Determine the port to use for the server."""
        if self._port is not None:
            target_port = self._port
            if not is_port_available(host=self._host, port=target_port):
                raise RuntimeError(
                    f"Port {target_port} is not available on {self._host}"
                )
        else:
            target_port = find_available_port(
                host=self._host, port_range=self._port_range
            )
            if target_port is None:
                raise RuntimeError(
                    f"No available ports found in range {self._port_range} "
                    f"on {self._host}"
                )
            console_logger.info(
                f"Found available port {target_port} in range {self._port_range}"
            )
        return target_port

    def stop(self) -> None:
        """Stop the convex decomposition server and cleanup resources."""
        if not self._running:
            console_logger.debug("Convex decomposition server already stopped")
            return

        console_logger.info("Stopping convex decomposition server...")
        self._running = False

        if self._server_process:
            pid = self._server_process.pid
            console_logger.debug(f"Terminating server process {pid}")

            self._server_process.terminate()
            try:
                exit_code = self._server_process.wait(timeout=5.0)
                console_logger.debug(
                    f"Server process {pid} exited with code {exit_code}"
                )
            except subprocess.TimeoutExpired:
                console_logger.warning(
                    f"Server process {pid} did not terminate gracefully, killing..."
                )
                self._server_process.kill()
                exit_code = self._server_process.wait()
                console_logger.debug(
                    f"Server process {pid} killed with code {exit_code}"
                )
            finally:
                if self._server_process.stdout:
                    self._server_process.stdout.close()
                if self._server_process.stderr:
                    self._server_process.stderr.close()

            self._server_process = None

        self._actual_port = None
        time.sleep(self._port_cleanup_delay)
        console_logger.info("Convex decomposition server stopped")

    def is_running(self) -> bool:
        """Check if the server is currently running.

        Returns:
            True if the server is running, False otherwise.
        """
        return self._running

    def get_url(self) -> str:
        """Get the URL where the server is running.

        Returns:
            The server URL.

        Raises:
            RuntimeError: If the server is not running.
        """
        if not self.is_running():
            status = self.get_process_status()
            raise RuntimeError(f"Server is not running (status: {status})")
        return f"http://{self._host}:{self._actual_port}"

    def get_host(self) -> str:
        """Get the host address."""
        return self._host

    def get_port(self) -> int | None:
        """Get the actual port the server is running on."""
        return self._actual_port

    def get_process_status(self) -> str:
        """Get the status of the server process for debugging.

        Returns:
            Human-readable status string.
        """
        if not self._server_process:
            return "No process"

        poll_result = self._server_process.poll()
        if poll_result is None:
            return f"Running (PID {self._server_process.pid})"
        else:
            return f"Exited with code {poll_result}"

    def wait_until_ready(self, timeout: float = 10.0) -> None:
        """Wait until the server is ready to accept HTTP requests.

        Args:
            timeout: Maximum time to wait in seconds.

        Raises:
            RuntimeError: If server is not running or doesn't become ready within
                timeout.
        """
        if not self.is_running():
            raise RuntimeError("Server is not running")

        console_logger.debug(
            f"Waiting for convex decomposition server to be ready (timeout: {timeout}s)"
        )

        start_time = time.time()
        max_retries = int(timeout * 2)  # Check twice per second.
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.get_url()}/health", timeout=2)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    console_logger.debug(
                        f"Convex decomposition server is ready after {elapsed:.1f}s"
                    )
                    return
            except requests.RequestException:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise RuntimeError(
                        f"Convex decomposition server failed to become ready "
                        f"within {timeout}s"
                    )
                if i < max_retries - 1:
                    time.sleep(0.5)
                    continue

        raise RuntimeError(
            f"Convex decomposition server did not become ready within {timeout}s"
        )

    def get_client(self) -> "ConvexDecompositionClient":
        """Get a client connected to this server.

        Returns:
            A ConvexDecompositionClient instance configured to connect to this server.

        Raises:
            RuntimeError: If server is not running.
        """
        if not self.is_running():
            raise RuntimeError("Server is not running")

        return ConvexDecompositionClient(host=self._host, port=self._actual_port)
