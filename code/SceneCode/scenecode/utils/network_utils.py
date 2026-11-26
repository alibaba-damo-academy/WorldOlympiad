"""Network utilities for server management."""

import random
import socket


def is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding.

    Args:
        host: The host address to check.
        port: The port number to check.

    Returns:
        True if the port is available, False otherwise.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Allow re-binding after transient TCP cleanup states (for example TIME_WAIT).
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def find_available_port(host: str, port_range: tuple[int, int]) -> int | None:
    """Find an available port in the given range.

    Ports are tried in random order to reduce collision probability when
    multiple processes are searching for ports simultaneously.

    Args:
        host: The host address to check.
        port_range: Tuple of (start_port, end_port) inclusive.

    Returns:
        An available port, or None if no ports are available.
    """
    start_port, end_port = port_range
    ports = list(range(start_port, end_port + 1))
    random.shuffle(ports)
    for port in ports:
        if is_port_available(host, port):
            return port
    return None
