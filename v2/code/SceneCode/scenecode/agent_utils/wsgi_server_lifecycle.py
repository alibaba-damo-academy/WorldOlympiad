"""Shared lifecycle helpers for in-process Werkzeug WSGI servers."""

from __future__ import annotations

import logging

from threading import Event, Thread
from typing import Any, Callable

from werkzeug.serving import BaseWSGIServer, make_server


def create_threaded_wsgi_server(
    host: str, port: int, app: Any
) -> BaseWSGIServer:
    """Create a threaded Werkzeug WSGI server owned by the caller."""
    return make_server(host, port, app, threaded=True)


def run_wsgi_server(
    wsgi_server: BaseWSGIServer,
    shutdown_event: Event,
    server_name: str,
    logger: logging.Logger,
) -> None:
    """Run a WSGI server until shutdown is requested."""
    try:
        wsgi_server.serve_forever()
    except Exception as e:
        logger.error(f"{server_name} server thread failed: {e}")
        shutdown_event.set()


def stop_threaded_wsgi_server(
    wsgi_server: BaseWSGIServer,
    server_thread: Thread,
    shutdown_event: Event,
    server_name: str,
    timeout_s: float = 5.0,
) -> None:
    """Stop a managed WSGI server and raise if the thread does not exit."""
    shutdown_event.set()
    wsgi_server.shutdown()
    wsgi_server.server_close()
    server_thread.join(timeout=timeout_s)
    if server_thread.is_alive():
        raise RuntimeError(
            f"{server_name} server thread did not stop within {timeout_s} seconds"
        )


def trigger_shutdown_callback_async(
    callback: Callable[[], None] | None,
    thread_name: str,
) -> bool:
    """Run a shutdown callback asynchronously if one is registered."""
    if callback is None:
        return False
    Thread(target=callback, name=thread_name, daemon=True).start()
    return True
