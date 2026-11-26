import logging
import threading
import time
import unittest

from unittest.mock import MagicMock

import flask

from scenecode.agent_utils.wsgi_server_lifecycle import (
    create_threaded_wsgi_server,
    run_wsgi_server,
    stop_threaded_wsgi_server,
)


class TestWsgiServerLifecycle(unittest.TestCase):
    def test_run_and_stop_threaded_wsgi_server(self):
        app = flask.Flask(__name__)

        @app.route("/health")
        def health():
            return {"status": "ok"}

        wsgi_server = create_threaded_wsgi_server("127.0.0.1", 0, app)
        shutdown_event = threading.Event()
        original_server_close = wsgi_server.server_close
        wsgi_server.server_close = MagicMock(side_effect=original_server_close)
        server_thread = threading.Thread(
            target=run_wsgi_server,
            args=(wsgi_server, shutdown_event, "Test", logging.getLogger(__name__)),
            daemon=False,
        )

        server_thread.start()
        time.sleep(0.1)
        self.assertTrue(server_thread.is_alive())

        stop_threaded_wsgi_server(
            wsgi_server, server_thread, shutdown_event, "Test", timeout_s=2.0
        )

        self.assertTrue(shutdown_event.is_set())
        self.assertFalse(server_thread.is_alive())
        self.assertGreaterEqual(wsgi_server.server_close.call_count, 1)

    def test_stop_threaded_wsgi_server_raises_when_thread_stays_alive(self):
        wsgi_server = MagicMock()
        server_thread = MagicMock()
        server_thread.is_alive.return_value = True
        shutdown_event = threading.Event()

        with self.assertRaises(RuntimeError):
            stop_threaded_wsgi_server(
                wsgi_server,
                server_thread,
                shutdown_event,
                "Test",
                timeout_s=0.01,
            )

        self.assertTrue(shutdown_event.is_set())
        wsgi_server.shutdown.assert_called_once()
        wsgi_server.server_close.assert_called_once()
        server_thread.join.assert_called_once_with(timeout=0.01)
