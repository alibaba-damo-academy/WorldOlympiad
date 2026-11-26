"""Unit tests for BlenderServer retry logic."""

import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from scenecode.agent_utils.blender.server_manager import BlenderServer


class TestBlenderServerRetry(unittest.TestCase):
    """Test BlenderServer auto-restart and retry behavior."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.server = BlenderServer(port=9999)
        self.server._running = True
        self.server._actual_port = 9999
        self.server._host = "127.0.0.1"
        # Mock server process.
        self.server._server_process = MagicMock()

    def test_request_succeeds_first_try(self) -> None:
        """Test that successful request returns without retry."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output_path": "/tmp/output.gltf"}

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = self.server._make_request_with_retry(
                endpoint="/convert_glb_to_gltf",
                timeout=60.0,
                json={"input_path": "/tmp/input.glb"},
                result_key="output_path",
            )

        self.assertEqual(result, Path("/tmp/output.gltf"))
        mock_post.assert_called_once()

    def test_server_crash_triggers_restart_and_retry(self) -> None:
        """Test that server crash triggers restart and retry."""
        # First call fails with ConnectionError, server process is dead.
        self.server._server_process.poll.return_value = -11  # SIGSEGV
        self.server._server_process.returncode = -11

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"output_path": "/tmp/output.gltf"}

        call_count = 0

        def mock_post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.ConnectionError("Connection refused")
            return mock_response

        with (
            patch("requests.post", side_effect=mock_post_side_effect),
            patch.object(self.server, "_restart_server") as mock_restart,
        ):
            result = self.server._make_request_with_retry(
                endpoint="/convert_glb_to_gltf",
                timeout=60.0,
                json={"input_path": "/tmp/input.glb"},
                result_key="output_path",
            )

        self.assertEqual(result, Path("/tmp/output.gltf"))
        mock_restart.assert_called_once()
        self.assertEqual(call_count, 2)

    def test_server_crash_max_retries_exceeded(self) -> None:
        """Test that error is raised after max retries exceeded."""
        # Server always dead.
        self.server._server_process.poll.return_value = -11
        self.server._server_process.returncode = -11

        with (
            patch(
                "requests.post",
                side_effect=requests.ConnectionError("Connection refused"),
            ),
            patch.object(self.server, "_restart_server"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.server._make_request_with_retry(
                    endpoint="/convert_glb_to_gltf",
                    timeout=60.0,
                    json={"input_path": "/tmp/input.glb"},
                    max_retries=2,
                )

        self.assertIn("crashed", str(ctx.exception))
        self.assertIn("failed after 2 restart attempts", str(ctx.exception))

    def test_connection_error_server_alive_no_retry(self) -> None:
        """Test that connection error with live server raises immediately."""
        # Server process still alive.
        self.server._server_process.poll.return_value = None

        with patch(
            "requests.post", side_effect=requests.ConnectionError("Connection refused")
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.server._make_request_with_retry(
                    endpoint="/convert_glb_to_gltf",
                    timeout=60.0,
                    json={"input_path": "/tmp/input.glb"},
                )

        self.assertIn("failed", str(ctx.exception))
        self.assertNotIn("restart", str(ctx.exception))

    def test_http_error_raises_without_retry(self) -> None:
        """Test that HTTP errors (non-crash) raise without retry."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.return_value = {"description": "bpy failed"}

        with patch("requests.post", return_value=mock_response) as mock_post:
            with self.assertRaises(RuntimeError) as ctx:
                self.server._make_request_with_retry(
                    endpoint="/convert_glb_to_gltf",
                    timeout=60.0,
                    json={"input_path": "/tmp/input.glb"},
                )

        self.assertIn("bpy failed", str(ctx.exception))
        mock_post.assert_called_once()  # No retry for HTTP errors.

    def test_result_key_returns_list_of_paths(self) -> None:
        """Test that list result_key values are converted to list of Paths."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "image_paths": ["/tmp/view1.png", "/tmp/view2.png"]
        }

        with patch("requests.post", return_value=mock_response):
            result = self.server._make_request_with_retry(
                endpoint="/render_multiview",
                timeout=60.0,
                json={"output_dir": "/tmp"},
                result_key="image_paths",
            )

        self.assertEqual(result, [Path("/tmp/view1.png"), Path("/tmp/view2.png")])


if __name__ == "__main__":
    unittest.main()
