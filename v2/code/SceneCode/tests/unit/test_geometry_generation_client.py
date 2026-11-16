import unittest

from unittest.mock import MagicMock, patch

import requests

from scenecode.agent_utils.geometry_generation_server.client import (
    GeometryGenerationClient,
)
from scenecode.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationError,
    GeometryGenerationServerRequest,
    GeometryGenerationServerResponse,
)


class TestGeometryGenerationClient(unittest.TestCase):
    """Test GeometryGenerationClient functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = GeometryGenerationClient()
        self.test_request = GeometryGenerationServerRequest(
            image_path="/test/image.png",
            output_dir="/test/output",
            prompt="A modern wooden chair",
            debug_folder="/test/debug",
            output_filename="test_chair.glb",
        )

    def test_initialization(self):
        """Test client initialization."""
        self.assertEqual(self.client.base_url, "http://127.0.0.1:7000")
        self.assertIsNotNone(self.client.session)

    @patch("requests.Session.get")
    def test_health_check_success(self, mock_get):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.client.health_check()

        self.assertTrue(result)
        mock_get.assert_called_once_with("http://127.0.0.1:7000/health", timeout=5)

    @patch("requests.Session.get")
    def test_health_check_connection_error(self, mock_get):
        """Test health check with connection error."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        result = self.client.health_check()

        self.assertFalse(result)

    @patch("requests.Session.get")
    def test_health_check_http_error(self, mock_get):
        """Test health check with HTTP error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_get.return_value = mock_response

        result = self.client.health_check()

        self.assertFalse(result)

    @patch("requests.Session.get")
    def test_health_check_timeout(self, mock_get):
        """Test health check with timeout."""
        mock_get.side_effect = requests.exceptions.Timeout("Timeout")

        result = self.client.health_check()

        self.assertFalse(result)

    @patch("requests.Session.post")
    def test_generate_geometries_success(self, mock_post):
        """Test successful batch geometry generation."""
        # Mock streaming response with NDJSON.
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b'{"index": 0, "status": "success", '
            b'"data": {"geometry_path": "/test/output/chair1.glb"}}',
            b'{"index": 1, "status": "success", '
            b'"data": {"geometry_path": "/test/output/chair2.glb"}}',
        ]
        mock_post.return_value = mock_response

        geometry_requests = [
            GeometryGenerationServerRequest(
                image_path="/test/image1.png",
                output_dir="/test/output",
                prompt="Chair 1",
            ),
            GeometryGenerationServerRequest(
                image_path="/test/image2.png",
                output_dir="/test/output",
                prompt="Chair 2",
            ),
        ]

        results = list(self.client.generate_geometries(geometry_requests))

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 0)  # Index
        self.assertEqual(results[0][1].geometry_path, "/test/output/chair1.glb")
        self.assertEqual(results[1][0], 1)  # Index
        self.assertEqual(results[1][1].geometry_path, "/test/output/chair2.glb")

        mock_post.assert_called_once_with(
            "http://127.0.0.1:7000/generate_geometries",
            json=[req.to_dict() for req in geometry_requests],
            stream=True,
            timeout=(10, 3600),
        )

    @patch("requests.Session.post")
    def test_generate_geometries_empty_list_error(self, mock_post):
        """Test batch generation with empty request list."""
        with self.assertRaises(ValueError) as context:
            list(self.client.generate_geometries([]))

        self.assertIn("Requests list cannot be empty", str(context.exception))
        mock_post.assert_not_called()

    @patch("requests.Session.post")
    def test_generate_geometries_partial_failure(self, mock_post):
        """Test batch generation with partial failure in streaming response.

        When one request fails, the client should yield an error object for that
        request and continue processing remaining results.
        """
        # Mock streaming response with success, error, success pattern.
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b'{"index": 0, "status": "success", '
            b'"data": {"geometry_path": "/test/output/chair1.glb"}}',
            b'{"index": 1, "status": "error", "error": "Generation failed"}',
            b'{"index": 2, "status": "success", '
            b'"data": {"geometry_path": "/test/output/chair3.glb"}}',
        ]
        mock_post.return_value = mock_response

        geometry_requests = [
            GeometryGenerationServerRequest(
                image_path="/test/image1.png",
                output_dir="/test/output",
                prompt="Chair 1",
            ),
            GeometryGenerationServerRequest(
                image_path="/test/image2.png",
                output_dir="/test/output",
                prompt="Chair 2",
            ),
            GeometryGenerationServerRequest(
                image_path="/test/image3.png",
                output_dir="/test/output",
                prompt="Chair 3",
            ),
        ]

        results = list(self.client.generate_geometries(geometry_requests))

        # Should have 3 results (2 successes, 1 error).
        self.assertEqual(len(results), 3)

        # First result should succeed.
        self.assertEqual(results[0][0], 0)
        self.assertIsInstance(results[0][1], GeometryGenerationServerResponse)
        self.assertEqual(results[0][1].geometry_path, "/test/output/chair1.glb")

        # Second result should be an error.
        self.assertEqual(results[1][0], 1)
        self.assertIsInstance(results[1][1], GeometryGenerationError)
        self.assertEqual(results[1][1].index, 1)
        self.assertEqual(results[1][1].error_message, "Generation failed")

        # Third result should succeed.
        self.assertEqual(results[2][0], 2)
        self.assertIsInstance(results[2][1], GeometryGenerationServerResponse)
        self.assertEqual(results[2][1].geometry_path, "/test/output/chair3.glb")

    @patch("requests.Session.post")
    def test_generate_geometries_invalid_json(self, mock_post):
        """Test batch generation with invalid JSON in stream."""
        # Mock streaming response with invalid JSON.
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b"invalid json line",
        ]
        mock_post.return_value = mock_response

        geometry_requests = [
            GeometryGenerationServerRequest(
                image_path="/test/image1.png",
                output_dir="/test/output",
                prompt="Chair 1",
            ),
        ]

        results_iter = self.client.generate_geometries(geometry_requests)

        with self.assertRaises(RuntimeError) as context:
            next(results_iter)

        self.assertIn("Invalid JSON in streaming response", str(context.exception))

    @patch("requests.Session.post")
    @patch("time.sleep")
    def test_generate_geometries_connection_error_with_retry(
        self, mock_sleep, mock_post
    ):
        """Test batch generation connection error with retry logic."""
        # First call raises ConnectionError, second succeeds.
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b'{"index": 0, "status": "success", '
            b'"data": {"geometry_path": "/test/output/chair1.glb"}}',
        ]

        mock_post.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            mock_response,
        ]

        geometry_requests = [
            GeometryGenerationServerRequest(
                image_path="/test/image1.png",
                output_dir="/test/output",
                prompt="Chair 1",
            ),
        ]

        results = list(
            self.client.generate_geometries(geometry_requests, max_retries=2)
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 0)
        self.assertEqual(results[0][1].geometry_path, "/test/output/chair1.glb")
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1


if __name__ == "__main__":
    unittest.main()
