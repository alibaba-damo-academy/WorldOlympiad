import unittest

from unittest.mock import MagicMock, patch

import requests

from scenecode.agent_utils.hssd_retrieval_server.client import HssdRetrievalClient
from scenecode.agent_utils.hssd_retrieval_server.dataclasses import (
    HssdRetrievalServerRequest,
)


class TestHssdRetrievalClient(unittest.TestCase):
    """Test HssdRetrievalClient functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = HssdRetrievalClient()
        self.test_request = HssdRetrievalServerRequest(
            object_description="Modern wooden desk",
            object_type="FURNITURE",
            output_dir="/tmp/hssd_test",
            desired_dimensions=(1.2, 0.6, 0.75),
        )

    def test_initialization(self):
        """Test client initialization."""
        self.assertEqual(self.client.base_url, "http://127.0.0.1:7001")
        self.assertIsNotNone(self.client.session)

    @patch("requests.Session.get")
    def test_health_check_success(self, mock_get):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.client.health_check()

        self.assertTrue(result)
        mock_get.assert_called_once_with("http://127.0.0.1:7001/health", timeout=5)

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
    def test_retrieve_success(self, mock_post):
        """Test successful batch HSSD retrieval."""
        # Mock streaming response with NDJSON.
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b'{"index": 0, "status": "success", '
            b'"data": {"results": [{"mesh_path": "/tmp/hssd/desk1.glb", '
            b'"hssd_id": "abc123", "object_name": "Modern wooden desk", '
            b'"similarity_score": 0.95, "size": [1.2, 0.6, 0.75], '
            b'"category": "tables"}], "query_description": "Modern wooden desk"}}',
            b'{"index": 1, "status": "success", '
            b'"data": {"results": [{"mesh_path": "/tmp/hssd/chair1.glb", '
            b'"hssd_id": "def456", "object_name": "Office chair", '
            b'"similarity_score": 0.92, "size": [0.6, 0.6, 1.0], '
            b'"category": "seating"}], "query_description": "Office chair"}}',
        ]
        mock_post.return_value = mock_response

        retrieval_requests = [
            HssdRetrievalServerRequest(
                object_description="Modern wooden desk",
                object_type="FURNITURE",
                output_dir="/tmp/hssd_test",
                desired_dimensions=(1.2, 0.6, 0.75),
            ),
            HssdRetrievalServerRequest(
                object_description="Office chair",
                object_type="FURNITURE",
                output_dir="/tmp/hssd_test",
                desired_dimensions=(0.6, 0.6, 1.0),
            ),
        ]

        results = list(self.client.retrieve_objects(retrieval_requests))

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 0)  # Index
        self.assertEqual(results[0][1].results[0].mesh_path, "/tmp/hssd/desk1.glb")
        self.assertEqual(results[0][1].results[0].hssd_id, "abc123")
        self.assertEqual(results[1][0], 1)  # Index
        self.assertEqual(results[1][1].results[0].mesh_path, "/tmp/hssd/chair1.glb")

        mock_post.assert_called_once_with(
            "http://127.0.0.1:7001/retrieve_objects",
            json=[req.to_dict() for req in retrieval_requests],
            stream=True,
            timeout=(10, 3600),
        )

    @patch("requests.Session.post")
    def test_retrieve_empty_list_error(self, mock_post):
        """Test batch retrieval with empty request list."""
        with self.assertRaises(ValueError) as context:
            list(self.client.retrieve_objects([]))

        self.assertIn("Requests list cannot be empty", str(context.exception))
        mock_post.assert_not_called()

    @patch("requests.Session.post")
    def test_retrieve_error_in_stream(self, mock_post):
        """Test batch retrieval with error in streaming response."""
        # Mock streaming response with an error.
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b'{"index": 0, "status": "success", '
            b'"data": {"results": [{"mesh_path": "/tmp/hssd/desk1.glb", '
            b'"hssd_id": "abc123", "object_name": "Modern wooden desk", '
            b'"similarity_score": 0.95, "size": [1.2, 0.6, 0.75], '
            b'"category": "tables"}], "query_description": "Modern wooden desk"}}',
            b'{"index": 1, "status": "error", "error": "Retrieval failed"}',
        ]
        mock_post.return_value = mock_response

        retrieval_requests = [
            HssdRetrievalServerRequest(
                object_description="Modern wooden desk",
                object_type="FURNITURE",
                output_dir="/tmp/hssd_test",
            ),
            HssdRetrievalServerRequest(
                object_description="Office chair",
                object_type="FURNITURE",
                output_dir="/tmp/hssd_test",
            ),
        ]

        results_iter = self.client.retrieve_objects(retrieval_requests)

        # First result should succeed.
        first_result = next(results_iter)
        self.assertEqual(first_result[0], 0)
        self.assertEqual(first_result[1].results[0].mesh_path, "/tmp/hssd/desk1.glb")

        # Second result should raise RuntimeError.
        with self.assertRaises(RuntimeError) as context:
            next(results_iter)

        self.assertIn(
            "HSSD retrieval failed for request 1",
            str(context.exception),
        )
        self.assertIn("Retrieval failed", str(context.exception))

    @patch("requests.Session.post")
    def test_retrieve_invalid_json(self, mock_post):
        """Test batch retrieval with invalid JSON in stream."""
        # Mock streaming response with invalid JSON.
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b"invalid json line",
        ]
        mock_post.return_value = mock_response

        retrieval_requests = [
            HssdRetrievalServerRequest(
                object_description="Modern wooden desk",
                object_type="FURNITURE",
                output_dir="/tmp/hssd_test",
            ),
        ]

        results_iter = self.client.retrieve_objects(retrieval_requests)

        with self.assertRaises(RuntimeError) as context:
            next(results_iter)

        self.assertIn("Invalid JSON in streaming response", str(context.exception))

    @patch("requests.Session.post")
    @patch("time.sleep")
    def test_retrieve_connection_error_with_retry(self, mock_sleep, mock_post):
        """Test batch retrieval connection error with retry logic."""
        # First call raises ConnectionError, second succeeds.
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b'{"index": 0, "status": "success", '
            b'"data": {"results": [{"mesh_path": "/tmp/hssd/desk1.glb", '
            b'"hssd_id": "abc123", "object_name": "Modern wooden desk", '
            b'"similarity_score": 0.95, "size": [1.2, 0.6, 0.75], '
            b'"category": "tables"}], "query_description": "Modern wooden desk"}}',
        ]

        mock_post.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            mock_response,
        ]

        retrieval_requests = [
            HssdRetrievalServerRequest(
                object_description="Modern wooden desk",
                object_type="FURNITURE",
                output_dir="/tmp/hssd_test",
            ),
        ]

        results = list(self.client.retrieve_objects(retrieval_requests, max_retries=2))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 0)
        self.assertEqual(results[0][1].results[0].mesh_path, "/tmp/hssd/desk1.glb")
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1


if __name__ == "__main__":
    unittest.main()
