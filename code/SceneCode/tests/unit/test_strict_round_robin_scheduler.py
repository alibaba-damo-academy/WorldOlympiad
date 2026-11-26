import time
import unittest

from unittest.mock import MagicMock

from scenecode.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationServerRequest,
)
from scenecode.agent_utils.geometry_generation_server.server_app import (
    StrictRoundRobinScheduler,
)


class TestStrictRoundRobinScheduler(unittest.TestCase):
    """Test StrictRoundRobinScheduler fairness and ordering logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = StrictRoundRobinScheduler()
        self.mock_callback = MagicMock()

    def test_empty_scheduler(self):
        """Test scheduler behavior when empty."""
        self.assertIsNone(self.scheduler.get_next_request())
        self.assertEqual(self.scheduler.get_queue_size(), 0)
        self.assertEqual(self.scheduler.get_client_count(), 0)

    def test_single_client_single_request(self):
        """Test scheduler with single client and single request."""
        requests = [
            GeometryGenerationServerRequest(
                image_path="/test/image.png",
                output_dir="/test/output",
                prompt="Test object",
            )
        ]

        self.scheduler.add_batch(
            "client_a", requests, self.mock_callback, received_timestamp=time.time()
        )

        # Check initial state.
        self.assertEqual(self.scheduler.get_queue_size(), 1)
        self.assertEqual(self.scheduler.get_client_count(), 1)

        # Get request.
        request = self.scheduler.get_next_request()
        self.assertIsNotNone(request)
        self.assertEqual(request.client_id, "client_a")
        self.assertEqual(request.request_index, 0)
        self.assertEqual(request.request.prompt, "Test object")

        # Scheduler should be empty now.
        self.assertEqual(self.scheduler.get_queue_size(), 0)
        self.assertEqual(self.scheduler.get_client_count(), 0)
        self.assertIsNone(self.scheduler.get_next_request())

    def test_single_client_multiple_requests(self):
        """Test scheduler with single client and multiple requests."""
        requests = [
            GeometryGenerationServerRequest(
                image_path="/test/image1.png",
                output_dir="/test/output",
                prompt="Object 1",
            ),
            GeometryGenerationServerRequest(
                image_path="/test/image2.png",
                output_dir="/test/output",
                prompt="Object 2",
            ),
        ]

        self.scheduler.add_batch(
            "client_a", requests, self.mock_callback, received_timestamp=time.time()
        )

        # Check initial state.
        self.assertEqual(self.scheduler.get_queue_size(), 2)
        self.assertEqual(self.scheduler.get_client_count(), 1)

        # Get first request.
        request1 = self.scheduler.get_next_request()
        self.assertEqual(request1.client_id, "client_a")
        self.assertEqual(request1.request_index, 0)
        self.assertEqual(request1.request.prompt, "Object 1")

        # Get second request.
        request2 = self.scheduler.get_next_request()
        self.assertEqual(request2.client_id, "client_a")
        self.assertEqual(request2.request_index, 1)
        self.assertEqual(request2.request.prompt, "Object 2")

        # Scheduler should be empty now.
        self.assertEqual(self.scheduler.get_queue_size(), 0)
        self.assertEqual(self.scheduler.get_client_count(), 0)

    def test_two_clients_fair_round_robin(self):
        """Test fair round-robin scheduling with two clients."""
        # Client A submits 3 requests.
        requests_a = [
            GeometryGenerationServerRequest(
                image_path=f"/test/image_a{i}.png",
                output_dir="/test/output",
                prompt=f"Object A{i}",
            )
            for i in range(3)
        ]

        # Client B submits 2 requests.
        requests_b = [
            GeometryGenerationServerRequest(
                image_path=f"/test/image_b{i}.png",
                output_dir="/test/output",
                prompt=f"Object B{i}",
            )
            for i in range(2)
        ]

        self.scheduler.add_batch(
            "client_a", requests_a, self.mock_callback, received_timestamp=time.time()
        )
        self.scheduler.add_batch(
            "client_b", requests_b, self.mock_callback, received_timestamp=time.time()
        )

        # Check initial state.
        self.assertEqual(self.scheduler.get_queue_size(), 5)
        self.assertEqual(self.scheduler.get_client_count(), 2)

        # Expected order: A0 → B0 → A1 → B1 → A2
        expected_order = [
            ("client_a", 0, "Object A0"),
            ("client_b", 0, "Object B0"),
            ("client_a", 1, "Object A1"),
            ("client_b", 1, "Object B1"),
            ("client_a", 2, "Object A2"),
        ]

        for expected_client, expected_index, expected_prompt in expected_order:
            request = self.scheduler.get_next_request()
            self.assertIsNotNone(request)
            self.assertEqual(request.client_id, expected_client)
            self.assertEqual(request.request_index, expected_index)
            self.assertEqual(request.request.prompt, expected_prompt)

        # Scheduler should be empty now.
        self.assertEqual(self.scheduler.get_queue_size(), 0)
        self.assertEqual(self.scheduler.get_client_count(), 0)

    def test_three_clients_arrival_order_preservation(self):
        """Test that new clients join at the END of rotation order."""
        # Client A arrives first with 2 requests.
        requests_a = [
            GeometryGenerationServerRequest(
                image_path=f"/test/image_a{i}.png",
                output_dir="/test/output",
                prompt=f"Object A{i}",
            )
            for i in range(2)
        ]

        # Client B arrives second with 2 requests.
        requests_b = [
            GeometryGenerationServerRequest(
                image_path=f"/test/image_b{i}.png",
                output_dir="/test/output",
                prompt=f"Object B{i}",
            )
            for i in range(2)
        ]

        self.scheduler.add_batch(
            "client_a", requests_a, self.mock_callback, received_timestamp=time.time()
        )
        self.scheduler.add_batch(
            "client_b", requests_b, self.mock_callback, received_timestamp=time.time()
        )

        # Process first round: A0 → B0
        request1 = self.scheduler.get_next_request()
        self.assertEqual(request1.client_id, "client_a")
        self.assertEqual(request1.request_index, 0)

        request2 = self.scheduler.get_next_request()
        self.assertEqual(request2.client_id, "client_b")
        self.assertEqual(request2.request_index, 0)

        # Client C arrives with 1 request (should join at END).
        requests_c = [
            GeometryGenerationServerRequest(
                image_path="/test/image_c0.png",
                output_dir="/test/output",
                prompt="Object C0",
            )
        ]
        self.scheduler.add_batch(
            "client_c", requests_c, self.mock_callback, received_timestamp=time.time()
        )

        # Expected remaining order: A1 → B1 → C0
        # (C joins at end, doesn't interrupt A and B)
        expected_remaining = [
            ("client_a", 1, "Object A1"),
            ("client_b", 1, "Object B1"),
            ("client_c", 0, "Object C0"),
        ]

        for expected_client, expected_index, expected_prompt in expected_remaining:
            request = self.scheduler.get_next_request()
            self.assertIsNotNone(request)
            self.assertEqual(request.client_id, expected_client)
            self.assertEqual(request.request_index, expected_index)
            self.assertEqual(request.request.prompt, expected_prompt)

    def test_client_cleanup_after_completion(self):
        """Test that clients are removed after all requests are processed."""
        requests_a = [
            GeometryGenerationServerRequest(
                image_path="/test/image_a0.png",
                output_dir="/test/output",
                prompt="Object A0",
            )
        ]

        requests_b = [
            GeometryGenerationServerRequest(
                image_path="/test/image_b0.png",
                output_dir="/test/output",
                prompt="Object B0",
            ),
            GeometryGenerationServerRequest(
                image_path="/test/image_b1.png",
                output_dir="/test/output",
                prompt="Object B1",
            ),
        ]

        self.scheduler.add_batch(
            "client_a", requests_a, self.mock_callback, received_timestamp=time.time()
        )
        self.scheduler.add_batch(
            "client_b", requests_b, self.mock_callback, received_timestamp=time.time()
        )

        # Initial state: 2 clients, 3 requests total.
        self.assertEqual(self.scheduler.get_client_count(), 2)
        self.assertEqual(self.scheduler.get_queue_size(), 3)

        # Process A0 (Client A should be removed).
        request1 = self.scheduler.get_next_request()
        self.assertEqual(request1.client_id, "client_a")
        self.assertEqual(self.scheduler.get_client_count(), 1)  # A removed
        self.assertEqual(self.scheduler.get_queue_size(), 2)

        # Process B0.
        request2 = self.scheduler.get_next_request()
        self.assertEqual(request2.client_id, "client_b")
        self.assertEqual(self.scheduler.get_client_count(), 1)  # B still active
        self.assertEqual(self.scheduler.get_queue_size(), 1)

        # Process B1 (Client B should be removed).
        request3 = self.scheduler.get_next_request()
        self.assertEqual(request3.client_id, "client_b")
        self.assertEqual(self.scheduler.get_client_count(), 0)  # B removed
        self.assertEqual(self.scheduler.get_queue_size(), 0)

    def test_add_requests_to_existing_client(self):
        """Test adding more requests to an existing client."""
        # Add initial batch for client A.
        initial_requests = [
            GeometryGenerationServerRequest(
                image_path="/test/image_a0.png",
                output_dir="/test/output",
                prompt="Object A0",
            )
        ]
        self.scheduler.add_batch(
            "client_a",
            initial_requests,
            self.mock_callback,
            received_timestamp=time.time(),
        )

        # Add more requests to the same client.
        additional_requests = [
            GeometryGenerationServerRequest(
                image_path="/test/image_a1.png",
                output_dir="/test/output",
                prompt="Object A1",
            ),
            GeometryGenerationServerRequest(
                image_path="/test/image_a2.png",
                output_dir="/test/output",
                prompt="Object A2",
            ),
        ]
        self.scheduler.add_batch(
            "client_a",
            additional_requests,
            self.mock_callback,
            received_timestamp=time.time(),
        )

        # Should have 1 client with 3 total requests.
        self.assertEqual(self.scheduler.get_client_count(), 1)
        self.assertEqual(self.scheduler.get_queue_size(), 3)

        # Process all requests. Note: indices are per-batch, not cumulative.
        expected_indices = [0, 0, 1]  # First batch: A0, Second batch: A1, A2
        expected_prompts = ["Object A0", "Object A1", "Object A2"]

        for i in range(3):
            request = self.scheduler.get_next_request()
            self.assertEqual(request.client_id, "client_a")
            self.assertEqual(request.request_index, expected_indices[i])
            self.assertEqual(request.request.prompt, expected_prompts[i])


if __name__ == "__main__":
    unittest.main()
