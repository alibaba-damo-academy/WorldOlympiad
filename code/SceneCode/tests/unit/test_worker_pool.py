"""Unit tests for GPU worker pool."""

import subprocess
import unittest

from unittest.mock import MagicMock, patch

from scenecode.agent_utils.geometry_generation_server.dataclasses import (
    GeometryGenerationServerRequest,
)
from scenecode.agent_utils.geometry_generation_server.gpu_worker import (
    ShutdownRequest,
    WorkerReady,
    WorkRequest,
    WorkResult,
)
from scenecode.agent_utils.geometry_generation_server.worker_pool import (
    GPUWorkerPool,
    PoolStats,
)


class TestGPUDetection(unittest.TestCase):
    """Test GPU detection logic."""

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "scenecode.agent_utils.geometry_generation_server.worker_pool.subprocess.run"
    )
    def test_detect_gpu_ids_multiple_gpus(self, mock_run):
        """Test detection of multiple GPUs via nvidia-smi."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0\n1\n2\n3\n",
        )

        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        self.assertEqual(gpu_ids, [0, 1, 2, 3])
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        self.assertIn("nvidia-smi", call_args[0][0])

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "scenecode.agent_utils.geometry_generation_server.worker_pool.subprocess.run"
    )
    def test_detect_gpu_ids_single_gpu(self, mock_run):
        """Test detection of single GPU."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0\n",
        )

        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        self.assertEqual(gpu_ids, [0])

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "scenecode.agent_utils.geometry_generation_server.worker_pool.subprocess.run"
    )
    def test_detect_gpu_ids_nvidia_smi_fails(self, mock_run):
        """Test fallback when nvidia-smi fails."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")

        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        # Should fall back to GPU 0.
        self.assertEqual(gpu_ids, [0])

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "scenecode.agent_utils.geometry_generation_server.worker_pool.subprocess.run"
    )
    def test_detect_gpu_ids_command_not_found(self, mock_run):
        """Test fallback when nvidia-smi is not installed."""
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")

        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        # Should fall back to GPU 0.
        self.assertEqual(gpu_ids, [0])

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "scenecode.agent_utils.geometry_generation_server.worker_pool.subprocess.run"
    )
    def test_detect_gpu_ids_timeout(self, mock_run):
        """Test fallback when nvidia-smi times out."""
        mock_run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 5)

        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        # Should fall back to GPU 0.
        self.assertEqual(gpu_ids, [0])

    @patch.dict("os.environ", {}, clear=True)
    @patch(
        "scenecode.agent_utils.geometry_generation_server.worker_pool.subprocess.run"
    )
    def test_detect_gpu_ids_empty_output(self, mock_run):
        """Test handling of empty nvidia-smi output."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        # Empty output should fall back to GPU 0.
        self.assertEqual(gpu_ids, [0])

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1,2,3,4,5,6,7"})
    def test_detect_gpu_ids_respects_cuda_visible_devices(self):
        """Test that CUDA_VISIBLE_DEVICES is respected."""
        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        self.assertEqual(gpu_ids, [1, 2, 3, 4, 5, 6, 7])

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "3"})
    def test_detect_gpu_ids_single_cuda_visible_device(self):
        """Test single GPU in CUDA_VISIBLE_DEVICES."""
        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        self.assertEqual(gpu_ids, [3])

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "  2 , 5 , 7  "})
    def test_detect_gpu_ids_handles_whitespace(self):
        """Test that whitespace in CUDA_VISIBLE_DEVICES is handled."""
        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        self.assertEqual(gpu_ids, [2, 5, 7])

    @patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""})
    @patch(
        "scenecode.agent_utils.geometry_generation_server.worker_pool.subprocess.run"
    )
    def test_detect_gpu_ids_empty_cuda_visible_devices(self, mock_run):
        """Test empty CUDA_VISIBLE_DEVICES falls back to nvidia-smi."""
        mock_run.return_value = MagicMock(returncode=0, stdout="0\n1\n")

        gpu_ids = GPUWorkerPool._detect_gpu_ids()

        self.assertEqual(gpu_ids, [0, 1])
        mock_run.assert_called_once()


class TestPoolStats(unittest.TestCase):
    """Test PoolStats dataclass."""

    def test_pool_stats_creation(self):
        """Test PoolStats creation with all fields."""
        stats = PoolStats(
            num_workers=4,
            total_requests=100,
            completed_requests=95,
            failed_requests=5,
            avg_processing_time_s=25.5,
            avg_end_to_end_latency_s=30.0,
            avg_queue_wait_s=4.5,
            max_queue_wait_s=10.0,
            worker_details=[
                {"gpu_id": 0, "pid": 1234, "alive": True},
                {"gpu_id": 1, "pid": 1235, "alive": True},
            ],
        )

        self.assertEqual(stats.num_workers, 4)
        self.assertEqual(stats.total_requests, 100)
        self.assertEqual(stats.completed_requests, 95)
        self.assertEqual(stats.failed_requests, 5)
        self.assertEqual(stats.avg_processing_time_s, 25.5)
        self.assertEqual(stats.avg_end_to_end_latency_s, 30.0)
        self.assertEqual(stats.avg_queue_wait_s, 4.5)
        self.assertEqual(stats.max_queue_wait_s, 10.0)
        self.assertEqual(len(stats.worker_details), 2)

    def test_pool_stats_none_avg_time(self):
        """Test PoolStats with no average processing time."""
        stats = PoolStats(
            num_workers=1,
            total_requests=0,
            completed_requests=0,
            failed_requests=0,
            avg_processing_time_s=None,
            avg_end_to_end_latency_s=None,
            avg_queue_wait_s=None,
            max_queue_wait_s=None,
            worker_details=[],
        )

        self.assertIsNone(stats.avg_processing_time_s)
        self.assertIsNone(stats.avg_end_to_end_latency_s)
        self.assertIsNone(stats.avg_queue_wait_s)
        self.assertIsNone(stats.max_queue_wait_s)


class TestWorkRequestResult(unittest.TestCase):
    """Test WorkRequest and WorkResult dataclasses."""

    def test_work_request_creation(self):
        """Test WorkRequest creation."""
        request = GeometryGenerationServerRequest(
            image_path="/test/image.png",
            output_dir="/test/output",
            prompt="A wooden chair",
        )

        work_request = WorkRequest(
            request_id="test-123",
            request=request,
            received_timestamp=1234567890.0,
        )

        self.assertEqual(work_request.request_id, "test-123")
        self.assertEqual(work_request.request.prompt, "A wooden chair")
        self.assertEqual(work_request.received_timestamp, 1234567890.0)

    def test_work_result_success(self):
        """Test WorkResult for successful request."""
        result = WorkResult(
            request_id="test-123",
            worker_id=0,
            status="success",
            data={"geometry_path": "/test/output/chair.glb"},
            error=None,
        )

        self.assertEqual(result.request_id, "test-123")
        self.assertEqual(result.worker_id, 0)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.data["geometry_path"], "/test/output/chair.glb")
        self.assertIsNone(result.error)

    def test_work_result_error(self):
        """Test WorkResult for failed request."""
        result = WorkResult(
            request_id="test-456",
            worker_id=1,
            status="error",
            data=None,
            error="Generation failed: out of memory",
        )

        self.assertEqual(result.status, "error")
        self.assertIsNone(result.data)
        self.assertEqual(result.error, "Generation failed: out of memory")


class TestShutdownRequest(unittest.TestCase):
    """Test ShutdownRequest sentinel class."""

    def test_shutdown_request_is_distinct_type(self):
        """Test that ShutdownRequest is distinguishable from other types."""
        shutdown = ShutdownRequest()
        work_request = WorkRequest(
            request_id="test",
            request=GeometryGenerationServerRequest(
                image_path="/test/image.png",
                output_dir="/test/output",
                prompt="test",
            ),
            received_timestamp=1234567890.0,
        )

        self.assertIsInstance(shutdown, ShutdownRequest)
        self.assertNotIsInstance(work_request, ShutdownRequest)
        self.assertNotIsInstance("string", ShutdownRequest)


class TestWorkerReady(unittest.TestCase):
    """Test WorkerReady signal class."""

    def test_worker_ready_creation(self):
        """Test WorkerReady creation with worker ID."""
        ready = WorkerReady(worker_id=3)

        self.assertEqual(ready.worker_id, 3)

    def test_worker_ready_is_distinct_type(self):
        """Test that WorkerReady is distinguishable from other message types."""
        ready = WorkerReady(worker_id=0)
        shutdown = ShutdownRequest()
        work_result = WorkResult(
            request_id="test",
            worker_id=0,
            status="success",
            data={"geometry_path": "/test/output.glb"},
            error=None,
        )

        self.assertIsInstance(ready, WorkerReady)
        self.assertNotIsInstance(shutdown, WorkerReady)
        self.assertNotIsInstance(work_result, WorkerReady)


class TestWorkerPoolInitialization(unittest.TestCase):
    """Test GPUWorkerPool initialization (without starting)."""

    @patch.object(GPUWorkerPool, "_detect_gpu_ids", return_value=[0, 1, 2, 3])
    def test_pool_initialization_defaults(self, mock_detect):
        """Test pool initialization with default parameters."""
        pool = GPUWorkerPool()

        self.assertEqual(pool.num_workers, 4)
        self.assertEqual(pool._use_mini, False)
        self.assertEqual(pool._backend, "hunyuan3d")
        self.assertIsNone(pool._sam3d_config)
        self.assertTrue(pool._preload_pipeline)
        # Verify multiprocessing context is created.
        self.assertIsNotNone(pool._mp_ctx)

    @patch.object(GPUWorkerPool, "_detect_gpu_ids", return_value=[0, 1])
    def test_pool_initialization_custom_params(self, mock_detect):
        """Test pool initialization with custom parameters."""
        sam3d_config = {"sam3_checkpoint": "/path/to/sam3.pt"}

        pool = GPUWorkerPool(
            use_mini=True,
            backend="sam3d",
            sam3d_config=sam3d_config,
            preload_pipeline=False,
        )

        self.assertEqual(pool.num_workers, 2)
        self.assertTrue(pool._use_mini)
        self.assertEqual(pool._backend, "sam3d")
        self.assertEqual(pool._sam3d_config, sam3d_config)
        self.assertFalse(pool._preload_pipeline)

    @patch.object(GPUWorkerPool, "_detect_gpu_ids", return_value=[0])
    def test_pool_stats_before_start(self, mock_detect):
        """Test getting stats before pool is started."""
        pool = GPUWorkerPool()

        stats = pool.get_stats()

        self.assertEqual(stats.num_workers, 1)
        self.assertEqual(stats.total_requests, 0)
        self.assertEqual(stats.completed_requests, 0)
        self.assertEqual(stats.failed_requests, 0)
        self.assertIsNone(stats.avg_processing_time_s)
        self.assertEqual(stats.worker_details, [])


if __name__ == "__main__":
    unittest.main()
