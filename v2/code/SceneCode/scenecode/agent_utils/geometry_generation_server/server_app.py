"""Flask application for geometry generation server with multi-GPU support.

This module provides the HTTP interface for geometry generation. It uses a
GPU worker pool to distribute requests across all available GPUs.

CRITICAL: This module must NOT import any CUDA-dependent code at module level.
CUDA imports are deferred to worker processes to enable proper GPU isolation.
"""

import logging
import time
import uuid

from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Callable

import flask

from scenecode.agent_utils.geometry_generation_server.worker_pool import GPUWorkerPool
from scenecode.agent_utils.scheduler import StrictRoundRobinScheduler
from scenecode.agent_utils.wsgi_server_lifecycle import (
    trigger_shutdown_callback_async,
)

from .dataclasses import GeometryGenerationServerRequest, StreamedResult

console_logger = logging.getLogger(__name__)


class GeometryGenerationApp(flask.Flask):
    """Flask application for geometry generation server with multi-GPU support.

    This application manages a pool of GPU worker processes and distributes
    geometry generation requests across them using fair round-robin scheduling.

    The worker pool automatically detects all available GPUs and spawns one
    worker process per GPU. Use CUDA_VISIBLE_DEVICES to control which GPUs
    are used.
    """

    def __init__(
        self,
        use_mini: bool = False,
        backend: str = "hunyuan3d",
        sam3d_config: dict | None = None,
        preload_pipeline: bool = True,
        log_file: Path | None = None,
    ) -> None:
        """Initialize the Flask app with GPU worker pool.

        Args:
            use_mini: Whether to use the mini model variant (0.6B parameters) instead
                of the full model. The mini model is faster with lower memory usage
                but may have reduced quality. Default: False (use full model).
            backend: 3D generation backend to use ("hunyuan3d" or "sam3d").
                Default: "hunyuan3d".
            sam3d_config: Configuration for SAM3D backend. Required if backend="sam3d".
                Should contain sam3_checkpoint and sam3d_checkpoint paths.
            preload_pipeline: Whether to preload pipelines in workers on start.
                Default: True.
            log_file: Optional path to log file for worker logging (e.g., experiment.log).
        """
        super().__init__("geometry_generation_server")

        self._use_mini = use_mini
        self._backend = backend
        self._sam3d_config = sam3d_config
        self._preload_pipeline = preload_pipeline

        # Fair scheduling across clients.
        self._scheduler = StrictRoundRobinScheduler()

        # GPU worker pool (created but not started).
        self._worker_pool = GPUWorkerPool(
            use_mini=use_mini,
            backend=backend,
            sam3d_config=sam3d_config,
            preload_pipeline=preload_pipeline,
            log_file=log_file,
        )

        # Coordinator thread dispatches from scheduler to worker pool.
        self._processing_thread: Thread | None = None
        self._processing_active = False
        self._shutdown_callback: Callable[[], None] | None = None

        # Setup routes.
        self.add_url_rule("/health", "health", self._health_endpoint, methods=["GET"])
        self.add_url_rule(
            "/shutdown", "shutdown", self._shutdown_endpoint, methods=["POST"]
        )
        self.add_url_rule(
            "/generate_geometries",
            "generate_geometries",
            self._generate_geometries_endpoint,
            methods=["POST"],
        )

    def start_processing(self) -> None:
        """Start the GPU worker pool and coordinator thread."""
        if self._processing_active:
            console_logger.warning("Processing already active")
            return

        console_logger.info("Starting geometry generation processing...")

        # Start the worker pool first.
        self._worker_pool.start()
        num_gpus = self._worker_pool.num_workers
        console_logger.info(f"Started worker pool with {num_gpus} GPU(s)")

        # Start coordinator thread.
        self._processing_active = True
        self._processing_thread = Thread(target=self._process_queue, daemon=False)
        self._processing_thread.start()

        console_logger.info("Geometry generation processing started")

    def stop_processing(self) -> None:
        """Stop the coordinator thread and GPU worker pool gracefully."""
        if not self._processing_active:
            return

        console_logger.info("Stopping geometry generation processing...")
        self._processing_active = False

        # Wait for coordinator thread to complete.
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5)
            if self._processing_thread.is_alive():
                console_logger.warning("Coordinator thread did not stop gracefully")

        # Stop the worker pool.
        self._worker_pool.stop()

        console_logger.info("Geometry generation processing stopped")

    def _process_queue(self) -> None:
        """Dispatch requests from scheduler to worker pool.

        This runs in a coordinator thread, pulling requests from the fair
        scheduler and dispatching them to the GPU worker pool. The dispatch
        blocks until a worker is available, preserving fair ordering.
        """
        try:
            console_logger.info("Coordinator thread started")

            while self._processing_active:
                # Get next request from fair scheduler.
                queued_request = self._scheduler.get_next_request()
                if queued_request:
                    console_logger.debug(
                        f"Dispatching request from {queued_request.client_id}: "
                        f"{queued_request.request.prompt}"
                    )

                    # Dispatch to worker pool (blocks until worker available).
                    self._worker_pool.submit_request(
                        request=queued_request.request,
                        callback=queued_request.callback,
                        request_index=queued_request.request_index,
                        received_timestamp=queued_request.received_timestamp,
                    )
                else:
                    # No requests available, sleep briefly.
                    time.sleep(0.1)

        except Exception as e:
            console_logger.error(f"Coordinator thread failed: {e}")

        finally:
            console_logger.info("Coordinator thread stopped")

    def _health_endpoint(self) -> flask.Response:
        """Health check endpoint with pool and scheduler details."""
        scheduler_queue_size = self._scheduler.get_queue_size()
        active_clients = self._scheduler.get_client_count()
        pool_stats = self._worker_pool.get_stats()

        return flask.jsonify(
            {
                "status": "healthy",
                "num_workers": pool_stats.num_workers,
                "scheduler_queue_size": scheduler_queue_size,
                "active_clients": active_clients,
                "processing_active": self._processing_active,
                "total_requests": pool_stats.total_requests,
                "completed_requests": pool_stats.completed_requests,
                "failed_requests": pool_stats.failed_requests,
                "avg_processing_time_seconds": pool_stats.avg_processing_time_s,
                "avg_end_to_end_latency_seconds": pool_stats.avg_end_to_end_latency_s,
                "avg_queue_wait_seconds": pool_stats.avg_queue_wait_s,
                "max_queue_wait_seconds": pool_stats.max_queue_wait_s,
                "workers": pool_stats.worker_details,
            }
        )

    def register_shutdown_callback(
        self, callback: Callable[[], None] | None
    ) -> None:
        """Register the manager-owned shutdown callback."""
        self._shutdown_callback = callback

    def _shutdown_endpoint(self) -> flask.Response:
        """Shutdown endpoint for graceful server termination."""
        console_logger.info("Shutdown endpoint called")
        if not trigger_shutdown_callback_async(
            self._shutdown_callback, "GeometryGenerationShutdown"
        ):
            console_logger.warning("Shutdown callback not registered.")
            return (
                flask.jsonify({"status": "error", "message": "shutdown unavailable"}),
                503,
            )

        return flask.jsonify({"status": "shutting down"}), 200

    def _generate_geometries_endpoint(self) -> flask.Response:
        """Handle batch geometry generation requests with streaming response."""
        try:
            data = flask.request.json
            if not data:
                return flask.jsonify({"error": "No JSON data provided"}), 400

            if not isinstance(data, list):
                return flask.jsonify({"error": "Expected a list of requests"}), 400

            if len(data) == 0:
                return flask.jsonify({"error": "Empty request list"}), 400

            # Validate each request in the batch.
            required_fields = ["image_path", "output_dir", "prompt"]
            for i, request_data in enumerate(data):
                if not isinstance(request_data, dict):
                    return (
                        flask.jsonify({"error": f"Request {i} is not an object"}),
                        400,
                    )

                for field in required_fields:
                    if field not in request_data:
                        return (
                            flask.jsonify(
                                {"error": f"Request {i} missing field: {field}"}
                            ),
                            400,
                        )

            # Create batch request.
            batch_requests = [
                GeometryGenerationServerRequest(**req_data) for req_data in data
            ]

            # Use scene_id for fair scheduling if provided, otherwise generate UUID.
            # All requests in a batch share the same scene_id, so check first request.
            first_scene_id = batch_requests[0].scene_id if batch_requests else None
            batch_id = first_scene_id if first_scene_id else str(uuid.uuid4())

            # Create result queue for this client.
            client_result_queue: Queue = Queue()
            results_received = 0
            batch_size = len(batch_requests)

            def result_callback(index: int, result: tuple[str, dict]) -> None:
                """Route results back to this client's queue."""
                nonlocal results_received
                client_result_queue.put((index, result))

            # Add batch to fair scheduler with timestamp for latency tracking.
            received_timestamp = time.time()
            self._scheduler.add_batch(
                client_id=batch_id,
                requests=batch_requests,
                callback=result_callback,
                received_timestamp=received_timestamp,
            )

            def generate():
                """Generator function for streaming NDJSON responses."""
                nonlocal results_received

                while results_received < batch_size:
                    index, (status, result_data) = client_result_queue.get()

                    if status == "success":
                        streamed_result = StreamedResult(
                            index=index, status="success", data=result_data
                        )
                    else:
                        streamed_result = StreamedResult(
                            index=index, status="error", error=result_data
                        )

                    yield streamed_result.to_json() + "\n"
                    results_received += 1

            return flask.Response(generate(), mimetype="application/x-ndjson")

        except Exception as e:
            console_logger.error(f"Batch request handling failed: {e}")
            return flask.jsonify({"error": str(e)}), 500
