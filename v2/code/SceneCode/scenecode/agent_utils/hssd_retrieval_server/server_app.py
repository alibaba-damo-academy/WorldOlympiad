"""Flask application for HSSD retrieval server with round-robin scheduling.

Adapted from geometry_generation_server for CLIP-based semantic search.
"""

import logging
import time
import uuid

from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Callable

import flask
import numpy as np

from scenecode.agent_utils.hssd_retrieval.retrieval import HssdRetriever
from scenecode.agent_utils.scheduler import QueuedRequest, StrictRoundRobinScheduler
from scenecode.agent_utils.wsgi_server_lifecycle import (
    trigger_shutdown_callback_async,
)

from .dataclasses import (
    HssdRetrievalResult,
    HssdRetrievalServerRequest,
    HssdRetrievalServerResponse,
    StreamedResult,
)

console_logger = logging.getLogger(__name__)


class HssdRetrievalApp(flask.Flask):
    """Flask application for HSSD retrieval with round-robin scheduling."""

    def __init__(
        self,
        preload_retriever: bool = True,
        hssd_data_path: str | None = None,
        hssd_preprocessed_path: str | None = None,
        hssd_top_k: int = 5,
        clip_device: str | None = None,
    ) -> None:
        """Initialize Flask app.

        Args:
            preload_retriever: Whether to preload HSSD retriever (and CLIP model)
                on startup. Default: True for consistent latency.
            hssd_data_path: Path to HSSD models directory. If None, uses environment
                variable HSSD_DATA_PATH or default "data/hssd-models".
            hssd_preprocessed_path: Path to preprocessed data directory. If None,
                uses environment variable HSSD_PREPROCESSED_PATH or default
                "data/preprocessed".
            hssd_top_k: Number of top CLIP candidates before size ranking (default: 5).
            clip_device: Target device for CLIP model (e.g., "cuda:0"). If None,
                uses default (cuda if available, else cpu).
        """
        super().__init__("hssd_retrieval_server")

        self._retriever: HssdRetriever | None = None

        # Store HSSD config parameters for lazy initialization.
        self._hssd_data_path = hssd_data_path
        self._hssd_preprocessed_path = hssd_preprocessed_path
        self._hssd_top_k = hssd_top_k
        self._clip_device = clip_device

        self._scheduler = StrictRoundRobinScheduler()
        self._processing_thread: Thread | None = None
        self._processing_active = False
        self._current_processing: str | None = None
        self._shutdown_callback: Callable[[], None] | None = None

        # Metrics.
        self._total_requests = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._request_times: list[float] = []

        # Preload retriever if requested.
        if preload_retriever:
            console_logger.info("Preloading HSSD retriever and CLIP model...")
            start_time = time.time()
            self._get_retriever()
            load_time = time.time() - start_time
            console_logger.info(
                f"HSSD retriever preloaded in {load_time:.2f}s " "(includes CLIP model)"
            )

        # Setup routes.
        self.add_url_rule("/health", "health", self._health_endpoint, methods=["GET"])
        self.add_url_rule(
            "/shutdown", "shutdown", self._shutdown_endpoint, methods=["POST"]
        )
        self.add_url_rule(
            "/retrieve_objects",
            "retrieve_objects",
            self._retrieve_objects_endpoint,
            methods=["POST"],
        )

    def _get_retriever(self) -> HssdRetriever:
        """Get or create HSSD retriever (singleton per server process)."""
        if self._retriever is None:
            import os

            from scenecode.agent_utils.hssd_retrieval.config import HssdConfig

            # Use provided paths or fall back to environment variables/defaults.
            data_path = self._hssd_data_path or os.environ.get(
                "HSSD_DATA_PATH", "data/hssd-models"
            )
            preprocessed_path = self._hssd_preprocessed_path or os.environ.get(
                "HSSD_PREPROCESSED_PATH", "data/preprocessed"
            )

            # Resolve relative paths to project root.
            data_path = Path(data_path)
            preprocessed_path = Path(preprocessed_path)
            if not data_path.is_absolute():
                project_root = Path(__file__).parent.parent.parent.parent
                data_path = project_root / data_path
            if not preprocessed_path.is_absolute():
                project_root = Path(__file__).parent.parent.parent.parent
                preprocessed_path = project_root / preprocessed_path

            config = HssdConfig(
                data_path=data_path,
                preprocessed_path=preprocessed_path,
                use_top_k=self._hssd_top_k,
                object_type_mapping=None,  # Will use defaults from __post_init__
            )
            self._retriever = HssdRetriever(
                config=config, clip_device=self._clip_device
            )
        return self._retriever

    def start_processing(self) -> None:
        """Start background processing thread."""
        if self._processing_active:
            console_logger.warning("Processing thread already active")
            return

        console_logger.info("Starting HSSD retrieval processing thread")
        self._processing_active = True
        self._processing_thread = Thread(target=self._process_queue, daemon=False)
        self._processing_thread.start()

    def stop_processing(self) -> None:
        """Stop background processing thread gracefully."""
        if not self._processing_active:
            return

        console_logger.info("Stopping HSSD retrieval processing thread")
        self._processing_active = False

        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5)
            if self._processing_thread.is_alive():
                console_logger.warning("Processing thread did not stop gracefully")

        # Clean up GPU memory by clearing CLIP model.
        if self._retriever is not None:
            console_logger.info("Cleaning up GPU memory (CLIP model)...")
            self._retriever = None

    def _process_queue(self) -> None:
        """Process retrieval requests using round-robin scheduling."""
        try:
            console_logger.info("HSSD retrieval processing started")

            while self._processing_active:
                request = self._scheduler.get_next_request()
                if request:
                    self._process_round_robin_request(request)
                else:
                    time.sleep(0.1)

        except Exception as e:
            console_logger.error(f"Processing queue failed: {e}")

        finally:
            self._current_processing = None
            console_logger.info("HSSD retrieval processing thread stopped")

    def _process_round_robin_request(self, queued_request: QueuedRequest) -> None:
        """Process a single retrieval request."""
        try:
            self._current_processing = (
                f"{queued_request.client_id}[{queued_request.request_index}]: "
                f"{queued_request.request.object_description}"
            )

            console_logger.info(
                f"Processing request from {queued_request.client_id}: "
                f"{queued_request.request.object_description}"
            )

            start_time = time.time()
            result = self._retrieve_internal(queued_request.request)
            queued_request.callback(
                queued_request.request_index, ("success", result.to_dict())
            )

            self._completed_requests += 1
            processing_time = time.time() - start_time
            self._request_times.append(processing_time)
            if len(self._request_times) > 100:
                self._request_times.pop(0)

        except Exception as e:
            console_logger.error(
                f"Request from {queued_request.client_id} "
                f"[{queued_request.request_index}] failed: {e}"
            )
            queued_request.callback(queued_request.request_index, ("error", str(e)))
            self._failed_requests += 1

        finally:
            self._current_processing = None

    def _retrieve_internal(
        self, request: HssdRetrievalServerRequest
    ) -> HssdRetrievalServerResponse:
        """Perform HSSD retrieval with CLIP.

        Returns candidates sorted by bbox_score (best first). Number of candidates
        is controlled by request.num_candidates (default 1 for non-router path,
        higher for router validation retries).

        Args:
            request: Retrieval request with description and constraints.

        Returns:
            Response with matching HSSD objects and their exported mesh paths.

        Raises:
            RuntimeError: If retrieval fails.
        """
        retriever = self._get_retriever()

        # Convert dimensions tuple to numpy array if provided.
        desired_dimensions = None
        if request.desired_dimensions:
            desired_dimensions = np.array(request.desired_dimensions)

        # Retrieve candidates sorted by bbox_score, limited by num_candidates.
        candidates = retriever.retrieve_multiple(
            description=request.object_description,
            object_type=request.object_type,
            desired_dimensions=desired_dimensions,
            max_candidates=request.num_candidates,
        )

        if not candidates:
            raise ValueError(f"No candidates found for '{request.object_description}'")

        # Export candidate meshes to client-specified output directory.
        if not request.output_dir:
            raise ValueError("output_dir must be specified in request")
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Derive category from HSSD config mapping.
        category = retriever.config.object_type_mapping.get(
            request.object_type.upper(), "unknown"
        )

        results: list[HssdRetrievalResult] = []
        for candidate in candidates:
            mesh_filename = f"{candidate.mesh_id}.glb"
            mesh_path = output_dir / mesh_filename
            candidate.mesh.export(str(mesh_path))

            console_logger.debug(f"Exported candidate mesh to {mesh_path}")

            result = HssdRetrievalResult(
                mesh_path=str(mesh_path),
                hssd_id=candidate.mesh_id,
                object_name=request.object_description,
                similarity_score=float(candidate.clip_score),
                size=tuple(candidate.mesh.extents.tolist()),
                category=category,
            )
            results.append(result)

        console_logger.info(
            f"Returning {len(results)} candidates for '{request.object_description}'"
        )

        return HssdRetrievalServerResponse(
            results=results, query_description=request.object_description
        )

    def _health_endpoint(self) -> flask.Response:
        """Health check endpoint."""
        scheduler_queue_size = self._scheduler.get_queue_size()
        active_clients = self._scheduler.get_client_count()
        pending_requests = scheduler_queue_size + (1 if self._current_processing else 0)

        avg_processing_time = None
        if self._request_times:
            avg_processing_time = sum(self._request_times) / len(self._request_times)

        return flask.jsonify(
            {
                "status": "healthy",
                "scheduler_queue_size": scheduler_queue_size,
                "active_clients": active_clients,
                "pending_requests": pending_requests,
                "currently_processing": self._current_processing,
                "total_requests": self._total_requests,
                "completed_requests": self._completed_requests,
                "failed_requests": self._failed_requests,
                "processing_active": self._processing_active,
                "avg_processing_time_seconds": avg_processing_time,
                "retriever_loaded": self._retriever is not None,
            }
        )

    def register_shutdown_callback(
        self, callback: Callable[[], None] | None
    ) -> None:
        """Register the manager-owned shutdown callback."""
        self._shutdown_callback = callback

    def _shutdown_endpoint(self) -> flask.Response:
        """Shutdown endpoint for graceful termination."""
        console_logger.info("Shutdown endpoint called")
        if not trigger_shutdown_callback_async(
            self._shutdown_callback, "HssdRetrievalShutdown"
        ):
            console_logger.warning("Shutdown callback not registered.")
            return (
                flask.jsonify({"status": "error", "message": "shutdown unavailable"}),
                503,
            )

        return flask.jsonify({"status": "shutting down"}), 200

    def _retrieve_objects_endpoint(self) -> flask.Response:
        """Handle batch retrieval requests with streaming response."""
        try:
            data = flask.request.json
            if not data:
                return flask.jsonify({"error": "No JSON data provided"}), 400

            if not isinstance(data, list):
                return flask.jsonify({"error": "Expected a list of requests"}), 400

            if len(data) == 0:
                return flask.jsonify({"error": "Empty request list"}), 400

            # Validate requests.
            required_fields = ["object_description", "object_type", "output_dir"]
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

            # Create batch.
            batch_requests = [
                HssdRetrievalServerRequest(**req_data) for req_data in data
            ]

            # Use scene_id for fair scheduling if provided, otherwise generate UUID.
            # All requests in a batch share the same scene_id, so check first request.
            first_scene_id = batch_requests[0].scene_id if batch_requests else None
            batch_id = first_scene_id if first_scene_id else str(uuid.uuid4())

            client_result_queue = Queue()
            results_received = 0
            batch_size = len(batch_requests)

            def result_callback(index: int, result: tuple[str, dict]) -> None:
                """Route results to client queue."""
                nonlocal results_received
                client_result_queue.put((index, result))

            # Add to scheduler.
            self._scheduler.add_batch(
                client_id=batch_id,
                requests=batch_requests,
                callback=result_callback,
                received_timestamp=time.time(),
            )

            self._total_requests += batch_size

            def generate():
                """Stream NDJSON responses."""
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
