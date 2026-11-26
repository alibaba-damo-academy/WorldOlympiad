"""Flask application for materials retrieval server with round-robin scheduling.

Provides CLIP-based semantic search over the AmbientCG material library.
"""

import logging
import shutil
import time
import uuid

from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Callable

import flask

from omegaconf import DictConfig

from scenecode.agent_utils.clip_embeddings import reset_clip_model_cache
from scenecode.agent_utils.materials_retrieval_server.config import MaterialsConfig
from scenecode.agent_utils.materials_retrieval_server.retrieval import (
    MaterialsRetriever,
)
from scenecode.agent_utils.scheduler import QueuedRequest, StrictRoundRobinScheduler
from scenecode.agent_utils.wsgi_server_lifecycle import (
    trigger_shutdown_callback_async,
)
from scenecode.utils.material import Material

from .dataclasses import (
    MaterialRetrievalResult,
    MaterialsRetrievalServerRequest,
    MaterialsRetrievalServerResponse,
    StreamedResult,
)

console_logger = logging.getLogger(__name__)


class MaterialsRetrievalApp(flask.Flask):
    """Flask application for materials retrieval with round-robin scheduling."""

    def __init__(
        self,
        preload_retriever: bool = True,
        materials_config: MaterialsConfig | DictConfig | None = None,
        clip_device: str | None = None,
    ) -> None:
        """Initialize Flask app.

        Args:
            preload_retriever: Whether to preload materials retriever (and CLIP
                model) on startup. Default: True for consistent latency.
            materials_config: Configuration for materials retrieval. Can be
                MaterialsConfig or DictConfig from Hydra. If None, uses default
                configuration from project defaults.
            clip_device: Target device for CLIP model (e.g., "cuda:0"). If None,
                uses default (cuda if available, else cpu).
        """
        super().__init__("materials_retrieval_server")

        self._retriever: MaterialsRetriever | None = None

        # Store materials config for lazy initialization.
        self._materials_config = materials_config
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
            console_logger.info("Preloading materials retriever and CLIP model...")
            start_time = time.time()
            self._get_retriever()
            load_time = time.time() - start_time
            console_logger.info(
                f"Materials retriever preloaded in {load_time:.2f}s "
                "(includes CLIP model)"
            )

        # Setup routes.
        self.add_url_rule("/health", "health", self._health_endpoint, methods=["GET"])
        self.add_url_rule(
            "/shutdown", "shutdown", self._shutdown_endpoint, methods=["POST"]
        )
        self.add_url_rule(
            "/retrieve_materials",
            "retrieve_materials",
            self._retrieve_materials_endpoint,
            methods=["POST"],
        )

    def _get_retriever(self) -> MaterialsRetriever:
        """Get or create materials retriever (singleton per server process)."""
        if self._retriever is None:
            config = self._materials_config

            # Convert DictConfig to MaterialsConfig if needed.
            if config is not None and not isinstance(config, MaterialsConfig):
                config = MaterialsConfig.from_config(config)

            # Fall back to defaults if no config provided.
            if config is None:
                project_root = Path(__file__).parent.parent.parent.parent
                data_path = project_root / "data" / "materials"
                embeddings_path = data_path / "embeddings"

                config = MaterialsConfig(
                    data_path=data_path,
                    embeddings_path=embeddings_path,
                    use_top_k=5,
                )

            self._retriever = MaterialsRetriever(
                config=config, clip_device=self._clip_device
            )

            # Initialize retriever (loads data).
            if not self._retriever.initialize():
                console_logger.error("Failed to initialize materials retriever")
                raise RuntimeError("Materials retriever initialization failed")

        return self._retriever

    def start_processing(self) -> None:
        """Start background processing thread."""
        if self._processing_active:
            console_logger.warning("Processing thread already active")
            return

        console_logger.info("Starting materials retrieval processing thread")
        self._processing_active = True
        self._processing_thread = Thread(target=self._process_queue, daemon=False)
        self._processing_thread.start()

    def stop_processing(self) -> None:
        """Stop background processing thread gracefully."""
        if not self._processing_active:
            return

        console_logger.info("Stopping materials retrieval processing thread")
        self._processing_active = False

        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5)
            if self._processing_thread.is_alive():
                console_logger.warning("Processing thread did not stop gracefully")

        # Clean up GPU memory by clearing CLIP model.
        if self._retriever is not None:
            console_logger.info("Cleaning up GPU memory (CLIP model)...")
            reset_clip_model_cache()
            self._retriever = None

    def _process_queue(self) -> None:
        """Process retrieval requests using round-robin scheduling."""
        try:
            console_logger.info("Materials retrieval processing started")

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
            console_logger.info("Materials retrieval processing thread stopped")

    def _process_round_robin_request(self, queued_request: QueuedRequest) -> None:
        """Process a single retrieval request."""
        try:
            self._current_processing = (
                f"{queued_request.client_id}[{queued_request.request_index}]: "
                f"{queued_request.request.material_description}"
            )

            console_logger.info(
                f"Processing request from {queued_request.client_id}: "
                f"{queued_request.request.material_description}"
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
        self, request: MaterialsRetrievalServerRequest
    ) -> MaterialsRetrievalServerResponse:
        """Perform materials retrieval with CLIP.

        Args:
            request: Retrieval request with description.

        Returns:
            Response with matching materials and their texture paths.

        Raises:
            ValueError: If retrieval fails or no candidates found.
        """
        retriever = self._get_retriever()

        # Retrieve candidates sorted by CLIP score.
        candidates = retriever.retrieve(
            description=request.material_description,
            top_k=request.num_candidates,
        )

        if not candidates:
            raise ValueError(
                f"No candidates found for '{request.material_description}'"
            )

        # Copy material files to client-specified output directory.
        if not request.output_dir:
            raise ValueError("output_dir must be specified in request")
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[MaterialRetrievalResult] = []
        for candidate in candidates:
            try:
                # Find texture files in source material folder.
                source_material_dir = retriever.config.data_path / candidate.material_id
                if not source_material_dir.exists():
                    console_logger.warning(
                        f"Material directory not found: {source_material_dir}"
                    )
                    continue

                # Find PBR textures using Material class.
                source_material = Material.from_path(source_material_dir)
                color_texture = source_material.get_texture("Color")
                normal_texture = source_material.get_texture("NormalGL")
                roughness_texture = source_material.get_texture("Roughness")

                if not all([color_texture, normal_texture, roughness_texture]):
                    console_logger.warning(
                        f"Missing PBR textures for {candidate.material_id}"
                    )
                    continue

                # Copy textures to output directory.
                material_output_dir = output_dir / candidate.material_id
                material_output_dir.mkdir(parents=True, exist_ok=True)

                color_dst = material_output_dir / color_texture.name
                normal_dst = material_output_dir / normal_texture.name
                roughness_dst = material_output_dir / roughness_texture.name

                shutil.copy2(color_texture, color_dst)
                shutil.copy2(normal_texture, normal_dst)
                shutil.copy2(roughness_texture, roughness_dst)

                console_logger.debug(
                    f"Copied material {candidate.material_id} to {material_output_dir}"
                )

                result = MaterialRetrievalResult(
                    material_path=str(material_output_dir),
                    material_id=candidate.material_id,
                    similarity_score=float(candidate.clip_score),
                    category=candidate.category,
                    color_texture=str(color_dst),
                    normal_texture=str(normal_dst),
                    roughness_texture=str(roughness_dst),
                )
                results.append(result)

            except Exception as e:
                console_logger.warning(
                    f"Failed to copy material {candidate.material_id}: {e}"
                )
                continue

        if not results:
            raise ValueError(
                f"Failed to copy materials for any candidates matching "
                f"'{request.material_description}'"
            )

        console_logger.info(
            f"Returning {len(results)} candidates for "
            f"'{request.material_description}'"
        )

        return MaterialsRetrievalServerResponse(
            results=results, query_description=request.material_description
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
            self._shutdown_callback, "MaterialsRetrievalShutdown"
        ):
            console_logger.warning("Shutdown callback not registered.")
            return (
                flask.jsonify({"status": "error", "message": "shutdown unavailable"}),
                503,
            )

        return flask.jsonify({"status": "shutting down"}), 200

    def _retrieve_materials_endpoint(self) -> flask.Response:
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
            required_fields = ["material_description", "output_dir"]
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
                MaterialsRetrievalServerRequest(**req_data) for req_data in data
            ]

            # Use scene_id for fair scheduling if provided, otherwise generate UUID.
            first_scene_id = batch_requests[0].scene_id if batch_requests else None
            batch_id = first_scene_id if first_scene_id else str(uuid.uuid4())

            client_result_queue: Queue = Queue()
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
