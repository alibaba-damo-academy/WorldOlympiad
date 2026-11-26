from __future__ import annotations

import gc
import logging
import threading
import time

from typing import Any, Literal, Tuple

import torch

console_logger = logging.getLogger(__name__)


class Hunyuan3DPipelineManager:
    """Singleton manager for Hunyuan3D generation pipelines to avoid repeated
    initialization.
    """

    _instance: Hunyuan3DPipelineManager | None = None
    _shape_pipeline: Any | None = None
    _texture_pipeline: Any | None = None
    _face_reducer: Any | None = None
    _background_remover: Any | None = None
    _current_model_variant: Literal["mini", "full"] | None = None
    _initialization_lock = threading.Lock()

    def __new__(cls) -> Hunyuan3DPipelineManager:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_pipelines(cls, use_mini: bool = False) -> Tuple[Any, Any, Any, Any]:
        """Get or create Hunyuan3D pipeline instances.

        Args:
            use_mini: Whether to use mini model variant (0.6B parameters).

        Returns:
            Tuple of (shape_pipeline, texture_pipeline, face_reducer,
            background_remover).
        """
        # Initialize or reinitialize if model configuration changed.
        current_variant = "mini" if use_mini else "full"
        with cls._initialization_lock:
            if (
                cls._shape_pipeline is None
                or cls._current_model_variant != current_variant
            ):
                console_logger.info(
                    "Initializing Hunyuan3D pipelines "
                    f"({'mini' if use_mini else 'full'} model)..."
                )
                cls._initialize_pipelines(use_mini)
                cls._current_model_variant = current_variant

        return (
            cls._shape_pipeline,
            cls._texture_pipeline,
            cls._face_reducer,
            cls._background_remover,
        )

    @classmethod
    def _initialize_pipelines(cls, use_mini: bool) -> None:
        """Initialize all Hunyuan3D pipelines.

        Args:
            use_mini: Whether to use mini model variant.
        """
        start_time = time.time()
        try:
            from hy3dgen.rembg import BackgroundRemover
            from hy3dgen.shapegen import FaceReducer, Hunyuan3DDiTFlowMatchingPipeline
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
        except ImportError as e:
            raise ImportError(
                "Hunyuan3D-2 is not installed. Please run scripts/install_hunyuan3d.sh"
            ) from e

        # Clear existing pipelines first.
        if cls._shape_pipeline is not None:
            cls._cleanup_existing_pipelines()

        # Configure model paths based on use_mini flag.
        if use_mini:
            model_path = "tencent/Hunyuan3D-2mini"
            subfolder = "hunyuan3d-dit-v2-mini-turbo"
        else:
            model_path = "tencent/Hunyuan3D-2"
            subfolder = "hunyuan3d-dit-v2-0-turbo"

        # Initialize shape generation pipeline.
        cls._shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path, subfolder=subfolder
        )
        cls._shape_pipeline.enable_flashvdm()

        # Initialize texture generation pipeline (always uses full model).
        cls._texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            "tencent/Hunyuan3D-2"
        )

        # Initialize post-processing tools.
        cls._face_reducer = FaceReducer()
        cls._background_remover = BackgroundRemover()

        console_logger.info(
            "Hunyuan3D pipelines initialized successfully in "
            f"{time.time() - start_time:.2f} seconds"
        )

    @classmethod
    def _cleanup_existing_pipelines(cls) -> None:
        """Clean up existing pipeline instances."""
        del cls._shape_pipeline
        del cls._texture_pipeline
        del cls._face_reducer
        del cls._background_remover

        # Force garbage collection and clear CUDA cache.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def reset_pipelines(cls) -> None:
        """Clear all Hunyuan3D pipelines and free GPU memory.

        Call this when you're done with asset generation to free up VRAM.
        The next call to get_pipelines() will reinitialize everything.
        """
        if cls._shape_pipeline is not None:
            console_logger.info(
                "Clearing Hunyuan3D pipelines and freeing GPU memory..."
            )
            cls._cleanup_existing_pipelines()

            cls._shape_pipeline = None
            cls._texture_pipeline = None
            cls._face_reducer = None
            cls._background_remover = None
            cls._current_model_variant = None

            console_logger.info("Hunyuan3D pipelines cleared successfully")

    @classmethod
    def are_pipelines_loaded(cls) -> bool:
        """Check if pipelines are currently loaded."""
        return cls._shape_pipeline is not None
