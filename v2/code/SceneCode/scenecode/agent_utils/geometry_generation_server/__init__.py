"""Geometry generation server components.

This module contains the complete geometry generation server implementation,
including server infrastructure and both Hunyuan3D and SAM3D backends.

For multi-GPU support, the server automatically detects available GPUs and
spawns one worker process per GPU. Use CUDA_VISIBLE_DEVICES to control which
GPUs are used.

IMPORTANT: The server-related imports (GeometryGenerationServer, GeometryGenerationClient)
do NOT initialize CUDA. CUDA-dependent imports (generate_geometry_from_image, pipeline
managers) are accessed via lazy loading to avoid accidental CUDA initialization in the
parent process when using multi-GPU mode.
"""

# Safe imports that don't initialize CUDA.
from .client import GeometryGenerationClient
from .dataclasses import (
    GeometryGenerationServerRequest,
    GeometryGenerationServerResponse,
)
from .server_manager import GeometryGenerationServer

# Lazy imports for CUDA-dependent modules.
# These should only be imported in GPU worker processes or when explicitly needed.


def __getattr__(name: str):
    """Lazy loading for CUDA-dependent modules.

    This prevents CUDA initialization in the parent process when using multi-GPU mode.
    """
    if name == "generate_geometry_from_image":
        from .geometry_generation import generate_geometry_from_image

        return generate_geometry_from_image

    if name == "Hunyuan3DPipelineManager":
        from .hunyuan3d_pipeline_manager import Hunyuan3DPipelineManager

        return Hunyuan3DPipelineManager

    if name == "SAM3DPipelineManager":
        from .sam3d_pipeline_manager import SAM3DPipelineManager

        return SAM3DPipelineManager

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Safe imports (no CUDA initialization).
    "GeometryGenerationClient",
    "GeometryGenerationServer",
    "GeometryGenerationServerRequest",
    "GeometryGenerationServerResponse",
    # Lazy imports (CUDA initialization on access).
    "generate_geometry_from_image",
    "Hunyuan3DPipelineManager",
    "SAM3DPipelineManager",
]
