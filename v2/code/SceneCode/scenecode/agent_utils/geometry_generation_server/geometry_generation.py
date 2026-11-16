"""Geometry generation module supporting multiple backends."""

# isort: off
# Configure CUDA environment BEFORE any CUDA-dependent imports.
# This is critical for nvdiffrast JIT compilation used by SAM3D.
# Must be the first import that could trigger CUDA setup.
from scenecode.agent_utils.geometry_generation_server.cuda_env_setup import (
    ensure_cuda_env,
)

ensure_cuda_env()

# Now safe to import standard library and other modules.
# isort: on
import logging
import time

from pathlib import Path
from typing import Literal

from PIL import Image

from scenecode.agent_utils.geometry_generation_server.hunyuan3d_pipeline_manager import (
    Hunyuan3DPipelineManager,
)
from scenecode.agent_utils.geometry_generation_server.sam3d_pipeline_manager import (
    generate_with_sam3d,
)

console_logger = logging.getLogger(__name__)


def generate_geometry_from_image(
    image_path: Path,
    output_path: Path,
    debug_folder: Path | None = None,
    use_mini: bool = False,
    use_pipeline_caching: bool = True,
    backend: Literal["hunyuan3d", "sam3d"] = "hunyuan3d",
    sam3d_config: dict | None = None,
) -> None:
    """Generate 3D geometry with texture from a 2D image.

    Args:
        image_path (Path): Path to the input image file. The image will be converted to
            RGBA format internally for automatic background removal (Hunyuan3D) or RGB
            format for segmentation (SAM3D).
        output_path (Path): Path where the generated 3D mesh will be saved. The output
            format depends on the file extension (e.g., .glb, .obj). GLB is recommended.
        debug_folder (Path | None): Path to the folder where the debug images will be
            saved. If None, no debug images will be saved. Must exist.
        use_mini (bool): Whether to use the mini model variant (0.6B parameters) for
            faster inference with reduced memory footprint. Only applies to Hunyuan3D
            backend. Defaults to False.
        use_pipeline_caching (bool): Whether to cache pipelines for faster subsequent
            generations.
        backend (Literal["hunyuan3d", "sam3d"]): Which 3D generation backend to use.
            Defaults to "hunyuan3d".
        sam3d_config (dict | None): Configuration for SAM3D backend. Required if
            backend="sam3d". Should contain:
            - sam3_checkpoint (Path): Path to SAM3 checkpoint
            - sam3d_checkpoint (Path): Path to SAM 3D Objects checkpoint
            - mode (Literal["foreground", "object_description"]): Segmentation mode
            - object_description (str | None): Object description (required if
              mode="object_description")
            - threshold (float): Confidence threshold for mask generation
    """
    if backend == "hunyuan3d":
        _generate_with_hunyuan3d(
            image_path=image_path,
            output_path=output_path,
            debug_folder=debug_folder,
            use_mini=use_mini,
            use_pipeline_caching=use_pipeline_caching,
        )
    elif backend == "sam3d":
        if sam3d_config is None:
            raise ValueError("sam3d_config is required when backend='sam3d'")
        generate_with_sam3d(
            image_path=image_path,
            output_path=output_path,
            sam3_checkpoint=sam3d_config["sam3_checkpoint"],
            sam3d_checkpoint=sam3d_config["sam3d_checkpoint"],
            mode=sam3d_config.get("mode", "foreground"),
            object_description=sam3d_config.get("object_description"),
            threshold=sam3d_config.get("threshold", 0.5),
            debug_folder=debug_folder,
            use_pipeline_caching=use_pipeline_caching,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _generate_with_hunyuan3d(
    image_path: Path,
    output_path: Path,
    debug_folder: Path | None = None,
    use_mini: bool = False,
    use_pipeline_caching: bool = True,
) -> None:
    """Generate 3D geometry using Hunyuan3D backend.

    Args:
        image_path: Path to the input image file.
        output_path: Path where the generated 3D mesh will be saved.
        debug_folder: Path to the folder where debug images will be saved.
        use_mini: Whether to use the mini model variant.
        use_pipeline_caching: Whether to cache pipelines for faster subsequent
            generations.
    """
    try:
        from hy3dgen.shapegen.pipelines import export_to_trimesh
    except ImportError as e:
        raise ImportError(
            "Hunyuan3D-2 is not installed. Please run scripts/install_hunyuan3d.sh"
        ) from e

    start_time = time.time()

    # Always get pipelines from manager for consistency.
    pipeline_shapegen, pipeline_texgen, face_reducer, background_remover = (
        Hunyuan3DPipelineManager.get_pipelines(use_mini=use_mini)
    )

    # Load and process image.
    image = Image.open(image_path).convert("RGBA")
    image = background_remover(image)

    if debug_folder:
        image.save(debug_folder / f"{image_path.stem}_without_background.png")

    # Generate shape with turbo parameters.
    outputs = pipeline_shapegen(
        image=image,
        num_inference_steps=5,
        guidance_scale=5.0,
        octree_resolution=256,
        num_chunks=8000,
        output_type="mesh",
    )

    # Convert to trimesh and apply face reduction.
    mesh = export_to_trimesh(outputs)[0]
    mesh = face_reducer(mesh)

    # Apply texture and export.
    mesh = pipeline_texgen(mesh, image=image)
    mesh.export(output_path)

    # Clean up pipelines if caching is disabled.
    if not use_pipeline_caching:
        Hunyuan3DPipelineManager.reset_pipelines()

    end_time = time.time()
    console_logger.info(
        f"Generated geometry from {image_path} in {end_time - start_time:.2f} seconds."
    )
