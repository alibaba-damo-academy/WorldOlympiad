from __future__ import annotations

# Configure CUDA environment BEFORE any CUDA-dependent imports.
# This is required for nvdiffrast JIT compilation used by SAM 3D Objects.
# Must be called before importing torch to ensure environment is set up.
from scenecode.agent_utils.geometry_generation_server.cuda_env_setup import (
    ensure_cuda_env,
)

ensure_cuda_env()

# Now safe to import CUDA-dependent code.
import gc
import logging
import threading
import time

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from PIL import Image
from scipy import ndimage

from scenecode.agent_utils.mesh_utils import load_mesh_as_trimesh

console_logger = logging.getLogger(__name__)


class SAM3DPipelineManager:
    """Singleton manager for SAM3D generation pipelines to avoid repeated
    initialization.
    """

    _instance: SAM3DPipelineManager | None = None
    _sam3_model: Any | None = None
    _sam3d_pipeline: Any | None = None
    _current_config: dict | None = None
    _initialization_lock = threading.Lock()

    def __new__(cls) -> SAM3DPipelineManager:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_pipelines(
        cls, sam3_checkpoint: Path, sam3d_checkpoint: Path
    ) -> tuple[Any, Any]:
        """Get or create SAM3D pipeline instances.

        Args:
            sam3_checkpoint: Path to SAM3 model checkpoint.
            sam3d_checkpoint: Path to SAM 3D Objects model checkpoint.

        Returns:
            Tuple of (sam3_model, sam3d_pipeline).
        """
        config = {
            "sam3_checkpoint": str(sam3_checkpoint),
            "sam3d_checkpoint": str(sam3d_checkpoint),
        }

        with cls._initialization_lock:
            if cls._sam3_model is None or cls._current_config != config:
                console_logger.info("Initializing SAM3D pipelines...")
                cls._initialize_pipelines(
                    sam3_checkpoint=sam3_checkpoint,
                    sam3d_checkpoint=sam3d_checkpoint,
                )
                cls._current_config = config

        return cls._sam3_model, cls._sam3d_pipeline

    @classmethod
    def _initialize_pipelines(
        cls, sam3_checkpoint: Path, sam3d_checkpoint: Path
    ) -> None:
        """Initialize all SAM3D pipelines.

        Args:
            sam3_checkpoint: Path to SAM3 model checkpoint.
            sam3d_checkpoint: Path to SAM 3D Objects model checkpoint.
        """
        start_time = time.time()
        console_logger.debug("Starting _initialize_pipelines...")
        console_logger.debug(f"PID={__import__('os').getpid()}")

        try:
            import os
            import sys

            console_logger.debug("Setting LIDRA_SKIP_INIT=1")
            # Skip sam3d_objects initialization (init module is not present).
            os.environ["LIDRA_SKIP_INIT"] = "1"

            # Add SAM3 to path.
            sam3_path = Path(__file__).parent.parent.parent.parent / "external" / "SAM3"
            if str(sam3_path) not in sys.path:
                sys.path.insert(0, str(sam3_path))
            console_logger.debug(f"Added SAM3 path: {sam3_path}")

            # Add SAM 3D Objects to path.
            sam3d_path = (
                Path(__file__).parent.parent.parent.parent
                / "external"
                / "sam-3d-objects"
            )
            if str(sam3d_path) not in sys.path:
                sys.path.insert(0, str(sam3d_path))
            console_logger.debug(f"Added SAM3D path: {sam3d_path}")

            console_logger.debug("Importing hydra.utils.instantiate...")
            from hydra.utils import instantiate

            console_logger.debug("Importing omegaconf.OmegaConf...")
            from omegaconf import OmegaConf

            console_logger.debug("Importing sam3.model.sam3_image_processor...")
            from sam3.model.sam3_image_processor import Sam3Processor

            console_logger.debug("Importing sam3.model_builder...")
            from sam3.model_builder import build_sam3_image_model

            console_logger.debug("All SAM3 imports completed successfully")
        except ImportError as e:
            console_logger.error(f"Import failed: {e}")
            raise ImportError(
                "SAM3D is not installed. Please run scripts/install_sam3d.sh"
            ) from e

        # Monkey-patch render_utils to use gsplat backend instead of inria.
        # The SAM3D code defaults to "inria" backend which requires
        # diff_gaussian_rasterization (not available via pip). We have gsplat
        # installed, so we patch render_multiview to use it instead.
        console_logger.debug("Patching gsplat backend...")
        cls._patch_gsplat_backend()
        console_logger.debug("gsplat backend patched")

        # Clear existing pipelines first.
        if cls._sam3_model is not None:
            console_logger.debug("Cleaning up existing pipelines...")
            cls._cleanup_existing_pipelines()

        # Initialize SAM3 image model.
        console_logger.debug(f"Loading SAM3 image model from {sam3_checkpoint}")
        load_start = time.time()
        sam3_model = build_sam3_image_model(checkpoint_path=str(sam3_checkpoint))
        console_logger.debug(f"SAM3 model built in {time.time() - load_start:.2f}s")
        cls._sam3_model = Sam3Processor(sam3_model)
        console_logger.debug("Sam3Processor created")

        # Initialize SAM 3D Objects pipeline from Hydra config.
        console_logger.debug(f"Loading SAM 3D Objects from {sam3d_checkpoint}")

        # Load the pipeline config.
        checkpoint_dir = Path(sam3d_checkpoint).parent
        console_logger.debug(f"Loading config from {sam3d_checkpoint}")
        config = OmegaConf.load(sam3d_checkpoint)
        console_logger.debug("Config loaded")

        # Disable model compilation to avoid warmup bug (missing run_layout_model
        # method). This follows the official demo.py pattern.
        config.compile_model = False

        # Use nvdiffrast rendering engine for full quality (matches demo_text_to_3d.py).
        # Pre-initializing nvdiffrast before Warp (via _pre_init_nvdiffrast) ensures
        # CUDA context compatibility between the two libraries.
        config.rendering_engine = "nvdiffrast"

        # Update paths to be absolute from checkpoint directory.
        for key in config.keys():
            if key.endswith(("_path", "_ckpt_path", "_config_path")):
                if config[key] and not Path(config[key]).is_absolute():
                    config[key] = str(checkpoint_dir / config[key])
        console_logger.debug("Config paths updated")

        # Use Hydra to instantiate the entire pipeline with all nested configs.
        # This properly handles depth_model, preprocessors, and other nested components.
        console_logger.debug("Instantiating SAM3D pipeline (this may take a while)...")
        instantiate_start = time.time()
        cls._sam3d_pipeline = instantiate(config)
        console_logger.debug(
            f"SAM3D pipeline instantiated in {time.time() - instantiate_start:.2f}s"
        )

        console_logger.info(
            f"SAM3D pipelines initialized successfully in "
            f"{time.time() - start_time:.2f} seconds"
        )

    @classmethod
    def _cleanup_existing_pipelines(cls) -> None:
        """Clean up existing pipeline instances."""
        cls._sam3_model = None
        cls._sam3d_pipeline = None

        # Force garbage collection and clear CUDA cache.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def _pre_init_nvdiffrast(cls) -> None:
        """Pre-initialize nvdiffrast CUDA context before Warp.

        This MUST be called before any code that imports Warp (e.g., gsplat
        rendering). Warp's CUDA mempool initialization conflicts with nvdiffrast
        if nvdiffrast is initialized after Warp. Pre-initializing nvdiffrast
        ensures its CUDA context is created first, allowing both libraries to
        coexist.

        This follows the initialization pattern from the working demo_text_to_3d.py
        which implicitly initializes nvdiffrast before Warp through import order.
        """
        try:
            import utils3d.torch

            # Create a dummy RastContext to trigger nvdiffrast CUDA initialization.
            # This initializes nvdiffrast's CUDA backend before Warp gets a chance
            # to set up its mempool.
            console_logger.info("Pre-initializing nvdiffrast CUDA context...")
            dummy_ctx = utils3d.torch.RastContext(backend="cuda")
            del dummy_ctx
            console_logger.info("nvdiffrast CUDA context initialized successfully")

        except ImportError:
            console_logger.warning(
                "Could not pre-initialize nvdiffrast - utils3d not available"
            )
        except Exception as e:
            console_logger.warning(f"nvdiffrast pre-initialization failed: {e}")

    @classmethod
    def _patch_gsplat_backend(cls) -> None:
        """Monkey-patch SAM3D render_utils to use gsplat backend.

        The SAM3D code defaults to "inria" backend for Gaussian rendering, which
        requires the diff_gaussian_rasterization package (not available on PyPI,
        must be built from source). We have gsplat installed instead, so we patch
        render_multiview to pass backend="gsplat" to render_frames.

        This patch only affects render_multiview which is called during texture
        baking. It wraps the original function to inject the backend option.
        """
        try:
            from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

            # Check if already patched.
            if hasattr(render_utils, "_original_render_multiview"):
                console_logger.debug("render_utils already patched for gsplat backend")
                return

            # Store original function.
            original_render_multiview = render_utils.render_multiview

            def patched_render_multiview(sample, resolution=512, nviews=30):
                """Patched render_multiview that uses gsplat backend."""
                r = 2
                fov = 40
                cams = [
                    render_utils.sphere_hammersley_sequence(i, nviews)
                    for i in range(nviews)
                ]
                yaws = [cam[0] for cam in cams]
                pitchs = [cam[1] for cam in cams]
                extrinsics, intrinsics = (
                    render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
                        yaws, pitchs, r, fov
                    )
                )
                # Use gsplat backend instead of default inria.
                res = render_utils.render_frames(
                    sample,
                    extrinsics,
                    intrinsics,
                    {
                        "resolution": resolution,
                        "bg_color": (0, 0, 0),
                        "backend": "gsplat",
                    },
                )
                return res["color"], extrinsics, intrinsics

            # Apply patch.
            render_utils._original_render_multiview = original_render_multiview
            render_utils.render_multiview = patched_render_multiview

            console_logger.info(
                "Patched render_utils.render_multiview to use gsplat backend"
            )

        except ImportError:
            console_logger.warning(
                "Could not patch render_utils - SAM3D modules not yet imported"
            )

    @classmethod
    def reset_pipelines(cls) -> None:
        """Clear all SAM3D pipelines and free GPU memory.

        Call this when you're done with asset generation to free up VRAM.
        The next call to get_pipelines() will reinitialize everything.
        """
        if cls._sam3_model is not None:
            console_logger.info("Clearing SAM3D pipelines and freeing GPU memory...")
            cls._cleanup_existing_pipelines()

            cls._sam3_model = None
            cls._sam3d_pipeline = None
            cls._current_config = None

            console_logger.info("SAM3D pipelines cleared successfully")

    @classmethod
    def are_pipelines_loaded(cls) -> bool:
        """Check if pipelines are currently loaded."""
        return cls._sam3_model is not None


def generate_mask(
    image: Image.Image,
    sam3_processor: Any,
    mode: Literal["foreground", "object_description"],
    object_description: str | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Generate segmentation mask using SAM3.

    Args:
        image: Input image (PIL Image).
        sam3_processor: SAM3 processor instance (Sam3Processor).
        mode: Segmentation mode.
            - "foreground": Detect background and invert.
            - "object_description": Use the object description that generated the
              image. This provides semantic context about what object to segment.
        object_description: Object description for segmentation (required if
            mode="object_description"). Uses the same description that was used
            to generate the image for better segmentation accuracy.
        threshold: Confidence threshold for mask generation.

    Returns:
        Binary mask as numpy array (H, W) with values in {0, 1}.
    """
    if mode == "object_description" and object_description is None:
        raise ValueError(
            "object_description is required when mode='object_description'"
        )

    def tensor_to_numpy(value: Any) -> Any:
        """Convert torch tensors to NumPy, normalizing unsupported dtypes."""
        if not torch.is_tensor(value):
            return value

        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.to(dtype=torch.float32)
        return value.numpy()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        # Set image in processor.
        inference_state = sam3_processor.set_image(image)

        if mode == "foreground":
            # For foreground segmentation, detect "background" and invert the mask.
            prompt = "background"
        else:
            # Use the object description that generated the image for better
            # segmentation. This provides semantic context about what to segment.
            prompt = object_description

        # Generate segmentation.
        # set_text_prompt returns updated inference_state with masks/scores.
        inference_state = sam3_processor.set_text_prompt(
            state=inference_state, prompt=prompt
        )

    # Extract results from inference_state.
    masks = inference_state["masks"]
    scores = inference_state["scores"]

    # Convert tensors to numpy (move from GPU to CPU if needed).
    scores = tensor_to_numpy(scores)
    masks = tensor_to_numpy(masks)

    # Check if any masks were detected.
    if len(scores) == 0:
        raise ValueError(
            f"No instances detected with prompt '{prompt}'. "
            "Try a different image or lower confidence threshold."
        )

    # Select mask with highest confidence.
    best_mask_idx = np.argmax(scores)
    mask = masks[best_mask_idx]

    # Squeeze any extra dimensions to ensure mask is 2D (H, W).
    while mask.ndim > 2:
        mask = np.squeeze(mask, axis=0)

    # Invert mask if in foreground mode (we detected background).
    if mode == "foreground":
        mask = ~mask

    # Convert to binary mask (H, W) with values in {0, 1}.
    if mask.dtype != np.uint8:
        mask = (mask > threshold).astype(np.uint8)

    # Remove edge-connected white regions (artifacts from background inversion).
    # Only apply in foreground mode - object_description mode may have objects at edges.
    if mode == "foreground":
        labeled, _ = ndimage.label(mask)

        # Find labels that touch any edge.
        edge_labels = set()
        edge_labels.update(labeled[0, :].flatten())  # top edge
        edge_labels.update(labeled[-1, :].flatten())  # bottom edge
        edge_labels.update(labeled[:, 0].flatten())  # left edge
        edge_labels.update(labeled[:, -1].flatten())  # right edge
        edge_labels.discard(0)  # 0 is background, not an artifact

        # Zero out edge-connected components.
        for label in edge_labels:
            mask[labeled == label] = 0

    return mask


def generate_3d_from_mask(
    image: Image.Image, mask: np.ndarray, sam3d_pipeline: Any, output_path: Path
) -> None:
    """Generate 3D mesh from image and mask using SAM 3D Objects.

    Args:
        image: Input image (PIL Image).
        mask: Binary segmentation mask (H, W) with values in {0, 1}.
        sam3d_pipeline: SAM 3D Objects pipeline instance.
        output_path: Path where the generated 3D mesh will be saved.
    """
    # Convert image to numpy array (H, W, 3).
    image_np = np.array(image)

    # Convert mask to uint8 with values 0-255.
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Merge mask into RGBA image (H, W, 4).
    # Mask becomes the alpha channel (0=transparent/background, 255=opaque/foreground).
    rgba_image = np.concatenate([image_np, mask_uint8[..., None]], axis=-1)

    # Generate 3D mesh using SAM 3D Objects pipeline.
    # Pass RGBA image with mask=None (mask is in alpha channel).
    output = sam3d_pipeline.run(
        rgba_image,
        mask=None,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=True,
        use_vertex_color=False,  # Use UV-mapped textures from texture baking
    )

    # Export GLB mesh.
    output["glb"].export(str(output_path))


def generate_with_sam3d(
    image_path: Path,
    output_path: Path,
    sam3_checkpoint: Path,
    sam3d_checkpoint: Path,
    mode: Literal["foreground", "object_description"] = "foreground",
    object_description: str | None = None,
    threshold: float = 0.5,
    debug_folder: Path | None = None,
    use_pipeline_caching: bool = True,
) -> None:
    """Generate 3D geometry from a 2D image using SAM3D pipeline.

    Args:
        image_path: Path to the input image file.
        output_path: Path where the generated 3D mesh will be saved.
        sam3_checkpoint: Path to SAM3 model checkpoint.
        sam3d_checkpoint: Path to SAM 3D Objects model checkpoint.
        mode: Segmentation mode.
            - "foreground": Automatic foreground detection (default).
            - "object_description": Use the object description that generated
              the image for semantic-aware segmentation.
        object_description: Object description used to generate the image
            (required if mode="object_description"). Provides semantic context
            for better segmentation accuracy.
        threshold: Confidence threshold for mask generation.
        debug_folder: Path to folder where debug images will be saved. If None, no
            debug images will be saved. Must exist.
        use_pipeline_caching: Whether to cache pipelines for faster subsequent
            generations.

    Raises:
        RuntimeError: If CUDA is not available (required for SAM3D).
    """
    # Verify CUDA is available before attempting SAM3D generation.
    from .cuda_env_setup import ensure_cuda_env

    if not ensure_cuda_env():
        raise RuntimeError(
            "SAM3D requires CUDA 12.x for nvdiffrast JIT compilation. "
            "Please install CUDA toolkit or set CUDA_HOME environment variable."
        )

    start_time = time.time()

    # Get pipelines from manager.
    sam3_processor, sam3d_pipeline = SAM3DPipelineManager.get_pipelines(
        sam3_checkpoint=sam3_checkpoint, sam3d_checkpoint=sam3d_checkpoint
    )

    # Load image.
    image = Image.open(image_path).convert("RGB")

    # Generate mask.
    mask = generate_mask(
        image=image,
        sam3_processor=sam3_processor,
        mode=mode,
        object_description=object_description,
        threshold=threshold,
    )

    if debug_folder:
        # Save mask visualization.
        mask_vis = Image.fromarray((mask * 255).astype(np.uint8))
        mask_vis.save(debug_folder / f"{image_path.stem}_mask.png")

        # Save masked image.
        image_np = np.array(image)
        masked_image_np = image_np.copy()
        masked_image_np[mask == 0] = 0
        masked_image = Image.fromarray(masked_image_np)
        masked_image.save(debug_folder / f"{image_path.stem}_masked.png")

    # Generate 3D mesh.
    generate_3d_from_mask(
        image=image, mask=mask, sam3d_pipeline=sam3d_pipeline, output_path=output_path
    )

    # Validate output mesh file.
    if not output_path.exists():
        raise RuntimeError(f"SAM3D mesh export failed: {output_path} does not exist")

    if output_path.stat().st_size == 0:
        raise RuntimeError(f"SAM3D mesh export failed: {output_path} is empty")

    # Verify the file is a valid GLB by attempting to load it.
    # Use load_mesh_as_trimesh which properly handles Scene objects (GLB files
    # with multiple meshes) by extracting and merging all mesh components.
    try:
        mesh = load_mesh_as_trimesh(output_path)
        console_logger.debug(
            f"Validated SAM3D output: {len(mesh.vertices)} vertices, "
            f"{len(mesh.faces)} faces"
        )
    except ValueError as e:
        raise RuntimeError(
            f"SAM3D mesh export failed: {output_path} is not a valid GLB file: {e}"
        ) from e

    # Clean up pipelines if caching is disabled.
    if not use_pipeline_caching:
        SAM3DPipelineManager.reset_pipelines()

    end_time = time.time()
    console_logger.info(
        f"Generated geometry from {image_path} using SAM3D "
        f"in {end_time - start_time:.2f} seconds."
    )
