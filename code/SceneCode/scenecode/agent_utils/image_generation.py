import base64
import logging
import os
import subprocess
import time

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

from google import genai
from google.genai import types
from omegaconf import DictConfig
from openai import OpenAI
from PIL import Image

from scenecode.agent_utils.gpu_diagnostics import describe_gpu_snapshot
from scenecode.prompts import PROMPTS_DATA_DIR
from scenecode.prompts.manager import PromptManager
from scenecode.prompts.registry import ImageGenerationPrompts
from scenecode.utils.openai import create_openai_client

console_logger = logging.getLogger(__name__)


class AssetOperationType(Enum):
    """Type of asset operation for request categorization."""

    INITIAL = "initial"  # Initial scene population
    ADDITION = "addition"  # Adding new objects to existing scene
    REPLACEMENT = "replacement"  # Replacing specific objects


class BaseImageGenerator(ABC):
    """Abstract base class for image generation backends."""

    @abstractmethod
    def generate_images(
        self,
        style_prompt: str,
        object_descriptions: list[str],
        output_paths: list[Path],
        size: str | None = None,
        labels: list[str] | None = None,
    ) -> None:
        """Generate multiple images in parallel.

        Args:
            style_prompt: The style context for the images.
            object_descriptions: List of object descriptions to generate.
            output_paths: Paths where images will be saved.
            size: Optional image size/aspect ratio override.
                - OpenAI: "1024x1024", "1792x1024", "1024x1792"
                - Gemini: "1:1", "16:9", "9:16", "4:3", "3:4"
                If None, uses backend default (1024x1024 or instance aspect_ratio).
            labels: Optional labels for log messages. If not provided,
                object_descriptions are used for logging.
        """
        ...

    @abstractmethod
    def generate_furniture_context_image(
        self,
        reference_image_path: Path,
        scene_description: str,
        width_m: float,
        length_m: float,
        output_path: Path,
    ) -> Path:
        """Generate top-down room visualization for furniture placement.

        Uses a Blender render of the empty room as reference, then edits it
        to show suggested furniture placement. The reference image shows
        doors/windows that should not be blocked.

        Args:
            reference_image_path: Blender render of empty room showing openings.
            scene_description: Text description of the scene.
            width_m: Floor plan width in meters.
            length_m: Floor plan length in meters.
            output_path: Where to save the generated image.

        Returns:
            Path to the saved image.
        """
        ...

    @abstractmethod
    def generate_manipuland_context_image(
        self,
        reference_image_path: Path,
        furniture_description: str,
        furniture_dimensions: str,
        suggested_items: str,
        prompt_constraints: str,
        style_notes: str,
        output_path: Path,
    ) -> Path:
        """Generate context image showing objects placed on furniture.

        Uses a Blender render of empty furniture as reference, then edits it
        to show suggested manipuland placement. Provides visual guidance for
        the manipuland designer agent.

        Args:
            reference_image_path: Blender render of furniture (may include
                context furniture like chairs around a table).
            furniture_description: Text description of the furniture.
            furniture_dimensions: Human-readable dimensions (e.g., "1.2m wide").
            suggested_items: Items to place on the furniture.
            prompt_constraints: Placement constraints from VLM analysis.
            style_notes: Style guidance for the scene.
            output_path: Where to save the generated image.

        Returns:
            Path to the saved image.
        """
        ...


class OpenAIImageGenerator(BaseImageGenerator):
    """Image generation using OpenAI gpt-image-1.5 via the Images API."""

    def __init__(
        self,
        client: OpenAI | None = None,
        quality: str = "low",
        api_base: str | None = None,
    ):
        """Initialize the generator.

        Args:
            client: Optional OpenAI client to reuse. If None, creates a new one.
            quality: Image quality. Options: "auto", "low", "medium", "high".
            api_base: Optional OpenAI-compatible API base URL.

        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set.
        """
        if client is None and not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAI image "
                "generation. Set it with: export OPENAI_API_KEY='your-key'"
            )
        self.client = client or create_openai_client(api_base=api_base)
        self.prompt_manager = PromptManager(prompts_dir=PROMPTS_DATA_DIR)
        self.image_quality = quality
        # self.model = "gpt-image-1.5"
        self.model = "gpt-image-2"

    def generate_images(
        self,
        style_prompt: str,
        object_descriptions: list[str],
        output_paths: list[Path],
        size: str | None = None,
        labels: list[str] | None = None,
    ) -> None:
        """Generate multiple images in parallel using gpt-image-1.5.

        Args:
            style_prompt: The style context for the images.
            object_descriptions: List of object descriptions to generate.
            output_paths: Paths where images will be saved.
            size: Optional size override ("1024x1024", "1792x1024", "1024x1792").
            labels: Optional labels for log messages.
        """
        if len(object_descriptions) != len(output_paths):
            raise ValueError("Number of descriptions must match number of output paths")

        # Use provided size or default to square.
        image_size = size if size else "1024x1024"
        # Use labels for logging if provided, otherwise use descriptions.
        effective_labels = labels if labels else object_descriptions

        console_logger.info(
            f"Generating {len(object_descriptions)} images (OpenAI gpt-image-1.5)"
        )

        def generate_single_image(
            description: str, output_path: Path, label: str
        ) -> None:
            prompt = self.prompt_manager.get_prompt(
                ImageGenerationPrompts.ASSET_IMAGE_INITIAL,
                description=description,
                style_prompt=style_prompt,
            )
            start_time = time.time()
            response = self.client.images.generate(
                model=self.model,
                prompt=prompt,
                size=image_size,
                n=1,
                output_format="png",
                quality=self.image_quality,
                background="opaque",
                moderation="low",
            )
            end_time = time.time()
            console_logger.info(
                f"Generated image for {label} in {end_time - start_time:.2f} seconds."
            )
            _extract_and_save_openai_image(
                response=response, output_path=output_path, description=description
            )

        # Generate all images concurrently.
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(generate_single_image, desc, path, lbl)
                for desc, path, lbl in zip(
                    object_descriptions, output_paths, effective_labels
                )
            ]
            # Wait for all to complete and raise any exceptions.
            for future in as_completed(futures):
                future.result()

    def generate_furniture_context_image(
        self,
        reference_image_path: Path,
        scene_description: str,
        width_m: float,
        length_m: float,
        output_path: Path,
    ) -> Path:
        """Generate top-down room visualization for furniture placement.

        Edits a Blender render of the empty room to show suggested furniture
        placement. The reference image shows doors/windows that should not
        be blocked.

        Args:
            reference_image_path: Blender render of empty room showing openings.
            scene_description: Text description of the scene.
            width_m: Floor plan width in meters.
            length_m: Floor plan length in meters.
            output_path: Where to save the generated image.

        Returns:
            Path to the saved image.
        """
        prompt = self.prompt_manager.get_prompt(
            ImageGenerationPrompts.FURNITURE_CONTEXT_IMAGE,
            scene_description=scene_description,
            width_m=width_m,
            length_m=length_m,
        )

        console_logger.info("Generating furniture placement context image")

        result = self._edit_image(
            prompt=prompt,
            reference_image_path=reference_image_path,
            output_path=output_path,
        )

        console_logger.info(f"Saved context image to {output_path}")

        return result

    def _edit_image(
        self,
        prompt: str,
        reference_image_path: Path,
        output_path: Path,
        size: str = "1024x1024",
    ) -> Path:
        """Edit an existing image with the given prompt.

        Uses OpenAI images.edit() API to modify a reference image based on
        the prompt. This is the core editing logic used by context image
        generation methods.

        Args:
            prompt: The editing instruction for the image.
            reference_image_path: Path to the reference image to edit.
            output_path: Where to save the edited image.
            size: Image size for output. Defaults to "1024x1024".

        Returns:
            Path to the saved edited image.

        API Reference: https://platform.openai.com/docs/api-reference/images
        """
        console_logger.info(f"Editing image {reference_image_path} (OpenAI)")

        start_time = time.time()
        with open(reference_image_path, "rb") as image_file:
            response = self.client.images.edit(
                model=self.model, image=image_file, prompt=prompt, size=size
            )
        end_time = time.time()

        console_logger.info(f"Edited image in {end_time - start_time:.2f} seconds")

        _extract_and_save_openai_image(
            response=response, output_path=output_path, description="edited image"
        )

        return output_path

    def generate_manipuland_context_image(
        self,
        reference_image_path: Path,
        furniture_description: str,
        furniture_dimensions: str,
        suggested_items: str,
        prompt_constraints: str,
        style_notes: str,
        output_path: Path,
    ) -> Path:
        """Generate context image showing objects placed on furniture.

        Args:
            reference_image_path: Blender render of furniture.
            furniture_description: Text description of the furniture.
            furniture_dimensions: Human-readable dimensions.
            suggested_items: Items to place on the furniture.
            prompt_constraints: Placement constraints from VLM analysis.
            style_notes: Style guidance for the scene.
            output_path: Where to save the generated image.

        Returns:
            Path to the saved image.
        """
        prompt = self.prompt_manager.get_prompt(
            ImageGenerationPrompts.MANIPULAND_CONTEXT_IMAGE,
            furniture_description=furniture_description,
            furniture_dimensions=furniture_dimensions,
            suggested_items=suggested_items,
            prompt_constraints=prompt_constraints,
            style_notes=style_notes,
        )

        console_logger.info("Generating manipuland placement context image")

        result = self._edit_image(
            prompt=prompt,
            reference_image_path=reference_image_path,
            output_path=output_path,
        )

        console_logger.info(f"Saved manipuland context image to {output_path}")

        return result


class GeminiImageGenerator(BaseImageGenerator):
    """Image generation using Google Gemini (gemini-3-pro-image-preview)."""

    def __init__(
        self,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
    ):
        """Initialize the Gemini generator.

        Args:
            aspect_ratio: Aspect ratio for generated images (e.g., "1:1", "16:9").
            image_size: Output image size ("1K", "2K", "4K").

        Raises:
            ValueError: If GOOGLE_API_KEY environment variable is not set.
        """
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Gemini image "
                "generation. Set it with: export GOOGLE_API_KEY='your-key'"
            )

        self.client = genai.Client()
        self.aspect_ratio = aspect_ratio
        self.image_size = image_size
        self.model = "gemini-3-pro-image-preview"
        self.prompt_manager = PromptManager(prompts_dir=PROMPTS_DATA_DIR)

    def generate_images(
        self,
        style_prompt: str,
        object_descriptions: list[str],
        output_paths: list[Path],
        size: str | None = None,
        labels: list[str] | None = None,
    ) -> None:
        """Generate multiple images in parallel using Gemini.

        Args:
            style_prompt: The style context for the images.
            object_descriptions: List of object descriptions to generate.
            output_paths: Paths where images will be saved.
            size: Optional aspect ratio override ("1:1", "16:9", "9:16", "4:3", "3:4").
            labels: Optional labels for log messages.
        """
        if len(object_descriptions) != len(output_paths):
            raise ValueError("Number of descriptions must match number of output paths")

        # Use provided aspect ratio or instance default.
        aspect_ratio = size if size else self.aspect_ratio
        # Use labels for logging if provided, otherwise use descriptions.
        effective_labels = labels if labels else object_descriptions

        console_logger.info(f"Generating {len(object_descriptions)} images (Gemini)")

        def generate_single_image(
            description: str, output_path: Path, label: str
        ) -> None:
            prompt = self.prompt_manager.get_prompt(
                ImageGenerationPrompts.ASSET_IMAGE_INITIAL,
                description=description,
                style_prompt=style_prompt,
            )

            start_time = time.time()
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=self.image_size,
                    ),
                ),
            )
            end_time = time.time()

            console_logger.info(
                f"Generated image for {label} in {end_time - start_time:.2f} "
                "seconds (Gemini)."
            )

            _extract_and_save_gemini_image(
                response=response, output_path=output_path, description=description
            )

        # Generate all images concurrently.
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(generate_single_image, desc, path, lbl)
                for desc, path, lbl in zip(
                    object_descriptions, output_paths, effective_labels
                )
            ]
            # Wait for all to complete and raise any exceptions.
            for future in as_completed(futures):
                future.result()

    def generate_furniture_context_image(
        self,
        reference_image_path: Path,
        scene_description: str,
        width_m: float,
        length_m: float,
        output_path: Path,
    ) -> Path:
        """Generate top-down room visualization for furniture placement using Gemini.

        Edits a Blender render of the empty room to show suggested furniture
        placement. The reference image shows doors/windows that should not
        be blocked.

        Args:
            reference_image_path: Blender render of empty room showing openings.
            scene_description: Text description of the scene.
            width_m: Floor plan width in meters.
            length_m: Floor plan length in meters.
            output_path: Where to save the generated image.

        Returns:
            Path to the saved image.
        """
        prompt = self.prompt_manager.get_prompt(
            ImageGenerationPrompts.FURNITURE_CONTEXT_IMAGE,
            scene_description=scene_description,
            width_m=width_m,
            length_m=length_m,
        )

        console_logger.info("Generating furniture placement context image (Gemini)")

        result = self._edit_image(
            prompt=prompt,
            reference_image_path=reference_image_path,
            output_path=output_path,
        )

        console_logger.info(f"Saved context image to {output_path}")

        return result

    def _edit_image(
        self,
        prompt: str,
        reference_image_path: Path,
        output_path: Path,
    ) -> Path:
        """Edit an existing image with the given prompt.

        Uses Gemini generate_content() with image input for multimodal editing.
        This is the core editing logic used by context image generation methods.

        Args:
            prompt: The editing instruction for the image.
            reference_image_path: Path to the reference image to edit.
            output_path: Where to save the edited image.

        Returns:
            Path to the saved edited image.

        API Reference: https://ai.google.dev/gemini-api/docs/image-generation
        """
        console_logger.info(f"Editing image {reference_image_path} (Gemini)")

        image_input = Image.open(reference_image_path)

        start_time = time.time()
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, image_input],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=self.aspect_ratio,
                ),
            ),
        )
        end_time = time.time()

        console_logger.info(
            f"Edited image in {end_time - start_time:.2f} seconds (Gemini)"
        )

        _extract_and_save_gemini_image(
            response=response, output_path=output_path, description="edited image"
        )

        return output_path

    def generate_manipuland_context_image(
        self,
        reference_image_path: Path,
        furniture_description: str,
        furniture_dimensions: str,
        suggested_items: str,
        prompt_constraints: str,
        style_notes: str,
        output_path: Path,
    ) -> Path:
        """Generate context image showing objects placed on furniture.

        Args:
            reference_image_path: Blender render of furniture.
            furniture_description: Text description of the furniture.
            furniture_dimensions: Human-readable dimensions.
            suggested_items: Items to place on the furniture.
            prompt_constraints: Placement constraints from VLM analysis.
            style_notes: Style guidance for the scene.
            output_path: Where to save the generated image.

        Returns:
            Path to the saved image.
        """
        prompt = self.prompt_manager.get_prompt(
            ImageGenerationPrompts.MANIPULAND_CONTEXT_IMAGE,
            furniture_description=furniture_description,
            furniture_dimensions=furniture_dimensions,
            suggested_items=suggested_items,
            prompt_constraints=prompt_constraints,
            style_notes=style_notes,
        )

        console_logger.info("Generating manipuland placement context image (Gemini)")

        result = self._edit_image(
            prompt=prompt,
            reference_image_path=reference_image_path,
            output_path=output_path,
        )

        console_logger.info(f"Saved manipuland context image to {output_path}")

        return result


class FluxKleinImageGenerator(BaseImageGenerator):
    """Image generation using a local FLUX.2-klein checkpoint via subprocess."""

    def __init__(self, config: DictConfig):
        """Initialize the FLUX.2-klein generator."""
        self.prompt_manager = PromptManager(prompts_dir=PROMPTS_DATA_DIR)
        self.python_executable = Path(config.python_executable)
        self.model_path = Path(config.model_path)
        self.width = int(config.width)
        self.height = int(config.height)
        self.num_inference_steps = int(config.num_inference_steps)
        self.guidance_scale = float(config.guidance_scale)
        self.max_sequence_length = int(config.max_sequence_length)
        self.seed = int(config.seed)
        self.worker_script = Path(__file__).with_name("flux_klein_worker.py")

    def generate_images(
        self,
        style_prompt: str,
        object_descriptions: list[str],
        output_paths: list[Path],
        size: str | None = None,
        labels: list[str] | None = None,
    ) -> None:
        """Generate multiple images sequentially using FLUX.2-klein."""
        if len(object_descriptions) != len(output_paths):
            raise ValueError("Number of descriptions must match number of output paths")

        width, height = _parse_flux_size(
            size=size,
            default_width=self.width,
            default_height=self.height,
        )
        effective_labels = labels if labels else object_descriptions

        console_logger.info(
            f"Generating {len(object_descriptions)} images (FLUX.2-klein, sequential)"
        )

        for description, output_path, label in zip(
            object_descriptions, output_paths, effective_labels
        ):
            prompt = self.prompt_manager.get_prompt(
                ImageGenerationPrompts.ASSET_IMAGE_INITIAL,
                description=description,
                style_prompt=style_prompt,
            )
            start_time = time.time()
            self._run_worker(
                mode="generate",
                prompt=prompt,
                output_path=output_path,
                width=width,
                height=height,
            )
            end_time = time.time()
            console_logger.info(
                f"Generated image for {label} in {end_time - start_time:.2f} seconds "
                "(FLUX.2-klein)."
            )

    def generate_furniture_context_image(
        self,
        reference_image_path: Path,
        scene_description: str,
        width_m: float,
        length_m: float,
        output_path: Path,
    ) -> Path:
        """Generate top-down room visualization for furniture placement."""
        prompt = self.prompt_manager.get_prompt(
            ImageGenerationPrompts.FURNITURE_CONTEXT_IMAGE,
            scene_description=scene_description,
            width_m=width_m,
            length_m=length_m,
        )

        console_logger.info(
            "Generating furniture placement context image (FLUX.2-klein)"
        )

        result = self._edit_image(
            prompt=prompt,
            reference_image_path=reference_image_path,
            output_path=output_path,
        )

        console_logger.info(f"Saved context image to {output_path}")
        return result

    def generate_manipuland_context_image(
        self,
        reference_image_path: Path,
        furniture_description: str,
        furniture_dimensions: str,
        suggested_items: str,
        prompt_constraints: str,
        style_notes: str,
        output_path: Path,
    ) -> Path:
        """Generate context image showing objects placed on furniture."""
        prompt = self.prompt_manager.get_prompt(
            ImageGenerationPrompts.MANIPULAND_CONTEXT_IMAGE,
            furniture_description=furniture_description,
            furniture_dimensions=furniture_dimensions,
            suggested_items=suggested_items,
            prompt_constraints=prompt_constraints,
            style_notes=style_notes,
        )

        console_logger.info(
            "Generating manipuland placement context image (FLUX.2-klein)"
        )

        result = self._edit_image(
            prompt=prompt,
            reference_image_path=reference_image_path,
            output_path=output_path,
        )

        console_logger.info(f"Saved manipuland context image to {output_path}")
        return result

    def _edit_image(
        self,
        prompt: str,
        reference_image_path: Path,
        output_path: Path,
    ) -> Path:
        """Edit an existing image with FLUX.2-klein."""
        console_logger.info(f"Editing image {reference_image_path} (FLUX.2-klein)")

        start_time = time.time()
        self._run_worker(
            mode="edit",
            prompt=prompt,
            output_path=output_path,
            reference_image_path=reference_image_path,
            width=self.width,
            height=self.height,
        )
        end_time = time.time()

        console_logger.info(
            f"Edited image in {end_time - start_time:.2f} seconds (FLUX.2-klein)"
        )
        return output_path

    def _run_worker(
        self,
        mode: str,
        prompt: str,
        output_path: Path,
        width: int,
        height: int,
        reference_image_path: Path | None = None,
    ) -> None:
        """Run the standalone FLUX worker in the dedicated Python environment."""
        if not self.python_executable.exists():
            raise FileNotFoundError(
                "FLUX Python executable not found: "
                f"{self.python_executable}"
            )
        if not self.worker_script.exists():
            raise FileNotFoundError(
                f"FLUX worker script not found: {self.worker_script}"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(f"FLUX model path not found: {self.model_path}")
        if reference_image_path is not None and not reference_image_path.exists():
            raise FileNotFoundError(
                f"FLUX reference image not found: {reference_image_path}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        parent_snapshot = describe_gpu_snapshot(
            f"FLUX parent before launch mode={mode} output={output_path.name}"
        )
        console_logger.info(parent_snapshot)

        cmd = [
            str(self.python_executable),
            str(self.worker_script),
            "--mode",
            mode,
            "--prompt",
            prompt,
            "--output-path",
            str(output_path),
            "--model-path",
            str(self.model_path),
            "--width",
            str(width),
            "--height",
            str(height),
            "--num-inference-steps",
            str(self.num_inference_steps),
            "--guidance-scale",
            str(self.guidance_scale),
            "--max-sequence-length",
            str(self.max_sequence_length),
            "--seed",
            str(self.seed),
        ]

        if reference_image_path is not None:
            cmd.extend(["--reference-image-path", str(reference_image_path)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            console_logger.info("FLUX worker stdout:\n%s", result.stdout.rstrip())
        if result.stderr.strip():
            console_logger.warning("FLUX worker stderr:\n%s", result.stderr.rstrip())
        if result.returncode != 0:
            raise RuntimeError(
                "FLUX.2-klein subprocess failed with exit code "
                f"{result.returncode}.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}\n"
                f"{parent_snapshot}"
            )
        if not output_path.exists():
            raise FileNotFoundError(
                "FLUX.2-klein subprocess completed but did not create the output "
                f"image: {output_path}"
            )


class FallbackImageGenerator(BaseImageGenerator):
    """Wrapper that retries image generation with a fallback backend."""

    def __init__(
        self,
        primary_backend: str,
        primary_generator: BaseImageGenerator,
        fallback_backend: str,
        fallback_generator: BaseImageGenerator,
    ):
        self.primary_backend = primary_backend
        self.primary_generator = primary_generator
        self.fallback_backend = fallback_backend
        self.fallback_generator = fallback_generator

    def _run_with_fallback(self, operation_name: str, fn):
        try:
            return fn(self.primary_generator)
        except Exception as exc:
            console_logger.warning(
                "Image generation backend '%s' failed during %s; "
                "falling back to '%s': %s",
                self.primary_backend,
                operation_name,
                self.fallback_backend,
                exc,
                exc_info=True,
            )
            return fn(self.fallback_generator)

    def generate_images(
        self,
        style_prompt: str,
        object_descriptions: list[str],
        output_paths: list[Path],
        size: str | None = None,
        labels: list[str] | None = None,
    ) -> None:
        self._run_with_fallback(
            operation_name="generate_images",
            fn=lambda generator: generator.generate_images(
                style_prompt=style_prompt,
                object_descriptions=object_descriptions,
                output_paths=output_paths,
                size=size,
                labels=labels,
            ),
        )

    def generate_furniture_context_image(
        self,
        reference_image_path: Path,
        scene_description: str,
        width_m: float,
        length_m: float,
        output_path: Path,
    ) -> Path:
        return self._run_with_fallback(
            operation_name="generate_furniture_context_image",
            fn=lambda generator: generator.generate_furniture_context_image(
                reference_image_path=reference_image_path,
                scene_description=scene_description,
                width_m=width_m,
                length_m=length_m,
                output_path=output_path,
            ),
        )

    def generate_manipuland_context_image(
        self,
        reference_image_path: Path,
        furniture_description: str,
        furniture_dimensions: str,
        suggested_items: str,
        prompt_constraints: str,
        style_notes: str,
        output_path: Path,
    ) -> Path:
        return self._run_with_fallback(
            operation_name="generate_manipuland_context_image",
            fn=lambda generator: generator.generate_manipuland_context_image(
                reference_image_path=reference_image_path,
                furniture_description=furniture_description,
                furniture_dimensions=furniture_dimensions,
                suggested_items=suggested_items,
                prompt_constraints=prompt_constraints,
                style_notes=style_notes,
                output_path=output_path,
            ),
        )


def _create_single_image_generator(
    backend: str, config: DictConfig, api_base: str | None = None
) -> BaseImageGenerator:
    """Create a single concrete image generator without fallback wrapping."""
    if backend == "openai":
        return OpenAIImageGenerator(
            quality=config.openai.quality,
            api_base=api_base,
        )
    elif backend == "gemini":
        return GeminiImageGenerator(
            aspect_ratio=config.gemini.aspect_ratio,
            image_size=config.gemini.image_size,
        )
    elif backend == "flux-klein":
        return FluxKleinImageGenerator(config=config.flux_klein)
    else:
        raise ValueError(f"Unknown image generation backend: {backend}")


def create_image_generator(
    backend: str, config: DictConfig, api_base: str | None = None
) -> BaseImageGenerator:
    """Factory function to create the appropriate image generator.

    Args:
        backend: Backend to use ("openai", "gemini", or "flux-klein").
        config: Configuration object with backend-specific settings.
            Expected structure:
            - config.fallback_backend (optional fallback backend name)
            - config.openai.quality (for openai backend)
            - config.gemini.aspect_ratio (for gemini backend)
            - config.gemini.image_size (for gemini backend)
            - config.flux_klein.* (for flux-klein backend)

    Returns:
        Configured image generator instance.

    Raises:
        ValueError: If unknown backend is specified.
    """
    fallback_backend = getattr(config, "fallback_backend", None)

    try:
        primary_generator = _create_single_image_generator(
            backend=backend,
            config=config,
            api_base=api_base,
        )
    except Exception as exc:
        if not fallback_backend or fallback_backend == backend:
            raise
        console_logger.warning(
            "Failed to initialize image generation backend '%s'; "
            "falling back to '%s': %s",
            backend,
            fallback_backend,
            exc,
            exc_info=True,
        )
        return _create_single_image_generator(
            backend=fallback_backend,
            config=config,
            api_base=api_base,
        )

    if not fallback_backend or fallback_backend == backend:
        return primary_generator

    try:
        fallback_generator = _create_single_image_generator(
            backend=fallback_backend,
            config=config,
            api_base=api_base,
        )
    except Exception as exc:
        console_logger.warning(
            "Failed to initialize fallback image generation backend '%s'; "
            "continuing with primary backend '%s': %s",
            fallback_backend,
            backend,
            exc,
            exc_info=True,
        )
        return primary_generator

    return FallbackImageGenerator(
        primary_backend=backend,
        primary_generator=primary_generator,
        fallback_backend=fallback_backend,
        fallback_generator=fallback_generator,
    )


def _parse_flux_size(
    size: str | None, default_width: int, default_height: int
) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT size overrides for the FLUX backend."""
    if size is None:
        return default_width, default_height

    width_str, separator, height_str = size.lower().partition("x")
    if not separator or not width_str.isdigit() or not height_str.isdigit():
        raise ValueError(
            "FLUX size override must be in WIDTHxHEIGHT format, "
            f"got: {size}"
        )
    return int(width_str), int(height_str)


def _extract_and_save_openai_image(
    response, output_path: Path, description: str
) -> None:
    """Extract image data from OpenAI Images API response and save to file.

    Args:
        response: OpenAI Images API response object.
        output_path: Path where image will be saved.
        description: Description of the object for error messages.
    """
    if not response.data:
        raise ValueError(f"No image data returned from OpenAI for {description}")

    image_base64 = response.data[0].b64_json
    if not image_base64:
        raise ValueError(f"No base64 image data in response for {description}")

    with open(output_path, "wb") as f:
        f.write(base64.b64decode(image_base64))


def _extract_and_save_gemini_image(
    response, output_path: Path, description: str
) -> None:
    """Extract image data from Gemini response and save to file.

    Args:
        response: Gemini response object.
        output_path: Path where image will be saved.
        description: Description of the object for error messages.
    """
    # Gemini returns images in response.parts (simplified API).
    if not response.parts:
        raise ValueError(f"No parts in Gemini response for {description}")

    # Find the image part using as_image().
    for part in response.parts:
        image = part.as_image()
        if image is not None:
            image.save(str(output_path))
            return

    raise ValueError(f"No image data found in Gemini response for {description}")
