"""AI-based material generation for thin coverings.

Generates PBR materials using AI image generation when library retrieval fails.
Supports tileable textures (rugs, carpets) and single artwork images (posters).
"""

import logging
import re
import time

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from PIL import Image

from scenecode.prompts.manager import PromptManager
from scenecode.prompts.registry import MaterialGenerationPrompts
from scenecode.utils.material import Material

if TYPE_CHECKING:
    from scenecode.agent_utils.image_generation import BaseImageGenerator

console_logger = logging.getLogger(__name__)


@dataclass
class MaterialGeneratorConfig:
    """Configuration for AI-based material generation."""

    enabled: bool = False
    backend: str = "openai"  # "openai" or "gemini"
    max_retries: int = 2
    default_roughness: int = 128  # 0-255 grayscale
    texture_scale: float = 0.5  # Meters per tile for tileable textures


class MaterialGenerator:
    """Generates PBR materials using AI image generation.

    Creates complete PBR material folders with:
    - Color texture from AI image generation
    - Flat normal map (RGB 128,128,255)
    - Uniform roughness map (grayscale)

    Supports two modes:
    - Tileable: Seamless textures for repeating patterns (rugs, carpets)
    - Artwork: Single images spanning entire surface (posters, paintings)
    """

    def __init__(
        self,
        config: MaterialGeneratorConfig,
        output_dir: Path,
        image_generator: "BaseImageGenerator",
    ) -> None:
        """Initialize material generator.

        Args:
            config: Generator configuration.
            output_dir: Directory for generated material folders.
            image_generator: AI image generator (OpenAI or Gemini).
        """
        self.config = config
        self.output_dir = output_dir
        self.image_generator = image_generator
        self.prompt_manager = PromptManager(
            prompts_dir=Path(__file__).parent.parent / "prompts" / "data"
        )

    def generate_material(
        self, description: str, material_id: str | None = None
    ) -> Material | None:
        """Generate tileable texture with seamless post-processing.

        Creates a PBR material suitable for repeating patterns.
        Applies seamless edge blending for natural tiling.
        Always generates square textures since tiling handles non-square surfaces.

        Args:
            description: Material description (e.g., "persian rug pattern").
            material_id: Optional custom ID. Auto-generated if not provided.

        Returns:
            Material with texture_scale=0.5 (tiles every 0.5m), or None if failed.
        """
        if material_id is None:
            material_id = _sanitize_material_id(description)

        material_dir = self.output_dir / material_id
        material_dir.mkdir(parents=True, exist_ok=True)

        console_logger.info(f"Generating tileable material: {description}")

        # For tileable textures, always use square (1:1) regardless of surface dimensions.
        # Tiling handles non-square surfaces naturally.
        image_size = get_optimal_aspect_ratio(
            width=1.0, height=1.0, backend=self.config.backend
        )

        try:
            # Generate color texture.
            color_path = material_dir / f"{material_id}_Color.png"
            prompt = self.prompt_manager.get_prompt(
                MaterialGenerationPrompts.SEAMLESS_TEXTURE,
                material_description=description,
            )

            # Use image generator to create color texture.
            self.image_generator.generate_images(
                style_prompt="",  # No additional style needed.
                object_descriptions=[prompt],
                output_paths=[color_path],
                size=image_size,
                labels=[description],  # Log short description, not full prompt.
            )

            if not color_path.exists():
                console_logger.error(
                    f"Failed to generate color texture for {description}"
                )
                return None

            # Apply seamless post-processing.
            color_image = Image.open(color_path)
            seamless_image = make_seamless(color_image)
            seamless_image.save(color_path)
            console_logger.info(f"Applied seamless processing to {color_path.name}")

            # Generate flat normal and roughness maps.
            normal_path = material_dir / f"{material_id}_NormalGL.png"
            roughness_path = material_dir / f"{material_id}_Roughness.png"

            normal_map = create_flat_normal_map(size=seamless_image.width)
            roughness_map = create_uniform_roughness_map(
                size=seamless_image.width, value=self.config.default_roughness
            )

            normal_map.save(normal_path)
            roughness_map.save(roughness_path)

            console_logger.info(f"Created PBR material at {material_dir}")

            return Material(
                path=material_dir,
                material_id=material_id,
                texture_scale=self.config.texture_scale,
            )

        except Exception as e:
            console_logger.error(f"Material generation failed: {e}")
            return None

    def generate_artwork(
        self,
        description: str,
        width: float,
        height: float,
        material_id: str | None = None,
    ) -> Material | None:
        """Generate single artwork image (poster, painting, wall art).

        Creates a PBR material where the image spans the entire surface.
        No tiling or seamless processing applied.

        Args:
            description: Artwork description (e.g., "Van Gogh Starry Night").
            width: Physical width in meters (for aspect ratio selection).
            height: Physical height in meters (for aspect ratio selection).
            material_id: Optional custom ID. Auto-generated if not provided.

        Returns:
            Material with texture_scale=None (cover mode), or None if failed.
        """
        if material_id is None:
            material_id = _sanitize_material_id(description)

        material_dir = self.output_dir / material_id
        material_dir.mkdir(parents=True, exist_ok=True)

        console_logger.info(f"Generating artwork material: {description}")

        # For artwork, match aspect ratio to surface dimensions if provided.
        image_size = get_optimal_aspect_ratio(
            width=width, height=height, backend=self.config.backend
        )
        console_logger.info(
            f"Using aspect ratio {image_size} for {width}x{height}m surface"
        )

        try:
            # Generate artwork image.
            color_path = material_dir / f"{material_id}_Color.png"
            prompt = self.prompt_manager.get_prompt(
                MaterialGenerationPrompts.ARTWORK_IMAGE,
                artwork_description=description,
            )

            # Use image generator to create artwork.
            self.image_generator.generate_images(
                style_prompt="",  # No additional style needed.
                object_descriptions=[prompt],
                output_paths=[color_path],
                size=image_size,
                labels=[description],  # Log short description, not full prompt.
            )

            if not color_path.exists():
                console_logger.error(f"Failed to generate artwork for {description}")
                return None

            # Get image dimensions for PBR map sizing.
            artwork_image = Image.open(color_path)
            img_size = artwork_image.width

            # Generate flat normal and roughness maps.
            normal_path = material_dir / f"{material_id}_NormalGL.png"
            roughness_path = material_dir / f"{material_id}_Roughness.png"

            normal_map = create_flat_normal_map(size=img_size)
            roughness_map = create_uniform_roughness_map(
                size=img_size, value=self.config.default_roughness
            )

            normal_map.save(normal_path)
            roughness_map.save(roughness_path)

            console_logger.info(f"Created artwork material at {material_dir}")

            # Cover mode: texture_scale=None means image spans entire surface.
            return Material(
                path=material_dir, material_id=material_id, texture_scale=None
            )

        except Exception as e:
            console_logger.error(f"Artwork generation failed: {e}")
            return None


def make_seamless(image: Image.Image, blend_width: int = 64) -> Image.Image:
    """Make texture seamless by blending edges.

    Uses gradient blending at borders to ensure left/right and top/bottom
    edges match when tiled.

    Args:
        image: Input image to make seamless.
        blend_width: Width of blend zone in pixels.

    Returns:
        Image with seamless edges for tiling.
    """
    arr = np.array(image)
    h, w = arr.shape[:2]

    # Clamp blend width to half the image size.
    blend_width = min(blend_width, w // 4, h // 4)

    if blend_width < 2:
        # Image too small for blending.
        return image

    # Create horizontal blend mask (left-right continuity).
    h_mask = np.linspace(0, 1, blend_width).reshape(1, -1, 1)

    # Blend left edge with right edge.
    left_strip = arr[:, :blend_width].astype(np.float32)
    right_strip = arr[:, -blend_width:].astype(np.float32)
    blended_h = (left_strip * (1 - h_mask) + right_strip * h_mask).astype(np.uint8)
    arr[:, :blend_width] = blended_h
    arr[:, -blend_width:] = blended_h

    # Create vertical blend mask (top-bottom continuity).
    v_mask = np.linspace(0, 1, blend_width).reshape(-1, 1, 1)

    # Blend top edge with bottom edge.
    top_strip = arr[:blend_width, :].astype(np.float32)
    bottom_strip = arr[-blend_width:, :].astype(np.float32)
    blended_v = (top_strip * (1 - v_mask) + bottom_strip * v_mask).astype(np.uint8)
    arr[:blend_width, :] = blended_v
    arr[-blend_width:, :] = blended_v

    return Image.fromarray(arr)


def get_optimal_aspect_ratio(width: float, height: float, backend: str) -> str:
    """Select closest supported aspect ratio for the given dimensions.

    Args:
        width: Physical width in meters.
        height: Physical height in meters.
        backend: "openai" or "gemini".

    Returns:
        Backend-specific aspect ratio or size string.
    """
    ratio = width / height

    if backend == "openai":
        # OpenAI gpt-image-1.5 supported sizes: 1024x1024, 1024x1536, 1536x1024, auto.
        if ratio > 1.3:
            return "1536x1024"  # landscape (1.5:1)
        elif ratio < 0.77:
            return "1024x1536"  # portrait (1:1.5)
        else:
            return "1024x1024"  # square

    elif backend == "gemini":
        # Gemini ratios: 1:1, 16:9, 9:16, 3:4, 4:3.
        supported = {
            1.0: "1:1",
            16 / 9: "16:9",  # 1.78
            9 / 16: "9:16",  # 0.56
            4 / 3: "4:3",  # 1.33
            3 / 4: "3:4",  # 0.75
        }
        closest = min(supported.keys(), key=lambda r: abs(r - ratio))
        return supported[closest]

    return "1:1"


def create_flat_normal_map(size: int = 1024) -> Image.Image:
    """Create flat normal map (RGB 128,128,255).

    Represents a surface pointing straight up with no bumps.

    Args:
        size: Image dimensions in pixels (creates square image).

    Returns:
        PIL Image with flat normal map.
    """
    arr = np.full((size, size, 3), [128, 128, 255], dtype=np.uint8)
    return Image.fromarray(arr)


def create_uniform_roughness_map(size: int = 1024, value: int = 128) -> Image.Image:
    """Create uniform roughness map (grayscale).

    Args:
        size: Image dimensions in pixels (creates square image).
        value: Roughness value (0=smooth/shiny, 255=rough/matte).

    Returns:
        PIL Image with uniform roughness.
    """
    arr = np.full((size, size), value, dtype=np.uint8)
    return Image.fromarray(arr)


def _sanitize_material_id(description: str) -> str:
    """Convert description to valid folder name.

    Args:
        description: Material description text.

    Returns:
        Safe folder name with timestamp for uniqueness.
    """
    # Remove special chars, replace spaces with underscores.
    sanitized = re.sub(r"[^\w\s-]", "", description)
    sanitized = re.sub(r"\s+", "_", sanitized)
    # Truncate to reasonable length.
    sanitized = sanitized[:50]
    # Add timestamp for uniqueness.
    return f"{sanitized}_{int(time.time())}"
