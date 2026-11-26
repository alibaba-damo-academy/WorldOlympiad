"""Unit tests for MaterialGenerator.

Focuses on contracts and business logic rather than implementation details.
"""

import unittest

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from PIL import Image

from scenecode.agent_utils.material_generator import (
    MaterialGenerator,
    MaterialGeneratorConfig,
    create_flat_normal_map,
    create_uniform_roughness_map,
    get_optimal_aspect_ratio,
    make_seamless,
)


class TestPBRMapGeneration(unittest.TestCase):
    """Tests for PBR texture map generation - these are core contracts."""

    def test_flat_normal_map_correct_values(self) -> None:
        """Flat normal map has correct RGB values for 'pointing up' surface.

        This is the contract: (128, 128, 255) represents a flat surface in normal maps.
        """
        normal = create_flat_normal_map(size=64)
        arr = np.array(normal)

        assert arr.shape == (64, 64, 3)
        # All pixels must be (128, 128, 255).
        assert np.all(arr[:, :, 0] == 128)
        assert np.all(arr[:, :, 1] == 128)
        assert np.all(arr[:, :, 2] == 255)

    def test_roughness_map_uniform_value(self) -> None:
        """Roughness map has uniform grayscale value."""
        roughness = create_uniform_roughness_map(size=64, value=200)
        arr = np.array(roughness)

        assert arr.shape == (64, 64)
        assert np.all(arr == 200)


class TestSeamlessProcessing(unittest.TestCase):
    """Tests for seamless texture processing - edge blending contract."""

    def test_preserves_dimensions(self) -> None:
        """Seamless processing preserves image dimensions."""
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        result = make_seamless(img)
        assert result.size == img.size

    def test_handles_small_images(self) -> None:
        """Small images don't crash (graceful degradation)."""
        img = Image.fromarray(np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8))
        result = make_seamless(img)
        assert result.size == img.size


class TestAspectRatioMapping(unittest.TestCase):
    """Tests for aspect ratio selection - maps surface dimensions to backend sizes."""

    def test_openai_wide_landscape(self) -> None:
        """OpenAI wide surface returns landscape size."""
        result = get_optimal_aspect_ratio(width=2.0, height=1.0, backend="openai")
        self.assertEqual(result, "1536x1024")

    def test_openai_tall_portrait(self) -> None:
        """OpenAI tall surface returns portrait size."""
        result = get_optimal_aspect_ratio(width=1.0, height=2.0, backend="openai")
        self.assertEqual(result, "1024x1536")

    def test_openai_square(self) -> None:
        """OpenAI square surface returns square size."""
        result = get_optimal_aspect_ratio(width=1.0, height=1.0, backend="openai")
        self.assertEqual(result, "1024x1024")

    def test_gemini_square(self) -> None:
        """Gemini square surface returns 1:1 ratio."""
        result = get_optimal_aspect_ratio(width=1.0, height=1.0, backend="gemini")
        self.assertEqual(result, "1:1")

    def test_gemini_wide_landscape(self) -> None:
        """Gemini wide surface returns 16:9 ratio."""
        result = get_optimal_aspect_ratio(width=1.78, height=1.0, backend="gemini")
        self.assertEqual(result, "16:9")

    def test_gemini_tall_portrait(self) -> None:
        """Gemini tall surface returns 9:16 ratio."""
        result = get_optimal_aspect_ratio(width=1.0, height=1.78, backend="gemini")
        self.assertEqual(result, "9:16")

    def test_unknown_backend_defaults(self) -> None:
        """Unknown backend defaults to 1:1."""
        result = get_optimal_aspect_ratio(width=2.0, height=1.0, backend="unknown")
        self.assertEqual(result, "1:1")


class TestMaterialGeneratorContracts(unittest.TestCase):
    """Tests for MaterialGenerator - focus on return value contracts."""

    def _create_mock_generator(
        self, tmp_path: Path
    ) -> tuple[MaterialGenerator, MagicMock]:
        """Create MaterialGenerator with mock image generator."""
        mock_img_gen = MagicMock()

        # Default mock behavior: create color image when called.
        def create_image(**kwargs) -> None:
            for path in kwargs.get("output_paths", []):
                img = Image.fromarray(
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                )
                img.save(path)

        mock_img_gen.generate_images.side_effect = create_image

        config = MaterialGeneratorConfig(
            enabled=True, backend="openai", texture_scale=0.5
        )
        generator = MaterialGenerator(
            config=config, output_dir=tmp_path, image_generator=mock_img_gen
        )
        return generator, mock_img_gen

    def test_generate_material_returns_material_with_texture_scale(self) -> None:
        """generate_material returns Material with configured texture_scale."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            generator, _ = self._create_mock_generator(tmp_path)
            material = generator.generate_material(description="test pattern")

            self.assertIsNotNone(material)
            self.assertEqual(material.texture_scale, 0.5)  # From config.
            self.assertTrue(material.path.exists())

    def test_generate_artwork_returns_material_with_no_tiling(self) -> None:
        """generate_artwork returns Material with texture_scale=None (cover mode)."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            generator, _ = self._create_mock_generator(tmp_path)
            material = generator.generate_artwork(
                description="test artwork", width=1.5, height=1.0
            )

            self.assertIsNotNone(material)
            self.assertIsNone(material.texture_scale)  # Cover mode.
            self.assertTrue(material.path.exists())

    def test_returns_none_when_image_generation_fails(self) -> None:
        """Returns None when image generator fails to create files."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            generator, mock_img_gen = self._create_mock_generator(tmp_path)
            # Override to not create any files.
            mock_img_gen.generate_images.side_effect = None

            material = generator.generate_material(description="test")

            self.assertIsNone(material)


if __name__ == "__main__":
    unittest.main()
