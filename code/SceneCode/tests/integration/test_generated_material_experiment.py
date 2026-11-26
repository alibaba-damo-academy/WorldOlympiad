"""Opt-in integration tests for generated material experiments.

These tests exercise generated materials directly without changing the scene
pipeline. Real image generation is guarded by an environment variable because it
may call OpenAI, fall back to local FLUX, consume API credits, or require GPU.
"""

import os
import sys
import unittest

from pathlib import Path
from unittest.mock import MagicMock, call, patch

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scenecode.agent_utils.image_generation import (
    FallbackImageGenerator,
    create_image_generator,
)
from scenecode.agent_utils.material_generator import (
    MaterialGenerator,
    MaterialGeneratorConfig,
)
from scenecode.utils.material import Material


RUN_REAL_IMAGE_GEN_TESTS = os.environ.get("SCENECODE_RUN_REAL_IMAGE_GEN_TESTS") == "1"
DEFAULT_GENERATED_MATERIAL_OUTPUT_DIR = REPO_ROOT / "outputs/test_material"


def _load_image_generation_config():
    """Load the same image-generation config shape used by placement agents."""
    cfg = OmegaConf.load(REPO_ROOT / "configs/furniture_agent/base_furniture_agent.yaml")
    return cfg.asset_manager.image_generation


def _assert_material_has_pbr_textures(
    test_case: unittest.TestCase, material: Material | None
) -> Material:
    test_case.assertIsNotNone(material)
    assert material is not None
    test_case.assertTrue(material.path.exists())

    textures = material.get_all_textures()
    test_case.assertIn("color", textures)
    test_case.assertIn("normal", textures)
    test_case.assertIn("roughness", textures)

    for texture_path in textures.values():
        test_case.assertTrue(texture_path.exists(), f"Missing texture: {texture_path}")
        test_case.assertGreater(texture_path.stat().st_size, 0)

    return material


class TestGeneratedMaterialImageGeneratorFactory(unittest.TestCase):
    """Fast contract tests that do not invoke real image generation services."""

    def test_image_generator_factory_has_openai_primary_flux_fallback_contract(self):
        image_generation_cfg = _load_image_generation_config()

        openai_generator = MagicMock(name="openai_generator")
        flux_generator = MagicMock(name="flux_generator")

        with patch(
            "scenecode.agent_utils.image_generation._create_single_image_generator",
            side_effect=[openai_generator, flux_generator],
        ) as mock_create:
            generator = create_image_generator(
                backend="openai",
                config=image_generation_cfg,
                api_base="https://example.invalid/v1",
            )

        self.assertIsInstance(generator, FallbackImageGenerator)
        self.assertEqual(generator.primary_backend, "openai")
        self.assertEqual(generator.fallback_backend, "flux-klein")
        self.assertIs(generator.primary_generator, openai_generator)
        self.assertIs(generator.fallback_generator, flux_generator)
        mock_create.assert_has_calls(
            [
                call(
                    backend="openai",
                    config=image_generation_cfg,
                    api_base="https://example.invalid/v1",
                ),
                call(
                    backend="flux-klein",
                    config=image_generation_cfg,
                    api_base="https://example.invalid/v1",
                ),
            ]
        )


@unittest.skipUnless(
    RUN_REAL_IMAGE_GEN_TESTS,
    "Set SCENECODE_RUN_REAL_IMAGE_GEN_TESTS=1 to run real image generation tests.",
)
class TestGeneratedMaterialExperimentWithRealImageGenerator(unittest.TestCase):
    """Real generated-material tests for the experimental non-retrieval path."""

    def setUp(self):
        output_dir = os.environ.get("SCENECODE_GENERATED_MATERIAL_OUTPUT_DIR")
        self.output_dir = (
            Path(output_dir) if output_dir else DEFAULT_GENERATED_MATERIAL_OUTPUT_DIR
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        image_generation_cfg = _load_image_generation_config()
        self.image_generator = create_image_generator(
            backend="openai",
            config=image_generation_cfg,
        )
        self.material_generator = MaterialGenerator(
            config=MaterialGeneratorConfig(
                enabled=True,
                backend="openai",
                max_retries=1,
                default_roughness=128,
                texture_scale=0.5,
            ),
            output_dir=self.output_dir,
            image_generator=self.image_generator,
        )

    def tearDown(self):
        # Keep generated materials on disk for manual inspection.
        pass

    @patch(
        "scenecode.agent_utils.materials_retrieval_server.client."
        "MaterialsRetrievalClient.retrieve_materials",
        side_effect=AssertionError("Generated-material experiment used retrieval"),
    )
    def test_real_image_generator_creates_floor_wall_and_rug_materials_without_retrieval(
        self, mock_retrieve_materials
    ):
        material_descriptions = [
            "warm oak hardwood floor",
            "smooth white plaster wall",
            "persian rug pattern",
        ]

        for description in material_descriptions:
            with self.subTest(description=description):
                material = self.material_generator.generate_material(
                    description=description
                )
                material = _assert_material_has_pbr_textures(self, material)
                self.assertEqual(material.texture_scale, 0.5)

        mock_retrieve_materials.assert_not_called()

    @patch(
        "scenecode.agent_utils.materials_retrieval_server.client."
        "MaterialsRetrievalClient.retrieve_materials",
        side_effect=AssertionError("Generated-material experiment used retrieval"),
    )
    def test_real_image_generator_creates_artwork_cover_material(
        self, mock_retrieve_materials
    ):
        material = self.material_generator.generate_artwork(
            description="abstract geometric poster with saturated red and blue shapes",
            width=0.8,
            height=1.2,
        )

        material = _assert_material_has_pbr_textures(self, material)
        self.assertIsNone(material.texture_scale)
        mock_retrieve_materials.assert_not_called()


if __name__ == "__main__":
    unittest.main()
