"""Tests for floor plan code-generated material configuration."""

import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf
from PIL import Image

from scenecode.agent_utils.material_generator import MaterialGeneratorConfig
from scenecode.experiments.base_experiment import BaseExperiment
from scenecode.floor_plan_agents.base_floor_plan_agent import BaseFloorPlanAgent
from scenecode.floor_plan_agents.tools.materials_resolver import (
    MaterialsConfig,
    MaterialsResolver,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeImageGenerator:
    """Small image generator fake that writes solid color texture files."""

    def __init__(self):
        self.calls = []

    def generate_images(
        self,
        style_prompt: str,
        object_descriptions: list[str],
        output_paths: list[Path],
        size: str | None = None,
        labels: list[str] | None = None,
    ):
        self.calls.append(
            {
                "style_prompt": style_prompt,
                "object_descriptions": object_descriptions,
                "output_paths": output_paths,
                "size": size,
                "labels": labels,
            }
        )
        for output_path in output_paths:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), color=(180, 150, 110)).save(output_path)
        return output_paths


class DummyFloorPlanAgent(BaseFloorPlanAgent):
    """Minimal concrete floor plan agent for configuration tests."""

    async def generate_house_layout(self, prompt: str, output_dir: Path):
        raise NotImplementedError


class TestFloorPlanMaterialsConfig(unittest.TestCase):
    """Tests for floor plan generated material configuration."""

    def test_build_floor_plan_agent_keeps_materials_server_args_for_compatibility(self):
        captured_kwargs = {}

        class FakeFloorPlanAgent:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        cfg_dict = {
            "floor_plan_agent": {"_name": "fake_floor_plan_agent"},
            "experiment": {
                "materials_retrieval_server": {
                    "host": "10.0.0.12",
                    "port": 7012,
                }
            },
        }

        BaseExperiment.build_floor_plan_agent(
            cfg_dict=cfg_dict,
            compatible_agents={"fake_floor_plan_agent": FakeFloorPlanAgent},
            logger=MagicMock(),
            render_gpu_id=2,
        )

        self.assertEqual(captured_kwargs["materials_server_host"], "10.0.0.12")
        self.assertEqual(captured_kwargs["materials_server_port"], 7012)
        self.assertEqual(captured_kwargs["render_gpu_id"], 2)

    def test_base_floor_plan_material_config_defaults_to_generation(self):
        cfg = OmegaConf.load(
            REPO_ROOT / "configs/floor_plan_agent/base_floor_plan_agent.yaml"
        )
        agent = DummyFloorPlanAgent(
            cfg=cfg,
            logger=MagicMock(),
            materials_server_host="10.0.0.22",
            materials_server_port=7022,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            agent.layout = SimpleNamespace(house_dir=Path(temp_dir))
            materials_config = agent._create_materials_config()

        self.assertFalse(materials_config.use_retrieval_server)
        self.assertTrue(materials_config.generator.enabled)
        self.assertEqual(materials_config.generator.backend, "openai")
        self.assertEqual(materials_config.generator.max_retries, 2)
        self.assertEqual(materials_config.generator.default_roughness, 128)
        self.assertEqual(materials_config.generator.texture_scale, 0.5)
        self.assertEqual(materials_config.server_host, "10.0.0.22")
        self.assertEqual(materials_config.server_port, 7022)

    @patch(
        "scenecode.agent_utils.materials_retrieval_server.client."
        "MaterialsRetrievalClient.retrieve_materials"
    )
    def test_materials_resolver_generates_pbr_material_without_retrieval(
        self, mock_retrieve_materials
    ):
        mock_retrieve_materials.side_effect = AssertionError(
            "material retrieval should not be called"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config = MaterialsConfig(
                use_retrieval_server=True,
                output_dir=output_dir,
                generator=MaterialGeneratorConfig(
                    enabled=True,
                    backend="openai",
                    max_retries=2,
                    default_roughness=128,
                    texture_scale=0.5,
                ),
            )
            fake_image_generator = FakeImageGenerator()

            resolver = MaterialsResolver(
                config,
                image_generator=fake_image_generator,
            )
            material = resolver.get_material("warm oak hardwood floor")

            self.assertIsNotNone(material)
            self.assertEqual(material.texture_scale, 0.5)
            self.assertEqual(
                material.path.parent,
                output_dir / "materials" / "generated_materials",
            )
            self.assertTrue(
                (material.path / f"{material.material_id}_Color.png").exists()
            )
            self.assertTrue(
                (material.path / f"{material.material_id}_NormalGL.png").exists()
            )
            self.assertTrue(
                (material.path / f"{material.material_id}_Roughness.png").exists()
            )
            self.assertEqual(
                set(material.get_all_textures().keys()),
                {"color", "normal", "roughness"},
            )

            # A fresh resolver can resolve generated material IDs from disk.
            fresh_resolver = MaterialsResolver(config, image_generator=None)
            resolved = fresh_resolver.get_material_by_id(material.material_id)

        mock_retrieve_materials.assert_not_called()
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.path, material.path)
        self.assertEqual(resolved.texture_scale, 0.5)
        self.assertEqual(len(fake_image_generator.calls), 1)


if __name__ == "__main__":
    unittest.main()
