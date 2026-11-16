import unittest

from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestImageGenerationConfigDefaults(unittest.TestCase):
    def test_base_agent_configs_default_to_openai_with_flux_fallback(self):
        expected_configs = [
            "configs/furniture_agent/base_furniture_agent.yaml",
            "configs/manipuland_agent/base_manipuland_agent.yaml",
            "configs/wall_agent/base_wall_agent.yaml",
            "configs/ceiling_agent/base_ceiling_agent.yaml",
            "configs/ceiling_agent/base_ceiling_agent_4.1.yaml",
        ]

        for relative_path in expected_configs:
            with self.subTest(config=relative_path):
                cfg = OmegaConf.load(REPO_ROOT / relative_path)
                image_generation = cfg.asset_manager.image_generation

                self.assertEqual(image_generation.backend, "openai")
                self.assertEqual(image_generation.fallback_backend, "flux-klein")


if __name__ == "__main__":
    unittest.main()
