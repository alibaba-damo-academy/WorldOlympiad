import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import ANY, patch

from omegaconf import OmegaConf

from scenecode.agent_utils.room import AgentType
from scenecode.furniture_agents.base_furniture_agent import BaseFurnitureAgent
from scenecode.furniture_agents.stateful_furniture_agent import StatefulFurnitureAgent
from tests.unit.mock_utils import create_mock_logger


class ConcreteFurnitureAgent(BaseFurnitureAgent):
    """Concrete implementation for testing abstract base class."""

    async def add_furniture(self, scene, scene_prompt):
        """Test implementation."""
        return "Test furniture added"


class TestBaseFurnitureAgent(unittest.TestCase):
    """Test BaseFurnitureAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_logger = create_mock_logger(self.temp_dir)

        # Load base and specific configs from actual config files.
        base_config_path = (
            Path(__file__).parent.parent.parent
            / "configs/furniture_agent/base_furniture_agent.yaml"
        )
        specific_config_path = (
            Path(__file__).parent.parent.parent
            / "configs/furniture_agent/stateful_furniture_agent.yaml"
        )
        base_config = OmegaConf.load(base_config_path)
        specific_config = OmegaConf.load(specific_config_path)

        # First merge base with specific config.
        merged_config = OmegaConf.merge(base_config, specific_config)

        # Define test overrides for fast testing.
        # Note: service_tier in agent configs references ${openai.service_tier} from
        # the top-level config.yaml which isn't loaded in tests. Provide both the
        # top-level key and override the interpolation in the agent config.
        test_overrides = {
            "openai": {
                "service_tier": None,  # Top-level openai.service_tier for interpolation
            },
            "furniture_agent": {
                "openai": {
                    "model": "gpt-4o-mini",  # Cheaper model for testing
                    "service_tier": None,  # Override interpolation directly
                },
            },
        }
        # Merge configs (base config provides all other values).
        self.test_config = OmegaConf.merge(merged_config, test_overrides)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("scenecode.furniture_agents.base_furniture_agent.AssetManager")
    @patch("scenecode.furniture_agents.base_furniture_agent.VLMService")
    @patch("scenecode.furniture_agents.base_furniture_agent.RenderingManager")
    @patch("scenecode.furniture_agents.base_furniture_agent.ConvexDecompositionServer")
    @patch("scenecode.furniture_agents.base_furniture_agent.BlenderServer")
    def test_base_furniture_agent_initialization(
        self,
        mock_blender_server_class,
        mock_convex_decomposition_server_class,
        mock_rendering_manager_class,
        mock_vlm_service_class,
        mock_asset_manager_class,
    ):
        """Test that base class initializes dependencies properly."""
        # Configure mock BlenderServer.
        mock_blender_server_class.return_value.is_running.return_value = True

        agent = ConcreteFurnitureAgent(cfg=self.test_config, logger=self.mock_logger)

        # Verify agent was initialized.
        self.assertIsNotNone(agent)
        self.assertEqual(agent.cfg, self.test_config)
        self.assertEqual(agent.logger, self.mock_logger)

        # Verify dependencies were created.
        mock_vlm_service_class.assert_called_once()
        mock_asset_manager_class.assert_called_once_with(
            logger=self.mock_logger,
            vlm_service=mock_vlm_service_class.return_value,
            blender_server=ANY,
            collision_client=ANY,
            cfg=self.test_config,
            agent_type=AgentType.FURNITURE,
            geometry_server_host="127.0.0.1",
            geometry_server_port=7000,
            hssd_server_host="127.0.0.1",
            hssd_server_port=7001,
            articulated_server_host="127.0.0.1",
            articulated_server_port=7002,
            materials_server_host="127.0.0.1",
            materials_server_port=7008,
        )
        mock_rendering_manager_class.assert_called_once_with(
            cfg=self.test_config.rendering,
            logger=self.mock_logger,
            subdirectory="furniture",
        )

    def test_abstract_method_enforcement(self):
        """Test that abstract method must be implemented."""
        # Cannot instantiate abstract base class directly.
        with self.assertRaises(TypeError):
            BaseFurnitureAgent(
                cfg=self.test_config,
                logger=self.mock_logger,
            )


class TestStatefulFurnitureAgent(unittest.TestCase):
    """Test StatefulFurnitureAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_logger = create_mock_logger(self.temp_dir)

        # Load base configuration from actual config file.
        config_path = (
            Path(__file__).parent.parent.parent
            / "configs/furniture_agent/stateful_furniture_agent.yaml"
        )
        base_config = OmegaConf.load(config_path)
        # Define test overrides for fast testing.
        test_overrides = {
            "furniture_agent": {
                "openai": {
                    "model": "gpt-4o-mini",  # Cheaper model for testing
                },
            },
        }
        # Merge configs (base config provides all other values).
        self.test_config = OmegaConf.merge(base_config, test_overrides)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_stateful_agent_class_exists(self):
        """Test that StatefulFurnitureAgent class can be imported."""
        self.assertTrue(hasattr(StatefulFurnitureAgent, "add_furniture"))
        self.assertTrue(callable(getattr(StatefulFurnitureAgent, "add_furniture")))


if __name__ == "__main__":
    unittest.main()
