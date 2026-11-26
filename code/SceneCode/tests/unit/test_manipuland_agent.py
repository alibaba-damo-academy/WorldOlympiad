import asyncio
import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

from omegaconf import OmegaConf

from scenecode.agent_utils.room import AgentType, RoomScene
from scenecode.manipuland_agents.base_manipuland_agent import BaseManipulandAgent
from tests.unit.mock_utils import create_mock_logger


class ConcreteManipulandAgent(BaseManipulandAgent):
    """Concrete implementation for testing abstract base class."""

    async def add_manipulands(self, scene):
        """Test implementation."""
        return "Test manipulands added"


class TestBaseManipulandAgent(unittest.TestCase):
    """Test BaseManipulandAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_logger = create_mock_logger(self.temp_dir)

        # Load configuration from actual config file.
        config_path = (
            Path(__file__).parent.parent.parent
            / "configs/manipuland_agent/base_manipuland_agent.yaml"
        )
        base_config = OmegaConf.load(config_path)

        # Note: service_tier in agent configs references ${openai.service_tier} from
        # the top-level config.yaml which isn't loaded in tests. Provide both the
        # top-level key and override the interpolation in the agent config.
        test_overrides = {
            "openai": {
                "service_tier": None,  # Top-level openai.service_tier for interpolation
            },
            "manipuland_agent": {
                "openai": {
                    "service_tier": None,  # Override interpolation directly
                },
            },
        }
        self.config = OmegaConf.merge(base_config, test_overrides)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("scenecode.manipuland_agents.base_manipuland_agent.AssetManager")
    @patch("scenecode.manipuland_agents.base_manipuland_agent.VLMService")
    @patch("scenecode.manipuland_agents.base_manipuland_agent.RenderingManager")
    @patch(
        "scenecode.manipuland_agents.base_manipuland_agent.ConvexDecompositionServer"
    )
    @patch("scenecode.manipuland_agents.base_manipuland_agent.BlenderServer")
    def test_initialization(
        self,
        mock_blender_server_class,
        mock_convex_decomposition_server_class,
        mock_rendering_manager_class,
        mock_vlm_service_class,
        mock_asset_manager_class,
    ):
        """Test BaseManipulandAgent initialization."""
        # Configure mock BlenderServer.
        mock_blender_server_class.return_value.is_running.return_value = True

        agent = ConcreteManipulandAgent(cfg=self.config, logger=self.mock_logger)

        self.assertEqual(agent.cfg, self.config)
        self.assertEqual(agent.logger, self.mock_logger)

        # Verify dependencies were created.
        mock_vlm_service_class.assert_called_once()
        mock_convex_decomposition_server_class.assert_called_once()
        mock_asset_manager_class.assert_called_once_with(
            logger=self.mock_logger,
            vlm_service=mock_vlm_service_class.return_value,
            blender_server=ANY,
            collision_client=ANY,
            cfg=self.config,
            agent_type=AgentType.MANIPULAND,
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
            cfg=self.config.rendering, logger=self.mock_logger
        )

    @patch("scenecode.manipuland_agents.base_manipuland_agent.AssetManager")
    @patch("scenecode.manipuland_agents.base_manipuland_agent.VLMService")
    @patch("scenecode.manipuland_agents.base_manipuland_agent.RenderingManager")
    @patch(
        "scenecode.manipuland_agents.base_manipuland_agent.ConvexDecompositionServer"
    )
    @patch("scenecode.manipuland_agents.base_manipuland_agent.BlenderServer")
    def test_abstract_method_implemented(
        self,
        mock_blender_server_class,
        mock_convex_decomposition_server_class,
        mock_rendering_manager_class,
        mock_vlm_service_class,
        mock_asset_manager_class,
    ):
        """Test that concrete class implements abstract method."""
        # Configure mock BlenderServer.
        mock_blender_server_class.return_value.is_running.return_value = True

        agent = ConcreteManipulandAgent(cfg=self.config, logger=self.mock_logger)

        # Should be able to call add_manipulands without TypeError.
        mock_scene = MagicMock(spec=RoomScene)
        result = asyncio.run(agent.add_manipulands(mock_scene))
        self.assertIsNotNone(result)

    def test_abstract_method_not_implemented_raises_error(self):
        """Test that instantiating abstract class directly raises TypeError."""
        with self.assertRaises(TypeError):
            BaseManipulandAgent(cfg=self.config, logger=self.mock_logger)


if __name__ == "__main__":
    unittest.main()
