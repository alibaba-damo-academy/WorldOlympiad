import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock

from omegaconf import OmegaConf

from scenecode.agent_utils.room import RoomScene
from scenecode.agent_utils.scene_analyzer import SceneAnalyzer


class TestSceneAnalyzer(unittest.TestCase):
    """Test SceneAnalyzer class contracts."""

    # Test configuration constants.
    TEST_MODEL = "gpt-4o-mini"
    TEST_REASONING_EFFORT = "low"

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_vlm_service = Mock()
        self.mock_rendering_manager = Mock()
        self.mock_scene = Mock(spec=RoomScene)

        # Create mock BlenderServer.
        self.mock_blender_server = Mock()
        self.mock_blender_server.is_running.return_value = True

        # Create test config (only OpenAI settings needed).
        test_config_dict = {
            "openai": {
                "model": self.TEST_MODEL,
                "vision_detail": "low",
                "reasoning_effort": {"scene_critique": self.TEST_REASONING_EFFORT},
                "verbosity": {"scene_critique": "low"},
            },
        }
        # Convert to OmegaConf to match expected structure.
        self.test_config = OmegaConf.create(test_config_dict)

        self.scene_analyzer = SceneAnalyzer(
            vlm_service=self.mock_vlm_service,
            rendering_manager=self.mock_rendering_manager,
            cfg=self.test_config,
            blender_server=self.mock_blender_server,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scene_analyzer_initialization(self):
        """Test that SceneAnalyzer initializes properly."""
        self.assertIsNotNone(self.scene_analyzer)
        self.assertEqual(self.scene_analyzer.vlm_service, self.mock_vlm_service)
        self.assertEqual(
            self.scene_analyzer.rendering_manager, self.mock_rendering_manager
        )
        self.assertEqual(self.scene_analyzer.cfg, self.test_config)
        self.assertEqual(self.scene_analyzer.blender_server, self.mock_blender_server)

    def test_configuration_access(self):
        """Test that SceneAnalyzer can access configuration values."""
        # Verify configuration was stored and accessible.
        self.assertEqual(self.scene_analyzer.cfg["openai"]["model"], self.TEST_MODEL)
        self.assertEqual(
            self.scene_analyzer.cfg["openai"]["reasoning_effort"]["scene_critique"],
            self.TEST_REASONING_EFFORT,
        )


if __name__ == "__main__":
    unittest.main()
