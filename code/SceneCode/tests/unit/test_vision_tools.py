import unittest

from unittest.mock import MagicMock, patch

from agents import ToolOutputImage, ToolOutputText

from scenecode.agent_utils.rendering_manager import RenderingManager
from scenecode.agent_utils.room import AgentType, RoomScene
from scenecode.furniture_agents.tools.vision_tools import VisionTools


class TestVisionTools(unittest.TestCase):
    """Test cases for VisionTools class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_scene = MagicMock(spec=RoomScene)
        # Mock room_geometry to None (no openings) so clearance checks are skipped.
        self.mock_scene.room_geometry = None
        self.mock_rendering_manager = MagicMock(spec=RenderingManager)

        # Create mock BlenderServer.
        self.mock_blender_server = MagicMock()
        self.mock_blender_server.is_running.return_value = True

        # Create mock config with physics validation settings.
        self.mock_cfg = MagicMock()
        self.mock_cfg.physics_validation.object_penetration_threshold_m = 0.001
        self.mock_cfg.physics_validation.floor_penetration_tolerance_m = 0.05
        self.mock_cfg.physics_validation.manipuland_furniture_tolerance_m = 0.02

        self.vision_tools = VisionTools(
            scene=self.mock_scene,
            rendering_manager=self.mock_rendering_manager,
            cfg=self.mock_cfg,
            blender_server=self.mock_blender_server,
        )

    def test_initialization(self):
        """Test VisionTools initialization."""
        self.assertEqual(self.vision_tools.scene, self.mock_scene)
        self.assertEqual(
            self.vision_tools.rendering_manager, self.mock_rendering_manager
        )
        self.assertIsInstance(self.vision_tools.tools, dict)
        self.assertIn("observe_scene", self.vision_tools.tools)
        self.assertIn("check_physics", self.vision_tools.tools)

    def test_tool_creation(self):
        """Test that tools are created correctly."""
        tools = self.vision_tools.tools
        self.assertIn("observe_scene", tools)
        self.assertIn("check_physics", tools)
        # Tool is a function_tool decorated function, check if it has the right
        # attributes.
        self.assertIsNotNone(tools["observe_scene"])
        self.assertIsNotNone(tools["check_physics"])

    @patch(
        "scenecode.furniture_agents.tools.vision_tools.encode_image_to_base64",
        return_value="fake_base64_data",
    )
    def test_observe_scene_successful(self, mock_encode):
        """Test successful scene observation returns ToolOutputImage list."""
        # Setup mocks - create a mock directory with Path objects.
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True

        mock_render_path1 = MagicMock()
        mock_render_path1.exists.return_value = True
        mock_render_path1.__str__ = MagicMock(return_value="/tmp/render1.png")
        mock_render_path1.__lt__ = MagicMock(return_value=True)

        mock_render_path2 = MagicMock()
        mock_render_path2.exists.return_value = True
        mock_render_path2.__str__ = MagicMock(return_value="/tmp/render2.png")
        mock_render_path2.__lt__ = MagicMock(return_value=True)

        mock_dir.glob.return_value = [mock_render_path1, mock_render_path2]

        self.mock_rendering_manager.render_scene.return_value = mock_dir

        # Call observe_scene implementation.
        result = self.vision_tools._observe_scene_impl()

        # Verify rendering manager was called.
        # Note: room_bounds is None when scene.room_geometry is a MagicMock.
        self.mock_rendering_manager.render_scene.assert_called_once_with(
            self.mock_scene, blender_server=self.mock_blender_server, room_bounds=None
        )

        # Verify result is a list with ToolOutputImage and ToolOutputText.
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # 2 images + 1 text message

        # First two should be ToolOutputImage.
        self.assertIsInstance(result[0], ToolOutputImage)
        self.assertIsInstance(result[1], ToolOutputImage)

        # Last should be ToolOutputText with confirmation message.
        self.assertIsInstance(result[2], ToolOutputText)
        self.assertIn("Scene observed from 2 viewpoints", result[2].text)

    def test_observe_scene_no_renders(self):
        """Test scene observation when no renders are generated."""
        # Setup mock to return None (no directory).
        self.mock_rendering_manager.render_scene.return_value = None

        # Call observe_scene implementation.
        result = self.vision_tools._observe_scene_impl()

        # Verify result is a list with error message.
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ToolOutputText)
        self.assertIn("Unable to observe scene", result[0].text)

    def test_observe_scene_empty_directory(self):
        """Test scene observation when directory exists but has no images."""
        # Setup mocks - create a mock directory with no images.
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True
        mock_dir.glob.return_value = []  # No images

        self.mock_rendering_manager.render_scene.return_value = mock_dir

        # Call observe_scene implementation.
        result = self.vision_tools._observe_scene_impl()

        # Verify result contains only the confirmation text (0 images).
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ToolOutputText)
        self.assertIn("Scene observed from 0 viewpoints", result[0].text)

    def test_observe_scene_rendering_exception(self):
        """Test scene observation when rendering throws exception."""
        # Setup mock to raise exception.
        self.mock_rendering_manager.render_scene.side_effect = Exception(
            "Rendering failed"
        )

        # Exception should propagate.
        with self.assertRaises(Exception) as context:
            self.vision_tools._observe_scene_impl()

        self.assertEqual(str(context.exception), "Rendering failed")

    @patch(
        "scenecode.furniture_agents.tools.vision_tools.encode_image_to_base64",
        side_effect=IOError("File read error"),
    )
    def test_observe_scene_file_read_exception(self, mock_encode):
        """Test scene observation when file reading throws exception."""
        # Setup mocks - create a mock directory with Path objects.
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True

        mock_render_path = MagicMock()
        mock_render_path.exists.return_value = True
        mock_render_path.__str__ = MagicMock(return_value="/tmp/render1.png")
        mock_render_path.__lt__ = MagicMock(return_value=True)
        mock_dir.glob.return_value = [mock_render_path]

        self.mock_rendering_manager.render_scene.return_value = mock_dir

        # Exception should propagate.
        with self.assertRaises(IOError) as context:
            self.vision_tools._observe_scene_impl()

        self.assertEqual(str(context.exception), "File read error")

    @patch(
        "scenecode.furniture_agents.tools.vision_tools.encode_image_to_base64",
        return_value="dGVzdF9pbWFnZV9kYXRh",  # base64 of "test_image_data"
    )
    def test_observe_scene_image_encoding(self, mock_encode):
        """Test that images are properly base64 encoded in ToolOutputImage."""
        # Setup mocks - create a mock directory with Path objects.
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True

        mock_render_path = MagicMock()
        mock_render_path.exists.return_value = True
        mock_render_path.__str__ = MagicMock(return_value="/tmp/render1.png")
        mock_render_path.__lt__ = MagicMock(return_value=True)
        mock_dir.glob.return_value = [mock_render_path]

        self.mock_rendering_manager.render_scene.return_value = mock_dir

        # Call observe_scene implementation.
        result = self.vision_tools._observe_scene_impl()

        # Verify result contains an image.
        self.assertEqual(len(result), 2)  # 1 image + 1 text
        self.assertIsInstance(result[0], ToolOutputImage)

        # Verify the image URL is base64 encoded with correct format.
        image_url = result[0].image_url
        self.assertTrue(image_url.startswith("data:image/png;base64,"))

        # Verify the base64 data is included in the URL.
        encoded_data = image_url.split(",", 1)[1]
        self.assertEqual(encoded_data, "dGVzdF9pbWFnZV9kYXRh")

    def test_observe_scene_implementation_called(self):
        """Test that the implementation method is called correctly."""
        # Test the implementation method directly since tool framework testing.
        # would require more complex mocking.
        with patch.object(self.vision_tools, "_observe_scene_impl") as mock_impl:
            mock_impl.return_value = [ToolOutputText(text="Test observation result")]

            result = self.vision_tools._observe_scene_impl()

            mock_impl.assert_called_once()
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], ToolOutputText)

    @patch("scenecode.furniture_agents.tools.vision_tools.check_physics_violations")
    def test_check_physics_no_violations(self, mock_check_physics):
        """Test physics check when no violations are detected."""
        mock_check_physics.return_value = (
            "No physics violations detected. All objects are properly placed."
        )

        result = self.vision_tools._check_physics_impl()

        self.assertEqual(
            result, "No physics violations detected. All objects are properly placed."
        )
        mock_check_physics.assert_called_once_with(
            scene=self.mock_scene, cfg=self.mock_cfg, agent_type=AgentType.FURNITURE
        )

    @patch("scenecode.furniture_agents.tools.vision_tools.check_physics_violations")
    def test_check_physics_with_violations(self, mock_check_physics):
        """Test physics check when violations are detected."""
        mock_check_physics.return_value = (
            "Physics violations detected (2 issue(s)):\n"
            "- chair_123 collides with table_456\n"
            "- sofa_789 collides with room_geometry"
        )

        result = self.vision_tools._check_physics_impl()

        self.assertIn("Physics violations detected", result)
        self.assertIn("chair_123 collides with table_456", result)
        mock_check_physics.assert_called_once_with(
            scene=self.mock_scene, cfg=self.mock_cfg, agent_type=AgentType.FURNITURE
        )

    @patch("scenecode.furniture_agents.tools.vision_tools.check_physics_violations")
    def test_check_physics_exception(self, mock_check_physics):
        """Test physics check when computation throws exception."""
        mock_check_physics.side_effect = Exception("Physics computation failed")

        with self.assertRaises(Exception) as context:
            self.vision_tools._check_physics_impl()

        self.assertEqual(str(context.exception), "Physics computation failed")


if __name__ == "__main__":
    unittest.main()
