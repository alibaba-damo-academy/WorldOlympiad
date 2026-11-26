import json
import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from scenecode.agent_utils.rendering_manager import RenderingManager
from scenecode.agent_utils.room import RoomScene
from tests.unit.mock_utils import create_mock_logger


class TestRenderingManager(unittest.TestCase):
    """Test RenderingManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_logger = create_mock_logger(self.temp_dir)
        self.mock_scene = Mock(spec=RoomScene)
        self.mock_scene.objects = {}
        self.mock_scene.text_description = "Test scene"
        # Mock content_hash method for automatic caching tests.
        self.mock_scene.content_hash.return_value = "default_content_hash"

        # Mock BlenderServer for rendering.
        self.mock_blender_server = Mock()
        self.mock_blender_server.is_running.return_value = True

        # Create test config with layout-based format.
        # Note: This mimics cfg.rendering (what RenderingManager actually receives).
        test_config_dict = {
            "layout": "top_plus_sides",
            "top_view_width": 512,
            "top_view_height": 512,
            "side_view_count": 4,
            "side_view_width": 256,
            "side_view_height": 256,
            "background_color": [1.0, 1.0, 1.0],
            "retry_count": 3,
            "retry_delay": 0.01,
        }
        self.test_config = OmegaConf.create(test_config_dict)

        self.rendering_manager = RenderingManager(
            cfg=self.test_config, logger=self.mock_logger
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_log_images_to_dir(self):
        """Create a mock function for log_images_to_dir that creates directories."""

        def mock_log_images_to_dir(images, dir):
            dir.mkdir(parents=True, exist_ok=True)

        return mock_log_images_to_dir

    def test_initialization(self):
        """Test that RenderingManager initializes properly."""
        self.assertIsNotNone(self.rendering_manager)
        self.assertEqual(self.rendering_manager.cfg, self.test_config)
        self.assertEqual(self.rendering_manager.logger, self.mock_logger)
        self.assertEqual(self.rendering_manager._render_counter, 0)
        self.assertEqual(self.rendering_manager._render_cache, {})

    @patch(
        "scenecode.agent_utils.rendering_manager.render_scene_for_agent_observation"
    )
    def test_render_scene_success(self, mock_render_function):
        """Test successful scene rendering."""
        # Mock successful rendering - returns list of paths to rendered images.
        mock_image_paths = [
            Path(self.temp_dir / "top_view.png"),
            Path(self.temp_dir / "side_view_0.png"),
        ]
        # Create the mock files so shutil.copy doesn't fail.
        for path in mock_image_paths:
            path.touch()
        mock_render_function.return_value = mock_image_paths

        # Mock logger to create directories when log_images_to_dir is called.
        self.mock_logger.log_images_to_dir.side_effect = (
            self._create_mock_log_images_to_dir()
        )

        # Test rendering.
        result = self.rendering_manager.render_scene(
            self.mock_scene, blender_server=self.mock_blender_server
        )

        # Verify result.
        self.assertIsInstance(result, Path)
        self.assertTrue(str(result).endswith("renders_001"))

        # Verify dependencies were called.
        mock_render_function.assert_called_once_with(
            scene=self.mock_scene,
            cfg=self.test_config,
            blender_server=self.mock_blender_server,
            include_objects=None,
            exclude_room_geometry=False,
            rendering_mode="furniture",
            support_surfaces=None,
            show_support_surface=False,
            articulated_open=False,
            wall_surfaces=None,
            annotate_object_types=None,
            wall_surfaces_for_labels=None,
            wall_furniture_map=None,
            room_bounds=None,
            ceiling_height=None,
            context_furniture_ids=None,
            side_view_elevation_degrees=None,
            side_view_start_azimuth_degrees=None,
            include_vertical_views=True,
            override_side_view_count=None,
        )

        # Verify render counter incremented.
        self.assertEqual(self.rendering_manager._render_counter, 1)

    @patch(
        "scenecode.agent_utils.rendering_manager.render_scene_for_agent_observation"
    )
    def test_render_scene_with_content_based_caching(self, mock_render_function):
        """Test that render caching works correctly with content-based keys."""
        # Mock successful rendering - returns list of paths.
        mock_image_paths = [Path(self.temp_dir / f"view_{i}.png") for i in range(2)]
        for path in mock_image_paths:
            path.touch()
        mock_render_function.return_value = mock_image_paths

        # Mock logger to create directories.
        self.mock_logger.log_images_to_dir.side_effect = (
            self._create_mock_log_images_to_dir()
        )

        # Mock scene.content_hash() to return consistent value.
        self.mock_scene.content_hash.return_value = "test_content_hash_123"

        # First render - should call rendering functions.
        result1 = self.rendering_manager.render_scene(
            self.mock_scene, blender_server=self.mock_blender_server
        )

        # Second render with same scene content - should use cache.
        result2 = self.rendering_manager.render_scene(
            self.mock_scene, blender_server=self.mock_blender_server
        )

        # Verify both results are the same.
        self.assertEqual(result1, result2)

        # Verify rendering function was called only once.
        mock_render_function.assert_called_once()

        # Verify cache contains the content-based key.
        expected_cache_key = "scene_content_test_content_hash_123"
        self.assertIn(expected_cache_key, self.rendering_manager._render_cache)
        self.assertEqual(
            self.rendering_manager._render_cache[expected_cache_key], result1
        )

    @patch("scenecode.agent_utils.rendering_manager.time.sleep")
    @patch(
        "scenecode.agent_utils.rendering_manager.render_scene_for_agent_observation"
    )
    def test_render_scene_with_retry_logic(self, mock_render_function, mock_sleep):
        """Test retry logic on rendering failures."""
        # Mock render function to fail twice, then succeed.
        mock_image_paths = [Path(self.temp_dir / f"view_{i}.png") for i in range(2)]
        for path in mock_image_paths:
            path.touch()

        mock_render_function.side_effect = [
            RuntimeError("First failure"),
            RuntimeError("Second failure"),
            mock_image_paths,  # Third attempt succeeds
        ]

        # Mock logger to create directories.
        self.mock_logger.log_images_to_dir.side_effect = (
            self._create_mock_log_images_to_dir()
        )

        # Test rendering.
        result = self.rendering_manager.render_scene(
            self.mock_scene, blender_server=self.mock_blender_server
        )

        # Verify result is returned after retries.
        self.assertIsInstance(result, Path)

        # Verify render function was called 3 times.
        self.assertEqual(mock_render_function.call_count, 3)

        # Verify sleep was called twice (after first two failures).
        self.assertEqual(mock_sleep.call_count, 2)

    @patch(
        "scenecode.agent_utils.rendering_manager.render_scene_for_agent_observation"
    )
    def test_render_scene_all_attempts_fail(self, mock_render_function):
        """Test that RuntimeError is raised when all attempts fail."""
        # Mock render function to always fail.
        mock_render_function.side_effect = RuntimeError("Persistent failure")

        # Test that error is raised.
        with self.assertRaises(RuntimeError) as context:
            self.rendering_manager.render_scene(
                self.mock_scene, blender_server=self.mock_blender_server
            )

        # Verify error message.
        self.assertIn("Scene rendering failed after 3 attempts", str(context.exception))

        # Verify render function was called 3 times.
        self.assertEqual(mock_render_function.call_count, 3)

    @patch(
        "scenecode.agent_utils.rendering_manager.render_scene_for_agent_observation"
    )
    def test_render_scene_empty_image_grid_failure(self, mock_render_function):
        """Test handling of empty image path list."""
        # Mock render function to return empty list of paths.
        mock_render_function.return_value = []  # Empty list

        # Test that error is raised for empty path list.
        with self.assertRaises(RuntimeError) as context:
            self.rendering_manager.render_scene(
                self.mock_scene, blender_server=self.mock_blender_server
            )

        # Verify error message.
        self.assertIn("No images", str(context.exception))

    @patch(
        "scenecode.agent_utils.rendering_manager.render_scene_for_agent_observation"
    )
    def test_scene_checkpoint_saving(self, mock_render_function):
        """Test that scene checkpoints are saved correctly via logger."""
        # Mock successful rendering - returns list of paths.
        mock_image_paths = [Path(self.temp_dir / f"view_{i}.png") for i in range(2)]
        for path in mock_image_paths:
            path.touch()
        mock_render_function.return_value = mock_image_paths

        # Mock scene.to_state_dict to return actual dictionary.
        mock_state_dict = {
            "scene_items": ["Test Object"],
            "object_positions": {
                "test_obj_123": {
                    "name": "Test Object",
                    "description": "A test object",
                    "position": [1.0, 2.0, 3.0],
                    "rotation": [0, 0, 0, 1],
                }
            },
            "text_description": "",
        }
        self.mock_scene.to_state_dict.return_value = mock_state_dict

        # Mock logger to create directories and handle log_scene call.
        self.mock_logger.log_images_to_dir.side_effect = (
            self._create_mock_log_images_to_dir()
        )

        # Mock log_scene to create checkpoint file.
        def mock_log_scene(scene, name=None, output_dir=None):
            if output_dir:
                # Create the checkpoint file that would be created by logger.
                checkpoint_file = output_dir / "scene_state.json"
                checkpoint_data = scene.to_state_dict()
                checkpoint_data["timestamp"] = 1234567890.0
                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint_data, f, indent=2)
                return output_dir
            return self.mock_logger.output_dir / "scene_states" / name

        self.mock_logger.log_scene.side_effect = mock_log_scene

        # Test rendering.
        result = self.rendering_manager.render_scene(
            self.mock_scene, blender_server=self.mock_blender_server
        )

        # Verify logger.log_scene was called with output_dir.
        self.mock_logger.log_scene.assert_called_once_with(
            scene=self.mock_scene, output_dir=result
        )

        # Verify checkpoint file was created.
        checkpoint_file = result / "scene_state.json"
        self.assertTrue(checkpoint_file.exists())

    def test_render_scene_with_different_content_hashes(self):
        """Test rendering with different content hashes doesn't use caching."""
        with patch(
            "scenecode.agent_utils.rendering_manager.render_scene_for_agent_observation"
        ) as mock_render:
            # Mock successful rendering - returns list of paths.
            mock_image_paths = [Path(self.temp_dir / f"view_{i}.png") for i in range(2)]
            for path in mock_image_paths:
                path.touch()
            mock_render.return_value = mock_image_paths

            # Mock logger to create directories.
            self.mock_logger.log_images_to_dir.side_effect = (
                self._create_mock_log_images_to_dir()
            )

            # Mock scene.content_hash() to return different values.
            self.mock_scene.content_hash.side_effect = ["hash1", "hash2"]

            # Render twice with different content hashes.
            _ = self.rendering_manager.render_scene(
                self.mock_scene, blender_server=self.mock_blender_server
            )
            _ = self.rendering_manager.render_scene(
                self.mock_scene, blender_server=self.mock_blender_server
            )

            # Verify render function was called twice (no caching due to different
            # hashes).
            self.assertEqual(mock_render.call_count, 2)

            # Verify cache contains both entries.
            self.assertEqual(len(self.rendering_manager._render_cache), 2)
            self.assertIn("scene_content_hash1", self.rendering_manager._render_cache)
            self.assertIn("scene_content_hash2", self.rendering_manager._render_cache)


if __name__ == "__main__":
    unittest.main()
