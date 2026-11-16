import json
import shutil
import tempfile
import unittest

from pathlib import Path

import lxml.etree as ET
import numpy as np

from pydrake.all import RigidTransform

from scenecode.agent_utils.house import RoomGeometry
from scenecode.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scenecode.utils.logging import ConsoleLogger


class TestConsoleLogger(unittest.TestCase):
    """Test ConsoleLogger functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_data = {"metric1": 1.0, "metric2": 2.0}
        self.test_hyperparams = {"learning_rate": 0.001, "batch_size": 32}
        self.test_obj = {"test": "data"}

        # Create a simple SDF tree for testing.
        self.sdf_root = ET.Element("sdf", version="1.7")
        world = ET.SubElement(self.sdf_root, "world", name="test_world")
        ET.SubElement(world, "light", name="test_light", type="directional")
        self.sdf_tree = ET.ElementTree(self.sdf_root)

        # Create a test scene.
        self.test_scene = RoomScene(
            room_geometry=RoomGeometry(sdf_tree=self.sdf_tree, sdf_path=None),
            text_description="Test scene for logging",
            scene_dir=self.temp_dir,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temp files and directories.
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_console_logger_api_compatibility(self):
        """Test that ConsoleLogger implements the BaseLogger API correctly."""
        logger = ConsoleLogger(output_dir=self.temp_dir)

        # Test all abstract methods are implemented and callable.
        self.assertTrue(hasattr(logger, "log"))
        self.assertTrue(callable(logger.log))

        self.assertTrue(hasattr(logger, "log_hyperparams"))
        self.assertTrue(callable(logger.log_hyperparams))

        self.assertTrue(hasattr(logger, "log_pickle"))
        self.assertTrue(callable(logger.log_pickle))

        self.assertTrue(hasattr(logger, "log_sdf"))
        self.assertTrue(callable(logger.log_sdf))

        self.assertTrue(hasattr(logger, "log_images_to_dir"))
        self.assertTrue(callable(logger.log_images_to_dir))

        self.assertTrue(hasattr(logger, "log_scene"))
        self.assertTrue(callable(logger.log_scene))

        # Test basic functionality without errors.
        logger.log(data=self.test_data)
        logger.log_hyperparams(data=self.test_hyperparams)
        logger.log_pickle(name="test.pkl", obj=self.test_obj, use_temp_file=False)
        sdf_path = logger.log_sdf(name="test.sdf", sdf_tree=self.sdf_tree)

        # Test log_images_to_dir.
        test_images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(2)
        ]
        images_dir = self.temp_dir / "test_images"
        logger.log_images_to_dir(images=test_images, dir=images_dir)

        # Test log_scene.
        scene_dir = logger.log_scene(scene=self.test_scene, name="test_scene")
        self.assertIsInstance(scene_dir, Path)

        # Verify files were created.
        self.assertTrue((self.temp_dir / "test.pkl").exists())
        self.assertTrue((self.temp_dir / "test.sdf").exists())
        self.assertTrue(images_dir.exists())
        image_files = list(images_dir.glob("*.png"))
        self.assertEqual(len(image_files), 2)
        self.assertIsInstance(sdf_path, Path)

    def test_console_logger_creates_output_dir(self):
        """Test that ConsoleLogger creates output directory if it doesn't exist."""
        non_existent_dir = self.temp_dir / "non_existent"
        self.assertFalse(non_existent_dir.exists())

        _logger = ConsoleLogger(output_dir=non_existent_dir)
        self.assertTrue(non_existent_dir.exists())

    def test_console_logger_scene_logging(self):
        """Test scene logging functionality."""
        logger = ConsoleLogger(output_dir=self.temp_dir)

        # Add some objects to the scene.
        test_object = SceneObject(
            object_id=UniqueID("test"),
            object_type=ObjectType.FURNITURE,
            name="test_asset",
            description="test_asset",
            transform=RigidTransform(),
        )
        self.test_scene.add_object(test_object)

        # Log the scene.
        scene_dir = logger.log_scene(
            scene=self.test_scene, name="test_scene_with_objects"
        )

        # Verify scene state and directive files were created.
        self.assertTrue((scene_dir / "scene_state.json").exists())
        self.assertTrue((scene_dir / "scene.dmd.yaml").exists())

        # Verify scene state content.
        with open(scene_dir / "scene_state.json", "r") as f:
            scene_state = json.load(f)

        self.assertIn("objects", scene_state)
        self.assertIn("timestamp", scene_state)
        self.assertEqual(len(scene_state["objects"]), 1)


if __name__ == "__main__":
    unittest.main()
