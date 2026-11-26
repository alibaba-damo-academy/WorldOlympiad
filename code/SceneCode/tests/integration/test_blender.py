import logging
import time
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from omegaconf import OmegaConf
from PIL import Image as PILImage
from pydrake.all import RigidTransform, RollPitchYaw

from scenecode.agent_utils.blender import BlenderServer
from scenecode.agent_utils.house import RoomGeometry
from scenecode.agent_utils.rendering import (
    render_scene,
    render_scene_for_agent_observation,
)
from scenecode.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID


class TestBlenderIntegration(unittest.TestCase):
    """Integration test for the complete Blender rendering pipeline.

    Tests the entire flow from RoomScene -> Drake Plant -> Blender Server -> PNG.
    Uses bpy singleton pattern with a single BlenderServer instance.
    """

    def test_blender_integration_pipeline(self):
        """Test complete Blender integration: server lifecycle + rendering pipeline."""
        # Set up test data paths.
        test_data_dir = Path(__file__).parent.parent / "test_data"
        floor_plan_path = test_data_dir / "simple_room_geometry.sdf"
        box_sdf_path = test_data_dir / "simple_box.sdf"
        sphere_sdf_path = test_data_dir / "simple_sphere.sdf"

        # Verify test data exists.
        for path in [floor_plan_path, box_sdf_path, sphere_sdf_path]:
            if not path.exists():
                self.fail(f"Test data file not found: {path}")

        # Start the server.
        server = BlenderServer(
            port_range=(8000, 8050),
            server_startup_delay=0.1,
            port_cleanup_delay=0.1,
        )
        server.start()

        try:
            # Wait for server to be ready.
            try:
                server.wait_until_ready()
            except RuntimeError as e:
                self.fail(f"BlenderServer failed to start: {e}")

            # Verify server is running.
            self.assertTrue(server.is_running())

            # Create test scene.
            floor_plan_tree = ET.parse(floor_plan_path)
            room_geometry = RoomGeometry(
                sdf_tree=floor_plan_tree,
                sdf_path=floor_plan_path,
            )
            scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

            # Add furniture object.
            furniture_obj = SceneObject(
                object_id=UniqueID("test_furniture"),
                object_type=ObjectType.FURNITURE,
                name="Test Furniture",
                description="A furniture box for testing",
                transform=RigidTransform(np.array([1.0, 0.0, 0.5])),
                sdf_path=box_sdf_path,
            )
            scene.add_object(furniture_obj)

            # Add manipuland object.
            manipuland_obj = SceneObject(
                object_id=UniqueID("test_manipuland"),
                object_type=ObjectType.MANIPULAND,
                name="Test Manipuland",
                description="A manipuland sphere for testing",
                transform=RigidTransform(np.array([0.0, 1.0, 1.0])),
                sdf_path=sphere_sdf_path,
            )
            scene.add_object(manipuland_obj)

            # Set camera pose.
            camera_pose = RigidTransform(
                RollPitchYaw(1.57, -0.2, 3.14), np.array([3.0, 0.0, 2.0])
            )

            # Render scene using Blender server (small image for speed).
            try:
                image = render_scene(
                    scene=scene,
                    camera_X_WC=camera_pose,
                    camera_width=20,
                    camera_height=30,
                    use_blender_server=True,
                    blender_server_url=server.get_url(),
                )
            except Exception as e:
                self.fail(f"Rendering failed: {e}")

            # Verify basic image properties.
            self.assertIsInstance(image, np.ndarray)
            self.assertEqual(image.shape, (30, 20, 4))  # Height, Width, RGBA
            self.assertEqual(image.dtype, np.uint8)

        finally:
            # Clean up server.
            if server.is_running():
                server.stop()
                # Give processes time to fully terminate.
                time.sleep(2)

    def test_flexible_rendering_integration(self):
        """Test flexible rendering integration with config-driven layouts."""
        # Set up test data paths.
        test_data_dir = Path(__file__).parent.parent / "test_data"
        floor_plan_path = test_data_dir / "simple_room_geometry.sdf"
        box_sdf_path = test_data_dir / "simple_box.sdf"

        # Verify test data exists.
        for path in [floor_plan_path, box_sdf_path]:
            if not path.exists():
                self.fail(f"Test data file not found: {path}")

        # Create test scene.
        floor_plan_tree = ET.parse(floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=floor_plan_tree,
            sdf_path=floor_plan_path,
        )
        scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

        # Add furniture object.
        furniture_obj = SceneObject(
            object_id=UniqueID("test_furniture"),
            object_type=ObjectType.FURNITURE,
            name="Test Furniture",
            description="Regular furniture object",
            transform=RigidTransform(np.array([1.0, 0.0, 0.5])),
            sdf_path=box_sdf_path,
        )
        scene.add_object(furniture_obj)

        # Load base rendering config and override for fast testing.
        config_path = (
            Path(__file__).parent.parent.parent
            / "configs/furniture_agent/base_furniture_agent.yaml"
        )
        base_config = OmegaConf.load(config_path)
        test_overrides = OmegaConf.create(
            {
                "top_view_width": 80,
                "top_view_height": 80,
                "side_view_count": 4,
                "side_view_width": 60,
                "side_view_height": 60,
                "server_startup_delay": 0.1,
                "port_cleanup_delay": 0.1,
            }
        )
        rendering_cfg = OmegaConf.merge(base_config.rendering, test_overrides)

        # Create BlenderServer for rendering.
        server = BlenderServer(
            port_range=(8001, 8010),
            server_startup_delay=0.1,
            port_cleanup_delay=0.1,
        )

        # Render scene with flexible rendering.
        try:
            server.start()
            server.wait_until_ready()

            image_paths = render_scene_for_agent_observation(
                scene=scene, cfg=rendering_cfg, blender_server=server
            )

            # Verify rendering produces correct number of views (1 top + 4 side).
            self.assertEqual(len(image_paths), 5)

            # Verify all image files exist and have content.
            for img_path in image_paths:
                self.assertTrue(img_path.exists(), f"Image not found: {img_path}")
                self.assertGreater(
                    img_path.stat().st_size, 0, f"Empty image file: {img_path}"
                )

                # Load and verify image has expected dimensions.
                img = PILImage.open(img_path)
                img_array = np.array(img)

                # Check if it's top or side view based on filename.
                if "top" in img_path.name:
                    expected_height, expected_width = 80, 80
                else:
                    expected_height, expected_width = 60, 60

                # Verify spatial dimensions (accept both RGB and RGBA).
                self.assertEqual(
                    img_array.shape[:2],
                    (expected_height, expected_width),
                    f"Unexpected image dimensions for {img_path.name}",
                )
                # Verify color channels (3 for RGB or 4 for RGBA).
                self.assertIn(
                    img_array.shape[2],
                    [3, 4],
                    f"Unexpected channel count for {img_path.name}",
                )

                # Verify image contains non-uniform content.
                channels = img_array.shape[2]
                unique_pixels = len(np.unique(img_array.reshape(-1, channels), axis=0))
                self.assertGreater(
                    unique_pixels,
                    5,
                    f"Image {img_path.name} should contain varied pixel content",
                )

        except Exception as e:
            self.fail(f"Flexible rendering failed: {e}")
        finally:
            # Clean up server.
            if server.is_running():
                server.stop()
                time.sleep(1)

    def test_concurrent_blender_servers(self):
        """Test that multiple BlenderServer instances can run concurrently."""
        # This test uses port ranges to avoid conflicts.
        server1 = BlenderServer(
            port_range=(8100, 8105),
            server_startup_delay=0.1,
            port_cleanup_delay=0.1,
        )
        server2 = BlenderServer(
            port_range=(8100, 8105),
            server_startup_delay=0.1,
            port_cleanup_delay=0.1,
        )

        try:
            # Start both servers - they should find different available ports.
            server1.start()
            server1.wait_until_ready()

            server2.start()
            server2.wait_until_ready()

            # Verify both servers are running.
            self.assertTrue(server1.is_running())
            self.assertTrue(server2.is_running())

            # Verify they are using different ports.
            url1 = server1.get_url()
            url2 = server2.get_url()
            self.assertNotEqual(url1, url2)

            # Extract ports from URLs and verify they're in the expected range.
            port1 = int(url1.split(":")[-1])
            port2 = int(url2.split(":")[-1])

            self.assertGreaterEqual(port1, 8100)
            self.assertLessEqual(port1, 8105)
            self.assertGreaterEqual(port2, 8100)
            self.assertLessEqual(port2, 8105)
            self.assertNotEqual(port1, port2)

        finally:
            # Clean up servers.
            for server in [server1, server2]:
                if server.is_running():
                    server.stop()
                    # Give processes time to fully terminate.
                    time.sleep(1)

    def test_single_top_layout(self):
        """Test single top view layout for ablations."""
        # Set up test data paths.
        test_data_dir = Path(__file__).parent.parent / "test_data"
        floor_plan_path = test_data_dir / "simple_room_geometry.sdf"
        box_sdf_path = test_data_dir / "simple_box.sdf"

        # Verify test data exists.
        for path in [floor_plan_path, box_sdf_path]:
            if not path.exists():
                self.fail(f"Test data file not found: {path}")

        # Create test scene.
        floor_plan_tree = ET.parse(floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=floor_plan_tree,
            sdf_path=floor_plan_path,
        )
        scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

        # Add furniture object.
        furniture_obj = SceneObject(
            object_id=UniqueID("test_furniture"),
            object_type=ObjectType.FURNITURE,
            name="Test Furniture",
            description="Furniture for layout test",
            transform=RigidTransform(np.array([1.0, 0.0, 0.5])),
            sdf_path=box_sdf_path,
        )
        scene.add_object(furniture_obj)

        # Load base rendering config and override for single_top layout.
        config_path = (
            Path(__file__).parent.parent.parent
            / "configs/furniture_agent/base_furniture_agent.yaml"
        )
        base_config = OmegaConf.load(config_path)
        test_overrides = OmegaConf.create(
            {
                "layout": "single_top",
                "top_view_width": 100,
                "top_view_height": 100,
                "side_view_count": 0,
                "server_startup_delay": 0.1,
                "port_cleanup_delay": 0.1,
            }
        )
        rendering_cfg = OmegaConf.merge(base_config.rendering, test_overrides)

        # Create BlenderServer for rendering.
        server = BlenderServer(
            port_range=(8002, 8007), server_startup_delay=0.1, port_cleanup_delay=0.1
        )

        # Render with BlenderServer.
        try:
            server.start()
            server.wait_until_ready()

            image_paths = render_scene_for_agent_observation(
                scene=scene, cfg=rendering_cfg, blender_server=server
            )

            # Verify rendering produces only top view.
            self.assertEqual(len(image_paths), 1)

            # Verify image dimensions.
            img = PILImage.open(image_paths[0])
            img_array = np.array(img)

            # Verify spatial dimensions (accept both RGB and RGBA).
            self.assertEqual(img_array.shape[:2], (100, 100))
            self.assertIn(img_array.shape[2], [3, 4], "Expected RGB or RGBA image")

            # Verify the image contains non-uniform content.
            channels = img_array.shape[2]
            unique_pixel_values = len(
                np.unique(img_array.reshape(-1, channels), axis=0)
            )
            self.assertGreater(
                unique_pixel_values,
                5,
                "Single top view should produce varied pixel content",
            )

        except Exception as e:
            self.fail(f"Single top layout rendering failed: {e}")
        finally:
            # Clean up server.
            if server.is_running():
                server.stop()
                time.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
