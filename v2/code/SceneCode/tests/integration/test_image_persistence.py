"""Integration tests for image persistence in agent sessions.

These tests verify that images returned from observe_scene persist in the
session and are visible to the agent across multiple API calls.
"""

import asyncio
import logging
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from agents import Agent, Runner, SQLiteSession
from omegaconf import OmegaConf
from pydrake.all import RigidTransform

from scenecode.agent_utils.blender import BlenderServer
from scenecode.agent_utils.house import RoomGeometry
from scenecode.agent_utils.rendering_manager import RenderingManager
from scenecode.agent_utils.room import ObjectType, RoomScene, SceneObject
from scenecode.furniture_agents.tools.vision_tools import VisionTools
from scenecode.utils.logging import ConsoleLogger
from tests.integration.common import has_openai_key

console_logger = logging.getLogger(__name__)


class TestImagePersistence(unittest.TestCase):
    """Test that images from observe_scene persist in session."""

    def _create_realistic_scene(self, output_dir: Path) -> RoomScene:
        """Create a realistic scene with furniture (like test_manipuland_placement)."""
        # Use existing test data.
        test_data_dir = Path(__file__).parent.parent / "test_data" / "realistic_scene"
        floor_plan_path = test_data_dir / "room_geometry.sdf"

        floor_plan_tree = ET.parse(floor_plan_path)
        wall_normals = {
            "left_wall": np.array([1.0, 0.0]),
            "right_wall": np.array([-1.0, 0.0]),
            "back_wall": np.array([0.0, 1.0]),
            "front_wall": np.array([0.0, -1.0]),
        }

        room_geometry = RoomGeometry(
            sdf_tree=floor_plan_tree,
            sdf_path=floor_plan_path,
            wall_normals=wall_normals,
        )

        scene = RoomScene(room_geometry=room_geometry, scene_dir=output_dir)

        # Add a desk to make the scene interesting.
        desk_sdf = (
            test_data_dir / "generated_assets/sdf/work_desk_1761578426/work_desk.sdf"
        )
        desk_gltf = (
            test_data_dir / "generated_assets/sdf/work_desk_1761578426/work_desk.gltf"
        )

        if desk_sdf.exists() and desk_gltf.exists():
            desk_obj = SceneObject(
                object_id=scene.generate_unique_id("work_desk"),
                object_type=ObjectType.FURNITURE,
                name="work_desk",
                description="Test desk",
                transform=RigidTransform(),
                sdf_path=desk_sdf,
                geometry_path=desk_gltf,
                bbox_min=np.array([-0.70, -0.365, 0.0]),
                bbox_max=np.array([0.70, 0.365, 0.761]),
            )
            scene.add_object(desk_obj)

        return scene

    def _count_images_in_session_items(self, items: list) -> int:
        """Count images stored in session items."""
        image_count = 0
        for item in items:
            if isinstance(item, dict):
                # Check function_call_output for images.
                if item.get("type") == "function_call_output":
                    output = item.get("output")
                    if isinstance(output, list):
                        for part in output:
                            if (
                                isinstance(part, dict)
                                and part.get("type") == "input_image"
                            ):
                                image_count += 1
                    # Also check string output that might contain image references.
                    if isinstance(output, str) and "data:image" in output:
                        image_count += 1
        return image_count

    @unittest.skipIf(not has_openai_key(), "Requires OPENAI_API_KEY")
    def test_observe_scene_images_persist_in_session(self):
        """Verify that observe_scene images are stored in session."""
        temp_dir = Path(tempfile.mkdtemp())
        output_dir = temp_dir / "image_persistence_test"
        output_dir.mkdir(exist_ok=True)

        try:
            # Create scene and tools (following test_manipuland_placement pattern).
            scene = self._create_realistic_scene(output_dir)
            logger = ConsoleLogger(output_dir=output_dir)

            # Load config.
            config_path = (
                Path(__file__).parent.parent.parent
                / "configs/furniture_agent/base_furniture_agent.yaml"
            )
            cfg = OmegaConf.load(config_path)

            # Create rendering manager.
            rendering_manager = RenderingManager(
                cfg=cfg.rendering, logger=logger, subdirectory="furniture"
            )

            # Create and start BlenderServer.
            blender_server = BlenderServer(
                port=8000,
                server_startup_delay=0.1,
                port_cleanup_delay=0.1,
            )
            blender_server.start()

            # Create VisionTools.
            vision_tools = VisionTools(
                scene=scene,
                rendering_manager=rendering_manager,
                cfg=cfg,
                blender_server=blender_server,
            )

            # Create agent with observe_scene tool.
            agent = Agent(
                name="test_observer",
                instructions="Call observe_scene to see the room.",
                model="gpt-4o-mini",
                tools=[vision_tools.tools["observe_scene"]],
            )

            # Create session.
            session = SQLiteSession(
                session_id="test_image_persistence",
                db_path=temp_dir / "test.db",
            )

            # Run agent to call observe_scene.
            console_logger.info("Running agent to call observe_scene...")
            asyncio.run(
                Runner.run(
                    starting_agent=agent,
                    input="Please call observe_scene.",
                    session=session,
                )
            )

            # Check if images are stored in session.
            items = asyncio.run(session.get_items())
            image_count = self._count_images_in_session_items(items)

            console_logger.info(f"Found {image_count} images in session items")
            console_logger.info(f"Session items: {len(items)}")

            self.assertGreater(
                image_count,
                0,
                "No images found in session items. observe_scene should persist "
                "images via ToolOutputImage.",
            )

        finally:
            blender_server.stop()
            shutil.rmtree(temp_dir, ignore_errors=True)

    @unittest.skipIf(not has_openai_key(), "Requires OPENAI_API_KEY")
    def test_multiple_observe_scene_calls_accumulate_images(self):
        """Verify calling observe_scene twice persists images from BOTH calls.

        Uses a SINGLE agent/session - call observe_scene, count, call again, count.
        This tests the real scenario: images accumulating in the same session.

        Test strategy (robust, independent of camera count):
        1. Call observe_scene once, count images (N)
        2. Call observe_scene again in same session, count total
        3. Verify: total == 2*N AND N > 0
        """
        temp_dir = Path(tempfile.mkdtemp())
        output_dir = temp_dir / "accumulate_test"
        output_dir.mkdir(exist_ok=True)

        try:
            scene = self._create_realistic_scene(output_dir)
            logger = ConsoleLogger(output_dir=output_dir)

            config_path = (
                Path(__file__).parent.parent.parent
                / "configs/furniture_agent/base_furniture_agent.yaml"
            )
            cfg = OmegaConf.load(config_path)

            rendering_manager = RenderingManager(
                cfg=cfg.rendering, logger=logger, subdirectory="furniture"
            )

            # Create and start BlenderServer.
            blender_server = BlenderServer(
                port=8001,
                server_startup_delay=0.1,
                port_cleanup_delay=0.1,
            )
            blender_server.start()

            vision_tools = VisionTools(
                scene=scene,
                rendering_manager=rendering_manager,
                cfg=cfg,
                blender_server=blender_server,
            )

            agent = Agent(
                name="test_observer",
                instructions="Call observe_scene exactly once per request.",
                model="gpt-4o-mini",
                tools=[vision_tools.tools["observe_scene"]],
            )

            session = SQLiteSession(
                session_id="test_accumulate",
                db_path=temp_dir / "test.db",
            )

            # First observe_scene call.
            console_logger.info("First observe_scene call...")
            asyncio.run(
                Runner.run(
                    starting_agent=agent,
                    input="Call observe_scene once.",
                    session=session,
                )
            )

            items_after_first = asyncio.run(session.get_items())
            images_after_first = self._count_images_in_session_items(items_after_first)
            console_logger.info(f"Images after 1st call: {images_after_first}")

            # Second observe_scene call (same session).
            console_logger.info("Second observe_scene call...")
            asyncio.run(
                Runner.run(
                    starting_agent=agent,
                    input="Call observe_scene once more.",
                    session=session,
                )
            )

            items_after_second = asyncio.run(session.get_items())
            images_after_second = self._count_images_in_session_items(
                items_after_second
            )
            console_logger.info(f"Images after 2nd call: {images_after_second}")

            # --- Assertions ---
            # 1. First call should produce images.
            self.assertGreater(
                images_after_first,
                0,
                "No images found after first observe_scene call.",
            )

            # 2. Second call should double the images (both calls persist).
            self.assertEqual(
                images_after_second,
                2 * images_after_first,
                f"Expected {2 * images_after_first} images after 2nd call "
                f"(2 x {images_after_first}), but found {images_after_second}. "
                "Images from first call are not persisting.",
            )

        finally:
            blender_server.stop()
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
