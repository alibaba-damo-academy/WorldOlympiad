"""
Integration test for wall snapping position and orientation.
Also renders the scene before and after the snap operation to
tests/integration/test_outputs.
"""

import json
import logging
import shutil
import time
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from omegaconf import DictConfig, OmegaConf
from pydrake.all import RigidTransform, RollPitchYaw

from scenecode.agent_utils.blender.server_manager import BlenderServer
from scenecode.agent_utils.house import RoomGeometry
from scenecode.agent_utils.physics_validation import compute_scene_collisions
from scenecode.agent_utils.rendering import render_scene_for_agent_observation
from scenecode.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scenecode.furniture_agents.tools.scene_tools import SceneTools

console_logger = logging.getLogger(__name__)


class TestWallSnapOrientation(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Start Blender server for rendering.
        self.blender_server = BlenderServer(
            host="127.0.0.1",
            port_range=(8010, 8020),
        )
        self.blender_server.start()

    def tearDown(self):
        """Clean up test fixtures."""
        if self.blender_server is not None and self.blender_server.is_running():
            self.blender_server.stop()

    def _render_and_save(
        self,
        scene: RoomScene,
        rendering_cfg: DictConfig,
        output_dir: Path,
        prefix: str,
    ) -> None:
        """Render scene and save images with given prefix."""
        images = render_scene_for_agent_observation(
            scene=scene,
            cfg=rendering_cfg,
            blender_server=self.blender_server,
        )
        for img_path in images:
            dest = output_dir / f"{prefix}_{img_path.name}"
            shutil.copy(img_path, dest)

    def _check_collisions(
        self, scene: RoomScene, context: str, expected_count: int = 0
    ) -> list:
        """Check collisions and verify expected count.

        Args:
            scene: RoomScene to check for collisions.
            context: Description of when check is happening (for logging).
            expected_count: Expected number of collisions (default 0).

        Returns:
            List of collision objects found.
        """
        console_logger.info(f"Checking collisions {context}...")
        collisions = compute_scene_collisions(scene=scene)
        console_logger.info(f"Found {len(collisions)} collision(s) {context}")

        for collision in collisions:
            console_logger.warning(f"  - {collision.to_description()}")

        self.assertEqual(
            len(collisions),
            expected_count,
            msg=(
                f"Expected {expected_count} collisions {context}, "
                f"found {len(collisions)}: "
                + ", ".join(c.to_description() for c in collisions)
            ),
        )
        return collisions

    def _execute_snap(
        self,
        scene_tools: SceneTools,
        object_id: str,
        target_id: str,
        orientation: str,
        context: str = "",
    ) -> dict:
        """Execute snap and verify success.

        Args:
            scene_tools: SceneTools instance to use.
            object_id: ID of object to snap.
            target_id: ID of target to snap to.
            orientation: Orientation mode ("toward", "away", "none").
            context: Description for logging (e.g., "desk to wall").

        Returns:
            Parsed result dictionary.
        """
        console_logger.info(f"Executing snap {context}...")
        result_json = scene_tools._snap_to_object_impl(
            object_id=object_id,
            target_id=target_id,
            orientation=orientation,
        )

        result = json.loads(result_json)
        console_logger.info(f"Snap result {context}: success={result.get('success')}")

        if not result.get("success"):
            console_logger.error(f"Snap failed {context}: {result.get('message')}")

        self.assertTrue(
            result["success"],
            msg=f"Snap operation {context} should succeed: {result.get('message')}",
        )
        return result

    def test_snap_to_wall_with_orientation(self):
        # Set up test data paths.
        test_data_dir = Path(__file__).parent.parent / "test_data" / "realistic_scene"
        floor_plan_path = test_data_dir / "room_geometry.sdf"

        # Verify test data exists.
        if not floor_plan_path.exists():
            self.fail(f"Test data file not found: {floor_plan_path}")

        # Create 5x5m room centered at origin.
        # back_wall at y=-2.5, normal pointing +Y into room.
        floor_plan_tree = ET.parse(floor_plan_path)
        wall_normals = {
            "left_wall": np.array([1.0, 0.0]),  # Points right.
            "right_wall": np.array([-1.0, 0.0]),  # Points left.
            "back_wall": np.array([0.0, 1.0]),  # Points forward (+Y).
            "front_wall": np.array([0.0, -1.0]),  # Points backward.
        }

        room_geometry = RoomGeometry(
            sdf_tree=floor_plan_tree,
            sdf_path=floor_plan_path,
            wall_normals=wall_normals,
        )
        scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

        # Manually create wall objects for the 5x5m room.
        # Room extends from -2.5 to +2.5 in both X and Y.
        # Wall dimensions: 0.1m thick, 2m tall.
        back_wall = SceneObject(
            object_id=UniqueID("back_wall"),
            object_type=ObjectType.WALL,
            name="back_wall",
            description="Back wall",
            transform=RigidTransform(p=[0.0, -2.5, 1.0]),
            geometry_path=None,
            bbox_min=np.array([-2.5, -0.05, 0.0]),
            bbox_max=np.array([2.5, 0.05, 2.0]),
            immutable=True,
        )
        scene.add_object(back_wall)

        # Add left wall for testing collision resolution.
        left_wall = SceneObject(
            object_id=UniqueID("left_wall"),
            object_type=ObjectType.WALL,
            name="left_wall",
            description="Left wall",
            transform=RigidTransform(p=[-2.5, 0.0, 1.0]),
            geometry_path=None,
            bbox_min=np.array([-0.05, -2.5, 0.0]),
            bbox_max=np.array([0.05, 2.5, 2.0]),
            immutable=True,
        )
        scene.add_object(left_wall)

        # Load work desk assets.
        desk_sdf = (
            test_data_dir / "generated_assets/sdf/work_desk_1761578426/work_desk.sdf"
        )
        desk_gltf = (
            test_data_dir / "generated_assets/sdf/work_desk_1761578426/work_desk.gltf"
        )
        if not desk_sdf.exists():
            self.fail(f"Asset not found: {desk_sdf}")
        if not desk_gltf.exists():
            self.fail(f"Asset not found: {desk_gltf}")

        # Place desk off-center at X=0.63 with incorrect yaw=90° (facing +X, sideways).
        # snap_to_object with orientation="away" should rotate it to yaw=0° (facing +Y).
        initial_yaw_deg = 90.0
        initial_transform = RigidTransform(
            RollPitchYaw(0, 0, np.deg2rad(initial_yaw_deg)),
            np.array([0.63, 1.0, 0.0]),  # Off-center position
        )
        desk_obj = SceneObject(
            object_id=scene.generate_unique_id("work_desk"),
            object_type=ObjectType.FURNITURE,
            name="work_desk",
            description="Test desk for wall snapping",
            transform=initial_transform,
            sdf_path=desk_sdf,
            geometry_path=desk_gltf,
            bbox_min=np.array([-0.75, -0.40, 0.0]),
            bbox_max=np.array([0.75, 0.40, 0.75]),
        )
        scene.add_object(desk_obj)

        # Load office chair assets.
        chair_sdf = (
            test_data_dir
            / "generated_assets/sdf/office_chair_1761578426/office_chair.sdf"
        )
        chair_gltf = (
            test_data_dir
            / "generated_assets/sdf/office_chair_1761578426/office_chair.gltf"
        )
        if not chair_sdf.exists():
            self.fail(f"Asset not found: {chair_sdf}")
        if not chair_gltf.exists():
            self.fail(f"Asset not found: {chair_gltf}")

        # Place chair at arbitrary position/orientation (will be snapped to desk).
        # Chair bbox: 0.7m × 0.7m × 1.0m (X × Y × Z).
        # snap_to_object with orientation="toward" should make chair face desk.
        # Position at x=2.0 to avoid collision with desk after orientation change.
        # (Desk at x=0.63 with yaw=0° extends to x=1.38, so chair must be > 1.73)
        chair_initial_yaw_deg = 45.0
        chair_initial_transform = RigidTransform(
            RollPitchYaw(0, 0, np.deg2rad(chair_initial_yaw_deg)),
            np.array([1.0, 0.0, 0.0]),  # Far enough to avoid desk bbox after rotation.
        )
        chair_obj = SceneObject(
            object_id=scene.generate_unique_id("office_chair"),
            object_type=ObjectType.FURNITURE,
            name="office_chair",
            description="Test chair for desk snapping",
            transform=chair_initial_transform,
            sdf_path=chair_sdf,
            geometry_path=chair_gltf,
            bbox_min=np.array([-0.35, -0.35, 0.0]),
            bbox_max=np.array([0.35, 0.35, 1.0]),
        )
        scene.add_object(chair_obj)

        # Add second table (dining table) IN COLLISION with left_wall.
        # This tests collision resolution (push out of wall).
        # left_wall inner face at x=-2.45, place table at x=-2.5 to create collision.
        dining_table_transform = RigidTransform(
            RollPitchYaw(0, 0, 0), np.array([-2.5, 0.0, 0.0])
        )
        dining_table_obj = SceneObject(
            object_id=scene.generate_unique_id("dining_table"),
            object_type=ObjectType.FURNITURE,
            name="dining_table",
            description="Second table for collision test",
            transform=dining_table_transform,
            sdf_path=desk_sdf,  # Reuse desk assets.
            geometry_path=desk_gltf,
            bbox_min=np.array([-0.75, -0.40, 0.0]),
            bbox_max=np.array([0.75, 0.40, 0.75]),
        )
        scene.add_object(dining_table_obj)

        # Create rendering config.
        rendering_cfg = OmegaConf.create(
            {
                "layout": "top_plus_sides",
                "top_view_width": 1024,
                "top_view_height": 1024,
                "side_view_count": 4,
                "side_view_width": 512,
                "side_view_height": 512,
                "background_color": [1.0, 1.0, 1.0],
                "server_startup_delay": 0.1,
                "port_cleanup_delay": 0.1,
                "annotations": {
                    "enable_set_of_mark_labels": True,
                    "enable_bounding_boxes": True,
                    "enable_direction_arrows": True,
                    "enable_partial_walls": True,
                    "enable_support_surface_debug": False,
                    "enable_convex_hull_debug": False,
                },
            }
        )

        # Create output directory.
        test_output_dir = Path(__file__).parent / "test_outputs"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        snap_test_dir = test_output_dir / f"wall_snap_test_{timestamp}"
        snap_test_dir.mkdir(parents=True, exist_ok=True)

        # Load base configuration and merge with test settings.
        config_path = (
            Path(__file__).parent.parent.parent
            / "configs/furniture_agent/base_furniture_agent.yaml"
        )
        base_cfg = OmegaConf.load(config_path)
        test_cfg = OmegaConf.merge(base_cfg, {})  # Use base config as-is.

        scene_tools = SceneTools(scene=scene, cfg=test_cfg)

        # Render "before" state (all 3 initial objects before any snaps).
        self._render_and_save(
            scene=scene,
            rendering_cfg=rendering_cfg,
            output_dir=snap_test_dir,
            prefix="before",
        )

        # Snap desk to back wall with orientation="away".
        result = self._execute_snap(
            scene_tools=scene_tools,
            object_id=str(desk_obj.object_id),
            target_id="back_wall",
            orientation="away",
            context="desk to back_wall",
        )

        # Render after desk snap for debugging.
        self._render_and_save(
            scene=scene,
            rendering_cfg=rendering_cfg,
            output_dir=snap_test_dir,
            prefix="after_desk",
        )

        # Snap chair to desk with orientation="toward".
        chair_result = self._execute_snap(
            scene_tools=scene_tools,
            object_id=str(chair_obj.object_id),
            target_id=str(desk_obj.object_id),
            orientation="toward",
            context="chair to desk",
        )

        # Render after chair snap for debugging.
        self._render_and_save(
            scene=scene,
            rendering_cfg=rendering_cfg,
            output_dir=snap_test_dir,
            prefix="after_chair",
        )

        # Snap dining table (in collision with left_wall) to left_wall.
        dining_table_result = self._execute_snap(
            scene_tools=scene_tools,
            object_id=str(dining_table_obj.object_id),
            target_id="left_wall",
            orientation="away",
            context="dining_table to left_wall (with collision)",
        )

        # Get dining_table's final position after snap to left_wall.
        final_dining_table = scene.objects[dining_table_obj.object_id]
        final_dining_table_pos = final_dining_table.transform.translation()
        console_logger.info(
            f"Dining table final position: ({final_dining_table_pos[0]:.3f}, "
            f"{final_dining_table_pos[1]:.3f}, {final_dining_table_pos[2]:.3f})"
        )

        # Check collisions after dining_table snap (should be zero).
        self._check_collisions(
            scene=scene,
            context="after dining_table snap",
            expected_count=0,
        )

        # Place extra_chair overlapping dining_table to test AABB push-out.
        # Offset by (0.2, 0) from table center to create X-axis penetration.
        # Push-out should move chair in X direction (minimum overlap axis).
        extra_chair_pos = final_dining_table_pos + np.array([0.2, 0.0, 0.0])
        extra_chair_transform = RigidTransform(
            RollPitchYaw(0, 0, np.deg2rad(180.0)), extra_chair_pos
        )
        extra_chair_obj = SceneObject(
            object_id=scene.generate_unique_id("extra_chair"),
            object_type=ObjectType.FURNITURE,
            name="extra_chair",
            description="Chair for mesh-to-mesh collision test",
            transform=extra_chair_transform,
            sdf_path=chair_sdf,
            geometry_path=chair_gltf,
            bbox_min=np.array([-0.35, -0.35, 0.0]),
            bbox_max=np.array([0.35, 0.35, 1.0]),
        )
        scene.add_object(extra_chair_obj)
        console_logger.info(
            "Added extra_chair overlapping dining_table at X offset to test AABB push-out"
        )

        # Snap extra_chair to dining_table (test AABB push-out moving chair farther).
        # Track initial position to verify push-out moved chair farther.
        initial_extra_chair_pos = extra_chair_obj.transform.translation()
        initial_distance_to_table = np.linalg.norm(
            initial_extra_chair_pos[:2] - final_dining_table_pos[:2]
        )

        extra_chair_result = self._execute_snap(
            scene_tools=scene_tools,
            object_id=str(extra_chair_obj.object_id),
            target_id=str(dining_table_obj.object_id),
            orientation="toward",
            context="extra_chair to dining_table",
        )

        # Render "after" state.
        self._render_and_save(
            scene=scene,
            rendering_cfg=rendering_cfg,
            output_dir=snap_test_dir,
            prefix="after",
        )

        # Get final desk state.
        final_desk = scene.objects[desk_obj.object_id]
        final_pos = final_desk.transform.translation()
        final_rpy = final_desk.transform.rotation().ToRollPitchYaw()
        final_yaw_deg = np.rad2deg(final_rpy.yaw_angle())

        # Log initial vs final state.
        console_logger.info(f"\nInitial position: (0.63, 1.0, 0.0)")
        console_logger.info(f"Initial yaw: {initial_yaw_deg}°")
        console_logger.info(
            f"Final position: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})"
        )
        console_logger.info(f"Final yaw: {final_yaw_deg:.1f}°")
        console_logger.info(f"Yaw change: {initial_yaw_deg}° → {final_yaw_deg:.1f}°")
        console_logger.info(f"\nRenders saved to: {snap_test_dir.absolute()}")

        # Verify snap succeeded.
        self.assertTrue(result["success"], "Snap operation should succeed")

        # Verify push-out moved chair farther from table.
        final_extra_chair = scene.objects[extra_chair_obj.object_id]
        final_extra_chair_pos = final_extra_chair.transform.translation()
        final_distance_to_table = np.linalg.norm(
            final_extra_chair_pos[:2] - final_dining_table_pos[:2]
        )
        console_logger.info(
            f"Extra chair distance to table: {initial_distance_to_table:.3f}m → "
            f"{final_distance_to_table:.3f}m (moved farther by "
            f"{final_distance_to_table - initial_distance_to_table:.3f}m)"
        )
        self.assertGreater(
            final_distance_to_table,
            initial_distance_to_table,
            msg=(
                f"Push-out should move chair farther from table, but distance decreased: "
                f"{initial_distance_to_table:.3f}m → {final_distance_to_table:.3f}m"
            ),
        )

        # Check 2: Verify snap-in refinement pulled chair back closer than AABB-only.
        # Chair bbox: 0.7m width, Table bbox: ~1.5m width after rotation.
        # With offset (0.2, 0): overlap_x ≈ 0.5m, so AABB push-out ≈ 0.5m + 0.03m margin.
        # Expected AABB-only distance: 0.2m + 0.53m = 0.73m.
        # Snap-in should refine this to be closer (e.g., 0.6-0.7m).
        aabb_only_distance = (
            initial_distance_to_table + 0.53
        )  # Approximate AABB push-out
        console_logger.info(
            f"AABB-only would achieve: {aabb_only_distance:.3f}m, "
            f"actual after snap-in: {final_distance_to_table:.3f}m"
        )
        self.assertLess(
            final_distance_to_table,
            aabb_only_distance,
            msg=(
                f"Snap-in should refine position closer than AABB-only ({aabb_only_distance:.3f}m), "
                f"but got {final_distance_to_table:.3f}m"
            ),
        )

        # Check 3: Verify chair is facing left toward table (orientation="toward").
        final_extra_chair_rpy = final_extra_chair.transform.rotation().ToRollPitchYaw()
        final_extra_chair_yaw_deg = np.rad2deg(final_extra_chair_rpy.yaw_angle())
        # Table is to the left (+Y direction from chair), so chair should face 90° (left).
        expected_yaw = 90.0
        yaw_error = min(
            abs(final_extra_chair_yaw_deg - expected_yaw),
            abs(final_extra_chair_yaw_deg - (expected_yaw + 360.0)),
            abs(final_extra_chair_yaw_deg - (expected_yaw - 360.0)),
        )
        console_logger.info(
            f"Extra chair orientation: {final_extra_chair_yaw_deg:.1f}° "
            f"(expected {expected_yaw}°, error={yaw_error:.1f}°)"
        )
        self.assertLess(
            yaw_error,
            10.0,
            msg=(
                f"Chair should face left toward table (yaw≈{expected_yaw}°), "
                f"but got {final_extra_chair_yaw_deg:.1f}°"
            ),
        )

        # Check collisions after extra_chair snap (should be zero).
        self._check_collisions(
            scene=scene,
            context="after extra_chair snap",
            expected_count=0,
        )

        # Verify position: desk back should touch wall's inner face.
        # back_wall inner face is at y=-2.45 (wall center -2.5 + half-thickness 0.05).
        # Desk collision geometry Y range: -0.372 to +0.372 (depth: 0.743m).
        # With orientation="away", desk faces away from wall (back toward wall).
        # For desk back (-0.372 local Y) to touch wall at -2.45:
        # desk center Y = -2.45 - (-0.372) = -2.078.
        # With snap margin (~1.5cm), expect y ≈ -2.063.
        expected_y = -2.063
        self.assertAlmostEqual(
            final_pos[1],
            expected_y,
            delta=0.05,
            msg=f"Desk center should be at y≈{expected_y} (back touching wall at -2.45)",
        )

        # Verify x stayed at off-center position (tests that snap preserves X).
        expected_x = 0.63
        self.assertAlmostEqual(
            final_pos[0],
            expected_x,
            delta=0.1,
            msg=f"Desk should remain at X≈{expected_x}",
        )

        # Verify z stayed at floor level (no floating).
        self.assertAlmostEqual(
            final_pos[2],
            0.0,
            delta=0.01,
            msg="Desk should remain on floor during snap",
        )

        # Verify orientation: desk should have rotated from 90° to face away (0°).
        # back_wall normal is (0, 1) pointing +Y.
        # "away" should make desk face +Y direction (yaw=0°).
        expected_yaw = 0.0
        yaw_error = abs(final_yaw_deg - expected_yaw)
        console_logger.info(
            f"\nOrientation check: expected yaw={expected_yaw}°, "
            f"got {final_yaw_deg:.1f}° (error={yaw_error:.1f}°)"
        )

        # Verify orientation was corrected (90° → 0°).
        self.assertLess(
            yaw_error,
            10.0,
            msg=(
                f"Desk should rotate from {initial_yaw_deg}° to face away "
                f"from wall (yaw≈{expected_yaw}°), but got {final_yaw_deg:.1f}°. "
                f"Orientation logic is broken."
            ),
        )

        # Verify chair snap succeeded.
        self.assertTrue(chair_result["success"], "Chair snap operation should succeed")

        # Get final chair state.
        final_chair = scene.objects[chair_obj.object_id]
        final_chair_pos = final_chair.transform.translation()
        final_chair_rpy = final_chair.transform.rotation().ToRollPitchYaw()
        final_chair_yaw_deg = np.rad2deg(final_chair_rpy.yaw_angle())

        # Log chair initial vs final state.
        console_logger.info(f"\n=== Chair Snap Verification ===")
        console_logger.info(f"Chair initial position: (2.0, 0.5, 0.0)")
        console_logger.info(f"Chair initial yaw: {chair_initial_yaw_deg}°")
        console_logger.info(
            f"Chair final position: ({final_chair_pos[0]:.3f}, "
            f"{final_chair_pos[1]:.3f}, {final_chair_pos[2]:.3f})"
        )
        console_logger.info(f"Chair final yaw: {final_chair_yaw_deg:.1f}°")
        console_logger.info(
            f"Chair yaw change: {chair_initial_yaw_deg}° → {final_chair_yaw_deg:.1f}°"
        )

        # Verify chair is positioned in front of desk (user-facing side).
        # Desk is at y≈-2.063, facing +Y (yaw=0°).
        # Desk depth is 0.743m, so front edge is at y≈-2.063 + 0.372 = -1.691.
        # Chair should be snapped to desk front, so chair.y > desk.y.
        self.assertGreater(
            final_chair_pos[1],
            final_pos[1],
            msg="Chair should be in front of desk (greater Y value)",
        )

        # Verify chair is facing the desk (orientation="toward").
        # Desk is facing +Y (yaw=0°), so chair should face -Y direction.
        # In Drake coordinates, facing -Y means yaw=180° (or -180°, equivalent).
        # We need to handle angle wrapping for comparison.
        expected_chair_yaw = 180.0  # Facing toward -Y (desk).
        chair_yaw_error = min(
            abs(final_chair_yaw_deg - expected_chair_yaw),
            abs(final_chair_yaw_deg - (expected_chair_yaw - 360.0)),
            abs(final_chair_yaw_deg - (expected_chair_yaw + 360.0)),
        )
        console_logger.info(
            f"\nChair orientation check: expected yaw≈{expected_chair_yaw}° "
            f"(facing desk), got {final_chair_yaw_deg:.1f}° (error={chair_yaw_error:.1f}°)"
        )

        self.assertLess(
            chair_yaw_error,
            10.0,
            msg=(
                f"Chair should face toward desk (yaw≈{expected_chair_yaw}°), "
                f"but got {final_chair_yaw_deg:.1f}°. "
                f"Orientation logic for chair-to-desk snap is incorrect."
            ),
        )

        # Verify chair is on the floor (z≈0).
        self.assertAlmostEqual(
            final_chair_pos[2],
            0.0,
            delta=0.01,
            msg="Chair should remain on floor during snap",
        )

        # Verify chair ends up close to desk (core snapping behavior).
        # Desk is at y≈-2.063, front edge at y≈-1.691.
        # Chair should be snapped close to desk front.
        # Use Euclidean distance in XY plane.
        chair_desk_distance_xy = np.linalg.norm(final_chair_pos[:2] - final_pos[:2])
        console_logger.info(
            f"\nChair-to-desk distance (XY plane): {chair_desk_distance_xy:.3f}m"
        )
        self.assertLess(
            chair_desk_distance_xy,
            1.5,
            msg=(
                f"Chair should end up close to desk after snapping "
                f"(distance < 1.5m), but got {chair_desk_distance_xy:.3f}m. "
                f"Snap-to-surface logic is broken."
            ),
        )

        # Note: X-alignment is NOT enforced because snap-to-surface chooses the
        # closest point, which can be any side of the desk (front, back, left,
        # right). Visual verification via renders confirms correct behavior.

        # Final validation: all 4 objects must have zero collisions.
        console_logger.info("\n=== FINAL COLLISION CHECK (ALL OBJECTS) ===")
        self._check_collisions(
            scene=scene,
            context="in final scene (all objects)",
            expected_count=0,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
