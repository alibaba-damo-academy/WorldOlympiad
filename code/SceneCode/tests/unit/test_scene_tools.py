import json
import math
import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import trimesh

from omegaconf import OmegaConf
from pydrake.all import RigidTransform, RollPitchYaw

from scenecode.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scenecode.furniture_agents.tools.scene_tools import SceneTools


class TestSnapToObject(unittest.TestCase):
    """Test snap_to_object_tool key contracts."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_scene = Mock(spec=RoomScene)
        self.mock_scene.objects = {}
        self.mock_scene.action_log_path = self.temp_dir / "action_log.json"

        # Add floor_plan mock with default empty wall_normals.
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.wall_normals = {}

        # Load base configuration from actual config file.
        config_path = (
            Path(__file__).parent.parent.parent
            / "configs/furniture_agent/base_furniture_agent.yaml"
        )
        self.cfg = OmegaConf.load(config_path)

        # Unit tests focus on testing snap contracts (rotation, validation, etc.).
        # Integration tests verify end-to-end behavior with real SDF files.
        self.scene_tools = SceneTools(scene=self.mock_scene, cfg=self.cfg)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_cube_mesh(self, size=1.0) -> Path:
        """Create a simple cube mesh for testing."""
        mesh = trimesh.creation.box(extents=[size, size, size])
        path = self.temp_dir / f"cube_{size}.glb"
        mesh.export(path)
        return path

    def test_snap_preserves_rotation(self):
        """Contract: Snap only translates, rotation is preserved."""
        obj_mesh = self._create_cube_mesh(size=1.0)
        target_mesh = self._create_cube_mesh(size=1.0)

        # Object with 45-degree rotation.
        initial_yaw = math.pi / 4
        obj = SceneObject(
            object_id=UniqueID("obj"),
            object_type=ObjectType.FURNITURE,
            name="Chair",
            description="Test",
            transform=RigidTransform(
                rpy=RollPitchYaw(0.0, 0.0, initial_yaw), p=[0.0, 0.0, 0.0]
            ),
            geometry_path=obj_mesh,
            bbox_min=np.array([-0.5, -0.5, -0.5]),
            bbox_max=np.array([0.5, 0.5, 0.5]),
        )

        target = SceneObject(
            object_id=UniqueID("target"),
            object_type=ObjectType.FURNITURE,
            name="Table",
            description="Test",
            transform=RigidTransform(p=[3.0, 0.0, 0.0]),
            geometry_path=target_mesh,
            bbox_min=np.array([-0.5, -0.5, -0.5]),
            bbox_max=np.array([0.5, 0.5, 0.5]),
        )

        # Set up scene.
        self.mock_scene.objects = {obj.object_id: obj, target.object_id: target}
        self.mock_scene.get_object = lambda uid: self.mock_scene.objects.get(uid)
        self.mock_scene.move_object = Mock(return_value=True)

        # Execute snap.
        result_json = self.scene_tools._snap_to_object_impl(
            object_id=str(obj.object_id), target_id=str(target.object_id)
        )
        result = json.loads(result_json)

        self.assertTrue(result["success"])

        # Verify rotation preserved in new transform.
        new_transform = self.mock_scene.move_object.call_args[1]["new_transform"]
        new_yaw = new_transform.rotation().ToRollPitchYaw().yaw_angle()

        self.assertAlmostEqual(
            new_yaw,
            initial_yaw,
            places=5,
            msg="Rotation should be preserved during snap",
        )

    def test_no_rotation_for_furniture_to_furniture(self):
        """Contract: Rotation only applies to wall snapping, not furniture."""

        obj_mesh = self._create_cube_mesh(size=1.0)
        target_mesh = self._create_cube_mesh(size=1.0)

        # Object with rotation.
        initial_yaw = math.radians(25.0)
        obj = SceneObject(
            object_id=UniqueID("chair"),
            object_type=ObjectType.FURNITURE,
            name="Chair",
            description="Test",
            transform=RigidTransform(
                rpy=RollPitchYaw(0.0, 0.0, initial_yaw), p=[0.0, 0.0, 0.0]
            ),
            geometry_path=obj_mesh,
            bbox_min=np.array([-0.5, -0.5, -0.5]),
            bbox_max=np.array([0.5, 0.5, 0.5]),
        )

        # Target is furniture (not a wall).
        target = SceneObject(
            object_id=UniqueID("table"),
            object_type=ObjectType.FURNITURE,
            name="Table",
            description="Test",
            transform=RigidTransform(p=[3.0, 0.0, 0.0]),
            geometry_path=target_mesh,
            bbox_min=np.array([-0.5, -0.5, -0.5]),
            bbox_max=np.array([0.5, 0.5, 0.5]),
        )

        # Set up scene.
        self.mock_scene.objects = {obj.object_id: obj, target.object_id: target}
        self.mock_scene.get_object = lambda uid: self.mock_scene.objects.get(uid)
        self.mock_scene.move_object = Mock(return_value=True)

        # Execute snap (alignment only applies to walls, not furniture).
        result_json = self.scene_tools._snap_to_object_impl(
            object_id=str(obj.object_id),
            target_id=str(target.object_id),
        )
        result = json.loads(result_json)

        self.assertTrue(result["success"])
        self.assertFalse(result["rotation_applied"])
        self.assertIsNone(result["rotation_angle_degrees"])

        # Verify rotation preserved.
        new_transform = self.mock_scene.move_object.call_args[1]["new_transform"]
        new_yaw = new_transform.rotation().ToRollPitchYaw().yaw_angle()
        self.assertAlmostEqual(new_yaw, initial_yaw, places=5)

    def test_no_rotation_when_config_disabled(self):
        """Contract: Rotation not applied when config disabled (for ablations)."""
        obj_mesh = self._create_cube_mesh(size=1.0)

        # Object with rotation.
        initial_yaw = math.radians(15.0)
        obj = SceneObject(
            object_id=UniqueID("wardrobe"),
            object_type=ObjectType.FURNITURE,
            name="Wardrobe",
            description="Test",
            transform=RigidTransform(
                rpy=RollPitchYaw(0.0, 0.0, initial_yaw), p=[0.0, 0.0, 0.5]
            ),
            geometry_path=obj_mesh,
            bbox_min=np.array([-0.5, -0.5, -0.5]),
            bbox_max=np.array([0.5, 0.5, 0.5]),
        )

        # Wall.
        wall = SceneObject(
            object_id=UniqueID("wall"),
            object_type=ObjectType.WALL,
            name="back_wall",
            description="Test",
            transform=RigidTransform(p=[5.0, 0.0, 1.5]),
            geometry_path=None,
            bbox_min=np.array([-0.1, -5.0, 0.0]),
            bbox_max=np.array([0.1, 5.0, 3.0]),
            immutable=True,
        )

        # Set up scene.
        self.mock_scene.objects = {obj.object_id: obj, wall.object_id: wall}
        self.mock_scene.get_object = lambda uid: self.mock_scene.objects.get(uid)
        self.mock_scene.move_object = Mock(return_value=True)
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.wall_normals = {"back_wall": np.array([1.0, 0.0])}

        # Execute snap with orientation="away" to test explicit alignment.
        result_json = self.scene_tools._snap_to_object_impl(
            object_id=str(obj.object_id),
            target_id=str(wall.object_id),
        )
        result = json.loads(result_json)

        self.assertTrue(result["success"])
        self.assertFalse(result["rotation_applied"])
        self.assertIsNone(result["rotation_angle_degrees"])

        # Verify rotation preserved.
        new_transform = self.mock_scene.move_object.call_args[1]["new_transform"]
        new_yaw = new_transform.rotation().ToRollPitchYaw().yaw_angle()
        self.assertAlmostEqual(new_yaw, initial_yaw, places=5)

    def test_orientation_none_preserves_rotation(self):
        """Contract: orientation='none' does not change rotation."""
        obj_mesh = self._create_cube_mesh(size=1.0)
        target_mesh = self._create_cube_mesh(size=1.0)

        initial_yaw = math.radians(37.0)  # Arbitrary angle.
        obj = SceneObject(
            object_id=UniqueID("nightstand"),
            object_type=ObjectType.FURNITURE,
            name="Nightstand",
            description="Bedroom Nightstand",
            transform=RigidTransform(
                rpy=RollPitchYaw(0.0, 0.0, initial_yaw), p=[2.0, 0.0, 0.0]
            ),
            geometry_path=obj_mesh,
            bbox_min=np.array([-0.5, -0.5, -0.5]),
            bbox_max=np.array([0.5, 0.5, 0.5]),
        )

        target = SceneObject(
            object_id=UniqueID("bed"),
            object_type=ObjectType.FURNITURE,
            name="Bed",
            description="Bedroom Bed",
            transform=RigidTransform(p=[0.0, 0.0, 0.0]),
            geometry_path=target_mesh,
            bbox_min=np.array([-1.0, -1.0, -0.5]),
            bbox_max=np.array([1.0, 1.0, 0.5]),
        )

        # Set up scene.
        self.mock_scene.objects = {obj.object_id: obj, target.object_id: target}
        self.mock_scene.get_object = lambda uid: self.mock_scene.objects.get(uid)
        self.mock_scene.move_object = Mock(return_value=True)
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.wall_normals = {}

        # Execute snap with orientation="none" (default).
        result_json = self.scene_tools._snap_to_object_impl(
            object_id=str(obj.object_id),
            target_id=str(target.object_id),
            orientation="none",
        )
        result = json.loads(result_json)

        self.assertTrue(result["success"])

        # Rotation should not be applied (unless for other reasons like wall).
        # Since target is not a wall, rotation_applied should be False.
        self.assertFalse(result["rotation_applied"])

    def test_orientation_toward_round_table(self):
        """Contract: orientation='toward' works for round tables."""
        obj_mesh = self._create_cube_mesh(size=0.5)

        # Create round table mesh (cylinder).
        table_mesh = trimesh.creation.cylinder(radius=1.0, height=0.1)
        table_path = self.temp_dir / "round_table.glb"
        table_mesh.export(table_path)

        # Chair north of round table.
        obj = SceneObject(
            object_id=UniqueID("chair"),
            object_type=ObjectType.FURNITURE,
            name="Chair",
            description="Dining Chair",
            transform=RigidTransform(
                rpy=RollPitchYaw(0.0, 0.0, 0.0), p=[0.0, 2.0, 0.0]
            ),
            geometry_path=obj_mesh,
            bbox_min=np.array([-0.25, -0.25, -0.25]),
            bbox_max=np.array([0.25, 0.25, 0.25]),
        )

        # Round table at origin.
        table = SceneObject(
            object_id=UniqueID("table"),
            object_type=ObjectType.FURNITURE,
            name="Round Table",
            description="Round Dining Table",
            transform=RigidTransform(p=[0.0, 0.0, 0.0]),
            geometry_path=table_path,
            bbox_min=np.array([-1.0, -1.0, -0.05]),
            bbox_max=np.array([1.0, 1.0, 0.05]),
        )

        # Set up scene.
        self.mock_scene.objects = {obj.object_id: obj, table.object_id: table}
        self.mock_scene.get_object = lambda uid: self.mock_scene.objects.get(uid)
        self.mock_scene.move_object = Mock(return_value=True)
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.wall_normals = {}

        # Execute snap with orientation="toward".
        result_json = self.scene_tools._snap_to_object_impl(
            object_id=str(obj.object_id),
            target_id=str(table.object_id),
            orientation="toward",
        )
        result = json.loads(result_json)

        self.assertTrue(result["success"])
        self.assertTrue(result["rotation_applied"])

        # Chair should face toward table center (south, toward -Y).
        # Expected yaw: 180 degrees (or -180, they're equivalent).
        final_yaw = result["rotation_angle_degrees"]
        self.assertIsNotNone(final_yaw)
        # Normalize angle to [-180, 180] range.
        normalized_yaw = ((final_yaw + 180.0) % 360.0) - 180.0
        self.assertAlmostEqual(abs(normalized_yaw), 180.0, delta=5.0)

    def test_orientation_with_penetration_resolution(self):
        """Contract: Orientation works with penetration resolution."""
        obj_mesh = self._create_cube_mesh(size=1.0)
        target_mesh = self._create_cube_mesh(size=2.0)

        # Chair overlapping table, facing wrong direction.
        obj = SceneObject(
            object_id=UniqueID("chair"),
            object_type=ObjectType.FURNITURE,
            name="Chair",
            description="Dining Chair",
            transform=RigidTransform(
                rpy=RollPitchYaw(0.0, 0.0, math.radians(45.0)), p=[0.5, 0.0, 0.0]
            ),
            geometry_path=obj_mesh,
            bbox_min=np.array([-0.5, -0.5, -0.5]),
            bbox_max=np.array([0.5, 0.5, 0.5]),
        )

        # Table at origin.
        target = SceneObject(
            object_id=UniqueID("table"),
            object_type=ObjectType.FURNITURE,
            name="Table",
            description="Dining Table",
            transform=RigidTransform(p=[0.0, 0.0, 0.0]),
            geometry_path=target_mesh,
            bbox_min=np.array([-1.0, -1.0, -0.5]),
            bbox_max=np.array([1.0, 1.0, 0.5]),
        )

        # Set up scene.
        self.mock_scene.objects = {obj.object_id: obj, target.object_id: target}
        self.mock_scene.get_object = lambda uid: self.mock_scene.objects.get(uid)
        self.mock_scene.move_object = Mock(return_value=True)
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.wall_normals = {}

        # Execute snap with orientation="toward".
        # Should: rotate, resolve penetration, then snap.
        result_json = self.scene_tools._snap_to_object_impl(
            object_id=str(obj.object_id),
            target_id=str(target.object_id),
            orientation="toward",
        )
        result = json.loads(result_json)

        self.assertTrue(result["success"])
        self.assertTrue(result["rotation_applied"])

        # Operation should succeed even with penetration.
        # Note: distance_moved might be 0 if already within snap threshold.
        self.assertIsNotNone(result["distance_moved"])

    def test_orientation_combined_with_wall_alignment(self):
        """Contract: Orientation and wall alignment both apply correctly."""
        obj_mesh = self._create_cube_mesh(size=1.0)

        # Wardrobe west of wall, with wall normal defined.
        obj = SceneObject(
            object_id=UniqueID("wardrobe"),
            object_type=ObjectType.FURNITURE,
            name="Wardrobe",
            description="Bedroom Wardrobe",
            transform=RigidTransform(
                rpy=RollPitchYaw(0.0, 0.0, math.radians(0.0)), p=[-2.0, 0.0, 0.5]
            ),
            geometry_path=obj_mesh,
            bbox_min=np.array([-0.5, -0.5, -0.5]),
            bbox_max=np.array([0.5, 0.5, 0.5]),
        )

        # Wall with normal defined.
        wall = SceneObject(
            object_id=UniqueID("wall"),
            object_type=ObjectType.WALL,
            name="east_wall",
            description="East Wall",
            transform=RigidTransform(p=[0.0, 0.0, 1.5]),
            geometry_path=None,
            bbox_min=np.array([-0.1, -5.0, 0.0]),
            bbox_max=np.array([0.1, 5.0, 3.0]),
            immutable=True,
        )

        # Set up scene with wall normal.
        self.mock_scene.objects = {obj.object_id: obj, wall.object_id: wall}
        self.mock_scene.get_object = lambda uid: self.mock_scene.objects.get(uid)
        self.mock_scene.move_object = Mock(return_value=True)
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.wall_normals = {
            "east_wall": np.array([-1.0, 0.0])  # Normal points west.
        }

        # Execute snap with orientation="away".
        # Should apply both orientation and wall alignment.
        result_json = self.scene_tools._snap_to_object_impl(
            object_id=str(obj.object_id),
            target_id=str(wall.object_id),
            orientation="away",
        )
        result = json.loads(result_json)

        self.assertTrue(result["success"])
        self.assertTrue(result["rotation_applied"])

        # Both orientation and wall alignment should have been applied.
        # Check that rotation was applied (exact angle depends on both operations).
        self.assertIsNotNone(result["rotation_angle_degrees"])


if __name__ == "__main__":
    unittest.main()
