"""Unit tests for scene reset/checkpoint functionality."""

import copy
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from pydrake.all import RigidTransform

from scenecode.agent_utils.house import RoomGeometry
from scenecode.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID


class TestSceneReset(unittest.TestCase):
    """Test scene checkpoint and restore functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Use existing test data floor plan.
        test_data_dir = Path(__file__).parent.parent / "test_data"
        floor_plan_path = test_data_dir / "simple_room_geometry.sdf"

        # Read the existing SDF file.
        with open(floor_plan_path, "r") as f:
            floor_plan_sdf = f.read()

        # Create RoomGeometry object.
        room_geometry_tree = ET.ElementTree(ET.fromstring(floor_plan_sdf))
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=floor_plan_path,
        )

        # Create test scene.
        self.scene = RoomScene(
            room_geometry=room_geometry,
            text_description="Test scene for reset functionality",
            scene_dir=test_data_dir,
        )

    def test_scene_checkpoint_and_restore(self):
        """Test saving and restoring scene state."""
        # Add initial object to scene.
        initial_obj = SceneObject(
            object_id=UniqueID("test_table"),
            object_type=ObjectType.FURNITURE,
            name="Test Table",
            description="A test table",
            transform=RigidTransform([1.0, 2.0, 0.0]),
        )
        self.scene.add_object(initial_obj)

        # Save checkpoint.
        checkpoint = copy.deepcopy(self.scene.to_state_dict())

        # Verify initial state.
        self.assertEqual(len(self.scene.objects), 1)
        self.assertIn(initial_obj.object_id, self.scene.objects)

        # Modify scene (simulate bad changes).
        self.scene.remove_object(initial_obj.object_id)
        second_obj = SceneObject(
            object_id=UniqueID("bad_chair"),
            object_type=ObjectType.FURNITURE,
            name="Bad Chair",
            description="A bad chair",
            transform=RigidTransform([3.0, 4.0, 0.0]),
        )
        self.scene.add_object(second_obj)

        # Verify scene changed.
        self.assertEqual(len(self.scene.objects), 1)
        self.assertNotIn(initial_obj.object_id, self.scene.objects)
        self.assertIn(second_obj.object_id, self.scene.objects)

        # Restore from checkpoint.
        self.scene.restore_from_state_dict(checkpoint)

        # Verify restoration.
        self.assertEqual(len(self.scene.objects), 1)
        self.assertIn(initial_obj.object_id, self.scene.objects)
        self.assertNotIn(second_obj.object_id, self.scene.objects)

        # Verify restored object properties.
        restored_obj = self.scene.get_object(initial_obj.object_id)
        self.assertEqual(restored_obj.name, initial_obj.name)
        self.assertEqual(restored_obj.description, initial_obj.description)
        self.assertEqual(restored_obj.object_type, initial_obj.object_type)

        # Verify transform was restored correctly.
        original_translation = initial_obj.transform.translation()
        restored_translation = restored_obj.transform.translation()
        np.testing.assert_array_almost_equal(
            original_translation, restored_translation, decimal=6
        )

    def test_empty_scene_checkpoint_restore(self):
        """Test checkpoint/restore with empty scene."""
        # Save checkpoint of empty scene.
        checkpoint = copy.deepcopy(self.scene.to_state_dict())

        # Add object.
        obj = SceneObject(
            object_id=UniqueID("temp_obj"),
            object_type=ObjectType.FURNITURE,
            name="Temp Object",
            description="Temporary object",
            transform=RigidTransform([0.0, 0.0, 0.0]),
        )
        self.scene.add_object(obj)
        self.assertEqual(len(self.scene.objects), 1)

        # Restore to empty state.
        self.scene.restore_from_state_dict(checkpoint)

        # Verify scene is empty again.
        self.assertEqual(len(self.scene.objects), 0)


if __name__ == "__main__":
    unittest.main()
