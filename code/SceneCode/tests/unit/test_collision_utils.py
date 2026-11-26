"""Unit tests for collision detection utilities."""

import unittest

from pathlib import Path

from pydrake.math import RigidTransform

from scenecode.agent_utils.room import ObjectType, SceneObject, UniqueID
from scenecode.utils.collision_utils import compute_pairwise_collisions

# Path to test data directory.
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


class TestComputePairwiseCollisions(unittest.TestCase):
    """Tests for compute_pairwise_collisions function."""

    def _create_scene_object(
        self,
        name: str,
        sdf_path: Path,
    ) -> SceneObject:
        """Create a SceneObject for testing."""
        return SceneObject(
            object_id=UniqueID(name),
            object_type=ObjectType.MANIPULAND,
            name=name,
            description=f"Test object {name}",
            transform=RigidTransform(),
            sdf_path=sdf_path,
        )

    def test_empty_list_returns_empty(self):
        """Empty objects list returns empty collisions."""
        collisions = compute_pairwise_collisions(objects=[], transforms=[])
        self.assertEqual(collisions, [])

    def test_single_object_returns_empty(self):
        """Single object cannot collide with itself."""
        sdf_path = TEST_DATA_DIR / "simple_box.sdf"
        obj = self._create_scene_object("box1", sdf_path)
        transform = RigidTransform()

        collisions = compute_pairwise_collisions(
            objects=[obj],
            transforms=[transform],
        )
        self.assertEqual(collisions, [])

    def test_separated_objects_no_collision(self):
        """Objects far apart have no collision."""
        sdf_path = TEST_DATA_DIR / "simple_box.sdf"
        obj1 = self._create_scene_object("box1", sdf_path)
        obj2 = self._create_scene_object("box2", sdf_path)

        # Place boxes 2m apart (each box is 0.5m, so 2m is well separated).
        transform1 = RigidTransform(p=[0, 0, 0])
        transform2 = RigidTransform(p=[2.0, 0, 0])

        collisions = compute_pairwise_collisions(
            objects=[obj1, obj2],
            transforms=[transform1, transform2],
        )
        self.assertEqual(collisions, [])

    def test_overlapping_objects_detected(self):
        """Overlapping objects are detected with correct indices."""
        sdf_path = TEST_DATA_DIR / "simple_box.sdf"
        obj1 = self._create_scene_object("box1", sdf_path)
        obj2 = self._create_scene_object("box2", sdf_path)

        # Place boxes 0.4m apart (each box is 0.5m, so 0.1m overlap).
        transform1 = RigidTransform(p=[0, 0, 0])
        transform2 = RigidTransform(p=[0.4, 0, 0])

        collisions = compute_pairwise_collisions(
            objects=[obj1, obj2],
            transforms=[transform1, transform2],
        )

        self.assertEqual(len(collisions), 1)
        collision = collisions[0]
        self.assertIn(collision.obj_a_idx, [0, 1])
        self.assertIn(collision.obj_b_idx, [0, 1])
        self.assertNotEqual(collision.obj_a_idx, collision.obj_b_idx)
        # Penetration should be approximately 0.1m (each box extends 0.25m).
        self.assertGreater(collision.penetration_m, 0.05)
        self.assertLess(collision.penetration_m, 0.15)

    def test_penetration_depth_accuracy(self):
        """Penetration depth is accurate for known overlap."""
        sdf_path = TEST_DATA_DIR / "simple_sphere.sdf"
        obj1 = self._create_scene_object("sphere1", sdf_path)
        obj2 = self._create_scene_object("sphere2", sdf_path)

        # Spheres with radius 0.2m placed 0.3m apart (0.1m overlap).
        transform1 = RigidTransform(p=[0, 0, 0])
        transform2 = RigidTransform(p=[0.3, 0, 0])

        collisions = compute_pairwise_collisions(
            objects=[obj1, obj2],
            transforms=[transform1, transform2],
        )

        self.assertEqual(len(collisions), 1)
        # Expected penetration: 2*0.2 - 0.3 = 0.1m.
        self.assertAlmostEqual(collisions[0].penetration_m, 0.1, places=2)

    def test_collisions_sorted_by_penetration(self):
        """Multiple collisions are sorted by penetration depth."""
        sdf_path = TEST_DATA_DIR / "simple_box.sdf"
        obj1 = self._create_scene_object("box1", sdf_path)
        obj2 = self._create_scene_object("box2", sdf_path)
        obj3 = self._create_scene_object("box3", sdf_path)

        # Box 1 and 2 have small overlap (0.45m apart).
        # Box 2 and 3 have larger overlap (0.3m apart).
        transform1 = RigidTransform(p=[0, 0, 0])
        transform2 = RigidTransform(p=[0.45, 0, 0])
        transform3 = RigidTransform(p=[0.75, 0, 0])  # 0.3m from box2

        collisions = compute_pairwise_collisions(
            objects=[obj1, obj2, obj3],
            transforms=[transform1, transform2, transform3],
        )

        # Should have 2 collisions, sorted by penetration (largest first).
        self.assertEqual(len(collisions), 2)
        self.assertGreater(collisions[0].penetration_m, collisions[1].penetration_m)

    def test_mismatched_lengths_raises(self):
        """Mismatched objects/transforms lengths raises ValueError."""
        sdf_path = TEST_DATA_DIR / "simple_box.sdf"
        obj1 = self._create_scene_object("box1", sdf_path)
        transform1 = RigidTransform()
        transform2 = RigidTransform(p=[1, 0, 0])

        with self.assertRaises(ValueError):
            compute_pairwise_collisions(
                objects=[obj1],
                transforms=[transform1, transform2],
            )


if __name__ == "__main__":
    unittest.main()
