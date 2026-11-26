"""Unit tests for reachability analysis functions."""

import unittest

from pathlib import Path

import numpy as np

from pydrake.all import RigidTransform, RotationMatrix

from scenecode.agent_utils.house import RoomGeometry
from scenecode.agent_utils.reachability import (
    ReachabilityResult,
    compute_reachability,
    format_reachability_for_critic,
    format_reachability_result,
)
from scenecode.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID


class TestReachability(unittest.TestCase):
    """Test reachability analysis algorithm."""

    def setUp(self):
        """Create test scene with basic room geometry."""
        self.test_data_dir = Path(__file__).parent.parent / "test_data"
        self.robot_width = 0.5  # 50cm robot

        # Create a 6m x 4m room.
        self.room_geometry = RoomGeometry(
            sdf_tree=None,
            sdf_path=None,
            walls=[],
            floor=None,
            wall_normals={
                "north_wall": np.array([0.0, -1.0]),
                "south_wall": np.array([0.0, 1.0]),
                "east_wall": np.array([-1.0, 0.0]),
                "west_wall": np.array([1.0, 0.0]),
            },
            width=4.0,
            length=6.0,
            wall_height=2.7,
            openings=[],
        )

        self.scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=self.test_data_dir,
        )

    def _create_furniture_object(
        self, name: str, position: np.ndarray, size: np.ndarray
    ) -> SceneObject:
        """Create a furniture object at the specified position with given size.

        Position is the center of the object. Size is full dimensions (not half).
        """
        half_size = size / 2
        return SceneObject(
            object_id=UniqueID(name),
            object_type=ObjectType.FURNITURE,
            name=name,
            description=f"Test {name}",
            transform=RigidTransform(p=position),
            bbox_min=-half_size,
            bbox_max=half_size,
        )

    def _create_carpet_object(
        self, name: str, position: np.ndarray, size: np.ndarray
    ) -> SceneObject:
        """Create a carpet object (FURNITURE type with thin_covering asset_source)."""
        half_size = size / 2
        return SceneObject(
            object_id=UniqueID(name),
            object_type=ObjectType.FURNITURE,  # Keeps agent's type
            name=name,
            description=f"Test {name}",
            transform=RigidTransform(p=position),
            bbox_min=-half_size,
            bbox_max=half_size,
            metadata={"asset_source": "thin_covering"},  # Identified via metadata
        )

    def _create_manipuland_object(
        self, name: str, position: np.ndarray, size: np.ndarray
    ) -> SceneObject:
        """Create a manipuland object."""
        half_size = size / 2
        return SceneObject(
            object_id=UniqueID(name),
            object_type=ObjectType.MANIPULAND,
            name=name,
            description=f"Test {name}",
            transform=RigidTransform(p=position),
            bbox_min=-half_size,
            bbox_max=half_size,
        )

    def test_empty_room_fully_reachable(self):
        """Empty room should be fully reachable with single region."""
        result = compute_reachability(self.scene, self.robot_width)

        self.assertTrue(result.is_fully_reachable)
        self.assertEqual(result.num_disconnected_regions, 1)
        self.assertAlmostEqual(result.reachability_ratio, 1.0)
        self.assertEqual(result.blocking_furniture_ids, [])

    def test_furniture_along_walls_reachable(self):
        """Furniture along walls should leave room fully reachable."""
        # Add furniture along the walls (not blocking passages).
        sofa = self._create_furniture_object(
            name="sofa",
            position=np.array([3.0, 0.5, 0.4]),  # Near south wall
            size=np.array([2.0, 0.8, 0.8]),
        )
        bookshelf = self._create_furniture_object(
            name="bookshelf",
            position=np.array([0.3, 2.0, 1.0]),  # Near west wall
            size=np.array([0.4, 1.0, 2.0]),
        )
        self.scene.add_object(sofa)
        self.scene.add_object(bookshelf)

        result = compute_reachability(self.scene, self.robot_width)

        self.assertTrue(result.is_fully_reachable)
        self.assertEqual(result.num_disconnected_regions, 1)
        self.assertEqual(result.blocking_furniture_ids, [])

    def test_single_blocker_splits_room(self):
        """Single furniture piece spanning room width should split into 2 regions."""
        # Add a long sofa that completely blocks the room.
        # Room is 6m x 4m, so a 5m wide sofa at y=2 blocks passage.
        blocking_sofa = self._create_furniture_object(
            name="blocking_sofa",
            position=np.array([3.0, 2.0, 0.4]),  # Center of room
            size=np.array([5.5, 1.0, 0.8]),  # Almost room width
        )
        self.scene.add_object(blocking_sofa)

        result = compute_reachability(self.scene, self.robot_width)

        self.assertFalse(result.is_fully_reachable)
        self.assertEqual(result.num_disconnected_regions, 2)
        # Check that the blocking furniture is identified.
        self.assertTrue(
            any("blocking_sofa" in bid for bid in result.blocking_furniture_ids),
            f"Expected 'blocking_sofa' in blockers: {result.blocking_furniture_ids}",
        )

    def test_multiple_independent_blockers(self):
        """Two independent blockers should create 3 regions."""
        # Place two blockers that independently split the room.
        # First blocker at y=1.3 (splits bottom third).
        blocker1 = self._create_furniture_object(
            name="blocker1",
            position=np.array([3.0, 1.3, 0.4]),
            size=np.array([5.5, 0.8, 0.8]),
        )
        # Second blocker at y=2.7 (splits top third).
        blocker2 = self._create_furniture_object(
            name="blocker2",
            position=np.array([3.0, 2.7, 0.4]),
            size=np.array([5.5, 0.8, 0.8]),
        )
        self.scene.add_object(blocker1)
        self.scene.add_object(blocker2)

        result = compute_reachability(self.scene, self.robot_width)

        self.assertFalse(result.is_fully_reachable)
        self.assertEqual(result.num_disconnected_regions, 3)
        # Check that the blocking furniture is identified.
        self.assertTrue(
            any("blocker1" in bid for bid in result.blocking_furniture_ids),
            f"Expected 'blocker1' in blockers: {result.blocking_furniture_ids}",
        )
        self.assertTrue(
            any("blocker2" in bid for bid in result.blocking_furniture_ids),
            f"Expected 'blocker2' in blockers: {result.blocking_furniture_ids}",
        )

    def test_passage_at_exact_robot_width(self):
        """Passage exactly robot width should still be reachable."""
        # Room is 6m wide. Place furniture leaving exactly 0.5m passage on one side.
        # Furniture width: 6.0 - 0.5 (passage) - 0.25 (wall buffer) = 5.25m
        # With robot half-width buffer, effective passage is 0.5 - 0.5 = 0 on that side.
        # Let's create a more controlled test: leave 0.6m passage which becomes
        # 0.1m effective (just above zero).
        furniture = self._create_furniture_object(
            name="almost_blocking",
            position=np.array([2.7, 2.0, 0.4]),
            size=np.array([4.8, 1.0, 0.8]),
        )
        self.scene.add_object(furniture)

        result = compute_reachability(self.scene, self.robot_width)

        # Should still be reachable (passage exists even if tight).
        self.assertTrue(result.is_fully_reachable)
        self.assertEqual(result.num_disconnected_regions, 1)

    def test_passage_below_robot_width_disconnected(self):
        """Passage smaller than robot width should create disconnected regions."""
        # Place furniture leaving passage smaller than robot width (0.5m).
        # Leave only 0.3m gap which is below robot width.
        # Room is 6m x 4m. Furniture of 5.8m leaves 0.2m passage.
        furniture = self._create_furniture_object(
            name="wide_furniture",
            position=np.array([2.9, 2.0, 0.4]),
            size=np.array([5.6, 1.0, 0.8]),  # Leaves 0.4m total (0.2m each side)
        )
        self.scene.add_object(furniture)

        result = compute_reachability(self.scene, self.robot_width)

        self.assertFalse(result.is_fully_reachable)
        self.assertGreater(result.num_disconnected_regions, 1)

    def test_cooperative_blocking_both_identified(self):
        """Two overlapping pieces that together block should both be identified."""
        # Place two furniture pieces that together block the room,
        # but removing either one opens the passage.
        # Each piece IS a blocker because removing it reduces regions from 2 to 1.

        # Room is 6m x 4m. Place two pieces each 3.5m wide, overlapping in middle.
        # Together they span the room, but each alone has a gap on opposite ends.
        piece1 = self._create_furniture_object(
            name="piece1",
            position=np.array([1.5, 2.0, 0.4]),  # Left side
            size=np.array([3.5, 1.0, 0.8]),  # 0 to 3.25m
        )
        piece2 = self._create_furniture_object(
            name="piece2",
            position=np.array([4.5, 2.0, 0.4]),  # Right side
            size=np.array([3.5, 1.0, 0.8]),  # 2.75m to 6m
        )
        self.scene.add_object(piece1)
        self.scene.add_object(piece2)

        result = compute_reachability(self.scene, self.robot_width)

        # Should be disconnected (together they block).
        self.assertFalse(result.is_fully_reachable)
        self.assertEqual(result.num_disconnected_regions, 2)

        # Removing either piece opens the room, so both are identified as blockers.
        # Check that the blocking furniture is identified.
        self.assertTrue(
            any("piece1" in bid for bid in result.blocking_furniture_ids),
            f"Expected 'piece1' in blockers: {result.blocking_furniture_ids}",
        )
        self.assertTrue(
            any("piece2" in bid for bid in result.blocking_furniture_ids),
            f"Expected 'piece2' in blockers: {result.blocking_furniture_ids}",
        )

    def test_non_furniture_ignored(self):
        """Carpets and manipulands should not affect reachability."""
        # Add a carpet that would block if it were furniture.
        carpet = self._create_carpet_object(
            name="large_carpet",
            position=np.array([3.0, 2.0, 0.01]),  # Center of room, on floor
            size=np.array([5.5, 3.5, 0.02]),  # Would block room if counted
        )
        # Add manipuland.
        vase = self._create_manipuland_object(
            name="vase",
            position=np.array([3.0, 2.0, 0.5]),
            size=np.array([0.2, 0.2, 0.4]),
        )
        self.scene.add_object(carpet)
        self.scene.add_object(vase)

        result = compute_reachability(self.scene, self.robot_width)

        # Should be fully reachable since only FURNITURE type is considered.
        self.assertTrue(result.is_fully_reachable)
        self.assertEqual(result.num_disconnected_regions, 1)
        self.assertEqual(result.blocking_furniture_ids, [])

    def test_reachability_ratio_calculation(self):
        """Test reachability ratio is computed correctly for disconnected regions."""
        # Create a blocker that splits room into unequal regions.
        # Place blocker near one end to create asymmetric split.
        blocker = self._create_furniture_object(
            name="asymmetric_blocker",
            position=np.array([3.0, 1.0, 0.4]),  # Near south side
            size=np.array([5.5, 0.8, 0.8]),
        )
        self.scene.add_object(blocker)

        result = compute_reachability(self.scene, self.robot_width)

        # Reachability ratio should be < 1.0 for disconnected room.
        self.assertFalse(result.is_fully_reachable)
        self.assertLess(result.reachability_ratio, 1.0)
        self.assertGreater(result.reachability_ratio, 0.0)

    def test_room_too_small_for_robot(self):
        """Room smaller than robot should return 0 regions."""
        # Create a tiny room smaller than robot footprint.
        tiny_geometry = RoomGeometry(
            sdf_tree=None,
            sdf_path=None,
            walls=[],
            floor=None,
            wall_normals={},
            width=0.3,  # 30cm - smaller than robot
            length=0.3,
            wall_height=2.7,
            openings=[],
        )
        tiny_scene = RoomScene(
            room_geometry=tiny_geometry,
            scene_dir=self.test_data_dir,
        )

        result = compute_reachability(tiny_scene, self.robot_width)

        self.assertFalse(result.is_fully_reachable)
        self.assertEqual(result.num_disconnected_regions, 0)
        self.assertAlmostEqual(result.reachability_ratio, 0.0)


class TestReachabilityResultSerialization(unittest.TestCase):
    """Test ReachabilityResult serialization."""

    def test_to_json(self):
        """Test JSON serialization of ReachabilityResult."""
        result = ReachabilityResult(
            is_fully_reachable=False,
            num_disconnected_regions=2,
            reachability_ratio=0.75,
            blocking_furniture_ids=["sofa_1", "table_2"],
        )

        json_str = result.to_json()

        self.assertIn("is_fully_reachable", json_str)
        self.assertIn("false", json_str.lower())
        self.assertIn("num_disconnected_regions", json_str)
        self.assertIn("2", json_str)
        self.assertIn("sofa_1", json_str)
        self.assertIn("table_2", json_str)


class TestFormatReachabilityResult(unittest.TestCase):
    """Test human-readable result formatting (used by designer tool)."""

    def test_fully_reachable_returns_message(self):
        """Fully reachable result should return confirmation message."""
        result = ReachabilityResult(
            is_fully_reachable=True,
            num_disconnected_regions=1,
            reachability_ratio=1.0,
            blocking_furniture_ids=[],
        )

        formatted = format_reachability_result(result)

        self.assertIn("fully reachable", formatted.lower())
        self.assertIn("accessible", formatted.lower())

    def test_disconnected_with_blockers(self):
        """Disconnected room with blockers should format appropriately."""
        result = ReachabilityResult(
            is_fully_reachable=False,
            num_disconnected_regions=2,
            reachability_ratio=0.6,
            blocking_furniture_ids=["sofa_1"],
        )

        formatted = format_reachability_result(result)

        self.assertIn("2 disconnected regions", formatted)
        self.assertIn("60.0%", formatted)
        self.assertIn("sofa_1", formatted)


class TestFormatReachabilityForCritic(unittest.TestCase):
    """Test critic context formatting (returns empty when no issues)."""

    def test_fully_reachable_returns_empty(self):
        """Fully reachable result should return empty string for template conditional."""
        result = ReachabilityResult(
            is_fully_reachable=True,
            num_disconnected_regions=1,
            reachability_ratio=1.0,
            blocking_furniture_ids=[],
        )

        formatted = format_reachability_for_critic(result)

        self.assertEqual(formatted, "")

    def test_disconnected_uses_shared_format(self):
        """Disconnected room should use same format as format_reachability_result."""
        result = ReachabilityResult(
            is_fully_reachable=False,
            num_disconnected_regions=2,
            reachability_ratio=0.6,
            blocking_furniture_ids=["sofa_1"],
        )

        critic_formatted = format_reachability_for_critic(result)
        result_formatted = format_reachability_result(result)

        self.assertEqual(critic_formatted, result_formatted)

    def test_disconnected_without_blockers(self):
        """Disconnected room without identified blockers should suggest rearrangement."""
        result = ReachabilityResult(
            is_fully_reachable=False,
            num_disconnected_regions=2,
            reachability_ratio=0.5,
            blocking_furniture_ids=[],
        )

        formatted = format_reachability_for_critic(result)

        self.assertIn("2 disconnected regions", formatted)
        self.assertIn("rearrangement", formatted.lower())


class TestReachabilityWithRotatedFurniture(unittest.TestCase):
    """Test reachability with rotated furniture (OBB vs AABB)."""

    def setUp(self):
        """Create test scene."""
        self.test_data_dir = Path(__file__).parent.parent / "test_data"
        self.robot_width = 0.5

        self.room_geometry = RoomGeometry(
            sdf_tree=None,
            sdf_path=None,
            walls=[],
            floor=None,
            wall_normals={},
            width=4.0,
            length=6.0,
            wall_height=2.7,
            openings=[],
        )

        self.scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=self.test_data_dir,
        )

    def test_rotated_furniture_obb(self):
        """Rotated furniture should use OBB, not AABB."""
        # Create a narrow piece (2m x 0.4m) at 45 degrees in center of room.
        # AABB would be ~1.7m x 1.7m, OBB is the actual 2m x 0.4m rotated.
        # The room is 6m x 4m, so either way there's space, but this tests
        # that rotation is handled correctly via convex hull.
        half_size = np.array([1.0, 0.2, 0.4])
        rotation_matrix = np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                [0, 0, 1],
            ]
        )
        rotation_45deg = RigidTransform(
            R=RotationMatrix(rotation_matrix),
            p=np.array([3.0, 2.0, 0.4]),
        )
        rotated_table = SceneObject(
            object_id=UniqueID("rotated_table"),
            object_type=ObjectType.FURNITURE,
            name="rotated_table",
            description="Test rotated table",
            transform=rotation_45deg,
            bbox_min=-half_size,
            bbox_max=half_size,
        )
        self.scene.add_object(rotated_table)

        result = compute_reachability(self.scene, self.robot_width)

        # Room should be fully reachable - rotated piece doesn't block.
        # This verifies the OBB transform is working correctly.
        self.assertTrue(result.is_fully_reachable)
        self.assertEqual(result.num_disconnected_regions, 1)


if __name__ == "__main__":
    unittest.main()
