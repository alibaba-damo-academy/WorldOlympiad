"""Unit tests for clearance zone validation functions."""

import unittest

from pathlib import Path

import numpy as np

from pydrake.all import RigidTransform

from scenecode.agent_utils.clearance_zones import (
    compute_door_clearance_violations,
    compute_open_connection_blocked_violations,
    compute_window_clearance_violations,
)
from scenecode.agent_utils.house import ClearanceOpeningData, RoomGeometry
from scenecode.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID


class TestClearanceZonesWallFiltering(unittest.TestCase):
    """Test that walls/floors are correctly filtered from clearance checks."""

    def setUp(self):
        """Create test scene with walls and openings."""
        self.test_data_dir = Path(__file__).parent.parent / "test_data"

        # Create a window opening on the north wall.
        self.window_opening = ClearanceOpeningData(
            opening_id="window_1",
            opening_type="window",
            wall_direction="north",
            center_world=[0.0, 2.0, 1.5],
            width=1.2,
            sill_height=1.0,
            height=1.2,
            clearance_bbox_min=[-0.6, 1.5, 0.0],
            clearance_bbox_max=[0.6, 2.0, 2.7],
            wall_start=[-2.0, 2.0],
            wall_end=[2.0, 2.0],
            position_along_wall=1.4,
        )

        # Create a door opening on the east wall.
        self.door_opening = ClearanceOpeningData(
            opening_id="door_1",
            opening_type="door",
            wall_direction="east",
            center_world=[2.0, 0.0, 1.0],
            width=0.9,
            sill_height=0.0,
            height=2.1,
            clearance_bbox_min=[1.5, -0.45, 0.0],
            clearance_bbox_max=[2.0, 0.45, 2.1],
            wall_start=[2.0, -2.0],
            wall_end=[2.0, 2.0],
            position_along_wall=1.55,
        )

        # Create an open connection on the south wall.
        self.open_connection = ClearanceOpeningData(
            opening_id="open_1",
            opening_type="open",
            wall_direction="south",
            center_world=[0.0, -2.0, 1.35],
            width=2.0,
            sill_height=0.0,
            height=2.7,
            clearance_bbox_min=None,
            clearance_bbox_max=None,
            wall_start=[-2.0, -2.0],
            wall_end=[2.0, -2.0],
            position_along_wall=1.0,
        )

        # Create room geometry with openings.
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
            length=4.0,
            wall_height=2.7,
            openings=[self.window_opening, self.door_opening, self.open_connection],
        )

        self.scene = RoomScene(
            room_geometry=self.room_geometry,
            scene_dir=self.test_data_dir,
        )

    def _create_wall_object(self, name: str, position: np.ndarray) -> SceneObject:
        """Create a wall object at the specified position."""
        return SceneObject(
            object_id=UniqueID(name),
            object_type=ObjectType.WALL,
            name=name,
            description=f"Room {name}",
            transform=RigidTransform(p=position),
            bbox_min=np.array([-2.0, -0.025, -1.35]),
            bbox_max=np.array([2.0, 0.025, 1.35]),
            immutable=True,
        )

    def _create_floor_object(self) -> SceneObject:
        """Create a floor object."""
        return SceneObject(
            object_id=UniqueID("floor"),
            object_type=ObjectType.FLOOR,
            name="Floor",
            description="Floor surface",
            transform=RigidTransform(),
            bbox_min=np.array([-2.0, -2.0, -0.1]),
            bbox_max=np.array([2.0, 2.0, 0.0]),
            immutable=True,
        )

    def _create_furniture_object(
        self, name: str, position: np.ndarray, size: np.ndarray
    ) -> SceneObject:
        """Create a furniture object at the specified position."""
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

    def test_window_clearance_ignores_structural_elements(self):
        """Test that walls and floor are not flagged as blocking windows."""
        # Add north wall (contains the window) and floor.
        north_wall = self._create_wall_object(
            "north_wall", position=np.array([0.0, 2.0, 1.35])
        )
        floor = self._create_floor_object()
        self.scene.add_object(north_wall)
        self.scene.add_object(floor)

        violations = compute_window_clearance_violations(self.scene)

        structural_violations = [
            v
            for v in violations
            if "wall" in v.furniture_id or "floor" in v.furniture_id
        ]
        self.assertEqual(
            len(structural_violations),
            0,
            "Structural elements should not be flagged as blocking windows",
        )

    def test_window_clearance_detects_furniture(self):
        """Test that tall furniture IS flagged as blocking windows."""
        # Add tall cabinet in front of window (above sill height of 1.0m).
        cabinet = self._create_furniture_object(
            name="tall_cabinet",
            position=np.array([0.0, 1.7, 1.2]),  # Center at z=1.2m
            size=np.array([0.6, 0.4, 2.4]),  # Top at z=2.4m > sill 1.0m
        )
        self.scene.add_object(cabinet)

        violations = compute_window_clearance_violations(self.scene)

        cabinet_violations = [v for v in violations if "cabinet" in v.furniture_id]
        self.assertEqual(
            len(cabinet_violations),
            1,
            "Tall cabinet should be flagged as blocking window",
        )

    def test_door_clearance_ignores_structural_elements(self):
        """Test that walls and floor are not flagged as blocking doors."""
        # Add east wall (contains the door) and floor.
        east_wall = SceneObject(
            object_id=UniqueID("east_wall"),
            object_type=ObjectType.WALL,
            name="east_wall",
            description="Room east_wall",
            transform=RigidTransform(p=np.array([2.0, 0.0, 1.35])),
            bbox_min=np.array([-0.025, -2.0, -1.35]),
            bbox_max=np.array([0.025, 2.0, 1.35]),
            immutable=True,
        )
        floor = self._create_floor_object()
        self.scene.add_object(east_wall)
        self.scene.add_object(floor)

        violations = compute_door_clearance_violations(self.scene)

        structural_violations = [
            v
            for v in violations
            if "wall" in v.furniture_id or "floor" in v.furniture_id
        ]
        self.assertEqual(
            len(structural_violations),
            0,
            "Structural elements should not be flagged as blocking doors",
        )

    def test_door_clearance_detects_furniture(self):
        """Test that furniture in door clearance zone IS flagged."""
        # Add furniture blocking the door.
        blocking_furniture = self._create_furniture_object(
            name="blocking_shelf",
            position=np.array([1.7, 0.0, 1.0]),  # In door clearance zone
            size=np.array([0.4, 0.6, 2.0]),
        )
        self.scene.add_object(blocking_furniture)

        violations = compute_door_clearance_violations(self.scene)

        furniture_violations = [v for v in violations if "shelf" in v.furniture_id]
        self.assertEqual(
            len(furniture_violations),
            1,
            "Furniture in door clearance zone should be flagged",
        )

    def test_open_connection_ignores_structural_elements(self):
        """Test that walls and floor are not flagged as blocking open connections."""
        # Add south wall (contains the open connection) and floor.
        south_wall = self._create_wall_object(
            "south_wall", position=np.array([0.0, -2.0, 1.35])
        )
        floor = self._create_floor_object()
        self.scene.add_object(south_wall)
        self.scene.add_object(floor)

        violations = compute_open_connection_blocked_violations(
            scene=self.scene,
            passage_size=0.5,
            open_connection_clearance=0.5,
        )

        structural_violations = [
            v
            for v in violations
            if any("wall" in fid or "floor" in fid for fid in v.blocking_furniture_ids)
        ]
        self.assertEqual(
            len(structural_violations),
            0,
            "Structural elements should not be flagged as blocking open connections",
        )

    def test_open_connection_detects_furniture(self):
        """Test that furniture blocking passage IS flagged."""
        # Add furniture blocking the entire open connection.
        blocking_furniture = self._create_furniture_object(
            name="blocking_sofa",
            position=np.array([0.0, -1.7, 0.4]),  # In open connection clearance
            size=np.array([2.5, 0.8, 0.8]),  # Wide enough to block passage
        )
        self.scene.add_object(blocking_furniture)

        violations = compute_open_connection_blocked_violations(
            scene=self.scene,
            passage_size=0.5,
            open_connection_clearance=0.5,
        )

        furniture_violations = [
            v
            for v in violations
            if any("sofa" in fid for fid in v.blocking_furniture_ids)
        ]
        self.assertEqual(
            len(furniture_violations),
            1,
            "Furniture blocking passage should be flagged",
        )


if __name__ == "__main__":
    unittest.main()
