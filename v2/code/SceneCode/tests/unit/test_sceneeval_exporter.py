"""Tests for SceneEvalExporter house export functionality."""

import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from pydrake.all import RigidTransform, RollPitchYaw

from scenecode.agent_utils.house import (
    HouseLayout,
    HouseScene,
    Opening,
    OpeningType,
    PlacedRoom,
    RoomSpec,
    Wall,
    WallDirection,
)
from scenecode.agent_utils.room import serialize_rigid_transform
from scenecode.agent_utils.sceneeval_exporter import (
    SceneEvalExportConfig,
    SceneEvalExporter,
)


def _create_minimal_room_scene(room_id: str, scene_dir: Path) -> MagicMock:
    """Create a minimal mock RoomScene for testing."""
    room = MagicMock()
    room.scene_dir = scene_dir
    room.objects = {}  # Empty objects for minimal test.
    return room


def _create_two_room_layout() -> HouseLayout:
    """Create a minimal two-room house layout for testing."""
    # Room specs: living room (4x5m) and kitchen (3x4m).
    living_spec = RoomSpec(
        room_id="living_room",
        room_type="living_room",
        length=4.0,
        width=5.0,
    )
    kitchen_spec = RoomSpec(
        room_id="kitchen",
        room_type="kitchen",
        length=3.0,
        width=4.0,
    )

    # Create walls for living room (4x5m).
    living_walls = [
        Wall(
            wall_id="living_room_north",
            room_id="living_room",
            direction=WallDirection.NORTH,
            start_point=(0.0, 5.0),
            end_point=(4.0, 5.0),
            length=4.0,
            is_exterior=True,
        ),
        Wall(
            wall_id="living_room_south",
            room_id="living_room",
            direction=WallDirection.SOUTH,
            start_point=(0.0, 0.0),
            end_point=(4.0, 0.0),
            length=4.0,
            is_exterior=True,
        ),
        Wall(
            wall_id="living_room_east",
            room_id="living_room",
            direction=WallDirection.EAST,
            start_point=(4.0, 0.0),
            end_point=(4.0, 5.0),
            length=5.0,
            is_exterior=False,
            faces_rooms=["kitchen"],
            openings=[
                Opening(
                    opening_id="door_living_kitchen",
                    opening_type=OpeningType.DOOR,
                    position_along_wall=1.5,
                    width=0.9,
                    height=2.1,
                    sill_height=0.0,
                )
            ],
        ),
        Wall(
            wall_id="living_room_west",
            room_id="living_room",
            direction=WallDirection.WEST,
            start_point=(0.0, 0.0),
            end_point=(0.0, 5.0),
            length=5.0,
            is_exterior=True,
        ),
    ]

    # Create walls for kitchen (3x4m), positioned at x=4.0.
    kitchen_walls = [
        Wall(
            wall_id="kitchen_north",
            room_id="kitchen",
            direction=WallDirection.NORTH,
            start_point=(0.0, 4.0),
            end_point=(3.0, 4.0),
            length=3.0,
            is_exterior=True,
        ),
        Wall(
            wall_id="kitchen_south",
            room_id="kitchen",
            direction=WallDirection.SOUTH,
            start_point=(0.0, 0.0),
            end_point=(3.0, 0.0),
            length=3.0,
            is_exterior=True,
        ),
        Wall(
            wall_id="kitchen_east",
            room_id="kitchen",
            direction=WallDirection.EAST,
            start_point=(3.0, 0.0),
            end_point=(3.0, 4.0),
            length=4.0,
            is_exterior=True,
        ),
        Wall(
            wall_id="kitchen_west",
            room_id="kitchen",
            direction=WallDirection.WEST,
            start_point=(0.0, 0.0),
            end_point=(0.0, 4.0),
            length=4.0,
            is_exterior=False,
            faces_rooms=["living_room"],
            openings=[
                Opening(
                    opening_id="door_living_kitchen",
                    opening_type=OpeningType.DOOR,
                    position_along_wall=1.5,
                    width=0.9,
                    height=2.1,
                    sill_height=0.0,
                )
            ],
        ),
    ]

    # Placed rooms with positions.
    # Convention: PlacedRoom.width = X dimension, PlacedRoom.depth = Y dimension.
    living_placed = PlacedRoom(
        room_id="living_room",
        position=(0.0, 0.0),
        width=4.0,  # X spans 0.0 to 4.0 (from wall definitions).
        depth=5.0,  # Y spans 0.0 to 5.0 (from wall definitions).
        walls=living_walls,
    )
    kitchen_placed = PlacedRoom(
        room_id="kitchen",
        position=(4.0, 0.0),  # Kitchen to the east of living room.
        width=3.0,  # X spans 0.0 to 3.0 local, 4.0 to 7.0 world.
        depth=4.0,  # Y spans 0.0 to 4.0.
        walls=kitchen_walls,
    )

    return HouseLayout(
        room_specs=[living_spec, kitchen_spec],
        placed_rooms=[living_placed, kitchen_placed],
        wall_height=2.7,
    )


class TestExportHouse(unittest.TestCase):
    """Test SceneEvalExporter.export_house() functionality."""

    def test_export_house_creates_valid_json(self) -> None:
        """export_house creates valid SceneEval JSON with correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            scene_dir = output_dir / "rooms"
            scene_dir.mkdir()

            # Create layout and house.
            layout = _create_two_room_layout()
            layout.house_dir = output_dir

            house = HouseScene(layout=layout, rooms={})

            # Add mock room scenes.
            house.rooms["living_room"] = _create_minimal_room_scene(
                "living_room", scene_dir
            )
            house.rooms["kitchen"] = _create_minimal_room_scene("kitchen", scene_dir)

            # Export.
            config = SceneEvalExportConfig(floor_thickness=0.15, wall_thickness=0.1)
            output_path = SceneEvalExporter.export_house(house, output_dir, config)

            # Verify file was created.
            assert output_path.exists()
            assert output_path.name == "sceneeval_state.json"

            # Load and verify structure.
            with open(output_path) as f:
                data = json.load(f)

            # Top-level structure.
            assert data["format"] == "sceneState"
            assert "scene" in data

            scene = data["scene"]
            assert scene["version"] == "scene@1.0.2"
            assert scene["unit"] == 1.0
            assert scene["up"] == [0, 0, 1]
            assert scene["front"] == [0, 1, 0]

    def test_export_house_architecture_has_both_rooms(self) -> None:
        """Architecture includes floor and walls for both rooms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            scene_dir = output_dir / "rooms"
            scene_dir.mkdir()

            layout = _create_two_room_layout()
            layout.house_dir = output_dir

            house = HouseScene(layout=layout, rooms={})
            house.rooms["living_room"] = _create_minimal_room_scene(
                "living_room", scene_dir
            )
            house.rooms["kitchen"] = _create_minimal_room_scene("kitchen", scene_dir)

            config = SceneEvalExportConfig(floor_thickness=0.15)
            output_path = SceneEvalExporter.export_house(house, output_dir, config)

            with open(output_path) as f:
                data = json.load(f)

            arch = data["scene"]["arch"]

            # Should have 2 floors + 8 walls = 10 elements.
            elements = arch["elements"]
            floors = [e for e in elements if e["type"] == "Floor"]
            walls = [e for e in elements if e["type"] == "Wall"]

            assert len(floors) == 2
            assert len(walls) == 8

            # Verify floor IDs.
            floor_ids = {f["id"] for f in floors}
            assert "floor|living_room" in floor_ids
            assert "floor|kitchen" in floor_ids

            # Verify floor thickness from config.
            for floor in floors:
                assert floor["depth"] == 0.15

            # Verify regions for both rooms.
            regions = arch["regions"]
            region_ids = {r["id"] for r in regions}
            assert "living_room" in region_ids
            assert "kitchen" in region_ids

    def test_export_house_room_positions_applied(self) -> None:
        """Room positions are correctly applied to floor polygons."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            scene_dir = output_dir / "rooms"
            scene_dir.mkdir()

            layout = _create_two_room_layout()
            layout.house_dir = output_dir

            house = HouseScene(layout=layout, rooms={})
            house.rooms["living_room"] = _create_minimal_room_scene(
                "living_room", scene_dir
            )
            house.rooms["kitchen"] = _create_minimal_room_scene("kitchen", scene_dir)

            config = SceneEvalExportConfig()
            output_path = SceneEvalExporter.export_house(house, output_dir, config)

            with open(output_path) as f:
                data = json.load(f)

            elements = data["scene"]["arch"]["elements"]

            # Find kitchen floor.
            kitchen_floor = next(e for e in elements if e["id"] == "floor|kitchen")
            points = kitchen_floor["points"]

            # Kitchen is at position (4.0, 0.0) with dimensions 3x4.
            # Floor polygon should be offset by room position.
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            # X should range from 4.0 to 7.0 (position.x + length).
            self.assertAlmostEqual(min(x_coords), 4.0)
            self.assertAlmostEqual(max(x_coords), 7.0)

            # Y should range from 0.0 to 4.0 (position.y + width).
            self.assertAlmostEqual(min(y_coords), 0.0)
            self.assertAlmostEqual(max(y_coords), 4.0)

    def test_export_house_door_becomes_hole(self) -> None:
        """Doors in walls are exported as holes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            scene_dir = output_dir / "rooms"
            scene_dir.mkdir()

            layout = _create_two_room_layout()
            layout.house_dir = output_dir

            house = HouseScene(layout=layout, rooms={})
            house.rooms["living_room"] = _create_minimal_room_scene(
                "living_room", scene_dir
            )
            house.rooms["kitchen"] = _create_minimal_room_scene("kitchen", scene_dir)

            config = SceneEvalExportConfig()
            output_path = SceneEvalExporter.export_house(house, output_dir, config)

            with open(output_path) as f:
                data = json.load(f)

            elements = data["scene"]["arch"]["elements"]
            walls = [e for e in elements if e["type"] == "Wall"]

            # Find walls with holes (doors).
            walls_with_holes = [w for w in walls if w.get("holes")]
            assert len(walls_with_holes) == 2  # One on each side of the door.

            # Verify hole structure.
            hole = walls_with_holes[0]["holes"][0]
            assert hole["type"] == "Door"
            assert "box" in hole
            assert "min" in hole["box"]
            assert "max" in hole["box"]

    def test_single_room_house_object_positions_offset_by_center(self) -> None:
        """Single-room house objects are offset by room center to match architecture.

        Architecture uses corner-based coordinates (0,0 to width,depth).
        Objects are stored in room-local center-based coordinates.
        This test verifies single-room houses correctly offset objects by room center.
        """
        from pydrake.math import RigidTransform, RotationMatrix

        from scenecode.agent_utils.room import ObjectType

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            scene_dir = output_dir / "room"
            scene_dir.mkdir()

            # Create a 10x8 single-room layout.
            room_spec = RoomSpec(
                room_id="gallery", room_type="gallery", length=10.0, width=8.0
            )

            walls = [
                Wall(
                    wall_id="gallery_north",
                    room_id="gallery",
                    direction=WallDirection.NORTH,
                    start_point=(0.0, 8.0),
                    end_point=(10.0, 8.0),
                    length=10.0,
                    is_exterior=True,
                ),
                Wall(
                    wall_id="gallery_south",
                    room_id="gallery",
                    direction=WallDirection.SOUTH,
                    start_point=(0.0, 0.0),
                    end_point=(10.0, 0.0),
                    length=10.0,
                    is_exterior=True,
                ),
                Wall(
                    wall_id="gallery_east",
                    room_id="gallery",
                    direction=WallDirection.EAST,
                    start_point=(10.0, 0.0),
                    end_point=(10.0, 8.0),
                    length=8.0,
                    is_exterior=True,
                ),
                Wall(
                    wall_id="gallery_west",
                    room_id="gallery",
                    direction=WallDirection.WEST,
                    start_point=(0.0, 0.0),
                    end_point=(0.0, 8.0),
                    length=8.0,
                    is_exterior=True,
                ),
            ]

            placed_room = PlacedRoom(
                room_id="gallery",
                position=(0.0, 0.0),
                width=10.0,
                depth=8.0,
                walls=walls,
            )

            layout = HouseLayout(
                room_specs=[room_spec],
                placed_rooms=[placed_room],
                wall_height=3.0,
            )
            layout.house_dir = output_dir

            # Create mock room with an object at room-local position (1, 2, 0.5).
            # Room center is (5, 4), so world position should be (6, 6, 0.5).
            room = MagicMock()
            room.scene_dir = scene_dir
            room.room_geometry = MagicMock()
            room.room_geometry.walls = []

            # Create mock object with transform at room-local (1, 2, 0.5).
            mock_obj = MagicMock()
            mock_obj.object_id = "test_table"
            mock_obj.object_type = ObjectType.FURNITURE
            mock_obj.transform = RigidTransform(
                R=RotationMatrix.Identity(), p=[1.0, 2.0, 0.5]
            )
            mock_obj.sdf_path = None
            mock_obj.metadata = {}

            room.objects = {"test_table": mock_obj}

            house = HouseScene(layout=layout, rooms={})
            house.rooms["gallery"] = room

            # Export.
            config = SceneEvalExportConfig()
            output_path = SceneEvalExporter.export_house(house, output_dir, config)

            with open(output_path) as f:
                data = json.load(f)

            # Verify object position is offset by room center (5, 4).
            objects = data["scene"]["object"]
            assert len(objects) == 1

            obj = objects[0]
            assert obj["id"] == "test_table"

            # Transform is column-major 4x4 matrix.
            # Indices 12, 13, 14 are x, y, z translation.
            matrix = obj["transform"]["data"]
            world_x = matrix[12]
            world_y = matrix[13]
            world_z = matrix[14]

            # Room-local (1, 2, 0.5) + center offset (5, 4, 0) = world (6, 6, 0.5).
            self.assertAlmostEqual(world_x, 6.0, places=5)
            self.assertAlmostEqual(world_y, 6.0, places=5)
            self.assertAlmostEqual(world_z, 0.5, places=5)


class TestSingleRoomExportWithHoles(unittest.TestCase):
    """Test SceneEvalExporter per-room export with doors/windows."""

    def test_get_wall_openings_maps_left_to_west(self) -> None:
        """_get_wall_openings correctly maps left_wall to west direction."""
        # Create HouseLayout with window on west wall.
        room_spec = RoomSpec(
            room_id="main", room_type="living_room", length=4.0, width=5.0
        )

        walls = [
            Wall(
                wall_id="main_west",
                room_id="main",
                direction=WallDirection.WEST,
                start_point=(0.0, 0.0),
                end_point=(0.0, 5.0),
                length=5.0,
                is_exterior=True,
                openings=[
                    Opening(
                        opening_id="window_main",
                        opening_type=OpeningType.WINDOW,
                        position_along_wall=2.0,
                        width=1.2,
                        height=1.0,
                        sill_height=0.9,
                    )
                ],
            ),
        ]

        placed_room = PlacedRoom(
            room_id="main", position=(0.0, 0.0), width=4.0, depth=5.0, walls=walls
        )

        house_layout = HouseLayout(
            room_specs=[room_spec], placed_rooms=[placed_room], wall_height=2.7
        )

        # Create minimal exporter with house_layout.
        config = SceneEvalExportConfig()
        exporter = SceneEvalExporter.__new__(SceneEvalExporter)
        exporter.house_layout = house_layout
        exporter.config = config

        # Query for left_wall (should map to west and find the window).
        holes = exporter._get_wall_openings("left_wall")
        self.assertEqual(len(holes), 1)
        self.assertEqual(holes[0]["type"], "Window")

    def test_get_wall_openings_maps_right_to_east(self) -> None:
        """_get_wall_openings correctly maps right_wall to east direction."""
        # Create HouseLayout with door on east wall.
        room_spec = RoomSpec(
            room_id="main", room_type="living_room", length=4.0, width=5.0
        )

        walls = [
            Wall(
                wall_id="main_east",
                room_id="main",
                direction=WallDirection.EAST,
                start_point=(4.0, 0.0),
                end_point=(4.0, 5.0),
                length=5.0,
                is_exterior=True,
                openings=[
                    Opening(
                        opening_id="door_main",
                        opening_type=OpeningType.DOOR,
                        position_along_wall=2.0,
                        width=0.9,
                        height=2.1,
                        sill_height=0.0,
                    )
                ],
            ),
        ]

        placed_room = PlacedRoom(
            room_id="main", position=(0.0, 0.0), width=4.0, depth=5.0, walls=walls
        )

        house_layout = HouseLayout(
            room_specs=[room_spec], placed_rooms=[placed_room], wall_height=2.7
        )

        config = SceneEvalExportConfig()
        exporter = SceneEvalExporter.__new__(SceneEvalExporter)
        exporter.house_layout = house_layout
        exporter.config = config

        # Query for right_wall (should map to east and find the door).
        holes = exporter._get_wall_openings("right_wall")
        self.assertEqual(len(holes), 1)
        self.assertEqual(holes[0]["type"], "Door")

    def test_get_wall_openings_maps_front_to_north(self) -> None:
        """_get_wall_openings correctly maps front_wall to north direction."""
        room_spec = RoomSpec(
            room_id="main", room_type="living_room", length=4.0, width=5.0
        )

        walls = [
            Wall(
                wall_id="main_north",
                room_id="main",
                direction=WallDirection.NORTH,
                start_point=(0.0, 5.0),
                end_point=(4.0, 5.0),
                length=4.0,
                is_exterior=True,
                openings=[
                    Opening(
                        opening_id="window_main",
                        opening_type=OpeningType.WINDOW,
                        position_along_wall=1.5,
                        width=1.5,
                        height=1.2,
                        sill_height=0.8,
                    )
                ],
            ),
        ]

        placed_room = PlacedRoom(
            room_id="main", position=(0.0, 0.0), width=4.0, depth=5.0, walls=walls
        )

        house_layout = HouseLayout(
            room_specs=[room_spec], placed_rooms=[placed_room], wall_height=2.7
        )

        config = SceneEvalExportConfig()
        exporter = SceneEvalExporter.__new__(SceneEvalExporter)
        exporter.house_layout = house_layout
        exporter.config = config

        # Query for front_wall (should map to north and find the window).
        holes = exporter._get_wall_openings("front_wall")
        self.assertEqual(len(holes), 1)
        self.assertEqual(holes[0]["type"], "Window")

    def test_get_wall_openings_maps_back_to_south(self) -> None:
        """_get_wall_openings correctly maps back_wall to south direction."""
        room_spec = RoomSpec(
            room_id="main", room_type="living_room", length=4.0, width=5.0
        )

        walls = [
            Wall(
                wall_id="main_south",
                room_id="main",
                direction=WallDirection.SOUTH,
                start_point=(0.0, 0.0),
                end_point=(4.0, 0.0),
                length=4.0,
                is_exterior=True,
                openings=[
                    Opening(
                        opening_id="door_main",
                        opening_type=OpeningType.DOOR,
                        position_along_wall=2.0,
                        width=0.9,
                        height=2.1,
                        sill_height=0.0,
                    )
                ],
            ),
        ]

        placed_room = PlacedRoom(
            room_id="main", position=(0.0, 0.0), width=4.0, depth=5.0, walls=walls
        )

        house_layout = HouseLayout(
            room_specs=[room_spec],
            placed_rooms=[placed_room],
            wall_height=2.7,
        )

        config = SceneEvalExportConfig()
        exporter = SceneEvalExporter.__new__(SceneEvalExporter)
        exporter.house_layout = house_layout
        exporter.config = config

        # Query for back_wall (should map to south and find the door).
        holes = exporter._get_wall_openings("back_wall")
        self.assertEqual(len(holes), 1)
        self.assertEqual(holes[0]["type"], "Door")

    def test_get_wall_openings_no_house_layout_returns_empty(self) -> None:
        """_get_wall_openings returns empty list when no house_layout."""
        config = SceneEvalExportConfig()
        exporter = SceneEvalExporter.__new__(SceneEvalExporter)
        exporter.house_layout = None
        exporter.config = config

        # Without house_layout, should return empty.
        self.assertEqual(exporter._get_wall_openings("left_wall"), [])
        self.assertEqual(exporter._get_wall_openings("right_wall"), [])


class TestSceneEvalExporterCompositeMembers(unittest.TestCase):
    """Regression tests for composite articulated member export."""

    def test_build_member_object_uses_effective_transform(self):
        exporter = SceneEvalExporter.__new__(SceneEvalExporter)
        exporter.config = SceneEvalExportConfig(asset_id_prefix="scenecode")
        exporter.scene_dir = Path("/tmp/scene")

        member = {
            "asset_id": "wardrobe_0",
            "name": "Wardrobe",
            "sdf_path": "/tmp/scene/generated_assets/wardrobe.sdf",
            "transform": serialize_rigid_transform(
                RigidTransform(p=[1.0, 2.0, 3.0])
            ),
            "internal_model_pose": serialize_rigid_transform(
                RigidTransform(RollPitchYaw(0.0, 0.0, np.pi), [0.0, -0.25, 0.0])
            ),
        }

        obj = exporter._build_member_object(member=member, index=0)

        matrix = np.array(obj["transform"]["data"]).reshape((4, 4), order="F")
        np.testing.assert_array_almost_equal(matrix[:3, 3], [1.0, 1.75, 3.0])


if __name__ == "__main__":
    unittest.main()
