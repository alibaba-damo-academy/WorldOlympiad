"""Tests for house.py dataclass serialization."""

import unittest

from pathlib import Path

from scenecode.agent_utils.house import (
    ConnectionType,
    Door,
    HouseLayout,
    Opening,
    OpeningType,
    PlacedRoom,
    RoomMaterials,
    RoomSpec,
    Wall,
    WallDirection,
    Window,
)
from scenecode.utils.material import Material


class TestRoundTrip(unittest.TestCase):
    """Test round-trip serialization for house dataclasses."""

    def test_opening_round_trip(self) -> None:
        """Opening survives to_dict/from_dict."""
        original = Opening(
            opening_id="opening_1",
            opening_type=OpeningType.WINDOW,
            position_along_wall=0.6,
            width=1.5,
            height=1.2,
            sill_height=0.9,
        )
        restored = Opening.from_dict(original.to_dict())
        assert restored.opening_id == original.opening_id
        assert restored.opening_type == original.opening_type
        assert restored.position_along_wall == original.position_along_wall
        assert restored.width == original.width
        assert restored.height == original.height
        assert restored.sill_height == original.sill_height

    def test_door_round_trip(self) -> None:
        """Door survives to_dict/from_dict."""
        original = Door(
            id="door_1",
            boundary_label="living_room|kitchen",
            position_segment=0.5,
            position_exact=2.5,
            door_type="interior",
            room_a="living_room",
            room_b="kitchen",
            width=0.9,
            height=2.1,
        )
        restored = Door.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.boundary_label == original.boundary_label
        assert restored.position_segment == original.position_segment
        assert restored.position_exact == original.position_exact
        assert restored.door_type == original.door_type
        assert restored.room_a == original.room_a
        assert restored.room_b == original.room_b
        assert restored.width == original.width
        assert restored.height == original.height

    def test_window_round_trip(self) -> None:
        """Window survives to_dict/from_dict."""
        original = Window(
            id="window_1",
            boundary_label="living_room|exterior",
            position_along_wall=0.6,
            room_id="living_room",
            wall_direction=WallDirection.NORTH,
            width=1.5,
            height=1.2,
            sill_height=0.9,
        )
        restored = Window.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.boundary_label == original.boundary_label
        assert restored.position_along_wall == original.position_along_wall
        assert restored.room_id == original.room_id
        assert restored.wall_direction == original.wall_direction
        assert restored.width == original.width
        assert restored.height == original.height
        assert restored.sill_height == original.sill_height

    def test_room_materials_round_trip(self) -> None:
        """RoomMaterials survives to_dict/from_dict."""
        original = RoomMaterials(
            wall_material=Material.from_path(Path("materials/plaster")),
            floor_material=Material.from_path(Path("materials/wood")),
        )
        restored = RoomMaterials.from_dict(original.to_dict())
        assert restored.wall_material == original.wall_material
        assert restored.floor_material == original.floor_material

    def test_wall_round_trip(self) -> None:
        """Wall survives to_dict/from_dict."""
        original = Wall(
            wall_id="living_room_north",
            room_id="living_room",
            direction=WallDirection.NORTH,
            start_point=(0.0, 6.0),
            end_point=(5.0, 6.0),
            length=5.0,
            is_exterior=True,
            faces_rooms=["kitchen"],
            openings=[
                Opening(
                    opening_id="opening_1",
                    opening_type=OpeningType.WINDOW,
                    position_along_wall=0.6,
                    width=1.5,
                    height=1.2,
                    sill_height=0.9,
                ),
            ],
        )
        restored = Wall.from_dict(original.to_dict())
        assert restored.wall_id == original.wall_id
        assert restored.room_id == original.room_id
        assert restored.direction == original.direction
        assert restored.start_point == original.start_point
        assert restored.end_point == original.end_point
        assert restored.length == original.length
        assert restored.is_exterior == original.is_exterior
        assert restored.faces_rooms == original.faces_rooms
        assert len(restored.openings) == 1
        assert restored.openings[0].opening_id == original.openings[0].opening_id

    def test_placed_room_round_trip(self) -> None:
        """PlacedRoom survives to_dict/from_dict."""
        original = PlacedRoom(
            room_id="living_room",
            position=(1.0, 2.0),
            width=5.0,
            depth=6.0,
            walls=[
                Wall(
                    wall_id="living_room_north",
                    room_id="living_room",
                    direction=WallDirection.NORTH,
                    start_point=(0.0, 6.0),
                    end_point=(5.0, 6.0),
                    length=5.0,
                    is_exterior=True,
                    faces_rooms=[],
                    openings=[],
                ),
            ],
        )
        restored = PlacedRoom.from_dict(original.to_dict())
        assert restored.room_id == original.room_id
        assert restored.position == original.position
        assert restored.width == original.width
        assert restored.depth == original.depth
        assert len(restored.walls) == 1
        assert restored.walls[0].wall_id == original.walls[0].wall_id

    def test_room_spec_round_trip(self) -> None:
        """RoomSpec survives to_dict/from_dict."""
        original = RoomSpec(
            room_id="living_room",
            room_type="living_room",
            prompt="A cozy living room",
            position=(1.0, 2.0),
            width=5.0,
            length=6.0,
            connections={
                "kitchen": ConnectionType.DOOR,
                "dining_room": ConnectionType.OPEN,
            },
        )
        restored = RoomSpec.from_dict(original.to_dict())
        assert restored.room_id == original.room_id
        assert restored.room_type == original.room_type
        assert restored.prompt == original.prompt
        assert restored.position == original.position
        assert restored.width == original.width
        assert restored.length == original.length
        assert restored.connections == original.connections

    def test_house_layout_round_trip(self) -> None:
        """HouseLayout with nested objects survives to_dict/from_dict."""
        original = HouseLayout(
            wall_height=2.8,
            room_specs=[
                RoomSpec(
                    room_id="living_room",
                    room_type="living_room",
                    prompt="A cozy living room",
                    position=(0.0, 0.0),
                    width=5.0,
                    length=6.0,
                    connections={"kitchen": ConnectionType.DOOR},
                ),
            ],
            doors=[
                Door(
                    id="door_1",
                    boundary_label="living_room|kitchen",
                    position_segment=0.5,
                    position_exact=2.5,
                    door_type="interior",
                    room_a="living_room",
                    room_b="kitchen",
                    width=0.9,
                    height=2.1,
                ),
            ],
            windows=[
                Window(
                    id="window_1",
                    boundary_label="living_room|exterior",
                    position_along_wall=0.6,
                    room_id="living_room",
                    wall_direction=WallDirection.NORTH,
                    width=1.5,
                    height=1.2,
                    sill_height=0.9,
                ),
            ],
            room_materials={
                "living_room": RoomMaterials(
                    wall_material=Material.from_path(Path("materials/plaster")),
                    floor_material=Material.from_path(Path("materials/wood")),
                ),
            },
            exterior_material=Material.from_path(Path("materials/brick")),
            placed_rooms=[
                PlacedRoom(
                    room_id="living_room",
                    position=(0.0, 0.0),
                    width=5.0,
                    depth=6.0,
                    walls=[
                        Wall(
                            wall_id="living_room_north",
                            room_id="living_room",
                            direction=WallDirection.NORTH,
                            start_point=(0.0, 6.0),
                            end_point=(5.0, 6.0),
                            length=5.0,
                            is_exterior=True,
                            faces_rooms=[],
                            openings=[],
                        ),
                    ],
                ),
            ],
            placement_valid=True,
            connectivity_valid=True,
            boundary_labels={
                "living_room|kitchen": ("living_room", "kitchen"),
            },
        )

        restored = HouseLayout.from_dict(original.to_dict())

        assert restored.wall_height == original.wall_height
        assert restored.placement_valid == original.placement_valid
        assert restored.connectivity_valid == original.connectivity_valid
        assert restored.exterior_material == original.exterior_material
        assert len(restored.room_specs) == 1
        assert restored.room_specs[0].room_id == original.room_specs[0].room_id
        assert len(restored.doors) == 1
        assert restored.doors[0].id == original.doors[0].id
        assert len(restored.windows) == 1
        assert restored.windows[0].id == original.windows[0].id
        assert "living_room" in restored.room_materials
        assert len(restored.placed_rooms) == 1
        assert restored.placed_rooms[0].room_id == original.placed_rooms[0].room_id
        assert restored.boundary_labels == original.boundary_labels


if __name__ == "__main__":
    unittest.main()
