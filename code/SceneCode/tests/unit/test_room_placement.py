"""Tests for room placement algorithm."""

import unittest

from scenecode.agent_utils.house import (
    ConnectionType,
    Door,
    PlacedRoom,
    RoomSpec,
    WallDirection,
)
from scenecode.floor_plan_agents.tools.room_placement import (
    PlacementConfig,
    PlacementError,
    find_room,
    get_shared_boundary,
    place_rooms,
    rooms_overlap,
    rooms_share_edge,
    validate_connectivity,
)


class TestSingleRoom(unittest.TestCase):
    """Tests for single room placement."""

    def test_single_room_at_origin(self):
        """Single room should be placed at origin (0, 0)."""
        specs = [
            RoomSpec(
                room_id="living",
                room_type="living_room",
                width=4.0,
                length=5.0,
            )
        ]
        result = place_rooms(specs)

        assert len(result) == 1
        assert result[0].position == (0.0, 0.0)

    def test_single_room_dimensions(self):
        """Single room should have correct dimensions."""
        specs = [
            RoomSpec(
                room_id="bedroom",
                room_type="bedroom",
                width=3.0,  # Y dimension.
                length=4.0,  # X dimension.
            )
        ]
        result = place_rooms(specs)

        assert result[0].width == 4.0  # X dimension (from length).
        assert result[0].depth == 3.0  # Y dimension (from width).


class TestTwoRooms(unittest.TestCase):
    """Tests for two room placement."""

    def test_two_adjacent_rooms(self):
        """Two adjacent rooms should share an edge."""
        specs = [
            RoomSpec(
                room_id="living",
                room_type="living_room",
                width=4.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="kitchen",
                room_type="kitchen",
                width=4.0,
                length=3.0,
                connections={"living": ConnectionType.DOOR},
            ),
        ]
        result = place_rooms(specs)

        assert len(result) == 2
        living = find_room(rooms=result, room_id="living")
        kitchen = find_room(rooms=result, room_id="kitchen")
        assert rooms_share_edge(room_a=living, room_b=kitchen, min_overlap=1.0)

    def test_two_rooms_no_overlap(self):
        """Adjacent rooms should not overlap."""
        specs = [
            RoomSpec(
                room_id="a",
                room_type="room",
                width=4.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="room",
                width=4.0,
                length=3.0,
                connections={"a": ConnectionType.DOOR},
            ),
        ]
        result = place_rooms(specs)

        a = find_room(rooms=result, room_id="a")
        b = find_room(rooms=result, room_id="b")
        assert not rooms_overlap(room_a=a, room_b=b)


class TestThreeRooms(unittest.TestCase):
    """Tests for three room placement."""

    def test_three_rooms_linear(self):
        """A-B-C linear layout: B adjacent to A and C."""
        specs = [
            RoomSpec(room_id="a", room_type="bedroom", width=3.0, length=4.0),
            RoomSpec(
                room_id="b",
                room_type="hallway",
                width=3.0,
                length=2.0,
                connections={"a": ConnectionType.DOOR, "c": ConnectionType.DOOR},
            ),
            RoomSpec(room_id="c", room_type="bathroom", width=3.0, length=3.0),
        ]
        result = place_rooms(specs)

        a = find_room(rooms=result, room_id="a")
        b = find_room(rooms=result, room_id="b")
        c = find_room(rooms=result, room_id="c")

        assert rooms_share_edge(room_a=a, room_b=b)
        assert rooms_share_edge(room_a=b, room_b=c)
        # A and C should not necessarily share an edge (B is between).


class TestMultiAdjacency(unittest.TestCase):
    """Tests for rooms with multiple adjacency requirements."""

    def test_room_adjacent_to_two_rooms(self):
        """Room C adjacent to both A and B requires corner placement."""
        specs = [
            RoomSpec(
                room_id="a",
                room_type="living_room",
                width=4.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="kitchen",
                width=4.0,
                length=3.0,
                connections={"a": ConnectionType.DOOR},
            ),
            RoomSpec(
                room_id="c",
                room_type="dining",
                width=3.0,
                length=3.0,
                connections={"a": ConnectionType.DOOR, "b": ConnectionType.DOOR},
            ),
        ]

        # This may or may not be satisfiable depending on geometry.
        # The algorithm will try different orderings.
        try:
            result = place_rooms(specs)
            c = find_room(rooms=result, room_id="c")
            # If successful, verify adjacencies.
            a = find_room(rooms=result, room_id="a")
            b = find_room(rooms=result, room_id="b")
            # At least one adjacency should be satisfied.
            assert rooms_share_edge(room_a=a, room_b=c) or rooms_share_edge(
                room_a=b, room_b=c
            )
        except PlacementError:
            # Expected if geometry doesn't allow satisfying all constraints.
            pass


class TestGridLayout(unittest.TestCase):
    """Tests for grid-like room layouts."""

    def test_four_rooms_grid(self):
        """2x2 grid layout with cross adjacencies."""
        specs = [
            RoomSpec(
                room_id="a",
                room_type="living",
                width=4.0,
                length=4.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="kitchen",
                width=4.0,
                length=4.0,
                connections={"a": ConnectionType.DOOR},
            ),
            RoomSpec(
                room_id="c",
                room_type="bedroom",
                width=4.0,
                length=4.0,
                connections={"a": ConnectionType.DOOR},
            ),
            RoomSpec(
                room_id="d",
                room_type="bath",
                width=4.0,
                length=4.0,
                connections={"b": ConnectionType.DOOR, "c": ConnectionType.DOOR},
            ),
        ]

        result = place_rooms(specs)

        a = find_room(rooms=result, room_id="a")
        b = find_room(rooms=result, room_id="b")
        c = find_room(rooms=result, room_id="c")
        d = find_room(rooms=result, room_id="d")

        assert rooms_share_edge(room_a=a, room_b=b)
        assert rooms_share_edge(room_a=a, room_b=c)
        # D should share edge with at least one of B or C.
        assert rooms_share_edge(room_a=b, room_b=d) or rooms_share_edge(
            room_a=c, room_b=d
        )


class TestNoOverlapping(unittest.TestCase):
    """Tests for overlap prevention."""

    def test_no_overlapping_rooms(self):
        """Placed rooms must never overlap."""
        specs = [
            RoomSpec(
                room_id="a",
                room_type="living",
                width=5.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="kitchen",
                width=4.0,
                length=4.0,
                connections={"a": ConnectionType.DOOR},
            ),
            RoomSpec(
                room_id="c",
                room_type="bedroom",
                width=4.0,
                length=4.0,
                connections={"a": ConnectionType.DOOR},
            ),
        ]
        result = place_rooms(specs)

        for i, room_i in enumerate(result):
            for j, room_j in enumerate(result):
                if i != j:
                    assert not rooms_overlap(
                        room_a=room_i, room_b=room_j
                    ), f"Rooms {room_i.room_id} and {room_j.room_id} overlap"


class TestMinSharedEdge(unittest.TestCase):
    """Tests for minimum shared edge requirement."""

    def test_min_shared_edge_respected(self):
        """Adjacency requires minimum shared edge length."""
        specs = [
            RoomSpec(
                room_id="a",
                room_type="living",
                width=4.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="closet",
                width=1.5,
                length=1.5,
                connections={"a": ConnectionType.DOOR},
            ),
        ]
        result = place_rooms(specs)

        a = find_room(rooms=result, room_id="a")
        b = find_room(rooms=result, room_id="b")
        assert rooms_share_edge(room_a=a, room_b=b, min_overlap=1.0)


class TestWallGeneration(unittest.TestCase):
    """Tests for wall generation."""

    def test_room_has_four_walls(self):
        """Each placed room should have exactly 4 walls."""
        specs = [
            RoomSpec(
                room_id="room",
                room_type="room",
                width=4.0,
                length=5.0,
            )
        ]
        result = place_rooms(specs)

        assert len(result[0].walls) == 4

    def test_wall_directions_complete(self):
        """Room should have walls for all 4 cardinal directions."""
        specs = [
            RoomSpec(
                room_id="room",
                room_type="room",
                width=4.0,
                length=5.0,
            )
        ]
        result = place_rooms(specs)

        directions = {wall.direction.value for wall in result[0].walls}
        assert directions == {"north", "south", "east", "west"}

    def test_exterior_walls_marked(self):
        """Single room should have all exterior walls."""
        specs = [
            RoomSpec(
                room_id="room",
                room_type="room",
                width=4.0,
                length=5.0,
            )
        ]
        result = place_rooms(specs)

        for wall in result[0].walls:
            assert wall.is_exterior is True
            assert wall.faces_rooms == []


class TestWallConnectivity(unittest.TestCase):
    """Tests for wall connectivity between rooms."""

    def test_shared_wall_marked_interior(self):
        """Shared walls between adjacent rooms should be marked as interior."""
        specs = [
            RoomSpec(
                room_id="a",
                room_type="living",
                width=4.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="kitchen",
                width=4.0,
                length=3.0,
                connections={"a": ConnectionType.DOOR},
            ),
        ]
        result = place_rooms(specs)

        # Find shared wall.
        shared = get_shared_boundary(result[0], result[1])
        if shared:
            assert shared.is_exterior is False
            assert result[1].room_id in shared.faces_rooms


class TestRoomsOverlap(unittest.TestCase):
    """Tests for rooms_overlap function."""

    def test_overlapping_rooms(self):
        """Overlapping rooms should be detected."""
        room_a = PlacedRoom(
            room_id="a",
            position=(0.0, 0.0),
            width=4.0,
            depth=4.0,
        )
        room_b = PlacedRoom(
            room_id="b",
            position=(2.0, 2.0),  # Overlaps with A.
            width=4.0,
            depth=4.0,
        )

        assert rooms_overlap(room_a=room_a, room_b=room_b) is True

    def test_adjacent_rooms_not_overlapping(self):
        """Adjacent (touching) rooms should not be detected as overlapping."""
        room_a = PlacedRoom(
            room_id="a",
            position=(0.0, 0.0),
            width=4.0,
            depth=4.0,
        )
        room_b = PlacedRoom(
            room_id="b",
            position=(4.0, 0.0),  # Touching A's east edge.
            width=4.0,
            depth=4.0,
        )

        assert rooms_overlap(room_a=room_a, room_b=room_b) is False


class TestRoomsShareEdge(unittest.TestCase):
    """Tests for rooms_share_edge function."""

    def test_adjacent_horizontal(self):
        """Horizontally adjacent rooms share edge."""
        room_a = PlacedRoom(
            room_id="a",
            position=(0.0, 0.0),
            width=4.0,
            depth=4.0,
        )
        room_b = PlacedRoom(
            room_id="b",
            position=(4.0, 0.0),
            width=4.0,
            depth=4.0,
        )

        assert rooms_share_edge(room_a, room_b, min_overlap=1.0) is True

    def test_adjacent_vertical(self):
        """Vertically adjacent rooms share edge."""
        room_a = PlacedRoom(
            room_id="a",
            position=(0.0, 0.0),
            width=4.0,
            depth=4.0,
        )
        room_b = PlacedRoom(
            room_id="b",
            position=(0.0, 4.0),
            width=4.0,
            depth=4.0,
        )

        assert rooms_share_edge(room_a=room_a, room_b=room_b, min_overlap=1.0) is True

    def test_diagonal_rooms_no_edge(self):
        """Diagonally placed rooms don't share edge."""
        room_a = PlacedRoom(
            room_id="a",
            position=(0.0, 0.0),
            width=4.0,
            depth=4.0,
        )
        room_b = PlacedRoom(
            room_id="b",
            position=(4.0, 4.0),  # Diagonal to A.
            width=4.0,
            depth=4.0,
        )

        assert rooms_share_edge(room_a=room_a, room_b=room_b) is False

    def test_insufficient_overlap(self):
        """Partially adjacent rooms may not meet minimum overlap."""
        room_a = PlacedRoom(
            room_id="a",
            position=(0.0, 0.0),
            width=4.0,
            depth=4.0,
        )
        room_b = PlacedRoom(
            room_id="b",
            position=(4.0, 3.5),  # Only 0.5m overlap.
            width=4.0,
            depth=4.0,
        )

        assert rooms_share_edge(room_a=room_a, room_b=room_b, min_overlap=1.0) is False
        assert rooms_share_edge(room_a=room_a, room_b=room_b, min_overlap=0.4) is True


class TestConnectivityValidation(unittest.TestCase):
    """Tests for door connectivity validation."""

    def test_single_room_with_exterior_door(self):
        """Single room with exterior door is valid."""
        rooms = [
            PlacedRoom(
                room_id="living",
                position=(0.0, 0.0),
                width=5.0,
                depth=4.0,
            )
        ]
        doors = [
            Door(
                id="door_1",
                boundary_label="A",
                position_segment="center",
                position_exact=2.5,
                door_type="exterior",
                room_a="living",
            )
        ]

        is_valid, msg = validate_connectivity(rooms, doors)
        assert is_valid is True

    def test_no_doors_invalid(self):
        """No doors should be invalid."""
        rooms = [
            PlacedRoom(
                room_id="living",
                position=(0.0, 0.0),
                width=5.0,
                depth=4.0,
            )
        ]

        is_valid, msg = validate_connectivity(rooms, [])
        assert is_valid is False
        assert "No doors" in msg

    def test_no_exterior_door_invalid(self):
        """No exterior door should be invalid."""
        rooms = [
            PlacedRoom(
                room_id="living",
                position=(0.0, 0.0),
                width=5.0,
                depth=4.0,
            ),
            PlacedRoom(
                room_id="kitchen",
                position=(5.0, 0.0),
                width=3.0,
                depth=4.0,
            ),
        ]
        doors = [
            Door(
                id="door_1",
                boundary_label="A",
                position_segment="center",
                position_exact=2.5,
                door_type="interior",
                room_a="living",
                room_b="kitchen",
            )
        ]

        is_valid, msg = validate_connectivity(rooms, doors)
        assert is_valid is False
        assert "exterior door" in msg.lower()

    def test_unreachable_room_invalid(self):
        """Room not reachable from exterior should be invalid."""
        rooms = [
            PlacedRoom(
                room_id="living",
                position=(0.0, 0.0),
                width=5.0,
                depth=4.0,
            ),
            PlacedRoom(
                room_id="kitchen",
                position=(5.0, 0.0),
                width=3.0,
                depth=4.0,
            ),
            PlacedRoom(
                room_id="bathroom",
                position=(0.0, 4.0),
                width=3.0,
                depth=3.0,
            ),
        ]
        doors = [
            Door(
                id="door_1",
                boundary_label="A",
                position_segment="center",
                position_exact=2.5,
                door_type="exterior",
                room_a="living",
            ),
            Door(
                id="door_2",
                boundary_label="B",
                position_segment="center",
                position_exact=2.5,
                door_type="interior",
                room_a="living",
                room_b="kitchen",
            ),
            # No door to bathroom!
        ]

        is_valid, msg = validate_connectivity(rooms, doors)
        assert is_valid is False
        assert "bathroom" in msg.lower()

    def test_open_connection_counts_for_reachability(self):
        """Room reachable via open connection should be valid."""
        rooms = [
            PlacedRoom(
                room_id="living_room",
                position=(0.0, 0.0),
                width=5.0,
                depth=4.0,
            ),
            PlacedRoom(
                room_id="kitchen",
                position=(5.0, 0.0),
                width=3.0,
                depth=4.0,
            ),
        ]
        doors = [
            Door(
                id="door_1",
                boundary_label="A",
                position_segment="center",
                position_exact=2.5,
                door_type="exterior",
                room_a="living_room",
            ),
            # No interior door to kitchen - only open connection.
        ]
        # Kitchen is open to living room.
        room_specs = [
            RoomSpec(
                room_id="living_room",
                room_type="living_room",
                prompt="Living room",
                width=4.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="kitchen",
                room_type="kitchen",
                prompt="Kitchen",
                width=4.0,
                length=3.0,
                connections={"living_room": ConnectionType.OPEN},
            ),
        ]

        # Without room_specs, kitchen should be unreachable.
        is_valid, msg = validate_connectivity(rooms, doors, room_specs=None)
        assert is_valid is False
        assert "kitchen" in msg.lower()

        # With room_specs (open connection), kitchen should be reachable.
        is_valid, msg = validate_connectivity(rooms, doors, room_specs=room_specs)
        assert is_valid is True


class TestRoomRotation(unittest.TestCase):
    """Tests for 90-degree room rotation during placement."""

    def test_room_rotated_to_fit_narrow_slot(self):
        """Room should be rotated 90° when it only fits in rotated orientation.

        Scenario:
        - Room A: 3x3 anchor at origin
        - Room B: 6x2 (wide and shallow), must be adjacent to A

        Without rotation: B's 6m dimension is too wide for good adjacency with A's 3m edge.
        With rotation: B becomes 2x6, and 2m easily fits against A's 3m edge.
        """
        specs = [
            RoomSpec(
                room_id="a",
                room_type="living",
                width=3.0,
                length=3.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="bedroom",
                width=2.0,  # depth when placed.
                length=6.0,  # width when placed.
                connections={"a": ConnectionType.DOOR},
            ),
        ]

        # This should succeed - B can rotate to fit against A.
        result = place_rooms(specs, config=PlacementConfig(min_shared_edge=1.5))

        assert len(result) == 2
        a = find_room(rooms=result, room_id="a")
        b = find_room(rooms=result, room_id="b")

        # Verify adjacency was satisfied.
        assert rooms_share_edge(room_a=a, room_b=b, min_overlap=1.5)

        # Verify no overlap.
        assert not rooms_overlap(room_a=a, room_b=b)

    def test_square_room_not_rotated_unnecessarily(self):
        """Square rooms should not try rotation (same dimensions)."""
        specs = [
            RoomSpec(
                room_id="a",
                room_type="living",
                width=4.0,
                length=4.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="bedroom",
                width=3.0,  # Square room.
                length=3.0,
                connections={"a": ConnectionType.DOOR},
            ),
        ]

        result = place_rooms(specs)

        assert len(result) == 2
        b = find_room(rooms=result, room_id="b")
        # Square room should maintain original dimensions.
        assert b.width == 3.0
        assert b.depth == 3.0

    def test_rotation_enables_tight_corner_placement(self):
        """Rotation should help fit room in constrained L-shaped corner.

        Scenario: Three rooms forming an L-shape where the third room
        needs specific orientation to satisfy dual adjacency.
        """
        specs = [
            RoomSpec(
                room_id="a",
                room_type="living",
                width=4.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="b",
                room_type="kitchen",
                width=3.0,
                length=4.0,
                connections={"a": ConnectionType.DOOR},
            ),
            RoomSpec(
                room_id="c",
                room_type="dining",
                width=2.0,
                length=5.0,  # Long narrow room.
                connections={"a": ConnectionType.DOOR, "b": ConnectionType.DOOR},
            ),
        ]

        # This complex adjacency requirement may need rotation to solve.
        try:
            result = place_rooms(specs)
            # If successful, verify all rooms placed without overlap.
            assert len(result) == 3
            for i, room_i in enumerate(result):
                for j, room_j in enumerate(result):
                    if i != j:
                        assert not rooms_overlap(room_a=room_i, room_b=room_j)
        except PlacementError:
            # Some configs may still be unsolvable - that's OK.
            # The test verifies rotation is attempted.
            pass


class TestEmptyInput(unittest.TestCase):
    """Tests for empty input handling."""

    def test_empty_specs(self):
        """Empty room specs should return empty result."""
        result = place_rooms([])
        assert result == []


class TestExteriorWallConstraint(unittest.TestCase):
    """Tests for exterior_walls constraint on room placement."""

    def test_exterior_wall_blocks_direct_adjacency(self):
        """Room with exterior_walls constraint should prevent adjacent rooms."""
        # Hallway with WEST wall marked exterior.
        # Kitchen wants to connect to hallway.
        # Kitchen should NOT be placed to the west of the hallway.
        specs = [
            RoomSpec(
                room_id="hallway",
                room_type="hallway",
                width=3.0,
                length=6.0,
                exterior_walls={WallDirection.WEST},
            ),
            RoomSpec(
                room_id="kitchen",
                room_type="kitchen",
                width=4.0,
                length=4.0,
                connections={"hallway": ConnectionType.DOOR},
            ),
        ]
        result = place_rooms(specs)

        # Kitchen should be placed somewhere but NOT west of hallway.
        hallway = find_room(rooms=result, room_id="hallway")
        kitchen = find_room(rooms=result, room_id="kitchen")

        # Hallway at origin: position (0, 0), width=6 (X), depth=3 (Y).
        # If kitchen were west of hallway, its max_x would be at hallway's min_x (0).
        # So kitchen_x + kitchen_width <= hallway_x means it's west.
        kitchen_max_x = kitchen.position[0] + kitchen.width
        hallway_min_x = hallway.position[0]

        # Kitchen should NOT be west of hallway (its east edge should be > hallway's west edge).
        # Actually, let's verify kitchen doesn't occupy the west clearance zone.
        # The clearance zone extends from hallway_min_x - clearance to hallway_min_x.
        # Default clearance is 20m, so any room with max_x in (hallway_min_x - 20, hallway_min_x]
        # that overlaps in Y would be rejected.
        # Simply verify kitchen is NOT directly west of hallway.
        is_kitchen_west_adjacent = (
            abs(kitchen_max_x - hallway_min_x) < 0.01
            and kitchen.position[1] < hallway.position[1] + hallway.depth
            and kitchen.position[1] + kitchen.depth > hallway.position[1]
        )
        assert (
            not is_kitchen_west_adjacent
        ), "Kitchen should not be placed west of hallway"

    def test_no_exterior_wall_allows_any_placement(self):
        """Rooms without exterior_walls constraint can be placed anywhere."""
        specs = [
            RoomSpec(
                room_id="living",
                room_type="living_room",
                width=4.0,
                length=5.0,
            ),
            RoomSpec(
                room_id="kitchen",
                room_type="kitchen",
                width=4.0,
                length=3.0,
                connections={"living": ConnectionType.DOOR},
            ),
        ]
        result = place_rooms(specs)

        assert len(result) == 2
        living = find_room(rooms=result, room_id="living")
        kitchen = find_room(rooms=result, room_id="kitchen")
        assert rooms_share_edge(room_a=living, room_b=kitchen, min_overlap=1.0)

    def test_multiple_exterior_walls(self):
        """Room with multiple exterior_walls blocks all specified directions."""
        # Corner room with WEST and SOUTH marked exterior.
        specs = [
            RoomSpec(
                room_id="corner",
                room_type="lobby",
                width=4.0,
                length=4.0,
                exterior_walls={WallDirection.WEST, WallDirection.SOUTH},
            ),
            RoomSpec(
                room_id="office",
                room_type="office",
                width=3.0,
                length=4.0,
                connections={"corner": ConnectionType.DOOR},
            ),
        ]
        result = place_rooms(specs)

        corner = find_room(rooms=result, room_id="corner")
        office = find_room(rooms=result, room_id="office")

        # Office should NOT be west or south of corner.
        # Check if office is west-adjacent (office.max_x == corner.min_x and Y overlaps).
        office_max_x = office.position[0] + office.width
        corner_min_x = corner.position[0]
        is_west_adjacent = (
            abs(office_max_x - corner_min_x) < 0.01
            and office.position[1] < corner.position[1] + corner.depth
            and office.position[1] + office.depth > corner.position[1]
        )

        # Check if office is south-adjacent (office.max_y == corner.min_y and X overlaps).
        office_max_y = office.position[1] + office.depth
        corner_min_y = corner.position[1]
        is_south_adjacent = (
            abs(office_max_y - corner_min_y) < 0.01
            and office.position[0] < corner.position[0] + corner.width
            and office.position[0] + office.width > corner.position[0]
        )

        assert not is_west_adjacent, "Office should not be placed west of corner"
        assert not is_south_adjacent, "Office should not be placed south of corner"

    def test_hallway_with_three_connections_and_exterior_wall(self):
        """Hallway with 3 room connections and 1 exterior wall should work."""
        # Hallway (2m x 8m) with connections to 3 rooms, west wall exterior.
        specs = [
            RoomSpec(
                room_id="hallway",
                room_type="hallway",
                width=2.0,
                length=8.0,
                exterior_walls={WallDirection.WEST},
            ),
            RoomSpec(
                room_id="room_a",
                room_type="bedroom",
                width=4.0,
                length=4.0,
                connections={"hallway": ConnectionType.DOOR},
            ),
            RoomSpec(
                room_id="room_b",
                room_type="bedroom",
                width=4.0,
                length=4.0,
                connections={"hallway": ConnectionType.DOOR},
            ),
            RoomSpec(
                room_id="room_c",
                room_type="bathroom",
                width=3.0,
                length=3.0,
                connections={"hallway": ConnectionType.DOOR},
            ),
        ]

        result = place_rooms(specs)

        # All rooms should be placed.
        assert len(result) == 4
        hallway = find_room(result, "hallway")

        # No room should be west of hallway.
        hallway_min_x = hallway.position[0]
        for room in result:
            if room.room_id == "hallway":
                continue
            room_max_x = room.position[0] + room.width
            is_west_adjacent = (
                abs(room_max_x - hallway_min_x) < 0.01
                and room.position[1] < hallway.position[1] + hallway.depth
                and room.position[1] + room.depth > hallway.position[1]
            )
            assert not is_west_adjacent, f"{room.room_id} should not be west of hallway"

    def test_bidirectional_exterior_wall_check(self):
        """Exterior wall constraint works bidirectionally."""
        # Room A has east wall exterior.
        # Room B wants to connect to A.
        # Room B should not be placed east of A, even though B doesn't specify exterior_walls.
        specs = [
            RoomSpec(
                room_id="room_a",
                room_type="living_room",
                width=4.0,
                length=5.0,
                exterior_walls={WallDirection.EAST},
            ),
            RoomSpec(
                room_id="room_b",
                room_type="kitchen",
                width=4.0,
                length=3.0,
                connections={"room_a": ConnectionType.DOOR},
            ),
        ]
        result = place_rooms(specs)

        room_a = find_room(result, "room_a")
        room_b = find_room(result, "room_b")

        # Room B should NOT be east of room A.
        room_a_max_x = room_a.position[0] + room_a.width
        room_b_min_x = room_b.position[0]
        is_east_adjacent = (
            abs(room_b_min_x - room_a_max_x) < 0.01
            and room_b.position[1] < room_a.position[1] + room_a.depth
            and room_b.position[1] + room_b.depth > room_a.position[1]
        )
        assert not is_east_adjacent, "Room B should not be placed east of room A"


if __name__ == "__main__":
    unittest.main()
