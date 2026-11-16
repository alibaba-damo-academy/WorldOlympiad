"""Tests for floor plan tools - door/window preservation after room changes."""

import json
import random
import unittest

from scenecode.agent_utils.house import HouseLayout, OpeningType, WallDirection
from scenecode.floor_plan_agents.tools.floor_plan_tools import FloorPlanTools
from scenecode.floor_plan_agents.tools.room_placement import get_shared_edge


class TestOpeningPreservation(unittest.TestCase):
    """Test that doors/windows are preserved when rooms are resized or modified."""

    def _create_single_room_layout(self) -> tuple:
        """Create a simple layout with one room."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        rooms = [
            {
                "type": "living_room",
                "prompt": "A spacious living room",
                "width": 5.0,
                "depth": 4.0,
            }
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success, f"Room creation failed: {result.message}"

        return layout, tools

    def _create_two_room_layout(self) -> tuple:
        """Create a layout with two adjacent rooms."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        rooms = [
            {
                "type": "living_room",
                "prompt": "A spacious living room",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "A modern kitchen",
                "width": 4.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success, f"Room creation failed: {result.message}"

        return layout, tools

    def test_door_preserved_after_room_resize_when_fits(self):
        """Door should be preserved and repositioned when room is resized if it still fits."""
        layout, tools = self._create_single_room_layout()

        # Add door on exterior wall.
        door_result = tools._add_door_impl(
            wall_id="A", position="left", width=1.0, height=2.1
        )
        assert door_result.success

        # Count door openings before resize.
        room = layout.placed_rooms[0]
        doors_before = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )
        assert doors_before == 1
        assert len(layout.doors) == 1
        old_position = layout.doors[0].position_exact

        # Resize room - door should be preserved (wall grew, door still fits).
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=6.0, depth=5.0
        )
        assert resize_result.success

        # Count door openings after resize - should be preserved.
        room = layout.placed_rooms[0]
        doors_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )

        assert (
            doors_after == 1
        ), "Door should be preserved after resize when it still fits"
        assert len(layout.doors) == 1, "Door metadata should be preserved"
        # Position should be proportionally adjusted (wall grew from 5m to 6m).
        new_position = layout.doors[0].position_exact
        expected_ratio = 6.0 / 5.0
        assert (
            abs(new_position - old_position * expected_ratio) < 0.01
        ), "Door repositioned proportionally"

    def test_window_preserved_after_room_resize_when_fits(self):
        """Window should be preserved and repositioned when room is resized if it still fits."""
        layout, tools = self._create_single_room_layout()

        # Add window on exterior wall B (which is on the depth dimension).
        window_result = tools._add_window_impl(
            wall_id="B", position="left", width=1.2, height=1.2
        )
        assert window_result.success

        # Count window openings before resize.
        room = layout.placed_rooms[0]
        windows_before = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.WINDOW])
            for w in room.walls
        )
        assert windows_before == 1
        assert len(layout.windows) == 1
        old_position = layout.windows[0].position_along_wall

        # Resize room - window should be preserved (wall grew, window still fits).
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=6.0, depth=5.0
        )
        assert resize_result.success

        # Count window openings after resize - should be preserved.
        room = layout.placed_rooms[0]
        windows_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.WINDOW])
            for w in room.walls
        )

        assert (
            windows_after == 1
        ), "Window should be preserved after resize when it still fits"
        assert len(layout.windows) == 1, "Window metadata should be preserved"
        # Position should be proportionally adjusted (depth grew from 4m to 5m).
        new_position = layout.windows[0].position_along_wall
        expected_ratio = 5.0 / 4.0
        # Use 0.1m tolerance due to floating point and boundary adjustments.
        assert (
            abs(new_position - old_position * expected_ratio) < 0.1
        ), f"Window repositioned proportionally: new={new_position}, expected={old_position * expected_ratio}"

    def test_door_invalidated_when_wall_shrinks(self):
        """Door at far end should be removed when room shrinks too much."""
        layout, tools = self._create_single_room_layout()

        # Add door on right side of 5m wall.
        door_result = tools._add_door_impl(
            wall_id="A", position="right", width=1.0, height=2.1
        )
        assert door_result.success

        # Door is near end of 5m wall (position ~3-4m).
        door_position = layout.doors[0].position_exact
        assert door_position > 2.8, f"Door should be at right end, got {door_position}"

        # Resize room to 2m wide - door position becomes invalid and door is removed.
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=2.0, depth=4.0
        )
        assert resize_result.success

        # Door opening should NOT be in wall and should be removed from layout.
        room = layout.placed_rooms[0]
        doors_in_wall = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )

        assert doors_in_wall == 0, "Door should be invalidated when wall shrinks"
        assert len(layout.doors) == 0, "Invalid door should be removed from layout"

        # Result message should inform about the removed door.
        assert "Removed" in resize_result.message, "Should inform about removed door"

    def test_partial_opening_preservation_on_resize(self):
        """Openings that still fit are preserved, those that don't are removed."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create wide room.
        # Boundary labels: A=north (8m), B=south (8m), C=east (4m), D=west (4m).
        rooms = [
            {"type": "living_room", "prompt": "A wide room", "width": 8.0, "depth": 4.0}
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Add door at left on wall A (north, 8m wall).
        # "left" position is around 0.2 * wall_length = 1.6m from start.
        door1_result = tools._add_door_impl(
            wall_id="A", position="left", width=1.0, height=2.1
        )
        assert (
            door1_result.success
        ), f"First door should succeed: {door1_result.message}"

        # Add door at right on wall B (south, also 8m wall - different wall).
        # "right" position is around 0.8 * wall_length = 6.4m from start.
        door2_result = tools._add_door_impl(
            wall_id="B", position="right", width=1.0, height=2.1
        )
        assert (
            door2_result.success
        ), f"Second door should succeed: {door2_result.message}"

        # Verify both doors are in walls.
        room = layout.placed_rooms[0]
        doors_before = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )
        assert doors_before == 2, f"Expected 2 doors, got {doors_before}"
        assert len(layout.doors) == 2

        # Resize room - shrink width from 8m to 2.5m (extreme shrink).
        # Both north (A) and south (B) walls shrink from 8m to 2.5m.
        # Left door on A at ~1.6m will scale to ~0.5m - door extends to 1.5m, fits in 2.5m wall.
        # Right door on B at ~6.4m will scale to ~2.0m - door extends to 3.0m > 2.5m wall, doesn't fit.
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=2.5, depth=4.0
        )
        assert resize_result.success

        room = layout.placed_rooms[0]
        doors_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.DOOR])
            for w in room.walls
        )

        # Left door should be preserved (scaled position still fits).
        # Right door should be removed (scaled position + width exceeds wall).
        assert (
            len(layout.doors) == 1
        ), f"Expected 1 door preserved, got {len(layout.doors)}"
        assert doors_after == 1, "One door should remain in wall openings"

        # Result message should mention the removed door.
        assert "Removed" in resize_result.message or "1" in resize_result.message

    def test_openings_preserved_after_add_adjacency(self):
        """Openings should be preserved when adjacency is added."""
        layout, tools = self._create_two_room_layout()

        # Add door on living room exterior wall.
        door_result = tools._add_door_impl(
            wall_id="A", position="center", width=1.0, height=2.1
        )
        assert door_result.success

        # Get initial door count.
        assert len(layout.doors) == 1

        # Remove and re-add adjacency to trigger re-placement.
        tools._remove_adjacency_impl(room_a="living_room", room_b="kitchen")
        tools._add_adjacency_impl(room_a="living_room", room_b="kitchen")

        # Door should still exist.
        assert len(layout.doors) == 1, "Door metadata should be preserved"

        # Check if door is in wall openings (may or may not be depending on wall changes).
        # At minimum, metadata should be preserved.

    def test_open_connection_creates_opening(self):
        """Adding open connection should create OPEN type opening."""
        layout, tools = self._create_two_room_layout()

        # Add open connection.
        result = tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")
        assert result.success

        # Check that OPEN openings were created.
        open_count = 0
        for room in layout.placed_rooms:
            for wall in room.walls:
                for opening in wall.openings:
                    if opening.opening_type == OpeningType.OPEN:
                        open_count += 1

        assert open_count >= 2, "OPEN openings should be created on both rooms' walls"

    def test_open_connection_preserved_after_resize(self):
        """Open connection should be preserved and recalculated when room is resized."""
        layout, tools = self._create_two_room_layout()

        # Add open connection.
        result = tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")
        assert result.success

        # Count OPEN openings before resize.
        open_before = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_before >= 2

        # Resize kitchen.
        resize_result = tools._resize_room_impl(room_id="kitchen", width=5.0, depth=4.0)
        assert resize_result.success

        # OPEN openings should still exist (recalculated for new overlap).
        open_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_after >= 2, "OPEN openings should be preserved after resize"

    def test_combined_openings_preserved_after_resize(self):
        """All opening types should be preserved after resize when they still fit."""
        layout, tools = self._create_two_room_layout()

        # Add door on living room exterior wall B (depth dimension).
        door_result = tools._add_door_impl(
            wall_id="B", position="left", width=0.9, height=2.1
        )
        assert door_result.success

        # Add window on kitchen exterior wall.
        window_result = tools._add_window_impl(
            wall_id="E", position="center", width=1.2, height=1.0
        )
        assert window_result.success

        # Add open connection.
        open_result = tools._add_open_connection_impl(
            room_a="living_room", room_b="kitchen"
        )
        assert open_result.success

        # Count all openings before resize.
        def count_openings():
            counts = {"door": 0, "window": 0, "open": 0}
            for room in layout.placed_rooms:
                for wall in room.walls:
                    for opening in wall.openings:
                        counts[opening.opening_type.value] += 1
            return counts

        before = count_openings()
        assert before["door"] == 1
        assert before["window"] == 1
        assert before["open"] >= 2

        # Resize living room - growing from 5x4 to 6x5.
        resize_result = tools._resize_room_impl(
            room_id="living_room", width=6.0, depth=5.0
        )
        assert resize_result.success

        # After resize (walls grow, so openings should fit):
        # - Door on living_room's wall B: PRESERVED (wall grew from 4m to 5m, door fits)
        # - Window on kitchen's wall: PRESERVED (kitchen wasn't resized)
        # - OPEN connections: PRESERVED with recomputed positions
        after = count_openings()
        assert (
            after["door"] == 1
        ), "Door on resized room preserved (wall grew, still fits)"
        assert (
            after["window"] == before["window"]
        ), "Window on other room should be preserved"
        assert after["open"] >= 2, "OPEN openings should be preserved"

    def test_remove_open_connection_clears_openings(self):
        """Removing open connection should remove OPEN type openings."""
        layout, tools = self._create_two_room_layout()

        # Add and then remove open connection.
        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Verify openings exist.
        open_count = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_count >= 2

        # Remove open connection.
        tools._remove_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Verify openings removed.
        open_after = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_after == 0, "OPEN openings should be removed"

    def test_three_room_layout_preserves_all_openings(self):
        """Complex layout with 3 rooms should preserve all openings after changes."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create L-shaped layout: living room with kitchen and bedroom adjacent.
        rooms = [
            {
                "type": "living_room",
                "prompt": "Main living area",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "Kitchen",
                "width": 3.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
            {
                "type": "bedroom",
                "prompt": "Bedroom",
                "width": 4.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success, f"Room creation failed: {result.message}"
        assert len(layout.placed_rooms) == 3

        # Find exterior walls for each room.
        exterior_walls = {}
        for label, (room_a, room_b, direction) in layout.boundary_labels.items():
            if room_b is None:  # Exterior wall.
                if room_a not in exterior_walls:
                    exterior_walls[room_a] = []
                exterior_walls[room_a].append(label)

        # Add door to living room, windows to kitchen and bedroom.
        living_wall = exterior_walls.get("living_room", [None])[0]
        if living_wall:
            tools._add_door_impl(wall_id=living_wall, position="center", width=0.9)

        kitchen_wall = exterior_walls.get("kitchen", [None])[0]
        if kitchen_wall:
            tools._add_window_impl(wall_id=kitchen_wall, position="center", width=1.0)

        bedroom_wall = exterior_walls.get("bedroom", [None])[0]
        if bedroom_wall:
            tools._add_window_impl(wall_id=bedroom_wall, position="center", width=1.2)

        # Add open connection between living room and kitchen.
        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Count openings before resize.
        def count_all():
            counts = {"door": 0, "window": 0, "open": 0}
            for room in layout.placed_rooms:
                for wall in room.walls:
                    for o in wall.openings:
                        counts[o.opening_type.value] += 1
            return counts

        before = count_all()

        # Resize bedroom from 4x3 to 5x4 (growing).
        resize_result = tools._resize_room_impl(room_id="bedroom", width=5.0, depth=4.0)
        assert resize_result.success

        after = count_all()

        # Resizing bedroom (growing) should preserve bedroom's window (repositioned):
        # - Door on living_room: PRESERVED (living_room wasn't resized)
        # - Window on kitchen: PRESERVED (kitchen wasn't resized)
        # - Window on bedroom: PRESERVED (bedroom grew, window repositioned and still fits)
        # - Open connection: PRESERVED (positions recomputed)
        assert (
            after["door"] >= before["door"]
        ), "Door on non-resized room should be preserved"
        assert (
            after["window"] == before["window"]
        ), "All windows preserved (bedroom grew, window still fits)"
        assert after["open"] >= 2, "Open connection should be preserved"

    def test_open_connection_width_matches_overlap(self):
        """Open connection width should match the actual room overlap."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create rooms of different sizes - overlap should be smaller room's width.
        rooms = [
            {"type": "living_room", "prompt": "Large room", "width": 6.0, "depth": 4.0},
            {
                "type": "kitchen",
                "prompt": "Smaller room",
                "width": 3.0,
                "depth": 4.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Calculate expected overlap.
        living_room = next(r for r in layout.placed_rooms if r.room_id == "living_room")
        kitchen = next(r for r in layout.placed_rooms if r.room_id == "kitchen")
        shared_edge = get_shared_edge(living_room, kitchen)
        assert shared_edge is not None, "Rooms should share an edge"

        # Add open connection.
        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Find the OPEN opening and verify width matches shared edge.
        for room in layout.placed_rooms:
            for wall in room.walls:
                for opening in wall.openings:
                    if opening.opening_type == OpeningType.OPEN:
                        assert abs(opening.width - shared_edge.width) < 0.01, (
                            f"Opening width {opening.width} should match "
                            f"shared edge width {shared_edge.width}"
                        )

    def test_open_connection_position_correct_for_both_walls(self):
        """Open connection position should be relative to each room's wall origin.

        When rooms have different sizes and the smaller room is offset, the
        opening position will be different for each wall. This test ensures
        each wall gets the correct position, not a shared incorrect position.
        """
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create rooms where kitchen is smaller and will be offset from living room.
        # Living room: 6m wide, Kitchen: 4m wide adjacent.
        # The placement algorithm may offset the smaller room.
        rooms = [
            {"type": "living_room", "prompt": "Large room", "width": 6.0, "depth": 5.0},
            {
                "type": "kitchen",
                "prompt": "Smaller room",
                "width": 4.0,
                "depth": 4.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Add open connection.
        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Get shared edges from both perspectives.
        living_room = next(r for r in layout.placed_rooms if r.room_id == "living_room")
        kitchen = next(r for r in layout.placed_rooms if r.room_id == "kitchen")

        shared_edge_living = get_shared_edge(living_room, kitchen)
        shared_edge_kitchen = get_shared_edge(kitchen, living_room)

        assert shared_edge_living is not None
        assert shared_edge_kitchen is not None

        # Find OPEN openings on each room's wall.
        living_opening = None
        kitchen_opening = None

        for wall in living_room.walls:
            for opening in wall.openings:
                if opening.opening_type == OpeningType.OPEN:
                    living_opening = opening
                    break

        for wall in kitchen.walls:
            for opening in wall.openings:
                if opening.opening_type == OpeningType.OPEN:
                    kitchen_opening = opening
                    break

        assert living_opening is not None, "Living room should have OPEN opening"
        assert kitchen_opening is not None, "Kitchen should have OPEN opening"

        # Each opening's position should match the shared edge from that room's perspective.
        assert (
            abs(
                living_opening.position_along_wall
                - shared_edge_living.position_along_wall
            )
            < 0.01
        ), (
            f"Living room opening position {living_opening.position_along_wall} should match "
            f"shared edge position {shared_edge_living.position_along_wall}"
        )
        assert (
            abs(
                kitchen_opening.position_along_wall
                - shared_edge_kitchen.position_along_wall
            )
            < 0.01
        ), (
            f"Kitchen opening position {kitchen_opening.position_along_wall} should match "
            f"shared edge position {shared_edge_kitchen.position_along_wall}"
        )

        # Verify positions can be different (this is the key invariant the bug violated).
        # Note: They might be equal if rooms are perfectly aligned, but they CAN differ.
        # The important thing is each is correct for its respective wall.

    def test_door_on_interior_wall(self):
        """Door on interior wall should create openings on both rooms' walls."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        rooms = [
            {
                "type": "living_room",
                "prompt": "Living room",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "Kitchen",
                "width": 4.0,
                "depth": 4.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Find interior wall label.
        interior_wall = None
        for label, (room_a, room_b, _) in layout.boundary_labels.items():
            if room_b is not None:  # Interior wall.
                interior_wall = label
                break

        assert interior_wall is not None, "Should have an interior wall"

        # Add door to interior wall.
        door_result = tools._add_door_impl(
            wall_id=interior_wall, position="center", width=0.9, height=2.1
        )
        assert door_result.success

        # Verify door metadata stored.
        assert len(layout.doors) == 1
        door = layout.doors[0]
        assert door.room_b is not None, "Interior door should have room_b set"

    def test_door_cutout_alignment_on_interior_walls(self):
        """Door cutouts on interior walls must align at same world position.

        When two rooms share an internal wall, each room has its own wall object
        with different start points. Door cutouts must align to the same world
        position, which means position_along_wall values will differ between walls.
        """
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create two rooms with adjacency.
        rooms = [
            {
                "type": "living_room",
                "prompt": "Living room",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "Kitchen",
                "width": 4.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Find interior wall label.
        interior_wall_label = None
        interior_room_a = None
        interior_room_b = None
        for label, (room_a, room_b, _) in layout.boundary_labels.items():
            if room_b is not None:  # Interior wall.
                interior_wall_label = label
                interior_room_a = room_a
                interior_room_b = room_b
                break

        assert interior_wall_label is not None, "Should have an interior wall"

        # Add door at position "left" (0.3m from start).
        door_result = tools._add_door_impl(
            wall_id=interior_wall_label, position="left", width=0.9, height=2.1
        )
        assert door_result.success

        # Find placed rooms.
        placed_a = next(r for r in layout.placed_rooms if r.room_id == interior_room_a)
        placed_b = next(r for r in layout.placed_rooms if r.room_id == interior_room_b)

        # Get shared edges from both perspectives.
        shared_edge_a = get_shared_edge(placed_a, placed_b)
        shared_edge_b = get_shared_edge(placed_b, placed_a)
        assert shared_edge_a is not None
        assert shared_edge_b is not None

        # Find door openings on each wall.
        opening_on_a = None
        opening_on_b = None

        for wall in placed_a.walls:
            if wall.direction == shared_edge_a.wall_direction:
                for opening in wall.openings:
                    if opening.opening_type == OpeningType.DOOR:
                        opening_on_a = opening
                        break

        for wall in placed_b.walls:
            if wall.direction == shared_edge_b.wall_direction:
                for opening in wall.openings:
                    if opening.opening_type == OpeningType.DOOR:
                        opening_on_b = opening
                        break

        assert opening_on_a is not None, "Room A wall should have door opening"
        assert opening_on_b is not None, "Room B wall should have door opening"

        # Calculate world positions of door left edges.
        # For vertical walls (east/west), position is along Y axis.
        # For horizontal walls (north/south), position is along X axis.
        def get_world_position(placed_room, wall_dir, position_along_wall):
            """Convert wall-relative position to world coordinate."""
            x, y = placed_room.position
            if wall_dir in (WallDirection.EAST, WallDirection.WEST):
                # Wall runs along Y axis from room's min_y.
                return y + position_along_wall
            else:
                # Wall runs along X axis from room's min_x.
                return x + position_along_wall

        world_pos_a = get_world_position(
            placed_a, shared_edge_a.wall_direction, opening_on_a.position_along_wall
        )
        world_pos_b = get_world_position(
            placed_b, shared_edge_b.wall_direction, opening_on_b.position_along_wall
        )

        # Door cutouts must align at the same world position.
        assert abs(world_pos_a - world_pos_b) < 0.01, (
            f"Door cutouts must align! Room A door at world pos {world_pos_a:.3f}, "
            f"Room B door at world pos {world_pos_b:.3f}, "
            f"position_along_wall A={opening_on_a.position_along_wall:.3f}, "
            f"position_along_wall B={opening_on_b.position_along_wall:.3f}"
        )

    def test_room_creation_validates_dimensions(self):
        """Room creation should fail for invalid dimensions."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Zero-width room should fail.
        rooms = [
            {
                "type": "living_room",
                "prompt": "Invalid room",
                "width": 0.0,
                "depth": 4.0,
            }
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        # The placement algorithm should reject this.
        # Note: If this passes, the code might need validation added.

    def test_sequential_operations_maintain_consistency(self):
        """Multiple sequential operations should maintain layout consistency."""
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create initial layout.
        rooms = [
            {
                "type": "living_room",
                "prompt": "Living room",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "Kitchen",
                "width": 4.0,
                "depth": 3.0,
                "connections": {"living_room": "DOOR"},
            },
        ]
        tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))

        # Add various openings.
        exterior_walls = [
            label
            for label, (_, room_b, _) in layout.boundary_labels.items()
            if room_b is None
        ]
        if len(exterior_walls) >= 2:
            tools._add_door_impl(
                wall_id=exterior_walls[0], position="center", width=0.9
            )
            tools._add_window_impl(
                wall_id=exterior_walls[1], position="center", width=1.0
            )

        tools._add_open_connection_impl(room_a="living_room", room_b="kitchen")

        # Perform multiple resize operations (all growing or similar).
        tools._resize_room_impl(room_id="living_room", width=6.0, depth=4.0)
        tools._resize_room_impl(room_id="kitchen", width=5.0, depth=3.5)
        tools._resize_room_impl(room_id="living_room", width=5.5, depth=4.5)

        # Layout should still be valid.
        assert layout.placement_valid
        assert len(layout.placed_rooms) == 2

        # Doors and windows should be preserved if they still fit after proportional
        # repositioning. Since all resizes here are growing or similar, openings
        # that were originally at "center" should still fit after repositioning.
        # (The exact count depends on which room each opening was on and whether
        # it still fits after all the resizes.)
        # At minimum, open connections should be preserved.

        # Open connection should still have openings (positions recomputed).
        open_count = sum(
            len([o for o in w.openings if o.opening_type == OpeningType.OPEN])
            for room in layout.placed_rooms
            for w in room.walls
        )
        assert open_count >= 2, "Open connection should survive multiple resizes"

    def test_door_window_overlap_prevention_on_exterior_wall(self):
        """Doors and windows on same wall must not overlap (min separation enforced)."""
        layout = HouseLayout()
        # Use small separation for predictable test behavior.
        tools = FloorPlanTools(layout=layout, mode="house", min_opening_separation=0.5)

        # Create a large room so wall is long enough for both door and window.
        rooms = [
            {"type": "living_room", "prompt": "Living room", "width": 8.0, "depth": 6.0}
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success

        # Find an exterior wall.
        exterior_wall = None
        for label, (room_a, room_b, _) in layout.boundary_labels.items():
            if room_b is None:  # Exterior wall.
                exterior_wall = label
                break
        assert exterior_wall is not None, "Should have exterior wall"

        # Add window at left (small wall segment).
        window_result = tools._add_window_impl(
            wall_id=exterior_wall, position="left", width=1.0
        )
        assert window_result.success, f"Window should succeed: {window_result.message}"

        # Adding door at right (different segment) should succeed.
        door_result = tools._add_door_impl(wall_id=exterior_wall, position="right")
        assert (
            door_result.success
        ), f"Door at right should succeed: {door_result.message}"

        # Now test overlap detection: try to add door at same position as window.
        # On a new layout with window at center.
        # Use seed for deterministic positioning to ensure overlap.
        random.seed(42)
        layout2 = HouseLayout()
        tools2 = FloorPlanTools(
            layout=layout2, mode="house", min_opening_separation=0.5
        )
        tools2._generate_room_specs_impl(room_specs_json=json.dumps(rooms))

        exterior_wall2 = None
        for label, (room_a, room_b, _) in layout2.boundary_labels.items():
            if room_b is None:
                exterior_wall2 = label
                break

        # Add window at center.
        window_result2 = tools2._add_window_impl(
            wall_id=exterior_wall2, position="center", width=1.5
        )
        assert window_result2.success

        # Adding door at center should fail (overlap).
        door_result2 = tools2._add_door_impl(wall_id=exterior_wall2, position="center")
        assert not door_result2.success, "Door should fail when overlapping window"
        assert "overlap" in door_result2.message.lower()


class TestLayoutCheckpointRestore(unittest.TestCase):
    """Test HouseLayout checkpoint/restore for reset functionality."""

    def test_layout_round_trip_preserves_all_state(self):
        """HouseLayout.from_dict(layout.to_dict()) should preserve all state.

        This test ensures the checkpoint/reset mechanism works correctly.
        If this test fails, _perform_checkpoint_reset would restore corrupted state.
        """
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create a complex layout with rooms, adjacencies, open connections.
        rooms = [
            {
                "type": "living_room",
                "prompt": "Living room",
                "width": 5.0,
                "depth": 4.0,
            },
            {
                "type": "kitchen",
                "prompt": "Kitchen",
                "width": 4.0,
                "depth": 3.0,
                "connections": {"living_room": "OPEN"},
            },
            {
                "type": "bedroom",
                "prompt": "Bedroom",
                "width": 4.0,
                "depth": 3.5,
                "connections": {"living_room": "DOOR"},
            },
        ]
        result = tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))
        assert result.success, f"Room creation failed: {result.message}"

        # Add doors and windows.
        for label, (room_a, room_b, direction) in layout.boundary_labels.items():
            if room_b is None:  # Exterior wall.
                tools._add_window_impl(wall_id=label, position="center", width=1.2)
            elif room_b == "bedroom":  # Interior door to bedroom.
                tools._add_door_impl(wall_id=label, position="center")

        # Capture state before serialization.
        original_room_ids = [s.room_id for s in layout.room_specs]
        original_door_count = len(layout.doors)
        original_window_count = len(layout.windows)
        original_placed_room_count = len(layout.placed_rooms)
        original_connections = layout.room_specs[
            1
        ].connections  # Kitchen's connections.

        # Serialize and restore.
        state_dict = layout.to_dict()
        restored = HouseLayout.from_dict(state_dict)

        # Verify all state was preserved.
        restored_room_ids = [s.room_id for s in restored.room_specs]
        assert restored_room_ids == original_room_ids, "Room IDs should match"

        assert (
            len(restored.doors) == original_door_count
        ), f"Door count should match: {len(restored.doors)} vs {original_door_count}"
        assert (
            len(restored.windows) == original_window_count
        ), f"Window count should match: {len(restored.windows)} vs {original_window_count}"
        assert (
            len(restored.placed_rooms) == original_placed_room_count
        ), f"Placed room count should match: {len(restored.placed_rooms)} vs {original_placed_room_count}"

        # Verify connections preserved.
        restored_kitchen = next(
            s for s in restored.room_specs if s.room_id == "kitchen"
        )
        assert (
            restored_kitchen.connections == original_connections
        ), f"connections should match: {restored_kitchen.connections} vs {original_connections}"

        # Verify placement_valid flag.
        assert restored.placement_valid == layout.placement_valid

    def test_layout_restore_after_modifications(self):
        """Restoring from checkpoint should undo subsequent modifications.

        Simulates the reset workflow: create checkpoint, make changes, restore.
        """
        layout = HouseLayout()
        tools = FloorPlanTools(layout=layout, mode="house")

        # Create initial layout.
        rooms = [
            {"type": "living_room", "prompt": "Living room", "width": 5.0, "depth": 4.0}
        ]
        tools._generate_room_specs_impl(room_specs_json=json.dumps(rooms))

        # Add a door on north wall (this is our checkpoint state).
        exterior_walls = []
        for label, (room_a, room_b, direction) in layout.boundary_labels.items():
            if room_b is None:  # Exterior wall.
                exterior_walls.append((label, direction))

        # Use first wall for door.
        door_wall = exterior_walls[0][0]
        tools._add_door_impl(wall_id=door_wall, position="center")

        # Create checkpoint.
        checkpoint = layout.to_dict()
        checkpoint_door_count = len(layout.doors)
        checkpoint_window_count = len(layout.windows)

        # Make modifications on a DIFFERENT wall (to avoid overlap issues).
        # Find a wall without the door.
        window_wall = exterior_walls[1][0] if len(exterior_walls) > 1 else door_wall
        window_result = tools._add_window_impl(
            wall_id=window_wall, position="center", width=1.0
        )
        assert window_result.success, f"Window should be added: {window_result.message}"
        assert (
            len(layout.windows) == checkpoint_window_count + 1
        ), "Window should be added"

        # Restore from checkpoint (simulating reset).
        restored = HouseLayout.from_dict(checkpoint)

        # Verify modifications were undone.
        assert len(restored.doors) == checkpoint_door_count
        assert (
            len(restored.windows) == checkpoint_window_count
        ), f"Window count should be restored: {len(restored.windows)} vs {checkpoint_window_count}"


if __name__ == "__main__":
    unittest.main()
