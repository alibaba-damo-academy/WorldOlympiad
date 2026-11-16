"""Open plan connection and layout change helper methods for FloorPlanTools.

This mixin provides methods for managing open connections between rooms (no wall),
and helper methods for handling layout changes (resizing, opening reapplication).
It is meant to be inherited by FloorPlanTools.
"""

import logging

from typing import TYPE_CHECKING

from scenecode.agent_utils.house import (
    ConnectionType,
    Opening,
    OpeningType,
    WallDirection,
)
from scenecode.floor_plan_agents.tools.ascii_generator import generate_ascii_floor_plan
from scenecode.floor_plan_agents.tools.room_placement import (
    PlacementConfig,
    PlacementError,
    get_shared_edge,
    place_rooms,
)

if TYPE_CHECKING:
    from scenecode.agent_utils.house import HouseLayout

console_logger = logging.getLogger(__name__)

# Epsilon for floating point comparison (meters).
POSITION_EPSILON = 0.001


class OpenPlanMixin:
    """Mixin providing open connection and layout change helper methods.

    Expected attributes on self:
        layout: HouseLayout - The house layout being modified.
        mode: str - Current mode ("room" or "house").
        placement_config: PlacementConfig - Configuration for room placement.

    Expected methods on self:
        _check_rooms_exist() -> Result | None
        _fail(message: str) -> Result
        _apply_door_to_wall(door: Door) -> str | None  (from DoorWindowMixin)
        _apply_window_to_wall(window: Window) -> str | None  (from DoorWindowMixin)
    """

    # Type hints for expected attributes (provided by FloorPlanTools).
    layout: "HouseLayout"
    mode: str
    placement_config: PlacementConfig

    def _add_open_connection_impl(self, room_a: str, room_b: str):
        """Create an open floor plan connection between two rooms (no wall).

        Makes two rooms share an open connection with no wall between them.
        The rooms will be placed adjacent but the shared wall is removed.
        Room mode: fails (single room has no connections).

        Args:
            room_a: First room ID.
            room_b: Second room ID.

        Returns:
            Result indicating success or failure.
        """
        # Import here to avoid circular dependency.
        from scenecode.floor_plan_agents.tools.floor_plan_tools import Result

        console_logger.info(
            f"Tool called: add_open_connection(room_a={room_a}, room_b={room_b})"
        )
        if self.mode == "room":
            return self._fail("Room mode: no open connections for single room.")

        error = self._check_rooms_exist()
        if error:
            return error

        spec_a = self.layout.get_room_spec(room_a)
        spec_b = self.layout.get_room_spec(room_b)

        if not spec_a:
            return self._fail(f"Room '{room_a}' not found.")
        if not spec_b:
            return self._fail(f"Room '{room_b}' not found.")

        # If rooms are already placed, verify they share an edge.
        if self.layout.placed_rooms:
            placed_a = next(
                (r for r in self.layout.placed_rooms if r.room_id == room_a), None
            )
            placed_b = next(
                (r for r in self.layout.placed_rooms if r.room_id == room_b), None
            )
            if placed_a and placed_b:
                shared_edge = get_shared_edge(room_a=placed_a, room_b=placed_b)
                if not shared_edge:
                    return self._fail(
                        f"Rooms '{room_a}' and '{room_b}' are not adjacent. "
                        "Cannot create open connection."
                    )

        # Track previous connection types for rollback on failure.
        prev_b_in_a = spec_a.connections.get(room_b)
        prev_a_in_b = spec_b.connections.get(room_a)

        # Set open connection (bidirectional).
        spec_a.connections[room_b] = ConnectionType.OPEN
        spec_b.connections[room_a] = ConnectionType.OPEN

        # If rooms are already placed, just add OPEN openings to existing walls.
        # No need to re-run placement - rooms are already adjacent.
        if self.layout.placed_rooms:
            # Apply the new open connection to existing walls.
            self._apply_open_connections_to_walls()

            # Invalidate geometry for both rooms (wall structure changed).
            if self.layout.invalidate_room_geometry(room_a):
                console_logger.debug(f"Invalidated geometry for room: {room_a}")
            if self.layout.invalidate_room_geometry(room_b):
                console_logger.debug(f"Invalidated geometry for room: {room_b}")

            msg = f"Added open connection: {room_a} <-> {room_b} (no wall between)."
            return Result(success=True, message=msg)

        # Rooms not yet placed - run placement with stability.
        # Both rooms being connected should have freedom to move together.
        try:
            config = PlacementConfig(
                timeout_seconds=self.placement_config.timeout_seconds,
                scoring_weights=self.placement_config.scoring_weights,
                previous_positions={
                    r.room_id: r.position for r in self.layout.placed_rooms
                },
                free_room_ids={room_a, room_b},
            )
            placed_rooms = place_rooms(
                room_specs=self.layout.room_specs,
                config=config,
            )
            self.layout.placed_rooms = placed_rooms
            self.layout.placement_valid = True
        except PlacementError as e:
            # Rollback: restore original connection state.
            if prev_b_in_a is not None:
                spec_a.connections[room_b] = prev_b_in_a
            else:
                del spec_a.connections[room_b]
            if prev_a_in_b is not None:
                spec_b.connections[room_a] = prev_a_in_b
            else:
                del spec_b.connections[room_a]
            return self._fail(f"Add open connection failed (layout unchanged): {e}")

        # Invalidate all geometry (placement re-run affects all rooms).
        invalidated = self.layout.invalidate_all_room_geometries()
        if invalidated > 0:
            console_logger.debug(f"Invalidated {invalidated} room geometries")

        # Regenerate ASCII labels after placement.
        ascii_result = generate_ascii_floor_plan(placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        # Restore all openings (doors, windows, open connections) to new walls.
        removed_doors, removed_windows = self._reapply_openings_to_walls()

        msg = f"Added open connection: {room_a} <-> {room_b} (no wall between)."
        msg += self._format_removal_message(removed_doors, removed_windows)

        return Result(success=True, message=msg)

    def _remove_open_connection_impl(self, room_a: str, room_b: str):
        """Remove an open floor plan connection and restore the wall.

        Removes the open connection between two rooms, restoring a wall
        between them. The rooms will remain adjacent (sharing a wall).
        Room mode: fails.

        Args:
            room_a: First room ID.
            room_b: Second room ID.

        Returns:
            Result indicating success or failure.
        """
        from scenecode.floor_plan_agents.tools.floor_plan_tools import Result

        console_logger.info(
            f"Tool called: remove_open_connection(room_a={room_a}, room_b={room_b})"
        )
        if self.mode == "room":
            return self._fail("Room mode: no open connections for single room.")

        error = self._check_rooms_exist()
        if error:
            return error

        spec_a = self.layout.get_room_spec(room_a)
        spec_b = self.layout.get_room_spec(room_b)

        if not spec_a:
            return self._fail(f"Room '{room_a}' not found.")
        if not spec_b:
            return self._fail(f"Room '{room_b}' not found.")

        # Check if open connection exists in spec OR on walls.
        # We check both to handle desyncs between spec connections and wall state.
        has_spec_connection = (
            spec_a.connections.get(room_b) == ConnectionType.OPEN
            or spec_b.connections.get(room_a) == ConnectionType.OPEN
        )
        has_wall_opening = self._has_open_wall_opening(room_a=room_a, room_b=room_b)

        if not has_spec_connection and not has_wall_opening:
            return self._fail(f"No open connection between '{room_a}' and '{room_b}'.")

        # Change OPEN to DOOR (rooms remain adjacent, just with a wall now).
        # Track previous state for rollback.
        prev_b_in_a = spec_a.connections.get(room_b)
        prev_a_in_b = spec_b.connections.get(room_a)

        if prev_b_in_a == ConnectionType.OPEN:
            spec_a.connections[room_b] = ConnectionType.DOOR
        if prev_a_in_b == ConnectionType.OPEN:
            spec_b.connections[room_a] = ConnectionType.DOOR

        # If rooms are already placed, just remove OPEN openings from walls.
        # No re-placement needed - rooms stay where they are.
        if self.layout.placed_rooms:
            # Remove OPEN openings for this room pair from all walls.
            open_id_prefix = f"open_{room_a}_{room_b}"
            open_id_prefix_rev = f"open_{room_b}_{room_a}"
            for placed_room in self.layout.placed_rooms:
                for wall in placed_room.walls:
                    wall.openings = [
                        o
                        for o in wall.openings
                        if not (
                            o.opening_type == OpeningType.OPEN
                            and (
                                o.opening_id.startswith(open_id_prefix)
                                or o.opening_id.startswith(open_id_prefix_rev)
                            )
                        )
                    ]

            # Invalidate geometry for both rooms (wall structure changed).
            if self.layout.invalidate_room_geometry(room_a):
                console_logger.debug(f"Invalidated geometry for room: {room_a}")
            if self.layout.invalidate_room_geometry(room_b):
                console_logger.debug(f"Invalidated geometry for room: {room_b}")

            msg = f"Removed open connection: {room_a} <-> {room_b} (wall restored)."
            return Result(success=True, message=msg)

        # Rooms not yet placed - run placement with stability.
        # No special rooms need freedom since we're just removing a connection.
        try:
            config = PlacementConfig(
                timeout_seconds=self.placement_config.timeout_seconds,
                scoring_weights=self.placement_config.scoring_weights,
                previous_positions={
                    r.room_id: r.position for r in self.layout.placed_rooms
                },
                free_room_ids=set(),
            )
            placed_rooms = place_rooms(
                room_specs=self.layout.room_specs,
                config=config,
            )
            self.layout.placed_rooms = placed_rooms
            self.layout.placement_valid = True
        except PlacementError as e:
            # Rollback: restore OPEN connection.
            if prev_b_in_a == ConnectionType.OPEN:
                spec_a.connections[room_b] = ConnectionType.OPEN
            if prev_a_in_b == ConnectionType.OPEN:
                spec_b.connections[room_a] = ConnectionType.OPEN
            return self._fail(f"Remove open connection failed (layout unchanged): {e}")

        # Invalidate all geometry (placement re-run affects all rooms).
        invalidated = self.layout.invalidate_all_room_geometries()
        if invalidated > 0:
            console_logger.debug(f"Invalidated {invalidated} room geometries")

        # Regenerate ASCII labels after placement.
        ascii_result = generate_ascii_floor_plan(placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        # Restore remaining openings (doors, windows, other open connections).
        removed_doors, removed_windows = self._reapply_openings_to_walls()

        msg = f"Removed open connection: {room_a} <-> {room_b} (wall restored)."
        msg += self._format_removal_message(removed_doors, removed_windows)

        return Result(success=True, message=msg)

    def _apply_open_connections_to_walls(self) -> None:
        """Apply OPEN openings for all open room connections.

        Creates OPEN type openings based on spec.connections with ConnectionType.OPEN.
        """
        if not self.layout.placed_rooms:
            return

        # Track processed pairs to avoid duplicates.
        processed_pairs: set[tuple[str, str]] = set()

        for spec in self.layout.room_specs:
            for other_room_id, conn_type in spec.connections.items():
                if conn_type != ConnectionType.OPEN:
                    continue

                # Create canonical pair key to avoid processing twice.
                pair_key = tuple(sorted([spec.room_id, other_room_id]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                # Find placed rooms.
                placed_a = next(
                    (r for r in self.layout.placed_rooms if r.room_id == spec.room_id),
                    None,
                )
                placed_b = next(
                    (r for r in self.layout.placed_rooms if r.room_id == other_room_id),
                    None,
                )

                if not placed_a or not placed_b:
                    console_logger.warning(
                        f"Open connection {spec.room_id}<->{other_room_id}: "
                        f"room(s) not placed"
                    )
                    continue

                # Get shared edge.
                shared_edge = get_shared_edge(room_a=placed_a, room_b=placed_b)
                if not shared_edge:
                    console_logger.warning(
                        f"Open connection {spec.room_id}<->{other_room_id}: "
                        f"no shared edge found"
                    )
                    continue

                # Create Opening.
                opening_id = f"open_{spec.room_id}_{other_room_id}"
                opening = Opening(
                    opening_id=opening_id,
                    opening_type=OpeningType.OPEN,
                    position_along_wall=shared_edge.position_along_wall,
                    width=shared_edge.width,
                    height=0.0,  # Ignored for OPEN type.
                    sill_height=0.0,
                )

                # Add to room_a's wall (skip if already exists).
                for wall in placed_a.walls:
                    if wall.direction == shared_edge.wall_direction:
                        if not any(o.opening_id == opening_id for o in wall.openings):
                            wall.openings.append(opening)
                        break

                # Add to room_b's wall (opposite direction).
                # Must create separate Opening with position relative to room_b's wall.
                shared_edge_b = get_shared_edge(placed_b, placed_a)
                if shared_edge_b:
                    opening_b_id = f"{opening_id}_b"
                    opening_b = Opening(
                        opening_id=opening_b_id,
                        opening_type=OpeningType.OPEN,
                        position_along_wall=shared_edge_b.position_along_wall,
                        width=shared_edge_b.width,
                        height=0.0,  # Ignored for OPEN type.
                        sill_height=0.0,
                    )
                    for wall in placed_b.walls:
                        if wall.direction == shared_edge_b.wall_direction:
                            if not any(
                                o.opening_id == opening_b_id for o in wall.openings
                            ):
                                wall.openings.append(opening_b)
                            break

    def _has_open_wall_opening(self, room_a: str, room_b: str) -> bool:
        """Check if OPEN type opening exists between two rooms on walls.

        This detects physical open connections even if spec.connections is out of sync.
        Useful for robustly handling desyncs between spec.connections and wall state.

        Args:
            room_a: First room ID.
            room_b: Second room ID.

        Returns:
            True if an OPEN opening exists between the rooms on walls.
        """
        if not self.layout.placed_rooms:
            return False

        open_id_prefix = f"open_{room_a}_{room_b}"
        open_id_prefix_rev = f"open_{room_b}_{room_a}"

        for placed_room in self.layout.placed_rooms:
            if placed_room.room_id not in (room_a, room_b):
                continue
            for wall in placed_room.walls:
                for opening in wall.openings:
                    if opening.opening_type == OpeningType.OPEN:
                        if opening.opening_id.startswith(
                            open_id_prefix
                        ) or opening.opening_id.startswith(open_id_prefix_rev):
                            return True
        return False

    def _find_boundary_for_rooms(
        self, room_a: str, room_b: str | None, direction: str | None = None
    ) -> str | None:
        """Find the boundary label for a room pair after layout change.

        Boundary labels (A, B, C...) are regenerated by the ASCII generator
        after each layout change. This method finds the new label for a
        specific room pair.

        Args:
            room_a: First room ID.
            room_b: Second room ID (None for exterior boundaries).
            direction: Wall direction ("north", "south", etc.) for exterior
                boundaries. Required when room_b is None to distinguish between
                multiple exterior walls of the same room.

        Returns:
            New boundary label, or None if room pair no longer shares a boundary.
        """
        for label, (bound_a, bound_b, bound_dir) in self.layout.boundary_labels.items():
            # Match room pair (order doesn't matter for interior boundaries).
            if room_b is None:
                # Exterior: room_a must match, room_b must be None, and direction
                # must match if specified.
                if bound_a == room_a and bound_b is None:
                    if direction is None or bound_dir == direction:
                        return label
            else:
                # Interior: check both orderings.
                if (bound_a == room_a and bound_b == room_b) or (
                    bound_a == room_b and bound_b == room_a
                ):
                    return label
        return None

    def _reapply_openings_to_walls(self) -> tuple[list[str], list[str]]:
        """Re-apply all doors, windows, and open connections to walls.

        Called after place_rooms() to restore openings that were lost when
        new Wall objects were created. Updates boundary labels if they changed
        during layout regeneration. Validates each opening still fits.
        Invalid openings are removed from layout lists.

        Returns:
            Tuple of (removed_door_ids, removed_window_ids).
        """
        removed_doors: list[str] = []
        removed_windows: list[str] = []

        if not self.layout.placed_rooms:
            return removed_doors, removed_windows

        # Re-apply doors, updating boundary labels and removing invalid ones.
        valid_doors = []
        for door in self.layout.doors:
            # Update boundary_label if it changed during layout regeneration.
            new_label = self._find_boundary_for_rooms(
                room_a=door.room_a, room_b=door.room_b
            )
            if new_label is None:
                removed_doors.append(
                    f"{door.id} ({door.boundary_label}): rooms no longer share boundary"
                )
                continue
            if new_label != door.boundary_label:
                console_logger.debug(
                    f"Door {door.id}: boundary label updated "
                    f"{door.boundary_label} -> {new_label}"
                )
                door.boundary_label = new_label

            error = self._apply_door_to_wall(door)
            if error:
                removed_doors.append(f"{door.id} ({door.boundary_label}): {error}")
            else:
                valid_doors.append(door)
        self.layout.doors = valid_doors

        # Re-apply windows, updating boundary labels and removing invalid ones.
        # Windows are exterior-only, so room_b is always None. Use wall_direction
        # to distinguish between multiple exterior walls of the same room.
        valid_windows = []
        for window in self.layout.windows:
            # Update boundary_label if it changed during layout regeneration.
            direction_str = (
                window.wall_direction.value if window.wall_direction else None
            )
            new_label = self._find_boundary_for_rooms(
                room_a=window.room_id, room_b=None, direction=direction_str
            )
            if new_label is None:
                removed_windows.append(
                    f"{window.id} ({window.boundary_label}): exterior boundary "
                    f"for room '{window.room_id}' {direction_str} no longer exists"
                )
                continue
            if new_label != window.boundary_label:
                console_logger.debug(
                    f"Window {window.id}: boundary label updated "
                    f"{window.boundary_label} -> {new_label}"
                )
                window.boundary_label = new_label

            error = self._apply_window_to_wall(window)
            if error:
                removed_windows.append(
                    f"{window.id} ({window.boundary_label}): {error}"
                )
            else:
                valid_windows.append(window)
        self.layout.windows = valid_windows

        # Re-apply open connections.
        self._apply_open_connections_to_walls()

        return removed_doors, removed_windows

    def _remove_openings_for_room(self, room_id: str) -> list[str]:
        """Remove all doors and windows for a specific room.

        Used when resizing a room - opening positions may no longer make sense.
        Open connections (OPEN type) are NOT removed here; their positions are
        recomputed automatically from shared edges during reapply.

        Args:
            room_id: Room ID to remove openings for.

        Returns:
            List of removed opening descriptions (for result message).
        """
        removed: list[str] = []

        # Remove doors where this room is room_a or room_b.
        doors_to_keep = []
        for door in self.layout.doors:
            if door.room_a == room_id or door.room_b == room_id:
                removed.append(f"door {door.id} ({door.boundary_label})")
                console_logger.info(
                    f"Removing door {door.id} due to room '{room_id}' resize"
                )
            else:
                doors_to_keep.append(door)
        self.layout.doors = doors_to_keep

        # Remove windows on this room's exterior walls.
        windows_to_keep = []
        for window in self.layout.windows:
            if window.room_id == room_id:
                removed.append(f"window {window.id} ({window.boundary_label})")
                console_logger.info(
                    f"Removing window {window.id} due to room '{room_id}' resize"
                )
            else:
                windows_to_keep.append(window)
        self.layout.windows = windows_to_keep

        return removed

    def _adjust_opening_positions_for_resize(
        self,
        room_id: str,
        old_width: float,
        old_depth: float,
        new_width: float,
        new_depth: float,
    ) -> None:
        """Proportionally adjust door/window positions after room resize.

        When a room's dimensions change, openings on walls whose length changed
        are proportionally repositioned. Openings on walls whose length didn't
        change stay at their current positions.

        Wall length mapping:
        - NORTH/SOUTH walls: length = room width (X dimension)
        - EAST/WEST walls: length = room depth (Y dimension)

        Args:
            room_id: Room that was resized.
            old_width: Previous room width (X dimension).
            old_depth: Previous room depth (Y dimension).
            new_width: New room width.
            new_depth: New room depth.
        """
        # Compute which dimensions changed.
        width_changed = abs(old_width - new_width) > POSITION_EPSILON
        depth_changed = abs(old_depth - new_depth) > POSITION_EPSILON

        if not width_changed and not depth_changed:
            return

        # Adjust door positions.
        for door in self.layout.doors:
            # Only adjust doors where this room is room_a
            # (position_exact is relative to room_a).
            if door.room_a != room_id:
                continue

            # Determine which wall the door is on by looking at boundary labels.
            # We'll use a simpler heuristic: if we have the boundary label, we can
            # look up the wall to find its direction.
            boundary_info = self.layout.boundary_labels.get(door.boundary_label)
            if not boundary_info:
                continue

            # boundary_info is (room_a_id, room_b_id or None, direction_str)
            _, _, direction_str = boundary_info
            if direction_str is None:
                continue

            direction = WallDirection(direction_str)

            # Determine old and new wall lengths based on direction.
            if direction in (WallDirection.NORTH, WallDirection.SOUTH):
                # Wall length = room width (X dimension).
                if not width_changed:
                    continue
                old_length = old_width
                new_length = new_width
            else:  # EAST or WEST
                # Wall length = room depth (Y dimension).
                if not depth_changed:
                    continue
                old_length = old_depth
                new_length = new_depth

            # Proportionally adjust position.
            if old_length > 0:
                ratio = new_length / old_length
                new_position = door.position_exact * ratio
                console_logger.debug(
                    f"Door {door.id}: repositioning {door.position_exact:.2f}m "
                    f"-> {new_position:.2f}m (ratio={ratio:.2f})"
                )
                door.position_exact = new_position

        # Adjust window positions.
        for window in self.layout.windows:
            if window.room_id != room_id:
                continue

            direction = window.wall_direction
            if direction is None:
                continue

            # Determine old and new wall lengths based on direction.
            if direction in (WallDirection.NORTH, WallDirection.SOUTH):
                if not width_changed:
                    continue
                old_length = old_width
                new_length = new_width
            else:  # EAST or WEST
                if not depth_changed:
                    continue
                old_length = old_depth
                new_length = new_depth

            # Proportionally adjust position.
            if old_length > 0:
                ratio = new_length / old_length
                new_position = window.position_along_wall * ratio
                console_logger.debug(
                    f"Window {window.id}: repositioning {window.position_along_wall:.2f}m "
                    f"-> {new_position:.2f}m (ratio={ratio:.2f})"
                )
                window.position_along_wall = new_position

    def _format_removal_message(
        self, removed_doors: list[str], removed_windows: list[str]
    ) -> str:
        """Format removal info for tool result messages.

        Args:
            removed_doors: List of removed door descriptions.
            removed_windows: List of removed window descriptions.

        Returns:
            Formatted string to append to result message, or empty if nothing removed.
        """
        if not removed_doors and not removed_windows:
            return ""

        parts = []
        if removed_doors:
            parts.append(
                f"Removed {len(removed_doors)} door(s): {'; '.join(removed_doors)}"
            )
        if removed_windows:
            parts.append(
                f"Removed {len(removed_windows)} window(s): {'; '.join(removed_windows)}"
            )

        return " | " + " | ".join(parts)
