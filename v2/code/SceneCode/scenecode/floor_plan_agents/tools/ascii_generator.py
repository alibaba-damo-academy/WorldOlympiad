"""ASCII floor plan visualization for LLM interface.

Generates text-based floor plan representations showing room boundaries,
room names, and wall segment labels for door/window placement.
"""

import logging

from dataclasses import dataclass

from scenecode.agent_utils.house import OpeningType, PlacedRoom, Wall, WallDirection
from scenecode.floor_plan_agents.tools.door_window_mixin import (
    SEGMENT_LEFT_END,
    SEGMENT_RIGHT_START,
)

console_logger = logging.getLogger(__name__)


# Character constants for ASCII drawing.
CORNER = "+"
HORIZONTAL = "-"
VERTICAL = "|"
SPACE = " "

# Scale factors for ASCII grid (increased for better label visibility).
# Terminal chars are ~2x tall as wide, so X=6, Y=3 gives ~1:1 visual aspect.
CHARS_PER_METER_X = 6  # Horizontal.
CHARS_PER_METER_Y = 3  # Vertical (enough rows for 1m rooms to show names).
MIN_ROOM_WIDTH_CHARS = 10  # Minimum to fit room names.


@dataclass
class AsciiFloorPlan:
    """Result of ASCII floor plan generation."""

    ascii_art: str
    """The ASCII representation of the floor plan."""

    boundary_labels: dict[str, tuple[str, str | None, str | None]]
    """Map of label to (room_a, room_b, direction) tuple.

    For interior walls: room_b is the adjacent room, direction is None.
    For exterior walls: room_b is None, direction is wall facing (e.g., "north").
    """

    legend: str
    """Legend explaining wall labels."""


def generate_ascii_floor_plan(placed_rooms: list[PlacedRoom]) -> AsciiFloorPlan:
    """Generate ASCII floor plan from placed rooms.

    Args:
        placed_rooms: List of placed rooms with walls.

    Returns:
        AsciiFloorPlan with ASCII art, labels, and legend.
    """
    if not placed_rooms:
        return AsciiFloorPlan(ascii_art="(No rooms)", boundary_labels={}, legend="")

    # Compute bounding box of all rooms.
    min_x = min(r.position[0] for r in placed_rooms)
    max_x = max(r.position[0] + r.width for r in placed_rooms)
    min_y = min(r.position[1] for r in placed_rooms)
    max_y = max(r.position[1] + r.depth for r in placed_rooms)

    # Create character grid with padding.
    padding = 2
    grid_width = int((max_x - min_x) * CHARS_PER_METER_X) + 2 * padding + 1
    grid_height = int((max_y - min_y) * CHARS_PER_METER_Y) + 2 * padding + 1

    # Ensure minimum dimensions.
    grid_width = max(grid_width, MIN_ROOM_WIDTH_CHARS + 4)
    grid_height = max(grid_height, 5)

    # Initialize grid with spaces.
    grid = [[SPACE for _ in range(grid_width)] for _ in range(grid_height)]

    # Draw each room.
    for room in placed_rooms:
        _draw_room(grid=grid, room=room, min_x=min_x, min_y=min_y, padding=padding)

    # Assign labels to walls/boundaries.
    boundary_labels = _assign_boundary_labels(placed_rooms)

    # Place labels on grid.
    _place_labels(
        grid=grid,
        placed_rooms=placed_rooms,
        boundary_labels=boundary_labels,
        min_x=min_x,
        min_y=min_y,
        padding=padding,
    )

    # Convert grid to string (flip Y axis - grid[0] is top, but Y=0 is bottom).
    lines = ["".join(row) for row in reversed(grid)]
    ascii_art = "\n".join(lines)

    # Generate legend.
    legend = _generate_legend(
        boundary_labels=boundary_labels, placed_rooms=placed_rooms
    )

    return AsciiFloorPlan(
        ascii_art=ascii_art, boundary_labels=boundary_labels, legend=legend
    )


def _draw_room(
    grid: list[list[str]],
    room: PlacedRoom,
    min_x: float,
    min_y: float,
    padding: int,
) -> None:
    """Draw a room on the grid.

    Args:
        grid: Character grid to draw on.
        room: Room to draw.
        min_x: Minimum X coordinate of all rooms.
        min_y: Minimum Y coordinate of all rooms.
        padding: Grid padding.
    """
    # Convert room coordinates to grid coordinates.
    x0 = int((room.position[0] - min_x) * CHARS_PER_METER_X) + padding
    x1 = int((room.position[0] + room.width - min_x) * CHARS_PER_METER_X) + padding
    y0 = int((room.position[1] - min_y) * CHARS_PER_METER_Y) + padding
    y1 = int((room.position[1] + room.depth - min_y) * CHARS_PER_METER_Y) + padding

    # Clamp to grid bounds.
    x0 = max(0, min(x0, len(grid[0]) - 1))
    x1 = max(0, min(x1, len(grid[0]) - 1))
    y0 = max(0, min(y0, len(grid) - 1))
    y1 = max(0, min(y1, len(grid) - 1))

    # Draw corners.
    grid[y0][x0] = CORNER
    grid[y0][x1] = CORNER
    grid[y1][x0] = CORNER
    grid[y1][x1] = CORNER

    # Draw horizontal walls (top=north, bottom=south).
    for x in range(x0 + 1, x1):
        if grid[y0][x] == SPACE:
            grid[y0][x] = HORIZONTAL
        if grid[y1][x] == SPACE:
            grid[y1][x] = HORIZONTAL

    # Draw vertical walls (left=west, right=east).
    for y in range(y0 + 1, y1):
        if grid[y][x0] == SPACE:
            grid[y][x0] = VERTICAL
        if grid[y][x1] == SPACE:
            grid[y][x1] = VERTICAL

    # Place room name in center.
    name = room.room_id
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2

    # Calculate available width for name.
    available_width = x1 - x0 - 2
    if available_width > 0 and center_y > 0 and center_y < len(grid):
        # Truncate name if necessary.
        truncated = name[:available_width]
        name_start = center_x - len(truncated) // 2

        for i, char in enumerate(truncated):
            col = name_start + i
            if 0 <= col < len(grid[0]) and x0 < col < x1:
                grid[center_y][col] = char


def _extract_other_room_from_open_id(opening_id: str, current_room: str) -> str | None:
    """Extract the other room ID from an OPEN opening's ID.

    Opening IDs have format: open_{room_a}_{room_b} or open_{room_a}_{room_b}_b.
    This returns the room that is NOT current_room.

    Args:
        opening_id: The opening ID (e.g., 'open_living_room_kitchen').
        current_room: The current room's ID.

    Returns:
        The other room's ID, or None if not parseable.
    """
    if not opening_id.startswith("open_"):
        return None

    # Remove 'open_' prefix and optional '_b' suffix.
    parts = opening_id[5:]  # Remove 'open_'
    if parts.endswith("_b"):
        parts = parts[:-2]

    # The parts now contain '{room_a}_{room_b}'.
    # We need to split this carefully since room names can contain underscores.
    # Try to find current_room in the string and extract the other room.
    if parts.startswith(current_room + "_"):
        return parts[len(current_room) + 1 :]
    elif parts.endswith("_" + current_room):
        return parts[: -(len(current_room) + 1)]

    return None


def _assign_boundary_labels(
    placed_rooms: list[PlacedRoom],
) -> dict[str, tuple[str, str | None, str | None]]:
    """Assign letter labels to wall boundaries.

    Interior walls are labeled first (A, B, C...), then exterior walls.

    Args:
        placed_rooms: All placed rooms.

    Returns:
        Dictionary mapping label to (room_a, room_b, direction) tuple.
        For interior walls: room_b is the adjacent room, direction is None.
        For exterior walls: room_b is None, direction is wall facing (e.g., "north").
    """
    labels: dict[str, tuple[str, str | None, str | None]] = {}
    label_ord = ord("A")

    # Collect all unique boundaries.
    interior_boundaries: list[tuple[str, str]] = []
    exterior_boundaries: list[tuple[str, WallDirection]] = []

    processed_pairs: set[tuple[str, ...]] = set()

    for room in placed_rooms:
        for wall in room.walls:
            if not wall.is_exterior:
                # Interior wall - find the room pair.
                for other_id in wall.faces_rooms:
                    pair = tuple(sorted([room.room_id, other_id]))
                    if pair not in processed_pairs:
                        interior_boundaries.append((room.room_id, other_id))
                        processed_pairs.add(pair)
            else:
                # Exterior wall - track room and direction.
                key = (room.room_id, wall.direction.value)
                if key not in processed_pairs:
                    exterior_boundaries.append((room.room_id, wall.direction))
                    processed_pairs.add(key)

    # Assign labels - interior first.
    for room_a, room_b in interior_boundaries:
        labels[chr(label_ord)] = (room_a, room_b, None)
        label_ord += 1

    # Then exterior - include direction for wall identification.
    for room_id, direction in exterior_boundaries:
        labels[chr(label_ord)] = (room_id, None, direction.value)
        label_ord += 1

    return labels


def _place_labels(
    grid: list[list[str]],
    placed_rooms: list[PlacedRoom],
    boundary_labels: dict[str, tuple[str, str | None, str | None]],
    min_x: float,
    min_y: float,
    padding: int,
) -> None:
    """Place boundary labels on the grid.

    Args:
        grid: Character grid.
        placed_rooms: All placed rooms.
        boundary_labels: Label assignments.
        min_x: Minimum X coordinate.
        min_y: Minimum Y coordinate.
        padding: Grid padding.
    """
    room_map = {r.room_id: r for r in placed_rooms}

    for label, (room_a, room_b, direction) in boundary_labels.items():
        room = room_map.get(room_a)
        if not room:
            continue

        # Find the wall for this boundary.
        target_wall = None
        for wall in room.walls:
            if room_b is None:
                # Exterior wall - match by direction.
                if wall.is_exterior and wall.direction.value == direction:
                    target_wall = wall
                    break
            else:
                # Interior wall - match by adjacent room.
                if room_b in wall.faces_rooms:
                    target_wall = wall
                    break

        if not target_wall:
            continue

        # Calculate label position at wall midpoint.
        mid_x = (target_wall.start_point[0] + target_wall.end_point[0]) / 2
        mid_y = (target_wall.start_point[1] + target_wall.end_point[1]) / 2

        grid_x = int((mid_x - min_x) * CHARS_PER_METER_X) + padding
        grid_y = int((mid_y - min_y) * CHARS_PER_METER_Y) + padding

        # Clamp to grid bounds.
        grid_x = max(0, min(grid_x, len(grid[0]) - 1))
        grid_y = max(0, min(grid_y, len(grid) - 1))

        # Place label.
        grid[grid_y][grid_x] = label


def _get_position_description(position_along_wall: float, wall_length: float) -> str:
    """Get coarse position description for an opening along a wall.

    Divides the wall into thirds and returns a description matching the
    vocabulary used by the floor plan agent tools (add_door, add_window).

    Args:
        position_along_wall: Distance from wall start to opening center in meters.
        wall_length: Total length of the wall in meters.

    Returns:
        One of: "left", "center", "right"
    """
    if wall_length <= 0:
        return "center"

    relative_position = position_along_wall / wall_length

    if relative_position < SEGMENT_LEFT_END:
        return "left"
    elif relative_position > SEGMENT_RIGHT_START:
        return "right"
    else:
        return "center"


def _generate_legend(
    boundary_labels: dict[str, tuple[str, str | None, str | None]],
    placed_rooms: list[PlacedRoom],
) -> str:
    """Generate legend explaining wall labels, doors, windows, and layout state.

    Args:
        boundary_labels: Label assignments (room_a, room_b, direction).
        placed_rooms: All placed rooms.

    Returns:
        Legend text including room summary, connectivity, walls, doors, and windows.
    """
    room_map = {r.room_id: r for r in placed_rooms}
    lines: list[str] = []

    # Room summary with dimensions.
    lines.extend(_generate_room_summary(placed_rooms))
    lines.append("")

    # Connectivity status.
    lines.extend(
        _generate_connectivity_status(
            placed_rooms=placed_rooms, boundary_labels=boundary_labels
        )
    )
    lines.append("")

    # Wall listings.
    interior_lines, exterior_lines = _generate_wall_listings(
        boundary_labels=boundary_labels, room_map=room_map
    )

    if len(interior_lines) > 1:
        lines.extend(interior_lines)
        lines.append("")
    if len(exterior_lines) > 1:
        lines.extend(exterior_lines)
        lines.append("")

    # Openings (doors, windows, open connections).
    open_lines, door_lines, window_lines, rooms_with_windows = (
        _generate_opening_listings(
            placed_rooms=placed_rooms, boundary_labels=boundary_labels
        )
    )

    if len(open_lines) > 1:
        lines.extend(open_lines)
        lines.append("")
    if len(door_lines) > 1:
        lines.extend(door_lines)
        lines.append("")
    if len(window_lines) > 1:
        lines.extend(window_lines)
        lines.append("")

    # Rooms without windows.
    rooms_without_windows = [
        r.room_id for r in placed_rooms if r.room_id not in rooms_with_windows
    ]
    if rooms_without_windows:
        formatted = [r for r in rooms_without_windows]
        lines.append(f"Rooms without windows: {', '.join(formatted)}")

    return "\n".join(lines)


def _generate_room_summary(placed_rooms: list[PlacedRoom]) -> list[str]:
    """Generate room summary with dimensions and areas."""
    lines = []
    total_area = sum(r.width * r.depth for r in placed_rooms)

    lines.append(f"Rooms ({len(placed_rooms)} total, {total_area:.1f}m² floor area):")
    for room in placed_rooms:
        area = room.width * room.depth
        lines.append(
            f"  - {room.room_id}: "
            f"{room.width:.1f}m × {room.depth:.1f}m ({area:.1f}m²)"
        )

    return lines


def _generate_connectivity_status(
    placed_rooms: list[PlacedRoom],
    boundary_labels: dict[str, tuple[str, str | None, str | None]],
) -> list[str]:
    """Generate connectivity status showing reachability and missing doors."""
    lines = []

    # Collect doors and open connections from walls.
    rooms_with_exterior_door: set[str] = set()
    interior_connections: dict[str, set[str]] = {r.room_id: set() for r in placed_rooms}
    open_connections: set[tuple[str, str]] = set()

    for room in placed_rooms:
        for wall in room.walls:
            for opening in wall.openings:
                if opening.opening_type == OpeningType.DOOR:
                    if wall.is_exterior:
                        rooms_with_exterior_door.add(room.room_id)
                    elif wall.faces_rooms:
                        other = wall.faces_rooms[0]
                        interior_connections[room.room_id].add(other)
                        interior_connections[other].add(room.room_id)
                elif opening.opening_type == OpeningType.OPEN:
                    if wall.faces_rooms:
                        other = wall.faces_rooms[0]
                        pair = tuple(sorted([room.room_id, other]))
                        open_connections.add(pair)
                        interior_connections[room.room_id].add(other)
                        interior_connections[other].add(room.room_id)

    # BFS to find reachable rooms.
    reachable: set[str] = set()
    if rooms_with_exterior_door:
        queue = list(rooms_with_exterior_door)
        reachable.update(queue)
        while queue:
            current = queue.pop(0)
            for neighbor in interior_connections.get(current, []):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)

    all_rooms = {r.room_id for r in placed_rooms}
    unreachable = all_rooms - reachable

    # Build status.
    if not rooms_with_exterior_door:
        lines.append("Connectivity: ✗ NO EXTERIOR DOOR - need at least one entry point")
    elif unreachable:
        formatted = [r for r in sorted(unreachable)]
        lines.append(
            f"Connectivity: ✗ INVALID - unreachable rooms: {', '.join(formatted)}"
        )

        # Suggest which doors are needed.
        lines.append("  Missing connections needed:")
        for room_id in sorted(unreachable):
            # Find adjacent rooms that ARE reachable.
            room = next((r for r in placed_rooms if r.room_id == room_id), None)
            if room:
                for wall in room.walls:
                    if not wall.is_exterior and wall.faces_rooms:
                        neighbor = wall.faces_rooms[0]
                        if neighbor in reachable:
                            wall_label = _find_wall_label(
                                boundary_labels=boundary_labels,
                                room_a=room_id,
                                room_b=neighbor,
                            )
                            label_hint = f" (wall {wall_label})" if wall_label else ""
                            lines.append(
                                f"    → {room_id} <-> " f"{neighbor}{label_hint}"
                            )
    else:
        lines.append("Connectivity: ✓ All rooms reachable from exterior")

    return lines


def _find_wall_label(
    boundary_labels: dict[str, tuple[str, str | None, str | None]],
    room_a: str,
    room_b: str,
) -> str | None:
    """Find the wall label for an interior wall between two rooms."""
    for label, (r_a, r_b, _) in boundary_labels.items():
        if (r_a == room_a and r_b == room_b) or (r_a == room_b and r_b == room_a):
            return label
    return None


def _compute_shared_edge_length(room_a: PlacedRoom, room_b: PlacedRoom) -> float:
    """Compute the length of the shared edge between two adjacent rooms."""
    # Room A bounds.
    a_x_min, a_y_min = room_a.position
    a_x_max = a_x_min + room_a.width
    a_y_max = a_y_min + room_a.depth

    # Room B bounds.
    b_x_min, b_y_min = room_b.position
    b_x_max = b_x_min + room_b.width
    b_y_max = b_y_min + room_b.depth

    # Check for vertical shared edge (rooms side by side horizontally).
    # A's east edge touches B's west edge, or A's west edge touches B's east edge.
    epsilon = 0.01
    if abs(a_x_max - b_x_min) < epsilon or abs(b_x_max - a_x_min) < epsilon:
        # Vertical edge - compute Y overlap.
        y_overlap_min = max(a_y_min, b_y_min)
        y_overlap_max = min(a_y_max, b_y_max)
        return max(0.0, y_overlap_max - y_overlap_min)

    # Check for horizontal shared edge (rooms stacked vertically).
    # A's north edge touches B's south edge, or A's south edge touches B's north edge.
    if abs(a_y_max - b_y_min) < epsilon or abs(b_y_max - a_y_min) < epsilon:
        # Horizontal edge - compute X overlap.
        x_overlap_min = max(a_x_min, b_x_min)
        x_overlap_max = min(a_x_max, b_x_max)
        return max(0.0, x_overlap_max - x_overlap_min)

    return 0.0


def _generate_wall_listings(
    boundary_labels: dict[str, tuple[str, str | None, str | None]],
    room_map: dict[str, PlacedRoom],
) -> tuple[list[str], list[str]]:
    """Generate interior and exterior wall listings."""
    interior_lines = ["Interior walls:"]
    exterior_lines = ["Exterior walls:"]

    for label, (room_a_id, room_b_id, direction) in sorted(boundary_labels.items()):
        room_a = room_map.get(room_a_id)
        if not room_a:
            continue

        # Find wall length for room_a.
        wall_a_length = 0.0
        for wall in room_a.walls:
            if room_b_id is None:
                if wall.is_exterior and wall.direction.value == direction:
                    wall_a_length = wall.length
                    break
            elif room_b_id in wall.faces_rooms:
                wall_a_length = wall.length
                break

        if room_b_id:
            room_b = room_map.get(room_b_id)
            if not room_b:
                continue

            # Find wall length for room_b.
            wall_b_length = 0.0
            for wall in room_b.walls:
                if room_a_id in wall.faces_rooms:
                    wall_b_length = wall.length
                    break

            # Compute actual shared edge length.
            shared_length = _compute_shared_edge_length(room_a, room_b)

            interior_lines.append(
                f"  {label}: {room_a_id} ({wall_a_length:.1f}m) <-> "
                f"{room_b_id} ({wall_b_length:.1f}m) "
                f"[shared: {shared_length:.1f}m]"
            )
        else:
            exterior_lines.append(
                f"  {label}: {room_a_id} {direction} " f"({wall_a_length:.1f}m)"
            )

    return interior_lines, exterior_lines


def _find_label_for_wall(
    boundary_labels: dict[str, tuple[str, str | None, str | None]],
    room_id: str,
    wall: Wall,
) -> str | None:
    """Find the boundary label for a wall.

    Args:
        boundary_labels: Mapping of label -> (room_a, room_b, direction).
        room_id: Room that owns this wall.
        wall: The wall to find a label for.

    Returns:
        The label (e.g., "A", "B") or None if not found.
    """
    for label, (room_a, room_b, direction) in boundary_labels.items():
        if wall.faces_rooms:
            # Interior wall - check if it connects these rooms.
            other_room = wall.faces_rooms[0]
            if {room_a, room_b} == {room_id, other_room}:
                return label
        else:
            # Exterior wall - check room and direction.
            if room_a == room_id and direction == wall.direction.value:
                return label
    return None


def _generate_opening_listings(
    placed_rooms: list[PlacedRoom],
    boundary_labels: dict[str, tuple[str, str | None, str | None]],
) -> tuple[list[str], list[str], list[str], set[str]]:
    """Generate listings for open connections, doors, and windows.

    Returns:
        Tuple of (open_lines, door_lines, window_lines, rooms_with_windows).
    """
    open_lines = ["Open connections:"]
    door_lines = ["Doors:"]
    exterior_door_lines: list[str] = []
    interior_door_lines: list[str] = []
    window_lines = ["Windows:"]
    rooms_with_windows: set[str] = set()

    processed: set[tuple[str, ...]] = set()

    for room in placed_rooms:
        for wall in room.walls:
            wall_label = _find_label_for_wall(
                boundary_labels=boundary_labels, room_id=room.room_id, wall=wall
            )
            label_prefix = f"wall {wall_label}: " if wall_label else ""

            for opening in wall.openings:
                position_desc = _get_position_description(
                    opening.position_along_wall, wall.length
                )

                if opening.opening_type == OpeningType.DOOR:
                    if wall.faces_rooms:
                        other_room = wall.faces_rooms[0]
                        door_key = tuple(sorted([room.room_id, other_room]))
                        if door_key in processed:
                            continue
                        processed.add(door_key)
                        interior_door_lines.append(
                            f"  - [{opening.opening_id}] {label_prefix}"
                            f"{room.room_id} <-> "
                            f"{other_room}, "
                            f"{opening.width:.1f}m × {opening.height:.1f}m, "
                            f"{position_desc}"
                        )
                    else:
                        exterior_door_lines.append(
                            f"  - [{opening.opening_id}] {label_prefix}"
                            f"{room.room_id} {wall.direction.value}, "
                            f"{opening.width:.1f}m × {opening.height:.1f}m, "
                            f"{position_desc} (ENTRY)"
                        )
                elif opening.opening_type == OpeningType.WINDOW:
                    rooms_with_windows.add(room.room_id)
                    window_lines.append(
                        f"  - [{opening.opening_id}] {label_prefix}"
                        f"{room.room_id} {wall.direction.value}, "
                        f"{opening.width:.1f}m × {opening.height:.1f}m, "
                        f"{position_desc}, sill {opening.sill_height:.1f}m"
                    )
                elif opening.opening_type == OpeningType.OPEN:
                    # Extract the other room from opening_id which has format:
                    # open_{room_a}_{room_b} or open_{room_a}_{room_b}_b
                    # This is more reliable than wall.faces_rooms[0] when a wall
                    # touches multiple rooms at different segments.
                    other_room = _extract_other_room_from_open_id(
                        opening.opening_id, room.room_id
                    )
                    if other_room:
                        open_key = tuple(sorted([room.room_id, other_room]))
                        if open_key in processed:
                            continue
                        processed.add(open_key)
                        open_lines.append(
                            f"  - {room.room_id} <-> "
                            f"{other_room}: "
                            f"{opening.width:.1f}m wide (full shared edge)"
                        )

    # Combine door lines with exterior first.
    if exterior_door_lines:
        door_lines.append("  Exterior:")
        door_lines.extend(f"  {line}" for line in exterior_door_lines)
    if interior_door_lines:
        door_lines.append("  Interior:")
        door_lines.extend(f"  {line}" for line in interior_door_lines)

    return open_lines, door_lines, window_lines, rooms_with_windows
