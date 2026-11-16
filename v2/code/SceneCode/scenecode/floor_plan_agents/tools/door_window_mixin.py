"""Door and window manipulation methods for FloorPlanTools.

This mixin provides methods for adding, removing, and managing doors and windows
in floor plan layouts. It is meant to be inherited by FloorPlanTools.
"""

import logging
import random

from typing import TYPE_CHECKING

from scenecode.agent_utils.house import (
    ConnectionType,
    Door,
    Opening,
    OpeningType,
    WallDirection,
    Window,
)
from scenecode.floor_plan_agents.tools.room_placement import (
    get_shared_edge,
    validate_connectivity,
)

if TYPE_CHECKING:
    from scenecode.agent_utils.house import HouseLayout

console_logger = logging.getLogger(__name__)

# Segment boundaries for left/center/right positioning (percentage of wall length).
SEGMENT_LEFT_END = 0.33
SEGMENT_RIGHT_START = 0.67

# Windows are inset from wall edges to avoid corners.
WINDOW_EDGE_INSET = 0.1


class DoorWindowMixin:
    """Mixin providing door and window manipulation methods.

    Expected attributes on self:
        layout: HouseLayout - The house layout being modified.
        min_opening_separation: float - Minimum separation between openings.
        door_window_config: DoorWindowConfig - Door/window constraints.

    Expected methods on self:
        _check_rooms_exist() -> Result | None
        _fail(message: str) -> Result
        _get_wall_by_boundary(wall_label: str, room_id: str) -> Wall
        _get_wall_length(room_id: str, wall_label: str) -> float
        _wall_faces_nearby_room(room_id: str, wall_label: str, threshold: float) -> str | None
    """

    # Type hints for expected attributes (provided by FloorPlanTools).
    layout: "HouseLayout"
    min_opening_separation: float
    # door_window_config is expected from FloorPlanTools.

    def _next_door_id(self) -> str:
        """Generate the next unique door ID.

        Finds the max existing door number and returns max + 1.
        Avoids collisions when doors are removed during layout changes.
        """
        max_num = 0
        for door in self.layout.doors:
            if door.id.startswith("door_"):
                try:
                    num = int(door.id[5:])
                    max_num = max(max_num, num)
                except ValueError:
                    pass
        return f"door_{max_num + 1}"

    def _next_window_id(self) -> str:
        """Generate the next unique window ID.

        Finds the max existing window number and returns max + 1.
        Avoids collisions when windows are removed during layout changes.
        """
        max_num = 0
        for window in self.layout.windows:
            if window.id.startswith("window_"):
                try:
                    num = int(window.id[7:])
                    max_num = max(max_num, num)
                except ValueError:
                    pass
        return f"window_{max_num + 1}"

    def _add_door_impl(
        self,
        wall_id: str,
        position: str = "center",
        width: float | None = None,
        height: float | None = None,
    ):
        """Add a door to a wall.

        At least one exterior door required. Interior doors connect rooms.

        Args:
            wall_id: Wall segment label (e.g., "A", "B").
            position: "left" | "center" | "right".
            width: Door width in valid range (uses config default if not specified).
            height: Door height in valid range (uses config default if not specified).

        Returns:
            Result indicating success or failure.
        """
        # Import here to avoid circular dependency.
        from scenecode.floor_plan_agents.tools.floor_plan_tools import Result

        # Apply defaults from config.
        cfg = self.door_window_config
        if width is None:
            width = cfg.door_default_width
        if height is None:
            height = cfg.door_default_height

        console_logger.info(
            f"Tool called: add_door(wall_id={wall_id}, position={position}, width={width})"
        )
        error = self._check_rooms_exist()
        if error:
            return error

        if position not in {"left", "center", "right"}:
            return self._fail(
                f"Position must be 'left', 'center', or 'right'. Got: {position}"
            )

        if not (cfg.door_width_min <= width <= cfg.door_width_max):
            return self._fail(
                f"Door width must be {cfg.door_width_min}-{cfg.door_width_max}m. "
                f"Got: {width}"
            )

        if not (cfg.door_height_min <= height <= cfg.door_height_max):
            return self._fail(
                f"Door height must be {cfg.door_height_min}-{cfg.door_height_max}m. "
                f"Got: {height}"
            )

        # Look up wall boundary.
        if wall_id not in self.layout.boundary_labels:
            available = ", ".join(sorted(self.layout.boundary_labels.keys()))
            return self._fail(f"Wall '{wall_id}' not found. Available: {available}")

        room_a, room_b, _direction = self.layout.boundary_labels[wall_id]
        door_type = "interior" if room_b else "exterior"

        # Check for open connection - cannot add door to open floor plan.
        if room_b:
            spec_a = self.layout.get_room_spec(room_a)
            if spec_a and spec_a.connections.get(room_b) == ConnectionType.OPEN:
                return self._fail(
                    f"Cannot add door between {room_a} and {room_b}: they have an "
                    f"open connection (no wall). Use remove_open_connection first "
                    f"if you want to add a door instead."
                )

        # Check exterior clearance for exterior doors.
        # Require configured clearance outside (not facing another room).
        if door_type == "exterior":
            clearance = cfg.exterior_door_clearance_m
            nearby_room = self._wall_faces_nearby_room(
                room_a, wall_id, threshold=clearance
            )
            if nearby_room:
                return self._fail(
                    f"Wall '{wall_id}' faces {nearby_room} within {clearance}m. "
                    f"Exterior doors require at least {clearance}m free space outside. "
                    f"Try a different exterior wall."
                )

        # Check for duplicate doors.
        for existing in self.layout.doors:
            if existing.boundary_label == wall_id:
                # Only one door per wall segment.
                return self._fail(f"Wall '{wall_id}' already has a door.")
            if door_type == "interior":
                # Interior: only one door between any two rooms.
                existing_rooms = {existing.room_a, existing.room_b}
                new_rooms = {room_a, room_b}
                if existing_rooms == new_rooms:
                    return self._fail(
                        f"Door already exists between '{room_a}' and '{room_b}'."
                    )

        # Calculate exact position within segment.
        # Segment ranges: left=0-33%, center=33-67%, right=67-100%.
        segment_ranges = {
            "left": (0.0, SEGMENT_LEFT_END),
            "center": (SEGMENT_LEFT_END, SEGMENT_RIGHT_START),
            "right": (SEGMENT_RIGHT_START, 1.0),
        }
        start_pct, end_pct = segment_ranges[position]

        # Find wall length.
        wall_length = self._get_wall_length(room_a, wall_id)

        # For interior doors, constrain position to shared edge (not full wall).
        # This ensures door fits within the overlapping region of both rooms.
        effective_start = 0.0
        effective_length = wall_length
        if door_type == "interior":
            placed_a = next(
                (r for r in self.layout.placed_rooms if r.room_id == room_a), None
            )
            placed_b = next(
                (r for r in self.layout.placed_rooms if r.room_id == room_b), None
            )
            if placed_a and placed_b:
                shared_edge = get_shared_edge(placed_a, placed_b)
                if shared_edge:
                    effective_start = shared_edge.position_along_wall
                    effective_length = shared_edge.width

        # Calculate position using center-based sampling.
        # Door center must be within segment, but edges must respect wall margins.
        margin = 0.3  # Door corner margin.
        half_width = width / 2

        # Valid center range based on wall boundaries (door must fit with margins).
        wall_center_min = effective_start + margin + half_width
        wall_center_max = effective_start + effective_length - margin - half_width

        # Segment boundaries for door center.
        segment_center_min = effective_start + effective_length * start_pct
        segment_center_max = effective_start + effective_length * end_pct

        # Intersection: center must be in segment AND within wall bounds.
        center_min = max(wall_center_min, segment_center_min)
        center_max = min(wall_center_max, segment_center_max)

        if center_max < center_min:
            return self._fail(
                f"Cannot place {width:.2f}m door in '{position}' segment of "
                f"{effective_length:.2f}m wall. Try a different segment or narrower door."
            )

        # Sample door center, convert to left edge position.
        door_center = random.uniform(center_min, center_max)
        position_exact = door_center - half_width

        # Check for overlap with existing windows on this wall.
        # Both door and window positions are LEFT EDGE.
        door_start = position_exact
        door_end = position_exact + width
        for window in self.layout.windows:
            if window.boundary_label == wall_id:
                window_start = window.position_along_wall
                window_end = window.position_along_wall + window.width
                # Check overlap with separation margin.
                if (
                    door_start < window_end + self.min_opening_separation
                    and door_end > window_start - self.min_opening_separation
                ):
                    return self._fail(
                        f"Door would overlap with window on wall '{wall_id}'. "
                        f"Door: {door_start:.2f}-{door_end:.2f}m, "
                        f"Window: {window_start:.2f}-{window_end:.2f}m. "
                        f"Min separation: {self.min_opening_separation}m. "
                        f"Try a different position."
                    )

        # Create door.
        door_id = self._next_door_id()
        door = Door(
            id=door_id,
            boundary_label=wall_id,
            position_segment=position,
            position_exact=position_exact,
            door_type=door_type,
            room_a=room_a,
            room_b=room_b,
            width=width,
            height=height,
        )

        # Add Opening to wall(s) for rendering/ASCII.
        # Only persist door if application succeeds.
        error = self._apply_door_to_wall(door)
        if error:
            return self._fail(f"Failed to add door at wall {wall_id}: {error}")

        self.layout.doors.append(door)

        # Invalidate geometry for affected rooms (wall openings changed).
        if self.layout.invalidate_room_geometry(room_a):
            console_logger.debug(f"Invalidated geometry for room: {room_a}")
        if room_b and self.layout.invalidate_room_geometry(room_b):
            console_logger.debug(f"Invalidated geometry for room: {room_b}")

        return Result(
            success=True,
            message=f"Added {door_type} door '{door_id}' at wall {wall_id} ({position}).",
        )

    def _apply_door_to_wall(self, door: Door) -> str | None:
        """Apply a door's Opening to the appropriate wall(s).

        Creates an Opening from the Door and adds it to the wall.
        For interior doors, adds to both rooms' walls with positions computed
        relative to each wall's start point.

        Args:
            door: The Door to apply.

        Returns:
            None if successful, error message string if door doesn't fit.
        """
        # Check wall still exists (boundary label may be invalid after layout change).
        if door.boundary_label not in self.layout.boundary_labels:
            msg = f"boundary '{door.boundary_label}' no longer exists"
            console_logger.warning(f"Door {door.id}: {msg}")
            return msg

        # Validate BOTH walls exist before adding to either (fail-fast).
        try:
            wall_a = self._get_wall_by_boundary(door.boundary_label, door.room_a)
        except ValueError as e:
            console_logger.warning(f"Door {door.id}: {e}")
            return str(e)

        wall_b = None
        if door.room_b:
            try:
                wall_b = self._get_wall_by_boundary(door.boundary_label, door.room_b)
            except ValueError as e:
                # Interior door must be applied to both walls or neither.
                console_logger.warning(f"Door {door.id} room_b: {e}")
                return str(e)

        # Validate exterior door clearance (may have changed after layout repositioning).
        if door.door_type == "exterior":
            clearance = self.door_window_config.exterior_door_clearance_m
            nearby_room = self._wall_faces_nearby_room(
                door.room_a, door.boundary_label, threshold=clearance
            )
            if nearby_room:
                msg = (
                    f"wall '{door.boundary_label}' now faces {nearby_room} "
                    f"within {clearance}m (exterior doors require clearance)"
                )
                console_logger.warning(f"Door {door.id}: {msg}")
                return msg

        # Validate position fits in wall (position_exact is left edge of door).
        if door.position_exact < 0:
            msg = f"position {door.position_exact:.2f}m is negative"
            console_logger.warning(f"Door {door.id}: {msg}")
            return msg
        if door.position_exact + door.width > wall_a.length:
            msg = (
                f"position {door.position_exact:.2f}m + width {door.width:.2f}m "
                f"exceeds wall length {wall_a.length:.2f}m"
            )
            console_logger.warning(f"Door {door.id}: {msg}")
            return msg

        # For interior doors, also validate that door fits within shared edge.
        # This prevents doors from being placed in parts of the wall that face exterior.
        shared_edge_a = None
        shared_edge_b = None
        placed_a = None
        placed_b = None

        if wall_b and door.room_b:
            placed_a = next(
                (r for r in self.layout.placed_rooms if r.room_id == door.room_a),
                None,
            )
            placed_b = next(
                (r for r in self.layout.placed_rooms if r.room_id == door.room_b),
                None,
            )

            if placed_a and placed_b:
                shared_edge_a = get_shared_edge(placed_a, placed_b)
                shared_edge_b = get_shared_edge(placed_b, placed_a)

                if shared_edge_a:
                    # Validate door fits within the shared edge on wall_a.
                    shared_start = shared_edge_a.position_along_wall
                    shared_end = shared_start + shared_edge_a.width
                    door_start = door.position_exact
                    door_end = door_start + door.width

                    if door_start < shared_start:
                        msg = (
                            f"door starts at {door_start:.2f}m but shared edge "
                            f"with {door.room_b} starts at {shared_start:.2f}m"
                        )
                        console_logger.warning(f"Door {door.id}: {msg}")
                        return msg
                    if door_end > shared_end:
                        msg = (
                            f"door ends at {door_end:.2f}m but shared edge "
                            f"with {door.room_b} ends at {shared_end:.2f}m "
                            f"(overlap width: {shared_edge_a.width:.2f}m)"
                        )
                        console_logger.warning(f"Door {door.id}: {msg}")
                        return msg
                else:
                    msg = (
                        f"no shared edge found between {door.room_a} and {door.room_b}"
                    )
                    console_logger.warning(f"Door {door.id}: {msg}")
                    return msg

        # Create Opening for wall_a.
        opening_a = Opening(
            opening_id=door.id,
            opening_type=OpeningType.DOOR,
            position_along_wall=door.position_exact,
            width=door.width,
            height=door.height,
            sill_height=0.0,
        )
        wall_a.openings.append(opening_a)

        # For interior doors, compute position on wall_b relative to its start.
        # Walls have different start points, so position_along_wall differs.
        if wall_b and door.room_b and shared_edge_a and shared_edge_b:
            # Door's offset from shared edge start on wall_a.
            door_offset = door.position_exact - shared_edge_a.position_along_wall
            # Position on wall_b = shared edge start on wall_b + same offset.
            position_on_wall_b = shared_edge_b.position_along_wall + door_offset

            opening_b = Opening(
                opening_id=door.id,
                opening_type=OpeningType.DOOR,
                position_along_wall=position_on_wall_b,
                width=door.width,
                height=door.height,
                sill_height=0.0,
            )
            wall_b.openings.append(opening_b)
        elif wall_b and door.room_b:
            # Fallback if shared edge computation failed (shouldn't happen).
            console_logger.warning(
                f"Door {door.id}: Could not compute shared edge for wall_b, "
                f"using same position (may be misaligned)"
            )
            wall_b.openings.append(opening_a)

        return None

    def _apply_window_to_wall(self, window: Window) -> str | None:
        """Apply a window's Opening to the appropriate wall.

        Creates an Opening from the Window and adds it to the wall.

        Args:
            window: The Window to apply.

        Returns:
            None if successful, error message string if window doesn't fit.
        """
        # Look up wall by stable (room_id, direction) if available.
        wall = None
        if window.wall_direction:
            placed_room = next(
                (r for r in self.layout.placed_rooms if r.room_id == window.room_id),
                None,
            )
            if placed_room:
                wall = next(
                    (
                        w
                        for w in placed_room.walls
                        if w.direction == window.wall_direction
                    ),
                    None,
                )
            if not wall:
                msg = (
                    f"Wall {window.wall_direction.value} not found for room "
                    f"'{window.room_id}'"
                )
                console_logger.warning(f"Window {window.id}: {msg}")
                return msg
        else:
            # Fallback to boundary_label lookup (legacy windows without wall_direction).
            if window.boundary_label not in self.layout.boundary_labels:
                msg = f"boundary '{window.boundary_label}' no longer exists"
                console_logger.warning(f"Window {window.id}: {msg}")
                return msg
            try:
                wall = self._get_wall_by_boundary(window.boundary_label, window.room_id)
            except ValueError as e:
                console_logger.warning(f"Window {window.id}: {e}")
                return str(e)

        # Validate position fits in wall.
        # Window position is LEFT EDGE (matches door/OPEN convention).
        if window.position_along_wall < 0:
            msg = (
                f"position {window.position_along_wall:.2f}m is before wall start "
                f"for {window.width:.2f}m window"
            )
            console_logger.warning(f"Window {window.id}: {msg}")
            return msg
        if window.position_along_wall + window.width > wall.length:
            msg = (
                f"position {window.position_along_wall:.2f}m too close to wall end "
                f"for {window.width:.2f}m window (wall is {wall.length:.2f}m)"
            )
            console_logger.warning(f"Window {window.id}: {msg}")
            return msg

        # Create Opening.
        opening = Opening(
            opening_id=window.id,
            opening_type=OpeningType.WINDOW,
            position_along_wall=window.position_along_wall,
            width=window.width,
            height=window.height,
            sill_height=window.sill_height,
        )

        # Add to wall if not already present.
        if not any(o.opening_id == window.id for o in wall.openings):
            wall.openings.append(opening)

        return None

    def _remove_door_impl(self, door_id: str):
        """Remove a door.

        Fails if removal breaks room connectivity.

        Args:
            door_id: Door identifier to remove.

        Returns:
            Result indicating success or failure.
        """
        from scenecode.floor_plan_agents.tools.floor_plan_tools import Result

        console_logger.info(f"Tool called: remove_door(door_id={door_id})")
        error = self._check_rooms_exist()
        if error:
            return error

        # Find door.
        door_idx = None
        for i, door in enumerate(self.layout.doors):
            if door.id == door_id:
                door_idx = i
                break

        if door_idx is None:
            return self._fail(f"Door '{door_id}' not found.")

        # Check connectivity without this door.
        doors_without = self.layout.doors[:door_idx] + self.layout.doors[door_idx + 1 :]
        is_valid, msg = validate_connectivity(
            self.layout.placed_rooms, doors_without, self.layout.room_specs
        )

        if not is_valid:
            return self._fail(f"Cannot remove door: {msg}")

        # Get door info before removing.
        door = self.layout.doors[door_idx]

        # Remove opening from wall(s).
        wall_a = self._get_wall_by_boundary(door.boundary_label, door.room_a)
        wall_a.openings = [o for o in wall_a.openings if o.opening_id != door_id]
        if door.room_b:
            wall_b = self._get_wall_by_boundary(door.boundary_label, door.room_b)
            wall_b.openings = [o for o in wall_b.openings if o.opening_id != door_id]

        # Remove door.
        self.layout.doors.pop(door_idx)

        # Invalidate geometry for affected rooms (wall openings changed).
        if self.layout.invalidate_room_geometry(door.room_a):
            console_logger.debug(f"Invalidated geometry for room: {door.room_a}")
        if door.room_b and self.layout.invalidate_room_geometry(door.room_b):
            console_logger.debug(f"Invalidated geometry for room: {door.room_b}")

        return Result(success=True, message=f"Removed door '{door_id}'.")

    def _add_window_impl(
        self,
        wall_id: str,
        position: str = "center",
        width: float | None = None,
        height: float | None = None,
        sill_height: float | None = None,
    ):
        """Add a window to an exterior wall.

        Fails if wall is interior or has a door.

        Args:
            wall_id: Exterior wall segment ID.
            position: "left" | "center" | "right".
            width: Window width in valid range (uses config default if not specified).
            height: Window height in valid range (uses config default if not specified).
            sill_height: Height from floor to window bottom (uses config default).

        Returns:
            Result indicating success or failure.
        """
        from scenecode.floor_plan_agents.tools.floor_plan_tools import Result

        # Apply defaults from config.
        cfg = self.door_window_config
        if width is None:
            width = cfg.window_default_width
        if height is None:
            height = cfg.window_default_height
        if sill_height is None:
            sill_height = cfg.window_default_sill_height

        console_logger.info(
            f"Tool called: add_window(wall_id={wall_id}, position={position}, "
            f"width={width}, sill_height={sill_height})"
        )
        error = self._check_rooms_exist()
        if error:
            return error

        if position not in {"left", "center", "right"}:
            return self._fail(
                f"Position must be 'left', 'center', or 'right'. Got: {position}"
            )

        # Validate dimensions.
        if not (cfg.window_width_min <= width <= cfg.window_width_max):
            return self._fail(
                f"Window width must be {cfg.window_width_min}-{cfg.window_width_max}m. "
                f"Got: {width}"
            )

        if not (cfg.window_height_min <= height <= cfg.window_height_max):
            return self._fail(
                f"Window height must be {cfg.window_height_min}-{cfg.window_height_max}m. "
                f"Got: {height}"
            )

        # Check wall exists and is exterior.
        if wall_id not in self.layout.boundary_labels:
            available = ", ".join(sorted(self.layout.boundary_labels.keys()))
            return self._fail(f"Wall '{wall_id}' not found. Available: {available}")

        room_a, room_b, _direction = self.layout.boundary_labels[wall_id]
        if room_b is not None:
            return self._fail(
                f"Wall '{wall_id}' is interior. Windows only on exterior walls."
            )

        # Check if wall faces another room across a small gap.
        nearby_room = self._wall_faces_nearby_room(room_a, wall_id, threshold=0.5)
        if nearby_room:
            return self._fail(
                f"Wall '{wall_id}' faces {nearby_room} across a small gap. "
                f"Windows only allowed on true exterior walls."
            )

        # Check for duplicate window on same wall/position.
        for existing in self.layout.windows:
            if existing.boundary_label == wall_id:
                # Get position segment for existing window.
                wall_length = self._get_wall_length(room_a, wall_id)
                existing_pct = existing.position_along_wall / wall_length
                if position == "left" and existing_pct < SEGMENT_LEFT_END:
                    return self._fail(
                        f"Wall '{wall_id}' already has a window on the left."
                    )
                if (
                    position == "center"
                    and SEGMENT_LEFT_END <= existing_pct <= SEGMENT_RIGHT_START
                ):
                    return self._fail(
                        f"Wall '{wall_id}' already has a window in the center."
                    )
                if position == "right" and existing_pct > SEGMENT_RIGHT_START:
                    return self._fail(
                        f"Wall '{wall_id}' already has a window on the right."
                    )

        # Calculate position with randomization.
        # Windows are inset from wall edges to avoid corners.
        segment_ranges = {
            "left": (WINDOW_EDGE_INSET, SEGMENT_LEFT_END),
            "center": (SEGMENT_LEFT_END, SEGMENT_RIGHT_START),
            "right": (SEGMENT_RIGHT_START, 1.0 - WINDOW_EDGE_INSET),
        }
        start_pct, end_pct = segment_ranges[position]

        wall_length = self._get_wall_length(room_a, wall_id)

        # Calculate position using center-based sampling.
        # Window center must be within segment, but edges must respect wall margins.
        margin = self.door_window_config.window_segment_margin
        half_width = width / 2

        # Valid center range based on wall boundaries (window must fit with margins).
        wall_center_min = margin + half_width
        wall_center_max = wall_length - margin - half_width

        # Segment boundaries for window center.
        segment_center_min = wall_length * start_pct
        segment_center_max = wall_length * end_pct

        # Intersection: center must be in segment AND within wall bounds.
        center_min = max(wall_center_min, segment_center_min)
        center_max = min(wall_center_max, segment_center_max)

        if center_max < center_min:
            return self._fail(
                f"Cannot place {width:.2f}m window in '{position}' segment of "
                f"{wall_length:.2f}m wall. Try a different segment or narrower window."
            )

        # Sample window center, convert to left edge position.
        window_center = random.uniform(center_min, center_max)
        position_along = window_center - half_width

        # Check for overlap with existing doors on this wall.
        window_start = position_along
        window_end = position_along + width
        for door in self.layout.doors:
            if door.boundary_label == wall_id:
                door_start = door.position_exact
                door_end = door.position_exact + door.width
                # Check overlap with separation margin.
                if (
                    window_start < door_end + self.min_opening_separation
                    and window_end > door_start - self.min_opening_separation
                ):
                    return self._fail(
                        f"Window would overlap with door on wall '{wall_id}'. "
                        f"Window: {window_start:.2f}-{window_end:.2f}m, "
                        f"Door: {door_start:.2f}-{door_end:.2f}m. "
                        f"Min separation: {self.min_opening_separation}m. "
                        f"Try a different position."
                    )

        # Check for overlap with existing windows on this wall.
        for existing_window in self.layout.windows:
            if existing_window.boundary_label == wall_id:
                existing_start = existing_window.position_along_wall
                existing_end = (
                    existing_window.position_along_wall + existing_window.width
                )
                # Check overlap with separation margin.
                if (
                    window_start < existing_end + self.min_opening_separation
                    and window_end > existing_start - self.min_opening_separation
                ):
                    return self._fail(
                        f"Window would overlap with existing window "
                        f"'{existing_window.id}' on wall '{wall_id}'. "
                        f"New window: {window_start:.2f}-{window_end:.2f}m, "
                        f"Existing: {existing_start:.2f}-{existing_end:.2f}m. "
                        f"Min separation: {self.min_opening_separation}m. "
                        f"Try a different position or remove the existing window."
                    )

        # Create window.
        # Get wall direction from boundary info for stable lookup.
        _, _, direction_str = self.layout.boundary_labels[wall_id]
        wall_direction = WallDirection(direction_str) if direction_str else None

        window_id = self._next_window_id()
        window = Window(
            id=window_id,
            boundary_label=wall_id,
            position_along_wall=position_along,
            room_id=room_a,
            wall_direction=wall_direction,
            width=width,
            height=height,
            sill_height=sill_height,
        )

        # Add Opening to wall for rendering/ASCII.
        error = self._apply_window_to_wall(window)
        if error:
            return self._fail(f"Failed to add window at wall {wall_id}: {error}")

        self.layout.windows.append(window)

        # Invalidate geometry for affected room (wall openings changed).
        if self.layout.invalidate_room_geometry(room_a):
            console_logger.debug(f"Invalidated geometry for room: {room_a}")

        return Result(
            success=True,
            message=f"Added window '{window_id}' at wall {wall_id} ({position}).",
        )

    def _remove_window_impl(self, window_id: str):
        """Remove a window.

        Args:
            window_id: Window identifier to remove.

        Returns:
            Result indicating success or failure.
        """
        from scenecode.floor_plan_agents.tools.floor_plan_tools import Result

        console_logger.info(f"Tool called: remove_window(window_id={window_id})")
        error = self._check_rooms_exist()
        if error:
            return error

        # Find window.
        window_idx = None
        for i, window in enumerate(self.layout.windows):
            if window.id == window_id:
                window_idx = i
                break

        if window_idx is None:
            return self._fail(f"Window '{window_id}' not found.")

        # Get window info before removing.
        window = self.layout.windows[window_idx]

        # Remove opening from wall using stable lookup.
        if window.wall_direction:
            placed_room = next(
                (r for r in self.layout.placed_rooms if r.room_id == window.room_id),
                None,
            )
            if placed_room:
                wall = next(
                    (
                        w
                        for w in placed_room.walls
                        if w.direction == window.wall_direction
                    ),
                    None,
                )
                if wall:
                    wall.openings = [
                        o for o in wall.openings if o.opening_id != window_id
                    ]
        elif window.boundary_label in self.layout.boundary_labels:
            # Fallback to boundary_label lookup.
            try:
                wall = self._get_wall_by_boundary(window.boundary_label, window.room_id)
                wall.openings = [o for o in wall.openings if o.opening_id != window_id]
            except ValueError:
                pass  # Wall no longer exists, just remove window.

        # Remove window.
        self.layout.windows.pop(window_idx)

        # Invalidate geometry for affected room (wall openings changed).
        if self.layout.invalidate_room_geometry(window.room_id):
            console_logger.debug(f"Invalidated geometry for room: {window.room_id}")

        return Result(success=True, message=f"Removed window '{window_id}'.")
