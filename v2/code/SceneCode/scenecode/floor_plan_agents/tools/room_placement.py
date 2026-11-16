"""Slot-based room placement algorithm with backtracking search.

This module implements an optimal room placement algorithm that satisfies
adjacency constraints while minimizing bounding box size and maintaining
layout stability across iterative edits.

Algorithm Overview
------------------
1. **Topological Sort**: Rooms sorted by adjacency dependencies. Anchor rooms
   (no adjacencies) placed first, then connector rooms that link them.

2. **First Room at Origin**: The first sorted room is placed at (0, 0).

3. **Slot-Based Attachment**: Each placed room exposes 4 edge slots (N/S/E/W).
   New rooms attach to slots of rooms they must be adjacent to.

4. **Backtracking Search**: Explores all valid placements recursively:
   - For each unplaced room, generate all valid candidate positions
   - Sort candidates by local score (adjacency satisfaction, compactness)
   - Recurse on each candidate, tracking the best complete layout found
   - Prune branches on timeout (anytime algorithm behavior)

5. **Global Scoring**: Complete layouts scored by:
   - Compactness: ratio of room area to bounding box area (higher = better)
   - Stability: proximity to previous positions (for iterative editing)

Properties
----------
- **Optimality**: Finds optimal layout within *fixed room ordering* and *discrete
  position space*. Two limitations:
  1. Room order fixed by topological sort (anchor rooms first, then connectors).
     Different orderings could yield different layouts, but exploring all O(n!)
     orderings is intractable.
  2. Positions sampled at 11 evenly-spaced points per slot edge (0%, 10%, ..., 100%).
  With ≤10 rooms and 5s timeout, typically explores all valid positions for the
  fixed ordering.

- **Anytime Behavior**: Returns best-found-so-far if timeout exceeded. Search
  is best-first (candidates sorted by score), so early layouts are good.

- **Soundness**: All returned layouts satisfy adjacency constraints. Candidates
  that don't satisfy required adjacencies to placed rooms are rejected during
  scoring (score=0), ensuring only valid placements are explored.

- **Completeness**: If a valid layout exists for the fixed room ordering and
  discrete position space, the algorithm will find it (given sufficient time).
  Returns PlacementError only when no valid layout exists within these constraints.

Features
--------
- 90° room rotation: Automatically tries rotated orientation for better fit
- Multi-adjacency: Corner-aligned positions for rooms adjacent to 2+ others
- Layout stability: Configurable preference for positions near previous layout
"""

import logging
import math
import time

from dataclasses import dataclass, field

from scenecode.agent_utils.house import PlacedRoom, RoomSpec, Wall, WallDirection

console_logger = logging.getLogger(__name__)


class PlacementError(Exception):
    """Raised when room placement fails."""


@dataclass
class ScoringWeights:
    """Weights for global layout scoring."""

    compactness: float = 1.0
    """Weight for bounding box minimization (higher = prefer compact layouts)."""

    stability: float = 1.0
    """Weight for staying near previous positions (higher = more stable)."""


@dataclass
class LayoutScoreBreakdown:
    """Breakdown of layout scoring components (all values are weighted)."""

    compactness: float
    stability: float
    total: float


@dataclass
class Slot:
    """An attachment slot on a placed room's edge.

    A slot represents an available edge where a new room can attach.
    """

    room_id: str
    """ID of the room that owns this slot."""

    direction: WallDirection
    """Direction of this slot (N/S/E/W edge of the room)."""

    start: float
    """Start coordinate along the slot edge."""

    end: float
    """End coordinate along the slot edge."""

    anchor_pos: tuple[float, float]
    """Position of the room that owns this slot (for reference)."""

    anchor_width: float
    """Width of the room that owns this slot."""

    anchor_depth: float
    """Depth of the room that owns this slot."""


@dataclass
class PlacementConfig:
    """Configuration for room placement algorithm."""

    min_shared_edge: float = 1.0
    """Minimum shared edge length for adjacency (meters)."""

    timeout_seconds: float = 5.0
    """Timeout for backtracking search. Returns best layout found when exceeded."""

    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    """Weights for global layout scoring (compactness, stability)."""

    previous_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Previous room positions for layout stability. Map of room_id to (x, y) position."""

    free_room_ids: set[str] = field(default_factory=set)
    """Room IDs that should have no position bias (can move freely).
    Typically includes rooms being resized or made adjacent."""

    exterior_wall_clearance_m: float = 20.0
    """Clearance zone for exterior_walls constraint (meters).

    Rooms with exterior_walls specified will have clearance zones created
    extending this distance outward from the specified walls. No other room
    can be placed within these zones, ensuring the walls remain accessible
    for exterior doors.
    """


@dataclass
class _SearchState:
    """Mutable state for backtracking search."""

    best_layout: list[PlacedRoom] | None = None
    """Best complete layout found so far."""

    best_score: float = float("-inf")
    """Score of the best layout."""

    start_time: float = 0.0
    """Search start time (from time.time())."""

    timed_out: bool = False
    """Whether the search has exceeded the timeout."""

    timeout_seconds: float = 5.0
    """Maximum search time before returning best-found-so-far."""


def _global_layout_score(
    placed_rooms: list[PlacedRoom], config: PlacementConfig
) -> LayoutScoreBreakdown:
    """Score a complete layout (higher = better).

    Considers:
    - Compactness: ratio of total room area to bounding box area
    - Stability: distance from previous positions (if provided)

    Args:
        placed_rooms: Complete room layout to score.
        config: Placement configuration with scoring weights and previous positions.

    Returns:
        Score breakdown with compactness, stability, and total scores.
    """
    if not placed_rooms:
        return LayoutScoreBreakdown(compactness=0.0, stability=0.0, total=0.0)

    weights = config.scoring_weights

    # Compactness score: room_area / bounding_box_area.
    total_room_area = sum(r.width * r.depth for r in placed_rooms)
    min_x = min(r.position[0] for r in placed_rooms)
    max_x = max(r.position[0] + r.width for r in placed_rooms)
    min_y = min(r.position[1] for r in placed_rooms)
    max_y = max(r.position[1] + r.depth for r in placed_rooms)
    bounding_box_area = (max_x - min_x) * (max_y - min_y)

    # Avoid division by zero (shouldn't happen with valid rooms).
    if bounding_box_area < 0.001:
        compactness_ratio = 1.0
    else:
        compactness_ratio = total_room_area / bounding_box_area

    # Scale scores to comparable ranges so neither dominates when weights are equal.
    # Compactness: ratio in [0, 1] -> scaled to [0, 100].
    compactness_score = compactness_ratio * 100.0 * weights.compactness

    # Stability: sum of per-room bonuses. Each room contributes up to 50 points
    # when at its previous position, decaying exponentially with distance.
    stability_score = 0.0
    if config.previous_positions and weights.stability > 0:
        for room in placed_rooms:
            if room.room_id in config.free_room_ids:
                continue  # Free rooms don't contribute to stability.
            if room.room_id in config.previous_positions:
                prev_pos = config.previous_positions[room.room_id]
                # Distance between room centers.
                curr_center = (
                    room.position[0] + room.width / 2,
                    room.position[1] + room.depth / 2,
                )
                prev_center = (
                    prev_pos[0] + room.width / 2,
                    prev_pos[1] + room.depth / 2,
                )
                distance = math.sqrt(
                    (curr_center[0] - prev_center[0]) ** 2
                    + (curr_center[1] - prev_center[1]) ** 2
                )
                # Exponential decay: full bonus (50) at distance=0, ~37% at distance=2m.
                scale = 2.0
                stability_score += (
                    math.exp(-distance / scale) * 50.0 * weights.stability
                )

    return LayoutScoreBreakdown(
        compactness=compactness_score,
        stability=stability_score,
        total=compactness_score + stability_score,
    )


def _topological_sort_rooms(room_specs: list[RoomSpec]) -> list[RoomSpec]:
    """Sort rooms for optimal placement order.

    Strategy for linear layouts like A-B-C where B connects A and C:
    1. Place anchor rooms (no adjacencies, not blocking connectors) first
    2. Place connector rooms (rooms with adjacencies, at least one satisfied) next
    3. Place remaining anchor rooms that were blocked by connectors last

    This ensures connectors like B are placed before their unmet deps (like C),
    so C can be placed adjacent to B rather than adjacent to A.

    Args:
        room_specs: List of room specifications.

    Returns:
        Sorted list optimized for successful placement.
    """
    if not room_specs:
        return []

    # Build adjacency lookup.
    sorted_list: list[RoomSpec] = []
    placed_ids: set[str] = set()
    remaining = list(room_specs)

    while remaining:
        # Capture remaining specs before sort (list.sort() empties list temporarily).
        remaining_snapshot = list(remaining)

        # Score each remaining room for placement priority.
        # Lower score = higher priority.
        def placement_score(spec: RoomSpec) -> tuple[int, int, float]:
            # Count how many adjacencies are already placed.
            placed_adj = sum(1 for a in spec.connections if a in placed_ids)
            total_adj = len(spec.connections)
            unmet = total_adj - placed_adj

            # Find reverse dependencies: unplaced rooms that require this spec.
            # Use snapshot because 'remaining' is empty during sort.
            reverse_deps = [
                s for s in remaining_snapshot if spec.room_id in s.connections
            ]

            # For rooms with no forward adjacencies.
            if total_adj == 0:
                # Check if any unplaced connector room requires us and is ready.
                # A connector is "ready" if it has at least one dep already placed.
                for r_spec in reverse_deps:
                    r_placed = sum(1 for a in r_spec.connections if a in placed_ids)
                    if r_placed > 0:
                        # This connector is ready - it should be placed before us.
                        # We wait so the connector can be positioned first.
                        return (2, 0, -(spec.width * spec.length))
                # No ready connectors blocking us - we're a true anchor.
                return (0, 0, -(spec.width * spec.length))

            # Priority 1: At least one adjacency placed (can attach to graph).
            if placed_adj > 0:
                return (1, unmet, -(spec.width * spec.length))

            # Priority 3: All adjacencies unplaced.
            # These need their deps placed first.
            return (3, total_adj, -(spec.width * spec.length))

        # Sort remaining by placement score.
        remaining.sort(key=placement_score)

        # Place the best candidate.
        chosen = remaining[0]
        sorted_list.append(chosen)
        placed_ids.add(chosen.room_id)
        remaining.remove(chosen)

    return sorted_list


def place_rooms(
    room_specs: list[RoomSpec], config: PlacementConfig | None = None
) -> list[PlacedRoom]:
    """Place rooms to satisfy adjacency constraints.

    Uses backtracking search with timeout to find globally optimal (or near-optimal)
    room layouts. Returns the best layout found within the timeout.

    For layout stability during iterative edits, pass previous_positions and
    free_room_ids in config. The algorithm will prefer positions close to
    previous locations for non-free rooms.

    Args:
        room_specs: List of room specifications with dimensions and adjacencies.
        config: Placement configuration. If None, uses default PlacementConfig.

    Returns:
        List of PlacedRoom with computed positions and walls.

    Raises:
        PlacementError: If no valid layout can be found.
    """
    if config is None:
        config = PlacementConfig()

    if not room_specs:
        return []

    if len(room_specs) == 1:
        # Single room: place at origin.
        spec = room_specs[0]
        return [_create_placed_room(spec=spec, position=(0.0, 0.0))]

    return _place_rooms_attempt(room_specs=room_specs, config=config)


def _place_rooms_attempt(
    room_specs: list[RoomSpec], config: PlacementConfig
) -> list[PlacedRoom]:
    """Find optimal room placement using backtracking with timeout.

    Args:
        room_specs: Room specifications.
        config: Placement configuration.

    Returns:
        List of placed rooms (best layout found within timeout).

    Raises:
        PlacementError: If no valid layout found.
    """
    # Topological sort: place rooms with fewer/no dependencies first.
    sorted_specs = _topological_sort_rooms(room_specs)

    # Build reverse adjacency map: which rooms require each room.
    reverse_adj_map: dict[str, list[str]] = {s.room_id: [] for s in room_specs}
    for s in room_specs:
        for adj_id in s.connections:
            if adj_id in reverse_adj_map:
                reverse_adj_map[adj_id].append(s.room_id)

    # Build room spec map for exterior_walls constraint checking.
    room_spec_map: dict[str, RoomSpec] = {s.room_id: s for s in room_specs}

    # Place first room at origin (should have no/fewest adjacencies).
    first_spec = sorted_specs[0]
    first_room = _create_placed_room(first_spec, (0.0, 0.0))
    initial_placed = [first_room]
    initial_slots = _get_room_slots(first_room)

    # Initialize search state.
    start_time = time.time()
    state = _SearchState(start_time=start_time, timeout_seconds=config.timeout_seconds)

    # Run backtracking search.
    _place_rooms_backtrack(
        remaining_specs=sorted_specs[1:],
        placed_rooms=initial_placed,
        available_slots=initial_slots,
        config=config,
        reverse_adj_map=reverse_adj_map,
        room_spec_map=room_spec_map,
        state=state,
    )

    # Calculate search duration.
    elapsed_time = time.time() - start_time

    if state.best_layout is None:
        console_logger.warning(
            f"Room placement failed after {elapsed_time:.2f}s. "
            f"Rooms: {[s.room_id for s in room_specs]}. Timed out: {state.timed_out}"
        )
        raise PlacementError(
            f"Room placement failed: no valid layout found. "
            f"Rooms: {[s.room_id for s in room_specs]}. "
            f"Timed out: {state.timed_out}"
        )

    # Log search results with score breakdown.
    score_breakdown = _global_layout_score(
        placed_rooms=state.best_layout, config=config
    )
    timeout_msg = "timed out" if state.timed_out else "completed"
    console_logger.info(
        f"Room placement {timeout_msg} in {elapsed_time:.2f}s "
        f"(compactness={score_breakdown.compactness:.1f}, "
        f"stability={score_breakdown.stability:.1f}, "
        f"total={score_breakdown.total:.1f})"
    )

    # Update wall connectivity.
    _update_wall_connectivity(state.best_layout)

    return state.best_layout


def _get_all_candidates(
    spec: RoomSpec,
    placed_rooms: list[PlacedRoom],
    available_slots: list[Slot],
    config: PlacementConfig,
    reverse_adj_map: dict[str, list[str]] | None = None,
    room_spec_map: dict[str, RoomSpec] | None = None,
) -> list[tuple[PlacedRoom, float]]:
    """Get all valid placement candidates for a room.

    Tries both orientations (original and 90° rotated) to maximize placement options.

    Args:
        spec: Room specification to place.
        placed_rooms: Already placed rooms.
        available_slots: Available attachment slots.
        config: Placement configuration.
        reverse_adj_map: Map of room_id to list of rooms that require it.
        room_spec_map: Map of room_id to RoomSpec for exterior_walls checking.

    Returns:
        List of (PlacedRoom, score) tuples for all valid placements.
    """
    candidates: list[tuple[PlacedRoom, float]] = []

    # Determine if this is a multi-adjacency case (needs corner placement).
    required_placed = [r for r in placed_rooms if r.room_id in spec.connections]
    is_multi_adjacency = len(required_placed) >= 2

    # Try both orientations: original and 90° rotated.
    # Original: X=spec.length, Y=spec.width
    # Rotated:  X=spec.width, Y=spec.length
    orientations = [
        (spec.length, spec.width),  # Original orientation.
    ]
    # Only add rotated if dimensions differ (avoid duplicate work for square rooms).
    if abs(spec.length - spec.width) > 0.001:
        orientations.append((spec.width, spec.length))  # 90° rotated.

    for room_x, room_y in orientations:
        for slot in available_slots:
            # Check if this slot's owner is in spec's connections.
            if spec.connections and slot.room_id not in spec.connections:
                # If room has specific adjacency requirements, only consider matching
                # slots.
                continue

            # Try placing room at this slot with current orientation.
            # For multi-adjacency, also consider corner-aligned positions.
            positions = _get_candidate_positions(
                spec=spec,
                slot=slot,
                min_shared_edge=config.min_shared_edge,
                placed_rooms=placed_rooms if is_multi_adjacency else None,
                room_x=room_x,
                room_y=room_y,
            )

            for pos in positions:
                room = _create_placed_room(
                    spec=spec, position=pos, room_width=room_x, room_depth=room_y
                )

                # Check for overlaps with existing rooms.
                if _has_overlap(room=room, placed_rooms=placed_rooms):
                    continue

                # Check exterior_walls clearance constraints.
                if room_spec_map and _violates_exterior_clearance(
                    candidate=room,
                    candidate_spec=spec,
                    placed_rooms=placed_rooms,
                    room_spec_map=room_spec_map,
                    clearance=config.exterior_wall_clearance_m,
                ):
                    continue

                # Check adjacencies with placed rooms.
                score = _score_placement(
                    room=room,
                    spec=spec,
                    placed_rooms=placed_rooms,
                    config=config,
                    reverse_adj_map=reverse_adj_map,
                )
                # Only add valid placements to candidates.
                if score > 0:
                    candidates.append((room, score))

    return candidates


def _place_rooms_backtrack(
    remaining_specs: list[RoomSpec],
    placed_rooms: list[PlacedRoom],
    available_slots: list[Slot],
    config: PlacementConfig,
    reverse_adj_map: dict[str, list[str]],
    room_spec_map: dict[str, RoomSpec],
    state: _SearchState,
) -> None:
    """Recursive backtracking search for optimal layout.

    Explores all valid layouts, updating state.best_layout when a better complete
    layout is found. Stops early if timeout is exceeded.

    Args:
        remaining_specs: Room specs still to be placed.
        placed_rooms: Rooms placed so far in this branch.
        available_slots: Available attachment slots.
        config: Placement configuration.
        reverse_adj_map: Map of room_id to list of rooms that require it.
        room_spec_map: Map of room_id to RoomSpec for exterior_walls checking.
        state: Mutable search state (best layout, score, timeout tracking).
    """
    # Check timeout.
    if time.time() - state.start_time > state.timeout_seconds:
        state.timed_out = True
        return

    # Base case: all rooms placed - score and possibly update best.
    if not remaining_specs:
        score = _global_layout_score(placed_rooms=placed_rooms, config=config)
        if score.total > state.best_score:
            state.best_score = score.total
            state.best_layout = list(placed_rooms)  # Copy the list.
        return

    # Get next room to place.
    spec = remaining_specs[0]
    rest = remaining_specs[1:]

    # Get all valid candidates for this room.
    candidates = _get_all_candidates(
        spec=spec,
        placed_rooms=placed_rooms,
        available_slots=available_slots,
        config=config,
        reverse_adj_map=reverse_adj_map,
        room_spec_map=room_spec_map,
    )

    # Sort candidates by score (highest first) for best-first exploration.
    candidates.sort(key=lambda x: -x[1])

    # Recurse on each candidate.
    for room, _ in candidates:
        if state.timed_out:
            return

        # Extend placed rooms and slots for this branch.
        new_placed = placed_rooms + [room]
        new_slots = available_slots + _get_room_slots(room)

        _place_rooms_backtrack(
            remaining_specs=rest,
            placed_rooms=new_placed,
            available_slots=new_slots,
            config=config,
            reverse_adj_map=reverse_adj_map,
            room_spec_map=room_spec_map,
            state=state,
        )


def _get_candidate_positions(
    spec: RoomSpec,
    slot: Slot,
    min_shared_edge: float,
    placed_rooms: list[PlacedRoom] | None = None,
    room_x: float | None = None,
    room_y: float | None = None,
) -> list[tuple[float, float]]:
    """Get candidate positions for placing a room at a slot.

    The room must share at least min_shared_edge with the slot's anchor room.
    Valid position range along the slot:
    - Horizontal slots (N/S): x varies from (anchor_x - room_x + min_shared_edge)
      to (anchor_x + anchor_width - min_shared_edge)
    - Vertical slots (E/W): y varies similarly

    Args:
        spec: Room to place.
        slot: Slot to attach to.
        min_shared_edge: Minimum shared edge length required.
        placed_rooms: If provided, also generate corner-aligned positions
            that could touch other placed rooms (for multi-adjacency).
        room_x: Explicit X dimension (overrides spec.length if provided).
        room_y: Explicit Y dimension (overrides spec.width if provided).

    Returns:
        List of candidate (x, y) positions.
    """
    positions = []

    # Room dimensions using length as x-dimension and width as y-dimension.
    if room_x is None:
        room_x = spec.length
    if room_y is None:
        room_y = spec.width

    # Calculate positions based on slot direction.
    if slot.direction == WallDirection.NORTH:
        # Slot is on north edge of anchor room.
        # New room attaches from north (its south edge touches slot).
        y = slot.anchor_pos[1] + slot.anchor_depth
        # Valid x range for sufficient edge sharing.
        x_min = slot.anchor_pos[0] - room_x + min_shared_edge
        x_max = slot.anchor_pos[0] + slot.anchor_width - min_shared_edge
        positions.extend(
            _generate_positions_in_range(
                var_min=x_min, var_max=x_max, fixed_coord=y, vary_axis="x"
            )
        )

    elif slot.direction == WallDirection.SOUTH:
        # New room attaches from south (its north edge touches slot).
        y = slot.anchor_pos[1] - room_y
        x_min = slot.anchor_pos[0] - room_x + min_shared_edge
        x_max = slot.anchor_pos[0] + slot.anchor_width - min_shared_edge
        positions.extend(
            _generate_positions_in_range(
                var_min=x_min, var_max=x_max, fixed_coord=y, vary_axis="x"
            )
        )

    elif slot.direction == WallDirection.EAST:
        # New room attaches from east (its west edge touches slot).
        x = slot.anchor_pos[0] + slot.anchor_width
        y_min = slot.anchor_pos[1] - room_y + min_shared_edge
        y_max = slot.anchor_pos[1] + slot.anchor_depth - min_shared_edge
        positions.extend(
            _generate_positions_in_range(
                var_min=y_min, var_max=y_max, fixed_coord=x, vary_axis="y"
            )
        )

    elif slot.direction == WallDirection.WEST:
        # New room attaches from west (its east edge touches slot).
        x = slot.anchor_pos[0] - room_x
        y_min = slot.anchor_pos[1] - room_y + min_shared_edge
        y_max = slot.anchor_pos[1] + slot.anchor_depth - min_shared_edge
        positions.extend(
            _generate_positions_in_range(
                var_min=y_min, var_max=y_max, fixed_coord=x, vary_axis="y"
            )
        )

    # Add corner-aligned positions for multi-adjacency.
    if placed_rooms:
        positions.extend(
            _get_corner_aligned_positions(
                spec=spec,
                slot=slot,
                placed_rooms=placed_rooms,
                room_x=room_x,
                room_y=room_y,
            )
        )

    return positions


def _generate_positions_in_range(
    var_min: float,
    var_max: float,
    fixed_coord: float,
    vary_axis: str,
) -> list[tuple[float, float]]:
    """Generate evenly spaced positions within a valid range.

    Args:
        var_min: Minimum value for the varying coordinate.
        var_max: Maximum value for the varying coordinate.
        fixed_coord: Fixed coordinate value.
        vary_axis: "x" if x varies, "y" if y varies.

    Returns:
        List of (x, y) positions.
    """
    positions = []

    # Generate positions at different points in the range.
    if var_max >= var_min:
        # Valid range exists.
        range_size = var_max - var_min
        # Generate 11 evenly spaced positions (0%, 10%, 20%, ..., 100%).
        for i in range(11):
            fraction = i / 10.0
            var_val = var_min + fraction * range_size
            if vary_axis == "x":
                positions.append((var_val, fixed_coord))
            else:
                positions.append((fixed_coord, var_val))

    return positions


def _get_corner_aligned_positions(
    spec: RoomSpec,
    slot: Slot,
    placed_rooms: list[PlacedRoom],
    room_x: float | None = None,
    room_y: float | None = None,
) -> list[tuple[float, float]]:
    """Get positions aligned with corners of other placed rooms.

    For multi-adjacency cases, tries to position the new room so it touches
    both the slot owner and other required adjacent rooms.

    Args:
        spec: Room to place.
        slot: Slot to attach to.
        placed_rooms: All placed rooms.
        room_x: Explicit X dimension (overrides spec.length if provided).
        room_y: Explicit Y dimension (overrides spec.width if provided).

    Returns:
        List of corner-aligned (x, y) positions.
    """
    positions = []
    if room_x is None:
        room_x = spec.length
    if room_y is None:
        room_y = spec.width

    # Find other rooms we need to be adjacent to.
    other_required = [
        r
        for r in placed_rooms
        if r.room_id in spec.connections and r.room_id != slot.room_id
    ]

    for other in other_required:
        # Calculate positions that would align our room's edges with the other room.
        other_left = other.position[0]
        other_right = other.position[0] + other.width
        other_bottom = other.position[1]
        other_top = other.position[1] + other.depth

        if slot.direction == WallDirection.NORTH:
            y = slot.anchor_pos[1] + slot.anchor_depth
            # Align left edge with other room's right edge.
            positions.append((other_right, y))
            # Align right edge with other room's left edge.
            positions.append((other_left - room_x, y))
            # Align left edge with other room's left edge.
            positions.append((other_left, y))
            # Align right edge with other room's right edge.
            positions.append((other_right - room_x, y))

        elif slot.direction == WallDirection.SOUTH:
            y = slot.anchor_pos[1] - room_y
            positions.append((other_right, y))
            positions.append((other_left - room_x, y))
            positions.append((other_left, y))
            positions.append((other_right - room_x, y))

        elif slot.direction == WallDirection.EAST:
            x = slot.anchor_pos[0] + slot.anchor_width
            # Align top edge with other room's bottom edge.
            positions.append((x, other_bottom - room_y))
            # Align bottom edge with other room's top edge.
            positions.append((x, other_top))
            # Align top edge with other room's top edge.
            positions.append((x, other_top - room_y))
            # Align bottom edge with other room's bottom edge.
            positions.append((x, other_bottom))

        elif slot.direction == WallDirection.WEST:
            x = slot.anchor_pos[0] - room_x
            positions.append((x, other_bottom - room_y))
            positions.append((x, other_top))
            positions.append((x, other_top - room_y))
            positions.append((x, other_bottom))

    return positions


def _score_placement(
    room: PlacedRoom,
    spec: RoomSpec,
    placed_rooms: list[PlacedRoom],
    config: PlacementConfig,
    reverse_adj_map: dict[str, list[str]] | None = None,
) -> float:
    """Score a placement based on adjacency satisfaction and compactness.

    Args:
        room: Proposed room placement.
        spec: Room specification with requirements.
        placed_rooms: Already placed rooms.
        config: Placement configuration.
        reverse_adj_map: Map of room_id to list of rooms that require it.

    Returns:
        Score (higher is better), 0 if invalid.
    """
    score = 100.0  # Base score.

    # Check required adjacencies (all connections require physical adjacency).
    required_adjacent = list(spec.connections.keys())

    if required_adjacent:
        satisfied = 0
        for adj_id in required_adjacent:
            adj_room = next((r for r in placed_rooms if r.room_id == adj_id), None)
            if adj_room and rooms_share_edge(room, adj_room, config.min_shared_edge):
                satisfied += 1
                score += 50.0  # Bonus for each satisfied adjacency.

        # If any required adjacency is not satisfied, reject.
        if satisfied < len(required_adjacent):
            # Check if the unmatched adjacencies are to unplaced rooms.
            placed_ids = {r.room_id for r in placed_rooms}
            unplaced_adjacencies = [a for a in required_adjacent if a not in placed_ids]
            # If all unsatisfied adjacencies are to unplaced rooms, still valid.
            if satisfied < len(required_adjacent) - len(unplaced_adjacencies):
                return 0.0  # Required adjacency to placed room not satisfied.

    # Check reverse adjacencies: placed rooms that require this room.
    # Give bonus for satisfying these (e.g., C placed adjacent to B when B needs C).
    if reverse_adj_map:
        requiring_rooms = reverse_adj_map.get(spec.room_id, [])
        for req_id in requiring_rooms:
            req_room = next((r for r in placed_rooms if r.room_id == req_id), None)
            if req_room and rooms_share_edge(room, req_room, config.min_shared_edge):
                score += 75.0  # Higher bonus for satisfying reverse adjacency.

    # Compactness bonus: prefer positions closer to center of mass.
    # This is a local heuristic for candidate ordering; global compactness is
    # evaluated by _global_layout_score on complete layouts.
    if placed_rooms:
        center_x = sum(r.position[0] + r.width / 2 for r in placed_rooms) / len(
            placed_rooms
        )
        center_y = sum(r.position[1] + r.depth / 2 for r in placed_rooms) / len(
            placed_rooms
        )
        room_center_x = room.position[0] + room.width / 2
        room_center_y = room.position[1] + room.depth / 2
        distance = math.sqrt(
            (room_center_x - center_x) ** 2 + (room_center_y - center_y) ** 2
        )
        score -= distance * 2  # Penalize distance from center.

    return score


def _has_overlap(room: PlacedRoom, placed_rooms: list[PlacedRoom]) -> bool:
    """Check if room overlaps with any placed room.

    Args:
        room: Room to check.
        placed_rooms: Existing rooms.

    Returns:
        True if any overlap detected.
    """
    for other in placed_rooms:
        if rooms_overlap(room_a=room, room_b=other):
            return True
    return False


def rooms_overlap(room_a: PlacedRoom, room_b: PlacedRoom) -> bool:
    """Check if two rooms have overlapping interiors.

    Args:
        room_a: First room.
        room_b: Second room.

    Returns:
        True if rooms overlap (share interior space).
    """
    # Room A bounds.
    a_min_x = room_a.position[0]
    a_max_x = room_a.position[0] + room_a.width
    a_min_y = room_a.position[1]
    a_max_y = room_a.position[1] + room_a.depth

    # Room B bounds.
    b_min_x = room_b.position[0]
    b_max_x = room_b.position[0] + room_b.width
    b_min_y = room_b.position[1]
    b_max_y = room_b.position[1] + room_b.depth

    # Check for overlap (strict inequality - touching edges are OK).
    x_overlap = a_min_x < b_max_x and a_max_x > b_min_x
    y_overlap = a_min_y < b_max_y and a_max_y > b_min_y

    return x_overlap and y_overlap


def _get_exterior_clearance_zones(
    room: PlacedRoom, spec: RoomSpec, clearance: float
) -> list[tuple[float, float, float, float]]:
    """Get rectangular clearance zones for exterior_walls.

    Each exterior_walls direction creates a forbidden zone extending outward.
    This prevents rooms from blocking exterior access either by direct adjacency
    or by wrapping around.

    Args:
        room: Placed room to compute zones for.
        spec: Room specification with exterior_walls constraints.
        clearance: Clearance distance in meters.

    Returns:
        List of (min_x, min_y, max_x, max_y) tuples representing forbidden zones.
    """
    zones: list[tuple[float, float, float, float]] = []
    for direction in spec.exterior_walls:
        if direction == WallDirection.WEST:
            # Zone extends clearance meters to the west.
            zones.append(
                (
                    room.position[0] - clearance,
                    room.position[1],
                    room.position[0],
                    room.position[1] + room.depth,
                )
            )
        elif direction == WallDirection.EAST:
            # Zone extends clearance meters to the east.
            zones.append(
                (
                    room.position[0] + room.width,
                    room.position[1],
                    room.position[0] + room.width + clearance,
                    room.position[1] + room.depth,
                )
            )
        elif direction == WallDirection.SOUTH:
            # Zone extends clearance meters to the south.
            zones.append(
                (
                    room.position[0],
                    room.position[1] - clearance,
                    room.position[0] + room.width,
                    room.position[1],
                )
            )
        elif direction == WallDirection.NORTH:
            # Zone extends clearance meters to the north.
            zones.append(
                (
                    room.position[0],
                    room.position[1] + room.depth,
                    room.position[0] + room.width,
                    room.position[1] + room.depth + clearance,
                )
            )
    return zones


def _overlaps_zone(room: PlacedRoom, zone: tuple[float, float, float, float]) -> bool:
    """Check if room overlaps with a clearance zone.

    Args:
        room: Room to check.
        zone: Clearance zone as (min_x, min_y, max_x, max_y).

    Returns:
        True if room overlaps with the zone.
    """
    z_min_x, z_min_y, z_max_x, z_max_y = zone
    r_min_x = room.position[0]
    r_max_x = room.position[0] + room.width
    r_min_y = room.position[1]
    r_max_y = room.position[1] + room.depth

    # Overlap if ranges intersect in both X and Y (strict inequality).
    x_overlap = r_min_x < z_max_x and r_max_x > z_min_x
    y_overlap = r_min_y < z_max_y and r_max_y > z_min_y
    return x_overlap and y_overlap


def _violates_exterior_clearance(
    candidate: PlacedRoom,
    candidate_spec: RoomSpec,
    placed_rooms: list[PlacedRoom],
    room_spec_map: dict[str, RoomSpec],
    clearance: float,
) -> bool:
    """Check if placement violates any exterior_walls clearance zones.

    Checks bidirectionally:
    - Candidate cannot be in any placed room's clearance zones.
    - Placed rooms cannot be in candidate's clearance zones.

    Args:
        candidate: Room placement candidate to check.
        candidate_spec: Specification for the candidate room.
        placed_rooms: Already placed rooms.
        room_spec_map: Map of room_id to RoomSpec for all rooms.
        clearance: Clearance distance in meters.

    Returns:
        True if placement violates exterior_walls constraints.
    """
    # Check if candidate is in any placed room's clearance zones.
    for placed in placed_rooms:
        placed_spec = room_spec_map.get(placed.room_id)
        if placed_spec and placed_spec.exterior_walls:
            zones = _get_exterior_clearance_zones(
                room=placed, spec=placed_spec, clearance=clearance
            )
            for zone in zones:
                if _overlaps_zone(room=candidate, zone=zone):
                    return True

    # Check if placed rooms are in candidate's clearance zones.
    if candidate_spec.exterior_walls:
        zones = _get_exterior_clearance_zones(
            room=candidate, spec=candidate_spec, clearance=clearance
        )
        for zone in zones:
            for placed in placed_rooms:
                if _overlaps_zone(room=placed, zone=zone):
                    return True

    return False


def rooms_share_edge(
    room_a: PlacedRoom, room_b: PlacedRoom, min_overlap: float = 0.0
) -> bool:
    """Check if two rooms share an edge with at least min_overlap length.

    Args:
        room_a: First room.
        room_b: Second room.
        min_overlap: Minimum shared edge length.

    Returns:
        True if rooms share sufficient edge.
    """
    # Room A bounds.
    a_min_x = room_a.position[0]
    a_max_x = room_a.position[0] + room_a.width
    a_min_y = room_a.position[1]
    a_max_y = room_a.position[1] + room_a.depth

    # Room B bounds.
    b_min_x = room_b.position[0]
    b_max_x = room_b.position[0] + room_b.width
    b_min_y = room_b.position[1]
    b_max_y = room_b.position[1] + room_b.depth

    # Check for shared vertical edge (A's east = B's west or vice versa).
    # Also check that rooms actually overlap in Y (not just touching corners).
    if abs(a_max_x - b_min_x) < 0.001 or abs(b_max_x - a_min_x) < 0.001:
        # Y ranges must have positive overlap (not just touching at a point).
        y_overlap_start = max(a_min_y, b_min_y)
        y_overlap_end = min(a_max_y, b_max_y)
        overlap = y_overlap_end - y_overlap_start
        if overlap > 0.001 and overlap >= min_overlap:
            return True

    # Check for shared horizontal edge (A's north = B's south or vice versa).
    # Also check that rooms actually overlap in X (not just touching corners).
    if abs(a_max_y - b_min_y) < 0.001 or abs(b_max_y - a_min_y) < 0.001:
        # X ranges must have positive overlap (not just touching at a point).
        x_overlap_start = max(a_min_x, b_min_x)
        x_overlap_end = min(a_max_x, b_max_x)
        overlap = x_overlap_end - x_overlap_start
        if overlap > 0.001 and overlap >= min_overlap:
            return True

    return False


@dataclass
class SharedEdge:
    """Describes the shared edge between two rooms."""

    wall_direction: WallDirection
    """Direction of the shared wall from room_a's perspective."""

    position_along_wall: float
    """Distance from wall start to where overlap begins (meters)."""

    width: float
    """Length of the overlapping segment (meters)."""


def get_shared_edge(room_a: PlacedRoom, room_b: PlacedRoom) -> SharedEdge | None:
    """Compute the shared edge segment between two rooms.

    Args:
        room_a: First room (perspective room for wall direction).
        room_b: Second room.

    Returns:
        SharedEdge describing the overlap, or None if rooms don't share an edge.
    """
    # Room A bounds.
    a_min_x = room_a.position[0]
    a_max_x = room_a.position[0] + room_a.width
    a_min_y = room_a.position[1]
    a_max_y = room_a.position[1] + room_a.depth

    # Room B bounds.
    b_min_x = room_b.position[0]
    b_max_x = room_b.position[0] + room_b.width
    b_min_y = room_b.position[1]
    b_max_y = room_b.position[1] + room_b.depth

    # Check for shared vertical edge.
    # A's east = B's west (room_b is to the east of room_a).
    if abs(a_max_x - b_min_x) < 0.001:
        y_overlap_start = max(a_min_y, b_min_y)
        y_overlap_end = min(a_max_y, b_max_y)
        overlap = y_overlap_end - y_overlap_start
        if overlap > 0.001:
            # Room A's east wall. Wall runs from a_min_y to a_max_y (south to north).
            position_along_wall = y_overlap_start - a_min_y
            return SharedEdge(
                wall_direction=WallDirection.EAST,
                position_along_wall=position_along_wall,
                width=overlap,
            )

    # B's east = A's west (room_b is to the west of room_a).
    if abs(b_max_x - a_min_x) < 0.001:
        y_overlap_start = max(a_min_y, b_min_y)
        y_overlap_end = min(a_max_y, b_max_y)
        overlap = y_overlap_end - y_overlap_start
        if overlap > 0.001:
            # Room A's west wall. Wall runs from a_min_y to a_max_y (south to north).
            position_along_wall = y_overlap_start - a_min_y
            return SharedEdge(
                wall_direction=WallDirection.WEST,
                position_along_wall=position_along_wall,
                width=overlap,
            )

    # Check for shared horizontal edge.
    # A's north = B's south (room_b is to the north of room_a).
    if abs(a_max_y - b_min_y) < 0.001:
        x_overlap_start = max(a_min_x, b_min_x)
        x_overlap_end = min(a_max_x, b_max_x)
        overlap = x_overlap_end - x_overlap_start
        if overlap > 0.001:
            # Room A's north wall. Wall runs from a_min_x to a_max_x (west to east).
            position_along_wall = x_overlap_start - a_min_x
            return SharedEdge(
                wall_direction=WallDirection.NORTH,
                position_along_wall=position_along_wall,
                width=overlap,
            )

    # B's north = A's south (room_b is to the south of room_a).
    if abs(b_max_y - a_min_y) < 0.001:
        x_overlap_start = max(a_min_x, b_min_x)
        x_overlap_end = min(a_max_x, b_max_x)
        overlap = x_overlap_end - x_overlap_start
        if overlap > 0.001:
            # Room A's south wall. Wall runs from a_min_x to a_max_x (west to east).
            position_along_wall = x_overlap_start - a_min_x
            return SharedEdge(
                wall_direction=WallDirection.SOUTH,
                position_along_wall=position_along_wall,
                width=overlap,
            )

    return None


def _create_placed_room(
    spec: RoomSpec,
    position: tuple[float, float],
    room_width: float | None = None,
    room_depth: float | None = None,
) -> PlacedRoom:
    """Create a PlacedRoom from spec at given position.

    Args:
        spec: Room specification.
        position: (x, y) min corner position.
        room_width: Explicit X dimension (overrides spec.length if provided).
        room_depth: Explicit Y dimension (overrides spec.width if provided).

    Returns:
        PlacedRoom with walls.
    """
    # Create walls for this room.
    # Using length as x-dimension (width in PlacedRoom) and width as y-dimension (depth).
    if room_width is None:
        room_width = spec.length  # X dimension.
    if room_depth is None:
        room_depth = spec.width  # Y dimension.

    walls = [
        Wall(
            wall_id=f"{spec.room_id}_north",
            room_id=spec.room_id,
            direction=WallDirection.NORTH,
            start_point=(position[0], position[1] + room_depth),
            end_point=(position[0] + room_width, position[1] + room_depth),
            length=room_width,
        ),
        Wall(
            wall_id=f"{spec.room_id}_south",
            room_id=spec.room_id,
            direction=WallDirection.SOUTH,
            start_point=(position[0], position[1]),
            end_point=(position[0] + room_width, position[1]),
            length=room_width,
        ),
        Wall(
            wall_id=f"{spec.room_id}_east",
            room_id=spec.room_id,
            direction=WallDirection.EAST,
            start_point=(position[0] + room_width, position[1]),
            end_point=(position[0] + room_width, position[1] + room_depth),
            length=room_depth,
        ),
        Wall(
            wall_id=f"{spec.room_id}_west",
            room_id=spec.room_id,
            direction=WallDirection.WEST,
            start_point=(position[0], position[1]),
            end_point=(position[0], position[1] + room_depth),
            length=room_depth,
        ),
    ]

    return PlacedRoom(
        room_id=spec.room_id,
        position=position,
        width=room_width,
        depth=room_depth,
        walls=walls,
    )


def _get_room_slots(room: PlacedRoom) -> list[Slot]:
    """Get available attachment slots for a room.

    Args:
        room: Placed room.

    Returns:
        List of slots (one per edge).
    """
    slots = []

    # North slot.
    slots.append(
        Slot(
            room_id=room.room_id,
            direction=WallDirection.NORTH,
            start=room.position[0],
            end=room.position[0] + room.width,
            anchor_pos=room.position,
            anchor_width=room.width,
            anchor_depth=room.depth,
        )
    )

    # South slot.
    slots.append(
        Slot(
            room_id=room.room_id,
            direction=WallDirection.SOUTH,
            start=room.position[0],
            end=room.position[0] + room.width,
            anchor_pos=room.position,
            anchor_width=room.width,
            anchor_depth=room.depth,
        )
    )

    # East slot.
    slots.append(
        Slot(
            room_id=room.room_id,
            direction=WallDirection.EAST,
            start=room.position[1],
            end=room.position[1] + room.depth,
            anchor_pos=room.position,
            anchor_width=room.width,
            anchor_depth=room.depth,
        )
    )

    # West slot.
    slots.append(
        Slot(
            room_id=room.room_id,
            direction=WallDirection.WEST,
            start=room.position[1],
            end=room.position[1] + room.depth,
            anchor_pos=room.position,
            anchor_width=room.width,
            anchor_depth=room.depth,
        )
    )

    return slots


def _update_wall_connectivity(placed_rooms: list[PlacedRoom]) -> None:
    """Update wall is_exterior and faces_rooms based on placement.

    Args:
        placed_rooms: All placed rooms (modified in place).
    """
    for room in placed_rooms:
        for wall in room.walls:
            # Check each other room to see if they share this wall.
            wall.is_exterior = True
            wall.faces_rooms = []

            for other_room in placed_rooms:
                if other_room.room_id == room.room_id:
                    continue

                # Check if this wall touches the other room.
                if _wall_touches_room(wall, other_room):
                    wall.is_exterior = False
                    if other_room.room_id not in wall.faces_rooms:
                        wall.faces_rooms.append(other_room.room_id)


def _wall_touches_room(wall: Wall, room: PlacedRoom) -> bool:
    """Check if a wall touches (is adjacent to) a room.

    Args:
        wall: Wall to check.
        room: Room to check against.

    Returns:
        True if wall touches room's boundary.
    """
    # Wall endpoints.
    w_start = wall.start_point
    w_end = wall.end_point

    # Room bounds.
    r_min_x = room.position[0]
    r_max_x = room.position[0] + room.width
    r_min_y = room.position[1]
    r_max_y = room.position[1] + room.depth

    # Check based on wall direction.
    if wall.direction == WallDirection.NORTH:
        # Wall is horizontal at y = w_start[1].
        # Check if it touches room's south edge.
        if abs(w_start[1] - r_min_y) < 0.001:
            # Check x overlap.
            x_overlap = max(w_start[0], r_min_x) < min(w_end[0], r_max_x)
            return x_overlap
    elif wall.direction == WallDirection.SOUTH:
        # Check if it touches room's north edge.
        if abs(w_start[1] - r_max_y) < 0.001:
            x_overlap = max(w_start[0], r_min_x) < min(w_end[0], r_max_x)
            return x_overlap
    elif wall.direction == WallDirection.EAST:
        # Wall is vertical at x = w_start[0].
        # Check if it touches room's west edge.
        if abs(w_start[0] - r_min_x) < 0.001:
            y_overlap = max(w_start[1], r_min_y) < min(w_end[1], r_max_y)
            return y_overlap
    elif wall.direction == WallDirection.WEST:
        # Check if it touches room's east edge.
        if abs(w_start[0] - r_max_x) < 0.001:
            y_overlap = max(w_start[1], r_min_y) < min(w_end[1], r_max_y)
            return y_overlap

    return False


def find_room(rooms: list[PlacedRoom], room_id: str) -> PlacedRoom:
    """Find a room by ID.

    Args:
        rooms: List of placed rooms.
        room_id: Room ID to find.

    Returns:
        The room with matching ID.

    Raises:
        ValueError: If room not found.
    """
    for room in rooms:
        if room.room_id == room_id:
            return room
    raise ValueError(f"Room '{room_id}' not found")


def get_shared_boundary(room_a: PlacedRoom, room_b: PlacedRoom) -> Wall | None:
    """Get the wall segment shared between two rooms.

    Args:
        room_a: First room.
        room_b: Second room.

    Returns:
        Wall from room_a that faces room_b, or None if not adjacent.
    """
    for wall in room_a.walls:
        if room_b.room_id in wall.faces_rooms:
            return wall
    return None


def validate_connectivity(
    placed_rooms: list[PlacedRoom],
    doors: list,  # list[Door] but avoiding circular import
    room_specs: list | None = None,  # list[RoomSpec] for open connections
) -> tuple[bool, str]:
    """Validate that all rooms are reachable from exterior via doors or open connections.

    Uses BFS from rooms with exterior doors.

    Args:
        placed_rooms: All placed rooms.
        doors: All doors in the house.
        room_specs: Room specifications containing open connections.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not placed_rooms:
        return True, ""

    if not doors:
        return False, "No doors defined. At least one exterior door required."

    # Find rooms with exterior doors.
    rooms_with_exterior_door = set()
    for door in doors:
        if door.door_type == "exterior":
            rooms_with_exterior_door.add(door.room_a)

    if not rooms_with_exterior_door:
        return False, "No exterior door found. At least one exterior door required."

    # Build adjacency graph from interior doors AND open connections.
    connections: dict[str, set[str]] = {r.room_id: set() for r in placed_rooms}

    # Add interior door connections.
    for door in doors:
        if door.door_type == "interior" and door.room_b:
            connections[door.room_a].add(door.room_b)
            connections[door.room_b].add(door.room_a)

    # Add open connections (rooms with no wall between them).
    if room_specs:
        for spec in room_specs:
            for other_room, conn_type in spec.connections.items():
                if conn_type.value == "OPEN":
                    if spec.room_id in connections and other_room in connections:
                        connections[spec.room_id].add(other_room)
                        connections[other_room].add(spec.room_id)

    # BFS from exterior doors.
    reachable = set()
    queue = list(rooms_with_exterior_door)
    reachable.update(queue)

    while queue:
        current = queue.pop(0)
        for neighbor in connections.get(current, []):
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)

    # Check all rooms are reachable.
    all_room_ids = {r.room_id for r in placed_rooms}
    unreachable = all_room_ids - reachable

    if unreachable:
        return (
            False,
            f"Rooms not reachable from exterior: {', '.join(sorted(unreachable))}",
        )

    return True, ""
