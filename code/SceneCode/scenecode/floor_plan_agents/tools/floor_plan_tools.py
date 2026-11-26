"""Floor plan designer tools for layout manipulation.

These tools allow the floor plan designer agent to create and modify house layouts,
including rooms, doors, windows, and materials.
"""

import json
import logging

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from agents import function_tool

from scenecode.agent_utils.house import (
    ConnectionType,
    HouseLayout,
    RoomMaterials,
    RoomSpec,
    Wall,
    WallDirection,
)
from scenecode.floor_plan_agents.tools.ascii_generator import generate_ascii_floor_plan
from scenecode.floor_plan_agents.tools.door_window_mixin import DoorWindowMixin
from scenecode.floor_plan_agents.tools.materials_resolver import (
    MaterialsConfig,
    MaterialsResolver,
)
from scenecode.floor_plan_agents.tools.open_plan_mixin import OpenPlanMixin
from scenecode.floor_plan_agents.tools.room_placement import (
    PlacementConfig,
    PlacementError,
    ScoringWeights,
    get_shared_edge,
    place_rooms,
    validate_connectivity,
)

if TYPE_CHECKING:
    from scenecode.agent_utils.image_generation import BaseImageGenerator

console_logger = logging.getLogger(__name__)


@dataclass
class RoomSpecsResult:
    """Result from generate_room_specs tool."""

    success: bool
    message: str
    ascii_floor_plan: str = ""
    wall_segment_labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Result:
    """Generic result from floor plan tools."""

    success: bool
    message: str


@dataclass
class MaterialResult:
    """Result from get_material tool."""

    success: bool
    message: str
    material_id: str = ""


@dataclass
class DoorWindowConfig:
    """Configuration for door and window constraints.

    All dimension values are in meters.
    """

    # Door constraints.
    door_width_min: float = 0.9
    door_width_max: float = 1.9
    door_height_min: float = 2.0
    door_height_max: float = 2.4
    door_default_width: float = 0.9
    door_default_height: float = 2.1

    # Window constraints.
    window_width_min: float = 0.6
    window_width_max: float = 3.0
    window_height_min: float = 0.6
    window_height_max: float = 2.0
    window_default_width: float = 1.2
    window_default_height: float = 1.2
    window_default_sill_height: float = 0.9
    window_segment_margin: float = 0.3  # Margin from segment boundary (meters).

    # Exterior door constraints.
    exterior_door_clearance_m: float = 1.0  # Min clearance outside exterior doors.


@dataclass
class MaterialsListResult:
    """Result from list_room_materials tool."""

    success: bool
    message: str
    materials: dict[str, dict[str, str]] = field(default_factory=dict)
    exterior_material_id: str = ""


@dataclass
class ValidationResult:
    """Result from validate tool."""

    layout: str  # "ok" or error message.
    connectivity: str  # "ok" or error message.


class FloorPlanTools(DoorWindowMixin, OpenPlanMixin):
    """Tools for floor plan designer agent.

    Follow the workflow phases:
    1. Room Layout - generate_room_specs, resize_room, add/remove_adjacency
    2. Wall Height - set_wall_height
    3. Doors - add_door, remove_door
    4. Windows - add_window, remove_window
    5. Materials - get_material, set_room_materials, set_exterior_material
    6. Validation - validate
    """

    def __init__(
        self,
        layout: HouseLayout,
        mode: Literal["room", "house"] = "room",
        materials_config: MaterialsConfig | None = None,
        material_image_generator: "BaseImageGenerator | None" = None,
        min_opening_separation: float = 0.5,
        placement_timeout_seconds: float = 5.0,
        placement_scoring_weights: ScoringWeights | None = None,
        placement_exterior_wall_clearance_m: float = 20.0,
        door_window_config: DoorWindowConfig | None = None,
        wall_height_min: float = 2.0,
        wall_height_max: float = 4.5,
        room_dim_min: float = 1.5,
        room_dim_max: float = 20.0,
    ):
        """Initialize floor plan tools.

        Args:
            layout: The HouseLayout to modify.
            mode: "room" (single room) or "house" (multi-room).
            materials_config: Materials resolver configuration.
            material_image_generator: Optional image generator for code materials.
            min_opening_separation: Minimum gap between door and window on same wall.
            placement_timeout_seconds: Backtracking search timeout for room placement.
            placement_scoring_weights: Weights for layout scoring (compactness, stability).
            placement_exterior_wall_clearance_m: Clearance zone for exterior_walls constraint.
            door_window_config: Door and window constraints configuration.
            wall_height_min: Minimum wall height in meters.
            wall_height_max: Maximum wall height in meters.
            room_dim_min: Minimum room dimension (width or depth) in meters.
            room_dim_max: Maximum room dimension (width or depth) in meters.
        """
        self.layout = layout
        self.mode = mode
        self.materials_resolver = MaterialsResolver(
            materials_config,
            image_generator=material_image_generator,
        )
        self.min_opening_separation = min_opening_separation
        self.placement_config = PlacementConfig(
            timeout_seconds=placement_timeout_seconds,
            scoring_weights=placement_scoring_weights or ScoringWeights(),
            exterior_wall_clearance_m=placement_exterior_wall_clearance_m,
        )
        self.door_window_config = door_window_config or DoorWindowConfig()
        self.wall_height_min = wall_height_min
        self.wall_height_max = wall_height_max
        self.room_dim_min = room_dim_min
        self.room_dim_max = room_dim_max

        # Build tools dictionary using closure pattern.
        # This avoids including 'self' in OpenAI function schemas.
        self.tools = self._create_tool_closures()

    def _check_rooms_exist(self) -> Result | None:
        """Check if rooms have been defined.

        Returns:
            Error Result if no rooms, None if OK.
        """
        if not self.layout.room_specs:
            return Result(
                success=False,
                message="No rooms defined. Call generate_room_specs first.",
            )
        return None

    def _fail(self, message: str) -> Result:
        """Log failure and return Result with success=False.

        Args:
            message: Failure message to log and return.

        Returns:
            Result with success=False and the message.
        """
        console_logger.info(f"Tool failed: {message}")
        return Result(success=False, message=message)

    def _create_tool_closures(self) -> dict:
        """Create tool closures with access to instance data.

        Uses closure pattern to avoid including 'self' in OpenAI function schemas.
        Each tool is a local function that captures self via closure.

        Returns:
            Dictionary mapping tool names to tool functions.
        """

        @function_tool
        def generate_room_specs(room_specs_json: str) -> RoomSpecsResult:
            """Create rooms with the specified dimensions and adjacencies.

            MUST be called first. Room mode: fails if >1 room specified.

            Args:
                room_specs_json: JSON string with list of room specifications. Example:
                    '[{"type": "living_room", "width": 5.0, "depth": 4.0},
                      {"type": "kitchen", "width": 3.0, "depth": 4.0,
                       "connections": {"living_room": "DOOR"}},
                      {"type": "hallway", "width": 2.0, "depth": 6.0,
                       "connections": {"living_room": "DOOR"},
                       "exterior_walls": ["west"]}]'

                    exterior_walls: Optional list of wall directions ("north", "south",
                    "east", "west") that MUST remain exterior (no rooms placed adjacent).
                    Use for rooms needing guaranteed external door access, e.g., a hallway
                    with multiple room connections that still needs an entrance door.

            Returns:
                RoomSpecsResult with placed rooms and wall segment labels.
            """
            return self._generate_room_specs_impl(room_specs_json)

        @function_tool
        def resize_room(room_id: str, width: float, depth: float) -> Result:
            """Change a room's dimensions.

            Args:
                room_id: Room to resize (e.g., "living_room").
                width: New width in meters.
                depth: New depth in meters.

            Returns:
                Result indicating success or failure.
            """
            return self._resize_room_impl(room_type=room_id, width=width, depth=depth)

        @function_tool
        def add_adjacency(room_a: str, room_b: str) -> Result:
            """Require two rooms to share a wall.

            Args:
                room_a: First room ID.
                room_b: Second room ID.

            Returns:
                Result indicating success or failure.
            """
            return self._add_adjacency_impl(room_a=room_a, room_b=room_b)

        @function_tool
        def remove_adjacency(room_a: str, room_b: str) -> Result:
            """Remove requirement for two rooms to share a wall.

            Args:
                room_a: First room ID.
                room_b: Second room ID.

            Returns:
                Result indicating success or failure.
            """
            return self._remove_adjacency_impl(room_a=room_a, room_b=room_b)

        @function_tool
        def add_open_connection(room_a: str, room_b: str) -> Result:
            """Remove wall between rooms for open floor plan (e.g., "living room open
            to kitchen").

            Creates floor-to-ceiling opening with NO wall - do NOT add doors after this.
            Use for: "open to", "open plan", "flows into", "combined" in prompt.

            Args:
                room_a: First room ID.
                room_b: Second room ID.

            Returns:
                Result indicating success or failure.
            """
            return self._add_open_connection_impl(room_a=room_a, room_b=room_b)

        @function_tool
        def remove_open_connection(room_a: str, room_b: str) -> Result:
            """Remove an open floor plan connection and restore the wall.

            Args:
                room_a: First room ID.
                room_b: Second room ID.

            Returns:
                Result indicating success or failure.
            """
            return self._remove_open_connection_impl(room_a=room_a, room_b=room_b)

        @function_tool
        def set_wall_height(height_meters: float) -> Result:
            """Set the wall height for all rooms.

            Args:
                height_meters: Wall height in meters.

            Returns:
                Result indicating success or failure.
            """
            return self._set_wall_height_impl(height_meters=height_meters)

        @function_tool
        def add_door(
            wall_id: str, position: str, width: float = 0.9, height: float = 2.1
        ) -> Result:
            """Add a door to a wall segment.

            Args:
                wall_id: Wall segment label (e.g., "A", "B") from the ASCII plan.
                position: "left", "center", or "right" third of the wall.
                width: Door width in meters.
                height: Door height in meters.

            Returns:
                Result indicating success or failure.
            """
            return self._add_door_impl(
                wall_id=wall_id, position=position, width=width, height=height
            )

        @function_tool
        def remove_door(door_id: str) -> Result:
            """Remove a door.

            Args:
                door_id: Door identifier to remove.

            Returns:
                Result indicating success or failure.
            """
            return self._remove_door_impl(door_id)

        @function_tool
        def add_window(
            wall_id: str,
            position: str,
            width: float = 1.2,
            height: float = 1.2,
            sill_height: float = 0.9,
        ) -> Result:
            """Add a window to an exterior wall segment.

            Args:
                wall_id: Wall segment label (e.g., "A", "B") from the ASCII plan.
                position: "left", "center", or "right" third of the wall.
                width: Window width in meters.
                height: Window height in meters.
                sill_height: Height from floor to window bottom in meters.

            Returns:
                Result indicating success or failure.
            """
            return self._add_window_impl(
                wall_id=wall_id,
                position=position,
                width=width,
                height=height,
                sill_height=sill_height,
            )

        @function_tool
        def remove_window(window_id: str) -> Result:
            """Remove a window.

            Args:
                window_id: Window identifier to remove.

            Returns:
                Result indicating success or failure.
            """
            return self._remove_window_impl(window_id)

        @function_tool
        def get_material(description: str) -> MaterialResult:
            """Search for a material by description.

            Args:
                description: Material description (e.g., "light oak wood floor").

            Returns:
                MaterialResult with material_id if found.
            """
            return self._get_material_impl(description)

        @function_tool
        def set_room_materials(
            room_id: str,
            floor_material_id: str = "",
            wall_material_id: str = "",
        ) -> Result:
            """Set materials for a room's floor and/or walls.

            Args:
                room_id: Room to set materials for.
                floor_material_id: Floor material ID from get_material (empty to skip).
                wall_material_id: Wall material ID from get_material (empty to skip).

            Returns:
                Result indicating success or failure.
            """
            return self._set_room_materials_impl(
                room_id=room_id,
                floor_material_id=floor_material_id,
                wall_material_id=wall_material_id,
            )

        @function_tool
        def set_exterior_material(material_id: str) -> Result:
            """Set exterior wall material.

            Args:
                material_id: Material ID from get_material.

            Returns:
                Result indicating success or failure.
            """
            return self._set_exterior_material_impl(material_id)

        @function_tool
        def list_room_materials() -> MaterialsListResult:
            """List current material assignments for all rooms.

            Returns:
                MaterialsListResult with material assignments.
            """
            return self._list_room_materials_impl()

        @function_tool
        def validate() -> ValidationResult:
            """Validate the floor plan for completeness.

            Returns:
                ValidationResult with any issues found.
            """
            return self._validate_impl()

        @function_tool
        def render_ascii() -> str:
            """Generate ASCII representation of the floor plan.

            Returns:
                ASCII floor plan with wall labels and legend.
            """
            return self._render_ascii_impl()

        return {
            "generate_room_specs": generate_room_specs,
            "resize_room": resize_room,
            "add_adjacency": add_adjacency,
            "remove_adjacency": remove_adjacency,
            "add_open_connection": add_open_connection,
            "remove_open_connection": remove_open_connection,
            "set_wall_height": set_wall_height,
            "add_door": add_door,
            "remove_door": remove_door,
            "add_window": add_window,
            "remove_window": remove_window,
            "get_material": get_material,
            "set_room_materials": set_room_materials,
            "set_exterior_material": set_exterior_material,
            "list_room_materials": list_room_materials,
            "validate": validate,
            "render_ascii": render_ascii,
        }

    def _generate_room_specs_impl(self, room_specs_json: str) -> RoomSpecsResult:
        """Create rooms with the specified dimensions and adjacencies.

        MUST be called first. Room mode: fails if >1 room specified.

        Args:
            room_specs_json: JSON string with list of room specifications.
                Each room must have: type and prompt (required).
                Optional fields: width, depth, connections.
                connections: dict mapping room_id to "DOOR" or "OPEN".
                Example:
                '[{"type": "living_room", "width": 5.0, "depth": 4.0,
                   "prompt": "A cozy modern living room with large windows."},
                  {"type": "kitchen", "width": 3.0, "depth": 4.0,
                   "prompt": "A bright kitchen with white cabinets.",
                   "connections": {"living_room": "DOOR"}}]'

        Returns:
            RoomSpecsResult with placed rooms and wall segment labels.
        """
        # Format JSON for readable logging.
        try:
            parsed_for_log = json.loads(room_specs_json)
            formatted_json = json.dumps(parsed_for_log, indent=2)
        except json.JSONDecodeError:
            formatted_json = room_specs_json  # Use raw string if invalid JSON.
        console_logger.info(
            f"Tool called: generate_room_specs(room_specs_json=\n{formatted_json})"
        )
        # Parse JSON input.
        try:
            room_specs = json.loads(room_specs_json)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON: {e}"
            console_logger.info(f"Tool failed: {msg}")
            return RoomSpecsResult(success=False, message=msg)

        if not isinstance(room_specs, list):
            msg = "room_specs_json must be a JSON array of room specifications."
            console_logger.info(f"Tool failed: {msg}")
            return RoomSpecsResult(success=False, message=msg)

        # Mode check.
        if self.mode == "room" and len(room_specs) > 1:
            msg = "Room mode: only 1 room allowed. Use house mode for multiple rooms."
            console_logger.info(f"Tool failed: {msg}")
            return RoomSpecsResult(success=False, message=msg)

        # Convert to RoomSpec objects.
        specs = []
        room_type_counts: dict[str, int] = {}
        for spec_dict in room_specs:
            room_type = spec_dict.get("type", "room")

            # Generate room_id from type.
            room_type_counts[room_type] = room_type_counts.get(room_type, 0) + 1
            if room_type_counts[room_type] == 1:
                room_id = room_type
            else:
                room_id = f"{room_type}_{room_type_counts[room_type]}"

            # In room mode, always use house prompt directly (ignore agent's prompt).
            # In house mode, prompt is required for each room.
            if self.mode == "room":
                prompt = self.layout.house_prompt
            else:
                prompt = spec_dict.get("prompt", "")
                if not prompt:
                    msg = (
                        f"Room '{room_type}' is missing required 'prompt' field. "
                        f"Each room must have a descriptive prompt."
                    )
                    console_logger.info(f"Tool failed: {msg}")
                    return RoomSpecsResult(success=False, message=msg)

            # Validate room dimensions.
            room_width = spec_dict.get("width", 5.0)
            room_depth = spec_dict.get("depth", 4.0)
            if not (self.room_dim_min <= room_width <= self.room_dim_max):
                msg = (
                    f"Room '{room_type}' width must be {self.room_dim_min}-"
                    f"{self.room_dim_max}m. Got: {room_width}"
                )
                console_logger.info(f"Tool failed: {msg}")
                return RoomSpecsResult(success=False, message=msg)
            if not (self.room_dim_min <= room_depth <= self.room_dim_max):
                msg = (
                    f"Room '{room_type}' depth must be {self.room_dim_min}-"
                    f"{self.room_dim_max}m. Got: {room_depth}"
                )
                console_logger.info(f"Tool failed: {msg}")
                return RoomSpecsResult(success=False, message=msg)

            # Parse connections from JSON.
            connections_raw = spec_dict.get("connections", {})
            connections = {k: ConnectionType(v) for k, v in connections_raw.items()}

            # Parse exterior_walls constraint from JSON.
            exterior_walls_raw = spec_dict.get("exterior_walls", [])
            exterior_walls = {WallDirection(w) for w in exterior_walls_raw}

            specs.append(
                RoomSpec(
                    room_id=room_id,
                    room_type=room_type,
                    prompt=prompt,
                    width=room_depth,  # Y dimension.
                    length=room_width,  # X dimension.
                    connections=connections,
                    exterior_walls=exterior_walls,
                )
            )

        # Run placement algorithm.
        try:
            placed_rooms = place_rooms(room_specs=specs, config=self.placement_config)
        except PlacementError as e:
            msg = f"Room placement failed: {e}"
            console_logger.info(f"Tool failed: {msg}")
            return RoomSpecsResult(success=False, message=msg)

        # Update layout.
        self.layout.room_specs = specs
        self.layout.placed_rooms = placed_rooms
        self.layout.placement_valid = True

        # Clear openings from previous layout (wall labels may have changed).
        self.layout.doors.clear()
        self.layout.windows.clear()

        # Invalidate all cached geometry (room specs completely changed).
        invalidated = self.layout.invalidate_all_room_geometries()
        console_logger.info(f"Invalidated {invalidated} room geometries")

        # Generate ASCII floor plan.
        ascii_result = generate_ascii_floor_plan(placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        # Log ASCII for visibility during runs.
        console_logger.info(
            "Floor plan layout:\n%s\n%s", ascii_result.ascii_art, ascii_result.legend
        )

        # Build wall segment labels description.
        labels_desc = {}
        for label, (room_a, room_b, direction) in ascii_result.boundary_labels.items():
            if room_b:
                labels_desc[label] = f"Interior: {room_a} <-> {room_b}"
            else:
                dir_str = f" ({direction})" if direction else ""
                labels_desc[label] = f"Exterior: {room_a}{dir_str}"

        return RoomSpecsResult(
            success=True,
            message=f"Created {len(specs)} room(s) successfully.",
            ascii_floor_plan=ascii_result.ascii_art,
            wall_segment_labels=labels_desc,
        )

    def _resize_room_impl(self, room_id: str, width: float, depth: float) -> Result:
        """Change a room's dimensions with layout stability.

        Doors and windows are preserved when possible:
        - Openings on walls whose length changed are proportionally repositioned
        - Openings that no longer fit after repositioning are removed
        - Open connections are preserved (positions recomputed from shared edges)

        Other rooms stay in approximately the same positions unless adjacency
        constraints require movement.

        Args:
            room_id: Room to resize (e.g., "living_room").
            width: New width in meters (within configured range).
            depth: New depth in meters (within configured range).

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: resize_room(room_id={room_id}, width={width}, depth={depth})"
        )
        error = self._check_rooms_exist()
        if error:
            return error

        # Validate dimensions.
        if not (self.room_dim_min <= width <= self.room_dim_max):
            return self._fail(
                f"Width must be {self.room_dim_min}-{self.room_dim_max}m. Got: {width}"
            )
        if not (self.room_dim_min <= depth <= self.room_dim_max):
            return self._fail(
                f"Depth must be {self.room_dim_min}-{self.room_dim_max}m. Got: {depth}"
            )

        # Find room spec.
        spec = self.layout.get_room_spec(room_id)
        if not spec:
            return self._fail(f"Room '{room_id}' not found.")

        # Store old state for rollback on failure.
        old_width = spec.length  # X dimension.
        old_depth = spec.width  # Y dimension.
        old_placed_rooms = self.layout.placed_rooms

        # Temporarily update dimensions for placement attempt.
        spec.length = width  # X dimension.
        spec.width = depth  # Y dimension.

        # Re-run placement with layout stability (other rooms stay in place).
        try:
            config = PlacementConfig(
                timeout_seconds=self.placement_config.timeout_seconds,
                scoring_weights=self.placement_config.scoring_weights,
                previous_positions={r.room_id: r.position for r in old_placed_rooms},
                free_room_ids={room_id},
            )
            placed_rooms = place_rooms(
                room_specs=self.layout.room_specs,
                config=config,
            )
            self.layout.placed_rooms = placed_rooms
            self.layout.placement_valid = True
        except PlacementError as e:
            # Rollback: restore old dimensions, keep layout valid.
            spec.length = old_width
            spec.width = old_depth
            # Note: placed_rooms unchanged since assignment only happens on success.
            return self._fail(f"Resize failed (layout unchanged): {e}")

        # Invalidate geometry for resized room (dimensions changed).
        if self.layout.invalidate_room_geometry(room_id):
            console_logger.debug(f"Invalidated geometry for resized room: {room_id}")

        # Regenerate ASCII labels after placement.
        ascii_result = generate_ascii_floor_plan(placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        # Proportionally adjust opening positions for walls whose length changed.
        self._adjust_opening_positions_for_resize(
            room_id=room_id,
            old_width=old_width,
            old_depth=old_depth,
            new_width=width,
            new_depth=depth,
        )

        # Reapply all doors/windows and open connections. This validates positions
        # and removes openings that no longer fit after resize.
        removed_doors, removed_windows = self._reapply_openings_to_walls()

        # Build result message with removal info.
        msg = f"Room '{room_id}' resized to {width}m x {depth}m."
        msg += self._format_removal_message(
            removed_doors=removed_doors, removed_windows=removed_windows
        )

        return Result(success=True, message=msg)

    def _add_adjacency_impl(self, room_a: str, room_b: str) -> Result:
        """Require two rooms to share a wall.

        Room mode: fails (single room has no adjacencies).

        Args:
            room_a: First room ID.
            room_b: Second room ID.

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: add_adjacency(room_a={room_a}, room_b={room_b})"
        )
        if self.mode == "room":
            return self._fail("Room mode: no adjacencies for single room.")

        error = self._check_rooms_exist()
        if error:
            return error

        spec_a = self.layout.get_room_spec(room_a)
        spec_b = self.layout.get_room_spec(room_b)

        if not spec_a:
            return self._fail(f"Room '{room_a}' not found.")
        if not spec_b:
            return self._fail(f"Room '{room_b}' not found.")

        # Track whether we need to add connections (for rollback on failure).
        added_b_to_a = room_b not in spec_a.connections
        added_a_to_b = room_a not in spec_b.connections

        # Add connection with DOOR type.
        if added_b_to_a:
            spec_a.connections[room_b] = ConnectionType.DOOR
        if added_a_to_b:
            spec_b.connections[room_a] = ConnectionType.DOOR

        # If rooms are already placed and already adjacent, no re-placement needed.
        if self.layout.placed_rooms:
            placed_a = next(
                (r for r in self.layout.placed_rooms if r.room_id == room_a), None
            )
            placed_b = next(
                (r for r in self.layout.placed_rooms if r.room_id == room_b), None
            )
            if placed_a and placed_b:
                shared_edge = get_shared_edge(placed_a, placed_b)
                if shared_edge:
                    # Already adjacent - constraint already satisfied.
                    msg = f"Added adjacency: {room_a} <-> {room_b}."
                    return Result(success=True, message=msg)

        # Rooms not yet placed or not currently adjacent - run placement with stability.
        # Both rooms being made adjacent should have freedom to move.
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
            # Rollback: remove connections we added.
            if added_b_to_a:
                del spec_a.connections[room_b]
            if added_a_to_b:
                del spec_b.connections[room_a]
            return self._fail(f"Add adjacency failed (layout unchanged): {e}")

        # Invalidate all geometry (adjacency affects positions and wall labels).
        invalidated = self.layout.invalidate_all_room_geometries()
        if invalidated > 0:
            console_logger.debug(f"Invalidated {invalidated} room geometries")

        # Regenerate ASCII labels after placement.
        ascii_result = generate_ascii_floor_plan(placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        # Restore openings (doors, windows, open connections) to new walls.
        removed_doors, removed_windows = self._reapply_openings_to_walls()

        msg = f"Added adjacency: {room_a} <-> {room_b}."
        msg += self._format_removal_message(
            removed_doors=removed_doors, removed_windows=removed_windows
        )

        return Result(success=True, message=msg)

    def _remove_adjacency_impl(self, room_a: str, room_b: str) -> Result:
        """Remove requirement for two rooms to share a wall.

        Room mode: fails.

        Args:
            room_a: First room ID.
            room_b: Second room ID.

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: remove_adjacency(room_a={room_a}, room_b={room_b})"
        )
        if self.mode == "room":
            return self._fail("Room mode: no adjacencies for single room.")

        error = self._check_rooms_exist()
        if error:
            return error

        spec_a = self.layout.get_room_spec(room_a)
        spec_b = self.layout.get_room_spec(room_b)

        if not spec_a:
            return self._fail(f"Room '{room_a}' not found.")
        if not spec_b:
            return self._fail(f"Room '{room_b}' not found.")

        # Track connections we remove (for rollback on failure).
        removed_b_from_a = spec_a.connections.pop(room_b, None)
        removed_a_from_b = spec_b.connections.pop(room_a, None)

        # If rooms are already placed, no re-placement needed.
        # Removing a constraint doesn't invalidate existing placement.
        if self.layout.placed_rooms:
            msg = f"Removed adjacency: {room_a} <-> {room_b}."
            return Result(success=True, message=msg)

        # Rooms not yet placed - run placement with stability.
        # No special rooms need freedom since we're just removing a constraint.
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
            # Rollback: restore connections we removed.
            if removed_b_from_a is not None:
                spec_a.connections[room_b] = removed_b_from_a
            if removed_a_from_b is not None:
                spec_b.connections[room_a] = removed_a_from_b
            return self._fail(f"Remove adjacency failed (layout unchanged): {e}")

        # Invalidate all geometry (adjacency affects positions and wall labels).
        invalidated = self.layout.invalidate_all_room_geometries()
        if invalidated > 0:
            console_logger.debug(f"Invalidated {invalidated} room geometries")

        # Regenerate ASCII labels after placement.
        ascii_result = generate_ascii_floor_plan(placed_rooms)
        self.layout.boundary_labels = ascii_result.boundary_labels

        # Restore openings (doors, windows, open connections) to new walls.
        removed_doors, removed_windows = self._reapply_openings_to_walls()

        msg = f"Removed adjacency: {room_a} <-> {room_b}."
        msg += self._format_removal_message(
            removed_doors=removed_doors, removed_windows=removed_windows
        )

        return Result(success=True, message=msg)

    def _set_wall_height_impl(self, height_meters: float) -> Result:
        """Set wall/ceiling height for entire house.

        Args:
            height_meters: Height in valid range (from config).

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: set_wall_height(height_meters={height_meters})"
        )
        if not (self.wall_height_min <= height_meters <= self.wall_height_max):
            return self._fail(
                f"Height must be between {self.wall_height_min} and "
                f"{self.wall_height_max}m. Got: {height_meters}"
            )

        self.layout.wall_height = height_meters

        # Invalidate all geometry (wall height affects all rooms).
        invalidated = self.layout.invalidate_all_room_geometries()
        if invalidated > 0:
            console_logger.debug(f"Invalidated {invalidated} room geometries")

        return Result(success=True, message=f"Wall height set to {height_meters}m.")

    def _get_wall_by_boundary(self, wall_label: str, room_id: str) -> Wall:
        """Get a Wall object by boundary label and room ID.

        Args:
            wall_label: Boundary label (e.g., "A", "B") from boundary_labels.
            room_id: Room ID owning the wall.

        Returns:
            The Wall object.

        Raises:
            ValueError: If wall cannot be found (fail-fast per CLAUDE.md).
        """
        # Look up what this boundary label refers to.
        boundary_info = self.layout.boundary_labels.get(wall_label)
        if not boundary_info:
            raise ValueError(
                f"Unknown wall label '{wall_label}'. "
                f"Available labels: {list(self.layout.boundary_labels.keys())}"
            )

        room_a_label, room_b_label, direction = boundary_info

        # Determine which room we're looking for on the wall.
        # If room_id is room_a, look for wall facing room_b.
        # If room_id is room_b, look for wall facing room_a.
        if room_id == room_a_label:
            target_room = room_b_label
        elif room_id == room_b_label:
            target_room = room_a_label
        else:
            # room_id doesn't match either end of this boundary.
            raise ValueError(
                f"Room '{room_id}' is not part of boundary '{wall_label}' "
                f"which connects {room_a_label} <-> {room_b_label}."
            )

        # Find the placed room.
        placed_room = None
        for placed in self.layout.placed_rooms:
            if placed.room_id == room_id:
                placed_room = placed
                break

        if not placed_room:
            raise ValueError(f"Room '{room_id}' not found in placed_rooms.")

        # Find the wall matching this boundary.
        for wall in placed_room.walls:
            if target_room is None:
                # Exterior wall - match by direction.
                if wall.is_exterior and wall.direction.value == direction:
                    return wall
            else:
                # Interior wall - check if this wall faces the target room.
                if target_room in wall.faces_rooms:
                    return wall

        raise ValueError(
            f"Wall for boundary '{wall_label}' not found in room '{room_id}'."
        )

    def _get_wall_length(self, room_id: str, wall_label: str) -> float:
        """Get the length of a wall by room and boundary label.

        Args:
            room_id: Room ID owning the wall.
            wall_label: Boundary label (e.g., "A", "B") from boundary_labels.

        Returns:
            Wall length in meters.

        Raises:
            ValueError: If wall cannot be found (fail-fast per CLAUDE.md).
        """
        wall = self._get_wall_by_boundary(wall_label=wall_label, room_id=room_id)
        return wall.length

    def _wall_faces_nearby_room(
        self, room_id: str, wall_label: str, threshold: float = 0.5
    ) -> str | None:
        """Check if an exterior wall faces another room within threshold distance.

        Used to prevent windows on walls that technically exterior but face another
        room across a small gap (e.g., 10cm gap between non-adjacent rooms).

        Args:
            room_id: Room owning the wall.
            wall_label: Boundary label from boundary_labels.
            threshold: Maximum distance (meters) to consider as "facing".

        Returns:
            Room ID of nearby room if found, None otherwise.
        """
        wall = self._get_wall_by_boundary(wall_label=wall_label, room_id=room_id)
        placed_room = next(
            (r for r in self.layout.placed_rooms if r.room_id == room_id), None
        )
        if not placed_room:
            return None

        # Get wall position based on direction.
        for other in self.layout.placed_rooms:
            if other.room_id == room_id:
                continue

            other_min_x = other.position[0]
            other_max_x = other.position[0] + other.width
            other_min_y = other.position[1]
            other_max_y = other.position[1] + other.depth

            # Check if wall faces this room within threshold.
            if wall.direction == WallDirection.NORTH:
                # Wall at y = placed_room.position[1] + depth, facing +Y.
                wall_y = placed_room.position[1] + placed_room.depth
                # Check if other room's south edge is within threshold.
                if 0 < other_min_y - wall_y <= threshold:
                    # Check x overlap.
                    if max(wall.start_point[0], other_min_x) < min(
                        wall.end_point[0], other_max_x
                    ):
                        return other.room_id
            elif wall.direction == WallDirection.SOUTH:
                # Wall at y = placed_room.position[1], facing -Y.
                wall_y = placed_room.position[1]
                # Check if other room's north edge is within threshold.
                if 0 < wall_y - other_max_y <= threshold:
                    if max(wall.start_point[0], other_min_x) < min(
                        wall.end_point[0], other_max_x
                    ):
                        return other.room_id
            elif wall.direction == WallDirection.EAST:
                # Wall at x = placed_room.position[0] + width, facing +X.
                wall_x = placed_room.position[0] + placed_room.width
                # Check if other room's west edge is within threshold.
                if 0 < other_min_x - wall_x <= threshold:
                    if max(wall.start_point[1], other_min_y) < min(
                        wall.end_point[1], other_max_y
                    ):
                        return other.room_id
            elif wall.direction == WallDirection.WEST:
                # Wall at x = placed_room.position[0], facing -X.
                wall_x = placed_room.position[0]
                # Check if other room's east edge is within threshold.
                if 0 < wall_x - other_max_x <= threshold:
                    if max(wall.start_point[1], other_min_y) < min(
                        wall.end_point[1], other_max_y
                    ):
                        return other.room_id

        return None

    def _get_material_impl(self, description: str) -> MaterialResult:
        """Find a material matching the description.

        Args:
            description: What the material should look like (e.g., "warm oak
                hardwood", "white hexagon tile", "red brick").

        Returns:
            MaterialResult with material_id.
        """
        console_logger.info(f"Tool called: get_material(description={description})")

        material = self.materials_resolver.get_material(description)

        if material:
            return MaterialResult(
                success=True,
                message=f"Found material: {material.material_id}",
                material_id=material.material_id,
            )
        else:
            msg = f"No material found matching '{description}'."
            console_logger.info(f"Tool failed: {msg}")
            return MaterialResult(success=False, message=msg)

    def _set_room_materials_impl(
        self, room_id: str, floor_material_id: str, wall_material_id: str
    ) -> Result:
        """Set wall and floor materials for a room using material IDs.

        Get material IDs from get_material() or list_room_materials().

        Args:
            room_id: Room to set materials for.
            floor_material_id: Material ID for floor (e.g., "Wood094_1K-JPG").
            wall_material_id: Material ID for walls (e.g., "Plaster001_1K-JPG").

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: set_room_materials(room_id={room_id}, "
            f"wall_material_id={wall_material_id}, floor_material_id={floor_material_id})"
        )
        error = self._check_rooms_exist()
        if error:
            return error

        spec = self.layout.get_room_spec(room_id)
        if not spec:
            return self._fail(f"Room '{room_id}' not found.")

        # Resolve materials.
        wall_mat = self.materials_resolver.get_material_by_id(wall_material_id)
        floor_mat = self.materials_resolver.get_material_by_id(floor_material_id)

        if not wall_mat:
            return self._fail(f"Wall material '{wall_material_id}' not found.")
        if not floor_mat:
            return self._fail(f"Floor material '{floor_material_id}' not found.")

        self.layout.room_materials[room_id] = RoomMaterials(
            wall_material=wall_mat,
            floor_material=floor_mat,
        )

        # Invalidate geometry for this room (materials baked into GLTF textures).
        if self.layout.invalidate_room_geometry(room_id):
            console_logger.debug(f"Invalidated geometry for room: {room_id}")

        return Result(success=True, message=f"Set materials for room '{room_id}'.")

    def _set_exterior_material_impl(self, material_id: str) -> Result:
        """Set material for exterior shell.

        Args:
            material_id: Material ID from get_material() (e.g., "Bricks001_1K-JPG").

        Returns:
            Result indicating success or failure.
        """
        console_logger.info(
            f"Tool called: set_exterior_material(material_id={material_id})"
        )
        material = self.materials_resolver.get_material_by_id(material_id)

        if not material:
            return self._fail(f"Exterior material '{material_id}' not found.")

        self.layout.exterior_material = material

        # Invalidate all geometry (exterior material affects rooms with exterior walls).
        invalidated = self.layout.invalidate_all_room_geometries()
        if invalidated > 0:
            console_logger.debug(f"Invalidated {invalidated} room geometries")

        return Result(
            success=True, message=f"Set exterior material to '{material_id}'."
        )

    def _list_room_materials_impl(self) -> MaterialsListResult:
        """List all materials currently assigned to rooms.

        Use to check existing materials before setting new ones for consistency.

        Returns:
            Dict mapping room_id to {wall_material_id, floor_material_id}.
            Also includes exterior_material_id if set.
        """
        console_logger.info("Tool called: list_room_materials")
        materials = {}

        for room_id, room_mat in self.layout.room_materials.items():
            materials[room_id] = {
                "wall_material_id": (
                    room_mat.wall_material.material_id if room_mat.wall_material else ""
                ),
                "floor_material_id": (
                    room_mat.floor_material.material_id
                    if room_mat.floor_material
                    else ""
                ),
            }

        exterior_id = ""
        if self.layout.exterior_material:
            exterior_id = self.layout.exterior_material.material_id

        return MaterialsListResult(
            success=True,
            message=f"Found materials for {len(materials)} room(s).",
            materials=materials,
            exterior_material_id=exterior_id,
        )

    def _validate_impl(self) -> ValidationResult:
        """Validate current design state.

        Call after room layout changes or door changes to catch issues early.
        Also use as final check before completing design.

        Returns:
            ValidationResult with status for each check:
            - layout: room placement (no overlaps, adjacencies satisfied)
            - connectivity: all rooms reachable from exterior via doors
        """
        console_logger.info("Tool called: validate")
        layout_status = "ok"
        connectivity_status = "ok"

        # Check layout.
        if not self.layout.placement_valid:
            layout_status = "error: room placement not completed or invalid"
        elif not self.layout.placed_rooms:
            layout_status = "error: no rooms placed"

        # Check connectivity.
        if self.layout.placed_rooms:
            is_valid, msg = validate_connectivity(
                self.layout.placed_rooms,
                self.layout.doors,
                self.layout.room_specs,
            )
            if not is_valid:
                connectivity_status = f"error: {msg}"
            self.layout.connectivity_valid = is_valid
        else:
            connectivity_status = "error: no rooms to validate"

        # Log validation result.
        is_valid = layout_status == "ok" and connectivity_status == "ok"
        if is_valid:
            console_logger.info("Validation passed: layout=ok, connectivity=ok")
        else:
            console_logger.info(
                f"Validation failed: layout={layout_status}, "
                f"connectivity={connectivity_status}"
            )

        return ValidationResult(layout=layout_status, connectivity=connectivity_status)

    def _render_ascii_impl(self) -> str:
        """Generate text representation of floor plan.

        Shows room boundaries, room names, and wall segment labels (A, B, C...).
        Use for quick layout overview or when planning door/window placement.

        Returns:
            ASCII floor plan string.
        """
        console_logger.info("Tool called: render_ascii")
        if not self.layout.placed_rooms:
            return "(No rooms to render)"

        result = generate_ascii_floor_plan(self.layout.placed_rooms)
        return f"{result.ascii_art}\n\n{result.legend}"
