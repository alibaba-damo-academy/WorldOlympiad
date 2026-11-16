"""Wall surface dataclass and extraction utilities.

Provides WallSurface for representing placeable wall areas and extraction
functions to create them from room geometry.
"""

from dataclasses import dataclass

import numpy as np

from pydrake.common.eigen_geometry import Quaternion
from pydrake.math import RigidTransform, RotationMatrix

from scenecode.agent_utils.house import (
    HouseLayout,
    OpeningType,
    RoomGeometry,
    Wall,
    WallDirection,
)
from scenecode.agent_utils.room import UniqueID


@dataclass
class WallSurface:
    """Represents a wall surface where objects can be mounted.

    Coordinate system (wall-local frame):
        Origin: wall start_point at floor level
        +X: along wall (from start to end)
        +Y: outward (away from room, opposite of wall normal into room)
        +Z: up (vertical)

    Note: +Y points outward (not into room) to ensure a right-handed coordinate
    system (Z = X × Y). This is required for valid quaternion representation.
    Wall-local Y coordinate is always 0 for placements since objects are on the
    wall surface.
    """

    surface_id: UniqueID
    """Unique identifier for the wall surface (e.g., WS_0)."""

    wall_id: str
    """Wall identifier within the room (e.g., 'living_room_north')."""

    wall_direction: WallDirection
    """Cardinal direction the wall faces (NORTH/SOUTH/EAST/WEST)."""

    bounding_box_min: list[float]
    """Minimum corner [x, y, z] in wall-local frame. Always [0, 0, 0]."""

    bounding_box_max: list[float]
    """Maximum corner [x, y, z] in wall-local frame. [length, 0.0, height]."""

    transform: RigidTransform
    """Pose of wall surface origin in world frame."""

    excluded_regions: list[tuple[float, float, float, float]]
    """Doors/windows as (x_min, z_min, x_max, z_max) in wall-local coordinates."""

    @property
    def length(self) -> float:
        """Wall length in meters."""
        return self.bounding_box_max[0]

    @property
    def height(self) -> float:
        """Wall height in meters."""
        return self.bounding_box_max[2]

    def to_world_pose(
        self, position_x: float, position_z: float, rotation_deg: float = 0.0
    ) -> RigidTransform:
        """Convert wall SE(2) position to world SE(3) pose.

        Args:
            position_x: Position along wall (meters from wall start).
            position_z: Height on wall (meters from floor).
            rotation_deg: Rotation around wall normal (degrees). Positive = CCW
                when looking at wall from inside room.

        Returns:
            World-frame RigidTransform for object placement.
        """
        # Position in wall-local coordinates.
        # Y = 0 since object is on the wall surface.
        local_position = np.array([position_x, 0.0, position_z])

        # Base rotation: flip 180° about Z so object front (+Y after
        # canonicalization) faces into the room (wall -Y) instead of outward.
        base_rotation = RotationMatrix.MakeZRotation(np.pi)

        # SE(2) rotation about Y axis (wall normal).
        # Positive rotation_deg = CCW when looking at wall from inside room.
        # Looking at wall = looking in +Y direction (from room at -Y toward wall).
        # CCW from viewer's perspective = +Z rotates toward -X = Ry(-θ).
        se2_rotation = RotationMatrix.MakeYRotation(-np.deg2rad(rotation_deg))

        # Combined: apply base rotation first (extrinsic), then SE(2) rotation.
        local_rotation = se2_rotation.multiply(base_rotation)

        # Combine local pose.
        local_pose = RigidTransform(R=local_rotation, p=local_position)

        # Transform to world frame.
        return self.transform.multiply(local_pose)

    def contains_point_2d(self, position_x: float, position_z: float) -> bool:
        """Check if point is within wall bounds and not in excluded region.

        Args:
            position_x: Position along wall (meters from wall start).
            position_z: Height on wall (meters from floor).

        Returns:
            True if point is valid for placement, False otherwise.
        """
        # Check wall bounds.
        if not (0 <= position_x <= self.length and 0 <= position_z <= self.height):
            return False

        # Check excluded regions (doors/windows).
        for x_min, z_min, x_max, z_max in self.excluded_regions:
            if x_min <= position_x <= x_max and z_min <= position_z <= z_max:
                return False

        return True

    def check_object_bounds(
        self,
        position_x: float,
        position_z: float,
        object_width: float,
        object_height: float,
    ) -> tuple[bool, str | None]:
        """Check if an object of given size can be placed at position.

        Args:
            position_x: Center X position along wall.
            position_z: Center Z position (height).
            object_width: Object width (along wall).
            object_height: Object height (vertical).

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        half_width = object_width / 2
        half_height = object_height / 2

        obj_x_min = position_x - half_width
        obj_x_max = position_x + half_width
        obj_z_min = position_z - half_height
        obj_z_max = position_z + half_height

        # Check wall bounds.
        if obj_x_min < 0 or obj_x_max > self.length:
            return False, f"Object extends beyond wall width (0 to {self.length:.2f}m)"
        if obj_z_min < 0 or obj_z_max > self.height:
            return False, f"Object extends beyond wall height (0 to {self.height:.2f}m)"

        # Check excluded regions (doors/windows).
        for x_min, z_min, x_max, z_max in self.excluded_regions:
            # Check AABB intersection.
            if (
                obj_x_min < x_max
                and obj_x_max > x_min
                and obj_z_min < z_max
                and obj_z_max > z_min
            ):
                return False, (
                    f"Object overlaps opening at "
                    f"({x_min:.2f}, {z_min:.2f}) to ({x_max:.2f}, {z_max:.2f})"
                )

        return True, None

    def to_dict(self) -> dict:
        """Serialize for checkpoints and action logs."""
        return {
            "surface_id": str(self.surface_id),
            "wall_id": self.wall_id,
            "wall_direction": self.wall_direction.value,
            "bounding_box_min": self.bounding_box_min,
            "bounding_box_max": self.bounding_box_max,
            "transform": _rigid_transform_to_list(self.transform),
            "excluded_regions": list(self.excluded_regions),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WallSurface":
        """Deserialize from dictionary."""
        return cls(
            surface_id=UniqueID(data["surface_id"]),
            wall_id=data["wall_id"],
            wall_direction=WallDirection(data["wall_direction"]),
            bounding_box_min=data["bounding_box_min"],
            bounding_box_max=data["bounding_box_max"],
            transform=_list_to_rigid_transform(data["transform"]),
            excluded_regions=[tuple(r) for r in data["excluded_regions"]],
        )


def _rigid_transform_to_list(transform: RigidTransform) -> list[float]:
    """Convert RigidTransform to flat list [x, y, z, qw, qx, qy, qz]."""
    p = transform.translation()
    q = transform.rotation().ToQuaternion()
    return [p[0], p[1], p[2], q.w(), q.x(), q.y(), q.z()]


def _list_to_rigid_transform(data: list[float]) -> RigidTransform:
    """Convert flat list [x, y, z, qw, qx, qy, qz] to RigidTransform."""
    p = np.array(data[:3])
    q = Quaternion(w=data[3], x=data[4], y=data[5], z=data[6])
    return RigidTransform(R=RotationMatrix(q), p=p)


def _compute_wall_rotation(direction: WallDirection) -> RotationMatrix:
    """Compute rotation matrix for wall surface coordinate frame.

    Creates a right-handed coordinate system:
        X: along wall
        Y: outward (away from room)
        Z: up (vertical)

    Args:
        direction: Wall direction (NORTH/SOUTH/EAST/WEST).

    Returns:
        RotationMatrix for the wall surface frame.
    """
    inward_normal = np.array(direction.get_inward_normal())
    outward_normal = -inward_normal  # Flip to outward for right-handed system.

    # Build rotation matrix using cross product to guarantee right-handed.
    col_y = np.array([outward_normal[0], outward_normal[1], 0.0])
    col_z = np.array([0.0, 0.0, 1.0])
    col_x = np.cross(col_y, col_z)  # Y × Z = X.

    rotation_matrix = np.column_stack([col_x, col_y, col_z])
    return RotationMatrix(rotation_matrix)


def _compute_wall_origin_and_length(
    direction_str: str, half_length: float, half_width: float, wall_thickness: float
) -> tuple[np.ndarray, float]:
    """Compute wall origin position and length based on direction.

    Wall origin is offset inward by wall_thickness so it's at the inside
    surface (facing room), not the outer room boundary.

    Args:
        direction_str: Wall direction ("north", "south", "east", "west").
        half_length: Half of room length (X dimension).
        half_width: Half of room width (Y dimension).
        wall_thickness: Wall thickness in meters.

    Returns:
        Tuple of (wall_origin as 2D array, wall_length in meters).
    """
    if direction_str == "north":
        # North wall at +Y, local X runs +X direction.
        origin = np.array([-half_length + wall_thickness, half_width - wall_thickness])
        length = 2 * half_length - 2 * wall_thickness
    elif direction_str == "south":
        # South wall at -Y, local X runs -X direction.
        origin = np.array([half_length - wall_thickness, -half_width + wall_thickness])
        length = 2 * half_length - 2 * wall_thickness
    elif direction_str == "east":
        # East wall at +X, local X runs -Y direction.
        origin = np.array([half_length - wall_thickness, half_width - wall_thickness])
        length = 2 * half_width - 2 * wall_thickness
    else:  # west
        # West wall at -X, local X runs +Y direction.
        origin = np.array([-half_length + wall_thickness, -half_width + wall_thickness])
        length = 2 * half_width - 2 * wall_thickness

    return origin, length


def _make_wall_id(room_id: str | None, direction_str: str) -> str:
    """Create wall ID from room ID and direction.

    Args:
        room_id: Optional room ID prefix (e.g., "dining_room").
        direction_str: Wall direction ("north", "south", "east", "west").

    Returns:
        Wall ID (e.g., "dining_room_north" or "north_wall").
    """
    if room_id:
        return f"{room_id}_{direction_str}"
    return f"{direction_str}_wall"


def extract_wall_surfaces(
    house_layout: HouseLayout,
    room_id: str,
    ceiling_height: float,
    wall_thickness: float = 0.05,
) -> list[WallSurface]:
    """Extract WallSurface objects from room walls.

    Creates a WallSurface for each of the 4 walls in the room, with excluded
    regions for doors and windows.

    Wall coordinates are transformed from house coordinates (room corner at
    placed_room.position) to room-local coordinates (room center at origin),
    matching the coordinate system used by RoomGeometry.openings and rendering.

    Args:
        house_layout: HouseLayout containing wall geometry.
        room_id: Room to extract surfaces for.
        ceiling_height: Height of ceiling (meters).
        wall_thickness: Wall thickness in meters for surface offset.

    Returns:
        List of WallSurface objects for the room (exactly 4).

    Raises:
        ValueError: If room not found in house layout.
    """
    placed_room = house_layout.get_placed_room(room_id=room_id)
    if placed_room is None:
        raise ValueError(f"Room '{room_id}' not found in house layout")

    # Compute offset to transform from house coords to room-local coords.
    # House coords: room corner at placed_room.position.
    # Room-local coords: room center at origin.
    # This matches compute_openings_data() in clearance_zones.py.
    offset_x = placed_room.position[0] + placed_room.width / 2
    offset_y = placed_room.position[1] + placed_room.depth / 2

    wall_surfaces = []

    for wall in placed_room.walls:
        surface = _create_wall_surface(
            wall=wall,
            ceiling_height=ceiling_height,
            offset_x=offset_x,
            offset_y=offset_y,
            wall_thickness=wall_thickness,
        )
        wall_surfaces.append(surface)

    return wall_surfaces


def _create_wall_surface(
    wall: Wall,
    ceiling_height: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    wall_thickness: float = 0.05,
) -> WallSurface:
    """Create a WallSurface from a Wall dataclass.

    Args:
        wall: Wall geometry from house layout (in house coordinates).
        ceiling_height: Height of ceiling in meters.
        offset_x: X offset to transform from house to room-local coords.
        offset_y: Y offset to transform from house to room-local coords.
        wall_thickness: Wall thickness in meters for surface offset.

    Returns:
        WallSurface for this wall (in room-local coordinates).
    """
    rotation = _compute_wall_rotation(wall.direction)
    inward_normal = np.array(wall.direction.get_inward_normal())

    # Transform wall coordinates from house to room-local coords.
    direction_str = wall.direction.value
    start = np.array([wall.start_point[0] - offset_x, wall.start_point[1] - offset_y])
    end = np.array([wall.end_point[0] - offset_x, wall.end_point[1] - offset_y])

    # For each direction, determine which corner is the origin based on
    # the cross-product-derived X axis direction.
    # Track whether origin is at start (for opening position conversion).
    origin_is_start = False
    if direction_str == "north":
        # X axis points +X (east), so origin is at west end (min X).
        origin_is_start = start[0] < end[0]
        wall_origin = start if origin_is_start else end
    elif direction_str == "south":
        # X axis points -X (west), so origin is at east end (max X).
        origin_is_start = start[0] > end[0]
        wall_origin = start if origin_is_start else end
    elif direction_str == "east":
        # X axis points -Y (south), so origin is at north end (max Y).
        origin_is_start = start[1] > end[1]
        wall_origin = start if origin_is_start else end
    else:  # west
        # X axis points +Y (north), so origin is at south end (min Y).
        origin_is_start = start[1] < end[1]
        wall_origin = start if origin_is_start else end

    # Offset wall origin inward by wall thickness so surface is at inside
    # face (facing room), not outer room boundary. This matches the offset
    # applied in _create_wall_surface_from_direction() for consistency.
    wall_origin = wall_origin + inward_normal * wall_thickness

    transform = RigidTransform(
        R=rotation,
        p=np.array([wall_origin[0], wall_origin[1], 0.0]),
    )

    # Extract excluded regions from openings.
    excluded_regions = []
    for opening in wall.openings:
        # Convert opening to wall-local (x along wall, z vertical).
        # position_along_wall is the LEFT EDGE of the opening measured from
        # wall.start_point. We need the CENTER for excluded region calculation.
        # Wall surface X coordinate is measured from wall_origin which may be
        # at start or end depending on the wall direction.
        opening_center_from_start = opening.position_along_wall + opening.width / 2
        if origin_is_start:
            x_center = opening_center_from_start
        else:
            # Origin at end: transform position from start-relative to end-relative.
            x_center = wall.length - opening_center_from_start

        half_width = opening.width / 2
        x_min = x_center - half_width
        x_max = x_center + half_width
        z_min = opening.sill_height
        # For OPEN type, use ceiling_height since opening.height may be 0.
        if opening.opening_type == OpeningType.OPEN:
            z_max = ceiling_height - opening.sill_height
        else:
            z_max = opening.sill_height + opening.height
        excluded_regions.append((x_min, z_min, x_max, z_max))

    return WallSurface(
        surface_id=UniqueID(wall.wall_id),
        wall_id=wall.wall_id,
        wall_direction=wall.direction,
        bounding_box_min=[0.0, 0.0, 0.0],
        bounding_box_max=[wall.length, 0.0, ceiling_height],
        transform=transform,
        excluded_regions=excluded_regions,
    )


def extract_wall_surfaces_from_room_geometry(
    room_geometry: "RoomGeometry", room_id: str | None = None
) -> list[WallSurface]:
    """Extract WallSurfaces from RoomGeometry for replay.

    This function reconstructs WallSurface objects from RoomGeometry's openings
    data, which contains wall geometry (start/end points, direction). This is
    used during replay when HouseLayout is not available.

    Args:
        room_geometry: RoomGeometry containing openings and wall_height.
        room_id: Optional room ID to prefix wall_ids (e.g., "dining_room").
            If provided, wall_ids will be "{room_id}_{direction}" (e.g.,
            "dining_room_north"). If None, wall_ids will be "{direction}_wall"
            (e.g., "north_wall").

    Returns:
        List of WallSurface objects (one per wall direction found).
    """
    # Group openings by wall direction.
    openings_by_wall: dict[str, list] = {}
    for opening in room_geometry.openings:
        direction = opening.wall_direction
        if direction not in openings_by_wall:
            openings_by_wall[direction] = []
        openings_by_wall[direction].append(opening)

    # If no openings, we need to infer walls from room dimensions.
    # For rooms without openings, create 4 walls based on width/length.
    if not openings_by_wall:
        return _create_wall_surfaces_from_dimensions(
            width=room_geometry.width,
            length=room_geometry.length,
            ceiling_height=room_geometry.wall_height,
            wall_thickness=room_geometry.wall_thickness,
            room_id=room_id,
        )

    wall_surfaces = []
    ceiling_height = room_geometry.wall_height

    for direction_str, openings in openings_by_wall.items():
        direction = WallDirection(direction_str)

        # Get wall geometry from first opening (all openings on same wall
        # share the same start/end points).
        first_opening = openings[0]

        # Compute wall length from opening data (magnitude is direction-invariant).
        wall_start_opening = np.array(first_opening.wall_start)
        wall_end_opening = np.array(first_opening.wall_end)
        wall_vec = wall_end_opening - wall_start_opening
        wall_length = float(np.linalg.norm(wall_vec))

        rotation = _compute_wall_rotation(direction)

        # Compute wall origin based on direction (ignores computed wall_length).
        wall_start, _ = _compute_wall_origin_and_length(
            direction_str=direction_str,
            half_length=room_geometry.length / 2,
            half_width=room_geometry.width / 2,
            wall_thickness=room_geometry.wall_thickness,
        )

        transform = RigidTransform(
            R=rotation, p=np.array([wall_start[0], wall_start[1], 0.0])
        )

        # Build excluded regions from openings.
        # Transform opening positions from opening.wall_start reference to
        # WallSurface local coordinates for consistent grid alignment.
        excluded_regions = []
        for opening in openings:
            # Compute opening center in world coordinates.
            # opening.position_along_wall is distance from opening.wall_start
            # to opening CENTER (per ClearanceOpeningData docstring).
            opening_wall_start = np.array(opening.wall_start)
            opening_wall_end = np.array(opening.wall_end)
            opening_wall_vec = opening_wall_end - opening_wall_start
            opening_wall_length = np.linalg.norm(opening_wall_vec)
            opening_wall_dir = opening_wall_vec / opening_wall_length

            # Opening center in world XY (Z=0 plane).
            # Note: position_along_wall in ClearanceOpeningData is copied from
            # Opening which uses LEFT EDGE convention, not center.
            opening_center_world = opening_wall_start + opening_wall_dir * (
                opening.position_along_wall + opening.width / 2
            )

            # Transform to WallSurface local coordinates.
            # WallSurface local: X along wall, Y outward, Z up.
            # local_pos = R^T @ (world_pos - origin)
            world_pos_3d = np.array(
                [opening_center_world[0], opening_center_world[1], 0.0]
            )
            local_pos = rotation.inverse().multiply(
                world_pos_3d - transform.translation()
            )
            opening_center_local_x = local_pos[0]

            # Compute excluded region bounds in WallSurface local coordinates.
            half_width_opening = opening.width / 2
            x_min = opening_center_local_x - half_width_opening
            x_max = opening_center_local_x + half_width_opening
            z_min = opening.sill_height
            z_max = opening.sill_height + opening.height
            excluded_regions.append((x_min, z_min, x_max, z_max))

        wall_id = _make_wall_id(room_id, direction_str)
        surface = WallSurface(
            surface_id=UniqueID(wall_id),
            wall_id=wall_id,
            wall_direction=direction,
            bounding_box_min=[0.0, 0.0, 0.0],
            bounding_box_max=[wall_length, 0.0, ceiling_height],
            transform=transform,
            excluded_regions=excluded_regions,
        )
        wall_surfaces.append(surface)

    # Add walls without openings based on room dimensions.
    existing_directions = set(openings_by_wall.keys())
    all_directions = {"north", "south", "east", "west"}
    missing_directions = all_directions - existing_directions

    for direction_str in missing_directions:
        surface = _create_wall_surface_from_direction(
            direction_str=direction_str,
            width=room_geometry.width,
            length=room_geometry.length,
            ceiling_height=ceiling_height,
            wall_thickness=room_geometry.wall_thickness,
            room_id=room_id,
        )
        wall_surfaces.append(surface)

    return wall_surfaces


def _create_wall_surfaces_from_dimensions(
    width: float,
    length: float,
    ceiling_height: float,
    wall_thickness: float,
    room_id: str | None = None,
) -> list[WallSurface]:
    """Create wall surfaces for a room without openings.

    Args:
        width: Room width (Y dimension).
        length: Room length (X dimension).
        ceiling_height: Wall height.
        wall_thickness: Wall thickness in meters.
        room_id: Optional room ID to prefix wall_ids.

    Returns:
        List of 4 WallSurfaces.
    """
    surfaces = []
    for direction_str in ["north", "south", "east", "west"]:
        surface = _create_wall_surface_from_direction(
            direction_str=direction_str,
            width=width,
            length=length,
            ceiling_height=ceiling_height,
            wall_thickness=wall_thickness,
            room_id=room_id,
        )
        surfaces.append(surface)
    return surfaces


def _create_wall_surface_from_direction(
    direction_str: str,
    width: float,
    length: float,
    ceiling_height: float,
    wall_thickness: float,
    room_id: str | None = None,
) -> WallSurface:
    """Create a WallSurface for a given direction based on room dimensions.

    Args:
        direction_str: Wall direction ("north", "south", "east", "west").
        width: Room width (Y dimension).
        length: Room length (X dimension).
        ceiling_height: Wall height.
        wall_thickness: Wall thickness in meters.
        room_id: Optional room ID to prefix wall_id.

    Returns:
        WallSurface for the specified direction.
    """
    direction = WallDirection(direction_str)

    wall_start, wall_length = _compute_wall_origin_and_length(
        direction_str=direction_str,
        half_length=length / 2,
        half_width=width / 2,
        wall_thickness=wall_thickness,
    )

    rotation = _compute_wall_rotation(direction)

    transform = RigidTransform(
        R=rotation,
        p=np.array([wall_start[0], wall_start[1], 0.0]),
    )

    wall_id = _make_wall_id(room_id, direction_str)
    return WallSurface(
        surface_id=UniqueID(wall_id),
        wall_id=wall_id,
        wall_direction=direction,
        bounding_box_min=[0.0, 0.0, 0.0],
        bounding_box_max=[wall_length, 0.0, ceiling_height],
        transform=transform,
        excluded_regions=[],
    )
