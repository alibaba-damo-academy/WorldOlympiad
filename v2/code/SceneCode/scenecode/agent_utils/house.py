"""House layout and room geometry data structures."""

import hashlib
import json
import logging
import os
import time
import xml.etree.ElementTree as ET

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from omegaconf import DictConfig

from scenecode.agent_utils.sceneeval_exporter import (
    SceneEvalExportConfig,
    SceneEvalExporter,
)
from scenecode.utils.material import Material
from scenecode.utils.package_utils import create_package_xml
from scenecode.utils.path_utils import safe_relative_path

if TYPE_CHECKING:
    from scenecode.agent_utils.room import ObjectType, RoomScene, SceneObject

console_logger = logging.getLogger(__name__)


def compute_wall_normals(walls: list["SceneObject"]) -> dict[str, np.ndarray]:
    """Compute room-facing normals for wall objects.

    Normals point from wall center toward room center (0, 0) in the XY plane,
    creating vectors that indicate the "inside" direction of each wall. These
    are used for snap-to-wall orientation calculations.

    Args:
        walls: List of wall SceneObjects.

    Returns:
        Dict mapping wall name to normalized 2D normal vector (X, Y).
    """
    # Room center is at origin (0, 0) for rectangular rooms.
    room_center = np.array([0.0, 0.0])

    wall_normals = {}
    for wall in walls:
        # Wall center position in XY plane.
        wall_center_2d = wall.transform.translation()[:2]

        # Normal points from wall toward room center.
        normal_2d = room_center - wall_center_2d

        # Normalize to unit vector.
        normal_length = np.linalg.norm(normal_2d)
        if normal_length > 1e-6:
            normal_2d = normal_2d / normal_length
        else:
            console_logger.warning(
                f"Wall {wall.name} is at room center, cannot compute normal"
            )
            normal_2d = np.array([0.0, 0.0])

        wall_normals[wall.name] = normal_2d

    return wall_normals


class WallDirection(Enum):
    """Cardinal direction for room walls."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

    def get_inward_normal(self) -> tuple[float, float]:
        """Get unit normal vector pointing INTO the room.

        Returns:
            (nx, ny) unit vector pointing into room interior.
        """
        if self == WallDirection.NORTH:
            return (0.0, -1.0)
        elif self == WallDirection.SOUTH:
            return (0.0, 1.0)
        elif self == WallDirection.EAST:
            return (-1.0, 0.0)
        else:  # WEST
            return (1.0, 0.0)


class OpeningType(Enum):
    """Type of wall opening."""

    DOOR = "door"
    WINDOW = "window"
    OPEN = "open"  # Open floor plan connection (no wall, floor-to-ceiling).


@dataclass
class Opening:
    """Opening (door/window/open connection) in a wall.

    Stored in Wall.openings list. Created when Door, Window, or open connection
    is added. For OPEN type, height is ignored at render time - uses wall_height.
    """

    opening_id: str
    """References Door.id or Window.id."""

    opening_type: OpeningType
    """Type of opening (door or window)."""

    position_along_wall: float
    """Meters from wall start_point."""

    width: float
    """Width of opening in meters."""

    height: float
    """Height of opening in meters."""

    sill_height: float = 0.0

    def to_dict(self) -> dict:
        """Serialize opening to dictionary."""
        return {
            "opening_id": self.opening_id,
            "opening_type": self.opening_type.value,
            "position_along_wall": self.position_along_wall,
            "width": self.width,
            "height": self.height,
            "sill_height": self.sill_height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Opening":
        """Deserialize opening from dictionary."""
        return cls(
            opening_id=data["opening_id"],
            opening_type=OpeningType(data["opening_type"]),
            position_along_wall=data["position_along_wall"],
            width=data["width"],
            height=data["height"],
            sill_height=data.get("sill_height", 0.0),
        )

    """Height from floor to bottom (0 for doors, >0 for windows)."""


@dataclass
class ClearanceOpeningData:
    """Opening data for clearance zone physics checks and label rendering.

    This extends the basic Opening data with computed world-space coordinates
    and clearance zone bounds for physics validation.
    """

    opening_id: str
    """Unique identifier from source Opening."""

    opening_type: str
    """Type: 'door', 'window', or 'open'."""

    wall_direction: str
    """Cardinal direction: 'north', 'south', 'east', 'west'."""

    center_world: list[float]
    """World coordinates [x, y, z] for label positioning."""

    width: float
    """Opening width in meters."""

    sill_height: float
    """Height from floor to bottom (0 for doors, >0 for windows)."""

    height: float
    """Height of opening in meters."""

    clearance_bbox_min: list[float] | None
    """Clearance zone AABB minimum [x, y, z], or None for OPEN type."""

    clearance_bbox_max: list[float] | None
    """Clearance zone AABB maximum [x, y, z], or None for OPEN type."""

    wall_start: list[float]
    """Wall start point [x, y] for open connection sweep."""

    wall_end: list[float]
    """Wall end point [x, y] for open connection sweep."""

    position_along_wall: float
    """Distance from wall start to opening center."""

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "opening_id": self.opening_id,
            "opening_type": self.opening_type,
            "wall_direction": self.wall_direction,
            "center_world": self.center_world,
            "width": self.width,
            "sill_height": self.sill_height,
            "height": self.height,
            "clearance_bbox_min": self.clearance_bbox_min,
            "clearance_bbox_max": self.clearance_bbox_max,
            "wall_start": self.wall_start,
            "wall_end": self.wall_end,
            "position_along_wall": self.position_along_wall,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClearanceOpeningData":
        """Deserialize from dictionary."""
        return cls(
            opening_id=data["opening_id"],
            opening_type=data["opening_type"],
            wall_direction=data["wall_direction"],
            center_world=data["center_world"],
            width=data["width"],
            sill_height=data.get("sill_height", 0.0),
            height=data["height"],
            clearance_bbox_min=data.get("clearance_bbox_min"),
            clearance_bbox_max=data.get("clearance_bbox_max"),
            wall_start=data["wall_start"],
            wall_end=data["wall_end"],
            position_along_wall=data["position_along_wall"],
        )


@dataclass
class Door:
    """Door in house layout.

    Interior doors create openings in both rooms' walls.
    """

    id: str
    """Unique door identifier."""

    boundary_label: str
    """ASCII label from HouseLayout.boundary_labels (e.g., 'A')."""

    position_segment: str
    """'left', 'center', or 'right'."""

    position_exact: float
    """Computed meters from boundary start (randomized within segment)."""

    door_type: str
    """'exterior' or 'interior'."""

    room_a: str
    """First room (or exterior-facing room for exterior doors)."""

    room_b: str | None = None
    """Second room (None if exterior door)."""

    width: float = 1.0
    """Designer-chosen within config range (0.9-1.9m)."""

    height: float = 2.1
    """Designer-chosen within config range (2.0-2.4m)."""

    def to_dict(self) -> dict:
        """Serialize door to dictionary."""
        return {
            "id": self.id,
            "boundary_label": self.boundary_label,
            "position_segment": self.position_segment,
            "position_exact": self.position_exact,
            "door_type": self.door_type,
            "room_a": self.room_a,
            "room_b": self.room_b,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Door":
        """Deserialize door from dictionary."""
        return cls(
            id=data["id"],
            boundary_label=data["boundary_label"],
            position_segment=data["position_segment"],
            position_exact=data["position_exact"],
            door_type=data["door_type"],
            room_a=data["room_a"],
            room_b=data.get("room_b"),
            width=data.get("width", 1.0),
            height=data.get("height", 2.1),
        )


@dataclass
class Window:
    """Window in house layout. Only on exterior walls."""

    id: str
    """Unique window identifier."""

    boundary_label: str
    """ASCII label from HouseLayout.boundary_labels (for agent display only)."""

    position_along_wall: float
    """Meters from wall start (left edge)."""

    room_id: str
    """Room this window belongs to."""

    wall_direction: WallDirection | None = None
    """Stable wall identifier (used instead of boundary_label for lookup)."""

    width: float = 1.2
    """Designer-chosen: small (0.6m) to picture window (3.0m)."""

    height: float = 1.2
    """Designer-chosen: small (0.6m) to floor-to-ceiling (2.0m)."""

    sill_height: float = 0.9
    """Designer-chosen: height from floor (typically 0.9m)."""

    def to_dict(self) -> dict:
        """Serialize window to dictionary."""
        return {
            "id": self.id,
            "boundary_label": self.boundary_label,
            "position_along_wall": self.position_along_wall,
            "room_id": self.room_id,
            "wall_direction": (
                self.wall_direction.value if self.wall_direction else None
            ),
            "width": self.width,
            "height": self.height,
            "sill_height": self.sill_height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Window":
        """Deserialize window from dictionary."""
        dir_str = data.get("wall_direction")
        return cls(
            id=data["id"],
            boundary_label=data["boundary_label"],
            position_along_wall=data["position_along_wall"],
            room_id=data["room_id"],
            wall_direction=WallDirection(dir_str) if dir_str else None,
            width=data.get("width", 1.2),
            height=data.get("height", 1.2),
            sill_height=data.get("sill_height", 0.9),
        )


@dataclass
class RoomMaterials:
    """Materials for a room's surfaces."""

    wall_material: Material | None = None
    """Wall material with PBR textures."""

    floor_material: Material | None = None
    """Floor material with PBR textures."""

    def to_dict(self) -> dict:
        """Serialize room materials to dictionary."""
        return {
            "wall_material": (
                self.wall_material.to_dict() if self.wall_material else None
            ),
            "floor_material": (
                self.floor_material.to_dict() if self.floor_material else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RoomMaterials":
        """Deserialize room materials from dictionary."""
        return cls(
            wall_material=(
                Material.from_dict(data["wall_material"])
                if data.get("wall_material")
                else None
            ),
            floor_material=(
                Material.from_dict(data["floor_material"])
                if data.get("floor_material")
                else None
            ),
        )


@dataclass
class Wall:
    """A room's wall in one direction.

    Each room has exactly 4 walls (N/S/E/W). Wall geometry is computed from
    room position + dimensions. A wall can face multiple rooms in T-junction
    layouts. This preserves 4-wall-per-room structure for rendering logic.
    """

    wall_id: str
    """Format: '{room_id}_{direction}' e.g., 'living_room_north'."""

    room_id: str
    """Room that owns this wall."""

    direction: WallDirection
    """Cardinal direction of this wall."""

    start_point: tuple[float, float]
    """(x, y) start of wall segment in global coordinates."""

    end_point: tuple[float, float]
    """(x, y) end of wall segment in global coordinates."""

    length: float
    """Wall length (computable from points but stored for convenience)."""

    is_exterior: bool = True
    """True if wall faces outside the house."""

    faces_rooms: list[str] = field(default_factory=list)
    """Room IDs on other side (empty if exterior, can be >1 for T-junctions)."""

    openings: list[Opening] = field(default_factory=list)
    """Openings (doors/windows) in this wall."""

    def cache_key(self, wall_height: float, material: Material | None = None) -> str:
        """Generate cache key for wall GLTF caching.

        The cache key includes all properties that affect the wall's geometry
        and appearance. If any of these change, the wall GLTF must be regenerated.

        Args:
            wall_height: Wall height in meters (from layout).
            material: Wall material (affects texture).

        Returns:
            SHA-256 hash string (first 16 chars) for cache lookup.
        """
        # Build state dict with all rendering-relevant properties.
        state = {
            "wall_id": self.wall_id,
            "start_point": self.start_point,
            "end_point": self.end_point,
            "direction": self.direction.value,
            "is_exterior": self.is_exterior,
            "wall_height": wall_height,
            "material": str(material.path) if material else None,
            "openings": [o.to_dict() for o in self.openings],
        }
        content_json = json.dumps(state, sort_keys=True)
        return hashlib.sha256(content_json.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Serialize wall to dictionary."""
        return {
            "wall_id": self.wall_id,
            "room_id": self.room_id,
            "direction": self.direction.value,
            "start_point": list(self.start_point),
            "end_point": list(self.end_point),
            "length": self.length,
            "is_exterior": self.is_exterior,
            "faces_rooms": self.faces_rooms,
            "openings": [opening.to_dict() for opening in self.openings],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Wall":
        """Deserialize wall from dictionary."""
        return cls(
            wall_id=data["wall_id"],
            room_id=data["room_id"],
            direction=WallDirection(data["direction"]),
            start_point=tuple(data["start_point"]),
            end_point=tuple(data["end_point"]),
            length=data["length"],
            is_exterior=data.get("is_exterior", True),
            faces_rooms=data.get("faces_rooms", []),
            openings=[Opening.from_dict(o) for o in data.get("openings", [])],
        )


@dataclass
class PlacedRoom:
    """Room with computed position (derived from RoomSpec).

    Regenerated when spec changes via placement algorithm.
    """

    room_id: str
    """Unique identifier matching RoomSpec.room_id."""

    position: tuple[float, float]
    """(x, y) min corner in global coordinates."""

    width: float
    """Room width in meters."""

    depth: float
    """Room depth in meters."""

    walls: list[Wall] = field(default_factory=list)
    """Exactly 4 walls: N, S, E, W."""

    def to_dict(self) -> dict:
        """Serialize placed room to dictionary."""
        return {
            "room_id": self.room_id,
            "position": list(self.position),
            "width": self.width,
            "depth": self.depth,
            "walls": [wall.to_dict() for wall in self.walls],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlacedRoom":
        """Deserialize placed room from dictionary."""
        return cls(
            room_id=data["room_id"],
            position=tuple(data["position"]),
            width=data["width"],
            depth=data["depth"],
            walls=[Wall.from_dict(w) for w in data.get("walls", [])],
        )


class ConnectionType(str, Enum):
    """Type of connection between adjacent rooms."""

    DOOR = "DOOR"
    """Rooms share a wall with a door opening."""

    OPEN = "OPEN"
    """Rooms have no wall between them (open floor plan)."""


@dataclass
class RoomSpec:
    """Specification for a single room in a house layout.

    Contains design-level information about a room that can be used by floor
    plan generators to create geometry. This is the input to geometry generation,
    not the output.
    """

    room_id: str
    """Unique identifier for the room (e.g., 'main', 'living_room', 'bedroom_1')."""

    room_type: str = "room"
    """Type of room (e.g., 'living_room', 'bedroom', 'kitchen', 'bathroom')."""

    prompt: str = ""
    """Text description/prompt for this room."""

    position: tuple[float, float] = (0.0, 0.0)
    """Position of room origin in house coordinates (x, y)."""

    width: float = 5.0
    """Room width in meters (y-dimension)."""

    length: float = 5.0
    """Room length in meters (x-dimension)."""

    connections: dict[str, ConnectionType] = field(default_factory=dict)
    """Room connections: maps room_id to ConnectionType (DOOR or OPEN)."""

    exterior_walls: set[WallDirection] = field(default_factory=set)
    """Walls that MUST remain exterior (no rooms can be placed adjacent to them).

    Use this for rooms like hallways, receptions, or lobbies that need
    external door access on specific walls. The placement algorithm will
    create clearance zones around these walls to prevent other rooms from
    blocking them.
    """

    def to_dict(self) -> dict:
        """Serialize room spec to dictionary."""
        return {
            "id": self.room_id,
            "type": self.room_type,
            "position": list(self.position),
            "width": self.width,
            "length": self.length,
            "prompt": self.prompt,
            "connections": {k: v.value for k, v in self.connections.items()},
            "exterior_walls": [w.value for w in self.exterior_walls],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RoomSpec":
        """Deserialize room spec from dictionary."""
        connections = {}
        if "connections" in data:
            connections = {k: ConnectionType(v) for k, v in data["connections"].items()}
        exterior_walls: set[WallDirection] = set()
        if "exterior_walls" in data:
            exterior_walls = {WallDirection(w) for w in data["exterior_walls"]}
        return cls(
            room_id=data["id"],
            room_type=data.get("type", "room"),
            prompt=data.get("prompt", ""),
            position=tuple(data.get("position", [0.0, 0.0])),
            width=data.get("width", 5.0),
            length=data.get("length", 6.0),
            connections=connections,
            exterior_walls=exterior_walls,
        )


@dataclass
class RoomGeometry:
    """Generated 3D geometry for a single room.

    Contains the physical structural elements (walls, floor) and their SDFormat
    representations. This is the the actual 3D geometry that gets loaded into Drake for
    simulation.

    Contrast with RoomSpec which is the design input (dimensions, position).
    """

    sdf_tree: ET.ElementTree
    """The SDF tree of the full room geometry."""

    sdf_path: Path
    """Path to the SDF file containing floor, walls, doors, windows, etc."""

    walls: list["SceneObject"] = field(default_factory=list)
    """Wall objects (immutable architectural elements)."""

    floor: "SceneObject | None" = None
    """Floor object (immutable architectural element, optional placement surface)."""

    wall_normals: dict[str, np.ndarray] = field(default_factory=dict)
    """Pre-computed room-facing normals for walls.

    Key: wall name (e.g., "north_wall", "south_wall", "east_wall", "west_wall")
    Value: 2D normalized normal vector (X, Y) pointing from wall center toward room center
    """

    width: float = 0.0
    """Room width in meters (y-dimension)."""

    length: float = 0.0
    """Room length in meters (x-dimension)."""

    wall_height: float = 2.5
    """Wall height in meters (needed for wall height violation check)."""

    wall_thickness: float = 0.05
    """Wall thickness in meters (needed for wall surface offset from room boundary)."""

    openings: list["ClearanceOpeningData"] = field(default_factory=list)
    """All door/window/open openings with physics and rendering data."""

    def content_hash(self) -> str:
        """Generate content hash for this floor plan."""
        floor_plan_dict = {
            "sdf_path": str(self.sdf_path) if self.sdf_path else "",
        }

        # Hash SDF file content.
        sdf_path_str = floor_plan_dict["sdf_path"]
        if sdf_path_str:
            try:
                path = Path(sdf_path_str)
                if path.exists():
                    # SDF files are XML-based text files, so read as UTF-8.
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    floor_plan_dict["sdf_path_content_hash"] = hashlib.sha256(
                        content.encode()
                    ).hexdigest()
                else:
                    floor_plan_dict["sdf_path_content_hash"] = ""
            except Exception as e:
                console_logger.warning(
                    f"Could not hash file content for {sdf_path_str}: {e}"
                )
                floor_plan_dict["sdf_path_content_hash"] = ""

        # Add walls and floor content hashes.
        floor_plan_dict["walls"] = [wall.content_hash() for wall in self.walls]
        floor_plan_dict["floor"] = self.floor.content_hash() if self.floor else None

        # Convert to JSON string with sorted keys for determinism.
        content_json = json.dumps(floor_plan_dict, sort_keys=True)

        # Generate SHA-256 hash.
        return hashlib.sha256(content_json.encode()).hexdigest()

    def to_dict(self, scene_dir: Path | None = None) -> dict[str, Any]:
        """Serialize RoomGeometry to dictionary.

        Args:
            scene_dir: Optional scene directory for path relativization.
                       If None, paths are stored as absolute paths.

        Returns:
            Dictionary containing floor plan state (excluding sdf_tree which
            will be re-parsed from file).
        """
        # Convert paths (relative or absolute).
        sdf_path_str = (
            safe_relative_path(self.sdf_path, scene_dir) if self.sdf_path else None
        )

        # Serialize floor if present.
        floor_data = None
        if self.floor:
            floor_data = self.floor.to_dict(scene_dir=scene_dir)

        # Serialize walls.
        walls_data = [w.to_dict(scene_dir=scene_dir) for w in self.walls]

        # Serialize wall_normals (convert numpy arrays to lists).
        wall_normals_data = {}
        for wall_name, normal_vec in self.wall_normals.items():
            wall_normals_data[wall_name] = normal_vec.tolist()

        return {
            "sdf_path": sdf_path_str,
            "walls": walls_data,
            "width": self.width,
            "length": self.length,
            "wall_height": self.wall_height,
            "wall_thickness": self.wall_thickness,
            "openings": [o.to_dict() for o in self.openings],
            "floor": floor_data,
            "wall_normals": wall_normals_data,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], scene_dir: Path | None = None
    ) -> "RoomGeometry":
        """Deserialize RoomGeometry from dictionary.

        Args:
            data: Dictionary containing floor plan state.
            scene_dir: Optional scene directory for path resolution.

        Returns:
            RoomGeometry instance reconstructed from dictionary.

        Raises:
            ValueError: If SDF file is missing (fail-fast for research codebase).

        Note:
            sdf_tree is re-parsed from the sdf_path file.
        """
        # Import here to avoid circular import.
        from scenecode.agent_utils.room import SceneObject

        # Resolve paths relative to scene_dir.
        sdf_path = None
        if data["sdf_path"]:
            sdf_path = (
                scene_dir / data["sdf_path"] if scene_dir else Path(data["sdf_path"])
            )

        # Re-parse sdf_tree from file.
        if not sdf_path or not sdf_path.exists():
            raise ValueError(
                f"SDF file not found at {sdf_path}. Floor plan cannot be restored "
                "without SDF file."
            )
        sdf_tree = ET.parse(sdf_path)

        # Restore floor if present.
        floor = None
        if data.get("floor"):
            floor = SceneObject.from_dict(data["floor"], scene_dir=scene_dir)

        # Restore walls if present.
        walls = []
        if "walls" in data:
            walls = [
                SceneObject.from_dict(w, scene_dir=scene_dir) for w in data["walls"]
            ]

        # Restore wall_normals (convert lists back to numpy arrays).
        wall_normals = {}
        if "wall_normals" in data:
            for wall_name, normal_list in data["wall_normals"].items():
                wall_normals[wall_name] = np.array(normal_list)

        return cls(
            sdf_tree=sdf_tree,
            sdf_path=sdf_path,
            walls=walls,
            floor=floor,
            wall_normals=wall_normals,
            width=data["width"],
            length=data["length"],
            wall_height=data.get("wall_height", 2.5),
            wall_thickness=data.get("wall_thickness", 0.05),
            openings=[
                ClearanceOpeningData.from_dict(o) for o in data.get("openings", [])
            ],
        )


@dataclass
class HouseLayout:
    """Layout specification for a house with one or more rooms.

    HouseLayout is the unified data structure for both room mode (single room)
    and house mode (multiple rooms). Room mode is simply a HouseLayout with
    one room. This eliminates separate code paths for the two modes.

    The floor plan generator receives a HouseLayout and populates the
    room_geometries dict with generated geometry for each room. Following stage agents
    don't interact with HouseLayout directly - they receive RoomScene instances
    with RoomGeometry.
    """

    wall_height: float = 2.5
    """Wall height in meters (default 2.5m, agent can override via set_wall_height)."""

    house_prompt: str = ""
    """Original user prompt for the house/room."""

    room_specs: list[RoomSpec] = field(default_factory=list)
    """Specifications for each room in the house."""

    room_geometries: dict[str, RoomGeometry] = field(default_factory=dict)
    """Generated room geometry for each room (room_id -> RoomGeometry)."""

    house_dir: Path | None = None
    """Directory for house-level outputs."""

    # Placed rooms (derived from specs via placement algorithm).
    placed_rooms: list[PlacedRoom] = field(default_factory=list)
    """Rooms with computed positions after placement algorithm."""

    # Doors and windows.
    doors: list[Door] = field(default_factory=list)
    """All doors in the house."""

    windows: list[Window] = field(default_factory=list)
    """All windows in the house."""

    # Materials per room (interior walls + floors).
    room_materials: dict[str, RoomMaterials] = field(default_factory=dict)
    """Materials for each room (room_id -> RoomMaterials)."""

    # Exterior shell material (consistent for entire house).
    exterior_material: Material | None = None
    """Exterior material (brick, siding, etc.) with PBR textures."""

    # Validation state.
    placement_valid: bool = False
    """True if room placement satisfies all constraints."""

    connectivity_valid: bool = False
    """True if all rooms are reachable from exterior via doors."""

    # ASCII boundary labels (generated dynamically).
    boundary_labels: dict[str, tuple[str, str | None, str | None]] = field(
        default_factory=dict
    )
    """Maps label (A, B, C...) to (room_a, room_b, direction).

    For interior walls: (room_a, room_b, None) - direction not needed.
    For exterior walls: (room_a, None, direction) - direction is wall facing (north, south, etc).
    """

    def __post_init__(self) -> None:
        """Create package.xml for Drake package://scene/ URI resolution."""
        if self.house_dir is not None:
            package_xml_path = self.house_dir / "package.xml"
            if not package_xml_path.exists():
                self.house_dir.mkdir(parents=True, exist_ok=True)
                create_package_xml(self.house_dir)
                console_logger.debug(
                    f"Created package.xml at {package_xml_path} for scene portability"
                )

    def get_room_spec(self, room_id: str) -> RoomSpec | None:
        """Get room specification by ID.

        Args:
            room_id: The room ID to look up.

        Returns:
            RoomSpec if found, None otherwise.
        """
        for spec in self.room_specs:
            if spec.room_id == room_id:
                return spec
        return None

    def get_placed_room(self, room_id: str) -> PlacedRoom | None:
        """Get placed room by ID.

        Args:
            room_id: The room ID to look up.

        Returns:
            PlacedRoom if found, None otherwise.
        """
        for placed_room in self.placed_rooms:
            if placed_room.room_id == room_id:
                return placed_room
        return None

    def get_room_geometry(self, room_id: str) -> RoomGeometry | None:
        """Get generated geometry for a room.

        Args:
            room_id: The room ID to look up.

        Returns:
            RoomGeometry if generated, None otherwise.
        """
        return self.room_geometries.get(room_id)

    def set_room_geometry(self, room_id: str, geometry: RoomGeometry) -> None:
        """Store generated geometry for a room.

        Args:
            room_id: The room ID.
            geometry: The generated RoomGeometry.

        Raises:
            ValueError: If room_id is not in room_specs.
        """
        if not any(spec.room_id == room_id for spec in self.room_specs):
            raise ValueError(f"Unknown room_id: {room_id}")
        self.room_geometries[room_id] = geometry

    def invalidate_room_geometry(self, room_id: str) -> bool:
        """Invalidate cached geometry for a specific room.

        Call this when room properties change (dimensions, walls, materials,
        openings) to force regeneration on next render.

        Args:
            room_id: The room ID to invalidate.

        Returns:
            True if geometry was invalidated, False if room had no cached geometry.
        """
        if room_id in self.room_geometries:
            del self.room_geometries[room_id]
            return True
        return False

    def invalidate_all_room_geometries(self) -> int:
        """Invalidate all cached room geometries.

        Call this when global properties change (wall_height, exterior materials)
        or when the entire layout is regenerated.

        Returns:
            Number of rooms that had cached geometry invalidated.
        """
        count = len(self.room_geometries)
        self.room_geometries.clear()
        return count

    @property
    def room_ids(self) -> list[str]:
        """Get list of all room IDs in order."""
        return [spec.room_id for spec in self.room_specs]

    def to_dict(self, scene_dir: Path | None = None) -> dict[str, Any]:
        """Serialize HouseLayout to dictionary for JSON export.

        Args:
            scene_dir: Optional scene directory for path relativization.
                       If None, paths are stored as absolute paths.

        Returns:
            Dictionary suitable for saving as house_layout.json.
        """
        # Serialize placed_rooms if present.
        placed_rooms_data = None
        if self.placed_rooms is not None:
            placed_rooms_data = [placed.to_dict() for placed in self.placed_rooms]

        # Serialize room_geometries if present.
        room_geometries_data = {}
        for room_id, geometry in self.room_geometries.items():
            room_geometries_data[room_id] = geometry.to_dict(scene_dir=scene_dir)

        return {
            "wall_height": self.wall_height,
            "house_prompt": self.house_prompt,
            "rooms": [spec.to_dict() for spec in self.room_specs],
            "placed_rooms": placed_rooms_data,
            "doors": [door.to_dict() for door in self.doors],
            "windows": [window.to_dict() for window in self.windows],
            "room_materials": {
                room_id: materials.to_dict()
                for room_id, materials in self.room_materials.items()
            },
            "exterior_material": (
                self.exterior_material.to_dict() if self.exterior_material else None
            ),
            "placement_valid": self.placement_valid,
            "connectivity_valid": self.connectivity_valid,
            "boundary_labels": {k: list(v) for k, v in self.boundary_labels.items()},
            "room_geometries": room_geometries_data,
        }

    def to_drake_directive(self, base_dir: Path | None = None) -> str:
        """Generate a Drake directive string for all room geometries.

        Creates a directive that includes all room geometry SDFs, with a
        house_frame at the root and room frames as children. Each room
        geometry is welded to its room frame.

        Args:
            base_dir: If provided, SDF paths are relative to this directory
                (for portable directives). The directive YAML file should be
                saved in this directory for Drake to resolve paths correctly.
                If None, absolute paths with file:// scheme are used.

        Returns:
            Drake directive in YAML format.

        Raises:
            ValueError: If no room geometries have been generated.
        """
        if not self.room_geometries:
            raise ValueError(
                "No room geometries have been generated. "
                "Call generate_house_layout first."
            )

        def format_sdf_path(sdf_path: Path | str | None) -> str:
            """Format SDF path as package:// URI or absolute file:// URI."""
            if sdf_path is None:
                return ""
            sdf_path = Path(sdf_path)
            if base_dir is not None:
                # Use package://scene/ for portable scenes.
                # Drake resolves this via PackageMap (set ROS_PACKAGE_PATH or
                # call parser.package_map().Add("scene", scene_dir)).
                rel_path = os.path.relpath(sdf_path, base_dir)
                return f"package://scene/{rel_path}"
            else:
                return f"file://{sdf_path.absolute()}"

        # Build lookup from room_id to PlacedRoom for positions.
        placed_room_lookup = {room.room_id: room for room in self.placed_rooms}

        directive = """directives:
- add_frame:
    name: house_frame
    X_PF:
      base_frame: world
      translation: [0, 0, 0]"""

        for room_id, room_geometry in self.room_geometries.items():
            # Get room position from placed_rooms (not room_specs).
            placed_room = placed_room_lookup.get(room_id)
            if placed_room is None:
                console_logger.warning(
                    f"Room '{room_id}' not found in placed_rooms, skipping"
                )
                continue

            # PlacedRoom.position is (x, y) of min corner.
            # Room geometry is centered at origin, so translate to room center.
            room_center_x = placed_room.position[0] + placed_room.width / 2
            room_center_y = placed_room.position[1] + placed_room.depth / 2

            room_frame_name = f"room_{room_id}_frame"
            model_name = f"room_geometry_{room_id}"
            room_geom_path = format_sdf_path(room_geometry.sdf_path)

            # Add room frame as child of house_frame.
            directive += f"""
- add_frame:
    name: {room_frame_name}
    X_PF:
      base_frame: house_frame
      translation: [{room_center_x}, {room_center_y}, 0.0]
- add_model:
    name: {model_name}
    file: {room_geom_path}
- add_weld:
    parent: {room_frame_name}
    child: {model_name}::room_geometry_body_link"""

        return directive

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], house_dir: Path | None = None
    ) -> "HouseLayout":
        """Restore HouseLayout from dictionary.

        Args:
            data: Dictionary from to_dict() or house_layout.json.
            house_dir: Directory for house outputs.

        Returns:
            Restored HouseLayout instance.
        """
        room_specs = [RoomSpec.from_dict(r) for r in data.get("rooms", [])]
        doors = [Door.from_dict(d) for d in data.get("doors", [])]
        windows = [Window.from_dict(w) for w in data.get("windows", [])]

        # Restore room materials.
        room_materials = {
            room_id: RoomMaterials.from_dict(mat_data)
            for room_id, mat_data in data.get("room_materials", {}).items()
        }

        # Restore exterior material.
        exterior_material = None
        if data.get("exterior_material"):
            exterior_material = Material.from_dict(data["exterior_material"])

        # Restore boundary labels.
        boundary_labels = {
            label: tuple(room_pair)
            for label, room_pair in data.get("boundary_labels", {}).items()
        }

        # Restore placed_rooms if present.
        placed_rooms = None
        if data.get("placed_rooms") is not None:
            placed_rooms = [PlacedRoom.from_dict(p) for p in data["placed_rooms"]]

        # Restore room_geometries if present.
        room_geometries = {}
        if data.get("room_geometries"):
            for room_id, geom_data in data["room_geometries"].items():
                room_geometries[room_id] = RoomGeometry.from_dict(
                    geom_data, scene_dir=house_dir
                )

        return cls(
            wall_height=data.get("wall_height", 2.5),
            house_prompt=data.get("house_prompt", ""),
            room_specs=room_specs,
            house_dir=house_dir,
            room_geometries=room_geometries,
            placed_rooms=placed_rooms,
            doors=doors,
            windows=windows,
            room_materials=room_materials,
            exterior_material=exterior_material,
            placement_valid=data.get("placement_valid", False),
            connectivity_valid=data.get("connectivity_valid", False),
            boundary_labels=boundary_labels,
        )

    def content_hash(self) -> str:
        """Generate deterministic hash of layout state for render caching.

        Creates a SHA-256 hash of all layout properties that affect rendering.
        Identical layouts produce identical hashes. Used to cache final renders.

        Returns:
            SHA-256 hash string (first 16 chars) of layout content.
        """
        # Build comprehensive state dict.
        state = {
            "wall_height": self.wall_height,
            "placed_rooms": [
                {
                    "room_id": r.room_id,
                    "position": r.position,
                    "width": r.width,
                    "depth": r.depth,
                    # Include wall cache keys for each wall.
                    "walls": [
                        w.cache_key(
                            wall_height=self.wall_height,
                            material=self._get_wall_material(r.room_id),
                        )
                        for w in r.walls
                    ],
                }
                for r in self.placed_rooms
            ],
            "room_materials": {
                room_id: {
                    "wall": str(m.wall_material.path) if m.wall_material else None,
                    "floor": str(m.floor_material.path) if m.floor_material else None,
                }
                for room_id, m in self.room_materials.items()
            },
            "exterior_material": (
                str(self.exterior_material.path) if self.exterior_material else None
            ),
        }
        content_json = json.dumps(state, sort_keys=True)
        return hashlib.sha256(content_json.encode()).hexdigest()[:16]

    def _get_wall_material(self, room_id: str) -> Material | None:
        """Get wall material for a room.

        Args:
            room_id: Room to get wall material for.

        Returns:
            Material or None if using default.
        """
        room_materials = self.room_materials.get(room_id)
        if room_materials:
            return room_materials.wall_material
        return None


@dataclass
class HouseScene:
    """Complete house scene: layout + populated rooms.

    Always use HouseScene as the top-level container. Room mode is just a
    HouseScene with a single room (room_id="main"). This unified model avoids
    code duplication between modes.

    HouseScene contains the HouseLayout (floor plan data) and populated
    RoomScene instances.
    """

    layout: HouseLayout
    """House layout containing room specs, geometry, and doors/windows."""

    rooms: dict[str, "RoomScene"] = field(default_factory=dict)
    """Dictionary mapping room_id to RoomScene instances."""

    @property
    def house_dir(self) -> Path:
        """Base directory for the house (from layout)."""
        if self.layout.house_dir is None:
            raise ValueError("HouseLayout.house_dir is not set")
        return self.layout.house_dir

    def _get_room_position(self, room_id: str) -> tuple[float, float]:
        """Get room center position from layout.

        Room geometry is centered at origin, so we need the center position
        (not corner) when placing rooms in the combined directive.

        Args:
            room_id: Room ID to look up.

        Returns:
            (x, y) center position tuple. Returns (0, 0) if room not found.
        """
        for placed in self.layout.placed_rooms:
            if placed.room_id == room_id:
                # Convert from corner to center position.
                center_x = placed.position[0] + placed.width / 2
                center_y = placed.position[1] + placed.depth / 2
                return (center_x, center_y)
        # Default to origin for single room mode or if placement not done.
        return (0.0, 0.0)

    def add_room(self, room: "RoomScene") -> None:
        """Add a room to the house.

        Args:
            room: RoomScene to add. room.room_id must be unique within this house.

        Raises:
            ValueError: If a room with the same room_id already exists.
        """
        if room.room_id in self.rooms:
            raise ValueError(f"Room with id '{room.room_id}' already exists")
        self.rooms[room.room_id] = room

    def get_room(self, room_id: str) -> "RoomScene | None":
        """Get a room by ID.

        Args:
            room_id: The room ID to look up.

        Returns:
            RoomScene if found, None otherwise.
        """
        return self.rooms.get(room_id)

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize HouseScene to dictionary for checkpointing.

        Returns:
            Dictionary containing complete house state including layout.
        """
        rooms_dict = {}
        for room_id, room in self.rooms.items():
            rooms_dict[room_id] = room.to_state_dict()

        return {
            "layout": self.layout.to_dict(scene_dir=self.house_dir),
            "rooms": rooms_dict,
        }

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, Any], house_dir: Path
    ) -> "HouseScene":
        """Create HouseScene from serialized dictionary.

        Args:
            state_dict: State dictionary from to_state_dict().
            house_dir: Base directory for the house (needed for path resolution).

        Returns:
            Restored HouseScene instance.
        """
        # Import here to avoid circular import.
        from scenecode.agent_utils.room import RoomScene

        # Restore layout.
        layout = HouseLayout.from_dict(state_dict["layout"], house_dir=house_dir)

        # Create HouseScene with restored layout.
        house_scene = cls(layout=layout)

        # Restore rooms.
        for room_id, room_data in state_dict["rooms"].items():
            room_dir = house_dir / f"room_{room_id}"
            room = RoomScene(
                room_geometry=None,  # Will be restored.
                scene_dir=room_dir,
                room_id=room_id,
            )
            room.restore_from_state_dict(room_data)
            house_scene.rooms[room_id] = room

        return house_scene

    def assemble(
        self,
        cfg: dict | DictConfig | None = None,
        output_name: str = "combined_house",
        include_object_types: "list[ObjectType] | None" = None,
    ) -> Path:
        """Assemble all rooms into combined house outputs.

        Creates the output directory with:
        - house.dmd.yaml: Drake directive with furniture as free bodies
          (only wall/ceiling-mounted objects welded)
        - house_furniture_welded.dmd.yaml: Drake directive with furniture welded
        - house_state.json: Combined state for all rooms
        - sceneeval_state.json: Combined SceneEval format
        - house.blend: Blender file for visualization (uses house.dmd.yaml)

        Single room is treated as a house with one room at identity transform.

        Note: Composite manipulands (stacks, piles) are always free bodies in
        both output files. This is only for final output - internal simulation
        still uses welded furniture and composites for physics.

        Args:
            cfg: Configuration (dict or OmegaConf). Required for blend export.
                If None, blend file will not be generated.
            output_name: Name of output directory (default: "combined_house").
                Use "combined_house_after_furniture" for intermediate saves.
            include_object_types: If provided, only include objects of these
                types in the output. Useful for intermediate snapshots.

        Returns:
            Path to the output directory.
        """
        combined_dir = self.house_dir / output_name
        combined_dir.mkdir(parents=True, exist_ok=True)

        # Generate house.dmd.yaml: furniture as free bodies, composites as free bodies.
        directive_free = self._generate_combined_directive(
            include_object_types=include_object_types,
            weld_furniture=False,
            weld_composite_members=False,
        )
        directive_path_free = combined_dir / "house.dmd.yaml"
        with open(directive_path_free, "w") as f:
            f.write(directive_free)
        console_logger.info(
            f"Saved Drake directive (furniture free): {directive_path_free}"
        )

        # Generate house_furniture_welded.dmd.yaml: furniture welded, composites free.
        directive_welded = self._generate_combined_directive(
            include_object_types=include_object_types,
            weld_furniture=True,
            weld_composite_members=False,
        )
        directive_path_welded = combined_dir / "house_furniture_welded.dmd.yaml"
        with open(directive_path_welded, "w") as f:
            f.write(directive_welded)
        console_logger.info(
            f"Saved Drake directive (furniture welded): {directive_path_welded}"
        )

        # Create package.xml for portability (only once per scene).
        package_xml_path = self.house_dir / "package.xml"
        if not package_xml_path.exists():
            create_package_xml(self.house_dir)
            console_logger.info(f"Created package.xml for scene portability")

        # Save combined house state.
        state_dict = self.to_state_dict()
        state_dict["timestamp"] = time.time()
        state_path = combined_dir / "house_state.json"
        with open(state_path, "w") as f:
            json.dump(state_dict, f, indent=2)
        console_logger.info(f"Saved combined house state: {state_path}")

        # Export combined SceneEval format.
        floor_thickness = cfg["floor_plan_agent"]["floor_thickness"] if cfg else 0.1
        config = SceneEvalExportConfig(floor_thickness=floor_thickness)
        SceneEvalExporter.export_house(
            house=self, output_dir=combined_dir, config=config
        )

        # Generate combined blend file.
        if cfg is not None:
            self._export_blend(output_dir=combined_dir, cfg=cfg)

        return combined_dir

    def _generate_combined_directive(
        self,
        include_object_types: "list[ObjectType] | None" = None,
        weld_furniture: bool = True,
        weld_composite_members: bool = True,
    ) -> str:
        """Generate Drake directive combining all rooms.

        Single room is just a house with one room at identity transform.
        Multi-room uses frames to position each room at its layout position.

        Args:
            include_object_types: If provided, only include objects of these
                types. Useful for intermediate snapshots.
            weld_furniture: If True (default), weld furniture to world frame.
                If False, furniture is added as free bodies.
            weld_composite_members: If True (default), weld composite manipuland
                members (stacks, piles) to their base. If False, all members
                are free bodies.

        Returns:
            Drake directive YAML string with package://scene/ URIs for portability.
        """
        directive = """directives:
- add_frame:
    name: house_frame
    X_PF:
      base_frame: world
      translation: [0, 0, 0]"""

        for room_id, room in self.rooms.items():
            geometry_name = f"room_geometry_{room_id}"
            room_frame_name = f"room_{room_id}_frame"

            # Get room position from layout.
            pos_x, pos_y = self._get_room_position(room_id)

            # Add room frame as child of house_frame.
            directive += f"""
- add_frame:
    name: {room_frame_name}
    X_PF:
      base_frame: house_frame
      translation: [{pos_x}, {pos_y}, 0]"""

            # Get room directive with parent_frame so all objects use
            # room-local coordinates relative to the room frame.
            room_directive = room.to_drake_directive(
                weld_room_geometry=False,
                room_geometry_name=geometry_name,
                model_name_prefix=f"{room_id}_",
                include_object_types=include_object_types,
                base_dir=self.house_dir,
                weld_furniture=weld_furniture,
                weld_stack_members=weld_composite_members,
                parent_frame=room_frame_name,
            )

            # Strip the "directives:" header.
            if room_directive.startswith("directives:"):
                room_directive = room_directive[len("directives:") :]
            directive += room_directive

            # Weld room geometry to room frame (no translation needed,
            # room geometry is centered at origin).
            directive += f"""
- add_weld:
    parent: {room_frame_name}
    child: {geometry_name}::room_geometry_body_link"""

        return directive

    def _export_blend(self, output_dir: Path, cfg: dict | DictConfig) -> None:
        """Export Blender file for all rooms to combined directory.

        Uses the combined directive for both single and multi-room cases.
        Single room is just a house with one room at identity transform.

        Args:
            output_dir: Directory to save house.blend.
            cfg: Configuration with rendering settings (dict or OmegaConf).
        """
        from scenecode.agent_utils.rendering import save_directive_as_blend

        directive_path = output_dir / "house.dmd.yaml"
        if not directive_path.exists():
            console_logger.error("Combined directive not found, skipping house.blend")
            return

        blend_output_path = output_dir / "house.blend"
        rendering_cfg = cfg["furniture_agent"]["rendering"]

        try:
            save_directive_as_blend(
                directive_path=directive_path,
                output_path=blend_output_path,
                blender_server_host=rendering_cfg["blender_server_host"],
                blender_server_port_range=tuple(
                    rendering_cfg["blender_server_port_range"]
                ),
                server_startup_delay=rendering_cfg["server_startup_delay"],
                port_cleanup_delay=rendering_cfg["port_cleanup_delay"],
                scene_dir=self.house_dir,
            )
            console_logger.info(f"Saved combined blend file: {blend_output_path}")
        except Exception as e:
            console_logger.error(f"Failed to export combined .blend file: {e}")
