"""SceneEval format exporter for scenecode scenes."""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scenecode.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    deserialize_composite_member_effective_transform,
)

if TYPE_CHECKING:
    from scenecode.agent_utils.house import HouseLayout, HouseScene, Opening, Wall

console_logger = logging.getLogger(__name__)

SCENEEVAL_VERSION = "scene@1.0.2"
ARCH_VERSION = "arch@1.0.2"


@dataclass
class SceneEvalExportConfig:
    """Configuration for SceneEval export."""

    asset_id_prefix: str = "scenecode"
    floor_thickness: float = 0.1  # Default 10cm, can be overridden from config.
    wall_thickness: float = 0.05  # Default 5cm, can be overridden from config.


class SceneEvalExporter:
    """Exports Scene to SceneEval SceneState JSON format.

    The exporter produces a JSON file compatible with SceneEval's scene loading
    pipeline. Key features:

    - Architecture (walls/floor) exported as 2D line segments with heights
    - Objects exported with full 4x4 transform matrices (column-major)
    - SDF paths included for Drake physics pass-through
    - Coordinate system: Z-up, Y-forward (matches both scenecode and SceneEval)
    """

    def __init__(
        self,
        scene: RoomScene,
        scene_dir: Path,
        config: SceneEvalExportConfig,
        house_layout: HouseLayout | None = None,
    ) -> None:
        """Initialize the exporter.

        Args:
            scene: RoomScene object to export.
            scene_dir: Base directory for the scene (paths are relative to this).
            config: Export configuration options.
            house_layout: Optional HouseLayout for door/window export.
        """
        self.scene = scene
        self.scene_dir = Path(scene_dir)
        self.config = config
        self.house_layout = house_layout

    def export(self) -> Path:
        """Export scene to SceneEval format.

        Returns:
            Path to the exported sceneeval_state.json file.
        """
        scene_state = self._build_scene_state()

        output_path = (
            self.scene_dir / "scene_states" / "final_scene" / "sceneeval_state.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(scene_state, f, indent=2)

        console_logger.info(f"Exported SceneEval format to: {output_path}")
        return output_path

    def _build_scene_state(self) -> dict:
        """Build complete SceneState dictionary."""
        return {
            "format": "sceneState",
            "scene": {
                "version": SCENEEVAL_VERSION,
                "id": f"scenecode-{self.scene_dir.name}",
                "unit": 1.0,
                "up": [0, 0, 1],
                "front": [0, 1, 0],
                "assetSource": [self.config.asset_id_prefix],
                "arch": self._build_architecture(),
                "object": self._build_objects(),
            },
        }

    def _transform_to_matrix(
        self, translation: list, rotation_wxyz: list
    ) -> list[float]:
        """Convert quaternion + translation to 4x4 column-major matrix.

        Args:
            translation: 3D translation vector [x, y, z].
            rotation_wxyz: Quaternion in wxyz order [w, x, y, z].

        Returns:
            Flattened 4x4 matrix in column-major order (16 floats).
        """
        w, x, y, z = rotation_wxyz
        # Rotation matrix from quaternion.
        rot = np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )
        mat4 = np.eye(4)
        mat4[:3, :3] = rot
        mat4[:3, 3] = translation
        # Column-major order for SceneEval.
        return mat4.T.flatten().tolist()

    def _opening_to_hole(self, opening: Opening, wall_height: float) -> dict | None:
        """Convert an Opening to SceneEval hole format.

        SceneEval holes use 2D coordinates relative to wall:
        - X: position along wall (meters from wall start)
        - Z: vertical position (meters from floor)

        Args:
            opening: Opening dataclass with position, width, height, sill_height.
            wall_height: Height of the wall (for OPEN type openings).

        Returns:
            Hole dictionary for SceneEval, or None for OPEN type.
        """
        from scenecode.agent_utils.house import OpeningType

        # Skip OPEN connections - they're handled separately as open_room_pairs.
        if opening.opening_type == OpeningType.OPEN:
            return None

        # All opening types use LEFT EDGE convention for position_along_wall.
        x_min = opening.position_along_wall
        x_max = opening.position_along_wall + opening.width

        z_min = opening.sill_height
        z_max = opening.sill_height + opening.height

        return {
            "id": opening.opening_id,
            "type": "Door" if opening.opening_type == OpeningType.DOOR else "Window",
            "box": {
                "min": [float(x_min), float(z_min)],
                "max": [float(x_max), float(z_max)],
            },
        }

    def _get_wall_openings(self, wall_name: str) -> list[dict]:
        """Get holes for a wall from HouseLayout.

        Args:
            wall_name: Wall name (e.g., 'north_wall', 'left_wall', 'south_wall').

        Returns:
            List of hole dictionaries for the wall.
        """
        if not self.house_layout or not self.house_layout.placed_rooms:
            return []

        holes = []
        wall_height = self.house_layout.wall_height

        # Map scene wall names to cardinal directions.
        # Single-room exports use left/right/front/back, multi-room uses NSEW.
        wall_direction_map = {
            "left": "west",
            "right": "east",
            "front": "north",
            "back": "south",
            "west": "west",
            "east": "east",
            "north": "north",
            "south": "south",
        }

        # Extract direction from wall name (e.g., "left_wall" -> "left").
        scene_wall_dir = wall_name.replace("_wall", "")
        target_dir = wall_direction_map.get(scene_wall_dir, scene_wall_dir)

        # Find the PlacedRoom and Wall matching this wall direction.
        for placed_room in self.house_layout.placed_rooms:
            for wall in placed_room.walls:
                # Match wall by wall_id suffix (e.g., 'living_room_north' -> 'north').
                wall_dir = wall.wall_id.split("_")[-1]
                if wall_dir == target_dir:
                    for opening in wall.openings:
                        hole = self._opening_to_hole(opening, wall_height)
                        if hole:
                            holes.append(hole)
                    break

        return holes

    def _get_open_room_pairs(self) -> list[list[str]]:
        """Get open room pairs from HouseLayout.

        Returns:
            List of room ID pairs that have open connections.
        """
        # Local import to avoid circular import.
        from scenecode.agent_utils.house import ConnectionType

        if not self.house_layout:
            return []

        open_pairs = []
        for spec in self.house_layout.room_specs:
            for other_room, conn_type in spec.connections.items():
                if conn_type != ConnectionType.OPEN:
                    continue
                # Avoid duplicates by sorting room IDs.
                pair = sorted([spec.room_id, other_room])
                if pair not in open_pairs:
                    open_pairs.append(pair)

        return open_pairs

    def _build_architecture(self) -> dict:
        """Convert room_geometry and walls to SceneEval architecture format.

        Returns:
            Architecture dictionary with floor and wall elements.
        """
        elements = []
        wall_ids = []

        # Floor element derived from wall positions.
        # The floor polygon represents the walkable area (interior of room).
        # We compute this from wall positions, offsetting inward by wall thickness.
        floor_points = self._compute_floor_polygon_from_walls()
        floor_depth = 0.1  # Standard floor thickness.
        elements.append(
            {
                "id": "floor|room",
                "roomId": "room",
                "type": "Floor",
                "depth": floor_depth,
                "points": floor_points,
            }
        )

        # Wall elements from RoomGeometry.walls (not Scene.objects).
        for wall_index, wall in enumerate(self.scene.room_geometry.walls):
            wall_elem = self._wall_to_element(wall, wall_index)
            elements.append(wall_elem)
            wall_ids.append(wall_index + 1)  # 0 is floor.

        arch = {
            "version": ARCH_VERSION,
            "id": "arch",
            "up": [0, 0, 1],
            "front": [0, 1, 0],
            "coords2d": [0, 1],
            "scaleToMeters": 1,
            "elements": elements,
            "regions": [{"id": "room", "type": "Other", "walls": wall_ids}],
            "holes": [],
        }

        # Add open_room_pairs if HouseLayout is available.
        open_pairs = self._get_open_room_pairs()
        if open_pairs:
            arch["open_room_pairs"] = open_pairs

        return arch

    def _wall_to_element(self, wall: SceneObject, index: int) -> dict:
        """Convert wall SceneObject to 2D line segment with height.

        The wall segment represents the room-facing surface, offset inward
        from the wall center by half the wall thickness. Uses pre-computed
        wall normals from RoomGeometry to determine inward direction.

        Args:
            wall: Wall SceneObject with transform and bounding box.
            index: Wall index for ID generation.

        Returns:
            Wall element dictionary for SceneEval architecture.
        """
        center = np.array(wall.transform.translation())
        bbox_min = np.array(wall.bbox_min)
        bbox_max = np.array(wall.bbox_max)
        extents = bbox_max - bbox_min

        # Get room-facing normal from room geometry (points into room).
        normal = self.scene.room_geometry.wall_normals.get(
            wall.name, np.array([0.0, 0.0])
        )

        # Determine wall orientation (thin dimension is thickness).
        # Offset wall center by half thickness in normal direction to get
        # room-facing surface.
        if extents[0] < extents[1]:
            # Y-aligned wall (thin in X).
            half_thickness = extents[0] / 2
            wall_x = center[0] + normal[0] * half_thickness
            p1 = [float(wall_x), float(center[1] + bbox_min[1]), 0.0]
            p2 = [float(wall_x), float(center[1] + bbox_max[1]), 0.0]
        else:
            # X-aligned wall (thin in Y).
            half_thickness = extents[1] / 2
            wall_y = center[1] + normal[1] * half_thickness
            p1 = [float(center[0] + bbox_min[0]), float(wall_y), 0.0]
            p2 = [float(center[0] + bbox_max[0]), float(wall_y), 0.0]

        height = float(extents[2])

        # Get holes (doors/windows) for this wall from HouseLayout.
        holes = self._get_wall_openings(wall.name)

        return {
            "id": f"wall|room|{wall.name}|{index}",
            "roomId": "room",
            "type": "Wall",
            "height": height,
            "depth": float(min(extents[0], extents[1])),  # Wall thickness.
            "points": [p1, p2],
            "holes": holes,
        }

    def _compute_floor_polygon_from_walls(self) -> list[list[float]]:
        """Compute floor polygon from wall inner surfaces.

        For a rectangular room, finds the 4 corners defined by wall intersections.
        Each wall contributes its room-facing surface position.

        Returns:
            List of 4 corner points [[x, y, z], ...] in counter-clockwise order.
        """
        # For rectangular rooms, we need to find the inner bounds.
        # Collect wall inner surface positions by orientation.
        min_x = float("inf")  # Right edge of left wall.
        max_x = float("-inf")  # Left edge of right wall.
        min_y = float("inf")  # Front edge of back wall.
        max_y = float("-inf")  # Back edge of front wall.

        for wall in self.scene.room_geometry.walls:
            center = np.array(wall.transform.translation())
            extents = np.array(wall.bbox_max) - np.array(wall.bbox_min)
            normal = self.scene.room_geometry.wall_normals.get(
                wall.name, np.array([0.0, 0.0])
            )

            # Determine if wall is X-aligned or Y-aligned.
            if extents[0] < extents[1]:
                # Y-aligned wall (thin in X, runs along Y).
                half_thickness = extents[0] / 2
                inner_x = center[0] + normal[0] * half_thickness
                if normal[0] > 0:  # Left wall, facing right.
                    min_x = min(min_x, inner_x)
                else:  # Right wall, facing left.
                    max_x = max(max_x, inner_x)
            else:
                # X-aligned wall (thin in Y, runs along X).
                half_thickness = extents[1] / 2
                inner_y = center[1] + normal[1] * half_thickness
                if normal[1] > 0:  # Back wall, facing forward.
                    min_y = min(min_y, inner_y)
                else:  # Front wall, facing backward.
                    max_y = max(max_y, inner_y)

        # Return counter-clockwise corners at z=0 (floor surface).
        return [
            [float(min_x), float(min_y), 0.0],
            [float(max_x), float(min_y), 0.0],
            [float(max_x), float(max_y), 0.0],
            [float(min_x), float(max_y), 0.0],
        ]

    def _build_member_object(self, member: dict, index: int) -> dict | None:
        """Build SceneEval object entry from a composite member asset.

        Args:
            member: Member asset dict with asset_id, name, transform, sdf_path.
            index: Object index for SceneEval.

        Returns:
            Object dictionary for SceneEval, or None if member lacks SDF.
        """
        sdf_path = member.get("sdf_path")
        if not sdf_path:
            return None

        # Extract transform from serialized format (fail-fast if missing).
        if "transform" not in member:
            raise ValueError(
                f"Composite member {index} missing 'transform' data. "
                f"Member: {member.get('name', 'unknown')}"
            )
        transform_data = member["transform"]
        if "translation" not in transform_data or "rotation_wxyz" not in transform_data:
            raise ValueError(
                f"Composite member {index} has incomplete transform data. "
                f"Member: {member.get('name', 'unknown')}, transform: {transform_data}"
            )
        transform = deserialize_composite_member_effective_transform(member)
        translation = [float(x) for x in transform.translation()]
        rotation_wxyz = [float(x) for x in transform.rotation().ToQuaternion().wxyz()]
        matrix = self._transform_to_matrix(translation, rotation_wxyz)

        # Build relative SDF path.
        sdf_path_obj = Path(sdf_path)
        if sdf_path_obj.is_absolute():
            try:
                sdf_path_str = str(sdf_path_obj.relative_to(self.scene_dir))
            except ValueError:
                sdf_path_str = str(sdf_path_obj)
        else:
            sdf_path_str = str(sdf_path_obj)

        # Use asset_id as the object ID.
        asset_id = member.get("asset_id", f"member_{index}")

        return {
            "index": index,
            "id": asset_id,
            "modelId": f"{self.config.asset_id_prefix}.{asset_id}",
            "parentId": "",
            "parentIndex": -1,
            "transform": {"data": matrix},
            "sdfPath": sdf_path_str,
        }

    def _build_objects(self) -> list[dict]:
        """Convert furniture/manipuland SceneObjects to object entries.

        Returns:
            List of object dictionaries for SceneEval scene state.
        """
        objects = []
        index = 0

        for obj in self.scene.objects.values():
            if obj.object_type in (ObjectType.WALL, ObjectType.FLOOR):
                continue

            # Handle composite objects by expanding member assets.
            composite_type = obj.metadata.get("composite_type")
            if composite_type == "stack":
                member_assets = obj.metadata.get("member_assets", [])
                for member in member_assets:
                    member_obj = self._build_member_object(member=member, index=index)
                    if member_obj:
                        objects.append(member_obj)
                        index += 1
                continue

            if composite_type == "filled_container":
                # Expand container_asset + fill_assets.
                container_asset = obj.metadata.get("container_asset")
                if container_asset:
                    container_obj = self._build_member_object(
                        member=container_asset, index=index
                    )
                    if container_obj:
                        objects.append(container_obj)
                        index += 1

                fill_assets = obj.metadata.get("fill_assets", [])
                for fill_asset in fill_assets:
                    fill_obj = self._build_member_object(member=fill_asset, index=index)
                    if fill_obj:
                        objects.append(fill_obj)
                        index += 1
                continue

            if composite_type == "pile":
                # Expand member_assets (same structure as stack).
                member_assets = obj.metadata.get("member_assets", [])
                for member in member_assets:
                    member_obj = self._build_member_object(member=member, index=index)
                    if member_obj:
                        objects.append(member_obj)
                        index += 1
                continue

            translation = list(obj.transform.translation())
            rotation_wxyz = list(obj.transform.rotation().ToQuaternion().wxyz())
            matrix = self._transform_to_matrix(translation, rotation_wxyz)

            # Get relative SDF path for Drake pass-through.
            sdf_path_str = ""
            if obj.sdf_path:
                sdf_path = Path(obj.sdf_path)
                if sdf_path.is_absolute():
                    try:
                        sdf_path_str = str(sdf_path.relative_to(self.scene_dir))
                    except ValueError:
                        # Path is not relative to scene_dir, use absolute.
                        sdf_path_str = str(sdf_path)
                else:
                    sdf_path_str = str(sdf_path)

            objects.append(
                {
                    "index": index,
                    "id": str(obj.object_id),
                    "modelId": f"{self.config.asset_id_prefix}.{obj.object_id}",
                    "parentId": "",
                    "parentIndex": -1,
                    "transform": {"data": matrix},
                    "sdfPath": sdf_path_str,
                }
            )
            index += 1

        return objects

    @classmethod
    def export_house(
        cls,
        house: "HouseScene",
        output_dir: Path,
        config: SceneEvalExportConfig,
    ) -> Path:
        """Export HouseScene to SceneEval format.

        Exports a complete SceneEval scene state including:
        - Architecture (floors, walls with doors/windows)
        - Objects (furniture, manipulands from all rooms)
        - Open room pairs (open floor plan connections)

        Args:
            house: HouseScene to export.
            output_dir: Directory to save sceneeval_state.json.
            config: Export configuration.

        Returns:
            Path to exported sceneeval_state.json.
        """
        # Build combined architecture and objects from all rooms.
        combined_arch = cls._build_house_architecture(house, config)
        combined_objects = cls._build_house_objects(house, config)

        scene_state = {
            "format": "sceneState",
            "scene": {
                "version": SCENEEVAL_VERSION,
                "id": f"scenecode-{output_dir.name}",
                "unit": 1.0,
                "up": [0, 0, 1],
                "front": [0, 1, 0],
                "assetSource": [config.asset_id_prefix],
                "arch": combined_arch,
                "object": combined_objects,
            },
        }

        output_path = output_dir / "sceneeval_state.json"
        with open(output_path, "w") as f:
            json.dump(scene_state, f, indent=2)
        console_logger.info(f"Saved combined SceneEval state: {output_path}")

        return output_path

    @classmethod
    def _build_house_architecture(
        cls,
        house: "HouseScene",
        config: SceneEvalExportConfig,
    ) -> dict:
        """Build combined architecture for all rooms in a house.

        Returns:
            Architecture dictionary with floors, walls (with doors/windows),
            regions, and open_room_pairs.
        """
        # Local import to avoid circular import.
        from scenecode.agent_utils.house import ConnectionType

        elements = []
        regions = []
        element_index = 0

        # Get open room pairs from layout.
        open_pairs = []
        for spec in house.layout.room_specs:
            for other_room, conn_type in spec.connections.items():
                if conn_type != ConnectionType.OPEN:
                    continue
                pair = sorted([spec.room_id, other_room])
                if pair not in open_pairs:
                    open_pairs.append(pair)

        # Process each room.
        for room_id in house.rooms:
            placed_room = house.layout.get_placed_room(room_id)
            if not placed_room:
                continue

            room_pos_x, room_pos_y = cls._get_house_room_corner_position(house, room_id)
            wall_indices = []

            # Floor element.
            floor_points = cls._get_room_floor_polygon(
                house, room_id, room_pos_x, room_pos_y
            )
            elements.append(
                {
                    "id": f"floor|{room_id}",
                    "roomId": room_id,
                    "type": "Floor",
                    "depth": config.floor_thickness,
                    "points": floor_points,
                }
            )
            element_index += 1

            # Wall elements with holes.
            for wall in placed_room.walls:
                wall_elem = cls._wall_to_house_element(
                    wall=wall,
                    room_id=room_id,
                    index=element_index,
                    room_offset=(room_pos_x, room_pos_y),
                    wall_height=house.layout.wall_height,
                    wall_thickness=config.wall_thickness,
                )
                elements.append(wall_elem)
                wall_indices.append(element_index)
                element_index += 1

            # Region for this room.
            regions.append(
                {
                    "id": room_id,
                    "type": "Other",
                    "walls": wall_indices,
                }
            )

        arch = {
            "version": ARCH_VERSION,
            "id": "arch",
            "up": [0, 0, 1],
            "front": [0, 1, 0],
            "coords2d": [0, 1],
            "scaleToMeters": 1,
            "elements": elements,
            "regions": regions,
            "holes": [],
        }

        if open_pairs:
            arch["open_room_pairs"] = open_pairs

        return arch

    @classmethod
    def _get_house_room_corner_position(
        cls, house: "HouseScene", room_id: str
    ) -> tuple[float, float]:
        """Get room corner position for multi-room house.

        Used for floor/wall polygon construction where we need the corner
        (min x, min y) as the starting point.

        Single room (room_id="main") is at origin.
        Multi-room uses PlacedRoom positions.
        """
        if len(house.rooms) == 1:
            return (0.0, 0.0)

        placed_room = house.layout.get_placed_room(room_id)
        if placed_room:
            return placed_room.position
        return (0.0, 0.0)

    @classmethod
    def _get_house_room_center_position(
        cls, house: "HouseScene", room_id: str
    ) -> tuple[float, float]:
        """Get room center position for house export.

        Used for object positioning where room geometry is centered at origin,
        so we need the center position to transform room-local coordinates
        to world coordinates (corner-based, matching architecture).

        Computes center from corner position + dimensions/2.
        """
        placed_room = house.layout.get_placed_room(room_id)
        if placed_room:
            # PlacedRoom.width = X dimension, PlacedRoom.depth = Y dimension.
            center_x = placed_room.position[0] + placed_room.width / 2
            center_y = placed_room.position[1] + placed_room.depth / 2
            return (center_x, center_y)
        return (0.0, 0.0)

    @classmethod
    def _get_room_floor_polygon(
        cls,
        house: "HouseScene",
        room_id: str,
        offset_x: float,
        offset_y: float,
    ) -> list[list[float]]:
        """Get floor polygon for a room in world coordinates.

        Args:
            house: HouseScene containing the room.
            room_id: Room ID.
            offset_x: X offset for room position.
            offset_y: Y offset for room position.

        Returns:
            List of 4 corner points in counter-clockwise order.
        """
        placed_room = house.layout.get_placed_room(room_id)
        if not placed_room:
            return []

        # Use PlacedRoom dimensions (accounts for rotation during placement).
        # PlacedRoom.width = X dimension, PlacedRoom.depth = Y dimension.
        min_x = offset_x
        max_x = offset_x + placed_room.width
        min_y = offset_y
        max_y = offset_y + placed_room.depth

        # Counter-clockwise order at z=0.
        return [
            [float(min_x), float(min_y), 0.0],
            [float(max_x), float(min_y), 0.0],
            [float(max_x), float(max_y), 0.0],
            [float(min_x), float(max_y), 0.0],
        ]

    @classmethod
    def _wall_to_house_element(
        cls,
        wall: "Wall",
        room_id: str,
        index: int,
        room_offset: tuple[float, float],
        wall_height: float,
        wall_thickness: float,
    ) -> dict:
        """Convert Wall to SceneEval element with holes.

        Args:
            wall: Wall dataclass with openings.
            room_id: Room ID this wall belongs to.
            index: Element index.
            room_offset: (x, y) offset for room position.
            wall_height: Wall height in meters.
            wall_thickness: Wall thickness in meters.

        Returns:
            Wall element dictionary for SceneEval.
        """
        from scenecode.agent_utils.house import OpeningType

        offset_x, offset_y = room_offset

        # Wall points in world coordinates.
        p1 = [
            float(wall.start_point[0] + offset_x),
            float(wall.start_point[1] + offset_y),
            0.0,
        ]
        p2 = [
            float(wall.end_point[0] + offset_x),
            float(wall.end_point[1] + offset_y),
            0.0,
        ]

        # Convert openings to holes.
        holes = []
        for opening in wall.openings:
            # Skip OPEN connections - they're in open_room_pairs.
            if opening.opening_type == OpeningType.OPEN:
                continue

            # All opening types use LEFT EDGE convention for position_along_wall.
            x_min = opening.position_along_wall
            x_max = opening.position_along_wall + opening.width
            z_min = opening.sill_height
            z_max = opening.sill_height + opening.height

            hole_type = "Door" if opening.opening_type == OpeningType.DOOR else "Window"
            holes.append(
                {
                    "id": opening.opening_id,
                    "type": hole_type,
                    "box": {
                        "min": [float(x_min), float(z_min)],
                        "max": [float(x_max), float(z_max)],
                    },
                }
            )

        return {
            "id": f"wall|{room_id}|{wall.direction.value}|{index}",
            "roomId": room_id,
            "type": "Wall",
            "height": float(wall_height),
            "depth": float(wall_thickness),
            "points": [p1, p2],
            "holes": holes,
        }

    @classmethod
    def _build_house_objects(
        cls,
        house: "HouseScene",
        config: SceneEvalExportConfig,
    ) -> list[dict]:
        """Build combined objects list from all rooms in a house.

        Args:
            house: HouseScene containing rooms.
            config: Export configuration.

        Returns:
            List of object dictionaries.
        """
        combined_objects = []
        object_index = 0

        for room_id, room in house.rooms.items():
            # Create exporter for this room to reuse _build_objects.
            exporter = cls(
                scene=room,
                scene_dir=room.scene_dir,
                config=config,
                house_layout=house.layout,
            )

            # Get room CENTER position offset for objects.
            # Room geometry is centered at origin, so objects need center offset.
            pos_x, pos_y = cls._get_house_room_center_position(house, room_id)

            # Build objects for this room.
            room_objects = exporter._build_objects()

            # Transform objects to world coordinates.
            for obj in room_objects:
                # Update index.
                obj["index"] = object_index
                object_index += 1

                # Transform position in matrix (column 12, 13 are x, y translation).
                if "transform" in obj and "data" in obj["transform"]:
                    matrix = obj["transform"]["data"]
                    # Column-major: indices 12, 13 are x, y translation.
                    matrix[12] += pos_x
                    matrix[13] += pos_y

                combined_objects.append(obj)

        return combined_objects
