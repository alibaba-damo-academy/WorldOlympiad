"""DMD Scene loader for success validation.

Loads scenes from scene_state.json (object metadata) and scene.dmd.yaml (Drake poses).
Provides access to object metadata and Drake physics queries.

Usage for robot evaluation:
    1. Robot receives initial scene.dmd.yaml
    2. Robot performs task and outputs modified scene.dmd.yaml
    3. Eval loads: scene_state.json (metadata) + modified dmd.yaml (new poses)
"""

import json
import logging

from dataclasses import dataclass
from pathlib import Path

from pydrake.all import (
    Context,
    Diagram,
    DiagramBuilder,
    MultibodyPlant,
    Quaternion,
    QueryObject,
    RigidTransform,
    SceneGraph,
)

from scenecode.agent_utils.drake_utils import create_plant_from_dmd
from scenecode.utils.sdf_utils import extract_base_link_name_from_sdf

console_logger = logging.getLogger(__name__)


@dataclass
class DMDScene:
    """Scene loaded from scene_state.json (metadata) and scene.dmd.yaml (poses).

    Provides access to:
    - Object metadata from scene_state.json (names, types, SDFs, bboxes)
    - Object poses from dmd.yaml (synced to scene_state after finalize)
    - Drake physics queries via finalized plant/scene_graph

    Usage:
        scene = load_scene_for_validation(
            scene_state_path=Path("scene_state.json"),
            dmd_path=Path("modified_scene.dmd.yaml"),
            task_description="Put apple on table",
        )
        scene.finalize()  # Build diagram, sync poses from Drake to scene_state
        query = scene.get_query_object()  # For physics queries
    """

    plant: MultibodyPlant
    """Drake MultibodyPlant with all objects loaded."""

    scene_graph: SceneGraph
    """Drake SceneGraph for geometry queries."""

    builder: DiagramBuilder
    """DiagramBuilder (used before finalization)."""

    scene_state: dict
    """Object metadata from scene_state.json. Poses updated from Drake after finalize()."""

    scene_dir: Path
    """Base directory for resolving package:// URIs."""

    dmd_path: Path | None = None
    """Path to the DMD file (for loading room geometry)."""

    task_description: str = ""
    """Task description for evaluation."""

    diagram: Diagram | None = None
    """Built diagram (after finalize())."""

    context: Context | None = None
    """Diagram context (after finalize())."""

    def finalize(self) -> None:
        """Build diagram, create context, and sync poses from Drake to scene_state.

        After finalize():
        - Physics queries are available via get_query_object()
        - scene_state["objects"][obj_id]["transform"] contains poses from Drake
        """
        if self.diagram is not None:
            console_logger.warning("Scene already finalized, skipping")
            return

        self.diagram = self.builder.Build()
        self.context = self.diagram.CreateDefaultContext()

        # Sync poses from Drake to scene_state.
        self._sync_poses_from_drake()
        console_logger.info("Scene finalized - poses synced from Drake to scene_state")

    def _sync_poses_from_drake(self) -> None:
        """Update scene_state with poses from Drake plant.

        For each object in scene_state, queries the corresponding model's pose
        from Drake and updates the transform in scene_state. This ensures
        scene_state reflects the actual poses from the dmd.yaml file.

        Uses the same base link detection as the main pipeline to correctly handle
        articulated objects.
        """
        if self.diagram is None or self.context is None:
            raise RuntimeError("Cannot sync poses before diagram is built")

        plant_context = self.plant.GetMyContextFromRoot(self.context)
        synced_count = 0

        for obj_id, obj_data in list(self.scene_state.get("objects", {}).items()):
            try:
                # Get model instance by object_id (model names match object_ids).
                model_instance = self.plant.GetModelInstanceByName(obj_id)

                # Get base link name from SDF (same approach as main pipeline).
                body = self._get_base_body(obj_id, obj_data, model_instance)
                if body is None:
                    continue

                pose = self.plant.EvalBodyPoseInWorld(plant_context, body)

                # Convert to scene_state format.
                translation = pose.translation().tolist()
                quat = pose.rotation().ToQuaternion()
                rotation_wxyz = [quat.w(), quat.x(), quat.y(), quat.z()]

                # Update scene_state.
                self.scene_state["objects"][obj_id]["transform"] = {
                    "translation": translation,
                    "rotation_wxyz": rotation_wxyz,
                }
                synced_count += 1

            except RuntimeError as e:
                # Model not found in Drake - may be room geometry or other non-object.
                console_logger.debug(f"Could not sync pose for {obj_id}: {e}")

        console_logger.info(f"Synced {synced_count} object poses from Drake")

    def _get_base_body(self, obj_id: str, obj_data: dict, model_instance):
        """Get the base body for a model, using SDF parsing like the main pipeline.

        Args:
            obj_id: Object identifier.
            obj_data: Object data from scene_state.
            model_instance: Drake model instance.

        Returns:
            Drake Body for the base link, or None if not found.
        """
        # Try to get base link name from SDF (same as main pipeline).
        sdf_path = obj_data.get("sdf_path")
        if sdf_path:
            try:
                base_link_name = extract_base_link_name_from_sdf(Path(sdf_path))
                return self.plant.GetBodyByName(
                    name=base_link_name, model_instance=model_instance
                )
            except (ValueError, RuntimeError) as e:
                console_logger.debug(
                    f"Could not get base link from SDF for {obj_id}: {e}"
                )

        # Fallback: use first body.
        body_indices = self.plant.GetBodyIndices(model_instance)
        if not body_indices:
            console_logger.debug(f"No bodies for model {obj_id}, skipping")
            return None

        return self.plant.get_body(body_indices[0])

    def get_query_object(self) -> QueryObject:
        """Get Drake QueryObject for geometry queries.

        Returns:
            QueryObject for signed distance and other geometry queries.

        Raises:
            RuntimeError: If scene has not been finalized.
        """
        if self.diagram is None or self.context is None:
            raise RuntimeError("Scene not finalized. Call finalize() first.")

        scene_graph_context = self.scene_graph.GetMyContextFromRoot(self.context)
        return self.scene_graph.get_query_output_port().Eval(scene_graph_context)

    def get_object_pose(self, obj_id: str) -> RigidTransform:
        """Get object pose from scene_state.

        After finalize(), this returns the pose from the dmd.yaml file
        (synced to scene_state by _sync_poses_from_drake).

        Args:
            obj_id: Object identifier.

        Returns:
            RigidTransform representing object pose in world frame.

        Raises:
            KeyError: If object not found in scene_state.
        """
        obj_data = self.scene_state["objects"][obj_id]
        trans = obj_data["transform"]["translation"]
        rot_wxyz = obj_data["transform"]["rotation_wxyz"]
        return RigidTransform(
            Quaternion(rot_wxyz[0], rot_wxyz[1], rot_wxyz[2], rot_wxyz[3]), trans
        )

    def get_sdf_path(self, obj_id: str) -> Path:
        """Get SDF path for an object.

        Args:
            obj_id: Object identifier.

        Returns:
            Path to the object's SDF file.

        Raises:
            KeyError: If object not found or has no SDF path.
        """
        sdf_path = self.scene_state["objects"][obj_id].get("sdf_path")
        if sdf_path is None:
            raise KeyError(f"Object {obj_id} has no SDF path")
        return Path(sdf_path)

    def get_object_bbox(self, obj_id: str) -> tuple[list[float], list[float]]:
        """Get object bounding box from scene_state.json.

        The bounding box is in object-local frame (not world frame).
        To get world-frame bounds, add the object's translation.

        Args:
            obj_id: Object identifier.

        Returns:
            Tuple of (bbox_min, bbox_max) as [x, y, z] lists in object frame.

        Raises:
            KeyError: If object not found or has no bbox.
        """
        obj_data = self.scene_state["objects"][obj_id]
        bbox_min = obj_data.get("bbox_min")
        bbox_max = obj_data.get("bbox_max")
        if bbox_min is None or bbox_max is None:
            raise KeyError(f"Object {obj_id} has no bounding box data")
        return bbox_min, bbox_max

    def get_object_dimensions(self, obj_id: str) -> list[float]:
        """Get object dimensions (width, depth, height) from bounding box.

        Args:
            obj_id: Object identifier.

        Returns:
            Dimensions as [width, depth, height] in meters.
        """
        bbox_min, bbox_max = self.get_object_bbox(obj_id)
        return [
            bbox_max[0] - bbox_min[0],  # width (x)
            bbox_max[1] - bbox_min[1],  # depth (y)
            bbox_max[2] - bbox_min[2],  # height (z)
        ]

    def get_all_object_ids(self) -> list[str]:
        """Get all object IDs in the scene.

        Returns:
            List of all object IDs.
        """
        return list(self.scene_state["objects"].keys())

    def get_object_info(self, obj_id: str) -> dict:
        """Get full object info from scene_state.

        Args:
            obj_id: Object identifier.

        Returns:
            Object data dictionary from scene_state.json.

        Raises:
            KeyError: If object not found.
        """
        return self.scene_state["objects"][obj_id]


def _expand_composite_members(obj_data: dict) -> dict[str, dict]:
    """Expand a composite object into its individual member objects.

    Composites (stack, pile, filled_container) contain multiple physical objects
    that are grouped together. This function extracts each member so they can be
    individually matched by the category matcher.

    Args:
        obj_data: Object data dict from scene_state containing metadata.

    Returns:
        Dict mapping drake_model_name -> object data for each member.
        Empty dict if not a composite or no members found.
    """
    metadata = obj_data.get("metadata", {})
    composite_type = metadata.get("composite_type")
    # member_model_names can be at top level or inside metadata.
    member_names = obj_data.get("member_model_names") or metadata.get(
        "member_model_names", []
    )

    if not composite_type or not member_names:
        return {}

    members = {}

    if composite_type in ("stack", "pile"):
        # member_assets[i] corresponds to member_model_names[i].
        for i, asset in enumerate(metadata.get("member_assets", [])):
            if i < len(member_names):
                name = asset.get("name", "")
                members[member_names[i]] = {
                    "object_id": asset.get("asset_id"),
                    "name": name,
                    "description": name.replace("_", " "),
                    "sdf_path": asset.get("sdf_path"),
                    "transform": asset.get("transform"),
                    "object_type": "manipuland",
                }

    elif composite_type == "filled_container":
        # Container model ends with "_c".
        container = metadata.get("container_asset", {})
        container_model = next((n for n in member_names if n.endswith("_c")), None)
        if container and container_model:
            name = container.get("name", "")
            members[container_model] = {
                "object_id": container.get("asset_id"),
                "name": name,
                "description": name.replace("_", " "),
                "sdf_path": container.get("sdf_path"),
                "transform": container.get("transform"),
                "object_type": "manipuland",
            }

        # Fill assets - match by name substring in model name.
        # Track matched names to avoid duplicates (multiple fill_assets may have same name).
        matched_models: set[str] = set()
        for asset in metadata.get("fill_assets", []):
            asset_name = asset.get("name", "")
            fill_model = next(
                (
                    n
                    for n in member_names
                    if asset_name in n
                    and not n.endswith("_c")
                    and n not in matched_models
                ),
                None,
            )
            if fill_model:
                matched_models.add(fill_model)
                members[fill_model] = {
                    "object_id": asset.get("asset_id"),
                    "name": asset_name,
                    "description": asset_name.replace("_", " "),
                    "sdf_path": asset.get("sdf_path"),
                    "transform": asset.get("transform"),
                    "object_type": "manipuland",
                }

    return members


def _normalize_scene_state(raw_state: dict) -> dict:
    """Normalize scene state to have top-level 'objects' dict.

    Handles two formats:
    1. Per-room scene_state.json: {"objects": {...}, "room_geometry": {...}}
    2. Combined house_state.json: {"rooms": {"bedroom": {"objects": {...}}, ...}}

    For combined house format, merges objects from all rooms with room_id prefix
    to match Drake model names (e.g., "bedroom_table_0" instead of "table_0").

    Also extracts wall data from room_geometry for wall hiding in renders.

    Args:
        raw_state: Raw scene state dictionary.

    Returns:
        Normalized state with top-level "objects" dict and "_walls" list.
    """
    # Already has top-level objects - per-room format.
    if "objects" in raw_state:
        # Extract walls from room_geometry for wall hiding.
        walls = []
        room_geometry = raw_state.get("room_geometry", {})
        walls.extend(room_geometry.get("walls", []))
        return {**raw_state, "_walls": walls}

    # Combined house format - merge objects from all rooms.
    if "rooms" in raw_state:
        merged_objects = {}
        all_walls = []

        # Build room position lookup from layout for coordinate transformation.
        room_positions = {}
        layout = raw_state.get("layout", {})
        for placed_room in layout.get("placed_rooms", []):
            room_positions[placed_room["room_id"]] = placed_room["position"]

        for room_id, room_data in raw_state["rooms"].items():
            room_objects = room_data.get("objects", {})
            for obj_id, obj_data in room_objects.items():
                # Check if composite - expand to individual members instead.
                metadata = obj_data.get("metadata", {})
                if metadata.get("composite_type") in (
                    "stack",
                    "pile",
                    "filled_container",
                ):
                    expanded = _expand_composite_members(obj_data)
                    merged_objects.update(expanded)
                else:
                    # Prefix with room_id to match Drake model names in combined DMD.
                    prefixed_id = f"{room_id}_{obj_id}"
                    # Copy obj_data and fix sdf_path to be relative to scene root.
                    # Original sdf_path is relative to room dir (e.g., generated_assets/...)
                    # but scene root has room_xxx/generated_assets/... structure.
                    obj_data_copy = dict(obj_data)
                    if obj_data_copy.get("sdf_path"):
                        sdf_path = obj_data_copy["sdf_path"]
                        # Only prefix if not already absolute or package URI.
                        if not sdf_path.startswith(("/", "package://", "file://")):
                            obj_data_copy["sdf_path"] = f"room_{room_id}/{sdf_path}"
                    merged_objects[prefixed_id] = obj_data_copy

            # Extract walls from room_geometry for wall hiding.
            room_geometry = room_data.get("room_geometry", {})
            for wall in room_geometry.get("walls", []):
                wall_copy = dict(wall)
                # Prefix wall IDs to match room context.
                wall_copy["object_id"] = f"{room_id}_{wall.get('object_id', '')}"
                all_walls.append(wall_copy)

            # Add floor object from room_geometry if present.
            # Transform bbox from room-local (centered) to world coordinates (corner origin).
            floor_data = room_geometry.get("floor")
            if floor_data:
                floor_id = f"{room_id}_floor"
                local_bbox_min = floor_data.get("bbox_min", [-5, -5, -0.1])
                local_bbox_max = floor_data.get("bbox_max", [5, 5, 0.0])

                # Room geometry uses centered coordinates, Drake uses corner-origin.
                # Transform: world = local + (length/2, width/2).
                room_length = room_geometry.get("length", 6.0)
                room_width = room_geometry.get("width", 4.5)
                offset_x = room_length / 2
                offset_y = room_width / 2

                world_bbox_min = [
                    local_bbox_min[0] + offset_x,
                    local_bbox_min[1] + offset_y,
                    local_bbox_min[2],
                ]
                world_bbox_max = [
                    local_bbox_max[0] + offset_x,
                    local_bbox_max[1] + offset_y,
                    local_bbox_max[2],
                ]

                merged_objects[floor_id] = {
                    "object_id": floor_data.get("object_id", floor_id),
                    "object_type": "floor",
                    "name": f"{room_id.capitalize()} Floor",
                    "description": f"Floor surface of {room_id}",
                    "transform": {
                        "translation": [offset_x, offset_y, 0],
                        "rotation_wxyz": [1, 0, 0, 0],
                    },
                    "bbox_min": world_bbox_min,
                    "bbox_max": world_bbox_max,
                    "immutable": True,
                }

        return {
            "objects": merged_objects,
            "_walls": all_walls,
            "_source": "combined_house",
        }

    # Unknown format - return as-is with empty objects.
    console_logger.warning("Unknown scene_state format - no objects found")
    return {"objects": {}, "_walls": []}


def load_scene_for_validation(
    scene_state_path: Path,
    dmd_path: Path,
    task_description: str = "",
    scene_dir: Path | None = None,
) -> DMDScene:
    """Load scene from scene_state.json (metadata) and dmd.yaml (poses).

    This supports the robot evaluation workflow:
    1. Robot receives initial scene.dmd.yaml
    2. Robot performs task and outputs modified scene.dmd.yaml
    3. Eval loads: original scene_state.json + modified dmd.yaml

    After calling finalize(), scene_state will contain poses from the dmd.yaml.

    Supports two input formats:
    - Per-room: scene_state.json with top-level "objects" dict
    - Combined house: house_state.json with nested "rooms.<room_id>.objects"

    Args:
        scene_state_path: Path to scene_state.json or house_state.json.
        dmd_path: Path to scene.dmd.yaml (Drake directives with poses).
        task_description: Task description for evaluation.
        scene_dir: Base directory for resolving package:// URIs in the DMD.
            If None, uses the parent directory of dmd_path.

    Returns:
        DMDScene ready for validation (call finalize() before physics queries).

    Raises:
        FileNotFoundError: If required files are missing.
    """
    # Validate required files exist.
    if not scene_state_path.exists():
        raise FileNotFoundError(f"Scene state file not found: {scene_state_path}")
    if not dmd_path.exists():
        raise FileNotFoundError(f"DMD file not found: {dmd_path}")

    # Default scene_dir to parent of dmd_path.
    if scene_dir is None:
        scene_dir = dmd_path.parent

    # Load and normalize scene_state.
    with open(scene_state_path) as f:
        raw_state = json.load(f)
    scene_state = _normalize_scene_state(raw_state)
    console_logger.info(
        f"Loaded scene state with {len(scene_state.get('objects', {}))} objects"
    )

    # Load Drake plant from DMD with package:// URI resolution.
    builder, plant, scene_graph = create_plant_from_dmd(
        directive_path=dmd_path, scene_dir=scene_dir
    )
    console_logger.info(f"Loaded Drake plant from {dmd_path}")

    return DMDScene(
        plant=plant,
        scene_graph=scene_graph,
        builder=builder,
        scene_state=scene_state,
        scene_dir=scene_dir,
        dmd_path=dmd_path,
        task_description=task_description,
    )
