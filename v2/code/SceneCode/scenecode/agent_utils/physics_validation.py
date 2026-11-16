"""Physics validation for scenes using Drake."""

import logging
import time

from dataclasses import dataclass

import numpy as np

from pydrake.all import DiagramBuilder, GeometryId, QueryObject, SceneGraphInspector

from scenecode.agent_utils.clearance_zones import (
    DoorClearanceViolation,
    OpenConnectionBlockedViolation,
    WallHeightExceededViolation,
    WindowClearanceViolation,
)
from scenecode.agent_utils.drake_utils import (
    create_drake_plant_and_scene_graph_from_scene,
)
from scenecode.agent_utils.room import AgentType, ObjectType, RoomScene, UniqueID

console_logger = logging.getLogger(__name__)

# Floor penetration is evaluated against z=-0.05m instead of Drake's signed
# distance to the full floor slab. This avoids meter-scale false positives
# where the shortest exit path from the slab is horizontal rather than vertical.
FLOOR_PENETRATION_Z_THRESHOLD_M = -0.05


@dataclass
class CollisionPair:
    """Represents a collision between two objects."""

    object_a_name: str
    """Name of the first object in collision."""

    object_a_id: str
    """UniqueID as string of the first object."""

    object_b_name: str
    """Name of the second object in collision."""

    object_b_id: str
    """UniqueID as string of the second object."""

    penetration_depth: float
    """Penetration depth in meters (positive value indicates penetration)."""

    def to_description(self) -> str:
        """Format for agent consumption with directly usable object IDs."""
        penetration_cm = self.penetration_depth * 100
        # Only show penetration depth if meaningful (>0.1mm).
        if penetration_cm > 0.01:
            return (
                f"{self.object_a_id} collides with {self.object_b_id} "
                f"({penetration_cm:.1f}cm penetration)"
            )
        else:
            return f"{self.object_a_id} collides with {self.object_b_id} (touching)"


def _get_furniture_id_for_manipuland(
    manipuland_id: str, scene: RoomScene
) -> str | None:
    """
    Get the furniture or floor ID that owns the surface this manipuland is placed on.

    Args:
        manipuland_id: String ID of the manipuland object.
        scene: RoomScene containing objects.

    Returns:
        String ID of the furniture/floor that owns the surface, or None if not found.
    """
    manipuland = scene.objects.get(UniqueID(manipuland_id))
    if not manipuland or not manipuland.placement_info:
        return None

    surface_id = manipuland.placement_info.parent_surface_id

    # Find furniture, floor, or wall-mounted object that owns this surface.
    for obj_id, obj in scene.objects.items():
        if obj.object_type in (
            ObjectType.FURNITURE,
            ObjectType.FLOOR,
            ObjectType.WALL_MOUNTED,
        ):
            for surface in obj.support_surfaces:
                if surface.surface_id == surface_id:
                    return str(obj_id)

    # Also check room_geometry.floor (not always in scene.objects).
    if scene.room_geometry and scene.room_geometry.floor:
        floor = scene.room_geometry.floor
        for surface in floor.support_surfaces:
            if surface.surface_id == surface_id:
                return str(floor.object_id)

    return None


def _compute_relevant_objects_for_collision(
    scene: RoomScene, current_furniture_id: UniqueID | None
) -> list[UniqueID] | None:
    """
    Determine which objects are relevant for collision checking.

    When current_furniture_id is provided (manipuland agent workflow):
    - Include all furniture (manipulands might extend beyond current surface)
    - Include only manipulands on current furniture/floor
    - For floor placement: floor manipulands must be checked against all furniture

    Returns:
        List of object IDs to include, or None for all objects.
    """
    if current_furniture_id is None:
        return None  # Full scene collision check.

    relevant_objects = []

    # Include all furniture and wall-mounted objects (current + surrounding).
    for obj in scene.objects.values():
        if obj.object_type in (ObjectType.FURNITURE, ObjectType.WALL_MOUNTED):
            relevant_objects.append(obj.object_id)

    # Check if current target is the floor.
    current_obj = scene.objects.get(current_furniture_id)
    is_floor_placement = current_obj and current_obj.object_type == ObjectType.FLOOR

    # Include only manipulands on current furniture/floor.
    for obj_id, obj in scene.objects.items():
        if obj.object_type == ObjectType.MANIPULAND:
            parent_id = _get_furniture_id_for_manipuland(str(obj_id), scene)
            if parent_id == str(current_furniture_id):
                relevant_objects.append(obj_id)

    if is_floor_placement:
        num_furniture = len(
            [
                o
                for o in relevant_objects
                if scene.objects.get(o)
                and scene.objects.get(o).object_type == ObjectType.FURNITURE
            ]
        )
        console_logger.info(
            f"Floor placement mode: including floor manipulands for collision check "
            f"against {num_furniture} furniture items"
        )

    console_logger.debug(
        f"Early filtering: {len(relevant_objects)} objects for collision check "
        f"(out of {len(scene.objects)} total)"
    )

    return relevant_objects


def _compute_floor_penetration_depth(
    scene: RoomScene,
    object_a_info: dict[str, str],
    object_b_info: dict[str, str],
) -> float | None:
    """Compute true floor penetration from world-space object bounds.

    For floor collisions, Drake's signed distance against the full floor slab can
    report a large horizontal "exit distance" even when an object's bottom is
    merely touching the floor. Instead, compute penetration as how far the
    object's lowest world-space point falls below z=-0.05m.

    Returns:
        Penetration depth in meters, or None if the object's bounds are unknown.
    """
    if object_a_info["name"] == "floor":
        non_floor_id = object_b_info["id"]
    elif object_b_info["name"] == "floor":
        non_floor_id = object_a_info["id"]
    else:
        return None

    scene_object = scene.get_object(UniqueID(non_floor_id))
    if scene_object is None:
        return None

    world_bounds = scene_object.compute_world_bounds()
    if world_bounds is None:
        return None

    world_min_z = float(world_bounds[0][2])
    return max(0.0, FLOOR_PENETRATION_Z_THRESHOLD_M - world_min_z)


def compute_scene_collisions(
    scene: RoomScene,
    penetration_threshold: float = 0.001,
    floor_penetration_tolerance: float = 0.05,
    current_furniture_id: UniqueID | None = None,
    manipuland_furniture_tolerance_m: float = 0.02,
) -> list[CollisionPair]:
    """
    Compute collision violations. Also checks for collisions between welded bodies.

    Args:
        scene: RoomScene to check for collisions.
        penetration_threshold: Minimum penetration depth to report (meters).
            Only penetrations deeper than this threshold are considered
            collisions.
        floor_penetration_tolerance: Tolerance for furniture-floor penetration
            (meters). Floor collisions with penetration less than this amount
            are ignored.
        current_furniture_id: Optional ID of furniture currently being populated
            by manipuland agent. When provided, filters out collisions involving
            manipulands from other furniture (unless they collide with current
            furniture's manipulands).
        manipuland_furniture_tolerance_m: Tolerance for current manipuland-current
            furniture surface contact (meters). Mild collisions within this threshold
            are filtered as expected contact. Default 0.02 (2cm).

    Returns:
        List of CollisionPair objects representing detected collisions.
        penetration_depth values are positive (positive = penetration,
        zero = touching).

    Raises:
        RuntimeError: If Drake physics validation fails.
    """
    collision_start_time = time.time()

    # Compute relevant objects for early filtering.
    # When current_furniture_id is provided, only load relevant objects into Drake.
    include_objects = _compute_relevant_objects_for_collision(
        scene=scene, current_furniture_id=current_furniture_id
    )

    # Create Drake scene graph with all objects as free bodies (not welded).
    # This allows broadphase query to detect all collisions.
    # weld_furniture=False makes furniture free, and free_mounted_objects_for_collision=True
    # also makes wall-mounted and ceiling-mounted objects free for collision detection.
    builder = DiagramBuilder()
    _, scene_graph = create_drake_plant_and_scene_graph_from_scene(
        scene=scene,
        builder=builder,
        include_objects=include_objects,
        weld_furniture=False,
        free_mounted_objects_for_collision=True,
    )
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # Get query object for collision detection.
    scene_graph_context = scene_graph.GetMyContextFromRoot(context)
    query_object: QueryObject = scene_graph.get_query_output_port().Eval(
        scene_graph_context
    )

    # Use broadphase collision detection (all objects are free bodies now).
    # This uses Drake's internal broadphase (BVH) for efficient filtering.
    try:
        inspector = query_object.inspector()
        all_signed_distance_pairs = (
            query_object.ComputeSignedDistancePairwiseClosestPoints(
                max_distance=0.0  # Only penetrating pairs (distance < 0).
            )
        )

        # Filter by penetration threshold.
        signed_distance_pairs = [
            pair
            for pair in all_signed_distance_pairs
            if pair.distance <= -penetration_threshold
        ]

        console_logger.debug(
            f"Broadphase found {len(all_signed_distance_pairs)} penetrating pairs, "
            f"{len(signed_distance_pairs)} above threshold"
        )
    except Exception as e:
        error_msg = f"Physics validation failed: {str(e)}"
        console_logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    # Convert to CollisionPair objects with filtering and deduplication.
    collisions = []
    seen_pairs = set()  # Track unique collision pairs to avoid duplicates
    for pair in signed_distance_pairs:
        # Map geometry IDs to object names and IDs.
        object_a_info = _get_object_info_from_geometry_id(
            geometry_id=pair.id_A, scene=scene, query_object=query_object
        )
        object_b_info = _get_object_info_from_geometry_id(
            geometry_id=pair.id_B, scene=scene, query_object=query_object
        )

        # Check if this is a floor collision and apply tolerance.
        is_floor_collision = (
            object_a_info["name"] == "floor" or object_b_info["name"] == "floor"
        )
        penetration_depth = abs(pair.distance)
        if is_floor_collision:
            # Use world-space object bounds for floor penetration instead of
            # Drake's signed distance to the full floor slab.
            computed_floor_penetration = _compute_floor_penetration_depth(
                scene=scene,
                object_a_info=object_a_info,
                object_b_info=object_b_info,
            )
            if computed_floor_penetration is not None:
                penetration_depth = computed_floor_penetration

            # Skip floor collisions that are within tolerance.
            if penetration_depth <= floor_penetration_tolerance:
                continue

        # Create a unique identifier for this collision pair to avoid duplicates.
        # Sort the IDs to ensure A->B and B->A are treated as the same collision.
        pair_id = tuple(
            sorted(
                [
                    f"{object_a_info['name']}[{object_a_info['id']}]",
                    f"{object_b_info['name']}[{object_b_info['id']}]",
                ]
            )
        )

        # Skip if we've already seen this collision pair.
        if pair_id in seen_pairs:
            continue
        seen_pairs.add(pair_id)

        collision = CollisionPair(
            object_a_name=object_a_info["name"],
            object_a_id=object_a_info["id"],
            object_b_name=object_b_info["name"],
            object_b_id=object_b_info["id"],
            penetration_depth=penetration_depth,
        )

        # Post-computation filtering: Apply distance-based filtering.
        # ONLY filters current manipuland × current furniture with ≤2cm penetration.
        # All other collision types use strict no-tolerance policy.
        should_skip, skip_reason = _should_skip_collision_pair(
            gid_a=pair.id_A,
            gid_b=pair.id_B,
            inspector=inspector,
            scene=scene,
            query_object=query_object,
            current_furniture_id=current_furniture_id,
            collision=collision,
            manipuland_furniture_tolerance_m=manipuland_furniture_tolerance_m,
        )
        if should_skip:
            console_logger.debug(f"Skipping collision: {skip_reason}")
            continue

        collisions.append(collision)

    collision_end_time = time.time()
    console_logger.info(
        "Computed scene collisions in "
        f"{collision_end_time - collision_start_time:.2f} seconds. "
        f"Found {len(collisions)} collisions."
    )

    # Log detailed collision information.
    if collisions:
        console_logger.info(f"=== Collision Details ({len(collisions)} total) ===")
        for i, collision in enumerate(collisions, 1):
            console_logger.info(
                f"Collision {i}: {collision.object_a_name} [{collision.object_a_id}] "
                f"<-> {collision.object_b_name} [{collision.object_b_id}] | "
                f"Penetration: {collision.penetration_depth * 100:.2f}cm"
            )
        console_logger.info("=" * 60)

    return collisions


def _should_skip_collision_pair(
    gid_a: GeometryId,
    gid_b: GeometryId,
    inspector: SceneGraphInspector,
    scene: RoomScene,
    query_object: QueryObject,
    current_furniture_id: UniqueID | None,
    collision: CollisionPair | None = None,
    manipuland_furniture_tolerance_m: float = 0.02,
) -> tuple[bool, str]:
    """
    Determine if a collision pair should be skipped.

    Returns True for self-collisions, wall-to-wall collisions,
    (when filtering is enabled) collisions involving non-current manipulands/furniture,
    and (when collision is provided) mild current manipuland-furniture contact.

    This function is called in two stages:
    1. Pre-computation (collision=None): Filters based on geometry relationships
    2. Post-computation (collision provided): Applies distance-based filtering

    Args:
        gid_a: First geometry ID.
        gid_b: Second geometry ID.
        inspector: Drake geometry inspector.
        scene: RoomScene containing objects.
        query_object: Drake query object for geometry inspection.
        current_furniture_id: Optional ID of furniture currently being populated.
            When provided, filters out collisions involving manipulands/furniture from
            other furniture pieces.
        collision: Optional CollisionPair for post-computation distance-based filtering.
            When None, only geometry-based filtering is applied.
        manipuland_furniture_tolerance_m: Tolerance for current manipuland-current
            furniture contact (meters). Only used when collision is provided.

    Returns:
        Tuple of (should_skip: bool, reason: str).
    """
    try:
        # Get frame IDs for both geometries.
        frame_a = inspector.GetFrameId(gid_a)
        frame_b = inspector.GetFrameId(gid_b)

        # Skip self-collisions (same frame).
        if frame_a == frame_b:
            return True, "self-collision (same frame)"

        # Get geometry names to check for wall-to-wall collisions.
        name_a = inspector.GetName(gid_a).lower()
        name_b = inspector.GetName(gid_b).lower()

        # Skip wall-to-wall collisions.
        if "wall" in name_a and "wall" in name_b:
            return True, "wall-to-wall collision"

        # Get object info for both geometries.
        obj_a_info = _get_object_info_from_geometry_id(
            geometry_id=gid_a, scene=scene, query_object=query_object
        )
        obj_b_info = _get_object_info_from_geometry_id(
            geometry_id=gid_b, scene=scene, query_object=query_object
        )

        obj_a_id = obj_a_info["id"]
        obj_b_id = obj_b_info["id"]

        # Skip intra-object collisions (e.g., articulated links, stack members).
        # Both geometries resolve to the same parent object ID.
        if obj_a_id == obj_b_id:
            return True, f"intra-object collision (same parent: {obj_a_id})"

        # Apply manipuland filtering if current_furniture_id is provided.
        if current_furniture_id is not None:
            # Check if objects are manipulands or furniture.
            obj_a_scene_obj = scene.objects.get(UniqueID(obj_a_id))
            obj_b_scene_obj = scene.objects.get(UniqueID(obj_b_id))

            obj_a_is_manipuland = (
                obj_a_scene_obj and obj_a_scene_obj.object_type == ObjectType.MANIPULAND
            )
            obj_b_is_manipuland = (
                obj_b_scene_obj and obj_b_scene_obj.object_type == ObjectType.MANIPULAND
            )

            obj_a_is_furniture = (
                obj_a_scene_obj and obj_a_scene_obj.object_type == ObjectType.FURNITURE
            )
            obj_b_is_furniture = (
                obj_b_scene_obj and obj_b_scene_obj.object_type == ObjectType.FURNITURE
            )

            # Determine if manipulands belong to current furniture.
            is_current_a = False
            is_current_b = False

            if obj_a_is_manipuland:
                furniture_a = _get_furniture_id_for_manipuland(
                    manipuland_id=obj_a_id, scene=scene
                )
                is_current_a = furniture_a == str(current_furniture_id)

            if obj_b_is_manipuland:
                furniture_b = _get_furniture_id_for_manipuland(
                    manipuland_id=obj_b_id, scene=scene
                )
                is_current_b = furniture_b == str(current_furniture_id)

            # Skip if one is non-current manipuland and other is non-current furniture.
            # We have no control over objects from other furniture pieces.
            if obj_a_is_manipuland and not is_current_a and obj_b_is_furniture:
                if str(obj_b_id) != str(current_furniture_id):
                    return True, "non-current manipuland with non-current furniture"
            if obj_b_is_manipuland and not is_current_b and obj_a_is_furniture:
                if str(obj_a_id) != str(current_furniture_id):
                    return True, "non-current manipuland with non-current furniture"

            # Skip if both are non-current manipulands.
            if obj_a_is_manipuland and obj_b_is_manipuland:
                if not is_current_a and not is_current_b:
                    return True, "both non-current manipulands"

            # Skip if one is non-current manipuland and other is floor/wall.
            # (Floor/wall has obj_id == "room_geometry").
            if obj_a_is_manipuland and not is_current_a and obj_b_id == "room_geometry":
                return True, "non-current manipuland with floor/wall"
            if obj_b_is_manipuland and not is_current_b and obj_a_id == "room_geometry":
                return True, "non-current manipuland with floor/wall"

            # Skip if both are non-current furniture.
            # We have no control over furniture we're not currently working with.
            if obj_a_is_furniture and obj_b_is_furniture:
                is_current_furniture_a = str(obj_a_id) == str(current_furniture_id)
                is_current_furniture_b = str(obj_b_id) == str(current_furniture_id)
                if not is_current_furniture_a and not is_current_furniture_b:
                    return True, "both non-current furniture"

            # Skip if one is non-current furniture and other is floor/wall.
            # Other furniture's floor contact is irrelevant to current work.
            if obj_a_is_furniture and obj_b_id == "room_geometry":
                if str(obj_a_id) != str(current_furniture_id):
                    return True, "non-current furniture with floor/wall"
            if obj_b_is_furniture and obj_a_id == "room_geometry":
                if str(obj_b_id) != str(current_furniture_id):
                    return True, "non-current furniture with floor/wall"

        # Wall-mounted object filtering.
        # Wall objects use Drake collision for wall↔wall and wall↔furniture checks.
        # Skip wall↔room_geometry (attached to walls).
        obj_a_scene_obj = scene.objects.get(UniqueID(obj_a_id))
        obj_b_scene_obj = scene.objects.get(UniqueID(obj_b_id))
        obj_a_is_wall_mounted = (
            obj_a_scene_obj and obj_a_scene_obj.object_type == ObjectType.WALL_MOUNTED
        )
        obj_b_is_wall_mounted = (
            obj_b_scene_obj and obj_b_scene_obj.object_type == ObjectType.WALL_MOUNTED
        )
        if obj_a_is_wall_mounted or obj_b_is_wall_mounted:
            # Get the other object's type.
            other_id = obj_b_id if obj_a_is_wall_mounted else obj_a_id

            # Skip wall object ↔ room geometry (wall objects are attached to walls).
            # Room geometry IDs can be "room_geometry" or "room_geometry::north_wall" etc.
            if other_id == "room_geometry" or other_id.startswith("room_geometry::"):
                return True, "wall-mounted object with room geometry"

        # Apply distance-based filtering if collision is provided.
        # This is the post-computation stage after penetration depth is known.
        if collision is not None and current_furniture_id is not None:
            obj_a_id = collision.object_a_id
            obj_b_id = collision.object_b_id

            # Get object types.
            obj_a = scene.objects.get(UniqueID(obj_a_id))
            obj_b = scene.objects.get(UniqueID(obj_b_id))

            obj_a_is_manipuland = obj_a and obj_a.object_type == ObjectType.MANIPULAND
            obj_b_is_manipuland = obj_b and obj_b.object_type == ObjectType.MANIPULAND

            obj_a_is_furniture = obj_a and obj_a.object_type == ObjectType.FURNITURE
            obj_b_is_furniture = obj_b and obj_b.object_type == ObjectType.FURNITURE

            # Check if we have a manipuland-furniture collision.
            if (obj_a_is_manipuland and obj_b_is_furniture) or (
                obj_b_is_manipuland and obj_a_is_furniture
            ):
                # Determine which is manipuland and which is furniture.
                manipuland_id = obj_a_id if obj_a_is_manipuland else obj_b_id
                furniture_id = obj_a_id if obj_a_is_furniture else obj_b_id

                # Check if manipuland belongs to current furniture.
                manipuland_furniture_id = _get_furniture_id_for_manipuland(
                    manipuland_id=manipuland_id, scene=scene
                )
                is_current_manipuland = manipuland_furniture_id == str(
                    current_furniture_id
                )

                # Check if furniture is current furniture.
                is_current_furniture = furniture_id == str(current_furniture_id)

                # Only filter if BOTH are current (manipuland on current furniture ×
                # current furniture). This represents expected surface contact.
                # Do NOT filter if furniture is non-current (we need to know if our
                # manipuland collides with nearby furniture).
                if is_current_manipuland and is_current_furniture:
                    # Apply distance threshold.
                    if collision.penetration_depth <= manipuland_furniture_tolerance_m:
                        penetration_cm = collision.penetration_depth * 100
                        return (
                            True,
                            "mild current manipuland-furniture contact "
                            f"({penetration_cm:.1f}cm ≤ 2cm)",
                        )

        return False, ""

    except Exception as e:
        # If we can't determine the relationship, don't skip.
        console_logger.error(f"Could not determine collision pair relationship: {e}")
        return False, "unknown"


def _find_composite_by_model_name(
    frame_name: str, scene: RoomScene
) -> dict[str, str] | None:
    """Find parent composite (stack or filled_container) by direct model name lookup.

    Uses member_model_names stored in composite metadata during to_drake_directive()
    for reliable O(1) lookup without regex parsing.

    Args:
        frame_name: Drake frame name (e.g., "plate_abc12345_s0001_2::base_link").
        scene: RoomScene containing composite objects.

    Returns:
        Dict with 'name' and 'id' keys if match found, None otherwise.
    """
    # Extract model name from frame (strip ::link_name suffix).
    model_name = frame_name.split("::")[0]

    # Direct lookup in composite metadata (stacks, filled containers, and piles).
    for object_id, scene_object in scene.objects.items():
        composite_type = scene_object.metadata.get("composite_type")
        if composite_type not in ("stack", "filled_container", "pile"):
            continue

        member_model_names = scene_object.metadata.get("member_model_names", [])
        if model_name in member_model_names:
            console_logger.debug(
                f"Composite lookup: model_name={model_name} -> "
                f"composite_type={composite_type}, id={object_id}, "
                f"name={scene_object.name}"
            )
            return {"name": scene_object.name, "id": str(object_id)}

    # Log when no match found - this helps debug association issues.
    console_logger.warning(
        f"Composite lookup FAILED for model_name={model_name}. "
        f"No composite has this in member_model_names."
    )
    return None


# Alias for backwards compatibility.
_find_stack_by_model_name = _find_composite_by_model_name


def _get_object_info_from_geometry_id(
    geometry_id: GeometryId, scene: RoomScene, query_object: QueryObject
) -> dict[str, str]:
    """
    Map a Drake geometry ID to scene object name and ID.

    Args:
        geometry_id: Drake geometry ID.
        scene: RoomScene containing objects.
        query_object: Drake query object for geometry inspection.

    Returns:
        Dictionary with 'name' and 'id' keys.
    """
    inspector = query_object.inspector()

    try:
        # Get frame ID from geometry.
        frame_id = inspector.GetFrameId(geometry_id)
        frame_name = inspector.GetName(frame_id)

        # Special handling for room geometry elements (walls, floor).
        if "room_geometry" in frame_name:
            # Try to extract specific wall/floor name from geometry.
            geometry_name = inspector.GetName(geometry_id)
            if "wall" in geometry_name.lower():
                # Extract wall ID from geometry name (e.g., "west_wall_collision" -> "west_wall").
                wall_id = geometry_name.lower().rsplit("_collision", 1)[0]
                return {"name": geometry_name, "id": wall_id}
            elif "floor" in geometry_name.lower() or "ground" in geometry_name.lower():
                # Floor or ground element.
                return {"name": "floor", "id": "room_geometry"}
            else:
                # Generic room geometry element (fallback).
                return {"name": "wall", "id": "room_geometry"}

        # Extract object ID from frame name for regular scene objects.
        for object_id, scene_object in scene.objects.items():
            # Reconstruct expected model name using same logic as to_drake_directive().
            # This ensures exact matching and avoids suffix collisions.
            base_name = scene_object.name.lower().replace(" ", "_")
            id_suffix = str(object_id).split("_")[-1][:8]
            expected_model_name = f"{base_name}_{id_suffix}"

            # Extract model name from frame (strip ::link_name suffix).
            model_name = frame_name.split("::")[0]

            # Check for exact model name match.
            if model_name == expected_model_name:
                return {"name": scene_object.name, "id": str(object_id)}

        # Try matching to stack objects via direct model name lookup.
        # Uses member_model_names stored in stack metadata.
        stack_match = _find_stack_by_model_name(frame_name=frame_name, scene=scene)
        if stack_match:
            return stack_match

        raise RuntimeError(
            f"Could not map geometry ID {geometry_id} with frame name '{frame_name}' "
            f"to any scene object. This indicates a mismatch between Drake's internal "
            f"naming and our scene object IDs."
        )

    except Exception as e:
        raise RuntimeError(
            f"Error mapping geometry ID {geometry_id} to object: {e}"
        ) from e


def _get_object_type_for_collision_id(
    object_id: str, scene: RoomScene
) -> ObjectType | None:
    """Get ObjectType for a collision object ID.

    Handles special cases like room_geometry (walls/floor).

    Args:
        object_id: Object ID from collision pair.
        scene: RoomScene for looking up object types.

    Returns:
        ObjectType or None if unknown.
    """
    # Handle room geometry special cases.
    if object_id == "room_geometry" or object_id.startswith("room_geometry::"):
        return None  # Room geometry is not modifiable by any agent.

    # Handle specific wall/floor IDs.
    if "_wall" in object_id.lower():
        return None  # Walls are not modifiable.

    scene_obj = scene.objects.get(UniqueID(object_id))
    return scene_obj.object_type if scene_obj else None


def _is_collision_relevant_to_agent(
    collision: CollisionPair,
    scene: RoomScene,
    agent_type: AgentType,
    current_furniture_id: UniqueID | None = None,
) -> bool:
    """Determine if a collision is relevant to the specified agent.

    A collision is relevant if at least one object in the pair is of a type
    that the agent can modify.

    Args:
        collision: CollisionPair to check.
        scene: RoomScene for object type lookups.
        agent_type: Type of agent checking collisions.
        current_furniture_id: For ManipulandAgent, the furniture being populated.

    Returns:
        True if collision is relevant to the agent.
    """
    type_a = _get_object_type_for_collision_id(
        object_id=collision.object_a_id, scene=scene
    )
    type_b = _get_object_type_for_collision_id(
        object_id=collision.object_b_id, scene=scene
    )

    target_type = agent_type.to_object_type()
    if target_type is None:
        # FLOOR_PLAN agent has no object type - can't modify any objects.
        return False

    if agent_type == AgentType.MANIPULAND:
        # Special case: manipuland must belong to current furniture.
        if current_furniture_id is None:
            return False

        def is_current_manipuland(obj_id: str) -> bool:
            scene_obj = scene.objects.get(UniqueID(obj_id))
            if not scene_obj or scene_obj.object_type != ObjectType.MANIPULAND:
                return False
            parent_id = _get_furniture_id_for_manipuland(
                manipuland_id=obj_id, scene=scene
            )
            return parent_id == str(current_furniture_id)

        return is_current_manipuland(collision.object_a_id) or is_current_manipuland(
            collision.object_b_id
        )

    # Standard case: at least one object must match agent's target type.
    return type_a == target_type or type_b == target_type


def filter_collisions_by_agent(
    collisions: list[CollisionPair],
    scene: RoomScene,
    agent_type: AgentType,
    current_furniture_id: UniqueID | None = None,
) -> list[CollisionPair]:
    """Filter collisions to show only those relevant to the specified agent.

    Each agent type can only modify certain object types:
    - FurnitureAgent: FURNITURE objects
    - ManipulandAgent: MANIPULAND objects on current furniture
    - WallAgent: WALL_MOUNTED objects
    - CeilingAgent: CEILING_MOUNTED objects

    A collision is relevant if at least one object in the pair is of a type
    the agent can modify.

    Args:
        collisions: List of collision pairs to filter.
        scene: RoomScene for looking up object types.
        agent_type: Type of agent requesting collisions.
        current_furniture_id: For ManipulandAgent, the furniture being populated.

    Returns:
        Filtered list of collision pairs.
    """
    filtered = []
    for collision in collisions:
        if _is_collision_relevant_to_agent(
            collision=collision,
            scene=scene,
            agent_type=agent_type,
            current_furniture_id=current_furniture_id,
        ):
            filtered.append(collision)

    if len(filtered) < len(collisions):
        console_logger.debug(
            f"Filtered collisions by agent type {agent_type.value}: "
            f"{len(collisions)} -> {len(filtered)}"
        )

    return filtered


def _get_thin_covering_owner_agent(
    covering_id: str, scene: RoomScene
) -> AgentType | None:
    """Determine which agent type owns a thin covering.

    Args:
        covering_id: UniqueID of the thin covering.
        scene: RoomScene for object lookups.

    Returns:
        AgentType that owns the covering, or None if unknown.
    """
    scene_obj = scene.objects.get(UniqueID(covering_id))
    if not scene_obj:
        return None

    # Wall coverings are owned by wall agent.
    if scene_obj.metadata.get("is_wall_covering", False):
        return AgentType.WALL_MOUNTED

    # Check object type - this tells us which agent placed it.
    if scene_obj.object_type == ObjectType.MANIPULAND:
        return AgentType.MANIPULAND
    elif scene_obj.object_type == ObjectType.FURNITURE:
        # Floor coverings (rugs, carpets) are placed by furniture agent.
        return AgentType.FURNITURE

    return None


def filter_thin_covering_overlaps_by_agent(
    overlaps: list["ThinCoveringOverlap"],
    scene: RoomScene,
    agent_type: AgentType,
    current_furniture_id: UniqueID | None = None,
) -> list["ThinCoveringOverlap"]:
    """Filter thin covering overlaps to show only those relevant to the agent.

    Args:
        overlaps: List of thin covering overlaps to filter.
        scene: RoomScene for object lookups.
        agent_type: Type of agent requesting the violations.
        current_furniture_id: For ManipulandAgent, the furniture being populated.

    Returns:
        Filtered list of overlaps.
    """
    filtered = []
    for overlap in overlaps:
        owner_a = _get_thin_covering_owner_agent(
            covering_id=overlap.covering_a_id, scene=scene
        )
        owner_b = _get_thin_covering_owner_agent(
            covering_id=overlap.covering_b_id, scene=scene
        )

        # Check if at least one covering belongs to this agent.
        if owner_a == agent_type or owner_b == agent_type:
            # For manipuland agent, additionally check if on current furniture.
            if agent_type == AgentType.MANIPULAND and current_furniture_id is not None:
                # Check if at least one covering is on current furniture.
                def is_on_current_furniture(covering_id: str) -> bool:
                    obj = scene.objects.get(UniqueID(covering_id))
                    if not obj or not obj.placement_info:
                        return False
                    parent_id = _get_furniture_id_for_manipuland(
                        manipuland_id=covering_id, scene=scene
                    )
                    return parent_id == str(current_furniture_id)

                if is_on_current_furniture(
                    overlap.covering_a_id
                ) or is_on_current_furniture(overlap.covering_b_id):
                    filtered.append(overlap)
            else:
                filtered.append(overlap)

    if len(filtered) < len(overlaps):
        console_logger.debug(
            f"Filtered thin covering overlaps by agent type {agent_type.value}: "
            f"{len(overlaps)} -> {len(filtered)}"
        )

    return filtered


def filter_thin_covering_boundary_violations_by_agent(
    violations: list["ThinCoveringBoundaryViolation"],
    agent_type: AgentType,
) -> list["ThinCoveringBoundaryViolation"]:
    """Filter thin covering boundary violations by agent type.

    Only floor coverings can have boundary violations (extending beyond room walls).
    Wall coverings and surface coverings don't have floor boundary constraints.

    Args:
        violations: List of boundary violations to filter.
        agent_type: Type of agent requesting the violations.

    Returns:
        Filtered list of violations (only FurnitureAgent sees floor covering violations).
    """
    # Only furniture agent places floor coverings, which are subject to room boundaries.
    if agent_type != AgentType.FURNITURE:
        return []

    # For furniture agent, show all floor covering boundary violations.
    return violations


@dataclass
class ThinCoveringOverlap:
    """Represents an overlap between two thin covering objects."""

    covering_a_name: str
    """Name of the first thin covering."""

    covering_a_id: str
    """UniqueID as string of the first thin covering."""

    covering_b_name: str
    """Name of the second thin covering."""

    covering_b_id: str
    """UniqueID as string of the second thin covering."""

    def to_description(self) -> str:
        """Format for human/VLM consumption."""
        return (
            f"Thin covering '{self.covering_a_name}' [{self.covering_a_id}] overlaps "
            f"with '{self.covering_b_name}' [{self.covering_b_id}]"
        )


@dataclass
class ThinCoveringBoundaryViolation:
    """Represents a thin covering extending beyond floor plan boundaries."""

    covering_id: str
    """UniqueID as string of the violating thin covering."""

    exceeded_boundaries: list[str]
    """List of boundary names exceeded, e.g., ["north", "east"]."""

    def to_description(self) -> str:
        """Format for human/VLM consumption."""
        boundaries_str = ", ".join(self.exceeded_boundaries)
        suffix = "boundaries" if len(self.exceeded_boundaries) > 1 else "boundary"
        return f"Thin covering [{self.covering_id}] extends beyond {boundaries_str} {suffix}"


def _get_obb_corners_2d(
    center_x: float, center_y: float, half_w: float, half_d: float, yaw: float
) -> np.ndarray:
    """Compute 2D OBB corner points given center, half-extents, and rotation.

    Args:
        center_x: X position of center.
        center_y: Y position of center.
        half_w: Half-width (X extent before rotation).
        half_d: Half-depth (Y extent before rotation).
        yaw: Rotation angle in radians (around Z axis).

    Returns:
        Array of shape (4, 2) with corner points in counter-clockwise order.
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Local corners (before rotation).
    local_corners = np.array(
        [
            [-half_w, -half_d],
            [+half_w, -half_d],
            [+half_w, +half_d],
            [-half_w, +half_d],
        ]
    )

    # Rotation matrix.
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

    # Transform to world space.
    world_corners = (rot @ local_corners.T).T + np.array([center_x, center_y])
    return world_corners


def _obb_overlap_2d(corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
    """Check if two 2D OBBs overlap using Separating Axis Theorem.

    Args:
        corners_a: Array of shape (4, 2) with corners of first OBB.
        corners_b: Array of shape (4, 2) with corners of second OBB.

    Returns:
        True if OBBs overlap, False otherwise.
    """

    def get_axes(corners: np.ndarray) -> list[np.ndarray]:
        """Get edge normal axes for SAT test."""
        axes = []
        for i in range(4):
            edge = corners[(i + 1) % 4] - corners[i]
            # Perpendicular (normal) to edge.
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        return axes

    def project(corners: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
        """Project corners onto axis, return min and max."""
        dots = corners @ axis
        return dots.min(), dots.max()

    # Test all 4 axes (2 from each OBB).
    for axis in get_axes(corners_a) + get_axes(corners_b):
        min_a, max_a = project(corners_a, axis)
        min_b, max_b = project(corners_b, axis)

        # Check for separation on this axis.
        if max_a < min_b or max_b < min_a:
            return False

    return True


def compute_thin_covering_overlaps(scene: RoomScene) -> list[ThinCoveringOverlap]:
    """Check for overlapping floor/manipuland thin coverings using 2D OBB intersection.

    Floor thin coverings (rugs, carpets) and manipuland thin coverings
    (tablecloths, placemats) have no collision geometry (purely decorative),
    so they won't appear in Drake collision detection. This function performs
    a separate check for thin covering overlaps using oriented bounding boxes.

    Wall thin coverings (paintings, posters) are excluded - they have collision
    geometry and use Drake collision detection instead.

    Args:
        scene: RoomScene containing thin covering objects.

    Returns:
        List of ThinCoveringOverlap objects representing detected overlaps.
    """
    coverings = []
    for obj_id, obj in scene.objects.items():
        if obj.metadata.get("asset_source") != "thin_covering":
            continue

        # Wall coverings use Drake collision detection (have collision geometry).
        # Skip them here - only check floor/manipuland thin coverings.
        if obj.metadata.get("is_wall_covering", False):
            continue

        width = obj.metadata.get("width_m")
        depth = obj.metadata.get("depth_m")
        if width is None or depth is None:
            console_logger.error(
                f"Thin covering '{obj.name}' [{obj_id}] missing width_m/depth_m "
                "metadata, skipping overlap check"
            )
            continue

        transform = obj.transform
        pos = transform.translation()

        rot_matrix = transform.rotation().matrix()
        yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])

        corners = _get_obb_corners_2d(
            center_x=pos[0],
            center_y=pos[1],
            half_w=width / 2.0,
            half_d=depth / 2.0,
            yaw=yaw,
        )

        coverings.append({"id": obj_id, "name": obj.name, "corners": corners})

    overlaps: list[ThinCoveringOverlap] = []
    for i in range(len(coverings)):
        for j in range(i + 1, len(coverings)):
            a = coverings[i]
            b = coverings[j]

            if _obb_overlap_2d(a["corners"], b["corners"]):
                overlaps.append(
                    ThinCoveringOverlap(
                        covering_a_name=a["name"],
                        covering_a_id=str(a["id"]),
                        covering_b_name=b["name"],
                        covering_b_id=str(b["id"]),
                    )
                )

    if overlaps:
        console_logger.info(f"Found {len(overlaps)} thin covering overlap(s)")
        for overlap in overlaps:
            console_logger.info(f"  {overlap.to_description()}")

    return overlaps


def compute_thin_covering_boundary_violations(
    scene: RoomScene, wall_thickness: float
) -> list[ThinCoveringBoundaryViolation]:
    """Check if floor thin coverings extend beyond the usable floor area.

    Floor thin coverings (rugs, carpets) have no collision geometry (purely
    decorative), so they won't appear in Drake collision detection. This
    function checks if thin coverings extend beyond floor plan boundaries,
    accounting for wall thickness.

    Wall thin coverings (paintings, posters) are excluded - they're mounted
    on walls, not on floors, and use Drake collision detection.

    The usable floor area is smaller than the room dimensions because walls
    extend inward by wall_thickness/2 on each side.

    Args:
        scene: RoomScene containing thin covering objects and room_geometry.
        wall_thickness: Wall thickness in meters. Walls extend inward by
            wall_thickness/2 from the room boundary.

    Returns:
        List of ThinCoveringBoundaryViolation objects representing detected violations.
    """
    # Compute usable floor bounds (inside walls).
    half_wall = wall_thickness / 2.0
    room_length = scene.room_geometry.length  # x-dimension
    room_width = scene.room_geometry.width  # y-dimension

    inner_min_x = -room_length / 2.0 + half_wall
    inner_max_x = room_length / 2.0 - half_wall
    inner_min_y = -room_width / 2.0 + half_wall
    inner_max_y = room_width / 2.0 - half_wall

    violations: list[ThinCoveringBoundaryViolation] = []

    for obj_id, obj in scene.objects.items():
        if obj.metadata.get("asset_source") != "thin_covering":
            continue

        # Wall coverings are on walls, not floors - skip floor boundary check.
        if obj.metadata.get("is_wall_covering", False):
            continue

        width = obj.metadata.get("width_m")
        depth = obj.metadata.get("depth_m")
        shape = obj.metadata.get("shape", "rectangular")

        if width is None or depth is None:
            raise ValueError(
                f"Thin covering [{obj_id}] missing width_m/depth_m metadata. "
                f"This is a bug in thin covering creation."
            )

        transform = obj.transform
        pos = transform.translation()
        center_x, center_y = pos[0], pos[1]

        exceeded_boundaries: list[str] = []

        if shape == "circular":
            # For circular thin coverings, use radius = min(width, depth) / 2.
            radius = min(width, depth) / 2.0

            if center_x - radius < inner_min_x:
                exceeded_boundaries.append("west")
            if center_x + radius > inner_max_x:
                exceeded_boundaries.append("east")
            if center_y - radius < inner_min_y:
                exceeded_boundaries.append("south")
            if center_y + radius > inner_max_y:
                exceeded_boundaries.append("north")
        else:
            # For rectangular thin coverings, compute OBB corners.
            rot_matrix = transform.rotation().matrix()
            yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])

            corners = _get_obb_corners_2d(
                center_x=center_x,
                center_y=center_y,
                half_w=width / 2.0,
                half_d=depth / 2.0,
                yaw=yaw,
            )

            # Check if any corner exceeds bounds.
            min_corner_x = np.min(corners[:, 0])
            max_corner_x = np.max(corners[:, 0])
            min_corner_y = np.min(corners[:, 1])
            max_corner_y = np.max(corners[:, 1])

            if min_corner_x < inner_min_x:
                exceeded_boundaries.append("west")
            if max_corner_x > inner_max_x:
                exceeded_boundaries.append("east")
            if min_corner_y < inner_min_y:
                exceeded_boundaries.append("south")
            if max_corner_y > inner_max_y:
                exceeded_boundaries.append("north")

        if exceeded_boundaries:
            # Sort for consistent output.
            exceeded_boundaries.sort()
            violations.append(
                ThinCoveringBoundaryViolation(
                    covering_id=str(obj_id),
                    exceeded_boundaries=exceeded_boundaries,
                )
            )

    if violations:
        console_logger.info(
            f"Found {len(violations)} thin covering boundary violation(s)"
        )
        for v in violations:
            console_logger.info(f"  {v.to_description()}")

    return violations


def filter_window_violations_by_agent(
    violations: list[WindowClearanceViolation], scene: RoomScene, agent_type: AgentType
) -> list[WindowClearanceViolation]:
    """Filter window clearance violations by agent type.

    Only shows violations where the blocking object is of a type the agent can modify.

    Args:
        violations: List of window clearance violations to filter.
        scene: RoomScene for object lookups.
        agent_type: Type of agent requesting the violations.

    Returns:
        Filtered list of violations for objects the agent can move.
    """
    target_object_type = agent_type.to_object_type()
    if target_object_type is None:
        # Floor plan agent doesn't place objects that can block windows.
        return []

    filtered = []
    for v in violations:
        obj_type = _get_object_type_for_collision_id(v.furniture_id, scene)
        if obj_type == target_object_type:
            filtered.append(v)

    return filtered


def filter_wall_height_violations_by_agent(
    violations: list[WallHeightExceededViolation],
    scene: RoomScene,
    agent_type: AgentType,
) -> list[WallHeightExceededViolation]:
    """Filter wall height violations by agent type.

    Only shows violations where the object exceeding wall height is of a type
    the agent can modify.

    Args:
        violations: List of wall height violations to filter.
        scene: RoomScene for object lookups.
        agent_type: Type of agent requesting the violations.

    Returns:
        Filtered list of violations for objects the agent can move.
    """
    target_object_type = agent_type.to_object_type()
    if target_object_type is None:
        # Floor plan agent doesn't place objects that can exceed wall height.
        return []

    filtered = []
    for v in violations:
        obj_type = _get_object_type_for_collision_id(v.object_id, scene)
        if obj_type == target_object_type:
            filtered.append(v)

    return filtered


def filter_door_violations_by_agent(
    violations: list[DoorClearanceViolation],
    scene: RoomScene,
    agent_type: AgentType,
) -> list[DoorClearanceViolation]:
    """Filter door clearance violations by agent type.

    Only shows violations where the blocking object is of a type the agent can modify.

    Args:
        violations: List of door clearance violations to filter.
        scene: RoomScene for object lookups.
        agent_type: Type of agent requesting the violations.

    Returns:
        Filtered list of violations for objects the agent can move.
    """
    target_object_type = agent_type.to_object_type()
    if target_object_type is None:
        return []

    filtered = []
    for v in violations:
        obj_type = _get_object_type_for_collision_id(v.furniture_id, scene)
        if obj_type == target_object_type:
            filtered.append(v)

    return filtered


def filter_open_connection_violations_by_agent(
    violations: list[OpenConnectionBlockedViolation],
    scene: RoomScene,
    agent_type: AgentType,
) -> list[OpenConnectionBlockedViolation]:
    """Filter open connection violations by agent type.

    Only shows violations where at least one blocking object is of a type the agent
    can modify.

    Args:
        violations: List of open connection violations to filter.
        scene: RoomScene for object lookups.
        agent_type: Type of agent requesting the violations.

    Returns:
        Filtered list of violations for objects the agent can move.
    """
    target_object_type = agent_type.to_object_type()
    if target_object_type is None:
        return []

    filtered = []
    for v in violations:
        # Check if any blocking furniture is of the agent's type.
        for furniture_id in v.blocking_furniture_ids:
            obj_type = _get_object_type_for_collision_id(furniture_id, scene)
            if obj_type == target_object_type:
                filtered.append(v)
                break  # Only add once per violation.

    return filtered
