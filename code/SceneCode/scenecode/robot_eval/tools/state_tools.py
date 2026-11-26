"""State tools for validation and policy interface agents.

Provides geometric facts about the scene (distances, support, spatial relations)
that agents use to assess task completion and verify preconditions. Returns processed
metrics with raw context for the agent to reason about.
"""

import json
import logging

import numpy as np

from agents import FunctionTool, function_tool
from scipy.spatial.transform import Rotation

from scenecode.robot_eval.dmd_scene import DMDScene
from scenecode.robot_eval.tools.response_dataclasses import (
    DistanceResult,
    ObjectDetailInfo,
    ObjectInfo,
    SpatialRelationResult,
    SupportResult,
)

console_logger = logging.getLogger(__name__)


def create_state_tools(scene: DMDScene) -> list[FunctionTool]:
    """Create state tools for agents.

    Creates closure-based tools that capture the scene state, providing
    geometric facts the agent can use for reasoning about task completion
    or candidate selection.

    Args:
        scene: DMDScene with finalized Drake plant and scene_state metadata.

    Returns:
        List of FunctionTool objects for the agent.
    """

    @function_tool
    def list_objects() -> str:
        """List all objects in the scene with their positions.

        Returns a JSON array of objects with id, name, type, and position.
        Use this first to understand what objects are in the scene.
        """
        console_logger.info("Tool called: list_objects")
        objects = []
        for obj_id, obj_data in scene.scene_state["objects"].items():
            info = ObjectInfo(
                id=obj_id,
                description=obj_data.get("description", ""),
                object_type=obj_data.get("object_type", "unknown"),
                position=obj_data["transform"]["translation"],
                bbox_min=obj_data.get("bbox_min"),
                bbox_max=obj_data.get("bbox_max"),
            )
            objects.append(
                {
                    "id": info.id,
                    "description": info.description,
                    "type": info.object_type,
                    "position": [round(p, 3) for p in info.position],
                }
            )
        return json.dumps(objects, indent=2)

    @function_tool
    def get_object_info(object_id: str) -> str:
        """Get detailed information about a specific object.

        Args:
            object_id: The unique identifier of the object.

        Returns:
            JSON with position, orientation (Euler angles), tilt, and dimensions.
        """
        console_logger.info(f"Tool called: get_object_info({object_id})")
        try:
            obj_data = scene.scene_state["objects"][object_id]

            # Get dimensions from bbox if available.
            bbox_min = obj_data.get("bbox_min", [0, 0, 0])
            bbox_max = obj_data.get("bbox_max", [0, 0, 0])
            dimensions = [round(bbox_max[i] - bbox_min[i], 3) for i in range(3)]

            # Convert quaternion to Euler angles.
            quat_wxyz = obj_data["transform"]["rotation_wxyz"]
            # scipy uses xyzw format.
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
            rot = Rotation.from_quat(quat_xyzw)
            euler = rot.as_euler("xyz", degrees=True)
            euler_dict = {
                "roll": round(euler[0], 1),
                "pitch": round(euler[1], 1),
                "yaw": round(euler[2], 1),
            }

            # Compute tilt from upright (angle between Z-axis and world Z).
            rot_matrix = rot.as_matrix()
            z_component = abs(rot_matrix[2, 2])  # Dot product with world Z.
            tilt_deg = round(np.degrees(np.arccos(np.clip(z_component, -1, 1))), 1)

            info = ObjectDetailInfo(
                id=object_id,
                description=obj_data.get("description", ""),
                object_type=obj_data.get("object_type", "unknown"),
                position=[round(p, 3) for p in obj_data["transform"]["translation"]],
                orientation_euler_deg=euler_dict,
                tilt_from_upright_deg=tilt_deg,
                dimensions=dimensions,
            )

            return json.dumps(
                {
                    "id": info.id,
                    "description": info.description,
                    "type": info.object_type,
                    "position_m": info.position,
                    "orientation_euler_deg": info.orientation_euler_deg,
                    "tilt_from_upright_deg": info.tilt_from_upright_deg,
                    "dimensions_m": info.dimensions,
                },
                indent=2,
            )
        except KeyError:
            return json.dumps({"error": f"Object '{object_id}' not found in scene"})

    @function_tool
    def get_distance(object_a: str, object_b: str) -> str:
        """Get surface-to-surface distance between two objects.

        Uses a signed distance query for accurate measurements.
        Negative distance indicates penetration/overlap. Includes object
        dimensions to provide scale context for interpreting the distance.

        Args:
            object_a: First object ID.
            object_b: Second object ID.

        Returns:
            JSON with distance, contact status, and object dimensions.
        """
        console_logger.info(f"Tool called: get_distance({object_a}, {object_b})")
        try:
            # Get object dimensions for context.
            dims_a = _get_object_dimensions(object_a, scene)
            dims_b = _get_object_dimensions(object_b, scene)

            # Compute minimum distance using proper Drake API.
            min_distance = _compute_min_distance_between_models(
                object_a=object_a, object_b=object_b, scene=scene
            )

            if min_distance is not None:
                result = DistanceResult(
                    object_a=object_a,
                    object_b=object_b,
                    # Cast to float to avoid numpy.float64 JSON serialization issues.
                    distance=float(round(min_distance, 4)),
                    in_contact=bool(min_distance <= 0.001),
                    a_dimensions=dims_a,
                    b_dimensions=dims_b,
                )
                return json.dumps(
                    {
                        "object_a": result.object_a,
                        "object_b": result.object_b,
                        "distance_m": result.distance,
                        "in_contact": result.in_contact,
                        "a_dimensions_m": result.a_dimensions,
                        "b_dimensions_m": result.b_dimensions,
                    },
                    indent=2,
                )

            # Objects not found or no collision geometry.
            return json.dumps(
                {
                    "object_a": object_a,
                    "object_b": object_b,
                    "distance_m": "not_measured",
                    "a_dimensions_m": dims_a,
                    "b_dimensions_m": dims_b,
                    "note": "Model not found or has no collision geometry",
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    @function_tool
    def get_spatial_relation(object_a: str, object_b: str) -> str:
        """Get spatial relationship between two objects using surface distances.

        Computes bbox-based surface gaps to accurately determine if objects are
        above/beside/touching each other.

        Args:
            object_a: First object ID.
            object_b: Second object ID.

        Returns:
            JSON with surface gaps, overlap flag, and object dimensions.
        """
        console_logger.info(
            f"Tool called: get_spatial_relation({object_a}, {object_b})"
        )
        try:
            data_a = scene.scene_state["objects"][object_a]
            data_b = scene.scene_state["objects"][object_b]

            pos_a = np.array(data_a["transform"]["translation"])
            pos_b = np.array(data_b["transform"]["translation"])

            bbox_a_min = np.array(data_a.get("bbox_min", [0, 0, 0]))
            bbox_a_max = np.array(data_a.get("bbox_max", [0, 0, 0]))
            bbox_b_min = np.array(data_b.get("bbox_min", [0, 0, 0]))
            bbox_b_max = np.array(data_b.get("bbox_max", [0, 0, 0]))

            # Compute world-frame bbox bounds.
            a_min = pos_a + bbox_a_min
            a_max = pos_a + bbox_a_max
            b_min = pos_b + bbox_b_min
            b_max = pos_b + bbox_b_max

            # Vertical surface gap: bottom of A minus top of B.
            vertical_surface_gap = a_min[2] - b_max[2]

            # Horizontal surface gap: minimum gap between bbox edges in XY.
            x_gap = max(0, max(a_min[0] - b_max[0], b_min[0] - a_max[0]))
            y_gap = max(0, max(a_min[1] - b_max[1], b_min[1] - a_max[1]))
            horizontal_surface_gap = np.sqrt(x_gap**2 + y_gap**2)

            # Check if footprints overlap in XY.
            # Cast to bool to avoid numpy.bool_ JSON serialization issues.
            x_overlap = a_min[0] < b_max[0] and a_max[0] > b_min[0]
            y_overlap = a_min[1] < b_max[1] and a_max[1] > b_min[1]
            footprint_overlaps = bool(x_overlap and y_overlap)

            # Compute dimensions.
            # Cast to float to avoid numpy.float64 JSON serialization issues.
            dims_a = [float(round(bbox_a_max[i] - bbox_a_min[i], 3)) for i in range(3)]
            dims_b = [float(round(bbox_b_max[i] - bbox_b_min[i], 3)) for i in range(3)]

            result = SpatialRelationResult(
                object_a=object_a,
                object_b=object_b,
                vertical_surface_gap=float(round(vertical_surface_gap, 3)),
                horizontal_surface_gap=float(round(horizontal_surface_gap, 3)),
                a_footprint_overlaps_b=footprint_overlaps,
                a_dimensions=dims_a,
                b_dimensions=dims_b,
            )

            return json.dumps(
                {
                    "object_a": result.object_a,
                    "object_b": result.object_b,
                    "vertical_surface_gap_m": result.vertical_surface_gap,
                    "horizontal_surface_gap_m": result.horizontal_surface_gap,
                    "a_footprint_overlaps_b": result.a_footprint_overlaps_b,
                    "a_dimensions_m": result.a_dimensions,
                    "b_dimensions_m": result.b_dimensions,
                },
                indent=2,
            )
        except KeyError as e:
            return json.dumps({"error": f"Object not found: {e}"})

    @function_tool
    def get_support(target: str, surface: str) -> str:
        """Check if target object is supported by a surface.

        Computes vertical gap (bbox-based), contact status (Drake signed distance),
        and footprint overlap percentage.

        Args:
            target: ID of the object being supported.
            surface: ID of the supporting surface/object.

        Returns:
            JSON with vertical gap, contact status, footprint overlap, and dimensions.
        """
        console_logger.info(f"Tool called: get_support({target}, {surface})")
        try:
            result = _compute_support(target_id=target, surface_id=surface, scene=scene)
            return json.dumps(
                {
                    "target": result.target,
                    "surface": result.surface,
                    "vertical_gap_m": result.vertical_gap,
                    "in_contact": result.in_contact,
                    "footprint_on_surface_pct": result.footprint_on_surface_pct,
                    "target_dimensions_m": result.target_dimensions,
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    return [
        list_objects,
        get_object_info,
        get_distance,
        get_spatial_relation,
        get_support,
    ]


def _get_model_name_from_geometry(geometry_id, query_object) -> str:
    """Map geometry ID to model name via frame lookup."""
    inspector = query_object.inspector()
    frame_id = inspector.GetFrameId(geometry_id)
    frame_name = inspector.GetName(frame_id)
    # Strip ::link_name suffix to get model name.
    return frame_name.split("::")[0]


def _compute_min_distance_between_models(
    object_a: str, object_b: str, scene: DMDScene
) -> float | None:
    """Compute minimum signed distance between two model instances.

    Uses Drake's collision geometry and signed distance queries. For models
    with multiple collision geometries (e.g., bed with mattress + frame),
    computes all pairwise distances and returns the minimum.

    Args:
        object_a: First model instance name (e.g., "bedroom_alarm_clock_0").
        object_b: Second model instance name (e.g., "bedroom_bed_0").
        scene: DMDScene with finalized plant.

    Returns:
        Minimum signed distance in meters. Negative means penetration.
        Returns None if models not found or have no collision geometry.
    """
    try:
        plant = scene.plant
        query_object = scene.get_query_object()

        # Get model instances.
        model_a = plant.GetModelInstanceByName(object_a)
        model_b = plant.GetModelInstanceByName(object_b)

        # Collect all collision geometries for both models.
        geoms_a = []
        geoms_b = []

        for body_idx in plant.GetBodyIndices(model_a):
            body = plant.get_body(body_idx)
            geoms_a.extend(plant.GetCollisionGeometriesForBody(body))

        for body_idx in plant.GetBodyIndices(model_b):
            body = plant.get_body(body_idx)
            geoms_b.extend(plant.GetCollisionGeometriesForBody(body))

        if not geoms_a or not geoms_b:
            console_logger.debug(
                f"No collision geometries: {object_a}={len(geoms_a)}, "
                f"{object_b}={len(geoms_b)}"
            )
            return None

        # Compute minimum distance across all geometry pairs.
        min_distance = None
        for geom_a in geoms_a:
            for geom_b in geoms_b:
                try:
                    result = query_object.ComputeSignedDistancePairClosestPoints(
                        geometry_id_A=geom_a, geometry_id_B=geom_b
                    )
                    if min_distance is None or result.distance < min_distance:
                        min_distance = result.distance
                except RuntimeError:
                    # Geometry pair may not support signed distance.
                    continue

        return min_distance

    except RuntimeError as e:
        console_logger.debug(f"Distance computation failed: {e}")
        return None


def _get_object_dimensions(object_id: str, scene: DMDScene) -> list[float] | None:
    """Get object dimensions from scene_state bbox.

    Args:
        object_id: ID of the object.
        scene: DMDScene with scene_state metadata.

    Returns:
        Dimensions [width, depth, height] in meters, or None if not available.
    """
    try:
        obj_data = scene.scene_state["objects"][object_id]
        bbox_min = obj_data.get("bbox_min")
        bbox_max = obj_data.get("bbox_max")
        if bbox_min and bbox_max:
            # Cast to float to ensure Python-native types for JSON serialization.
            return [float(round(bbox_max[i] - bbox_min[i], 3)) for i in range(3)]
        return None
    except KeyError:
        return None


def _compute_support(target_id: str, surface_id: str, scene: DMDScene) -> SupportResult:
    """Check if target is supported by surface.

    Computes:
    - Vertical gap using bbox (bottom of target to top of surface)
    - Contact status using Drake signed distance (ground truth)
    - Footprint overlap percentage

    Handles floor support specially: if surface_id looks like a floor (ends with
    "_floor" or is "floor") but doesn't exist in scene_state, assumes z=0 floor.

    Args:
        target_id: ID of object being supported.
        surface_id: ID of supporting surface/object.
        scene: DMDScene with scene_state metadata and finalized plant.

    Returns:
        SupportResult with support analysis.
    """
    target_data = scene.scene_state["objects"][target_id]

    # Handle floor specially - may not exist in scene_state as explicit object.
    is_floor_surface = surface_id.endswith("_floor") or surface_id == "floor"
    if surface_id not in scene.scene_state["objects"] and is_floor_surface:
        # Create virtual floor at z=0 that covers the whole scene.
        surface_data = {
            "object_type": "floor",
            "transform": {"translation": [0, 0, 0], "rotation_wxyz": [1, 0, 0, 0]},
            "bbox_min": [-100, -100, -0.1],
            "bbox_max": [100, 100, 0.0],
        }
        console_logger.debug(f"Using virtual floor for {surface_id}")
    else:
        surface_data = scene.scene_state["objects"][surface_id]

    target_pos = np.array(target_data["transform"]["translation"])
    surface_pos = np.array(surface_data["transform"]["translation"])

    target_bbox_min = np.array(target_data.get("bbox_min", [0, 0, 0]))
    target_bbox_max = np.array(target_data.get("bbox_max", [0, 0, 0]))
    surface_bbox_min = np.array(surface_data.get("bbox_min", [0, 0, 0]))
    surface_bbox_max = np.array(surface_data.get("bbox_max", [0, 0, 0]))

    # Vertical gap: bottom of target to top of surface.
    target_bottom_z = target_pos[2] + target_bbox_min[2]
    surface_top_z = surface_pos[2] + surface_bbox_max[2]
    vertical_gap = target_bottom_z - surface_top_z

    # Check contact.
    # For floor objects: use vertical_gap (floor collision is in room_geometry model).
    # For other objects: use Drake signed distance (ground truth).
    surface_type = surface_data.get("object_type", "")
    if surface_type == "floor":
        # Floor support: objects within 5mm of floor surface are in contact.
        # Cast to bool to avoid numpy.bool_ JSON serialization issues.
        in_contact = bool(abs(vertical_gap) < 0.005)
    else:
        in_contact = _check_contact_signed_distance(
            object_a=target_id, object_b=surface_id, scene=scene
        )

    # Compute footprint overlap percentage.
    target_min = target_pos + target_bbox_min
    target_max = target_pos + target_bbox_max
    surface_min = surface_pos + surface_bbox_min
    surface_max = surface_pos + surface_bbox_max

    inter_min_x = max(target_min[0], surface_min[0])
    inter_max_x = min(target_max[0], surface_max[0])
    inter_min_y = max(target_min[1], surface_min[1])
    inter_max_y = min(target_max[1], surface_max[1])

    if inter_max_x > inter_min_x and inter_max_y > inter_min_y:
        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        target_area = (target_max[0] - target_min[0]) * (target_max[1] - target_min[1])
        footprint_pct = inter_area / target_area if target_area > 0 else 0.0
    else:
        footprint_pct = 0.0

    # Compute target dimensions.
    # Cast to float to avoid numpy.float64 JSON serialization issues.
    target_dims = [
        float(round(target_bbox_max[i] - target_bbox_min[i], 3)) for i in range(3)
    ]

    return SupportResult(
        target=target_id,
        surface=surface_id,
        vertical_gap=float(round(vertical_gap, 3)),
        in_contact=in_contact,
        footprint_on_surface_pct=float(round(footprint_pct, 2)),
        target_dimensions=target_dims,
    )


def _check_contact_signed_distance(
    object_a: str, object_b: str, scene: DMDScene, threshold: float = 0.005
) -> bool:
    """Check if two objects are in contact using Drake signed distance.

    Args:
        object_a: First object ID.
        object_b: Second object ID.
        scene: DMDScene with finalized plant.
        threshold: Maximum distance to consider as contact (default 5mm).

    Returns:
        True if objects are in contact (signed distance < threshold).
    """
    min_distance = _compute_min_distance_between_models(
        object_a=object_a, object_b=object_b, scene=scene
    )
    if min_distance is not None:
        return min_distance < threshold
    return False
