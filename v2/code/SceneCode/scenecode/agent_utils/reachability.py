"""Reachability analysis for room traversability.

Computes whether all areas of a room are accessible by a square robot footprint,
identifying which furniture pieces block passages. Uses Shapely polygon operations
for precise 2D geometry.
"""

import json
import logging

from dataclasses import asdict, dataclass

import numpy as np

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from scenecode.agent_utils.room import ObjectType, RoomScene, SceneObject

console_logger = logging.getLogger(__name__)


@dataclass
class ReachabilityResult:
    """Result of room reachability analysis."""

    is_fully_reachable: bool
    """True if room has single connected walkable region."""

    num_disconnected_regions: int
    """Number of disconnected walkable regions (1 = fully reachable)."""

    reachability_ratio: float
    """Ratio of largest region area to total walkable area (1.0 = fully reachable)."""

    blocking_furniture_ids: list[str]
    """IDs of furniture pieces that individually block passages."""

    def to_json(self) -> str:
        """Serialize result to JSON string."""
        return json.dumps(asdict(self))


def compute_reachability(scene: RoomScene, robot_width: float) -> ReachabilityResult:
    """Compute room reachability with given robot footprint.

    Algorithm:
    1. Create floor polygon from room dimensions
    2. Shrink floor by half robot_width for center clearance
    3. Get 2D OBB for each furniture, buffer by half robot_width
    4. Subtract buffered furniture from shrunk floor
    5. Count resulting regions (Polygon=1, MultiPolygon=N)
    6. If disconnected, identify blockers via removal test

    Args:
        scene: Room scene containing furniture objects.
        robot_width: Side length of square robot footprint in meters.

    Returns:
        ReachabilityResult with connectivity analysis.
    """
    half_robot = robot_width / 2

    # Get floor polygon from room dimensions.
    floor = _get_floor_polygon(scene)

    # Shrink floor for robot center clearance.
    walkable = floor.buffer(-half_robot)
    if walkable.is_empty:
        console_logger.warning(
            f"Room too small for robot (width={robot_width}m), "
            f"floor dimensions: {scene.room_geometry.length}x{scene.room_geometry.width}m"
        )
        return ReachabilityResult(
            is_fully_reachable=False,
            num_disconnected_regions=0,
            reachability_ratio=0.0,
            blocking_furniture_ids=[],
        )

    # Get furniture OBBs (only FURNITURE type, not carpets/manipulands).
    furniture_obbs = _get_furniture_obbs(scene)

    # Subtract buffered furniture from walkable area.
    if furniture_obbs:
        buffered = [obb.buffer(half_robot) for _, obb in furniture_obbs]
        furniture_union = unary_union(buffered)
        walkable = walkable.difference(furniture_union)

    # Count regions.
    num_regions = _count_regions(walkable)

    # Compute reachability ratio.
    if walkable.is_empty:
        reachability_ratio = 0.0
    elif isinstance(walkable, MultiPolygon):
        areas = [g.area for g in walkable.geoms]
        reachability_ratio = max(areas) / sum(areas)
    else:
        reachability_ratio = 1.0

    # Identify blockers if disconnected.
    blockers: list[str] = []
    if num_regions > 1:
        blockers = _identify_blockers(
            floor=floor,
            furniture_obbs=furniture_obbs,
            half_robot=half_robot,
            baseline_regions=num_regions,
        )

    is_fully_reachable = num_regions == 1

    console_logger.info(
        f"Reachability: regions={num_regions}, ratio={reachability_ratio:.2f}, "
        f"blockers={blockers}"
    )

    return ReachabilityResult(
        is_fully_reachable=is_fully_reachable,
        num_disconnected_regions=num_regions,
        reachability_ratio=reachability_ratio,
        blocking_furniture_ids=blockers,
    )


def _get_floor_polygon(scene: RoomScene) -> Polygon:
    """Create floor polygon from room dimensions.

    Floor is a rectangle from (0,0) to (length, width) in world coordinates.
    """
    length = scene.room_geometry.length
    width = scene.room_geometry.width
    return Polygon([(0, 0), (length, 0), (length, width), (0, width)])


def _get_furniture_obb_2d(obj: SceneObject) -> Polygon:
    """Get 2D oriented bounding box of furniture in world frame.

    Transforms the 8 corners of the object-frame AABB to world frame,
    projects to XY plane, and computes convex hull.

    Args:
        obj: SceneObject with bbox_min/bbox_max and transform.

    Returns:
        Shapely Polygon representing 2D footprint.
    """
    bmin, bmax = obj.bbox_min, obj.bbox_max

    # Generate 8 corners of 3D AABB in object frame.
    corners_3d = np.array(
        [
            [bmin[0], bmin[1], bmin[2]],
            [bmax[0], bmin[1], bmin[2]],
            [bmax[0], bmax[1], bmin[2]],
            [bmin[0], bmax[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]],
            [bmax[0], bmin[1], bmax[2]],
            [bmax[0], bmax[1], bmax[2]],
            [bmin[0], bmax[1], bmax[2]],
        ]
    )

    # Transform to world frame.
    world_corners = np.array([obj.transform @ c for c in corners_3d])

    # Project to XY and compute convex hull.
    xy_points = [(c[0], c[1]) for c in world_corners]
    return Polygon(xy_points).convex_hull


def _get_furniture_obbs(scene: RoomScene) -> list[tuple[str, Polygon]]:
    """Get OBBs for all FURNITURE objects (excludes thin coverings, manipulands).

    Args:
        scene: Room scene to extract furniture from.

    Returns:
        List of (object_id, polygon) tuples for furniture with valid bounds.
    """
    result = []
    for obj_id, obj in scene.objects.items():
        if obj.object_type != ObjectType.FURNITURE:
            continue
        # Thin coverings (carpets, rugs) don't block walking.
        if obj.metadata.get("asset_source") == "thin_covering":
            continue
        if obj.bbox_min is None or obj.bbox_max is None:
            console_logger.error(f"Skipping {obj_id} in reachability: no bounding box")
            continue
        result.append((str(obj_id), _get_furniture_obb_2d(obj)))
    return result


def _count_regions(walkable: Polygon | MultiPolygon) -> int:
    """Count disconnected walkable regions.

    Args:
        walkable: Result of floor minus furniture.

    Returns:
        Number of regions (0 if empty, 1 if single polygon, N if multi-polygon).
    """
    if walkable.is_empty:
        return 0
    if isinstance(walkable, MultiPolygon):
        return len(walkable.geoms)
    return 1


def _identify_blockers(
    floor: Polygon,
    furniture_obbs: list[tuple[str, Polygon]],
    half_robot: float,
    baseline_regions: int,
) -> list[str]:
    """Identify blocking furniture via removal test.

    For each furniture piece, recompute walkable area without it.
    If removing reduces the number of regions, that piece was blocking.

    Args:
        floor: Floor polygon.
        furniture_obbs: List of (id, polygon) for all furniture.
        half_robot: Half of robot width for buffering.
        baseline_regions: Number of regions with all furniture present.

    Returns:
        List of furniture IDs that individually block passages.
    """
    blockers = []
    for i, (obj_id, _) in enumerate(furniture_obbs):
        # Recompute walkable area without this furniture.
        obbs_without = furniture_obbs[:i] + furniture_obbs[i + 1 :]
        walkable = floor.buffer(-half_robot)
        if obbs_without:
            buffered = [obb.buffer(half_robot) for _, obb in obbs_without]
            walkable = walkable.difference(unary_union(buffered))
        regions_without = _count_regions(walkable)

        # If removing reduces regions, it was blocking.
        if regions_without < baseline_regions:
            blockers.append(obj_id)

    return blockers


def format_reachability_result(result: ReachabilityResult) -> str:
    """Format reachability result as human-readable text.

    Used by both the designer tool and critic context injection.

    Args:
        result: ReachabilityResult from compute_reachability.

    Returns:
        Formatted string describing reachability status.
    """
    if result.is_fully_reachable:
        return "Room is fully reachable - all areas accessible."

    lines = []
    lines.append(f"Room has {result.num_disconnected_regions} disconnected regions")
    lines.append(f"Reachability ratio: {result.reachability_ratio:.1%}")

    if result.blocking_furniture_ids:
        lines.append(f"Blocking furniture: {', '.join(result.blocking_furniture_ids)}")
    else:
        lines.append("No single blocker identified - furniture may need rearrangement")

    return "\n".join(lines)


def format_reachability_for_critic(result: ReachabilityResult) -> str:
    """Format reachability result for critic context injection.

    Returns empty string if fully reachable (no issues to report in prompt).

    Args:
        result: ReachabilityResult from compute_reachability.

    Returns:
        Formatted string for critic prompt, or empty string if no issues.
    """
    if result.is_fully_reachable:
        return ""  # Empty string = no issues for template conditional
    return format_reachability_result(result)
