#!/usr/bin/env python
"""Convert a generated scene into robot-executable poses.

This is the Policy Interface stage of the robot evaluation pipeline. It converts
a natural language task into concrete object IDs and exact poses that a robot
policy can execute.

The unified PolicyInterfaceAgent:
1. Parses the task → goal predicates + preconditions
2. Finds objects in scene matching categories
3. Verifies preconditions using state/vision tools
4. Returns ranked valid (target, reference) bindings
5. Computes exact poses for each binding

Usage:
    python scripts/robot_eval/policy_interface.py \
        --scene-state outputs/.../scene_002/combined_house/house_state.json \
        --dmd outputs/.../scene_002/combined_house/house.dmd.yaml \
        --scene-dir outputs/.../scene_002 \
        --task "Find a speaker and place it on the bed" \
        --door-clearance-m 1.0

Inputs:
    --scene-state: Path to scene_state.json (per-room) or house_state.json (combined house)
    --dmd: Path to scene.dmd.yaml or house.dmd.yaml (Drake scene with poses)
    --scene-dir: Scene root directory for package:// URI resolution (default: parent of DMD)
    --door-clearance-m: Distance from door for robot start position (default: 1.0)

Output:
    JSON with robot_start_xy and robot-executable commands with target poses.
"""

import argparse
import asyncio
import json
import logging

from pathlib import Path

from scenecode.agent_utils.blender import BlenderServer
from scenecode.robot_eval import create_robot_eval_config
from scenecode.robot_eval.dmd_scene import load_scene_for_validation
from scenecode.robot_eval.policy_interface.predicate_resolver import PredicateResolver

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Outward normal vectors for each wall direction (robot faces into the room).
OUTWARD_NORMALS = {
    "north": (0.0, 1.0),
    "south": (0.0, -1.0),
    "east": (1.0, 0.0),
    "west": (-1.0, 0.0),
}


def compute_robot_start_xy(
    raw_state: dict, door_clearance_m: float
) -> tuple[float, float] | None:
    """Compute robot start XY from exterior door position.

    For house scenes with exterior doors, places the robot centered at the door
    and offset outward by door_clearance_m. For scenes without exterior doors,
    falls back to room center.

    Args:
        raw_state: Raw scene state dict (house_state.json or scene_state.json).
        door_clearance_m: Distance from door center to robot position.

    Returns:
        (x, y) start position in world coordinates, or None if cannot compute.
    """
    layout = raw_state.get("layout")
    if layout is None:
        logger.debug("No layout in scene state, cannot compute robot start position")
        return None

    placed_rooms = layout.get("placed_rooms", [])
    doors = layout.get("doors", [])

    exterior_door = next((d for d in doors if d.get("door_type") == "exterior"), None)
    if exterior_door is None:
        # Fall back to room center.
        if placed_rooms:
            room = placed_rooms[0]
            pos = room["position"]
            center_x = pos[0] + room["width"] / 2
            center_y = pos[1] + room["depth"] / 2
            logger.info(
                "No exterior door found, using room center: "
                f"({center_x:.2f}, {center_y:.2f})"
            )
            return (center_x, center_y)
        logger.warning(
            "No exterior door and no placed rooms, cannot compute start position"
        )
        return None

    # Find the room and wall containing the door.
    room_id = exterior_door["room_a"]
    door_id = exterior_door["id"]
    target_room = next((r for r in placed_rooms if r["room_id"] == room_id), None)
    if target_room is None:
        logger.warning(f"Room {room_id} not found in placed_rooms")
        return None

    # Find wall with this door opening.
    target_wall = next(
        (
            wall
            for wall in target_room.get("walls", [])
            if any(o.get("opening_id") == door_id for o in wall.get("openings", []))
        ),
        None,
    )
    if target_wall is None:
        logger.warning(f"Wall containing door {door_id} not found")
        return None

    # Compute door center position along wall.
    door_position_exact = exterior_door["position_exact"]
    door_width = exterior_door.get("width", 1.0)
    door_center_along_wall = door_position_exact + door_width / 2

    # Interpolate along wall to get world position.
    start_point = target_wall["start_point"]
    end_point = target_wall["end_point"]
    wall_length = target_wall["length"]

    if wall_length <= 0:
        logger.warning(f"Invalid wall length: {wall_length}")
        return None

    t = door_center_along_wall / wall_length
    door_center_x = start_point[0] + t * (end_point[0] - start_point[0])
    door_center_y = start_point[1] + t * (end_point[1] - start_point[1])

    # Get outward normal and compute robot position.
    direction = target_wall["direction"]
    outward = OUTWARD_NORMALS.get(direction)
    if outward is None:
        logger.warning(f"Unknown wall direction: {direction}")
        return None

    robot_x = door_center_x + door_clearance_m * outward[0]
    robot_y = door_center_y + door_clearance_m * outward[1]

    logger.info(
        f"Robot start position: ({robot_x:.2f}, {robot_y:.2f}) "
        f"[{door_clearance_m}m from {direction} door {door_id}]"
    )
    return (robot_x, robot_y)


def compute_world_bounds(
    raw_state: dict, inflation_m: float = 2.0
) -> dict[str, list[float]] | None:
    """Compute world bounds from room geometry, inflated by given distance.

    For sampling-based motion planners that need bounded degrees of freedom.

    Args:
        raw_state: Raw scene state dict (house_state.json or scene_state.json).
        inflation_m: Distance to inflate XY bounds by (default: 2.0m).

    Returns:
        Dict with "min" and "max" keys containing [x, y, z] coordinates,
        or None if cannot compute.
    """
    layout = raw_state.get("layout")
    if layout is None:
        return None

    placed_rooms = layout.get("placed_rooms", [])
    if not placed_rooms:
        return None

    # Get ceiling height (z max). Floor is at z=0.
    wall_height = layout.get("wall_height", 2.7)

    # Compute union of all room bounds.
    all_min_x, all_min_y = float("inf"), float("inf")
    all_max_x, all_max_y = float("-inf"), float("-inf")

    for room in placed_rooms:
        pos = room["position"]
        room_min_x = pos[0]
        room_min_y = pos[1]
        room_max_x = pos[0] + room["width"]
        room_max_y = pos[1] + room["depth"]

        all_min_x = min(all_min_x, room_min_x)
        all_min_y = min(all_min_y, room_min_y)
        all_max_x = max(all_max_x, room_max_x)
        all_max_y = max(all_max_y, room_max_y)

    # Inflate XY by specified distance. Z goes from floor (0) to ceiling.
    return {
        "min": [all_min_x - inflation_m, all_min_y - inflation_m, 0.0],
        "max": [all_max_x + inflation_m, all_max_y + inflation_m, wall_height],
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Convert scene + task into robot-executable poses"
    )
    parser.add_argument(
        "--scene-state",
        required=True,
        type=Path,
        help="Path to scene_state.json (object metadata)",
    )
    parser.add_argument(
        "--dmd",
        required=True,
        type=Path,
        help="Path to scene.dmd.yaml (Drake scene)",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Human task description (e.g., 'Find a fruit and place it on the table')",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="Model for policy interface agent (default: gpt-5.2)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional: Write robot commands to JSON file",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        help="Scene root directory for package:// URI resolution (default: parent of DMD)",
    )
    parser.add_argument(
        "--door-clearance-m",
        type=float,
        default=1.0,
        help="Distance from door for robot start position (default: 1.0)",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision tools (skip Blender server startup)",
    )
    args = parser.parse_args()

    cfg = create_robot_eval_config(model=args.model)

    # Load scene.
    logger.info(f"Task: {args.task}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Scene state: {args.scene_state}")
    logger.info(f"DMD: {args.dmd}")
    if args.scene_dir:
        logger.info(f"Scene dir: {args.scene_dir}")

    # Load raw state to compute robot start position and world bounds.
    with open(args.scene_state) as f:
        raw_state = json.load(f)
    robot_start_xy = compute_robot_start_xy(raw_state, args.door_clearance_m)
    world_bounds = compute_world_bounds(raw_state, inflation_m=2.0)

    scene = load_scene_for_validation(
        scene_state_path=args.scene_state,
        dmd_path=args.dmd,
        task_description=args.task,
        scene_dir=args.scene_dir,
    )
    scene.finalize()

    # Start Blender server for vision tools (unless disabled).
    blender_server = None
    if not args.no_vision:
        logger.info("Starting Blender server for vision tools...")
        blender_server = BlenderServer()
        blender_server.start()

    try:
        # Resolve task to poses using unified agent.
        logger.info("Resolving task with policy interface agent...")
        resolver = PredicateResolver(
            scene=scene, cfg=cfg, blender_server=blender_server
        )
        result = await resolver.resolve_async(task_description=args.task)

        for note in result.notes:
            logger.info(f"  {note}")

        logger.info("")
        logger.info("=" * 60)
        logger.info("ROBOT OUTPUT")
        logger.info("=" * 60)

        if robot_start_xy:
            logger.info(
                f"Robot start XY: [{robot_start_xy[0]:.4f}, {robot_start_xy[1]:.4f}]"
            )
        else:
            logger.info("Robot start XY: None (could not compute)")

        if world_bounds:
            logger.info(
                f"World bounds: [{world_bounds['min'][0]:.2f}, "
                f"{world_bounds['min'][1]:.2f}, {world_bounds['min'][2]:.2f}] → "
                f"[{world_bounds['max'][0]:.2f}, {world_bounds['max'][1]:.2f}, "
                f"{world_bounds['max'][2]:.2f}]"
            )

        logger.info("")
        logger.info("Commands:")
        robot_commands = []
        for exact in result.poses:
            pos = exact.target_position
            cmd = {
                "action": exact.action,
                "rank": exact.rank,
                "confidence": exact.confidence,
                "drake_model_name": exact.drake_model_name,
                "target_position": [round(p, 4) for p in pos],
                "placement_bounds_min": (
                    [round(p, 4) for p in exact.placement_bounds_min]
                    if exact.placement_bounds_min
                    else None
                ),
                "placement_bounds_max": (
                    [round(p, 4) for p in exact.placement_bounds_max]
                    if exact.placement_bounds_max
                    else None
                ),
                "reasoning": exact.reasoning,
            }
            robot_commands.append(cmd)
            logger.info(
                f"  rank={exact.rank} conf={exact.confidence:.2f}: "
                f"{exact.action}({exact.drake_model_name}) → "
                f"position [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
            )
            if exact.reasoning:
                logger.info(f"    reasoning: {exact.reasoning}")
            if exact.placement_bounds_min and exact.placement_bounds_max:
                bounds_min = exact.placement_bounds_min
                bounds_max = exact.placement_bounds_max
                logger.info(
                    f"    placement region: [{bounds_min[0]:.3f}, {bounds_min[1]:.3f}, {bounds_min[2]:.3f}] → "
                    f"[{bounds_max[0]:.3f}, {bounds_max[1]:.3f}, {bounds_max[2]:.3f}]"
                )

        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            output = {
                "task": args.task,
                "robot_start_xy": list(robot_start_xy) if robot_start_xy else None,
                "world_bounds": world_bounds,
                "commands": robot_commands,
            }
            with open(args.output_json, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Robot output written to: {args.output_json}")
    finally:
        if blender_server is not None:
            logger.info("Stopping Blender server...")
            blender_server.stop()


if __name__ == "__main__":
    asyncio.run(main())
