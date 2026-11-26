"""Snapping algorithm helpers for scene tools.

This module contains the core snapping algorithms used to position furniture
and objects in the scene, including collision detection and resolution.
"""

import gc
import logging
import time

import numpy as np
import trimesh
import trimesh.collision

from omegaconf import DictConfig
from pydrake.all import RigidTransform

from scenecode.agent_utils.room import ObjectType, SceneObject
from scenecode.furniture_agents.tools.response_dataclasses import (
    FurnitureErrorType,
    SnapToObjectResult,
)
from scenecode.utils.geometry_utils import (
    convert_mesh_yup_to_zup,
    rigid_transform_to_matrix,
)
from scenecode.utils.mesh_loading import (
    get_collision_vertices_world,
    load_object_collision_geometry,
)

console_logger = logging.getLogger(__name__)

# Numerical thresholds for snapping algorithms.
DEGENERATE_VOLUME_THRESHOLD = 1e-6  # Zero-volume detection.
ZERO_DISTANCE_THRESHOLD = 1e-6  # Distance comparison threshold.


def snap_mesh_to_aabb(
    obj: SceneObject, target: SceneObject, cfg: DictConfig
) -> tuple[np.ndarray, float]:
    """Snap object to another object using collision geometry and AABB.

    Uses CollisionManager for accurate collision detection with the object's
    collision geometry (from SDF) and the target's AABB. This approach is more
    accurate than vertex-based detection and doesn't require downsampling.

    This function handles three cases:
    1. Gap: Object outside target → moves object closer to touch surface.
    2. Touching: Object surface touching target → no movement needed.
    3. Penetration: Objects overlapping → pushes out, then snaps to surface.

    Args:
        obj: Object to move (must have sdf_path with collision geometry).
        target: Target object (must have bbox).
        cfg: Configuration with snap_to_object.snap_margin_m for gap size.

    Returns:
        Tuple of (movement_vector, distance_moved).

    Raises:
        ValueError: If collision geometry cannot be loaded or no bbox.
    """
    # Load collision geometry with automatic fallback to visual geometry.
    obj_collision_meshes = load_object_collision_geometry(obj)

    console_logger.info(
        f"Mesh-to-AABB snap: obj={len(obj_collision_meshes)} collision pieces, "
        f"target=AABB"
    )

    # Get target's world-space AABB.
    world_bounds = target.compute_world_bounds()
    if world_bounds is None:
        raise ValueError(f"Target {target.name} has no bounding box")

    bbox_min, bbox_max = world_bounds

    # Create AABB box mesh for target.
    box_extents = bbox_max - bbox_min
    box_center = (bbox_min + bbox_max) / 2
    target_box_mesh = trimesh.creation.box(extents=box_extents)
    target_box_mesh.apply_translation(box_center)

    # Create collision managers.
    obj_matrix = rigid_transform_to_matrix(obj.transform)
    obj_manager = trimesh.collision.CollisionManager()
    for i, piece in enumerate(obj_collision_meshes):
        obj_manager.add_object(f"obj_{i}", piece, transform=obj_matrix)

    target_manager = trimesh.collision.CollisionManager()
    target_manager.add_object("target_aabb", target_box_mesh)

    # Check current distance (negative = penetrating, positive = separated).
    min_distance = target_manager.min_distance_other(obj_manager)

    # Margin is the desired gap between objects after snapping to prevent collision.
    # Small gap (1cm) prevents floating-point errors and collision detection mismatches.
    margin = cfg.snap_to_object.snap_margin_m
    total_movement = np.zeros(3)

    # Handle penetration case.
    if min_distance < 0:
        console_logger.info(f"Penetration detected: min_distance={min_distance:.3f}m")

        # Compute push-out direction based on object/target centers.
        obj_center = obj.transform.translation()
        target_center = (bbox_min + bbox_max) / 2
        pushout_direction = obj_center - target_center
        pushout_norm = np.linalg.norm(pushout_direction)

        if pushout_norm < DEGENERATE_VOLUME_THRESHOLD:
            # Objects at same position - push along +X.
            pushout_direction = np.array([1.0, 0.0, 0.0])
        else:
            pushout_direction = pushout_direction / pushout_norm

        # Push out until separated + margin.
        pushout_distance = abs(min_distance) + margin
        pushout_vector = pushout_direction * pushout_distance
        total_movement += pushout_vector

        console_logger.info(
            f"Pushed out {pushout_distance:.3f}m along {pushout_direction}"
        )

        # Update collision manager with new position.
        new_transform = RigidTransform(
            obj.transform.rotation(), obj.transform.translation() + pushout_vector
        )
        new_matrix = rigid_transform_to_matrix(new_transform)

        obj_manager = trimesh.collision.CollisionManager()
        for i, piece in enumerate(obj_collision_meshes):
            obj_manager.add_object(f"obj_{i}", piece, transform=new_matrix)

    # Compute snap distance using collision detection.
    # Binary search to find exact distance where objects touch.
    snap_direction = box_center - (obj.transform.translation() + total_movement)
    snap_norm = np.linalg.norm(snap_direction)

    if snap_norm < ZERO_DISTANCE_THRESHOLD:
        # Already at target center.
        return total_movement, np.linalg.norm(total_movement)

    snap_direction_unit = snap_direction / snap_norm

    # Use small steps to find collision point.
    step_size = cfg.snap_to_object.iterative_snap_step_m
    max_steps = int(snap_norm / step_size) + 100
    snap_distance = 0.0

    for i in range(max_steps):
        test_distance = (i + 1) * step_size
        if test_distance > snap_norm:
            snap_distance = snap_norm
            break

        test_pos = (
            obj.transform.translation()
            + total_movement
            + (snap_direction_unit * test_distance)
        )
        test_transform = RigidTransform(obj.transform.rotation(), test_pos)
        test_matrix = rigid_transform_to_matrix(test_transform)

        test_manager = trimesh.collision.CollisionManager()
        for j, piece in enumerate(obj_collision_meshes):
            test_manager.add_object(f"obj_{j}", piece, transform=test_matrix)

        test_distance_to_target = target_manager.min_distance_other(test_manager)

        if test_distance_to_target < margin:
            # Collision detected - back off by margin.
            snap_distance = max(0.0, test_distance - margin)
            break

    snap_vector = snap_direction_unit * snap_distance
    total_movement += snap_vector

    distance = np.linalg.norm(total_movement)
    console_logger.info(
        f"Snap complete: total_movement={distance:.3f}m "
        f"(pushout + snap={np.linalg.norm(total_movement - snap_vector):.3f}m "
        f"+ {np.linalg.norm(snap_vector):.3f}m)"
    )

    return total_movement, distance


def compute_snap_direction_mesh_to_mesh(
    obj: SceneObject, target: SceneObject, cfg: DictConfig
) -> np.ndarray:
    """Compute snap direction from obj to target using closest points on visual geometry.

    Args:
        obj: Object to move (must have geometry_path).
        target: Target object (must have geometry_path).
        cfg: Configuration with snap_to_object.max_sample_vertices setting.

    Returns:
        Unit direction vector from obj to target.

    Raises:
        ValueError: If geometry cannot be loaded.
    """
    start_time = time.time()

    # Load meshes.
    obj_mesh = trimesh.load(obj.geometry_path, force="mesh")
    target_mesh = trimesh.load(target.geometry_path, force="mesh")

    # Handle Scene objects (multiple meshes) by combining.
    if isinstance(obj_mesh, trimesh.Scene):
        meshes = [
            g for g in obj_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
        ]
        obj_mesh = trimesh.util.concatenate(meshes) if meshes else None
    if isinstance(target_mesh, trimesh.Scene):
        meshes = [
            g for g in target_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
        ]
        target_mesh = trimesh.util.concatenate(meshes) if meshes else None

    if not isinstance(obj_mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh from {obj.geometry_path}")
    if not isinstance(target_mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh from {target.geometry_path}")

    # Log mesh complexity for performance monitoring.
    console_logger.info(
        f"Computing snap direction: {obj.name} ({len(obj_mesh.vertices)} vertices) "
        f"→ {target.name} ({len(target_mesh.vertices)} vertices)"
    )

    # Convert meshes from Y-up (GLTF) to Z-up (Drake) before applying transforms.
    convert_mesh_yup_to_zup(obj_mesh)
    convert_mesh_yup_to_zup(target_mesh)

    # Apply runtime scale_factor (set by rescale operations).
    if obj.scale_factor != 1.0:
        obj_mesh.vertices *= obj.scale_factor
    if target.scale_factor != 1.0:
        target_mesh.vertices *= target.scale_factor

    # Transform meshes to world coordinates.
    obj_matrix = rigid_transform_to_matrix(obj.transform)
    target_matrix = rigid_transform_to_matrix(target.transform)

    obj_mesh.apply_transform(obj_matrix)
    target_mesh.apply_transform(target_matrix)

    # Downsample vertices to avoid O(n*m) memory exhaustion with high-poly meshes.
    max_sample_vertices = cfg.snap_to_object.max_sample_vertices
    obj_vertex_count = len(obj_mesh.vertices)
    if obj_vertex_count > max_sample_vertices:
        # Compute voxel pitch to achieve target vertex count.
        # Estimate: vertices ≈ (mesh_size / voxel_pitch)^3
        mesh_extents = obj_mesh.bounds[1] - obj_mesh.bounds[0]
        mesh_size = np.mean(mesh_extents)
        # Target: max_sample_vertices ≈ (mesh_size / pitch)^3
        # => pitch ≈ mesh_size / (max_sample_vertices)^(1/3)
        target_pitch = mesh_size / (max_sample_vertices ** (1 / 3))

        console_logger.info(
            f"Voxel downsampling {obj_vertex_count} vertices to ~{max_sample_vertices} "
            f"(pitch={target_pitch:.4f}m) for proximity query to avoid memory exhaustion"
        )

        # Create voxel-downsampled copy.
        downsampled_mesh = obj_mesh.copy()
        downsampled_mesh.merge_vertices()
        downsampled_mesh = downsampled_mesh.voxelized(pitch=target_pitch).marching_cubes

        sampled_points = downsampled_mesh.vertices
        console_logger.info(
            f"Downsampled to {len(sampled_points)} vertices "
            f"({len(sampled_points) / obj_vertex_count * 100:.1f}% of original)"
        )
    else:
        sampled_points = obj_mesh.vertices

    # Find closest points between meshes using downsampled vertices.
    closest_on_target, distances, _ = trimesh.proximity.closest_point(
        mesh=target_mesh, points=sampled_points
    )

    # Find vertex that's closest to target.
    min_idx = np.argmin(distances)
    closest_on_obj = sampled_points[min_idx]
    closest_on_target_pt = closest_on_target[min_idx]

    # Compute direction vector.
    direction = closest_on_target_pt - closest_on_obj
    distance = np.linalg.norm(direction)

    if distance < ZERO_DISTANCE_THRESHOLD:
        console_logger.warning("Objects are already touching, returning zero direction")
        return np.zeros(3)

    # Normalize to unit vector.
    direction_unit = direction / distance

    # Log computation time for performance monitoring.
    elapsed_time = time.time() - start_time
    console_logger.info(
        f"Snap direction computed in {elapsed_time:.2f}s (distance={distance:.3f}m)"
    )

    # Explicitly free mesh memory to prevent accumulation across multiple snaps.
    # Trimesh uses C++ libraries (e.g., FCL) that may not be immediately freed by
    # Python's garbage collector.
    del obj_mesh, target_mesh, closest_on_target, distances, sampled_points
    gc.collect()

    return direction_unit


def snap_with_iterative_collision_check(
    obj: SceneObject,
    target: SceneObject,
    direction: np.ndarray,
    cfg: DictConfig,
) -> tuple[np.ndarray, float]:
    """Snap object to target using iterative collision checking.

    Moves object step-by-step in the given direction until collision is detected.
    Uses trimesh CollisionManager with python-fcl for robust collision detection.
    This properly detects collisions between convex mesh pieces, avoiding the
    vertex projection problem where high vertices (e.g., table tops) block
    movement even when there's empty space underneath.

    Args:
        obj: Object to move (must have sdf_path with collision geometry).
        target: Target object (stays in place).
        direction: Unit direction vector in world coordinates.
        cfg: Configuration with snap_to_object.iterative_snap_step_m and
            snap_to_object.max_snap_distance_m.

    Returns:
        Tuple of (movement_vector, distance_moved).

    Raises:
        ValueError: If collision geometry cannot be loaded.
    """
    # Normalize direction.
    direction_norm = np.linalg.norm(direction)
    if direction_norm < ZERO_DISTANCE_THRESHOLD:
        console_logger.warning("Direction is zero, no movement needed")
        return np.zeros(3), 0.0
    direction_unit = direction / direction_norm

    # Load collision geometry for obj (applies SDF scale and runtime scale_factor).
    obj_collision_meshes = load_object_collision_geometry(obj)

    # Load collision geometry for target.
    if target.geometry_path and target.sdf_path:
        # Target has collision geometry (applies SDF scale and runtime scale_factor).
        target_collision_meshes = load_object_collision_geometry(target)
    else:
        # Target is a wall or object without mesh - create AABB box mesh.
        world_bounds = target.compute_world_bounds()
        if world_bounds is None:
            console_logger.warning(
                f"Target {target.name} has no bounds for collision check"
            )
            return np.zeros(3), 0.0

        bbox_min, bbox_max = world_bounds
        box_extents = bbox_max - bbox_min
        box_center = (bbox_min + bbox_max) / 2
        box_mesh = trimesh.creation.box(extents=box_extents)
        box_mesh.apply_translation(box_center)
        target_collision_meshes = [box_mesh]

    # Create collision manager for target (stays fixed).
    target_manager = trimesh.collision.CollisionManager()
    target_matrix = rigid_transform_to_matrix(target.transform)
    for i, piece in enumerate(target_collision_meshes):
        target_manager.add_object(f"target_{i}", piece, transform=target_matrix)

    # Create reusable collision manager for obj (will update transform each iteration).
    obj_manager = trimesh.collision.CollisionManager()

    # Iteratively move obj toward target.
    step_size = cfg.snap_to_object.iterative_snap_step_m
    max_snap_distance = cfg.snap_to_object.max_snap_distance_m
    max_iterations = int(max_snap_distance / step_size)
    total_movement = 0.0

    # Current position starts at obj's transform.
    current_position = obj.transform.translation().copy()

    # Precompute AABB projections for efficient "sliding through" safety check.
    # This prevents runaway snapping when object fits under target (e.g., chair under table).
    obj_matrix = rigid_transform_to_matrix(obj.transform)
    obj_vertices_local = np.vstack([m.vertices for m in obj_collision_meshes])
    obj_vertices_homogeneous = np.hstack(
        [obj_vertices_local, np.ones((len(obj_vertices_local), 1))]
    )
    obj_vertices_world = (obj_matrix @ obj_vertices_homogeneous.T).T[:, :3]

    # Project object vertices onto movement direction to find both edges.
    obj_projections = np.dot(obj_vertices_world, direction_unit)
    obj_max_projection = np.max(
        obj_projections
    )  # Leading edge (furthest along direction).
    obj_min_projection = np.min(
        obj_projections
    )  # Trailing edge (closest along direction).
    current_position_projection = np.dot(current_position, direction_unit)
    obj_leading_edge_offset = obj_max_projection - current_position_projection
    obj_trailing_edge_offset = obj_min_projection - current_position_projection

    # Compute target's leading edge projection (closest point to object along direction).
    target_vertices_local = np.vstack([m.vertices for m in target_collision_meshes])
    target_vertices_homogeneous = np.hstack(
        [target_vertices_local, np.ones((len(target_vertices_local), 1))]
    )
    target_vertices_world = (target_matrix @ target_vertices_homogeneous.T).T[:, :3]
    target_projections = np.dot(target_vertices_world, direction_unit)
    target_leading_edge = np.min(target_projections)

    console_logger.debug(
        f"AABB safety check setup: obj_leading_edge_offset={obj_leading_edge_offset:.3f}m, "
        f"obj_trailing_edge_offset={obj_trailing_edge_offset:.3f}m, "
        f"target_leading_edge={target_leading_edge:.3f}m"
    )

    for i in range(max_iterations):
        # Move position by one step.
        current_position += direction_unit * step_size
        total_movement += step_size

        # AABB-based safety check: has object's trailing edge passed target's leading edge?
        # This catches "sliding through" cases (e.g., chair under table) efficiently.
        current_position_projection = np.dot(current_position, direction_unit)
        object_trailing_edge = current_position_projection + obj_trailing_edge_offset

        if object_trailing_edge >= target_leading_edge:
            # Object passed through target - stop to prevent runaway snapping.
            total_movement -= step_size
            total_movement = max(0.0, total_movement)
            movement_vector = direction_unit * total_movement

            console_logger.info(
                f"Iterative snap: AABB safety check triggered at step {i + 1}, "
                f"object_trailing_edge={object_trailing_edge:.3f}m >= "
                f"target_leading_edge={target_leading_edge:.3f}m, "
                f"final distance={total_movement:.3f}m"
            )

            # Cleanup before early return.
            del (
                obj_manager,
                target_manager,
                obj_collision_meshes,
                target_collision_meshes,
            )
            gc.collect()

            return movement_vector, total_movement

        # Create transform for current obj position.
        current_transform = RigidTransform(obj.transform.rotation(), current_position)
        current_matrix = rigid_transform_to_matrix(current_transform)

        # Update collision manager with obj at current position.
        # Clear previous iteration's objects and add with new transform.
        obj_manager._objs.clear()  # Direct clear for performance.
        for j, piece in enumerate(obj_collision_meshes):
            obj_manager.add_object(f"obj_{j}", piece, transform=current_matrix)

        # Check collision using min_distance_other.
        # Returns negative if penetrating, positive if separated.
        min_distance = target_manager.min_distance_other(obj_manager)

        # Collision detection threshold: half step size provides good accuracy.
        # We've moved one full step, so checking within half step catches collision
        # before significant penetration while avoiding false positives from discretization.
        collision_threshold = step_size * 0.5
        if min_distance < collision_threshold:
            # Collision detected. Revert last step and subtract margin.
            total_movement -= step_size
            total_movement = max(0.0, total_movement)
            movement_vector = direction_unit * total_movement

            console_logger.info(
                f"Iterative snap: collision detected at step {i + 1}, "
                f"min_distance={min_distance:.4f}m, "
                f"final distance={total_movement:.3f}m"
            )

            # Cleanup before early return.
            del (
                obj_manager,
                target_manager,
                obj_collision_meshes,
                target_collision_meshes,
            )
            gc.collect()

            return movement_vector, total_movement

    # Max iterations reached without collision (rare but possible).
    console_logger.warning(
        f"Iterative snap: reached max iterations ({max_iterations}) without "
        f"collision, distance={total_movement:.3f}m"
    )
    movement_vector = direction_unit * total_movement

    # Cleanup collision managers and mesh data.
    del obj_manager, target_manager, obj_collision_meshes, target_collision_meshes
    gc.collect()

    return movement_vector, total_movement


def resolve_collision_if_penetrating(
    obj: SceneObject,
    target: SceneObject,
    cfg: DictConfig,
    wall_normals: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Resolve collision between two objects using conservative AABB push-out.

    Uses conservative AABB approach (treats object as circular by using max dimension)
    to ensure rotation won't reintroduce collision after push-out.

    Args:
        obj: Object to move (must have SDF path with collision geometry).
        target: Target object (stays in place).
        cfg: Configuration with snap_to_object.snap_margin_m for gap size.
        wall_normals: Dictionary mapping wall names to 2D normal vectors.
            Required when target is a wall.

    Returns:
        Movement vector applied to separate objects (zero vector if no collision).
    """
    console_logger.info(f"Checking collision: {obj.name} vs {target.name}")

    # Load collision geometry vertices in world coordinates.
    if not obj.sdf_path and not obj.geometry_path:
        console_logger.info(
            f"Object {obj.name} has no geometry, skipping collision check"
        )
        return np.zeros(3)

    try:
        obj_collision_meshes = load_object_collision_geometry(obj)
    except ValueError:
        console_logger.info(f"No collision geometry for {obj.name}, skipping")
        return np.zeros(3)

    obj_vertices_local = np.vstack([m.vertices for m in obj_collision_meshes])
    transform_matrix = rigid_transform_to_matrix(obj.transform)
    obj_vertices_homogeneous = np.hstack(
        [obj_vertices_local, np.ones((len(obj_vertices_local), 1))]
    )
    obj_vertices_world = (transform_matrix @ obj_vertices_homogeneous.T).T[:, :3]

    # Compute conservative AABB for object (circular treatment).
    obj_bbox_min_world = obj_vertices_world.min(axis=0)
    obj_bbox_max_world = obj_vertices_world.max(axis=0)
    obj_width = obj_bbox_max_world[0] - obj_bbox_min_world[0]
    obj_depth = obj_bbox_max_world[1] - obj_bbox_min_world[1]
    obj_max_dim = max(obj_width, obj_depth)

    # Create conservative square bbox.
    obj_center = (obj_bbox_min_world[:2] + obj_bbox_max_world[:2]) / 2.0
    obj_bbox_min_conservative = np.array(
        [
            obj_center[0] - obj_max_dim / 2.0,
            obj_center[1] - obj_max_dim / 2.0,
            obj_bbox_min_world[2],
        ]
    )
    obj_bbox_max_conservative = np.array(
        [
            obj_center[0] + obj_max_dim / 2.0,
            obj_center[1] + obj_max_dim / 2.0,
            obj_bbox_max_world[2],
        ]
    )

    console_logger.info(
        f"Conservative AABB: {obj.name} width={obj_width:.3f}m, "
        f"depth={obj_depth:.3f}m, max_dim={obj_max_dim:.3f}m (treating as circular)"
    )

    # Get target's world-space AABB.
    target_bounds = target.compute_world_bounds()
    if target_bounds is None:
        console_logger.info(
            f"Target {target.name} has no bounds, skipping collision check"
        )
        return np.zeros(3)
    target_bbox_min, target_bbox_max = target_bounds

    # Check for AABB overlap in XY plane.
    overlap_x = min(obj_bbox_max_conservative[0], target_bbox_max[0]) - max(
        obj_bbox_min_conservative[0], target_bbox_min[0]
    )
    overlap_y = min(obj_bbox_max_conservative[1], target_bbox_max[1]) - max(
        obj_bbox_min_conservative[1], target_bbox_min[1]
    )

    console_logger.info(
        f"AABB overlap check: {obj.name} vs {target.name}: "
        f"overlap_x={overlap_x:.4f}m, overlap_y={overlap_y:.4f}m"
    )

    # If no overlap in either axis, no collision.
    if overlap_x <= 0 or overlap_y <= 0:
        console_logger.info(
            f"No collision between {obj.name} and {target.name}, skipping resolution"
        )
        return np.zeros(3)

    console_logger.info(
        f"Collision detected: overlap_x={overlap_x:.3f}m, overlap_y={overlap_y:.3f}m"
    )

    # Determine push-out direction and distance.
    # Margin is the desired gap between objects after push-out to prevent re-collision.
    # Small gap (1cm) prevents floating-point errors and collision detection mismatches.
    margin = cfg.snap_to_object.snap_margin_m

    if target.object_type == ObjectType.WALL and wall_normals is not None:
        # Wall: push along wall normal (perpendicular to wall).
        wall_normal_2d = wall_normals.get(target.name)
        if wall_normal_2d is None:
            console_logger.warning(
                f"Wall {target.name} not found in wall_normals, "
                f"falling back to axis-based push-out"
            )
            # Fallback to axis-based push-out.
            if overlap_x < overlap_y:
                obj_center_x = (obj_bbox_min_world[0] + obj_bbox_max_world[0]) / 2.0
                target_center_x = (target_bbox_min[0] + target_bbox_max[0]) / 2.0
                direction = 1.0 if obj_center_x > target_center_x else -1.0
                pushout_distance = overlap_x + margin
                pushout_vector = np.array([direction * pushout_distance, 0.0, 0.0])
            else:
                obj_center_y = (obj_bbox_min_world[1] + obj_bbox_max_world[1]) / 2.0
                target_center_y = (target_bbox_min[1] + target_bbox_max[1]) / 2.0
                direction = 1.0 if obj_center_y > target_center_y else -1.0
                pushout_distance = overlap_y + margin
                pushout_vector = np.array([0.0, direction * pushout_distance, 0.0])
        else:
            # Use wall normal for push-out direction.
            # Compute penetration depth: project overlap onto wall normal.
            # For simplicity, use max of overlap_x and overlap_y.
            pushout_distance = max(overlap_x, overlap_y) + margin
            push_direction = np.array([wall_normal_2d[0], wall_normal_2d[1], 0.0])
            pushout_vector = push_direction * pushout_distance

            console_logger.info(
                f"Wall push-out: distance={pushout_distance:.3f}m along wall normal"
            )
    else:
        # Object: push along minimum overlap axis.
        if overlap_x < overlap_y:
            # Push in X direction.
            obj_center_x = (obj_bbox_min_world[0] + obj_bbox_max_world[0]) / 2.0
            target_center_x = (target_bbox_min[0] + target_bbox_max[0]) / 2.0
            direction = 1.0 if obj_center_x > target_center_x else -1.0

            pushout_distance = overlap_x + margin
            pushout_vector = np.array([direction * pushout_distance, 0.0, 0.0])
            console_logger.info(
                f"Pushed {obj.name} out by {pushout_distance:.3f}m in X direction"
            )
        else:
            # Push in Y direction.
            obj_center_y = (obj_bbox_min_world[1] + obj_bbox_max_world[1]) / 2.0
            target_center_y = (target_bbox_min[1] + target_bbox_max[1]) / 2.0
            direction = 1.0 if obj_center_y > target_center_y else -1.0

            pushout_distance = overlap_y + margin
            pushout_vector = np.array([0.0, direction * pushout_distance, 0.0])
            console_logger.info(
                f"Pushed {obj.name} out by {pushout_distance:.3f}m in Y direction"
            )

    # Apply movement to object.
    old_pos = obj.transform.translation()
    new_pos = old_pos + pushout_vector
    obj.transform = RigidTransform(R=obj.transform.rotation(), p=new_pos)

    console_logger.info(
        f"Collision resolved: moved {obj.name} by {np.linalg.norm(pushout_vector):.3f}m"
    )

    return pushout_vector


def snap_mesh_to_aabb_along_axis(
    obj: SceneObject,
    target: SceneObject,
    axis_world: np.ndarray,
    cfg: DictConfig,
) -> tuple[np.ndarray, float]:
    """Snap object to target by moving along a specific axis direction.

    Projects collision geometry onto the given axis and moves object to eliminate gap.
    This preserves the facing relationship by constraining movement to the facing axis.

    Args:
        obj: Object to move.
        target: Target object (stays in place).
        axis_world: Unit direction vector in world coordinates (e.g., object's +Y axis).
        cfg: Configuration with snap_to_object.snap_margin_m for gap size.

    Returns:
        Tuple of (movement_vector, distance) where distance is the gap along axis.
    """
    # Normalize axis.
    axis_unit = axis_world / np.linalg.norm(axis_world)

    # Load collision vertices in world coordinates.
    obj_vertices = get_collision_vertices_world(obj)

    # For target, use mesh vertices if available, otherwise use AABB.
    if target.geometry_path:
        target_vertices = get_collision_vertices_world(target)
    else:
        # Target is a wall or object without mesh - use AABB corners.
        world_bounds = target.compute_world_bounds()
        if world_bounds is None:
            console_logger.warning(f"Target {target.name} has no bounds for axis snap")
            return np.zeros(3), 0.0
        bbox_min, bbox_max = world_bounds
        # Generate 8 AABB corners.
        target_vertices = np.array(
            [
                [bbox_min[0], bbox_min[1], bbox_min[2]],
                [bbox_max[0], bbox_min[1], bbox_min[2]],
                [bbox_min[0], bbox_max[1], bbox_min[2]],
                [bbox_max[0], bbox_max[1], bbox_min[2]],
                [bbox_min[0], bbox_min[1], bbox_max[2]],
                [bbox_max[0], bbox_min[1], bbox_max[2]],
                [bbox_min[0], bbox_max[1], bbox_max[2]],
                [bbox_max[0], bbox_max[1], bbox_max[2]],
            ]
        )

    # Project vertices onto axis.
    obj_projections = np.dot(obj_vertices, axis_unit)
    target_projections = np.dot(target_vertices, axis_unit)

    # Find the extremes along the axis.
    # Object's "front" is the furthest vertex along the axis direction.
    # Target's "back" (nearest side) is the closest vertex along axis direction.
    obj_max_projection = np.max(obj_projections)
    target_min_projection = np.min(target_projections)

    # Gap along axis (positive = separated, negative = overlapping).
    gap = target_min_projection - obj_max_projection

    # Subtract margin to leave small gap between objects after snapping.
    # Small gap prevents floating-point errors and collision detection mismatches.
    margin = cfg.snap_to_object.snap_margin_m
    distance_to_move = gap - margin

    # Movement vector along axis.
    movement_vector = axis_unit * distance_to_move

    console_logger.info(
        f"Axis snap: obj_max={obj_max_projection:.3f}, "
        f"target_min={target_min_projection:.3f}, gap={gap:.3f}m, "
        f"margin={margin:.3f}m, move={distance_to_move:.3f}m"
    )

    return movement_vector, abs(distance_to_move)


def select_and_execute_snap_algorithm(
    obj: SceneObject,
    target: SceneObject,
    orientation: str,
    orientation_applied: bool,
    object_id: str,
    target_id: str,
    cfg: DictConfig,
) -> tuple[np.ndarray, float] | str:
    """Select and execute the appropriate snapping algorithm.

    Args:
        obj: Object to snap.
        target: Target to snap to.
        orientation: Orientation mode ("toward", "away", or "none").
        orientation_applied: Whether orientation was applied.
        object_id: Object ID string (for error messages).
        target_id: Target ID string (for error messages).
        cfg: Configuration object.

    Returns:
        Tuple of (movement_vector, distance) if successful,
        or error JSON string if failed.
    """
    start_time = time.time()
    try:
        if orientation_applied:
            # Axis-constrained snapping: move only along facing direction.
            # This preserves the facing relationship established.

            # Compute axis direction from object's current rotation.
            # For "toward": +Y axis (forward).
            # For "away": -Y axis (backward).
            local_axis = np.array([0.0, 1.0 if orientation == "toward" else -1.0, 0.0])
            axis_world = obj.transform.rotation() @ local_axis

            if target.geometry_path and target.sdf_path:
                # Target has mesh geometry: use iterative collision checking.
                movement_vector, distance = snap_with_iterative_collision_check(
                    obj=obj, target=target, direction=axis_world, cfg=cfg
                )
                algorithm = "iterative-axis-constrained"
                console_logger.info(
                    f"Using iterative axis-constrained snapping (mesh target, "
                    f"orientation='{orientation}')"
                )
            else:
                # Target is AABB (wall): use fast single-step approach.
                movement_vector, distance = snap_mesh_to_aabb_along_axis(
                    obj=obj, target=target, axis_world=axis_world, cfg=cfg
                )
                algorithm = "axis-constrained"
                console_logger.info(
                    f"Using axis-constrained snapping (AABB target, "
                    f"orientation='{orientation}')"
                )
        else:
            # Closest-point snapping.
            if obj.geometry_path and target.geometry_path:
                # Use iterative mesh-to-mesh algorithm.
                # Compute direction from closest points on visual geometry.
                direction = compute_snap_direction_mesh_to_mesh(
                    obj=obj, target=target, cfg=cfg
                )
                # Use iterative collision checking to find safe snap distance.
                movement_vector, distance = snap_with_iterative_collision_check(
                    obj=obj, target=target, direction=direction, cfg=cfg
                )
                algorithm = "iterative-mesh-to-mesh"
            elif obj.geometry_path:
                # Use mesh-to-AABB algorithm.
                movement_vector, distance = snap_mesh_to_aabb(obj, target, cfg)
                algorithm = "mesh-to-AABB"
            else:
                # Object missing geometry.
                return SnapToObjectResult(
                    success=False,
                    message=f"{obj.name} missing geometry_path - cannot snap",
                    object_id=object_id,
                    target_id=target_id,
                    error_type=FurnitureErrorType.INVALID_POSITION,
                    suggested_action=(
                        "Snap requires 3D geometry. Use a different object that has "
                        "geometry"
                    ),
                ).to_json()

            console_logger.info(f"Using closest-point snapping (orientation='none')")
    except Exception as e:
        console_logger.error(f"Error computing snap: {e}")
        return SnapToObjectResult(
            success=False,
            message=f"Failed to compute snap: {str(e)}",
            object_id=object_id,
            target_id=target_id,
            suggested_action="Check object and target IDs are valid and have geometry",
        ).to_json()

    console_logger.info(
        f"Snap computation ({algorithm}): distance={distance:.3f}m, "
        f"computed in {time.time() - start_time:.2f}s"
    )

    return (movement_vector, distance)
