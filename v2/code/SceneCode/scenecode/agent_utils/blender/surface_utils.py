"""Utility functions for multi-surface rendering support."""

import logging
import math

from pathlib import Path

import bpy
import matplotlib.pyplot as plt
import numpy as np

from mathutils import Vector
from PIL import Image, ImageDraw

from scenecode.agent_utils.blender.annotations import load_annotation_font

console_logger = logging.getLogger(__name__)


def is_point_occluded(
    camera_obj: "bpy.types.Object",
    target: Vector,
    surface_id: str,
    debug: bool = False,
) -> bool:
    """Check if a point is occluded from the camera by other geometry.

    Uses a forward raycast with distance checking: shoots a ray from the camera
    toward the target, then checks if it hits geometry BEFORE reaching the target.
    This correctly handles edge cases where the target is at or near furniture
    surface boundaries.

    Args:
        camera_obj: Blender camera object.
        target: Target point in world coordinates.
        surface_id: Surface identifier for debug logging.
        debug: If True, log detailed raycast information.

    Returns:
        True if point is occluded by geometry, False otherwise.
    """
    scene = bpy.context.scene
    camera_loc = camera_obj.location

    # Direction from camera toward target.
    direction = target - camera_loc
    distance_to_target = direction.length

    if distance_to_target <= 0:
        return True  # Camera at target position - consider occluded.
    direction.normalize()

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Temporarily hide overlay objects for accurate raycast.
    overlay_objects = [
        obj
        for obj in bpy.data.objects
        if obj.name.startswith("Overlay_") and obj.type == "MESH"
    ]
    original_states = {}
    for obj in overlay_objects:
        original_states[obj.name] = obj.hide_viewport
        obj.hide_viewport = True

    depsgraph.update()

    # Raycast from camera toward target, extending slightly past to ensure we
    # don't miss due to floating-point precision.
    result, hit_loc, _norm, _idx, hit_object, _matrix = scene.ray_cast(
        depsgraph,
        origin=camera_loc,
        direction=direction,
        distance=distance_to_target + 0.01,
    )

    # Restore overlay visibility.
    for obj in overlay_objects:
        obj.hide_viewport = original_states[obj.name]
    depsgraph.update()

    if not result:
        # Ray didn't hit anything - target is visible.
        if debug:
            console_logger.debug(
                f"Raycast {surface_id}: no hit, target visible, " f"target={target}"
            )
        return False

    # Check if hit was significantly before the target position.
    # If geometry blocks the ray before reaching the target, it's occluded.
    hit_distance = (hit_loc - camera_loc).length
    occlusion_tolerance = 0.02  # 2cm tolerance for surface boundary cases.

    is_occluded = hit_distance < distance_to_target - occlusion_tolerance

    if debug:
        hit_name = hit_object.name if hit_object else "None"
        console_logger.debug(
            f"Raycast {surface_id}: hit={hit_name}, hit_dist={hit_distance:.3f}, "
            f"target_dist={distance_to_target:.3f}, occluded={is_occluded}, "
            f"target={target}, hit_loc={hit_loc}"
        )

    return is_occluded


def find_best_label_position(
    convex_hull_vertices: np.ndarray,
    camera_obj: "bpy.types.Object",
    img_width: int,
    img_height: int,
    grid_size: int = 5,
    surface_id: str = "",
) -> tuple[float, float] | None:
    """Find best visible position for a surface label using grid sampling.

    Generates a grid of candidate points on the surface, projects them all
    to image coordinates, filters by visibility and occlusion, and returns
    the most central non-occluded point.

    Args:
        convex_hull_vertices: Nx3 array of surface vertices in world coords.
        camera_obj: Blender camera object for projection.
        img_width: Rendered image width in pixels.
        img_height: Rendered image height in pixels.
        grid_size: Number of grid points per axis (grid_size x grid_size total).
        surface_id: Surface identifier for occlusion checking (allows own overlay).

    Returns:
        (pixel_x, pixel_y) of best visible position, or None if no visible points.
    """
    # Get surface bounding box.
    x_min, y_min, z_min = convex_hull_vertices.min(axis=0)
    x_max, y_max, z_max = convex_hull_vertices.max(axis=0)
    surface_z = z_max  # Use top Z of surface.

    # Debug logging for occlusion investigation.
    console_logger.debug(
        f"find_best_label_position {surface_id}: bbox=({x_min:.3f},{y_min:.3f}) to "
        f"({x_max:.3f},{y_max:.3f}), z={surface_z:.3f}, camera={camera_obj.location}"
    )

    # Generate grid of candidate points on surface.
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack(
        [xx.ravel(), yy.ravel(), np.full(grid_size**2, surface_z)]
    )

    # Get camera matrices.
    scene = bpy.context.scene
    world_to_cam = np.array(camera_obj.matrix_world.inverted())
    proj_matrix = np.array(
        camera_obj.calc_matrix_camera(
            bpy.context.evaluated_depsgraph_get(),
            x=scene.render.resolution_x,
            y=scene.render.resolution_y,
        )
    )

    # Transform to homogeneous coordinates (Nx4).
    points_h = np.hstack([grid_points, np.ones((len(grid_points), 1))])

    # World to camera transform (vectorized).
    cam_coords = (world_to_cam @ points_h.T).T  # Nx4

    # Filter points behind camera (camera Z > 0 means behind in Blender).
    in_front_mask = cam_coords[:, 2] < 0

    if not in_front_mask.any():
        return None

    # Project to NDC (vectorized).
    ndc_coords = (proj_matrix @ cam_coords.T).T  # Nx4

    # Perspective divide.
    w = ndc_coords[:, 3:4]
    w = np.where(w != 0, w, 1)  # Avoid division by zero.
    ndc_normalized = ndc_coords / w

    # NDC to pixel coordinates.
    pixel_x = (ndc_normalized[:, 0] + 1.0) * 0.5 * img_width
    pixel_y = (1.0 - ndc_normalized[:, 1]) * 0.5 * img_height
    pixel_coords = np.column_stack([pixel_x, pixel_y])

    # Filter by bounds.
    in_bounds_mask = (
        (pixel_x >= 0)
        & (pixel_x <= img_width)
        & (pixel_y >= 0)
        & (pixel_y <= img_height)
    )

    # Combined visibility mask.
    visible_mask = in_front_mask & in_bounds_mask

    if not visible_mask.any():
        return None

    # Get center of surface in pixel space (for finding most central visible point).
    center_world = convex_hull_vertices.mean(axis=0)
    center_h = np.append(center_world, 1.0)
    center_cam = world_to_cam @ center_h
    center_ndc = proj_matrix @ center_cam
    if center_ndc[3] != 0:
        center_ndc = center_ndc / center_ndc[3]
    center_pixel = np.array(
        [
            (center_ndc[0] + 1.0) * 0.5 * img_width,
            (1.0 - center_ndc[1]) * 0.5 * img_height,
        ]
    )

    # Get visible candidate indices sorted by distance to center (most central first).
    visible_indices = np.where(visible_mask)[0]
    visible_pixels = pixel_coords[visible_indices]
    distances = np.linalg.norm(visible_pixels - center_pixel, axis=1)
    sorted_order = np.argsort(distances)

    # Check occlusion for all candidates and collect non-occluded ones.
    non_occluded_points: list[tuple[int, tuple[float, float]]] = []
    occluded_count = 0

    for order_idx in sorted_order:
        candidate_idx = visible_indices[order_idx]
        point_world = Vector(grid_points[candidate_idx].tolist())

        is_occluded = is_point_occluded(camera_obj, point_world, surface_id)
        if is_occluded:
            occluded_count += 1
        else:
            non_occluded_points.append((order_idx, tuple(pixel_coords[candidate_idx])))

    total_checked = len(sorted_order)
    visible_count = len(non_occluded_points)

    console_logger.debug(
        f"find_best_label {surface_id}: {occluded_count}/{total_checked} occluded, "
        f"{visible_count} visible"
    )

    # Require at least 40% of points to be visible for a surface to be considered
    # visible. This prevents edge artifacts (points shooting past mesh edges or
    # through gaps) from incorrectly marking occluded surfaces as visible.
    # With 25 grid points, 40% = 10 points minimum.
    min_visible_ratio = 0.40
    min_visible_count = max(3, int(total_checked * min_visible_ratio))

    if visible_count < min_visible_count:
        console_logger.debug(
            f"find_best_label_position {surface_id}: only {visible_count}/{total_checked} "
            f"points visible (need {min_visible_count}), treating as occluded"
        )
        return None

    # Return the most central non-occluded point.
    _, best_pixel = non_occluded_points[0]
    console_logger.debug(
        f"find_best_label_position {surface_id}: found {visible_count} visible points, "
        f"using most central at pixel={best_pixel}"
    )
    return best_pixel


def generate_surface_colors(surface_ids: list[str]) -> dict[str, tuple[int, int, int]]:
    """Generate unique colors for each support surface using matplotlib tab20.

    Args:
        surface_ids: List of surface ID strings.

    Returns:
        Dict mapping surface_id to RGB color tuple (0-255 range).
    """
    cmap = plt.get_cmap("tab20")
    colors = {}
    for idx, surface_id in enumerate(surface_ids):
        # Get color from tab20 colormap (cycles if more than 20 surfaces).
        rgba = cmap(idx % 20)
        # Convert from 0-1 float to 0-255 int, drop alpha channel.
        rgb = tuple(int(c * 255) for c in rgba[:3])
        colors[surface_id] = rgb
    return colors


def generate_multi_surface_views(support_surfaces: list[dict]) -> list[dict]:
    """Generate one top view per support surface for multi-surface furniture.

    Args:
        support_surfaces: List of support surface dicts, each containing surface_id,
            corners, and optional convex_hull_vertices.

    Returns:
        List of view dictionaries, one top view per surface with surface_id embedded.
    """
    views = []
    for idx, surface in enumerate(support_surfaces):
        surface_id = surface.get("surface_id", f"surface_{idx}")
        # Store surface data for use during rendering (coordinate markers, labels).
        view = {
            "name": f"{idx}_top_{surface_id}",
            "direction": Vector((0, 0, 1)),
            "is_side": False,
            "surface_data": surface,  # Store for coordinate marker filtering.
        }
        views.append(view)
    return views


def generate_angled_drawer_view(
    surface: dict,
    joint_name: str,
    drawer_direction: list[float] | None = None,
    view_index: int = 0,
) -> dict:
    """Generate an angled view for looking into a drawer interior.

    Creates a view at ~45° from top, angled to see inside the drawer.
    Camera is positioned opposite to drawer opening direction to look into it.

    Args:
        surface: Support surface dict containing surface_id, corners, etc.
        joint_name: Name of the joint controlling this drawer.
        drawer_direction: Direction the drawer moves when opening (from FK delta).
            If None, uses default front-opening assumption.
        view_index: Index for naming the view.

    Returns:
        View dictionary with angled direction for drawer interior visibility.
    """
    surface_id = surface.get("surface_id", f"surface_{view_index}")

    if drawer_direction is not None:
        # Camera positioned on SAME side as drawer opening to look INTO it.
        # If drawer slides toward +Y (front), camera is at +Y looking into the opening.
        dx, dy, _ = drawer_direction

        # Compute horizontal direction magnitude.
        horiz_mag = math.sqrt(dx * dx + dy * dy)

        if horiz_mag > 0.01:  # Drawer has horizontal movement.
            # Normalize horizontal component (same direction as drawer opening).
            cam_x = dx / horiz_mag
            cam_y = dy / horiz_mag

            # Add elevation: ~45° from horizontal (0.7 Z component).
            # Scale horizontal to maintain unit vector after adding Z.
            horiz_scale = 0.7  # cos(45°) ≈ 0.7
            direction = Vector((cam_x * horiz_scale, cam_y * horiz_scale, 0.7))
            direction.normalize()
        else:
            # Fallback for vertical-only movement (unusual for drawers).
            direction = Vector((0.0, 0.7, 0.7)).normalized()
    else:
        # Default: assume front-opening drawer (opens toward +Y in Drake coords).
        # Camera at +Y looking in, with 45° elevation.
        direction = Vector((0.0, 0.7, 0.7)).normalized()

    return {
        "name": f"drawer_{joint_name}_{surface_id}",
        "direction": direction,
        "is_side": False,  # Treat as top-like for camera setup.
        "surface_data": surface,
        "is_drawer_view": True,  # Flag for special handling.
    }


def add_surface_id_label(
    image_path: Path,
    surface_id: str,
    surface_colors: dict[str, tuple[int, int, int]],
) -> None:
    """Add surface ID label to rendered image.

    Args:
        image_path: Path to the image file.
        surface_id: Surface identifier to display.
        surface_colors: Mapping of surface IDs to RGB colors.
    """
    try:
        # Load image.
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Load font using same logic as object labels.
        # Divisor 25 gives ~41pt on 1024px, ~20pt on 512px for readable labels.
        font = load_annotation_font(img.width, base_font_size_divisor=25)

        # Add label in top-right corner with background.
        label_text = f"Surface ID: {surface_id}"

        # Get text bounding box for background.
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position in top-right corner with padding.
        padding = 5
        x = img.width - text_width - padding * 2
        y = padding

        # Draw background with surface color if available, else default blue.
        if surface_colors and surface_id in surface_colors:
            bg_color = surface_colors[surface_id]
        else:
            bg_color = (77, 153, 255)  # Default blue fallback.

        draw.rectangle(
            [
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding,
            ],
            fill=bg_color,
        )

        # Draw text in white.
        draw.text((x, y), label_text, fill=(255, 255, 255), font=font)

        # Save image.
        img.save(image_path)
    except Exception as e:
        console_logger.warning(f"Failed to add surface ID label: {e}")


def add_surface_labels_to_side_view(
    image_path: Path,
    camera_obj: bpy.types.Object,
    support_surfaces: list[dict],
    surface_colors: dict[str, tuple[int, int, int]],
) -> None:
    """Add surface ID labels to a side view for all visible surfaces.

    Uses grid-based sampling to find the best visible position for each label.
    This is more robust than single-point projection as it can find visible
    positions even when the surface center is occluded or out of bounds.

    Args:
        image_path: Path to the rendered side view image.
        camera_obj: Blender camera object for projection.
        support_surfaces: List of all support surfaces.
        surface_colors: Mapping of surface IDs to RGB colors.
    """
    if support_surfaces is None or len(support_surfaces) == 0:
        return

    try:
        # Load image.
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Load font using same logic as object labels.
        font = load_annotation_font(
            img.width, base_font_size_divisor=50, min_font_size=5
        )

        # Process each surface.
        for surface_data in support_surfaces:
            surface_id = surface_data.get("surface_id", "unknown")

            # Get surface vertices for grid-based position finding.
            convex_hull_vertices = surface_data.get("convex_hull_vertices")
            if convex_hull_vertices is None or len(convex_hull_vertices) == 0:
                # Fallback to bounding box corners if mesh data not available.
                corners = surface_data.get("corners")
                if corners is None or len(corners) != 8:
                    console_logger.warning(
                        f"Surface {surface_id} missing geometry, skipping label"
                    )
                    continue
                convex_hull_vertices = corners

            vertices_array = np.array(convex_hull_vertices)

            # Find best visible label position using grid sampling.
            best_position = find_best_label_position(
                convex_hull_vertices=vertices_array,
                camera_obj=camera_obj,
                img_width=img.width,
                img_height=img.height,
                grid_size=5,  # 5x5 = 25 candidate points.
                surface_id=surface_id,
            )

            if best_position is None:
                console_logger.info(
                    f"Surface {surface_id}: no visible label positions found"
                )
                continue

            pixel_x, pixel_y = best_position

            # Draw label at projected position.
            label_text = surface_id

            # Get text bounding box.
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Center text on projected point.
            x = int(pixel_x - text_width / 2)
            y = int(pixel_y - text_height / 2)

            # Draw background with surface color if available, else default blue.
            if surface_colors and surface_id in surface_colors:
                bg_color = surface_colors[surface_id]
            else:
                bg_color = (77, 153, 255)  # Default blue fallback.

            padding = 5
            draw.rectangle(
                [
                    x - padding,
                    y - padding,
                    x + text_width + padding,
                    y + text_height + padding,
                ],
                fill=bg_color,
            )

            # Draw text in white.
            draw.text((x, y), label_text, fill=(255, 255, 255), font=font)
            console_logger.info(f"Surface {surface_id}: label placed at ({x}, {y})")

        # Save annotated image.
        img.save(image_path)

    except Exception as e:
        console_logger.warning(f"Failed to add surface labels to side view: {e}")
