"""Mixin class for coordinate grid functionality in BlenderRenderer.

This mixin encapsulates all coordinate grid and marker management methods for
rendering, including grid generation, marker filtering, and debug visualization.
"""

import logging

from pathlib import Path

import bpy
import numpy as np

from mathutils import Vector

from scenecode.agent_utils.blender.camera_utils import get_pixel_coordinates
from scenecode.agent_utils.blender.scene_utils import get_floor_bounds
from scenecode.utils.geometry_utils import compute_ordered_convex_hull_vertices_2d

console_logger = logging.getLogger(__name__)

# Coordinate grid constants.
CONVEX_HULL_MARKER_BUFFER_METERS = 0.01  # 1cm buffer for boundary markers.


class CoordinateGridMixin:
    """Mixin providing coordinate grid and marker methods for BlenderRenderer.

    This mixin contains methods for generating visual reference marks, filtering
    markers by convex hull containment, and debug visualization.

    Attributes:
        _surface_corners: Corner points defining the current surface bounds.
        _client_objects: Blender collection containing imported scene objects.
        _grid_origin: Origin point for the coordinate grid.
        _grid_axis_x: X-axis direction for the coordinate grid.
        _grid_axis_y: Y-axis direction for the coordinate grid.
        _grid_axis_z: Z-axis direction for the coordinate grid.
        _current_convex_hull: Current convex hull vertices for marker filtering.
    """

    def _get_visual_marks(
        self,
        scene: bpy.types.Scene,
        camera_obj: bpy.types.Object,
        is_top_view: bool = False,
        is_drawer_view: bool = False,
        ceiling_height: float | None = None,
        room_bounds: tuple[float, float, float, float] | None = None,
    ) -> dict[tuple[float, float], tuple[int, int]]:
        """Generate visual reference marks at grid intersections.

        Creates small sphere markers at grid intersection points to help visualize
        coordinate positions. For manipuland mode, uses support surface bounds.
        For furniture/ceiling mode, uses room bounds (falls back to floor bounds).

        Args:
            scene: Blender scene
            camera_obj: Camera object for projection
            is_top_view: Whether this is a top-down view
            is_drawer_view: Whether this is a per-drawer angled view (uses 3x3 grid)
            ceiling_height: Height of ceiling for ceiling mode (places grid at ceiling)
            room_bounds: Room XY bounds (min_x, min_y, max_x, max_y) for stable grid

        Returns:
            dict: Mapping from world coordinates (x, y) to pixel coordinates (px, py).
        """
        # Use support surface bounds if available (manipuland mode), otherwise floor.
        if self._surface_corners is not None:
            # Manipuland mode: use oriented local bounds from corner.
            # Grid transformation uses corner-based coordinates for correct rendering.
            # Labels will be offset to show center-based coordinates separately.
            edge_x = self._surface_corners[1] - self._surface_corners[0]
            edge_y = self._surface_corners[2] - self._surface_corners[0]
            edge_z = self._surface_corners[4] - self._surface_corners[0]

            extent_x = np.linalg.norm(edge_x)
            extent_y = np.linalg.norm(edge_y)
            extent_z = np.linalg.norm(edge_z)

            # Grid in corner-based coordinates (0 to extent) for transformation.
            min_x, min_y = 0, 0
            max_x, max_y = extent_x, extent_y

            # Z height is at the surface (local Z = 0).
            marker_z = 0.0  # Local Z

            # Always compute grid origin and axes from current surface corners.
            # _surface_corners is guaranteed to be updated for each view (line 548-552).
            self._grid_origin = self._surface_corners[0]
            self._grid_axis_x = (
                edge_x / extent_x if extent_x > 0 else np.array([1, 0, 0])
            )
            self._grid_axis_y = (
                edge_y / extent_y if extent_y > 0 else np.array([0, 1, 0])
            )
            self._grid_axis_z = (
                edge_z / extent_z if extent_z > 0 else np.array([0, 0, 1])
            )

            # Store extents for label offset computation (to show center-based coords).
            self._extent_x = extent_x
            self._extent_y = extent_y

        else:
            # Use room_bounds if provided, otherwise fall back to floor bounds.
            if room_bounds is not None:
                min_x, min_y, max_x, max_y = room_bounds
            else:
                # Fall back to floor bounds (computed from scene objects).
                floor_bounds = get_floor_bounds(self._client_objects)
                min_x, min_y, _, max_x, max_y = floor_bounds
            # Use ceiling height if provided (ceiling mode), otherwise floor Z (0).
            marker_z = ceiling_height if ceiling_height is not None else 0

        marks = {}

        if is_top_view or is_drawer_view:
            # Top view: 5x5 grid uniformly distributed across bounds.
            # Drawer view: 3x3 grid (smaller surface, less clutter).
            num_divisions = 3 if is_drawer_view else 5

            # Generate X positions uniformly from min_x to max_x.
            x_positions = []
            for i in range(num_divisions):
                x = min_x + (max_x - min_x) * i / (num_divisions - 1)
                x_positions.append(x)

            # Generate Y positions uniformly from min_y to max_y.
            y_positions = []
            for i in range(num_divisions):
                y = min_y + (max_y - min_y) * i / (num_divisions - 1)
                y_positions.append(y)

            # Create all labeled markers on uniform grid.
            for local_x in x_positions:
                for local_y in y_positions:
                    # Transform from local grid coordinates to world coordinates.
                    if hasattr(self, "_grid_origin"):
                        # Manipuland mode: transform local coords to world
                        # using surface orientation.
                        world_pos = (
                            self._grid_origin
                            + local_x * self._grid_axis_x
                            + local_y * self._grid_axis_y
                            + marker_z * self._grid_axis_z
                        )
                        world_coord = Vector(world_pos.tolist())
                    else:
                        # Furniture mode: grid positions are already in
                        # world coordinates.
                        world_coord = Vector((local_x, local_y, marker_z))

                    # Filter by convex hull if provided (multi-surface mode).
                    if self._should_include_marker(world_coord):
                        pixel_x, pixel_y = get_pixel_coordinates(
                            scene, camera_obj, world_coord
                        )
                        if pixel_x is not None and pixel_y is not None:
                            # For manipuland mode, offset labels to show center-based coords.
                            if hasattr(self, "_extent_x"):
                                label_x = local_x - self._extent_x / 2
                                label_y = local_y - self._extent_y / 2
                                marks[(label_x, label_y)] = (pixel_x, pixel_y)
                            else:
                                marks[(local_x, local_y)] = (pixel_x, pixel_y)
        else:
            # Side views: 9 strategic positions (corners, edge midpoints, center).
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            strategic_positions = [
                (min_x, min_y),
                (max_x, min_y),
                (min_x, max_y),
                (max_x, max_y),
                (center_x, min_y),
                (center_x, max_y),
                (min_x, center_y),
                (max_x, center_y),
                (center_x, center_y),
            ]

            for local_x, local_y in strategic_positions:
                # Transform from local grid coordinates to world coordinates.
                if hasattr(self, "_grid_origin"):
                    # Manipuland mode: transform local coords to world
                    # using surface orientation.
                    world_pos = (
                        self._grid_origin
                        + local_x * self._grid_axis_x
                        + local_y * self._grid_axis_y
                        + marker_z * self._grid_axis_z
                    )
                    world_coord = Vector(world_pos.tolist())
                else:
                    # Furniture mode: grid positions are already in world coordinates.
                    world_coord = Vector((local_x, local_y, marker_z))

                # Filter by convex hull if provided (multi-surface mode).
                if self._should_include_marker(world_coord):
                    pixel_x, pixel_y = get_pixel_coordinates(
                        scene, camera_obj, world_coord
                    )
                    if pixel_x is not None and pixel_y is not None:
                        # For manipuland mode, offset labels to show center-based coords.
                        if hasattr(self, "_extent_x"):
                            label_x = local_x - self._extent_x / 2
                            label_y = local_y - self._extent_y / 2
                            marks[(label_x, label_y)] = (pixel_x, pixel_y)
                        else:
                            marks[(local_x, local_y)] = (pixel_x, pixel_y)

        return marks

    def _should_include_marker(self, world_coord: Vector) -> bool:
        """Check if a coordinate marker should be included based on convex hull.

        For multi-surface mode, filters markers to only show those inside the
        surface's convex hull. For single-surface or furniture mode, includes all.

        Args:
            world_coord: World-space coordinate of the marker.

        Returns:
            True if marker should be included, False otherwise.
        """
        # If no convex hull filtering, include all markers.
        if (
            not hasattr(self, "_current_convex_hull")
            or self._current_convex_hull is None
        ):
            return True

        # Use shapely for robust point-in-polygon test.
        try:
            from shapely.geometry import Point, Polygon

            # Compute convex hull to get properly ordered vertices.
            # Unordered vertices can create self-intersecting polygons.
            ordered_vertices_2d = compute_ordered_convex_hull_vertices_2d(
                vertices_3d=self._current_convex_hull
            )

            # Create polygon from ordered vertices.
            polygon = Polygon(ordered_vertices_2d)

            # Buffer polygon to include boundary markers.
            buffered_polygon = polygon.buffer(CONVEX_HULL_MARKER_BUFFER_METERS)

            # Test if point is inside buffered polygon.
            point = Point(world_coord[0], world_coord[1])
            return buffered_polygon.contains(point)
        except Exception as e:
            # If shapely fails, include the marker (fail-safe).
            console_logger.warning(
                f"Failed to check marker inclusion with shapely: {e}, including marker"
            )
            return True

    def _debug_visualize_convex_hull(
        self, image_path: Path, camera_obj: bpy.types.Object
    ) -> None:
        """Debug visualization: Draw convex hull outline on rendered image.

        Args:
            image_path: Path to the rendered image.
            camera_obj: Camera object for projection.
        """
        if (
            not hasattr(self, "_current_convex_hull")
            or self._current_convex_hull is None
        ):
            return

        try:
            from mathutils import Vector
            from PIL import Image, ImageDraw

            # Compute convex hull to get properly ordered vertices.
            ordered_vertices_2d = compute_ordered_convex_hull_vertices_2d(
                vertices_3d=self._current_convex_hull
            )

            # Load image.
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)

            # Project ordered vertices to pixel coordinates.
            scene = bpy.context.scene
            hull_pixels = []
            for vertex_2d in ordered_vertices_2d:
                # Use Z coordinate from first convex hull vertex.
                z_coord = self._current_convex_hull[0][2]
                world_coord = Vector((vertex_2d[0], vertex_2d[1], z_coord))
                pixel_x, pixel_y = get_pixel_coordinates(scene, camera_obj, world_coord)
                if pixel_x is not None and pixel_y is not None:
                    hull_pixels.append((pixel_x, pixel_y))

            # Draw convex hull outline if we have enough points.
            if len(hull_pixels) >= 3:
                # Close the polygon by adding first point at end.
                hull_pixels.append(hull_pixels[0])
                # Draw thick outline in cyan for visibility.
                draw.line(hull_pixels, fill=(0, 255, 255), width=4)

            # Save annotated image.
            img.save(image_path)

        except Exception as e:
            console_logger.warning(f"Failed to visualize convex hull: {e}")
