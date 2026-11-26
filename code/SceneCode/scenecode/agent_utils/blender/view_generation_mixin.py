"""Mixin class for view generation functionality in BlenderRenderer.

This mixin encapsulates all view generation methods for multi-view rendering,
including grid views, single views, top-plus-sides, and wall-specific configs.
"""

import math

from mathutils import Vector

# View generation constants.
NUM_GRID_VIEWS_SIDE = 8
GRID_VIEW_ELEVATION_ANGLE_RAD = math.pi / 6  # 30 degrees.


class ViewGenerationMixin:
    """Mixin providing view generation methods for BlenderRenderer.

    This mixin contains methods for generating different view configs
    used in multi-view rendering (grid, single top, top-plus-sides).
    """

    def _generate_grid_3x3_views(self) -> list[dict]:
        """Generate 9 views for 3x3 grid: 1 top + 8 sides at 45° intervals.

        Returns:
            List of view dictionaries with name, direction, and is_side fields.
        """
        views = []
        views.append(
            {"name": "0_top", "direction": Vector((0, 0, 1)), "is_side": False}
        )

        # 8 side views at 45° intervals around vertical axis with 30° elevation.
        elevation_angle = GRID_VIEW_ELEVATION_ANGLE_RAD
        for i in range(NUM_GRID_VIEWS_SIDE):
            horizontal_angle = 2 * math.pi * i / NUM_GRID_VIEWS_SIDE

            # Convert spherical to cartesian coordinates.
            x = math.cos(elevation_angle) * math.cos(horizontal_angle)
            y = math.cos(elevation_angle) * math.sin(horizontal_angle)
            z = math.sin(elevation_angle)

            dir_vec = Vector((x, y, z))
            views.append({"name": f"{i+1}_side", "direction": dir_vec, "is_side": True})

        return views

    def _generate_single_top_view(self) -> list[dict]:
        """Generate single top-down view.

        Returns:
            List containing one view dictionary.
        """
        return [{"name": "0_top", "direction": Vector((0, 0, 1)), "is_side": False}]

    def _generate_top_plus_sides_views(
        self,
        count: int,
        furniture_rotation_z: float | None = None,
        is_multi_surface_mode: bool = False,
        elevation_degrees: float | None = None,
        start_azimuth_degrees: float | None = None,
        include_vertical_views: bool = True,
    ) -> list[dict]:
        """Generate 1 top + N side views from corners.

        Args:
            count: Number of side views (typically 4 for corner views at 45°).
            furniture_rotation_z: Optional furniture rotation angle (radians) to align
                side view angles with furniture orientation in manipuland mode.
            is_multi_surface_mode: If True, skip 45° offset for perpendicular views
                in multi-surface rendering mode.
            elevation_degrees: Optional elevation angle in degrees for side views.
                Overrides default 30 degrees.
            start_azimuth_degrees: Optional starting azimuth angle in degrees.
                90 degrees positions camera at +Y (front). Overrides default.
            include_vertical_views: Whether to include top-down view. Defaults to True.

        Returns:
            List of view dictionaries with name, direction, and is_side fields.
        """
        views = []

        # Add top view unless disabled for angled-only rendering.
        if include_vertical_views:
            views.append(
                {"name": "0_top", "direction": Vector((0, 0, 1)), "is_side": False}
            )

        # Use provided elevation or default to 30 degrees.
        elevation_angle = (
            math.radians(elevation_degrees)
            if elevation_degrees is not None
            else math.pi / 6
        )

        for i in range(count):
            # Determine starting azimuth.
            if start_azimuth_degrees is not None:
                # Use explicit start azimuth (in radians) without 45° offset.
                offset = math.radians(start_azimuth_degrees)
            elif is_multi_surface_mode:
                # For multi-surface mode, use perpendicular angles (no offset).
                offset = 0
            else:
                # For other modes, offset by 45° to position cameras at corners.
                offset = math.pi / 4

            horizontal_angle = (2 * math.pi * i / count) + offset

            # Rotate angles to align with furniture orientation in manipuland mode.
            # Skip this if explicit start_azimuth is provided.
            if furniture_rotation_z is not None and start_azimuth_degrees is None:
                horizontal_angle += furniture_rotation_z

            x = math.cos(elevation_angle) * math.cos(horizontal_angle)
            y = math.cos(elevation_angle) * math.sin(horizontal_angle)
            z = math.sin(elevation_angle)
            dir_vec = Vector((x, y, z))
            views.append({"name": f"{i}_side", "direction": dir_vec, "is_side": True})

        return views

    def _generate_wall_context_views(self) -> list[dict]:
        """Generate single top-down view for wall context.

        Same as single_top but used for wall_context rendering mode which
        applies different annotation filtering (only wall objects annotated).

        Returns:
            List containing one view dictionary for top-down context.
        """
        return [
            {
                "name": "wall_context_top",
                "direction": Vector((0, 0, 1)),
                "is_side": False,
                "is_wall_context": True,
            }
        ]

    def _generate_wall_orthographic_view(
        self, wall_surfaces: list[dict] | None = None
    ) -> list[dict]:
        """Generate orthographic views perpendicular to wall surface(s).

        Creates views looking at wall(s) from the room interior, perpendicular
        to the wall surface. The camera uses orthographic projection to show
        each wall in correct proportions for coordinate-based placement.

        Args:
            wall_surfaces: List of wall surface dicts. Each contains:
                - wall_id: Wall identifier (e.g., "living_room_north")
                - direction: Wall direction (e.g., "north", "south", "east", "west")
                - length: Wall length in meters
                - height: Wall height in meters
                - transform: [x, y, z, qw, qx, qy, qz] pose in world frame
                - excluded_regions: List of [x_min, z_min, x_max, z_max] for doors/windows

        Returns:
            List of view dictionaries for wall orthographic views.
        """
        if not wall_surfaces:
            raise ValueError("wall_orthographic layout requires wall_surfaces")

        # Direction map: camera looks toward wall.
        direction_map = {
            "north": Vector((0, 1, 0)),  # North wall at +Y, camera looks toward +Y.
            "south": Vector((0, -1, 0)),  # South wall at -Y, camera looks toward -Y.
            "east": Vector((1, 0, 0)),  # East wall at +X, camera looks toward +X.
            "west": Vector((-1, 0, 0)),  # West wall at -X, camera looks toward -X.
        }

        views = []
        for surface in wall_surfaces:
            wall_id = surface.get("wall_id", "unknown")
            direction = surface.get("direction", "north")
            camera_direction = direction_map.get(direction.lower(), Vector((0, 1, 0)))

            views.append(
                {
                    "name": f"wall_{wall_id}_ortho",
                    "direction": camera_direction,
                    "is_side": False,
                    "is_wall_orthographic": True,
                    "is_orthographic": True,
                    "wall_surface": surface,
                }
            )

        return views

    def _generate_wall_views(self, wall_surfaces: list[dict]) -> list[dict]:
        """Generate combined wall views: context top-down + per-wall orthographic.

        Produces all wall agent views in a single batch:
        - 1 top-down context view (wall_context)
        - N orthographic views (one per wall surface)

        Args:
            wall_surfaces: List of wall surface dicts for orthographic rendering.

        Returns:
            List of view dictionaries (context first, then orthographic views).
        """
        views = []

        # 1. Add top-down context view.
        views.append(
            {
                "name": "wall_context_top",
                "direction": Vector((0, 0, 1)),
                "is_side": False,
                "is_wall_context": True,
            }
        )

        # 2. Add orthographic views for each wall.
        direction_map = {
            "north": Vector((0, 1, 0)),
            "south": Vector((0, -1, 0)),
            "east": Vector((1, 0, 0)),
            "west": Vector((-1, 0, 0)),
        }

        for surface in wall_surfaces:
            wall_id = surface.get("wall_id", "unknown")
            direction = surface.get("direction", "north")
            camera_direction = direction_map.get(direction.lower(), Vector((0, 1, 0)))

            views.append(
                {
                    "name": f"wall_{wall_id}_ortho",
                    "direction": camera_direction,
                    "is_side": False,
                    "is_wall_orthographic": True,
                    "is_orthographic": True,
                    "wall_surface": surface,
                }
            )

        return views

    def _generate_ceiling_perspective_view(
        self,
        room_bounds: tuple[float, float, float, float],
        ceiling_height: float,
    ) -> list[dict]:
        """Generate elevated perspective view for ceiling observation.

        Creates a perspective camera positioned above the room looking down
        at the ceiling plane. Camera is elevated above ceiling height and
        positioned at a corner to provide clear view of ceiling fixtures
        with furniture context below.

        Args:
            room_bounds: Room XY bounds (min_x, min_y, max_x, max_y) in meters.
            ceiling_height: Height of ceiling above floor in meters.

        Returns:
            List containing one view dictionary for ceiling perspective.
        """
        min_x, min_y, max_x, max_y = room_bounds
        room_center_x = (min_x + max_x) / 2
        room_center_y = (min_y + max_y) / 2
        room_width = max_x - min_x
        room_depth = max_y - min_y

        # Position camera at elevated corner looking toward room center.
        # Camera height: ceiling + 2m for good overview.
        # Camera position: offset from corner for angled view.
        camera_height = ceiling_height + 2.0
        camera_offset = max(room_width, room_depth) * 0.3

        # Camera at corner offset, looking toward center and down.
        camera_x = min_x - camera_offset
        camera_y = min_y - camera_offset

        # Direction from camera toward room center (normalized).
        dx = room_center_x - camera_x
        dy = room_center_y - camera_y
        dz = ceiling_height - camera_height  # Looking down at ceiling.

        return [
            {
                "name": "ceiling_perspective",
                "direction": Vector((dx, dy, dz)).normalized(),
                "is_side": False,
                "is_ceiling_perspective": True,
                "camera_position": Vector((camera_x, camera_y, camera_height)),
                "room_bounds": room_bounds,
                "ceiling_height": ceiling_height,
            }
        ]
