"""Vision tools for wall-mounted object placement.

Provides rendering capabilities for wall agent observation, including:
- Top-down context view (full room with furniture + wall objects)
- Per-wall orthographic views with grid overlay
"""

import logging

from pathlib import Path
from typing import TYPE_CHECKING

from agents import ToolOutputImage, ToolOutputText, function_tool
from omegaconf import DictConfig

from scenecode.agent_utils.house import WallDirection
from scenecode.agent_utils.physics_tools import check_physics_violations
from scenecode.agent_utils.rendering_manager import RenderingManager
from scenecode.agent_utils.room import AgentType, ObjectType, RoomScene
from scenecode.utils.openai import encode_image_to_base64
from scenecode.wall_agents.tools.wall_surface import WallSurface

if TYPE_CHECKING:
    from scenecode.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)


class WallVisionTools:
    """Vision tools for wall-mounted object placement.

    Provides rendering for wall agent observation with context and wall views.
    """

    def __init__(
        self,
        scene: RoomScene,
        rendering_manager: RenderingManager,
        wall_surfaces: list[WallSurface],
        cfg: DictConfig,
        blender_server: "BlenderServer",
    ):
        """Initialize wall vision tools.

        Args:
            scene: RoomScene to observe.
            rendering_manager: Manager for rendering operations.
            wall_surfaces: List of wall surfaces for the room.
            cfg: Configuration with rendering settings.
            blender_server: BlenderServer instance for rendering. REQUIRED - forked
                workers cannot safely use embedded bpy due to GPU/OpenGL state
                corruption from fork.
        """
        self.scene = scene
        self.rendering_manager = rendering_manager
        self.wall_surfaces = wall_surfaces
        self.cfg = cfg
        self.blender_server = blender_server
        self.tools = self._create_tool_closures()

    def _create_tool_closures(self) -> dict:
        """Create tool closures for vision operations."""

        @function_tool
        def observe_scene() -> list[ToolOutputImage | ToolOutputText]:
            """Observe the current scene state for wall placement.

            Renders views of the room to help with wall object placement:
            - Top-down context view showing all furniture and wall objects
            - Per-wall orthographic views with coordinate grid for all 4 walls

            The top-down view provides room context (furniture arrangement,
            traffic flow, focal points). Per-wall views show exact placement
            positions and excluded regions (doors/windows).

            Returns:
                List of images and confirmation message.
            """
            return self._observe_scene_impl()

        @function_tool
        def check_physics() -> list[ToolOutputText]:
            """Check physics validity of current wall placements.

            Returns:
                Physics validation results.
            """
            return self._check_physics_impl()

        return {
            "observe_scene": observe_scene,
            "check_physics": check_physics,
        }

    def _observe_scene_impl(self) -> list[ToolOutputImage | ToolOutputText]:
        """Implementation for scene observation.

        Renders wall views in a single call:
        1. Top-down context view (all furniture + all wall objects)
        2. Per-wall orthographic views for all 4 walls

        Returns:
            List of images and confirmation message.
        """
        console_logger.info("Tool called: observe_scene")

        outputs: list[ToolOutputImage | ToolOutputText] = []

        # Get all object IDs by type.
        all_furniture_ids = [
            obj.object_id
            for obj in self.scene.get_objects_by_type(ObjectType.FURNITURE)
        ]
        all_wall_object_ids = [
            obj.object_id
            for obj in self.scene.get_objects_by_type(ObjectType.WALL_MOUNTED)
        ]

        console_logger.info(
            f"Rendering wall observation: {len(all_furniture_ids)} furniture, "
            f"{len(all_wall_object_ids)} wall objects"
        )

        # Compute room bounds from wall surfaces for camera positioning.
        # Each wall direction tells us about a room boundary.
        room_bounds = self._compute_room_bounds_from_walls()

        # Build wall_surfaces list for rendering (all walls).
        wall_surfaces_for_render = []
        for surface in self.wall_surfaces:
            wall_surface_dict = self._wall_surface_to_dict(
                surface, room_bounds=room_bounds
            )
            wall_surfaces_for_render.append(wall_surface_dict)

        # Build wall_furniture_map: mapping from surface_id to nearby furniture IDs.
        furniture_threshold_m = self.cfg.rendering.wall_furniture_threshold_m
        wall_furniture_map = {}
        for surface in self.wall_surfaces:
            nearby_furniture_ids = self._get_furniture_near_wall(
                surface=surface,
                threshold_m=furniture_threshold_m,
            )
            wall_furniture_map[str(surface.surface_id)] = nearby_furniture_ids

        # Render all views in a single call (context + per-wall orthographic).
        render_dir = self.rendering_manager.render_scene(
            scene=self.scene,
            blender_server=self.blender_server,
            include_objects=all_furniture_ids + all_wall_object_ids,
            exclude_room_geometry=False,
            rendering_mode="wall",
            wall_surfaces=wall_surfaces_for_render,
            wall_surfaces_for_labels=wall_surfaces_for_render,
            annotate_object_types=["wall_mounted"],
            wall_furniture_map=wall_furniture_map,
        )

        if render_dir and render_dir.exists():
            images = self._collect_images(render_dir)
            outputs.extend(images)
            console_logger.info(
                f"Rendered {len(images)} wall images (1 context + "
                f"{len(wall_surfaces_for_render)} orthographic)"
            )

        # Add summary message.
        num_images = sum(1 for o in outputs if isinstance(o, ToolOutputImage))
        wall_names = [s.wall_id for s in self.wall_surfaces]
        outputs.append(
            ToolOutputText(
                text=f"Scene observed from {num_images} viewpoints. "
                f"Walls shown: {wall_names}. "
                f"{len(all_wall_object_ids)} wall objects placed."
            )
        )

        return outputs

    def _wall_surface_to_dict(
        self,
        surface: WallSurface,
        room_bounds: tuple[float, float, float, float] | None = None,
    ) -> dict:
        """Convert WallSurface to dict for rendering.

        Args:
            surface: WallSurface to convert.
            room_bounds: Optional (min_x, min_y, max_x, max_y) room bounds.
                Used to compute room_depth for camera positioning.

        Returns:
            Dict with wall surface data for Blender.
        """
        wall_transform = surface.transform
        direction = surface.wall_direction.name.lower()

        # Compute room_depth based on wall direction and room bounds.
        # For north/south walls, depth is Y extent. For east/west, it's X extent.
        room_depth = None
        if room_bounds is not None:
            min_x, min_y, max_x, max_y = room_bounds
            if direction in ("north", "south"):
                room_depth = max_y - min_y
            else:  # east, west
                room_depth = max_x - min_x

        result = {
            "surface_id": str(surface.surface_id),
            "wall_id": surface.wall_id,
            "direction": direction,
            "length": surface.length,
            "height": surface.height,
            "transform": [
                wall_transform.translation()[0],
                wall_transform.translation()[1],
                wall_transform.translation()[2],
                wall_transform.rotation().ToQuaternion().w(),
                wall_transform.rotation().ToQuaternion().x(),
                wall_transform.rotation().ToQuaternion().y(),
                wall_transform.rotation().ToQuaternion().z(),
            ],
            "excluded_regions": list(surface.excluded_regions),
        }
        if room_depth is not None:
            result["room_depth"] = room_depth
        return result

    def _check_physics_impl(self) -> str:
        """Implementation for physics checking.

        Returns:
            String describing collision status.
        """
        console_logger.info("Tool called: check_physics")
        return check_physics_violations(
            scene=self.scene,
            cfg=self.cfg,
            agent_type=AgentType.WALL_MOUNTED,
        )

    def _collect_images(self, images_dir: Path) -> list[ToolOutputImage]:
        """Collect rendered images from directory."""
        outputs = []
        for img_path in sorted(images_dir.glob("*.png")):
            img_base64 = encode_image_to_base64(img_path)
            outputs.append(
                ToolOutputImage(image_url=f"data:image/png;base64,{img_base64}")
            )
        return outputs

    def _compute_room_bounds_from_walls(
        self,
    ) -> tuple[float, float, float, float] | None:
        """Compute room XY bounds from wall surfaces.

        Uses wall positions to determine room extents. Each wall direction
        gives us one boundary (north wall at max_y, south at min_y, etc.).

        Returns:
            Tuple (min_x, min_y, max_x, max_y) or None if walls are insufficient.
        """
        if len(self.wall_surfaces) < 2:
            return None

        # Extract wall Y positions for north/south and X positions for east/west.
        y_positions = []
        x_positions = []

        for surface in self.wall_surfaces:
            # Get wall center position from transform.
            pos = surface.transform.translation()
            direction = surface.wall_direction.name.lower()

            if direction in ("north", "south"):
                y_positions.append(pos[1])
            else:  # east, west
                x_positions.append(pos[0])

        # Compute bounds from wall positions.
        # If we don't have walls in both directions, use a fallback.
        if y_positions:
            min_y = min(y_positions)
            max_y = max(y_positions)
        else:
            # Fallback: use X-axis walls' Y positions (approximate).
            min_y, max_y = -3.0, 3.0

        if x_positions:
            min_x = min(x_positions)
            max_x = max(x_positions)
        else:
            # Fallback: use Y-axis walls' X positions (approximate).
            min_x, max_x = -3.0, 3.0

        return (min_x, min_y, max_x, max_y)

    def _get_furniture_near_wall(
        self, surface: WallSurface, threshold_m: float
    ) -> list:
        """Get furniture IDs within threshold distance of wall.

        Checks AABB nearest edge (not center) - a sofa touching the wall
        is included even if its center is far.

        Args:
            surface: Wall surface to check distance from.
            threshold_m: Maximum distance in meters.

        Returns:
            List of furniture object IDs near the wall.
        """
        nearby_ids = []

        for obj in self.scene.get_objects_by_type(ObjectType.FURNITURE):
            # Get object world bounds.
            if obj.bbox_min is None or obj.bbox_max is None:
                raise RuntimeError(
                    f"Furniture '{obj.name}' ({obj.object_id}) has no bbox. "
                    f"All placed furniture must have bounding boxes."
                )

            # Compute object world AABB (handles rotation and pre-scaled bbox values).
            world_bounds = obj.compute_world_bounds()
            if world_bounds is None:
                raise RuntimeError(
                    f"Furniture '{obj.name}' ({obj.object_id}) could not compute "
                    f"world bounds."
                )
            obj_min, obj_max = world_bounds

            # Get wall position.
            wall_pos = surface.transform.translation()

            # Check distance based on wall direction.
            # North/South walls are at constant Y, East/West at constant X.
            if surface.wall_direction in (WallDirection.NORTH, WallDirection.SOUTH):
                wall_y = wall_pos[1]
                dist = min(abs(obj_min[1] - wall_y), abs(obj_max[1] - wall_y))
            else:  # EAST, WEST
                wall_x = wall_pos[0]
                dist = min(abs(obj_min[0] - wall_x), abs(obj_max[0] - wall_x))

            if dist <= threshold_m:
                nearby_ids.append(obj.object_id)

        return nearby_ids

    def _get_wall_objects_on_surface(self, surface_id: str) -> list:
        """Get wall object IDs on a specific surface.

        Args:
            surface_id: Wall surface ID.

        Returns:
            List of wall object IDs on this surface.
        """
        object_ids = []

        for obj in self.scene.get_objects_by_type(ObjectType.WALL_MOUNTED):
            if obj.placement_info is None:
                raise RuntimeError(
                    f"Wall object '{obj.name}' ({obj.object_id}) has no placement_info. "
                    f"All wall objects must be placed via place_wall_object."
                )
            if str(obj.placement_info.parent_surface_id) == surface_id:
                object_ids.append(obj.object_id)

        return object_ids
