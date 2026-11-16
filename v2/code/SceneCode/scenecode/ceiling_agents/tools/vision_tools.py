"""Vision tools for ceiling-mounted object placement.

Provides rendering capabilities for ceiling agent observation, including:
- Elevated perspective view (ceiling + furniture context from above)
"""

import logging

from pathlib import Path
from typing import TYPE_CHECKING

from agents import ToolOutputImage, ToolOutputText, function_tool
from omegaconf import DictConfig

from scenecode.agent_utils.physics_tools import check_physics_violations
from scenecode.agent_utils.rendering_manager import RenderingManager
from scenecode.agent_utils.room import AgentType, ObjectType, RoomScene
from scenecode.utils.openai import encode_image_to_base64

if TYPE_CHECKING:
    from scenecode.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)


class CeilingVisionTools:
    """Vision tools for ceiling-mounted object placement.

    Provides rendering for ceiling agent observation with elevated perspective view.
    """

    def __init__(
        self,
        scene: RoomScene,
        rendering_manager: RenderingManager,
        room_bounds: tuple[float, float, float, float],
        ceiling_height: float,
        cfg: DictConfig,
        blender_server: "BlenderServer",
    ):
        """Initialize ceiling vision tools.

        Args:
            scene: RoomScene to observe.
            rendering_manager: Manager for rendering operations.
            room_bounds: Room XY bounds (min_x, min_y, max_x, max_y).
            ceiling_height: Height of ceiling above floor (meters).
            cfg: Configuration with rendering settings.
            blender_server: BlenderServer instance for rendering. REQUIRED - forked
                workers cannot safely use embedded bpy due to GPU/OpenGL state
                corruption from fork.
        """
        self.scene = scene
        self.rendering_manager = rendering_manager
        self.room_bounds = room_bounds
        self.ceiling_height = ceiling_height
        self.cfg = cfg
        self.blender_server = blender_server
        self.tools = self._create_tool_closures()

    def _create_tool_closures(self) -> dict:
        """Create tool closures for vision operations."""

        @function_tool
        def observe_scene() -> list[ToolOutputImage | ToolOutputText]:
            """Observe the current scene state for ceiling placement.

            Renders an elevated perspective view of the room showing:
            - Ceiling plane with coordinate grid
            - Furniture below for spatial context
            - Already placed ceiling objects

            The perspective view provides depth cues for understanding
            spatial relationships between ceiling fixtures and furniture.

            Returns:
                List of images and confirmation message.
            """
            return self._observe_scene_impl()

        @function_tool
        def check_physics() -> list[ToolOutputText]:
            """Check physics validity of current ceiling placements.

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

        Renders an elevated perspective view showing ceiling and furniture context.

        Returns:
            List of images and confirmation message.
        """
        console_logger.info("Tool called: observe_scene (ceiling)")

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
        all_ceiling_object_ids = [
            obj.object_id
            for obj in self.scene.get_objects_by_type(ObjectType.CEILING_MOUNTED)
        ]

        console_logger.info(
            f"Rendering ceiling observation: {len(all_furniture_ids)} furniture, "
            f"{len(all_wall_object_ids)} wall objects, "
            f"{len(all_ceiling_object_ids)} ceiling objects"
        )

        # Render elevated perspective view.
        # Include wall objects so ceiling agent can see paintings, shelves, mirrors,
        # etc. that might inform light placement decisions.
        render_dir = self.rendering_manager.render_scene(
            scene=self.scene,
            blender_server=self.blender_server,
            include_objects=all_furniture_ids
            + all_wall_object_ids
            + all_ceiling_object_ids,
            exclude_room_geometry=False,
            rendering_mode="ceiling_perspective",
            room_bounds=self.room_bounds,
            ceiling_height=self.ceiling_height,
            annotate_object_types=["ceiling_mounted"],
        )

        if render_dir and render_dir.exists():
            images = self._collect_images(render_dir)
            outputs.extend(images)
            console_logger.info(f"Rendered {len(images)} ceiling view image(s)")

        # Add summary message.
        num_images = sum(1 for o in outputs if isinstance(o, ToolOutputImage))
        min_x, min_y, max_x, max_y = self.room_bounds
        room_width = max_x - min_x
        room_depth = max_y - min_y
        outputs.append(
            ToolOutputText(
                text=f"Ceiling observed from elevated perspective. "
                f"Room size: {room_width:.1f}m x {room_depth:.1f}m, "
                f"ceiling height: {self.ceiling_height:.1f}m. "
                f"{len(all_ceiling_object_ids)} ceiling objects placed."
            )
        )

        return outputs

    def _check_physics_impl(self) -> str:
        """Implementation for physics checking.

        Returns:
            String describing collision status.
        """
        console_logger.info("Tool called: check_physics (ceiling)")
        return check_physics_violations(
            scene=self.scene, cfg=self.cfg, agent_type=AgentType.CEILING_MOUNTED
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
