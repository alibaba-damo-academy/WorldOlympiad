"""
Vision tools for scene observation and dynamic visual feedback.

This module provides capabilities for rendering scenes and returning images
directly via ToolOutputImage. Images returned this way are stored in the session
and persist across API calls, giving the agent visual memory.
"""

import logging

from typing import TYPE_CHECKING, Any

from agents import ToolOutputImage, ToolOutputText, function_tool
from omegaconf import DictConfig

from scenecode.agent_utils.physics_tools import check_physics_violations
from scenecode.agent_utils.reachability import (
    compute_reachability,
    format_reachability_result,
)
from scenecode.agent_utils.rendering_manager import RenderingManager
from scenecode.agent_utils.room import AgentType, RoomScene
from scenecode.utils.openai import encode_image_to_base64

if TYPE_CHECKING:
    from scenecode.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)


class VisionTools:
    """Tools for scene observation and image persistence.

    Provides capabilities for rendering scenes and returning images directly
    via ToolOutputImage. Images returned this way are stored in the session
    and persist across API calls.
    """

    def __init__(
        self,
        scene: RoomScene,
        rendering_manager: RenderingManager,
        cfg: DictConfig,
        blender_server: "BlenderServer",
    ):
        """Initialize vision tools.

        Args:
            scene: RoomScene to observe.
            rendering_manager: Manager for rendering operations.
            cfg: Configuration with physics validation settings.
            blender_server: BlenderServer instance for rendering. REQUIRED - forked
                workers cannot safely use embedded bpy due to GPU/OpenGL state
                corruption from fork.
        """
        self.scene = scene
        self.rendering_manager = rendering_manager
        self.cfg = cfg
        self.blender_server = blender_server
        self.tools = self._create_tool_closures()

    def _get_room_bounds(self) -> tuple[float, float, float, float] | None:
        """Extract room XY bounds from scene geometry.

        Returns:
            Tuple of (min_x, min_y, max_x, max_y) in meters, or None if unavailable.
        """
        room_geom = self.scene.room_geometry
        if room_geom is not None and room_geom.length > 0 and room_geom.width > 0:
            half_length = room_geom.length / 2
            half_width = room_geom.width / 2
            return (-half_length, -half_width, half_length, half_width)
        return None

    def _create_tool_closures(self) -> dict[str, Any]:
        """Create tool closures with access to scene dependencies.

        Returns:
            Dictionary mapping tool names to tool functions.
        """

        @function_tool
        async def observe_scene() -> list[ToolOutputImage | ToolOutputText]:
            """Take visual snapshots of the current furniture arrangement.

            After calling this, you'll see images of the room from multiple angles
            showing all the furniture that's currently placed. This helps you
            understand the current state and make decisions about changes.

            Returns:
                Images of the scene from multiple viewpoints, plus a confirmation
                message. These images persist in your conversation history.
            """
            return self._observe_scene_impl()

        @function_tool
        async def check_physics() -> str:
            """Check for physics violations (collisions) in the current scene.

            This detects small collisions that might not be visible in renders
            but would make the scene physically invalid. Use this before
            concluding your design to ensure all objects are properly placed.

            Returns:
                Description of any collisions detected, or confirmation of no issues.
            """
            return self._check_physics_impl()

        @function_tool
        async def check_reachability() -> str:
            """Check if all areas of the room are reachable.

            Analyzes whether a person can traverse to all parts of the room
            without furniture blocking passages. Identifies specific furniture
            pieces that create blockages if the room is not fully reachable.

            Returns:
                JSON with reachability analysis including:
                - is_fully_reachable: true if room has single connected region
                - num_disconnected_regions: count of separate walkable areas
                - reachability_ratio: largest region / total (1.0 = fully connected)
                - blocking_furniture_ids: IDs of furniture blocking passages
            """
            return self._check_reachability_impl()

        return {
            "observe_scene": observe_scene,
            "check_physics": check_physics,
            "check_reachability": check_reachability,
        }

    def _observe_scene_impl(self) -> list[ToolOutputImage | ToolOutputText]:
        """Implementation for scene observation.

        Returns images directly via ToolOutputImage so they persist in the session.

        Returns:
            List of images plus a confirmation message.
        """
        console_logger.info("Tool called: observe_scene")
        # Render current scene state with room bounds for stable grid markers.
        images_dir = self.rendering_manager.render_scene(
            self.scene,
            blender_server=self.blender_server,
            room_bounds=self._get_room_bounds(),
        )

        if not images_dir or not images_dir.exists():
            console_logger.error("No renders generated for scene observation")
            return [
                ToolOutputText(text="Unable to observe scene - no renders available.")
            ]

        # Collect images and return them directly.
        outputs: list[ToolOutputImage | ToolOutputText] = []

        for img_path in sorted(images_dir.glob("*.png")):
            img_base64 = encode_image_to_base64(img_path)
            outputs.append(
                ToolOutputImage(image_url=f"data:image/png;base64,{img_base64}")
            )

        # Add confirmation message.
        num_images = len(outputs)
        outputs.append(
            ToolOutputText(
                text=f"Scene observed from {num_images} viewpoints. "
                "Visual feedback is now available for analysis."
            )
        )

        console_logger.info(f"Returning {num_images} images via ToolOutputImage")
        return outputs

    def _check_physics_impl(self) -> str:
        """Implementation for physics validation.

        Returns:
            String describing collision status.
        """
        console_logger.info("Tool called: check_physics")
        return check_physics_violations(
            scene=self.scene, cfg=self.cfg, agent_type=AgentType.FURNITURE
        )

    def _check_reachability_impl(self) -> str:
        """Implementation for reachability validation.

        Returns:
            Human-readable string describing reachability status.
        """
        console_logger.info("Tool called: check_reachability")
        robot_width = self.cfg.reachability.robot_width
        result = compute_reachability(scene=self.scene, robot_width=robot_width)
        return format_reachability_result(result)
