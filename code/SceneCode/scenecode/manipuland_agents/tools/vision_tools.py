"""
Vision tools for focused scene observation and visual feedback.

This module provides capabilities for rendering focused views of individual
furniture pieces with their manipulands. Returns images directly via
ToolOutputImage so they persist in the session across API calls.
"""

import logging

from typing import TYPE_CHECKING, Any

from agents import ToolOutputImage, ToolOutputText, function_tool
from omegaconf import DictConfig

from scenecode.agent_utils.physics_tools import check_physics_violations
from scenecode.agent_utils.rendering_manager import RenderingManager
from scenecode.agent_utils.room import AgentType, ObjectType, RoomScene, UniqueID
from scenecode.utils.openai import encode_image_to_base64

if TYPE_CHECKING:
    from scenecode.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)


class ManipulandVisionTools:
    """Vision tools for focused scene observation.

    Provides capabilities for rendering focused views showing only specific
    furniture and its manipulands. Returns images directly via ToolOutputImage
    so they persist in the session across API calls.
    """

    def __init__(
        self,
        scene: RoomScene,
        rendering_manager: RenderingManager,
        cfg: DictConfig,
        current_furniture_id: UniqueID,
        blender_server: "BlenderServer",
        context_furniture_ids: list[UniqueID] | None = None,
    ):
        """Initialize vision tools.

        Args:
            scene: RoomScene to observe.
            rendering_manager: Manager for rendering operations.
            cfg: Configuration with rendering settings.
            current_furniture_id: ID of furniture currently being populated.
            blender_server: BlenderServer instance for rendering. REQUIRED - forked
                workers cannot safely use embedded bpy due to GPU/OpenGL state
                corruption from fork.
            context_furniture_ids: Optional list of nearby furniture IDs to include
                for spatial context (e.g., chairs around a table).
        """
        self.scene = scene
        self.rendering_manager = rendering_manager
        self.cfg = cfg
        self.current_furniture_id = current_furniture_id
        self.blender_server = blender_server
        self.context_furniture_ids = context_furniture_ids or []
        self.tools = self._create_tool_closures()

    def _create_tool_closures(self) -> dict[str, Any]:
        """Create tool closures with access to scene dependencies."""

        @function_tool
        async def observe_scene() -> list[ToolOutputImage | ToolOutputText]:
            """Observe the current furniture and manipulands placed on it.

            Shows:
            - The furniture currently being populated
            - Manipulands already placed on it
            - Minimal floor context

            Does NOT show:
            - Other furniture in the scene
            - Manipulands on other surfaces

            This focused view helps you see details of your current work without
            visual clutter from the rest of the scene.

            Returns:
                Images of the focused view plus a confirmation message. These
                images persist in your conversation history.
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

        return {
            "observe_scene": observe_scene,
            "check_physics": check_physics,
        }

    def _observe_scene_impl(self) -> list[ToolOutputImage | ToolOutputText]:
        """Implementation for focused scene observation.

        Renders only the current furniture + its manipulands using scene filtering.
        Returns images directly via ToolOutputImage so they persist in the session.

        Returns:
            List of images plus a confirmation message.
        """
        console_logger.info("Tool called: observe_scene (focused on current furniture)")

        # Get current furniture object.
        furniture = self.scene.get_object(self.current_furniture_id)
        if not furniture:
            console_logger.error(
                f"Furniture {self.current_furniture_id} not found for observation"
            )
            return [
                ToolOutputText(
                    text="Unable to observe scene - current furniture not found."
                )
            ]

        # Get support surfaces for this furniture.
        surface_ids = {s.surface_id for s in furniture.support_surfaces}

        # Get manipulands on these surfaces.
        manipulands_on_furniture = [
            obj
            for obj in self.scene.objects.values()
            if (
                obj.placement_info
                and obj.placement_info.parent_surface_id in surface_ids
            )
        ]

        # Detect if current furniture is the floor.
        is_floor_observation = (
            self.current_furniture_id == self.scene.room_geometry.floor.object_id
            if self.scene.room_geometry and self.scene.room_geometry.floor
            else False
        )

        manipuland_ids = [obj.object_id for obj in manipulands_on_furniture]

        # Build list of objects to include based on furniture type.
        if is_floor_observation:
            # Floor rendering needs full context: floor + all furniture + floor
            # manipulands.
            furniture_ids = [
                obj.object_id
                for obj in self.scene.objects.values()
                if obj.object_type == ObjectType.FURNITURE
            ]
            include_objects = furniture_ids + manipuland_ids
            exclude_room_geometry = False
            console_logger.info(
                f"Rendering floor view: all furniture + {len(manipuland_ids)} "
                "floor manipulands"
            )
        else:
            # Regular furniture: focused rendering with current furniture + manipulands.
            # Include context furniture for orientation decisions (e.g., chairs around table).
            valid_context_ids = [
                ctx_id
                for ctx_id in self.context_furniture_ids
                if ctx_id in self.scene.objects
            ]
            include_objects = (
                [self.current_furniture_id] + manipuland_ids + valid_context_ids
            )
            exclude_room_geometry = True
            context_info = (
                f" + {len(valid_context_ids)} context" if valid_context_ids else ""
            )
            console_logger.info(
                f"Rendering focused view: 1 furniture + {len(manipuland_ids)} "
                f"manipulands{context_info}"
            )

        # Get all support surfaces for this furniture.
        support_surfaces = (
            furniture.support_surfaces if furniture.support_surfaces else []
        )

        if not support_surfaces:
            console_logger.error(
                f"No support surfaces defined for {furniture.name}, "
                "rendering without support surface visualization"
            )

        # Check if current furniture is articulated (doors/drawers should be open).
        is_articulated = furniture.metadata.get("is_articulated", False)
        if is_articulated:
            console_logger.info(
                f"Furniture '{furniture.name}' is articulated, rendering with open state"
            )

        console_logger.info(
            f"Calling render_scene with include_objects="
            f"{[str(obj_id) for obj_id in include_objects]}, "
            f"{len(support_surfaces)} support surface(s)"
        )
        images_dir = self.rendering_manager.render_scene(
            scene=self.scene,
            blender_server=self.blender_server,
            include_objects=include_objects,
            exclude_room_geometry=exclude_room_geometry,
            rendering_mode="manipuland",
            support_surfaces=support_surfaces,  # Pass all surfaces.
            show_support_surface=self.cfg.rendering.annotations.enable_support_surface_debug,
            articulated_open=is_articulated,
            context_furniture_ids=self.context_furniture_ids,
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

        num_images = len(outputs)

        # Return appropriate message based on observation type.
        if is_floor_observation:
            num_furniture = len(furniture_ids)
            outputs.append(
                ToolOutputText(
                    text=f"Floor and full scene context observed from {num_images} "
                    f"viewpoints. Visual feedback now available. "
                    f"(Showing: floor + {num_furniture} furniture + {len(manipuland_ids)} "
                    "floor manipulands)"
                )
            )
        else:
            num_surfaces = len(support_surfaces)
            surface_text = (
                f"{num_surfaces} surface(s)" if num_surfaces > 1 else "1 surface"
            )
            outputs.append(
                ToolOutputText(
                    text=f"Current furniture and its manipulands observed from {num_images} "
                    f"viewpoints ({surface_text}). Visual feedback now available. "
                    f"(Showing: {furniture.name} + {len(manipuland_ids)} manipulands)"
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
            scene=self.scene,
            cfg=self.cfg,
            current_furniture_id=self.current_furniture_id,
            agent_type=AgentType.MANIPULAND,
        )
