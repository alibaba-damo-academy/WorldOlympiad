"""
Stateful manipuland agent with planner/designer/critic workflow.

This module implements manipuland placement using persistent agents that work
per-furniture, with fresh contexts for each furniture surface to bound token usage.
"""

import logging
import math

from pathlib import Path
from typing import Any

from agents import Agent, FunctionTool, Runner, RunResult, custom_span
from omegaconf import DictConfig

from scenecode.agent_utils.base_stateful_agent import (
    BaseStatefulAgent,
    log_agent_usage,
)
from scenecode.agent_utils.physical_feasibility import (
    apply_per_furniture_postprocessing,
)
from scenecode.agent_utils.placement_noise import PlacementNoiseMode
from scenecode.agent_utils.rendering_manager import RenderingManager
from scenecode.agent_utils.room import (
    AgentType,
    ObjectType,
    RoomScene,
    SupportSurface,
    UniqueID,
    extract_and_propagate_support_surfaces,
)
from scenecode.agent_utils.scene_analyzer import FurnitureSelection, SceneAnalyzer
from scenecode.agent_utils.scoring import (
    ManipulandCritiqueWithScores,
    log_agent_response,
)
from scenecode.agent_utils.support_surface_extraction import (
    SupportSurfaceExtractionConfig,
)
from scenecode.agent_utils.workflow_tools import WorkflowTools
from scenecode.manipuland_agents.base_manipuland_agent import BaseManipulandAgent
from scenecode.manipuland_agents.tools.manipuland_tools import ManipulandTools
from scenecode.manipuland_agents.tools.vision_tools import ManipulandVisionTools
from scenecode.prompts.registry import ManipulandAgentPrompts
from scenecode.utils.logging import BaseLogger

console_logger = logging.getLogger(__name__)


class StatefulManipulandAgent(BaseStatefulAgent, BaseManipulandAgent):
    """Manipuland placement with planner/designer/critic agents per furniture.

    Workflow:
    1. Initial analysis: Identify which furniture to populate
    2. Per-furniture loop: Create fresh agents for each furniture surface
    3. Per-furniture workflow: Planner coordinates designer/critic
    4. Agent-driven termination: Planner decides when surface is complete
    """

    @property
    def agent_type(self) -> AgentType:
        """Return agent type for collision filtering."""
        return AgentType.MANIPULAND

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
        geometry_server_host: str = "127.0.0.1",
        geometry_server_port: int = 7000,
        hssd_server_host: str = "127.0.0.1",
        hssd_server_port: int = 7001,
        articulated_server_host: str = "127.0.0.1",
        articulated_server_port: int = 7002,
        materials_server_host: str = "127.0.0.1",
        materials_server_port: int = 7008,
        num_workers: int = 1,
        render_gpu_id: int | None = None,
    ):
        # Initialize base agent (sessions, checkpoint state, prompt registry).
        BaseStatefulAgent.__init__(
            self,
            cfg=cfg,
            logger=logger,
            geometry_server_host=geometry_server_host,
            geometry_server_port=geometry_server_port,
            hssd_server_host=hssd_server_host,
            hssd_server_port=hssd_server_port,
        )
        # Initialize manipuland-specific base class.
        BaseManipulandAgent.__init__(
            self,
            cfg=cfg,
            logger=logger,
            geometry_server_host=geometry_server_host,
            geometry_server_port=geometry_server_port,
            hssd_server_host=hssd_server_host,
            hssd_server_port=hssd_server_port,
            articulated_server_host=articulated_server_host,
            articulated_server_port=articulated_server_port,
            materials_server_host=materials_server_host,
            materials_server_port=materials_server_port,
            num_workers=num_workers,
            render_gpu_id=render_gpu_id,
        )

        # Initialize pending images for image injection during critique.
        self.pending_images: list[dict[str, Any]] = []

        # Current furniture selection context (set per-furniture in workflow).
        self.current_furniture_selection: FurnitureSelection | None = None

        # Context image for manipuland designer initialization (per-furniture).
        self.manipuland_context_image_path: Path | None = None

    def _render_furniture_for_context(self) -> Path:
        """Render furniture with clean angled front view for context image input.

        Uses furniture_selection mode with empty annotate_object_types to get
        a clean render without any labels, bounding boxes, or coordinate overlays.
        For articulated furniture, opens joints to show interior surfaces.
        Includes context furniture (e.g., chairs around a table) for spatial reference.

        Uses adaptive camera elevation based on furniture type:
        - Tables (1 surface): High elevation (60°) - looking down at surface
        - Shelves (multiple surfaces): Low elevation (30°) - see all levels from front

        Camera is positioned to view the furniture's front face (+Y in local frame),
        accounting for the furniture's world rotation.

        Special case for floor: Renders top-down view of entire room with all
        furniture visible, similar to observe_scene. This provides spatial context
        for floor item placement (rugs, floor lamps, etc.).

        Returns:
            Path to directory containing rendered images.
        """
        furniture = self.scene.get_object(self.current_furniture_id)

        # Special case: Floor needs top-down view of entire room with all furniture.
        # This provides spatial context for floor item placement.
        if furniture.object_type == ObjectType.FLOOR:
            # Include all furniture objects for room context.
            all_furniture_ids = [
                obj.object_id
                for obj in self.scene.objects.values()
                if obj.object_type == ObjectType.FURNITURE
            ]
            return self.rendering_manager.render_scene(
                scene=self.scene,
                blender_server=self.blender_server,
                include_objects=[self.current_furniture_id] + all_furniture_ids,
                exclude_room_geometry=False,  # Include floor/walls for context
                rendering_mode="furniture_selection",  # Disables grid/frame
                annotate_object_types=[],  # Disables all labels/bboxes
                render_name=f"context_input_{self.current_furniture_id}",
                # Top-down view for floor context.
                include_vertical_views=True,  # Include top view
                override_side_view_count=0,  # No side views, just top
            )

        # Get context furniture IDs from current selection.
        context_ids = (
            self.current_furniture_selection.context_furniture_ids
            if self.current_furniture_selection
            else []
        )

        # Include current furniture + validated context furniture (same pattern as
        # observe_scene).
        valid_context_ids = [
            ctx_id for ctx_id in context_ids if ctx_id in self.scene.objects
        ]
        include_objects = [self.current_furniture_id] + valid_context_ids

        # Check if furniture is articulated (has doors/drawers).
        is_articulated = furniture.metadata.get("is_articulated", False)

        # Determine elevation based on furniture type (number of support surfaces).
        # Tables with 1 surface benefit from high angle looking down at surface.
        # Shelves with multiple surfaces need low angle to see all levels.
        num_surfaces = (
            len(furniture.support_surfaces) if furniture.support_surfaces else 1
        )
        if num_surfaces == 1:
            elevation = 60.0  # High angle - looking down at table surface
        else:
            elevation = 30.0  # Low angle - see all shelf levels from front

        # Calculate camera azimuth to view the furniture's front face.
        # Furniture "front" is +Y in local frame. We need to find where that
        # points in world frame and position the camera there.
        # For a Z-rotation (yaw) of θ, the camera should be at azimuth = 90° + θ.
        rotation_matrix = furniture.transform.rotation().matrix()
        # Extract yaw (Z rotation) from rotation matrix: atan2(R[1,0], R[0,0]).
        yaw_rad = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        # Camera azimuth: 90° (front at +Y) + furniture yaw rotation.
        front_azimuth = 90.0 + math.degrees(yaw_rad)

        return self.rendering_manager.render_scene(
            scene=self.scene,
            blender_server=self.blender_server,
            include_objects=include_objects,
            exclude_room_geometry=True,  # Furniture only, no floor/walls
            rendering_mode="furniture_selection",  # Disables grid/frame
            annotate_object_types=[],  # Disables all labels/bboxes
            articulated_open=is_articulated,  # Open joints to show interior surfaces
            context_furniture_ids=valid_context_ids,  # For proper visibility in render
            render_name=f"context_input_{self.current_furniture_id}",
            # Render single angled view from furniture's front face.
            include_vertical_views=False,  # No pure top/bottom views
            override_side_view_count=1,  # Single angled view
            side_view_start_azimuth_degrees=front_azimuth,  # Front of furniture
            side_view_elevation_degrees=elevation,  # Adaptive elevation
        )

    def _get_furniture_dimensions(self, furniture) -> str:
        """Compute human-readable furniture dimensions from bbox.

        Args:
            furniture: SceneObject with bbox_min and bbox_max.

        Returns:
            Human-readable dimensions string.
        """
        if furniture.bbox_min is None or furniture.bbox_max is None:
            return "dimensions unknown"

        dims = furniture.bbox_max - furniture.bbox_min
        width, depth, height = dims[0], dims[1], dims[2]
        return f"{width:.2f}m wide × {depth:.2f}m deep × {height:.2f}m tall"

    def _generate_manipuland_context_image(self) -> Path | None:
        """Generate context image for manipuland placement.

        Renders the furniture and uses image editing API to add suggested objects.
        This provides visual guidance for the manipuland designer agent.

        Returns:
            Path to generated context image, or None if generation fails or disabled.
        """
        if not self.cfg.context_image_generation.enabled:
            return None

        render_dir = self._render_furniture_for_context()

        selection = self.current_furniture_selection
        furniture = self.scene.get_object(selection.furniture_id)

        # Select correct image based on furniture type.
        # Floor uses top-down view; other furniture uses angled front view.
        if furniture.object_type == ObjectType.FLOOR:
            render = render_dir / "0_top.png"
        else:
            render = render_dir / "0_side.png"

        try:
            return self.asset_manager.image_generator.generate_manipuland_context_image(
                reference_image_path=render,
                furniture_description=furniture.description,
                furniture_dimensions=self._get_furniture_dimensions(furniture),
                suggested_items=selection.suggested_items,
                prompt_constraints=selection.prompt_constraints,
                style_notes=selection.style_notes,
                output_path=render_dir / "context_edited.png",
            )
        except Exception as e:
            console_logger.warning(f"Context image generation failed: {e}")
            return None

    def _get_context_image_path(self) -> Path | None:
        """Get the AI-generated context image for initial design.

        Returns:
            Path to manipuland context image if available, None otherwise.
        """
        return self.manipuland_context_image_path

    def _create_designer_tools(
        self,
        current_furniture_id: UniqueID,
        support_surfaces: dict[str, SupportSurface],
    ) -> list[FunctionTool]:
        """Create designer tools with captured dependencies.

        Args:
            current_furniture_id: ID of furniture being populated.
            support_surfaces: Dictionary mapping surface_id to SupportSurface.

        Returns:
            List of tools for the designer agent.
        """
        # Get context furniture from current selection.
        context_ids = []
        if self.current_furniture_selection:
            context_ids = self.current_furniture_selection.context_furniture_ids

        vision_tools = ManipulandVisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            current_furniture_id=current_furniture_id,
            blender_server=self.blender_server,
            context_furniture_ids=context_ids,
        )
        self.manipuland_tools = ManipulandTools(
            scene=self.scene,
            asset_manager=self.asset_manager,
            cfg=self.cfg,
            current_furniture_id=current_furniture_id,
            support_surfaces=support_surfaces,
        )
        workflow_tools = WorkflowTools()

        return [
            *vision_tools.tools.values(),
            *self.manipuland_tools.tools.values(),
            *workflow_tools.tools.values(),
        ]

    def _create_designer_agent(
        self, tools: list[FunctionTool], furniture_description: str
    ) -> Agent:
        """Create designer agent with furniture-specific context.

        Args:
            tools: Tools to provide to the designer.
            furniture_description: Description of furniture being populated.

        Returns:
            Configured designer agent.
        """
        designer_config = self.cfg.agents.designer_agent
        designer_prompt_enum = ManipulandAgentPrompts[designer_config.prompt]

        # Get structured assignment context from current furniture selection.
        selection = self.current_furniture_selection
        if not selection:
            raise ValueError("No current furniture selection set")

        return super()._create_designer_agent(
            tools=tools,
            prompt_enum=designer_prompt_enum,
            furniture_description=furniture_description,
            suggested_items=selection.suggested_items,
            prompt_constraints=selection.prompt_constraints,
            style_notes=selection.style_notes,
            has_reference_image=self.manipuland_context_image_path is not None,
        )

    def _create_critic_agent(
        self, tools: list[FunctionTool], furniture_description: str
    ) -> Agent:
        """Create critic agent with furniture-specific context.

        Args:
            tools: Tools to provide to the critic.
            furniture_description: Description of furniture being populated.

        Returns:
            Configured critic agent with structured output.
        """
        critic_config = self.cfg.agents.critic_agent
        critic_prompt_enum = ManipulandAgentPrompts[critic_config.prompt]

        # Get structured assignment context from current furniture selection.
        selection = self.current_furniture_selection
        if not selection:
            raise ValueError("No current furniture selection set")

        return super()._create_critic_agent(
            tools=tools,
            prompt_enum=critic_prompt_enum,
            output_type=ManipulandCritiqueWithScores,
            furniture_description=furniture_description,
            suggested_items=selection.suggested_items,
            prompt_constraints=selection.prompt_constraints,
            style_notes=selection.style_notes,
        )

    def _create_planner_agent(
        self, tools: list[FunctionTool], furniture_description: str
    ) -> Agent:
        """Create planner agent with furniture-specific context.

        Args:
            tools: Tools to provide to the planner.
            furniture_description: Description of furniture being populated.

        Returns:
            Configured planner agent.
        """
        planner_config = self.cfg.agents.planner_agent
        planner_prompt_enum = ManipulandAgentPrompts[planner_config.prompt]
        single_threshold = self.cfg.reset_single_category_threshold
        total_threshold = self.cfg.reset_total_sum_threshold

        # Get structured assignment context from current furniture selection.
        selection = self.current_furniture_selection
        if not selection:
            raise ValueError("No current furniture selection set")

        return super()._create_planner_agent(
            tools=tools,
            prompt_enum=planner_prompt_enum,
            furniture_description=furniture_description,
            suggested_items=selection.suggested_items,
            prompt_constraints=selection.prompt_constraints,
            style_notes=selection.style_notes,
            max_critique_rounds=self.cfg.max_critique_rounds,
            reset_single_category_threshold=single_threshold,
            reset_total_sum_threshold=total_threshold,
            early_finish_min_score=self.cfg.early_finish_min_score,
        )

    def _create_tools_for_furniture(
        self, furniture_id: UniqueID
    ) -> tuple[list[FunctionTool], list[FunctionTool], list[FunctionTool]]:
        """Create tools for planner, designer, and critic.

        Args:
            furniture_id: ID of current furniture.

        Returns:
            Tuple of (planner_tools, designer_tools, critic_tools).
        """
        # Get all support surfaces for this furniture.
        furniture = self.scene.get_object(furniture_id)
        if not furniture or not furniture.support_surfaces:
            raise ValueError(f"Furniture {furniture_id} has no support surfaces")

        # Build dict mapping surface_id strings to SupportSurface objects.
        support_surfaces = {
            str(surface.surface_id): surface for surface in furniture.support_surfaces
        }

        # Create designer tools using base class helper method.
        # This ensures consistency with furniture agent architecture and includes
        # WorkflowTools for task management.
        designer_tools = self._create_designer_tools(
            current_furniture_id=furniture_id, support_surfaces=support_surfaces
        )

        # Planner gets all designer tools (same access).
        planner_tools = designer_tools

        # Create critic tools using helper method.
        critic_tools = self._create_critic_tools(furniture_id=furniture_id)

        return planner_tools, designer_tools, critic_tools

    def _get_initial_design_prompt_enum(self) -> Any:
        """Get the prompt enum for initial design instruction.

        Returns:
            Manipuland-specific initial design instruction prompt.
        """
        return ManipulandAgentPrompts.DESIGNER_INITIAL_INSTRUCTION

    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Get prompt kwargs for initial design instruction.

        Returns:
            Dict with has_reference_image flag.
        """
        return {
            "has_reference_image": self.manipuland_context_image_path is not None,
        }

    def _get_design_change_prompt_enum(self) -> Any:
        """Get the prompt enum for design change instruction.

        Returns:
            Manipuland-specific design change instruction prompt.
        """
        return ManipulandAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION

    def _get_critique_prompt_enum(self) -> Any:
        """Get the prompt enum for critic runner instruction.

        Returns:
            Manipuland-specific critic instruction prompt.
        """
        return ManipulandAgentPrompts.MANIPULAND_CRITIC_RUNNER_INSTRUCTION

    def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Set placement noise profile for manipuland tools.

        Args:
            mode: Placement noise mode (NATURAL or PERFECT).
        """
        self.manipuland_tools.set_noise_profile(mode)

    def _create_critic_tools(self, furniture_id: UniqueID) -> list[FunctionTool]:
        """Create critic tools with read-only scene access.

        Args:
            furniture_id: ID of furniture being critiqued (for context rendering).

        Returns:
            List of tools for the critic (read-only scene validation tools).
        """
        # Get context furniture from current selection.
        context_ids = []
        if self.current_furniture_selection:
            context_ids = self.current_furniture_selection.context_furniture_ids

        # Create vision tools for critic (read-only operations).
        vision_tools = ManipulandVisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            current_furniture_id=furniture_id,
            blender_server=self.blender_server,
            context_furniture_ids=context_ids,
        )

        # Critic gets read-only tools (observe only).
        # Note: check_physics is NOT included since physics_context is already
        # injected via the critique runner instruction template.
        return [
            vision_tools.tools["observe_scene"],
            self.manipuland_tools.tools["get_current_scene_state"],
        ]

    def _setup_furniture_context(self, furniture_selection: FurnitureSelection) -> None:
        """Set up per-furniture rendering and analysis context.

        Args:
            furniture_selection: Selection data for this furniture including
                suggested items, prompt constraints, and style notes.
        """
        # Clear pending images from previous furniture iteration.
        # This prevents image leakage if session callback somehow doesn't trigger.
        self.pending_images = []

        furniture_id = furniture_selection.furniture_id

        # Create per-furniture rendering manager with subdirectory.
        self.rendering_manager = RenderingManager(
            cfg=self.cfg.rendering,
            logger=self.logger,
            subdirectory=f"manipulands_{furniture_id}",
        )

        # Update scene_analyzer to use per-furniture rendering manager.
        self.scene_analyzer = SceneAnalyzer(
            vlm_service=self.vlm_service,
            rendering_manager=self.rendering_manager,
            cfg=self.cfg,
            blender_server=self.blender_server,
        )

        # Store current furniture selection for agent creation.
        self.current_furniture_id = furniture_id
        self.current_furniture_selection = furniture_selection

    def _initialize_checkpoint_state(self) -> None:
        """Reset checkpoint state for new furniture iteration.

        Called at the start of each furniture iteration to clear checkpoint
        state from the previous furniture piece. The attributes themselves
        were initialized in __init__().
        """
        # Reset checkpoint state to None for new furniture iteration.
        self.previous_scene_checkpoint = None
        self.scene_checkpoint = None
        self.previous_checkpoint_scores = None
        self.checkpoint_scores = None
        self.previous_scores = None
        self.previous_checkpoint_render_dir = None
        self.checkpoint_render_dir = None
        # Keep placement_style as-is (it persists across furniture iterations).

    def _setup_furniture_agents(
        self, furniture_id: UniqueID, furniture_description: str
    ) -> None:
        """Create agents and sessions for this furniture piece.

        Args:
            furniture_id: ID of furniture being populated.
            furniture_description: Human-readable furniture description.
        """
        # Create fresh tools and agents for this furniture.
        # First create designer/critic tools.
        (
            _,  # planner_tools created later after agents exist
            designer_tools,
            critic_tools,
        ) = self._create_tools_for_furniture(furniture_id)

        # Create sessions using base class helper.
        # Sessions are stored as instance variables for planner tool closures.
        self.designer_session, self.critic_session = self._create_sessions(
            session_prefix=f"{furniture_id}_"
        )

        # Create agents using base class helpers with override methods.
        self.designer = self._create_designer_agent(
            tools=designer_tools, furniture_description=furniture_description
        )

        self.critic = self._create_critic_agent(
            tools=critic_tools, furniture_description=furniture_description
        )

        # Now create planner tools (can reference self.designer/critic/sessions).
        planner_tools = self._create_planner_tools()

        # Create planner agent using base class helper with override method.
        self.planner = self._create_planner_agent(
            tools=planner_tools, furniture_description=furniture_description
        )

    async def _run_furniture_workflow(self, furniture_id: UniqueID) -> None:
        """Execute the multi-agent workflow for a furniture piece.

        Args:
            furniture_id: ID of furniture being populated.
        """
        # Get runner instruction for planner to start workflow.
        planner_runner_prompt = (
            ManipulandAgentPrompts.MANIPULAND_PLANNER_RUNNER_INSTRUCTION
        )
        runner_instruction = self.prompt_registry.get_prompt(
            prompt_enum=planner_runner_prompt,
        )

        result: RunResult = await Runner.run(
            starting_agent=self.planner,
            input=runner_instruction,
            max_turns=self.cfg.agents.planner_agent.max_turns,
            run_config=self._create_run_config(),
        )
        log_agent_usage(result=result, agent_name="PLANNER (MANIPULAND)")

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="PLANNER (MANIPULAND)"
            )

        # Compute final critique and scores for completed furniture.
        # Check if scene changed since last checkpoint to avoid redundant critique.
        current_scene_hash = self.scene.content_hash()

        if (
            self.checkpoint_scene_hash is not None
            and current_scene_hash == self.checkpoint_scene_hash
        ):
            console_logger.info(
                "Scene unchanged since last critique, skipping final critique"
            )
        else:
            console_logger.info(
                "Scene changed since last critique, computing final critique"
            )
            # Pass update_checkpoint=False to preserve N-1 checkpoint for reset check.
            await self._request_critique_impl(update_checkpoint=False)

        # Validate final scene and save scores.
        await self._finalize_scene_and_scores()

        console_logger.info(
            f"Completed manipuland placement for furniture {furniture_id}"
        )

    def _get_final_scores_directory(self) -> Path:
        """Get the directory path for saving per-furniture manipuland placement state.

        Returns:
            Path to scene_states/manipuland_furniture_{id} directory.
        """
        return (
            self.logger.output_dir
            / "scene_states"
            / f"manipuland_furniture_{self.current_furniture_id}"
        )

    async def add_manipulands(self, scene: RoomScene) -> None:
        """Add manipulands to furniture surfaces in the scene.

        This method implements a two-phase workflow:
        1. VLM-based furniture analysis to identify which pieces need manipulands
        2. Per-furniture multi-agent workflow (planner/designer/critic) to
           populate selected furniture with appropriate small objects

        The scene is mutated in place to add manipuland objects. Fresh agent
        contexts are created for each furniture piece to bound token usage.

        Side effects:
        - Scene objects are added (manipulands placed on furniture)
        - Support surfaces are extracted and assigned to furniture
        - Render cache is cleared before processing
        - Per-furniture subdirectories created under logger output directory
        - Checkpoint state saved after each critique iteration
        - Final scores copied to furniture_<id>/final_scene/ directories

        Requirements:
        - Furniture must have geometry_path (non-None)
        - Furniture must have valid bounding boxes (bbox_min, bbox_max)
        - Scene must have text_description for agent context

        Args:
            scene: RoomScene with furniture already placed. Furniture objects must
                have geometry and bounding boxes to be considered for manipuland
                placement.

        Raises:
            Exception: If support surface extraction fails (indicates invalid
                furniture geometry). Agent execution errors are logged but do
                not halt processing of remaining furniture.
        """
        try:
            console_logger.info("Starting manipuland placement")
            self.scene = scene

            # Clear render cache to ensure fresh renders for manipulands.
            # This prevents cache key collisions when object IDs are reused.
            self.rendering_manager.clear_cache()

            # Phase 1: Initial analysis - identify which furniture to populate.
            furniture_data = await self._analyze_furniture_for_placement(scene)

            if not furniture_data:
                console_logger.info("No furniture identified for manipuland placement")
                return

            console_logger.info(
                f"Identified {len(furniture_data)} furniture pieces to populate"
            )

            # Phase 1b: Select context furniture for each selection.
            if self.cfg.context_furniture.enabled:
                # Get path to furniture_selection images (already rendered).
                furniture_selection_dir = (
                    self.rendering_manager._base_output_dir
                    / "scene_renders"
                    / "furniture_selection"
                )
                images_dir = (
                    furniture_selection_dir if furniture_selection_dir.exists() else None
                )

                context_map = self.scene_analyzer.select_context_furniture(
                    scene=scene,
                    furniture_selections=furniture_data,
                    furniture_selection_images_dir=images_dir,
                )

                # Attach context to each selection.
                for selection in furniture_data:
                    selection.context_furniture_ids = context_map.get(
                        selection.furniture_id, []
                    )

            # Phase 2: Per-furniture loop.
            for furniture_selection in furniture_data:
                furniture_id = furniture_selection.furniture_id
                # Create custom span for this furniture's manipuland placement.
                with custom_span(
                    name=f"manipulands_{furniture_id}",
                    data={"furniture_id": str(furniture_id)},
                ):
                    console_logger.info(f"Populating furniture: {furniture_id}")
                    if furniture_selection.suggested_items:
                        console_logger.info(
                            f"Suggested items: {furniture_selection.suggested_items}"
                        )
                        console_logger.info(
                            f"Prompt constraints: {furniture_selection.prompt_constraints}"
                        )
                        console_logger.info(
                            f"Style notes: {furniture_selection.style_notes}"
                        )

                    # Extract support surface for this furniture.
                    furniture = scene.get_object(furniture_id)
                    if not furniture:
                        console_logger.warning(
                            f"Furniture {furniture_id} not found, skipping"
                        )
                        continue

                    # Extract all support surfaces using HSM algorithm.
                    hsm_config = SupportSurfaceExtractionConfig.from_config(
                        cfg=self.cfg.support_surface_extraction
                    )
                    surfaces = extract_and_propagate_support_surfaces(
                        scene=self.scene, furniture_object=furniture, config=hsm_config
                    )

                    console_logger.info(
                        f"Extracted {len(surfaces)} support surface(s) for {furniture_id}"
                    )

                    # Skip furniture with no support surfaces (e.g., plants, unsuitable geometry).
                    if not surfaces:
                        console_logger.warning(
                            f"No support surfaces found for {furniture_id}, skipping manipuland placement"
                        )
                        continue

                    try:
                        # Set up per-furniture context.
                        self._setup_furniture_context(furniture_selection)

                        # Generate context image for manipuland placement (if enabled).
                        self.manipuland_context_image_path = (
                            self._generate_manipuland_context_image()
                        )

                        # Initialize checkpoint state.
                        self._initialize_checkpoint_state()

                        # Get furniture description for agent prompts.
                        furniture_obj = scene.get_object(furniture_id)
                        furniture_description = (
                            furniture_obj.description if furniture_obj else "furniture"
                        )

                        # Create agents and sessions.
                        self._setup_furniture_agents(
                            furniture_id=furniture_id,
                            furniture_description=furniture_description,
                        )

                        # Run multi-agent workflow.
                        await self._run_furniture_workflow(furniture_id)

                        # Per-furniture post-processing (after manipulands placed).
                        if self.cfg.per_furniture_postprocessing.enabled:
                            sim_cfg = self.cfg.per_furniture_postprocessing.simulation
                            sim_html_path = None
                            if sim_cfg.save_html:
                                sim_html_path = (
                                    self.scene.scene_dir
                                    / "simulation"
                                    / "per_furniture"
                                    / f"{furniture_id}_simulation.html"
                                )
                            self.scene = apply_per_furniture_postprocessing(
                                full_scene=self.scene,
                                furniture_id=furniture_id,
                                config=self.cfg.per_furniture_postprocessing,
                                simulation_html_path=sim_html_path,
                            )

                    except Exception as e:
                        console_logger.error(
                            f"Error populating furniture {furniture_id}: {e}", exc_info=True
                        )
                        # Continue to next furniture piece.
                        continue

            console_logger.info("Manipuland placement complete")
        finally:
            await self._cleanup_async_resources()

    async def _analyze_furniture_for_placement(
        self, scene: RoomScene
    ) -> list[FurnitureSelection]:
        """Analyze which furniture should have manipulands.

        Delegates to SceneAnalyzer for VLM-based furniture selection.

        Args:
            scene: RoomScene with furniture.

        Returns:
            List of FurnitureSelection objects with assignment context.
        """
        return self.scene_analyzer.analyze_furniture_for_manipulands(
            scene=scene,
            prompt_enum=ManipulandAgentPrompts.ANALYZE_FURNITURE_FOR_PLACEMENT,
        )
