"""Stateful ceiling-mounted object placement agent.

This module implements the main ceiling agent using the planner/designer/critic
trio architecture shared with furniture, wall, and manipuland agents.

The ceiling agent is per-room (like furniture agent), processing the entire
ceiling in a single session to enable coherent lighting layout.

Pipeline order: floor_plan -> furniture -> wall-mounted -> ceiling-mounted -> manipulands
"""

import logging

from pathlib import Path
from typing import Any

from agents import Agent, FunctionTool, Runner
from agents.run import RunResult
from agents.tracing import custom_span
from omegaconf import DictConfig

from scenecode.agent_utils.base_stateful_agent import (
    BaseStatefulAgent,
    log_agent_usage,
)
from scenecode.agent_utils.placement_noise import PlacementNoiseMode
from scenecode.agent_utils.room import AgentType, RoomScene
from scenecode.agent_utils.scoring import CeilingCritiqueWithScores, log_agent_response
from scenecode.agent_utils.workflow_tools import WorkflowTools
from scenecode.ceiling_agents.base_ceiling_agent import BaseCeilingAgent
from scenecode.ceiling_agents.tools.ceiling_tools import CeilingTools
from scenecode.ceiling_agents.tools.vision_tools import CeilingVisionTools
from scenecode.prompts.registry import CeilingAgentPrompts
from scenecode.utils.logging import BaseLogger

console_logger = logging.getLogger(__name__)


class StatefulCeilingAgent(BaseStatefulAgent, BaseCeilingAgent):
    """Ceiling-mounted object placement with planner/designer/critic agents.

    Workflow:
    1. Extract room bounds from scene geometry
    2. Create agents for ceiling decoration (single session)
    3. Planner coordinates designer/critic for balanced lighting layout
    4. Agent-driven termination: Planner decides when ceiling is complete

    Like furniture and wall agents, ceiling agent operates per-room
    to enable reasoning about the entire ceiling simultaneously.
    """

    @property
    def agent_type(self) -> AgentType:
        """Return agent type for collision filtering."""
        return AgentType.CEILING_MOUNTED

    def __init__(
        self,
        cfg: DictConfig,
        logger: BaseLogger,
        ceiling_height: float,
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
        """Initialize stateful ceiling agent.

        Args:
            cfg: Configuration object containing ceiling agent settings.
            logger: Logger instance for tracking operations.
            ceiling_height: Height of ceiling in meters.
            geometry_server_host: Host for geometry generation server.
            geometry_server_port: Port for geometry generation server.
            hssd_server_host: Host for HSSD retrieval server.
            hssd_server_port: Port for HSSD retrieval server.
            articulated_server_host: Host for articulated retrieval server.
            articulated_server_port: Port for articulated retrieval server.
            materials_server_host: Deprecated compatibility parameter; ignored.
            materials_server_port: Deprecated compatibility parameter; ignored.
            num_workers: Number of parallel workers.
            render_gpu_id: GPU device ID for Blender rendering. When set, uses
                bubblewrap to isolate the BlenderServer to this GPU.
        """
        # Initialize base stateful placement agent (sessions, checkpoint state).
        BaseStatefulAgent.__init__(
            self,
            cfg=cfg,
            logger=logger,
            geometry_server_host=geometry_server_host,
            geometry_server_port=geometry_server_port,
            hssd_server_host=hssd_server_host,
            hssd_server_port=hssd_server_port,
        )

        # Initialize ceiling-specific base class.
        BaseCeilingAgent.__init__(
            self,
            cfg=cfg,
            logger=logger,
            ceiling_height=ceiling_height,
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

        # Ceiling tools will be set when adding ceiling objects.
        self.ceiling_tools: CeilingTools | None = None

    def _create_designer_tools(self) -> list[FunctionTool]:
        """Create designer tools with captured dependencies.

        Returns:
            List of tools for the designer agent.
        """
        vision_tools = CeilingVisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            room_bounds=self.room_bounds,
            ceiling_height=self.ceiling_height,
            cfg=self.cfg,
            blender_server=self.blender_server,
        )
        self.ceiling_tools = CeilingTools(
            scene=self.scene,
            room_bounds=self.room_bounds,
            ceiling_height=self.ceiling_height,
            asset_manager=self.asset_manager,
            cfg=self.cfg,
        )
        workflow_tools = WorkflowTools()

        return [
            *vision_tools.tools.values(),
            *self.ceiling_tools.tools.values(),
            *workflow_tools.tools.values(),
        ]

    def _create_designer_agent(
        self, tools: list[FunctionTool], room_description: str
    ) -> Agent:
        """Create designer agent with room-specific context.

        Args:
            tools: Tools to provide to the designer.
            room_description: Description of the room being decorated.

        Returns:
            Configured designer agent.
        """
        designer_config = self.cfg.agents.designer_agent
        designer_prompt_enum = CeilingAgentPrompts[designer_config.prompt]

        min_x, min_y, max_x, max_y = self.room_bounds
        room_width = max_x - min_x
        room_depth = max_y - min_y

        return super()._create_designer_agent(
            tools=tools,
            prompt_enum=designer_prompt_enum,
            room_description=room_description,
            room_width=room_width,
            room_depth=room_depth,
            ceiling_height=self.ceiling_height,
        )

    def _create_critic_tools(self) -> list[FunctionTool]:
        """Create critic tools with read-only scene access.

        Returns:
            List of tools for the critic (read-only scene validation tools).
        """
        vision_tools = CeilingVisionTools(
            scene=self.scene,
            rendering_manager=self.rendering_manager,
            room_bounds=self.room_bounds,
            ceiling_height=self.ceiling_height,
            cfg=self.cfg,
            blender_server=self.blender_server,
        )

        # Critic gets read-only tools plus scene state.
        # Note: check_physics is NOT included since physics_context is already
        # injected via the critique runner instruction template.
        return [
            vision_tools.tools["observe_scene"],
            self.ceiling_tools.tools["get_current_scene_state"],
        ]

    def _create_critic_agent(
        self, tools: list[FunctionTool], room_description: str
    ) -> Agent:
        """Create critic agent with room-specific context.

        Args:
            tools: Tools to provide to the critic.
            room_description: Description of the room being decorated.

        Returns:
            Configured critic agent with structured output.
        """
        critic_config = self.cfg.agents.critic_agent
        critic_prompt_enum = CeilingAgentPrompts[critic_config.prompt]

        min_x, min_y, max_x, max_y = self.room_bounds
        room_width = max_x - min_x
        room_depth = max_y - min_y

        return super()._create_critic_agent(
            tools=tools,
            prompt_enum=critic_prompt_enum,
            output_type=CeilingCritiqueWithScores,
            room_description=room_description,
            room_width=room_width,
            room_depth=room_depth,
            ceiling_height=self.ceiling_height,
        )

    def _create_planner_agent(
        self, tools: list[FunctionTool], room_description: str
    ) -> Agent:
        """Create planner agent with room-specific context.

        Args:
            tools: Tools to provide to the planner.
            room_description: Description of the room being decorated.

        Returns:
            Configured planner agent.
        """
        planner_config = self.cfg.agents.planner_agent
        planner_prompt_enum = CeilingAgentPrompts[planner_config.prompt]
        single_threshold = self.cfg.reset_single_category_threshold
        total_threshold = self.cfg.reset_total_sum_threshold

        min_x, min_y, max_x, max_y = self.room_bounds
        room_width = max_x - min_x
        room_depth = max_y - min_y

        return super()._create_planner_agent(
            tools=tools,
            prompt_enum=planner_prompt_enum,
            room_description=room_description,
            room_width=room_width,
            room_depth=room_depth,
            ceiling_height=self.ceiling_height,
            max_critique_rounds=self.cfg.max_critique_rounds,
            reset_single_category_threshold=single_threshold,
            reset_total_sum_threshold=total_threshold,
            early_finish_min_score=self.cfg.early_finish_min_score,
        )

    def _get_initial_design_prompt_enum(self) -> Any:
        """Get the prompt enum for initial design instruction.

        Returns:
            Ceiling-specific initial design instruction prompt.
        """
        return CeilingAgentPrompts.DESIGNER_INITIAL_INSTRUCTION

    def _get_initial_design_prompt_kwargs(self) -> dict:
        """Get prompt kwargs for initial design instruction.

        Returns:
            Dictionary with room bounds information.
        """
        min_x, min_y, max_x, max_y = self.room_bounds
        room_width = max_x - min_x
        room_depth = max_y - min_y

        return {
            "room_width": room_width,
            "room_depth": room_depth,
            "ceiling_height": self.ceiling_height,
        }

    def _get_design_change_prompt_enum(self) -> Any:
        """Get the prompt enum for design change instruction.

        Returns:
            Ceiling-specific design change instruction prompt.
        """
        return CeilingAgentPrompts.DESIGNER_CRITIQUE_INSTRUCTION

    def _get_critique_prompt_enum(self) -> Any:
        """Get the prompt enum for critic runner instruction.

        Returns:
            Ceiling-specific critic instruction prompt.
        """
        return CeilingAgentPrompts.STATEFUL_CRITIC_RUNNER_INSTRUCTION

    def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
        """Set placement noise profile for ceiling tools.

        Args:
            mode: Placement noise mode (NATURAL or PERFECT).
        """
        if self.ceiling_tools is not None:
            self.ceiling_tools.set_noise_profile(mode=mode)

    def _get_final_scores_directory(self) -> Path:
        """Get the directory path for saving ceiling placement state.

        Returns:
            Path to scene_states/ceiling directory.
        """
        return self.logger.output_dir / "scene_states" / "ceiling"

    def _setup_ceiling_agents(self, room_description: str) -> None:
        """Create agents and sessions for ceiling decoration.

        Args:
            room_description: Human-readable room description.
        """
        # Create designer tools first.
        designer_tools = self._create_designer_tools()

        # Create sessions using base class helper.
        self.designer_session, self.critic_session = self._create_sessions(
            session_prefix="ceiling_"
        )

        # Create agents using base class helpers with override methods.
        self.designer = self._create_designer_agent(
            tools=designer_tools, room_description=room_description
        )

        # Create critic tools (needs ceiling_tools to be set first).
        critic_tools = self._create_critic_tools()
        self.critic = self._create_critic_agent(
            tools=critic_tools, room_description=room_description
        )

        # Create planner tools (can reference self.designer/critic/sessions).
        planner_tools = self._create_planner_tools()

        # Create planner agent using base class helper.
        self.planner = self._create_planner_agent(
            tools=planner_tools, room_description=room_description
        )

    async def _run_ceiling_workflow(self) -> None:
        """Execute the multi-agent workflow for ceiling decoration."""
        # Get runner instruction for planner to start workflow.
        planner_runner_prompt = CeilingAgentPrompts.STATEFUL_PLANNER_RUNNER_INSTRUCTION
        runner_instruction = self.prompt_registry.get_prompt(
            prompt_enum=planner_runner_prompt,
        )

        result: RunResult = await Runner.run(
            starting_agent=self.planner,
            input=runner_instruction,
            max_turns=self.cfg.agents.planner_agent.max_turns,
            run_config=self._create_run_config(),
        )
        log_agent_usage(result=result, agent_name="PLANNER (CEILING)")

        if result.final_output:
            log_agent_response(
                response=result.final_output, agent_name="PLANNER (CEILING)"
            )

        # Compute final critique and scores.
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
            await self._request_critique_impl(update_checkpoint=False)

        # Validate final scene and save scores.
        await self._finalize_scene_and_scores()

        console_logger.info("Ceiling decoration workflow complete")

    async def add_ceiling_objects(self, scene: RoomScene) -> None:
        """Add ceiling-mounted objects to a scene.

        This method implements the ceiling decoration workflow:
        1. Extract room bounds from scene geometry
        2. Create agents for ceiling decoration (single session)
        3. Run planner/designer/critic workflow
        4. Save final scores and scene state

        The scene is mutated in place to add ceiling-mounted objects.

        Side effects:
        - Scene objects are added (ceiling-mounted objects)
        - Render cache is cleared before processing
        - Checkpoint state saved after each critique iteration
        - Final scores saved to ceiling/ directory

        Args:
            scene: RoomScene with furniture already placed. The scene is
                mutated in place to add ceiling-mounted objects.
        """
        try:
            console_logger.info("Starting ceiling decoration")
            self.scene = scene

            # Extract room bounds.
            self.room_bounds = self._extract_room_bounds(scene=scene)
            min_x, min_y, max_x, max_y = self.room_bounds
            room_width = max_x - min_x
            room_depth = max_y - min_y

            console_logger.info(
                f"Room bounds: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f}), "
                f"size: {room_width:.1f}m x {room_depth:.1f}m, "
                f"ceiling height: {self.ceiling_height:.1f}m"
            )

            # Clear render cache to ensure fresh renders.
            self.rendering_manager.clear_cache()

            with custom_span(
                name="ceiling_decoration",
                data={
                    "room_id": scene.room_id,
                    "room_width": room_width,
                    "room_depth": room_depth,
                    "ceiling_height": self.ceiling_height,
                },
            ):
                try:
                    # Get room description for agent context.
                    room_description = (
                        scene.text_description
                        if scene.text_description
                        else f"Room {scene.room_id}"
                    )

                    # Create agents and sessions.
                    self._setup_ceiling_agents(room_description=room_description)

                    # Run multi-agent workflow.
                    await self._run_ceiling_workflow()

                except Exception as e:
                    console_logger.error(
                        f"Error during ceiling decoration for {scene.room_id}: {e}",
                        exc_info=True,
                    )
                    raise

            console_logger.info("Ceiling decoration complete")
        finally:
            await self._cleanup_async_resources()
