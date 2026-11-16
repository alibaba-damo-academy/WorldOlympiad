"""Unified policy interface agent.

Single agent that handles the entire task → object bindings pipeline:
1. Parses task to understand goals and preconditions
2. Finds objects in scene that match categories
3. Verifies preconditions using state/vision tools
4. Returns ALL valid (target, reference) pairs ranked by confidence
"""

import logging

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agents import Agent, Runner, RunResult
from omegaconf import DictConfig
from pydantic import BaseModel, Field

from scenecode.prompts import RobotEvalPrompts, prompt_registry
from scenecode.robot_eval.dmd_scene import DMDScene
from scenecode.robot_eval.tools import create_state_tools, create_vision_tools
from scenecode.utils.openai import (
    close_cached_async_openai_clients,
    create_run_config,
)

if TYPE_CHECKING:
    from scenecode.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)


class ObjectBinding(BaseModel):
    """A valid (target, reference) binding for a task."""

    target_id: str = Field(description="Object ID of the target (what to move)")
    reference_id: str = Field(description="Object ID of the reference (where to place)")
    rank: int = Field(description="Agent's preference ranking (1 = best)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    reasoning: str = Field(description="Why this binding satisfies the task")


class PolicyInterfaceOutput(BaseModel):
    """Structured output from the policy interface agent."""

    task_description: str = Field(description="Original task being resolved")

    # Parsed task components.
    goal_predicate: str = Field(
        description="Goal predicate type: 'on', 'inside', 'near'"
    )
    target_category: str = Field(description="Category of object to move")
    reference_category: str = Field(description="Category of placement location")
    target_precondition: str | None = Field(
        default=None,
        description="Extracted precondition for target (e.g., 'on floor')",
    )
    reference_precondition: str | None = Field(
        default=None,
        description="Extracted precondition for reference (e.g., 'on dining table')",
    )

    # Valid bindings.
    valid_bindings: list[ObjectBinding] = Field(
        description="Ranked list of valid (target, reference) bindings"
    )
    overall_success: bool = Field(
        description="True if at least one valid binding was found"
    )
    notes: list[str] = Field(
        default_factory=list, description="Additional notes from agent"
    )


@dataclass
class PolicyInterfaceAgent:
    """Unified agent for robot task planning.

    Takes a task description and scene, returns ranked object bindings.
    Combines the functionality of PredicateExtractor, CategoryMatcher,
    and CandidateSelectionAgent into a single coherent reasoning pass.

    Usage:
        scene = load_scene_for_validation(scene_state_path, dmd_path)
        scene.finalize()

        agent = PolicyInterfaceAgent(scene=scene, cfg=cfg)
        result = await agent.resolve("Pick a cup from the floor and put it in the sink")
    """

    scene: DMDScene
    """Scene with finalized Drake plant and scene_state metadata."""

    cfg: DictConfig
    """Configuration with model settings."""

    blender_server: "BlenderServer | None" = None
    """Optional Blender server for vision tools. If None, only state tools available."""

    _agent: Agent | None = field(default=None, init=False)
    """Lazily initialized agent."""

    def _create_agent(self) -> Agent:
        """Create the policy interface agent with tools and prompt."""
        # State tools (always available).
        state_tools = create_state_tools(self.scene)

        # Vision tools (if BlenderServer provided).
        if self.blender_server is not None:
            vision_tools = create_vision_tools(
                scene=self.scene, blender_server=self.blender_server
            )
            all_tools = state_tools + vision_tools
        else:
            console_logger.warning("No BlenderServer provided - vision tools disabled")
            all_tools = state_tools

        prompt = prompt_registry.get_prompt(RobotEvalPrompts.POLICY_AGENT)

        # Create agent with structured output.
        return Agent(
            name="policy_interface",
            model=self.cfg.openai.model,
            tools=all_tools,
            instructions=prompt,
            output_type=PolicyInterfaceOutput,
        )

    @property
    def agent(self) -> Agent:
        """Get or create the policy interface agent."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    async def resolve(
        self, task_description: str, max_turns: int = 1000
    ) -> PolicyInterfaceOutput:
        """Resolve a task to valid object bindings.

        Args:
            task_description: Natural language task (e.g., "Pick a cup from the floor").
            max_turns: Maximum agent turns before forcing output.

        Returns:
            PolicyInterfaceOutput with ranked valid bindings.
        """
        console_logger.info(f"Resolving task: {task_description}")

        try:
            result: RunResult = await Runner.run(
                starting_agent=self.agent,
                input=task_description,
                max_turns=max_turns,
                run_config=create_run_config(api_base=self.cfg.openai.api_base),
            )
            output = result.final_output_as(PolicyInterfaceOutput)

            console_logger.info(
                f"Resolved {len(output.valid_bindings)} bindings for "
                f"{output.goal_predicate}({output.target_category}, {output.reference_category})"
            )
            return output
        finally:
            await close_cached_async_openai_clients()
