"""Success validator agent for evaluating task completion.

Uses an LLM agent with state and vision tools to determine whether
a task has been successfully completed. Returns structured validation
results with per-requirement scores and explanations.
"""

import logging

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from agents import Agent, Runner, RunResult
from omegaconf import DictConfig
from pydantic import BaseModel, Field

from scenecode.prompts import RobotEvalPrompts, prompt_registry
from scenecode.robot_eval.dmd_scene import DMDScene, load_scene_for_validation
from scenecode.robot_eval.tools import create_state_tools, create_vision_tools
from scenecode.utils.openai import create_run_config

if TYPE_CHECKING:
    from scenecode.agent_utils.blender.server_manager import BlenderServer

console_logger = logging.getLogger(__name__)


class RequirementScore(str, Enum):
    """Discrete scores for consistent LLM output.

    Using discrete options rather than arbitrary floats produces more
    consistent results from LLMs.
    """

    NONE = "none"
    """Requirement not satisfied (0.0)."""

    PARTIAL = "partial"
    """Partially satisfied - technically there but not right (0.5)."""

    FULL = "full"
    """Fully satisfied (1.0)."""

    def to_float(self) -> float:
        """Convert to numeric score."""
        if self == RequirementScore.NONE:
            return 0.0
        elif self == RequirementScore.PARTIAL:
            return 0.5
        return 1.0


class RequirementResult(BaseModel):
    """Result for a single requirement extracted from the task."""

    description: str = Field(
        description="What requirement is being checked (e.g., 'plate on table')"
    )
    score: RequirementScore = Field(
        description="How well the requirement is satisfied: none, partial, or full"
    )
    reasoning: str = Field(
        description="Explanation for the score with evidence from state/vision tools"
    )


class ValidationResult(BaseModel):
    """Structured output from the validator agent."""

    task_description: str = Field(description="The original task being validated")
    requirements: list[RequirementResult] = Field(
        description="Per-requirement scores and reasoning"
    )
    overall_reasoning: str = Field(
        description="Summary of overall task completion status"
    )

    @property
    def overall_score(self) -> float:
        """Average of requirement scores (0.0 - 1.0)."""
        if not self.requirements:
            return 0.0
        total = sum(r.score.to_float() for r in self.requirements)
        return total / len(self.requirements)

    @property
    def overall_success(self) -> bool:
        """Whether task is considered successful (score >= 0.9)."""
        return self.overall_score >= 0.9


@dataclass
class SuccessValidatorAgent:
    """LLM agent that validates task success using state + vision tools.

    The validator:
    1. Receives the original human task description
    2. Extracts explicit requirements from the task
    3. Uses state tools to gather geometric facts
    4. Uses vision tools to observe the scene
    5. Reasons about whether requirements are satisfied
    6. Returns structured validation results

    Usage:
        scene = load_scene_from_dir(scene_dir, task_description="...")
        scene.finalize()

        validator = SuccessValidatorAgent(scene=scene, cfg=cfg)
        result = await validator.validate("Place the cup on the table")
    """

    scene: DMDScene
    """Scene to validate."""

    cfg: DictConfig
    """Configuration with model settings."""

    blender_server: "BlenderServer | None" = None
    """Optional Blender server for generating renders. If None, vision tools
    are not available and validation uses only state tools."""

    render_id: str | None = None
    """Optional unique ID for render output directories. Use when running
    multiple validations in parallel to avoid file conflicts."""

    _agent: Agent | None = field(default=None, init=False)
    """Lazily initialized agent."""

    def _create_agent(self) -> Agent:
        """Create the validator agent with tools and prompt."""
        # Create state tools (always available).
        state_tools = create_state_tools(self.scene)

        # Create vision tools only if blender_server is provided.
        if self.blender_server is not None:
            vision_tools = create_vision_tools(
                scene=self.scene,
                blender_server=self.blender_server,
                render_id=self.render_id,
            )
            all_tools = state_tools + vision_tools
        else:
            console_logger.warning(
                "No BlenderServer provided - vision tools disabled for validation"
            )
            all_tools = state_tools

        prompt = prompt_registry.get_prompt(
            prompt_enum=RobotEvalPrompts.SUCCESS_VALIDATOR,
        )

        # Create agent with structured output.
        return Agent(
            name="success_validator",
            model=self.cfg.openai.model,
            tools=all_tools,
            instructions=prompt,
            output_type=ValidationResult,
        )

    @property
    def agent(self) -> Agent:
        """Get or create the validator agent."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    async def validate(
        self, task_description: str, max_turns: int = 1000
    ) -> ValidationResult:
        """Validate whether the task has been completed successfully.

        Args:
            task_description: The human natural language task to validate.
            max_turns: Maximum number of agent turns before forcing output.

        Returns:
            ValidationResult with per-requirement scores and overall assessment.
        """
        console_logger.info(f"Validating task: {task_description[:100]}...")

        # Build the validation instruction from prompt registry.
        instruction = prompt_registry.get_prompt(
            RobotEvalPrompts.VALIDATION_INSTRUCTION,
            task_description=task_description,
        )

        # Run the agent.
        result: RunResult = await Runner.run(
            starting_agent=self.agent,
            input=instruction,
            max_turns=max_turns,
            run_config=create_run_config(api_base=self.cfg.openai.api_base),
        )

        # Extract structured output.
        validation_result = result.final_output_as(ValidationResult)

        # Log results.
        console_logger.info(
            f"Validation complete: score={validation_result.overall_score:.2f}, "
            f"success={validation_result.overall_success}"
        )
        for req in validation_result.requirements:
            console_logger.info(
                f"  [{req.score.value}] {req.description}: {req.reasoning[:100]}..."
            )

        return validation_result


async def validate_task(
    task_description: str,
    cfg: DictConfig,
    scene_state_path: Path,
    dmd_path: Path,
    blender_server: "BlenderServer | None" = None,
    scene_dir: Path | None = None,
    render_id: str | None = None,
) -> ValidationResult:
    """Validate a task after robot execution.

    Args:
        task_description: Task to validate.
        cfg: Configuration with model settings.
        scene_state_path: Path to scene_state.json (object metadata from scenecode).
        dmd_path: Path to scene.dmd.yaml (poses from robot output).
        blender_server: Optional Blender server for rendering.
        scene_dir: Scene root directory for package:// URI resolution.
        render_id: Optional unique ID for render output directories. Use when
            running multiple validations in parallel to avoid file conflicts.

    Returns:
        ValidationResult with scores and reasoning.
    """
    scene = load_scene_for_validation(
        scene_state_path=scene_state_path,
        dmd_path=dmd_path,
        task_description=task_description,
        scene_dir=scene_dir,
    )
    scene.finalize()

    validator = SuccessValidatorAgent(
        scene=scene, cfg=cfg, blender_server=blender_server, render_id=render_id
    )

    return await validator.validate(task_description)
