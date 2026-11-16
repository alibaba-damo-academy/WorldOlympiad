"""Scene prompt generator for task-based scene generation.

Converts human natural language task descriptions into diverse scene prompts
that enable the task without pre-solving it.
"""

import csv
import logging

from dataclasses import dataclass
from pathlib import Path

import yaml

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from scenecode.prompts import RobotEvalPrompts, prompt_registry
from scenecode.robot_eval.llm_utils import close_cached_clients, structured_llm_call

console_logger = logging.getLogger(__name__)


class TaskAnalysis(BaseModel):
    """Analysis of a human task's requirements and flexibility."""

    room_requirement: str | None = Field(
        default=None,
        description="Required room type if specified (e.g., 'kitchen', 'bedroom'), "
        "or null if any room works",
    )
    required_objects: list[str] = Field(
        description="Objects that MUST be present for the task "
        "(e.g., ['fruit', 'table'])",
    )
    flexible_dimensions: list[str] = Field(
        description="Aspects that CAN vary across scene prompts "
        "(e.g., ['style', 'additional_furniture', 'object_variants'])",
    )
    initial_state_constraint: str = Field(
        description="What must NOT be true initially to avoid pre-solving the task "
        "(e.g., 'fruit must NOT already be on table')",
    )


class ScenePrompt(BaseModel):
    """A single scene prompt with metadata."""

    prompt: str = Field(
        description="Natural language prompt for scene generation. Should be 1-5 "
        "sentences describing the room, required objects in non-goal positions, "
        "and additional contextual objects.",
    )
    style_variant: str = Field(
        description="Brief label for the style/variant of this prompt "
        "(e.g., 'modern', 'rustic', 'minimalist', 'Star Wars')",
    )


class ScenePromptGeneratorOutput(BaseModel):
    """Structured output from the scene prompt generator."""

    task_description: str = Field(
        description="The original human task (preserved for validator)",
    )
    analysis: TaskAnalysis = Field(
        description="Analysis of task requirements and flexibility",
    )
    scene_prompts: list[ScenePrompt] = Field(
        description="N diverse scene prompts that enable the task",
    )

    @property
    def prompt_texts(self) -> list[str]:
        """Get just the prompt text strings."""
        return [sp.prompt for sp in self.scene_prompts]


@dataclass
class ScenePromptGenerator:
    """LLM that generates diverse scene prompts from human tasks.

    The generator:
    1. Analyzes the human task to extract requirements and flexibility
    2. Identifies initial state constraints (don't pre-solve the task!)
    3. Generates N diverse scene prompts varying flexible dimensions
    4. Outputs prompts compatible with scenecode's CSV input format

    Usage:
        generator = ScenePromptGenerator(cfg=cfg, num_prompts=5)
        output = await generator.generate("Find a fruit and place it on the table")
        generator.write_prompts_csv(output, output_dir)
    """

    cfg: DictConfig
    """Configuration with model settings."""

    num_prompts: int = 5
    """Number of diverse scene prompts to generate."""

    async def generate(self, task_description: str) -> ScenePromptGeneratorOutput:
        """Generate diverse scene prompts from a human task.

        Args:
            task_description: Human natural language task description.

        Returns:
            ScenePromptGeneratorOutput with analysis and N scene prompts.
        """
        console_logger.info(f"Generating scene prompts for task: {task_description}")

        try:
            system_prompt = prompt_registry.get_prompt(
                prompt_enum=RobotEvalPrompts.SCENE_PROMPT_GENERATOR,
                num_prompts=self.num_prompts,
            )

            output = await structured_llm_call(
                model=self.cfg.openai.model,
                system_prompt=system_prompt,
                user_input=f"TASK: {task_description}",
                output_type=ScenePromptGeneratorOutput,
                api_base=self.cfg.openai.api_base,
            )

            console_logger.info(
                f"Generated {len(output.scene_prompts)} scene prompts "
                f"(room={output.analysis.room_requirement}, "
                f"objects={output.analysis.required_objects})"
            )

            return output
        finally:
            await close_cached_clients()

    def write_prompts_csv(
        self, output: ScenePromptGeneratorOutput, output_dir: Path
    ) -> Path:
        """Write prompts.csv for scenecode consumption.

        Creates:
        - prompts.csv: Scene prompts in scenecode CSV format
        - task_metadata.yaml: Task info for validation

        Args:
            output: Generator output with scene prompts.
            output_dir: Directory to write files to.

        Returns:
            Path to the created prompts.csv file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write prompts.csv in scenecode format.
        csv_path = output_dir / "prompts.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scene_index", "prompt"])
            for i, scene_prompt in enumerate(output.scene_prompts):
                writer.writerow([i, scene_prompt.prompt])

        console_logger.info(f"Wrote {len(output.scene_prompts)} prompts to {csv_path}")

        # Write task metadata for validation.
        task_metadata = {
            "task_description": output.task_description,
            "analysis": {
                "room_requirement": output.analysis.room_requirement,
                "required_objects": output.analysis.required_objects,
                "flexible_dimensions": output.analysis.flexible_dimensions,
                "initial_state_constraint": output.analysis.initial_state_constraint,
            },
            "num_scenes": len(output.scene_prompts),
            "style_variants": [sp.style_variant for sp in output.scene_prompts],
        }
        metadata_path = output_dir / "task_metadata.yaml"
        with open(metadata_path, "w") as f:
            yaml.dump(task_metadata, f, default_flow_style=False)

        console_logger.info(f"Wrote task metadata to {metadata_path}")

        return csv_path


async def generate_scene_prompts(
    task_description: str,
    cfg: DictConfig,
    output_dir: Path | None = None,
    num_prompts: int = 5,
) -> ScenePromptGeneratorOutput:
    """Convenience function to generate scene prompts from a task.

    Args:
        task_description: Human natural language task.
        cfg: Configuration with model settings.
        output_dir: Optional directory to write CSV output.
        num_prompts: Number of diverse prompts to generate.

    Returns:
        ScenePromptGeneratorOutput with analysis and prompts.
    """
    generator = ScenePromptGenerator(cfg=cfg, num_prompts=num_prompts)
    output = await generator.generate(task_description)

    if output_dir is not None:
        generator.write_prompts_csv(output=output, output_dir=output_dir)

    return output
