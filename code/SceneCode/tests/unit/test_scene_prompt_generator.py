"""Unit tests for robot_eval scene prompt generation."""

from unittest.mock import AsyncMock, patch

import pytest

from scenecode.robot_eval import create_robot_eval_config
from scenecode.robot_eval.task_generation.scene_prompt_generator import (
    ScenePrompt,
    ScenePromptGenerator,
    ScenePromptGeneratorOutput,
    TaskAnalysis,
)


@pytest.mark.asyncio
async def test_generate_closes_cached_clients_in_finally():
    """Scene prompt generation should close cached LLM clients after each run."""
    cfg = create_robot_eval_config(model="gpt-5.2")
    generator = ScenePromptGenerator(cfg=cfg, num_prompts=2)
    output = ScenePromptGeneratorOutput(
        task_description="Place fruit on a table",
        analysis=TaskAnalysis(
            room_requirement="kitchen",
            required_objects=["fruit", "table"],
            flexible_dimensions=["style"],
            initial_state_constraint="fruit must not already be on the table",
        ),
        scene_prompts=[
            ScenePrompt(prompt="A compact kitchen with fruit on a counter.", style_variant="compact"),
            ScenePrompt(prompt="A bright kitchen with fruit near the sink.", style_variant="bright"),
        ],
    )

    with patch(
        "scenecode.robot_eval.task_generation.scene_prompt_generator.structured_llm_call",
        new=AsyncMock(return_value=output),
    ), patch(
        "scenecode.robot_eval.task_generation.scene_prompt_generator.close_cached_clients",
        new=AsyncMock(),
    ) as mock_close_cached_clients:
        result = await generator.generate("Place fruit on a table")

    assert result == output
    mock_close_cached_clients.assert_awaited_once()
