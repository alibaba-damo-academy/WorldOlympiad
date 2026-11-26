"""Prompt management system for scene agent."""

from pathlib import Path

from .manager import PromptManager
from .registry import (
    AssetRouterPrompts,
    FloorPlanAgentPrompts,
    FurnitureAgentPrompts,
    ImageGenerationPrompts,
    ManipulandAgentPrompts,
    MeshPhysicsPrompts,
    PromptRegistry,
    RobotEvalPrompts,
)

# Initialize the prompt manager with the data directory.
PROMPTS_DATA_DIR = Path(__file__).parent / "data"

prompt_manager = PromptManager(prompts_dir=PROMPTS_DATA_DIR)

# Initialize the prompt registry.
prompt_registry = PromptRegistry(prompt_manager)

__all__ = [
    "prompt_manager",
    "prompt_registry",
    "AssetRouterPrompts",
    "FloorPlanAgentPrompts",
    "FurnitureAgentPrompts",
    "ImageGenerationPrompts",
    "ManipulandAgentPrompts",
    "MeshPhysicsPrompts",
    "RobotEvalPrompts",
]
