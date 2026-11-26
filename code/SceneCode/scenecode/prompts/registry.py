from enum import Enum, EnumMeta, nonmember
from typing import Any, Dict, Type

from .manager import PromptManager


class PromptEnumMeta(EnumMeta):
    """Metaclass for PromptEnum that handles _BASE_PATH validation."""

    def __new__(
        metacls: Type["PromptEnumMeta"],
        cls: str,
        bases: tuple[Type, ...],
        classdict: Dict[str, Any],
        **kwds: Any,
    ) -> Type["PromptEnum"]:
        """Create enum class with automatic _BASE_PATH validation."""
        # Store _BASE_PATH value before enum processes classdict.
        base_path_value = classdict.get("_BASE_PATH")

        # Create the enum class normally.
        enum_class = super().__new__(metacls, cls, bases, classdict, **kwds)

        # Validate _BASE_PATH exists for non-base classes.
        if cls != "PromptEnum" and base_path_value is None:
            raise AttributeError(
                f"{cls} must define a _BASE_PATH class attribute. "
                f"Example: _BASE_PATH = nonmember('agent_folder')"
            )

        return enum_class


class PromptEnum(str, Enum, metaclass=PromptEnumMeta):
    """Base class for all prompt enums.

    All prompt enums should inherit from this class and define a _BASE_PATH.
    The enum values are automatically combined with _BASE_PATH to create full paths.

    Example:
        class ManipulatorAgentPrompts(PromptEnum):
            _BASE_PATH = nonmember("agent")
            GRASP_PLANNING = "grasp_planning"  # Full path: "agent/grasp_planning"
            MOTION_PLANNING = "motion_planning"  # Full path: "agent/motion_planning"

    Note:
        _BASE_PATH must be wrapped with nonmember() to prevent it from becoming
        an enum member in Python 3.11+.
    """

    def __new__(cls, value):
        """Create enum member with _BASE_PATH prefix."""
        # Get base path, defaulting to empty string if not found.
        base_path = getattr(cls, "_BASE_PATH", "")

        if base_path:
            full_value = f"{base_path}/{value}"
        else:
            # Allow empty base path for special cases (like tests).
            full_value = value

        obj = str.__new__(cls, full_value)
        obj._value_ = full_value
        return obj


class FloorPlanAgentPrompts(PromptEnum):
    """Registry of floor plan agent prompts."""

    _BASE_PATH = nonmember("floor_plan_agent")

    # Planner prompts.
    PLANNER_AGENT = "planner_agent"
    PLANNER_RUNNER_INSTRUCTION = "planner_runner_instruction"

    # Designer prompts.
    DESIGNER_AGENT = "designer_agent"
    DESIGNER_INITIAL_INSTRUCTION = "designer_initial_instruction"
    DESIGNER_CRITIQUE_INSTRUCTION = "designer_critique_instruction"

    # Critic prompts.
    CRITIC_AGENT = "critic_agent"
    CRITIC_RUNNER_INSTRUCTION = "critic_runner_instruction"


class FurnitureAgentPrompts(PromptEnum):
    """Registry of furniture agent prompts."""

    _BASE_PATH = nonmember("furniture_agent")

    # Planner prompts.
    STATEFUL_PLANNER_AGENT = "stateful_planner_agent"
    STATEFUL_PLANNER_RUNNER_INSTRUCTION = "stateful_planner_runner_instruction"

    # Designer prompts.
    DESIGNER_AGENT = "designer_agent"
    DESIGNER_INITIAL_INSTRUCTION = "designer_initial_instruction"
    DESIGNER_CRITIQUE_INSTRUCTION_STATEFUL = "designer_critique_instruction_stateful"

    # Critic prompts.
    STATEFUL_CRITIC_AGENT = "stateful_critic_agent"
    STATEFUL_CRITIC_RUNNER_INSTRUCTION = "stateful_critic_runner_instruction"


class ManipulandAgentPrompts(PromptEnum):
    """Registry of manipuland agent prompts."""

    _BASE_PATH = nonmember("manipuland_agent")

    # Planner prompts.
    MANIPULAND_PLANNER_AGENT = "planner_agent"
    MANIPULAND_PLANNER_RUNNER_INSTRUCTION = "planner_runner_instruction"

    # Designer prompts.
    MANIPULAND_DESIGNER_AGENT = "designer_agent"
    DESIGNER_INITIAL_INSTRUCTION = "designer_initial_instruction"
    DESIGNER_CRITIQUE_INSTRUCTION = "designer_critique_instruction"

    # Critic prompts.
    MANIPULAND_CRITIC_AGENT = "critic_agent"
    MANIPULAND_CRITIC_RUNNER_INSTRUCTION = "critic_runner_instruction"

    # Analysis prompts.
    ANALYZE_FURNITURE_FOR_PLACEMENT = "analyze_furniture_for_placement"
    SELECT_CONTEXT_FURNITURE = "select_context_furniture"


class WallAgentPrompts(PromptEnum):
    """Registry of wall agent prompts."""

    _BASE_PATH = nonmember("wall_agent")

    # Planner prompts.
    STATEFUL_PLANNER_AGENT = "stateful_planner_agent"
    STATEFUL_PLANNER_RUNNER_INSTRUCTION = "stateful_planner_runner_instruction"

    # Designer prompts.
    DESIGNER_AGENT = "designer_agent"
    DESIGNER_INITIAL_INSTRUCTION = "designer_initial_instruction"
    DESIGNER_CRITIQUE_INSTRUCTION = "designer_critique_instruction"

    # Critic prompts.
    STATEFUL_CRITIC_AGENT = "stateful_critic_agent"
    STATEFUL_CRITIC_RUNNER_INSTRUCTION = "stateful_critic_runner_instruction"


class CeilingAgentPrompts(PromptEnum):
    """Registry of ceiling agent prompts."""

    _BASE_PATH = nonmember("ceiling_agent")

    # Planner prompts.
    STATEFUL_PLANNER_AGENT = "stateful_planner_agent"
    STATEFUL_PLANNER_RUNNER_INSTRUCTION = "stateful_planner_runner_instruction"

    # Designer prompts.
    DESIGNER_AGENT = "designer_agent"
    DESIGNER_INITIAL_INSTRUCTION = "designer_initial_instruction"
    DESIGNER_CRITIQUE_INSTRUCTION = "designer_critique_instruction"

    # Critic prompts.
    STATEFUL_CRITIC_AGENT = "stateful_critic_agent"
    STATEFUL_CRITIC_RUNNER_INSTRUCTION = "stateful_critic_runner_instruction"


class MeshPhysicsPrompts(PromptEnum):
    """Registry of mesh physics analysis prompts."""

    _BASE_PATH = nonmember("mesh_physics")

    GENERATED = "generated"
    HSSD = "hssd"
    ARTICULATED = "articulated"


class ImageGenerationPrompts(PromptEnum):
    """Registry of image generation prompts."""

    _BASE_PATH = nonmember("image_generation")

    ASSET_IMAGE_INITIAL = "asset_image_initial"
    ASSET_IMAGE_CONTINUATION = "asset_image_continuation"
    FURNITURE_CONTEXT_IMAGE = "furniture_context_image"
    MANIPULAND_CONTEXT_IMAGE = "manipuland_context_image"


class MaterialGenerationPrompts(PromptEnum):
    """Registry of AI-based material generation prompts."""

    _BASE_PATH = nonmember("material_generation")

    SEAMLESS_TEXTURE = "seamless_texture"
    ARTWORK_IMAGE = "artwork_image"


class AssetRouterPrompts(PromptEnum):
    """Registry of asset router prompts."""

    _BASE_PATH = nonmember("asset_router")

    REQUEST_ANALYSIS_FURNITURE = "request_analysis_furniture"
    REQUEST_ANALYSIS_MANIPULAND = "request_analysis_manipuland"
    REQUEST_ANALYSIS_WALL = "request_analysis_wall"
    REQUEST_ANALYSIS_CEILING = "request_analysis_ceiling"
    ASSET_VALIDATION = "asset_validation"
    ASSET_VALIDATION_LENIENT = "asset_validation_lenient"
    THIN_COVERING_VALIDATION_PROMPT = "thin_covering_validation"


class SessionMemoryPrompts(PromptEnum):
    """Registry of session memory management prompts."""

    _BASE_PATH = nonmember("session_memory")

    TURN_SUMMARIZATION = "turn_summarization"


class RobotEvalPrompts(PromptEnum):
    """Registry of robot evaluation prompts."""

    _BASE_PATH = nonmember("robot_eval")

    SUCCESS_VALIDATOR = "success_validator"
    VALIDATION_INSTRUCTION = "validation_instruction"
    SCENE_PROMPT_GENERATOR = "scene_prompt_generator"
    POLICY_AGENT = "policy_agent"


class PromptRegistry:
    """Central registry for all available prompts."""

    def __init__(self, prompt_manager: PromptManager):
        """Initialize registry with a prompt manager."""
        self.prompt_manager = prompt_manager

    def get_prompt(self, prompt_enum: str, **kwargs) -> str:
        """
        Get a prompt by its enum value with type safety.

        Args:
            prompt_enum: Prompt enum value
            **kwargs: Template variables for rendering

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If template variables don't match YAML declaration exactly
        """
        # Get template variables from YAML file.
        metadata = self.prompt_manager.get_prompt_metadata(prompt_enum)
        required_vars = set(metadata.get("template_variables", []))
        provided_vars = set(kwargs.keys())

        # Check for missing variables.
        missing_vars = required_vars - provided_vars
        if missing_vars:
            raise ValueError(
                f"Missing required template variables for {prompt_enum}: "
                f"{', '.join(sorted(missing_vars))}. "
                f"Required: {sorted(required_vars)}, Provided: {sorted(provided_vars)}"
            )

        # Check for extra variables.
        extra_vars = provided_vars - required_vars
        if extra_vars:
            raise ValueError(
                f"Unexpected template variables for {prompt_enum}: "
                f"{', '.join(sorted(extra_vars))}. "
                f"Required: {sorted(required_vars)}, Provided: {sorted(provided_vars)}"
            )

        return self.prompt_manager.get_prompt(prompt_enum, **kwargs)

    def validate_prompt_args(self, prompt_enum: str, **kwargs) -> bool:
        """
        Validate that all required variables are provided for a prompt.

        Args:
            prompt_enum: Prompt enum value
            **kwargs: Provided template variables

        Returns:
            True if all required variables match exactly
        """
        try:
            # Use the same validation logic as get_prompt but return bool.
            metadata = self.prompt_manager.get_prompt_metadata(prompt_enum)
            required_vars = set(metadata.get("template_variables", []))
            provided_vars = set(kwargs.keys())

            # Must match exactly (no missing, no extra).
            return required_vars == provided_vars
        except Exception:
            # If any error occurs (e.g., prompt not found), validation fails.
            return False
