import logging

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from jinja2 import Template

if TYPE_CHECKING:
    from scenecode.prompts.registry import PromptEnum

console_logger = logging.getLogger(__name__)


class PromptNotFoundError(Exception):
    """Raised when a prompt is not found."""


class PromptManager:
    """Manages loading and rendering of YAML prompts with Jinja2 template support."""

    def __init__(self, prompts_dir: Path):
        """
        Initialize the prompt manager.

        Args:
            prompts_dir: Directory containing YAML prompt files
        """
        self.prompts_dir = prompts_dir
        self._cache: dict[str, dict[str, Any]] = {}

        console_logger.debug(
            f"Initialized PromptManager with prompts_dir: {prompts_dir}"
        )

    @lru_cache(maxsize=128)
    def _load_prompt_yaml(self, prompt_path: Path) -> dict[str, Any]:
        """Load and cache a YAML prompt file."""
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise PromptNotFoundError(f"Prompt file not found: {prompt_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in prompt file {prompt_path}: {e}")

    def get_prompt_metadata(self, prompt_name: "PromptEnum") -> dict[str, Any]:
        """
        Get metadata for a prompt without loading the full prompt content.

        Args:
            prompt_name: PromptEnum value (any enum inheriting from PromptEnum)

        Returns:
            Dictionary containing prompt metadata
        """
        prompt_path = self.prompts_dir / f"{prompt_name.value}.yaml"
        prompt_data = self._load_prompt_yaml(prompt_path)

        # Return metadata without the prompt content.
        metadata = prompt_data.copy()
        metadata.pop("prompt", None)
        return metadata

    def get_prompt(self, prompt_name: "PromptEnum", **kwargs) -> str:
        """
        Load and render a prompt with optional template variables.

        Args:
            prompt_name: PromptEnum value (any enum inheriting from PromptEnum)
            **kwargs: Template variables to render in the prompt

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If template variables don't match YAML declaration exactly
        """
        prompt_path = self.prompts_dir / f"{prompt_name.value}.yaml"
        prompt_data = self._load_prompt_yaml(prompt_path)

        if "prompt" not in prompt_data:
            raise ValueError(f"No 'prompt' field found in {prompt_path}")

        # Strict validation of template variables.
        template_vars = set(prompt_data.get("template_variables", []))
        provided_vars = set(kwargs.keys())

        # Check for missing variables.
        missing_vars = template_vars - provided_vars
        if missing_vars:
            raise ValueError(
                f"Missing template variables for {prompt_name}: "
                f"{', '.join(sorted(missing_vars))}. "
                f"Required: {sorted(template_vars)}, Provided: {sorted(provided_vars)}"
            )

        # Check for extra variables.
        extra_vars = provided_vars - template_vars
        if extra_vars:
            raise ValueError(
                f"Unexpected template variables for {prompt_name}: "
                f"{', '.join(sorted(extra_vars))}. "
                f"Required: {sorted(template_vars)}, Provided: {sorted(provided_vars)}"
            )

        prompt_content = prompt_data["prompt"]

        # If no template variables required, return as-is.
        if not template_vars:
            return prompt_content

        # Render with Jinja2.
        template = Template(prompt_content)
        try:
            return template.render(**kwargs)
        except Exception as e:
            raise ValueError(f"Error rendering prompt {prompt_name}: {e}")

    def list_prompts(self) -> list[str]:
        """List all available prompts."""
        prompts = []
        for yaml_file in self.prompts_dir.rglob("*.yaml"):
            # Convert path to prompt name.
            relative_path = yaml_file.relative_to(self.prompts_dir)
            prompt_name = str(relative_path.with_suffix(""))
            prompts.append(prompt_name)
        return sorted(prompts)

    def validate_prompt(self, prompt_name: str) -> bool:
        """
        Validate that a prompt exists and has required fields.

        Args:
            prompt_name: Name of the prompt to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            prompt_data = self._load_prompt_yaml(
                self.prompts_dir / f"{prompt_name}.yaml"
            )
            return "prompt" in prompt_data
        except (PromptNotFoundError, ValueError):
            return False
