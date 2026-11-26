"""Unit tests for the prompt management system."""

import inspect
import tempfile
import unittest

from enum import Enum
from pathlib import Path
from unittest.mock import Mock

import yaml

from scenecode import prompts
from scenecode.prompts import FurnitureAgentPrompts, prompt_manager, prompt_registry
from scenecode.prompts.manager import PromptManager, PromptNotFoundError
from scenecode.prompts.registry import PromptEnum, PromptRegistry


class MockPromptEnum(PromptEnum):
    """Mock enum for testing PromptManager with enum-only API."""

    _BASE_PATH = ""  # Empty base path for test files at root level

    TEST_PROMPT = "test_prompt"
    SIMPLE_PROMPT = "simple_prompt"
    NONEXISTENT_PROMPT = "nonexistent_prompt"


class TestPromptManager(unittest.TestCase):
    """Test cases for PromptManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.prompts_dir = Path(self.temp_dir.name) / "prompts"
        self.prompts_dir.mkdir()

        # Create test prompt files.
        test_prompt_data = {
            "name": "test_prompt",
            "version": "1.0",
            "description": "A test prompt",
            "template_variables": ["var1", "var2"],
            "prompt": "This is a test prompt with {{ var1 }} and {{ var2 }}.",
        }

        simple_prompt_data = {
            "name": "simple_prompt",
            "version": "1.0",
            "prompt": "This is a simple prompt without variables.",
        }

        # Write test prompt files.
        with open(self.prompts_dir / "test_prompt.yaml", "w") as f:
            yaml.dump(test_prompt_data, f)

        with open(self.prompts_dir / "simple_prompt.yaml", "w") as f:
            yaml.dump(simple_prompt_data, f)

        self.manager = PromptManager(prompts_dir=self.prompts_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_get_prompt_simple(self):
        """Test getting a simple prompt without template variables."""
        prompt = self.manager.get_prompt(MockPromptEnum.SIMPLE_PROMPT)
        self.assertEqual(prompt, "This is a simple prompt without variables.")

    def test_get_prompt_with_template_variables(self):
        """Test getting a prompt with template variables."""
        prompt = self.manager.get_prompt(
            MockPromptEnum.TEST_PROMPT, var1="value1", var2="value2"
        )
        expected = "This is a test prompt with value1 and value2."
        self.assertEqual(prompt, expected)

    def test_get_prompt_missing_template_variables(self):
        """Test that missing template variables raise ValueError."""
        with self.assertRaises(ValueError) as context:
            self.manager.get_prompt(
                MockPromptEnum.TEST_PROMPT, var1="value1"
            )  # Missing var2

        error_msg = str(context.exception)
        self.assertIn("Missing template variables", error_msg)
        self.assertIn("var2", error_msg)

    def test_get_prompt_extra_template_variables(self):
        """Test that extra template variables raise ValueError."""
        with self.assertRaises(ValueError) as context:
            self.manager.get_prompt(
                MockPromptEnum.TEST_PROMPT,
                var1="value1",
                var2="value2",
                var3="extra",  # Extra variable
            )

        error_msg = str(context.exception)
        self.assertIn("Unexpected template variables", error_msg)
        self.assertIn("var3", error_msg)

    def test_get_prompt_empty_template_variables_with_args(self):
        """Test that providing variables to prompt with no template_variables raises
        ValueError."""
        with self.assertRaises(ValueError) as context:
            self.manager.get_prompt(MockPromptEnum.SIMPLE_PROMPT, unexpected="value")

        error_msg = str(context.exception)
        self.assertIn("Unexpected template variables", error_msg)
        self.assertIn("unexpected", error_msg)

    def test_get_prompt_not_found(self):
        """Test getting a non-existent prompt raises exception."""
        with self.assertRaises(PromptNotFoundError):
            self.manager.get_prompt(MockPromptEnum.NONEXISTENT_PROMPT)

    def test_get_prompt_metadata(self):
        """Test getting prompt metadata."""
        metadata = self.manager.get_prompt_metadata(MockPromptEnum.TEST_PROMPT)
        expected_metadata = {
            "name": "test_prompt",
            "version": "1.0",
            "description": "A test prompt",
            "template_variables": ["var1", "var2"],
        }
        self.assertEqual(metadata, expected_metadata)

    def test_list_prompts(self):
        """Test listing all available prompts."""
        prompts = self.manager.list_prompts()
        expected = ["simple_prompt", "test_prompt"]
        self.assertEqual(sorted(prompts), sorted(expected))

    def test_validate_prompt(self):
        """Test prompt validation."""
        self.assertTrue(self.manager.validate_prompt("test_prompt"))
        self.assertFalse(self.manager.validate_prompt("nonexistent"))


class TestPromptRegistry(unittest.TestCase):
    """Test cases for PromptRegistry."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_manager = Mock(spec=PromptManager)
        self.registry = PromptRegistry(self.mock_manager)

    def test_get_prompt(self):
        """Test getting a prompt through the registry."""
        expected_prompt = "Test prompt content"
        self.mock_manager.get_prompt.return_value = expected_prompt
        self.mock_manager.get_prompt_metadata.return_value = {
            "template_variables": ["test_var"]
        }

        result = self.registry.get_prompt(
            prompt_enum=FurnitureAgentPrompts.STATEFUL_CRITIC_RUNNER_INSTRUCTION,
            test_var="test value",
        )

        self.assertEqual(result, expected_prompt)
        self.mock_manager.get_prompt.assert_called_once_with(
            FurnitureAgentPrompts.STATEFUL_CRITIC_RUNNER_INSTRUCTION,
            test_var="test value",
        )

    def test_validate_prompt_args(self):
        """Test validating prompt arguments."""
        # Mock metadata for STATEFUL_CRITIC_RUNNER_INSTRUCTION.
        self.mock_manager.get_prompt_metadata.return_value = {
            "template_variables": ["test_var"]
        }

        # Valid arguments.
        is_valid = self.registry.validate_prompt_args(
            prompt_enum=FurnitureAgentPrompts.STATEFUL_CRITIC_RUNNER_INSTRUCTION,
            test_var="test value",
        )
        self.assertTrue(is_valid)

        # Missing required argument.
        is_valid = self.registry.validate_prompt_args(
            prompt_enum=FurnitureAgentPrompts.STATEFUL_CRITIC_RUNNER_INSTRUCTION,
        )
        self.assertFalse(is_valid)

        # Extra argument.
        is_valid = self.registry.validate_prompt_args(
            prompt_enum=FurnitureAgentPrompts.STATEFUL_CRITIC_RUNNER_INSTRUCTION,
            test_var="test value",
            extra="unexpected",
        )
        self.assertFalse(is_valid)

    def test_registry_strict_validation_missing_vars(self):
        """Test that registry raises ValueError for missing template variables."""
        self.mock_manager.get_prompt_metadata.return_value = {
            "template_variables": ["test_var"]
        }

        with self.assertRaises(ValueError) as context:
            self.registry.get_prompt(
                FurnitureAgentPrompts.STATEFUL_CRITIC_RUNNER_INSTRUCTION,
                # Missing test_var
            )

        error_msg = str(context.exception)
        self.assertIn("Missing required template variables", error_msg)
        self.assertIn("test_var", error_msg)

    def test_registry_strict_validation_extra_vars(self):
        """Test that registry raises ValueError for extra template variables."""
        self.mock_manager.get_prompt_metadata.return_value = {
            "template_variables": ["test_var"]
        }

        with self.assertRaises(ValueError) as context:
            self.registry.get_prompt(
                FurnitureAgentPrompts.STATEFUL_CRITIC_RUNNER_INSTRUCTION,
                test_var="test value",
                extra="unexpected",  # Extra variable
            )

        error_msg = str(context.exception)
        self.assertIn("Unexpected template variables", error_msg)
        self.assertIn("extra", error_msg)

    def test_prompt_enums(self):
        """Test that prompt enums work correctly."""
        self.assertEqual(
            FurnitureAgentPrompts.STATEFUL_PLANNER_AGENT,
            "furniture_agent/stateful_planner_agent",
        )
        self.assertEqual(
            FurnitureAgentPrompts.STATEFUL_CRITIC_AGENT,
            "furniture_agent/stateful_critic_agent",
        )


class TestPromptSystem(unittest.TestCase):
    """Test cases for the complete prompt system."""

    def test_system_initialization(self):
        """Test that the prompt system initializes correctly."""
        self.assertIsNotNone(prompt_manager)
        self.assertIsNotNone(prompt_registry)
        self.assertTrue(prompt_manager.prompts_dir.exists())

    def test_load_actual_prompts(self):
        """Test loading the actual prompt files in the system."""
        # Test stateful scene critic prompt (requires scene_description).
        stateful_prompt = prompt_manager.get_prompt(
            prompt_name=FurnitureAgentPrompts.STATEFUL_CRITIC_AGENT,
            scene_description="A modern living room",
        )
        self.assertIsInstance(stateful_prompt, str)
        self.assertGreater(len(stateful_prompt), 100)

    def test_template_rendering(self):
        """Test rendering prompts with template variables."""
        rendered_prompt = prompt_manager.get_prompt(
            prompt_name=FurnitureAgentPrompts.STATEFUL_CRITIC_AGENT,
            scene_description="A modern living room with a sofa",
        )

        self.assertIn("modern living room", rendered_prompt)

    def test_registry_functionality(self):
        """Test registry functionality with actual prompts."""
        # Test getting prompt through registry.
        prompt = prompt_registry.get_prompt(
            prompt_enum=FurnitureAgentPrompts.STATEFUL_CRITIC_AGENT,
            scene_description="Test living room",
        )

        self.assertIsInstance(prompt, str)
        self.assertIn("Test living room", prompt)

        # Test argument validation.
        self.assertTrue(
            prompt_registry.validate_prompt_args(
                prompt_enum=FurnitureAgentPrompts.STATEFUL_CRITIC_AGENT,
                scene_description="test",
            )
        )

        # Test validation with missing arguments.
        self.assertFalse(
            prompt_registry.validate_prompt_args(
                prompt_enum=FurnitureAgentPrompts.STATEFUL_CRITIC_AGENT,
                # Missing scene_description
            )
        )

        # Test validation with extra arguments.
        self.assertFalse(
            prompt_registry.validate_prompt_args(
                prompt_enum=FurnitureAgentPrompts.STATEFUL_CRITIC_AGENT,
                scene_description="test",
                extra="unexpected",  # Extra argument
            )
        )

    def test_prompt_listing(self):
        """Test listing available prompts."""
        prompts = prompt_manager.list_prompts()
        expected_prompts = [
            "furniture_agent/stateful_planner_agent",
            "furniture_agent/designer_agent",
            "furniture_agent/stateful_critic_agent",
        ]

        for expected in expected_prompts:
            self.assertIn(expected, prompts)

    def test_all_enum_prompts_have_files(self):
        """Test that all prompt enum values have corresponding YAML files."""
        # Find all enum classes in the prompts module.
        missing_files = []
        load_errors = []

        for _, obj in inspect.getmembers(prompts, inspect.isclass):
            # Check if it's an enum that inherits from str, Enum.
            if (
                issubclass(obj, Enum)
                and issubclass(obj, str)
                and obj is not Enum
                and obj is not str
            ):

                # Check each enum value.
                for enum_value in obj:
                    # Skip private enum members like _BASE_PATH.
                    if enum_value.name.startswith("_"):
                        continue

                    # Construct expected file path.
                    expected_file = (
                        prompt_manager.prompts_dir / f"{enum_value.value}.yaml"
                    )

                    if not expected_file.exists():
                        missing_files.append(
                            f"{obj.__name__}.{enum_value.name} -> {expected_file}"
                        )
                    else:
                        # Also test that the file can be loaded successfully.
                        try:
                            prompt_manager.get_prompt_metadata(enum_value)
                        except Exception as e:
                            load_errors.append(
                                f"{obj.__name__}.{enum_value.name}: {str(e)}"
                            )

        # Report all issues.
        error_messages = []
        if missing_files:
            error_messages.append(
                f"Missing prompt files:\n"
                + "\n".join(f"  - {f}" for f in missing_files)
            )

        if load_errors:
            error_messages.append(
                f"Failed to load prompt files:\n"
                + "\n".join(f"  - {e}" for e in load_errors)
            )

        if error_messages:
            self.fail("\n\n".join(error_messages))


if __name__ == "__main__":
    unittest.main()
