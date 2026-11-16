"""Unit tests for IntraTurnImageFilter.

Tests the main contracts: keeps last N observations, strips older ones,
preserves context images in user messages.
"""

import unittest

from unittest.mock import MagicMock

from omegaconf import OmegaConf

from scenecode.agent_utils.intra_turn_image_filter import IntraTurnImageFilter


def make_config(enabled: bool = True, keep_last_n: int = 2) -> OmegaConf:
    """Create a mock config for testing."""
    return OmegaConf.create(
        {
            "session_memory": {
                "enabled": True,
                "intra_turn_observation_stripping": {
                    "enabled": enabled,
                    "keep_last_n_observations": keep_last_n,
                },
            }
        }
    )


def make_observation_output(n_images: int = 2) -> dict:
    """Create a function_call_output item with images (observe_scene output)."""
    output = []
    for i in range(n_images):
        output.append(
            {"type": "input_image", "image_url": f"data:image/png;base64,fake_{i}"}
        )
    output.append(
        {"type": "input_text", "text": f"Scene observed from {n_images} viewpoints."}
    )
    return {
        "type": "function_call_output",
        "call_id": f"call_{id(output)}",
        "output": output,
    }


def make_user_message_with_image() -> dict:
    """Create a user message with a context image."""
    return {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Design the room"},
            {"type": "input_image", "image_url": "data:image/png;base64,context_image"},
        ],
    }


def make_text_tool_output() -> dict:
    """Create a function_call_output item without images (non-observe_scene)."""
    return {
        "type": "function_call_output",
        "call_id": "call_text_tool",
        "output": "Tool completed successfully.",
    }


def make_mock_call_data(items: list) -> MagicMock:
    """Create mock CallModelData for testing."""
    mock_model_data = MagicMock()
    mock_model_data.input = items
    mock_model_data.instructions = None
    mock_data = MagicMock()
    mock_data.model_data = mock_model_data
    return mock_data


class TestIntraTurnImageFilter(unittest.TestCase):
    """Tests for the main filter contracts."""

    def test_keeps_last_n_observations_strips_older(self):
        """Strips older observations, keeps last N with images intact."""
        filter_ = IntraTurnImageFilter(cfg=make_config(enabled=True, keep_last_n=2))

        items = [
            make_observation_output(n_images=1),  # Obs 0 - should be stripped.
            make_observation_output(n_images=1),  # Obs 1 - should be stripped.
            make_observation_output(n_images=1),  # Obs 2 - should be kept.
            make_observation_output(n_images=1),  # Obs 3 - should be kept.
        ]

        result = filter_(make_mock_call_data(items))

        # Observations 0 and 1 should have images stripped.
        for i in [0, 1]:
            has_placeholder = any(
                p.get("text") == "[image removed]" for p in result.input[i]["output"]
            )
            self.assertTrue(has_placeholder, f"Observation {i} should be stripped")

        # Observations 2 and 3 should have images intact.
        for i in [2, 3]:
            has_image = any(
                "data:image" in p.get("image_url", "")
                for p in result.input[i]["output"]
            )
            self.assertTrue(has_image, f"Observation {i} should have images")

    def test_preserves_context_images_in_user_messages(self):
        """Never strips images from user messages (context images)."""
        filter_ = IntraTurnImageFilter(cfg=make_config(enabled=True, keep_last_n=1))

        items = [
            make_user_message_with_image(),  # Context image - must be preserved.
            make_observation_output(n_images=1),  # Obs 0 - should be stripped.
            make_observation_output(n_images=1),  # Obs 1 - should be kept.
        ]

        result = filter_(make_mock_call_data(items))

        # User message with context image should be unchanged.
        user_msg = result.input[0]
        has_context_image = any(
            "context_image" in str(p.get("image_url", ""))
            for p in user_msg["content"]
            if isinstance(p, dict)
        )
        self.assertTrue(has_context_image, "Context image must be preserved")

    def test_disabled_filter_returns_unchanged(self):
        """Filter returns unchanged data when disabled."""
        filter_ = IntraTurnImageFilter(cfg=make_config(enabled=False))
        items = [make_observation_output(), make_observation_output()]

        result = filter_(make_mock_call_data(items))

        # Should return the original model_data unchanged.
        self.assertEqual(result.input, items)

    def test_caching_skips_restripping_when_no_new_observation(self):
        """Cache reuse: skips stripping when no new observation is added."""
        filter_ = IntraTurnImageFilter(cfg=make_config(enabled=True, keep_last_n=2))

        # Initial call with 3 observations - should strip obs 0.
        obs0 = make_observation_output(n_images=1)
        obs1 = make_observation_output(n_images=1)
        obs2 = make_observation_output(n_images=1)
        initial_items = [obs0, obs1, obs2]

        result1 = filter_(make_mock_call_data(initial_items))

        # Verify obs 0 was stripped.
        self.assertTrue(
            any(p.get("text") == "[image removed]" for p in result1.input[0]["output"])
        )

        # Second call: add a text tool output (not an observation).
        text_output = make_text_tool_output()
        extended_items = initial_items + [text_output]

        result2 = filter_(make_mock_call_data(extended_items))

        # Cache should be used - obs 0 should still be stripped from cache.
        self.assertTrue(
            any(p.get("text") == "[image removed]" for p in result2.input[0]["output"])
        )
        # New item should be appended.
        self.assertEqual(result2.input[-1], text_output)

        # Third call: add a new observation - should trigger re-stripping.
        obs3 = make_observation_output(n_images=1)
        new_obs_items = extended_items + [obs3]

        result3 = filter_(make_mock_call_data(new_obs_items))

        # Now obs 0 AND obs 1 should be stripped (4 obs, keep last 2).
        self.assertTrue(
            any(p.get("text") == "[image removed]" for p in result3.input[0]["output"])
        )
        self.assertTrue(
            any(p.get("text") == "[image removed]" for p in result3.input[1]["output"])
        )
        # obs 2 and obs 3 should have images intact.
        self.assertTrue(
            any(
                "data:image" in p.get("image_url", "")
                for p in result3.input[2]["output"]
            )
        )


if __name__ == "__main__":
    unittest.main()
