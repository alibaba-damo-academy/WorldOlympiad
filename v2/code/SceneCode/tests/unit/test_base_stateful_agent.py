"""Unit tests for BaseStatefulAgent request settings."""

import unittest

import pytest

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from omegaconf import OmegaConf

from scenecode.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenecode.agent_utils.placement_noise import PlacementNoiseMode
from scenecode.agent_utils.room import AgentType


class _TestableBaseStatefulAgent(BaseStatefulAgent):
    """Minimal concrete subclass for exercising shared helpers."""

    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def agent_type(self) -> AgentType:
        return AgentType.FURNITURE

    def _get_final_scores_directory(self) -> Path:
        return Path(".")

    def _get_critique_prompt_enum(self) -> Any:
        return None

    def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
        return None

    def _get_design_change_prompt_enum(self) -> Any:
        return None

    def _get_initial_design_prompt_enum(self) -> Any:
        return None

    def _get_initial_design_prompt_kwargs(self) -> dict:
        return {}


class TestBaseStatefulAgentModelSettings(unittest.TestCase):
    """Tests for shared ModelSettings construction."""

    def test_get_model_settings_omits_reasoning_and_verbosity(self):
        """Compatibility mode should not forward reasoning or verbosity."""
        cfg = OmegaConf.create(
            {
                "openai": {
                    "service_tier": "priority",
                    "reasoning_effort": {"designer": "high"},
                    "verbosity": {"designer": "low"},
                },
                "api_timeout": {
                    "connect": 1.0,
                    "read": 2.0,
                    "write": 3.0,
                    "pool": 4.0,
                },
            }
        )
        agent = _TestableBaseStatefulAgent(cfg)

        settings = agent._get_model_settings(
            settings_key="designer",
            tool_choice="observe_scene",
            parallel_tool_calls=False,
        )

        self.assertIsNotNone(settings)
        self.assertIsNone(settings.reasoning)
        self.assertIsNone(settings.verbosity)
        self.assertEqual(settings.tool_choice, "observe_scene")
        self.assertFalse(settings.parallel_tool_calls)

        serialized = settings.to_json_dict()
        self.assertIsNone(serialized["reasoning"])
        self.assertIsNone(serialized["verbosity"])
        self.assertEqual(serialized["tool_choice"], "observe_scene")
        self.assertFalse(serialized["parallel_tool_calls"])
        self.assertEqual(serialized["extra_args"]["service_tier"], "priority")
        self.assertIn("timeout", serialized["extra_args"])


if __name__ == "__main__":
    unittest.main()


@pytest.mark.asyncio
async def test_cleanup_async_resources_closes_sessions_and_cached_clients():
    """Shared async cleanup should close managed sessions and shared clients."""
    cfg = OmegaConf.create({"session_memory": {"intra_turn_observation_stripping": {"enabled": False}}, "openai": {}})
    agent = _TestableBaseStatefulAgent(cfg)
    designer_session = MagicMock()
    designer_session.close = AsyncMock()
    critic_session = MagicMock()
    critic_session.close = AsyncMock()
    agent.designer_session = designer_session
    agent.critic_session = critic_session

    with patch(
        "scenecode.agent_utils.base_stateful_agent.close_cached_async_openai_clients",
        new=AsyncMock(),
    ) as mock_close_cached_clients:
        await agent._cleanup_async_resources()

    designer_session.close.assert_awaited_once()
    critic_session.close.assert_awaited_once()
    mock_close_cached_clients.assert_awaited_once()
    assert agent.designer_session is None
    assert agent.critic_session is None
