"""Unit tests for finalization checkpoint reset logic."""

import asyncio
import shutil
import tempfile
import unittest

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from scenecode.agent_utils.base_stateful_agent import BaseStatefulAgent
from scenecode.agent_utils.placement_noise import PlacementNoiseMode
from scenecode.agent_utils.room import AgentType
from scenecode.agent_utils.scoring import CategoryScore, FurnitureCritiqueWithScores


def _make_scores(
    realism: int = 7,
    functionality: int = 7,
    layout: int = 7,
    holistic: int = 7,
    prompt: int = 7,
    reachability: int = 7,
) -> FurnitureCritiqueWithScores:
    """Create FurnitureCritiqueWithScores with specified grades."""
    return FurnitureCritiqueWithScores(
        critique="Test critique",
        realism=CategoryScore(name="Realism", grade=realism, comment="test"),
        functionality=CategoryScore(
            name="Functionality", grade=functionality, comment="test"
        ),
        layout=CategoryScore(name="Layout", grade=layout, comment="test"),
        holistic_completeness=CategoryScore(
            name="Holistic Completeness", grade=holistic, comment="test"
        ),
        prompt_following=CategoryScore(
            name="Prompt Following", grade=prompt, comment="test"
        ),
        reachability=CategoryScore(
            name="Reachability", grade=reachability, comment="test"
        ),
    )


class TestFinalizeSceneReset(unittest.TestCase):
    """Test that finalization resets to N-1 checkpoint when scores degrade."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create mock render directories with scores.yaml files.
        self.n2_render_dir = self.temp_dir / "renders_003"
        self.n1_render_dir = self.temp_dir / "renders_008"
        self.final_render_dir = self.temp_dir / "renders_009"

        for render_dir in [
            self.n2_render_dir,
            self.n1_render_dir,
            self.final_render_dir,
        ]:
            render_dir.mkdir(parents=True)
            (render_dir / "scores.yaml").write_text("test: scores")
            (render_dir / "view_0.png").write_text("test image")

        # Create final scores directory.
        self.final_scores_dir = self.temp_dir / "scene_states" / "furniture"
        self.final_scores_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_testable_agent(
        self, mock_scene: MagicMock, mock_rendering_manager: MagicMock
    ):
        """Create a concrete testable subclass of BaseStatefulPlacementAgent."""
        final_scores_dir = self.final_scores_dir

        mock_cfg = MagicMock()
        mock_cfg.reset_single_category_threshold = 3  # Trigger reset on 3+ point drop.
        mock_cfg.reset_total_sum_threshold = 6

        # Set up action_log_path as a real path for the action logger decorator.
        mock_scene.action_log_path = self.temp_dir / "action_log.json"

        class TestableAgent(BaseStatefulAgent):
            def __init__(self):
                # Skip parent __init__ to avoid complex setup.
                self.scene = mock_scene
                self.rendering_manager = mock_rendering_manager
                self.cfg = mock_cfg

            @property
            def agent_type(self) -> AgentType:
                return AgentType.FURNITURE

            def _get_final_scores_directory(self) -> Path:
                return final_scores_dir

            def _get_critique_prompt_enum(self) -> Any:
                return None

            def _get_design_change_prompt_enum(self) -> Any:
                return None

            def _get_initial_design_prompt_enum(self) -> Any:
                return None

            def _get_initial_design_prompt_kwargs(self) -> dict:
                return {}

            def _set_placement_noise_profile(self, mode: PlacementNoiseMode) -> None:
                pass

        return TestableAgent()

    def test_finalize_resets_to_n1_not_n2_when_scores_degrade(self):
        """When final scores are worse than N-1, reset should use N-1 state (not N-2).

        This tests for the bug where:
        - Comparison correctly uses checkpoint_scores (N-1)
        - But reset incorrectly uses previous_scene_checkpoint (N-2)

        The fix should make reset use scene_checkpoint (N-1) to match comparison.
        """
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()

        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)
        agent.final_scores_dir = self.final_scores_dir

        # Set up checkpoint state simulating the bug scenario:
        # N-2 (previous_scene_checkpoint): Realism=3 (old bad state)
        # N-1 (scene_checkpoint): Realism=6 (good checkpoint we want)
        # N (previous_scores/final): Realism=3 (degraded final)

        # N-2 state (what the bug incorrectly resets to).
        agent.previous_scene_checkpoint = {"state": "N-2", "objects": {"old": "state"}}
        agent.previous_checkpoint_scores = _make_scores(realism=3)
        agent.previous_checkpoint_render_dir = self.n2_render_dir

        # N-1 state (what we SHOULD reset to).
        agent.scene_checkpoint = {"state": "N-1", "objects": {"good": "state"}}
        agent.checkpoint_scores = _make_scores(realism=6)  # Good scores.
        agent.checkpoint_render_dir = self.n1_render_dir

        # Final scores (N) - degraded compared to N-1.
        agent.previous_scores = _make_scores(realism=3)  # 3 point drop triggers reset.
        agent.final_render_dir = self.final_render_dir

        # Run finalization.
        asyncio.run(agent._finalize_scene_and_scores())

        # ASSERTION: The scene should be restored to N-1 state, not N-2.
        # The bug causes restore_from_state_dict to be called with N-2.
        mock_scene.restore_from_state_dict.assert_called_once()
        call_args = mock_scene.restore_from_state_dict.call_args[0][0]

        # This assertion will FAIL with the current buggy code.
        # Current code passes previous_scene_checkpoint (N-2).
        # Fixed code should pass scene_checkpoint (N-1).
        self.assertEqual(
            call_args["state"],
            "N-1",
            f"Expected reset to N-1 state but got {call_args.get('state', 'unknown')}. "
            "The finalization reset is using the wrong checkpoint (N-2 instead of N-1).",
        )

        # Also verify the render dir is set to N-1.
        # Current buggy code sets it to previous_checkpoint_render_dir (N-2).
        self.assertEqual(
            agent.final_render_dir,
            self.n1_render_dir,
            f"Expected final_render_dir to be N-1 ({self.n1_render_dir}) "
            f"but got {agent.final_render_dir}",
        )

    def test_finalize_no_reset_when_scores_improve(self):
        """When final scores are better than N-1, no reset should occur."""
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()

        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)

        # Set up state where final scores are BETTER than checkpoint.
        agent.previous_scene_checkpoint = {"state": "N-2"}
        agent.previous_checkpoint_scores = _make_scores(realism=5)
        agent.previous_checkpoint_render_dir = self.n2_render_dir

        agent.scene_checkpoint = {"state": "N-1"}
        agent.checkpoint_scores = _make_scores(realism=6)  # N-1 scores.
        agent.checkpoint_render_dir = self.n1_render_dir

        # Final scores improved from N-1.
        agent.previous_scores = _make_scores(realism=8)  # Better than N-1.
        agent.final_render_dir = self.final_render_dir

        asyncio.run(agent._finalize_scene_and_scores())

        # No reset should occur.
        mock_scene.restore_from_state_dict.assert_not_called()

        # Final render dir should remain as the final iteration's render.
        self.assertEqual(agent.final_render_dir, self.final_render_dir)

    def test_finalize_resets_on_total_sum_drop(self):
        """Reset should trigger when total sum drops by threshold.

        Tests the alternative reset path (total sum) vs single category.
        Same bug applies: should reset to N-1, not N-2.
        """
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()

        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)

        # N-2 state.
        agent.previous_scene_checkpoint = {"state": "N-2"}
        agent.previous_checkpoint_scores = _make_scores()
        agent.previous_checkpoint_render_dir = self.n2_render_dir

        # N-1 state with good scores (all 7s = 42 total for 6 categories).
        agent.scene_checkpoint = {"state": "N-1"}
        agent.checkpoint_scores = _make_scores()  # All 7s.
        agent.checkpoint_render_dir = self.n1_render_dir

        # Final scores: each category drops by 2 (total drop = 12 > threshold 6).
        # No single category drops by 3, so only total sum triggers reset.
        agent.previous_scores = _make_scores(
            realism=5, functionality=5, layout=5, holistic=5, prompt=5, reachability=5
        )
        agent.final_render_dir = self.final_render_dir

        asyncio.run(agent._finalize_scene_and_scores())

        # Reset should occur.
        mock_scene.restore_from_state_dict.assert_called_once()
        call_args = mock_scene.restore_from_state_dict.call_args[0][0]

        # Should reset to N-1 (same bug as single category test).
        self.assertEqual(
            call_args["state"],
            "N-1",
            "Total sum reset should also use N-1 checkpoint, not N-2.",
        )

    def test_finalize_no_reset_when_checkpoint_scores_none(self):
        """No reset should occur when checkpoint_scores is None.

        This is an edge case at the start of iteration (no previous checkpoint).
        """
        mock_scene = MagicMock()
        mock_rendering_manager = MagicMock()

        agent = self._create_testable_agent(mock_scene, mock_rendering_manager)

        # No checkpoint scores (first iteration scenario).
        agent.checkpoint_scores = None
        agent.scene_checkpoint = None
        agent.checkpoint_render_dir = None

        agent.previous_scores = _make_scores(realism=3)  # Low scores.
        agent.final_render_dir = self.final_render_dir

        asyncio.run(agent._finalize_scene_and_scores())

        # No reset should occur since there's nothing to compare against.
        mock_scene.restore_from_state_dict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
