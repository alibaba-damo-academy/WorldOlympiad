"""Unit tests for policy interface components."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scenecode.robot_eval import create_robot_eval_config
from scenecode.robot_eval.policy_interface import (
    ObjectBinding,
    PolicyInterfaceAgent,
    PolicyInterfaceOutput,
    PredicateResolver,
)


class TestPolicyInterfaceOutput:
    """Tests for PolicyInterfaceOutput data structure."""

    def test_create_policy_interface_output(self):
        """Test creating a PolicyInterfaceOutput."""
        output = PolicyInterfaceOutput(
            task_description="Pick a cup from the floor and put it in the sink",
            goal_predicate="inside",
            target_category="cup",
            reference_category="sink",
            target_precondition="on floor",
            reference_precondition=None,
            valid_bindings=[
                ObjectBinding(
                    target_id="cup_01",
                    reference_id="sink_01",
                    rank=1,
                    confidence=0.95,
                    reasoning="Cup is on the floor",
                )
            ],
            overall_success=True,
            notes=["Found 1 valid binding"],
        )

        assert output.goal_predicate == "inside"
        assert output.target_category == "cup"
        assert output.target_precondition == "on floor"
        assert len(output.valid_bindings) == 1
        assert output.valid_bindings[0].target_id == "cup_01"
        assert output.overall_success is True

    def test_create_failed_output(self):
        """Test creating a PolicyInterfaceOutput with no valid bindings."""
        output = PolicyInterfaceOutput(
            task_description="Pick a cup from the floor and put it in the sink",
            goal_predicate="inside",
            target_category="cup",
            reference_category="sink",
            target_precondition="on floor",
            reference_precondition=None,
            valid_bindings=[],
            overall_success=False,
            notes=["No cups found on the floor"],
        )

        assert output.overall_success is False
        assert len(output.valid_bindings) == 0


class TestPredicateResolver:
    """Tests for PredicateResolver pose computation.

    Note: The resolver now uses a unified PolicyInterfaceAgent for
    task parsing and candidate selection. These tests focus on the
    geometric pose computation which doesn't require the agent.
    """

    @pytest.fixture
    def cfg(self):
        """Create test config."""
        return create_robot_eval_config(model="gpt-5.2")

    def _create_mock_scene(self):
        """Create a mock scene with test objects."""
        mock_scene = MagicMock()
        mock_scene.scene_state = {
            "objects": {
                "apple_01": {
                    "name": "Red Apple",
                    "description": "A fresh red apple",
                    "object_type": "manipuland",
                    "transform": {"translation": [0.5, 0.0, 1.0]},
                    "bbox_min": [-0.05, -0.05, -0.05],
                    "bbox_max": [0.05, 0.05, 0.05],
                },
                "cup_01": {
                    "name": "Coffee Cup",
                    "transform": {"translation": [1.0, 0.5, 1.0]},
                    "bbox_min": [-0.04, -0.04, 0.0],
                    "bbox_max": [0.04, 0.04, 0.10],
                },
                "table_01": {
                    "name": "Dining Table",
                    "description": "A wooden dining table",
                    "object_type": "furniture",
                    "transform": {"translation": [0.0, 0.0, 0.0]},
                    "bbox_min": [-0.5, -0.3, 0.0],
                    "bbox_max": [0.5, 0.3, 0.8],
                },
                "drawer_01": {
                    "name": "Drawer",
                    "description": "A wooden drawer",
                    "transform": {"translation": [2.0, 0.0, 0.5]},
                    "bbox_min": [-0.2, -0.15, -0.1],
                    "bbox_max": [0.2, 0.15, 0.1],
                },
            }
        }
        return mock_scene

    def test_compute_on_pose(self, cfg):
        """Test computing 'on' pose geometry."""
        mock_scene = self._create_mock_scene()
        resolver = PredicateResolver(scene=mock_scene, cfg=cfg)

        pose = resolver._compute_on_pose(target_id="cup_01", ref_id="table_01")

        assert pose is not None
        assert pose.action == "pick_and_place"
        assert pose.drake_model_name == "cup_01"
        assert pose.reference_id == "table_01"
        assert pose.source_predicate == "on"
        # Z should be table surface (0.8) + cup bottom offset (0.0).
        assert pose.target_position[2] == pytest.approx(0.8, abs=0.01)

    def test_compute_on_pose_placement_bounds(self, cfg):
        """Test that 'on' pose includes correct placement bounds."""
        mock_scene = self._create_mock_scene()
        resolver = PredicateResolver(scene=mock_scene, cfg=cfg)

        pose = resolver._compute_on_pose(target_id="cup_01", ref_id="table_01")

        # Placement bounds should be set (table surface shrunk by cup half-extents).
        assert pose.placement_bounds_min is not None
        assert pose.placement_bounds_max is not None
        # Cup half-extent = 0.04, table x=[-0.5, 0.5] → contained x=[-0.46, 0.46].
        assert pose.placement_bounds_min[0] == pytest.approx(-0.46, abs=0.01)
        assert pose.placement_bounds_max[0] == pytest.approx(0.46, abs=0.01)

    def test_compute_inside_pose(self, cfg):
        """Test computing 'inside' pose geometry."""
        mock_scene = self._create_mock_scene()
        resolver = PredicateResolver(scene=mock_scene, cfg=cfg)

        pose = resolver._compute_inside_pose(target_id="cup_01", ref_id="drawer_01")

        assert pose is not None
        assert pose.drake_model_name == "cup_01"
        assert pose.reference_id == "drawer_01"
        assert pose.source_predicate == "inside"

    def test_compute_near_pose(self, cfg):
        """Test computing 'near' pose geometry."""
        mock_scene = self._create_mock_scene()
        resolver = PredicateResolver(scene=mock_scene, cfg=cfg)

        pose = resolver._compute_near_pose(target_id="apple_01", ref_id="table_01")

        assert pose is not None
        assert pose.drake_model_name == "apple_01"
        assert pose.reference_id == "table_01"
        assert pose.source_predicate == "near"

    def test_compute_pose_missing_object(self, cfg):
        """Test that pose computation returns None for missing objects."""
        mock_scene = self._create_mock_scene()
        resolver = PredicateResolver(scene=mock_scene, cfg=cfg)

        pose = resolver._compute_on_pose(target_id="nonexistent", ref_id="table_01")
        assert pose is None

        pose = resolver._compute_on_pose(target_id="cup_01", ref_id="nonexistent")
        assert pose is None

    def test_resolve_drake_model_name_regular_object(self, cfg):
        """Test drake model name resolution for regular objects."""
        mock_scene = self._create_mock_scene()
        resolver = PredicateResolver(scene=mock_scene, cfg=cfg)

        # Regular objects return their own ID.
        assert resolver._resolve_drake_model_name("cup_01") == "cup_01"
        assert resolver._resolve_drake_model_name("table_01") == "table_01"

    def test_resolve_drake_model_name_composite(self, cfg):
        """Test drake model name resolution for composite objects (stacks)."""
        mock_scene = self._create_mock_scene()
        # Add a composite object.
        mock_scene.scene_state["objects"]["stack_01"] = {
            "name": "Book Stack",
            "transform": {"translation": [0, 0, 0]},
            "bbox_min": [0, 0, 0],
            "bbox_max": [0.2, 0.2, 0.3],
            "metadata": {
                "composite_type": "stack",
                "member_model_names": ["book_0", "book_1", "book_2"],
            },
        }

        resolver = PredicateResolver(scene=mock_scene, cfg=cfg)

        # Composite stack returns topmost item.
        assert resolver._resolve_drake_model_name("stack_01") == "book_2"


class TestPolicyInterfaceAgent:
    """Tests for PolicyInterfaceAgent instantiation."""

    @pytest.fixture
    def cfg(self):
        """Create test config."""
        return create_robot_eval_config(model="gpt-5.2")

    def test_agent_creation(self, cfg):
        """Test that PolicyInterfaceAgent can be instantiated."""
        mock_scene = MagicMock()
        mock_scene.scene_state = {"objects": {}}

        agent = PolicyInterfaceAgent(scene=mock_scene, cfg=cfg)

        assert agent.scene == mock_scene
        assert agent.cfg == cfg
        assert agent.blender_server is None
        assert agent._agent is None  # Lazy initialization.
        assert cfg.openai.api_base == "https://api.openai.com/v1"

    def test_agent_with_blender_server(self, cfg):
        """Test PolicyInterfaceAgent with BlenderServer."""
        mock_scene = MagicMock()
        mock_scene.scene_state = {"objects": {}}
        mock_blender = MagicMock()

        agent = PolicyInterfaceAgent(
            scene=mock_scene, cfg=cfg, blender_server=mock_blender
        )

        assert agent.blender_server == mock_blender

    @pytest.mark.asyncio
    async def test_resolve_closes_cached_clients_in_finally(self, cfg):
        """Policy interface cleanup should run even when Runner.run fails."""
        mock_scene = MagicMock()
        mock_scene.scene_state = {"objects": {}}
        agent = PolicyInterfaceAgent(scene=mock_scene, cfg=cfg)
        agent._agent = MagicMock()

        with patch(
            "scenecode.robot_eval.policy_interface.policy_agent.Runner.run",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ), patch(
            "scenecode.robot_eval.policy_interface.policy_agent.close_cached_async_openai_clients",
            new=AsyncMock(),
        ) as mock_close_cached_clients:
            with pytest.raises(RuntimeError, match="boom"):
                await agent.resolve(task_description="Move the cup")

        mock_close_cached_clients.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resolve_passes_run_config_with_api_base(self, cfg):
        """Policy interface uses the configured api_base via RunConfig."""
        mock_scene = MagicMock()
        mock_scene.scene_state = {"objects": {}}
        agent = PolicyInterfaceAgent(scene=mock_scene, cfg=cfg)
        agent._agent = MagicMock()

        mock_output = PolicyInterfaceOutput(
            task_description="Move the cup",
            goal_predicate="on",
            target_category="cup",
            reference_category="table",
            valid_bindings=[],
            overall_success=False,
        )

        mock_result = MagicMock()
        mock_result.final_output_as.return_value = mock_output

        with patch(
            "scenecode.robot_eval.policy_interface.policy_agent.Runner.run",
            new=AsyncMock(return_value=mock_result),
        ) as mock_run:
            await agent.resolve(task_description="Move the cup")

        run_config = mock_run.call_args.kwargs["run_config"]
        assert (
            run_config.model_provider.openai_provider._stored_base_url
            == cfg.openai.api_base
        )
