import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch, sentinel

from omegaconf import OmegaConf

from scenecode.agent_utils.asset_router.dataclasses import AssetItem
from scenecode.agent_utils.asset_router.router import AssetRouter
from scenecode.agent_utils.room import AgentType, ObjectType


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestBaseAgentAssetStrategyDefaults(unittest.TestCase):
    """Verify shipped base configs default to code generation plus thin coverings."""

    def test_base_agent_configs_disable_generated_and_articulated(self):
        expected_configs = [
            (
                "configs/furniture_agent/base_furniture_agent.yaml",
                True,
            ),
            (
                "configs/manipuland_agent/base_manipuland_agent.yaml",
                True,
            ),
            (
                "configs/wall_agent/base_wall_agent.yaml",
                True,
            ),
            (
                "configs/ceiling_agent/base_ceiling_agent.yaml",
                False,
            ),
        ]

        for relative_path, expected_thin_covering in expected_configs:
            with self.subTest(config=relative_path):
                cfg = OmegaConf.load(REPO_ROOT / relative_path)
                strategies = cfg.asset_manager.router.strategies

                self.assertEqual(cfg.asset_manager.general_asset_source, "code_generated")
                self.assertTrue(strategies.code_generated.enabled)
                self.assertFalse(strategies.code_articulated.enabled)
                self.assertFalse(strategies.generated.enabled)
                self.assertFalse(strategies.articulated.enabled)
                self.assertEqual(
                    strategies.thin_covering.enabled,
                    expected_thin_covering,
                )


class TestDefaultAssetRouterBehavior(unittest.TestCase):
    """Smoke tests for default strategy behavior under shipped configs."""

    @staticmethod
    def _load_router() -> AssetRouter:
        cfg = OmegaConf.load(
            REPO_ROOT / "configs/furniture_agent/base_furniture_agent.yaml"
        )
        return AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=MagicMock(),
            cfg=cfg,
        )

    @staticmethod
    def _make_item(*, strategies: list[str], thin_covering_type: str | None = None) -> AssetItem:
        return AssetItem(
            description="test asset",
            short_name="test_asset",
            dimensions=[1.0, 1.0, 1.0],
            object_type=ObjectType.FURNITURE,
            strategies=strategies,
            thin_covering_type=thin_covering_type,
        )

    def test_code_generated_strategy_still_runs_with_default_config(self):
        router = self._load_router()
        item = self._make_item(strategies=["code_generated"])

        with (
            patch.object(
                router, "_try_code_generated_strategy", return_value=sentinel.code_result
            ) as mock_code_generated,
            patch.object(router, "_try_generated_strategy") as mock_generated,
            patch.object(router, "_try_articulated_strategy") as mock_articulated,
            patch.object(router, "_try_thin_covering_strategy") as mock_thin_covering,
        ):
            result = router.generate_with_validation(
                item=item,
                geometry_client=None,
                code_object_runner=MagicMock(),
                image_generator=MagicMock(),
                images_dir=Path("/tmp"),
                geometry_dir=Path("/tmp"),
                code_object_dir=Path("/tmp"),
                debug_dir=Path("/tmp"),
            )

        self.assertIs(result, sentinel.code_result)
        mock_code_generated.assert_called_once()
        mock_generated.assert_not_called()
        mock_articulated.assert_not_called()
        mock_thin_covering.assert_not_called()

    def test_thin_covering_strategy_remains_available_without_materials_client(self):
        router = self._load_router()
        item = self._make_item(
            strategies=["thin_covering"],
            thin_covering_type="tileable",
        )

        with (
            patch.object(
                router, "_try_thin_covering_strategy", return_value=sentinel.thin_result
            ) as mock_thin_covering,
            patch.object(router, "_try_code_generated_strategy") as mock_code_generated,
            patch.object(router, "_try_generated_strategy") as mock_generated,
            patch.object(router, "_try_articulated_strategy") as mock_articulated,
        ):
            result = router.generate_with_validation(
                item=item,
                geometry_client=None,
                code_object_runner=MagicMock(),
                image_generator=MagicMock(),
                images_dir=Path("/tmp"),
                geometry_dir=Path("/tmp"),
                code_object_dir=Path("/tmp"),
                debug_dir=Path("/tmp"),
            )

        self.assertIs(result, sentinel.thin_result)
        mock_thin_covering.assert_called_once()
        mock_code_generated.assert_not_called()
        mock_generated.assert_not_called()
        mock_articulated.assert_not_called()

    def test_disabled_articulated_strategy_falls_back_to_code_generated(self):
        router = self._load_router()
        item = self._make_item(strategies=["articulated", "code_generated"])

        with (
            patch.object(router, "_try_articulated_strategy") as mock_articulated,
            patch.object(
                router, "_try_code_generated_strategy", return_value=sentinel.code_result
            ) as mock_code_generated,
            patch.object(router, "_try_generated_strategy") as mock_generated,
            patch.object(router, "_try_thin_covering_strategy") as mock_thin_covering,
        ):
            result = router.generate_with_validation(
                item=item,
                geometry_client=None,
                code_object_runner=MagicMock(),
                image_generator=MagicMock(),
                images_dir=Path("/tmp"),
                geometry_dir=Path("/tmp"),
                code_object_dir=Path("/tmp"),
                debug_dir=Path("/tmp"),
                articulated_client=MagicMock(),
            )

        self.assertIs(result, sentinel.code_result)
        mock_articulated.assert_not_called()
        mock_code_generated.assert_called_once()
        mock_generated.assert_not_called()
        mock_thin_covering.assert_not_called()


if __name__ == "__main__":
    unittest.main()
