"""Unit tests for the asset router module."""

import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from scenecode.agent_utils.asset_router import AssetRouter
from scenecode.agent_utils.asset_router.dataclasses import (
    AnalysisResult,
    AssetItem,
    GeneratedGeometry,
    ValidationResult,
)
from scenecode.agent_utils.room import AgentType, ObjectType
from scenecode.utils.material import Material


class TestAnalysisResultWasModified(unittest.TestCase):
    """Test the was_modified computed property logic."""

    def test_single_item_not_modified(self) -> None:
        """Single item with no original_description is not modified."""
        item = AssetItem(
            description="wooden ladder",
            short_name="ladder",
            dimensions=[0.5, 0.3, 2.0],
            object_type=ObjectType.FURNITURE,
            strategies=["generated"],
        )
        result = AnalysisResult(
            items=[item],
            original_description=None,
            discarded_manipulands=None,
        )
        assert not result.was_modified

    def test_with_original_description_is_modified(self) -> None:
        """Items with original_description set is modified (was split/filtered)."""
        items = [
            AssetItem(
                description="dining table",
                short_name="dining_table",
                dimensions=[1.5, 0.9, 0.75],
                object_type=ObjectType.FURNITURE,
                strategies=["generated"],
            ),
        ]
        result = AnalysisResult(
            items=items,
            original_description="dining table and four chairs",
            discarded_manipulands=None,
        )
        assert result.was_modified

    def test_with_discarded_manipulands_is_modified(self) -> None:
        """Request with discarded manipulands is modified."""
        item = AssetItem(
            description="ladder",
            short_name="ladder",
            dimensions=[0.5, 0.3, 2.0],
            object_type=ObjectType.FURNITURE,
            strategies=["generated"],
        )
        result = AnalysisResult(
            items=[item],
            original_description="ladder with flower pots",
            discarded_manipulands=["flower pots"],
        )
        assert result.was_modified


class TestAssetRouterItemTypeValidation(unittest.TestCase):
    """Test validate_item_types method behavior."""

    def test_furniture_items_valid_for_furniture_agent(self) -> None:
        """Furniture items are valid for furniture agent."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        items = [
            AssetItem(
                description="desk",
                short_name="desk",
                dimensions=[1.2, 0.6, 0.75],
                object_type=ObjectType.FURNITURE,
                strategies=["generated"],
            ),
        ]

        error = router.validate_item_types(items)
        assert error is None

    def test_manipuland_items_valid_for_manipuland_agent(self) -> None:
        """Manipuland items are valid for manipuland agent."""
        router = AssetRouter(
            agent_type=AgentType.MANIPULAND, vlm_service=MagicMock(), cfg=MagicMock()
        )

        items = [
            AssetItem(
                description="coffee mug",
                short_name="mug",
                dimensions=[0.08, 0.08, 0.1],
                object_type=ObjectType.MANIPULAND,
                strategies=["generated"],
            ),
        ]

        error = router.validate_item_types(items)
        assert error is None

    def test_either_type_valid_for_both_agents(self) -> None:
        """EITHER type items are valid for both furniture and manipuland agents."""
        item = AssetItem(
            description="potted plant",
            short_name="potted_plant",
            dimensions=[0.3, 0.3, 0.6],
            object_type=ObjectType.EITHER,
            strategies=["generated"],
        )

        furniture_router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )
        assert furniture_router.validate_item_types([item]) is None

        manipuland_router = AssetRouter(
            agent_type=AgentType.MANIPULAND, vlm_service=MagicMock(), cfg=MagicMock()
        )
        assert manipuland_router.validate_item_types([item]) is None

    def test_wrong_type_returns_error(self) -> None:
        """Wrong item type for agent returns error message."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        items = [
            AssetItem(
                description="coffee mug",
                short_name="mug",
                dimensions=[0.08, 0.08, 0.1],
                object_type=ObjectType.MANIPULAND,
                strategies=["generated"],
            ),
        ]

        error = router.validate_item_types(items)
        assert error is not None
        assert "manipuland" in error.lower()


class TestThinCoveringGeneratedMaterials(unittest.TestCase):
    """Thin coverings use generated materials, not material retrieval."""

    @staticmethod
    def _make_router() -> AssetRouter:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "router": {
                        "strategies": {
                            "thin_covering": {
                                "enabled": True,
                                "max_retries": 2,
                                "thickness_m": 0.003,
                                "texture_scale": 0.5,
                                "generator": {
                                    "enabled": True,
                                    "backend": "openai",
                                    "max_retries": 1,
                                    "default_roughness": 128,
                                    "texture_scale": 0.5,
                                },
                            }
                        }
                    }
                }
            }
        )
        return AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=MagicMock(),
            cfg=cfg,
        )

    def test_thin_covering_generates_material_with_no_retrieval_client(self) -> None:
        router = self._make_router()
        item = AssetItem(
            description="persian rug pattern",
            short_name="persian_rug",
            dimensions=[2.0, 1.5, 0.01],
            object_type=ObjectType.FURNITURE,
            strategies=["thin_covering"],
            thin_covering_type="tileable",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            material = Material(
                path=temp_path / "generated_material",
                material_id="generated_material",
                texture_scale=0.5,
            )
            generated_geometry = GeneratedGeometry(
                geometry_path=temp_path / "persian_rug.glb",
                item=item,
                asset_source="thin_covering",
            )

            with (
                patch(
                    "scenecode.agent_utils.materials_retrieval_server.client."
                    "MaterialsRetrievalClient.retrieve_materials"
                ) as mock_retrieve_materials,
                patch(
                    "scenecode.agent_utils.asset_router.router."
                    "MaterialGenerator.generate_material",
                    return_value=material,
                ) as mock_generate_material,
                patch.object(
                    router,
                    "_generate_thin_covering_geometry",
                    return_value=generated_geometry,
                ) as mock_generate_geometry,
                patch.object(
                    router,
                    "_validate_thin_covering",
                    return_value=ValidationResult(is_acceptable=True, reason="ok"),
                ) as mock_validate,
            ):
                result = router.generate_with_validation(
                    item=item,
                    geometry_client=None,
                    code_object_runner=None,
                    image_generator=MagicMock(),
                    images_dir=temp_path,
                    geometry_dir=temp_path,
                    code_object_dir=temp_path,
                    debug_dir=temp_path,
                    materials_client=None,
                )

        self.assertIs(result, generated_geometry)
        mock_retrieve_materials.assert_not_called()
        mock_generate_material.assert_called_once_with(description=item.description)
        mock_validate.assert_called_once()
        self.assertEqual(mock_generate_geometry.call_count, 1)
        geometry_kwargs = mock_generate_geometry.call_args.kwargs
        self.assertEqual(geometry_kwargs["material_path"], material.path)
        self.assertEqual(geometry_kwargs["texture_scale"], 0.5)
        self.assertEqual(geometry_kwargs["width"], 2.0)
        self.assertEqual(geometry_kwargs["second_dim"], 1.5)


class TestAnalysisResponseParsing(unittest.TestCase):
    """Test parsing of VLM analysis responses."""

    def test_parse_single_furniture_item(self) -> None:
        """Parse single furniture item response."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [
                {
                    "description": "wooden ladder",
                    "short_name": "ladder",
                    "dimensions": [0.5, 0.3, 2.0],
                    "object_type": "FURNITURE",
                    "strategies": ["generated"],
                }
            ],
            "original_description": None,
            "discarded_manipulands": None,
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 1
        assert result.items[0].description == "wooden ladder"
        assert result.items[0].object_type == ObjectType.FURNITURE
        assert not result.was_modified

    def test_parse_composite_split(self) -> None:
        """Parse response with composite split into multiple items."""
        router = AssetRouter(
            agent_type=AgentType.MANIPULAND, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [
                {
                    "description": "fruit bowl",
                    "short_name": "fruit_bowl",
                    "dimensions": [0.3, 0.3, 0.10],
                    "object_type": "MANIPULAND",
                    "strategies": ["generated"],
                },
                {
                    "description": "apple",
                    "short_name": "apple",
                    "dimensions": [0.08, 0.08, 0.08],
                    "object_type": "MANIPULAND",
                    "strategies": ["generated"],
                },
            ],
            "original_description": "fruit bowl with apples",
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 2
        assert result.was_modified
        assert result.original_description == "fruit bowl with apples"

    def test_parse_error_response(self) -> None:
        """Parse error response from VLM."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [],
            "original_description": None,
            "discarded_manipulands": None,
            "error": "Request is for a manipuland (coffee mug), not furniture.",
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 0
        assert result.error is not None
        assert "manipuland" in result.error.lower()

    def test_parse_error_response_preserves_original_description(self) -> None:
        """Error responses preserve original_description for debugging."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [],
            "original_description": "stack of 4 car tires",
            "discarded_manipulands": None,
            "error": "Stackable items should be handled by manipuland agent.",
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 0
        assert result.error is not None
        assert result.original_description == "stack of 4 car tires"

    def test_parse_with_discarded_manipulands(self) -> None:
        """Parse response with discarded manipulands (furniture agent filtering)."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [
                {
                    "description": "bookshelf",
                    "short_name": "bookshelf",
                    "dimensions": [1.0, 0.3, 2.0],
                    "object_type": "FURNITURE",
                    "strategies": ["generated"],
                }
            ],
            "original_description": "bookshelf with books and decorations",
            "discarded_manipulands": ["books", "decorations"],
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 1
        assert result.was_modified
        assert result.discarded_manipulands == ["books", "decorations"]

    def test_parse_lowercase_object_type(self) -> None:
        """Object type parsing is case-insensitive."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [
                {
                    "description": "desk",
                    "short_name": "desk",
                    "dimensions": [1.2, 0.6, 0.75],
                    "object_type": "furniture",  # lowercase
                    "strategies": ["generated"],
                }
            ],
            "original_description": None,
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 1
        assert result.items[0].object_type == ObjectType.FURNITURE

    def test_parse_code_generated_strategy_is_preserved(self) -> None:
        """code_generated stays distinct from legacy generated when parsing."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [
                {
                    "description": "desk lamp",
                    "short_name": "desk_lamp",
                    "dimensions": [0.2, 0.2, 0.45],
                    "object_type": "FURNITURE",
                    "strategies": ["code_generated"],
                }
            ],
            "original_description": None,
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 1
        assert result.items[0].strategies == ["code_generated"]


    def test_parse_code_object_profile_is_preserved(self) -> None:
        """code_object_profile is preserved for specialized code-generated assets."""
        router = AssetRouter(
            agent_type=AgentType.WALL_MOUNTED,
            vlm_service=MagicMock(),
            cfg=MagicMock(),
        )

        response = {
            "items": [
                {
                    "description": "framed landscape painting",
                    "short_name": "painting",
                    "dimensions": [0.8, 0.6, 0.05],
                    "object_type": "WALL_MOUNTED",
                    "strategies": ["code_generated"],
                    "code_object_profile": "wall_art",
                }
            ],
            "original_description": None,
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 1
        assert result.items[0].code_object_profile == "wall_art"

    def test_parse_non_wall_art_code_object_profiles_are_preserved(self) -> None:
        """Non-wall-art code_object_profile values pass through unchanged."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=MagicMock(),
            cfg=MagicMock(),
        )

        response = {
            "items": [
                {
                    "description": "simple decorative bowl",
                    "short_name": "bowl",
                    "dimensions": [0.3, 0.3, 0.2],
                    "object_type": "FURNITURE",
                    "strategies": ["code_generated"],
                    "code_object_profile": "SimpleObject",
                },
                {
                    "description": "small handled mug",
                    "short_name": "mug",
                    "dimensions": [0.12, 0.09, 0.11],
                    "object_type": "MANIPULAND",
                    "strategies": ["code_generated"],
                    "code_object_profile": "manipuland",
                },
            ],
            "original_description": None,
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 2
        assert result.items[0].code_object_profile == "SimpleObject"
        assert result.items[1].code_object_profile == "manipuland"


    def test_wall_art_descriptions_normalize_to_code_generated_wall_art(self) -> None:
        """Wall-art descriptions are stabilized onto the wall_art Code_Object path."""
        router = AssetRouter(
            agent_type=AgentType.WALL_MOUNTED,
            vlm_service=MagicMock(),
            cfg=MagicMock(),
        )

        descriptions = [
            "abstract painting",
            "canvas wall print",
            "framed landscape painting",
            "picture frame",
        ]
        for description in descriptions:
            with self.subTest(description=description):
                response = {
                    "items": [
                        {
                            "description": description,
                            "short_name": "wall_item",
                            "dimensions": [0.8, 0.6, 0.05],
                            "object_type": "WALL_MOUNTED",
                            "strategies": ["thin_covering"],
                            "thin_covering_type": "single_image",
                        }
                    ],
                    "original_description": None,
                }

                result = router._parse_analysis_response(response)
                item = result.items[0]
                assert item.strategies == ["code_generated"]
                assert item.code_object_profile == "wall_art"
                assert item.thin_covering_type is None

    def test_repeat_wall_coverings_remain_thin_covering(self) -> None:
        """Wallpaper-like coverings stay on thin_covering instead of wall_art."""
        router = AssetRouter(
            agent_type=AgentType.WALL_MOUNTED,
            vlm_service=MagicMock(),
            cfg=MagicMock(),
        )

        response = {
            "items": [
                {
                    "description": "wallpaper section",
                    "short_name": "wallpaper",
                    "dimensions": [1.2, 1.0, 0.01],
                    "object_type": "WALL_MOUNTED",
                    "strategies": ["code_generated"],
                    "code_object_profile": "wall_art",
                }
            ],
            "original_description": None,
        }

        result = router._parse_analysis_response(response)
        item = result.items[0]
        assert item.strategies == ["thin_covering"]
        assert item.thin_covering_type == "tileable"
        assert item.code_object_profile is None

    def test_parse_legacy_generated_strategy_is_preserved(self) -> None:
        """legacy generated remains supported as its own distinct strategy."""
        router = AssetRouter(
            agent_type=AgentType.FURNITURE, vlm_service=MagicMock(), cfg=MagicMock()
        )

        response = {
            "items": [
                {
                    "description": "desk lamp",
                    "short_name": "desk_lamp",
                    "dimensions": [0.2, 0.2, 0.45],
                    "object_type": "FURNITURE",
                    "strategies": ["generated"],
                }
            ],
            "original_description": None,
        }

        result = router._parse_analysis_response(response)
        assert len(result.items) == 1
        assert result.items[0].strategies == ["generated"]


if __name__ == "__main__":
    unittest.main()
