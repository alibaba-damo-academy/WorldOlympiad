import json
import shutil
import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

import trimesh

from omegaconf import OmegaConf

from scenecode.agent_utils.asset_router import AssetRouter
from scenecode.agent_utils.asset_router.dataclasses import (
    AssetItem,
    CodeArticulatedGeometry,
    GeneratedGeometry,
    ValidationResult,
)
from scenecode.agent_utils.code_object_generation import (
    CodeObjectGenerationResult,
    CodeObjectRunner,
)
from scenecode.agent_utils.wall_art_preprocess import prepare_wall_art_reference_assets
from scenecode.agent_utils.room import AgentType, ObjectType


class TestCodeObjectRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.image_path = self.temp_dir / "reference.png"
        self.image_path.write_bytes(b"png")
        self.config_path = self.temp_dir / "config.yaml"
        self.config_path.write_text("model: {}\n", encoding="utf-8")
        self.output_dir = self.temp_dir / "output"
        self.cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "code_object": {
                        "config_path": str(self.config_path),
                        "blender_mcp_port_range": [9900, 9999],
                        "max_concurrent_runs": 1,
                        "max_attempts": 3,
                    }
                }
            }
        )
        self.runner = CodeObjectRunner(self.cfg)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _make_pipeline_class(self, behaviors, calls, init_calls=None):
        class FakePipeline:
            def __init__(self, config_path: str, use_mock: bool, port: int) -> None:
                self.config_path = config_path
                self.use_mock = use_mock
                self.port = port
                if init_calls is not None:
                    init_calls.append(config_path)

            async def run(self, **kwargs):
                calls.append(kwargs.copy())
                behavior = behaviors.pop(0)
                output_dir = Path(kwargs["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)

                if behavior.get("create_plan"):
                    (output_dir / "ObjectPlan.json").write_text("{}", encoding="utf-8")

                part_names = behavior.get("create_parts", [])
                if part_names:
                    parts_dir = output_dir / "code" / "parts_test_object"
                    parts_dir.mkdir(parents=True, exist_ok=True)
                    for part_name in part_names:
                        (parts_dir / f"{part_name}.py").write_text(
                            f"# {part_name}\n", encoding="utf-8"
                        )

                mesh_path = None
                if behavior.get("create_mesh"):
                    mesh_path = output_dir / "mesh" / "test_object.glb"
                    mesh_path.parent.mkdir(parents=True, exist_ok=True)
                    trimesh.creation.box(extents=[1.0, 1.0, 1.0]).export(mesh_path)
                elif behavior.get("create_gltf"):
                    mesh_path = output_dir / "gltf" / "test_object.gltf"
                    mesh_path.parent.mkdir(parents=True, exist_ok=True)
                    trimesh.creation.box(extents=[1.0, 1.0, 1.0]).export(mesh_path)

                return SimpleNamespace(
                    success=behavior.get("success", False),
                    status=behavior.get("status", "generated"),
                    error=behavior.get("error"),
                    stages_failed=behavior.get("stages_failed", []),
                    stages_completed=behavior.get("stages_completed", []),
                    mesh_path=str(mesh_path) if mesh_path is not None else None,
                    gltf_path=(
                        str(mesh_path)
                        if behavior.get("legacy_gltf_field") and mesh_path is not None
                        else None
                    ),
                    urdf_path=behavior.get("urdf_path"),
                )

        return FakePipeline

    def _write_existing_plan_and_parts(self, part_names: list[str]) -> None:
        (self.output_dir / "ObjectPlan.json").parent.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "ObjectPlan.json").write_text("{}", encoding="utf-8")
        parts_dir = self.output_dir / "code" / "parts_test_object"
        parts_dir.mkdir(parents=True, exist_ok=True)
        for part_name in part_names:
            (parts_dir / f"{part_name}.py").write_text("# existing\n", encoding="utf-8")

    def test_generate_from_image_retries_same_output_dir_until_success(self) -> None:
        behaviors = [
            {"success": False, "error": "rate limited"},
            {"success": True, "create_mesh": True},
        ]
        calls = []
        with (
            patch.object(
                self.runner,
                "_load_pipeline_class",
                return_value=self._make_pipeline_class(behaviors, calls),
            ),
            patch.object(self.runner, "_reserve_port", return_value=9911),
            patch.object(self.runner, "_release_port"),
        ):
            result = self.runner.generate_from_image(
                image_path=self.image_path,
                output_dir=self.output_dir,
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["output_dir"], str(self.output_dir))
        self.assertEqual(calls[1]["output_dir"], str(self.output_dir))
        self.assertEqual(calls[0]["image_path"], str(self.image_path))
        self.assertEqual(calls[1]["image_path"], str(self.image_path))
        self.assertEqual(calls[0]["skip_stages"], [])
        self.assertEqual(calls[1]["skip_stages"], [])
        self.assertTrue(result.mesh_path.exists())
        self.assertEqual(result.mesh_path.suffix, ".glb")

    def test_existing_plan_does_not_force_plan_check_skip(self) -> None:
        behaviors = [
            {"success": False, "create_plan": True, "error": "401"},
            {"success": True, "create_mesh": True},
        ]
        calls = []
        with (
            patch.object(
                self.runner,
                "_load_pipeline_class",
                return_value=self._make_pipeline_class(behaviors, calls),
            ),
            patch.object(self.runner, "_reserve_port", return_value=9911),
            patch.object(self.runner, "_release_port"),
        ):
            result = self.runner.generate_from_image(
                image_path=self.image_path,
                output_dir=self.output_dir,
            )

        self.assertEqual(calls[0]["skip_stages"], [])
        self.assertEqual(calls[1]["skip_stages"], [])
        self.assertEqual(result.object_plan_path, self.output_dir / "ObjectPlan.json")

    def test_existing_part_causes_retry_to_skip_plan_check(self) -> None:
        behaviors = [
            {
                "success": False,
                "create_plan": True,
                "create_parts": ["legs"],
                "error": "401",
            },
            {"success": True, "create_mesh": True},
        ]
        calls = []
        with (
            patch.object(
                self.runner,
                "_load_pipeline_class",
                return_value=self._make_pipeline_class(behaviors, calls),
            ),
            patch.object(self.runner, "_reserve_port", return_value=9911),
            patch.object(self.runner, "_release_port"),
        ):
            self.runner.generate_from_image(
                image_path=self.image_path,
                output_dir=self.output_dir,
            )

        self.assertEqual(calls[0]["skip_stages"], [])
        self.assertEqual(calls[1]["skip_stages"], ["plan_check"])

    def test_existing_parts_only_skip_plan_check_not_construction_or_export(self) -> None:
        self._write_existing_plan_and_parts(["frame", "legs"])
        behaviors = [
            {"success": False, "stages_failed": ["export_glb"]},
            {"success": True, "create_mesh": True},
        ]
        calls = []
        with (
            patch.object(
                self.runner,
                "_load_pipeline_class",
                return_value=self._make_pipeline_class(behaviors, calls),
            ),
            patch.object(self.runner, "_reserve_port", return_value=9911),
            patch.object(self.runner, "_release_port"),
        ):
            self.runner.generate_from_image(
                image_path=self.image_path,
                output_dir=self.output_dir,
            )

        self.assertEqual(calls[0]["skip_stages"], ["plan_check"])
        self.assertEqual(calls[1]["skip_stages"], ["plan_check"])
        self.assertNotIn("construction", calls[0]["skip_stages"])
        self.assertNotIn("export_glb", calls[0]["skip_stages"])

    def test_all_attempts_failed_raises_summary_error(self) -> None:
        behaviors = [
            {"success": False, "error": "401 unauthorized"},
            {"success": False, "stages_failed": ["construction"]},
            {
                "success": False,
                "stages_failed": ["export_glb"],
                "error": "missing module",
            },
        ]
        calls = []
        with (
            patch.object(
                self.runner,
                "_load_pipeline_class",
                return_value=self._make_pipeline_class(behaviors, calls),
            ),
            patch.object(self.runner, "_reserve_port", return_value=9911),
            patch.object(self.runner, "_release_port") as mock_release_port,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.runner.generate_from_image(
                    image_path=self.image_path,
                    output_dir=self.output_dir,
                )

        message = str(ctx.exception)
        self.assertIn("after 3 attempt(s)", message)
        self.assertIn("attempt 3/3", message)
        self.assertIn("stages_failed=['export_glb']", message)
        self.assertEqual(len(calls), 3)
        mock_release_port.assert_called_once_with(9911)

    def test_resolve_mesh_path_supports_legacy_gltf_field(self) -> None:
        legacy_gltf = self.output_dir / "gltf" / "legacy_object.gltf"
        legacy_gltf.parent.mkdir(parents=True, exist_ok=True)
        trimesh.creation.box(extents=[1.0, 1.0, 1.0]).export(legacy_gltf)

        result = SimpleNamespace(
            success=True,
            mesh_path=None,
            gltf_path=str(legacy_gltf),
        )

        resolved = self.runner._resolve_mesh_path(result, self.output_dir)

        self.assertEqual(resolved, legacy_gltf.with_suffix(".glb"))
        self.assertTrue(resolved.exists())

    def test_resolve_mesh_path_prefers_new_mesh_directory(self) -> None:
        mesh_path = self.output_dir / "mesh" / "test_object.glb"
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        trimesh.creation.box(extents=[1.0, 1.0, 1.0]).export(mesh_path)

        resolved = self.runner._resolve_mesh_path(SimpleNamespace(), self.output_dir)

        self.assertEqual(resolved, mesh_path)

    def test_generate_articulated_from_image_returns_generated_status_and_urdf(self) -> None:
        urdf_path = self.output_dir / "test_object.urdf"
        behaviors = [
            {
                "success": True,
                "status": "generated",
                "create_mesh": True,
                "urdf_path": str(urdf_path),
            }
        ]
        calls = []
        with (
            patch.object(
                self.runner,
                "_load_pipeline_class",
                return_value=self._make_pipeline_class(behaviors, calls),
            ),
            patch.object(self.runner, "_reserve_port", return_value=9911),
            patch.object(self.runner, "_release_port"),
        ):
            urdf_path.parent.mkdir(parents=True, exist_ok=True)
            urdf_path.write_text("<robot/>", encoding="utf-8")
            result = self.runner.generate_articulated_from_image(
                image_path=self.image_path,
                output_dir=self.output_dir,
            )

        self.assertEqual(result.status, "generated")
        self.assertEqual(result.urdf_path, urdf_path)
        self.assertEqual(len(calls), 1)

    def test_generate_articulated_from_image_returns_no_movable_parts(self) -> None:
        behaviors = [
            {
                "success": True,
                "status": "no_movable_parts",
                "create_mesh": True,
            }
        ]
        calls = []
        with (
            patch.object(
                self.runner,
                "_load_pipeline_class",
                return_value=self._make_pipeline_class(behaviors, calls),
            ),
            patch.object(self.runner, "_reserve_port", return_value=9911),
            patch.object(self.runner, "_release_port"),
        ):
            result = self.runner.generate_articulated_from_image(
                image_path=self.image_path,
                output_dir=self.output_dir,
            )

        self.assertEqual(result.status, "no_movable_parts")
        self.assertIsNone(result.urdf_path)
        self.assertEqual(len(calls), 1)

    def test_generate_from_image_uses_config_override_when_provided(self) -> None:
        override_config_path = self.temp_dir / "config_WallArt.yaml"
        override_config_path.write_text("model: {}\n", encoding="utf-8")
        behaviors = [{"success": True, "create_mesh": True}]
        calls = []
        init_calls = []
        with (
            patch.object(
                self.runner,
                "_load_pipeline_class",
                return_value=self._make_pipeline_class(behaviors, calls, init_calls),
            ),
            patch.object(self.runner, "_reserve_port", return_value=9911),
            patch.object(self.runner, "_release_port"),
        ):
            self.runner.generate_from_image(
                image_path=self.image_path,
                output_dir=self.output_dir,
                config_path_override=override_config_path,
            )

        self.assertEqual(init_calls, [str(override_config_path.resolve())])


class TestAssetRouterCodeObjectCompatibility(unittest.TestCase):
    def test_router_calls_runner_once_per_candidate(self) -> None:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "side_view_elevation_degrees": 20.0,
                    "validation_taa_samples": 8,
                }
            }
        )
        router = AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=MagicMock(),
            cfg=cfg,
        )
        item = AssetItem(
            description="modern chair",
            short_name="chair",
            dimensions=[0.6, 0.6, 0.9],
            object_type=ObjectType.FURNITURE,
            strategies=["code_generated"],
        )
        temp_dir = Path(tempfile.mkdtemp())
        try:
            code_object_output_dir = temp_dir / "candidate"
            result_bundle = CodeObjectGenerationResult(
                output_dir=code_object_output_dir,
                mesh_path=code_object_output_dir / "mesh" / "chair.glb",
                object_plan_path=None,
                code_dir=None,
                pipeline_result_path=None,
                full_object_render_path=None,
            )
            runner = MagicMock()
            runner.generate_from_image.return_value = result_bundle
            image_generator = MagicMock()

            result = router._try_code_generated_strategy(
                item=item,
                max_retries=0,
                code_object_runner=runner,
                image_generator=image_generator,
                images_dir=temp_dir / "images",
                code_object_dir=temp_dir / "code_object",
                debug_dir=temp_dir / "debug",
                style_context="Modern style",
            )

            runner.generate_from_image.assert_called_once()
            call_kwargs = runner.generate_from_image.call_args.kwargs
            self.assertIsNone(call_kwargs["config_path_override"])
            image_generator.generate_images.assert_called_once()
            self.assertIsNotNone(result)
            self.assertEqual(result.asset_source, "code_generated")
            self.assertEqual(result.geometry_path.suffix, ".glb")
        finally:
            shutil.rmtree(temp_dir)

    def test_final_attempt_accepts_without_validation(self) -> None:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "side_view_elevation_degrees": 20.0,
                    "validation_taa_samples": 8,
                }
            }
        )
        router = AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=MagicMock(),
            cfg=cfg,
        )
        item = AssetItem(
            description="modern chair",
            short_name="chair",
            dimensions=[0.6, 0.6, 0.9],
            object_type=ObjectType.FURNITURE,
            strategies=["code_generated"],
        )
        temp_dir = Path(tempfile.mkdtemp())
        try:
            first_result = GeneratedGeometry(
                geometry_path=temp_dir / "first.glb",
                item=item,
                asset_source="code_generated",
            )
            final_result = GeneratedGeometry(
                geometry_path=temp_dir / "final.glb",
                item=item,
                asset_source="code_generated",
            )

            with (
                patch.object(
                    router,
                    "_generate_code_object_geometry",
                    side_effect=[first_result, final_result],
                ) as mock_generate,
                patch.object(
                    router,
                    "validate_asset",
                    return_value=ValidationResult(
                        is_acceptable=False,
                        reason="bad render",
                        suggestions=["retry"],
                    ),
                ) as mock_validate,
            ):
                result = router._try_code_generated_strategy(
                    item=item,
                    max_retries=2,
                    code_object_runner=MagicMock(),
                    image_generator=MagicMock(),
                    images_dir=temp_dir / "images",
                    code_object_dir=temp_dir / "code_object",
                    debug_dir=temp_dir / "debug",
                    style_context="Modern style",
                )

            self.assertIs(result, final_result)
            self.assertEqual(mock_generate.call_count, 2)
            mock_validate.assert_called_once_with(
                mesh_path=first_result.geometry_path,
                description=item.description,
                output_dir=temp_dir / "debug" / "chair_code_generated_validation",
                use_lenient=False,
            )
        finally:
            shutil.rmtree(temp_dir)

    def test_only_non_final_attempts_are_validated(self) -> None:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "side_view_elevation_degrees": 20.0,
                    "validation_taa_samples": 8,
                }
            }
        )
        router = AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=MagicMock(),
            cfg=cfg,
        )
        item = AssetItem(
            description="modern chair",
            short_name="chair",
            dimensions=[0.6, 0.6, 0.9],
            object_type=ObjectType.FURNITURE,
            strategies=["code_generated"],
        )
        temp_dir = Path(tempfile.mkdtemp())
        try:
            results = [
                GeneratedGeometry(
                    geometry_path=temp_dir / f"candidate_{idx}.glb",
                    item=item,
                    asset_source="code_generated",
                )
                for idx in range(3)
            ]

            with (
                patch.object(
                    router,
                    "_generate_code_object_geometry",
                    side_effect=results,
                ) as mock_generate,
                patch.object(
                    router,
                    "validate_asset",
                    side_effect=[
                        ValidationResult(
                            is_acceptable=False,
                            reason="bad render 1",
                            suggestions=["retry"],
                        ),
                        ValidationResult(
                            is_acceptable=False,
                            reason="bad render 2",
                            suggestions=["retry"],
                        ),
                    ],
                ) as mock_validate,
            ):
                result = router._try_code_generated_strategy(
                    item=item,
                    max_retries=3,
                    code_object_runner=MagicMock(),
                    image_generator=MagicMock(),
                    images_dir=temp_dir / "images",
                    code_object_dir=temp_dir / "code_object",
                    debug_dir=temp_dir / "debug",
                    style_context="Modern style",
                )

            self.assertIs(result, results[-1])
            self.assertEqual(mock_generate.call_count, 3)
            self.assertEqual(mock_validate.call_count, 2)
            self.assertEqual(
                [call.kwargs["mesh_path"] for call in mock_validate.call_args_list],
                [results[0].geometry_path, results[1].geometry_path],
            )
        finally:
            shutil.rmtree(temp_dir)

    def test_final_attempt_failure_does_not_accept(self) -> None:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "side_view_elevation_degrees": 20.0,
                    "validation_taa_samples": 8,
                }
            }
        )
        router = AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=MagicMock(),
            cfg=cfg,
        )
        item = AssetItem(
            description="modern chair",
            short_name="chair",
            dimensions=[0.6, 0.6, 0.9],
            object_type=ObjectType.FURNITURE,
            strategies=["code_generated"],
        )
        temp_dir = Path(tempfile.mkdtemp())
        try:
            first_result = GeneratedGeometry(
                geometry_path=temp_dir / "first.glb",
                item=item,
                asset_source="code_generated",
            )

            with (
                patch.object(
                    router,
                    "_generate_code_object_geometry",
                    side_effect=[first_result, None],
                ) as mock_generate,
                patch.object(
                    router,
                    "validate_asset",
                    return_value=ValidationResult(
                        is_acceptable=False,
                        reason="bad render",
                        suggestions=["retry"],
                    ),
                ) as mock_validate,
            ):
                result = router._try_code_generated_strategy(
                    item=item,
                    max_retries=2,
                    code_object_runner=MagicMock(),
                    image_generator=MagicMock(),
                    images_dir=temp_dir / "images",
                    code_object_dir=temp_dir / "code_object",
                    debug_dir=temp_dir / "debug",
                    style_context="Modern style",
                )

            self.assertIsNone(result)
            self.assertEqual(mock_generate.call_count, 2)
            mock_validate.assert_called_once()
        finally:
            shutil.rmtree(temp_dir)

    def test_code_articulated_strategy_returns_pending_articulated_geometry(self) -> None:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "side_view_elevation_degrees": 20.0,
                    "validation_taa_samples": 8,
                }
            }
        )
        router = AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=MagicMock(),
            cfg=cfg,
        )
        item = AssetItem(
            description="storage cabinet",
            short_name="cabinet",
            dimensions=[1.0, 0.5, 1.4],
            object_type=ObjectType.FURNITURE,
            strategies=["code_articulated"],
        )
        temp_dir = Path(tempfile.mkdtemp())
        try:
            pending = CodeArticulatedGeometry(
                urdf_path=temp_dir / "cabinet.urdf",
                item=item,
                image_path=temp_dir / "cabinet.png",
                geometry_path=temp_dir / "cabinet.glb",
            )
            with (
                patch.object(
                    router,
                    "_generate_code_articulated_geometry",
                    return_value=pending,
                ) as mock_generate,
                patch.object(router, "validate_asset") as mock_validate,
            ):
                result = router._try_code_articulated_strategy(
                    item=item,
                    max_retries=1,
                    code_object_runner=MagicMock(),
                    image_generator=MagicMock(),
                    images_dir=temp_dir / "images",
                    code_object_dir=temp_dir / "code_object",
                    debug_dir=temp_dir / "debug",
                    style_context="Modern style",
                )

            self.assertIs(result, pending)
            mock_generate.assert_called_once()
            mock_validate.assert_not_called()
        finally:
            shutil.rmtree(temp_dir)

    def test_code_articulated_generation_passes_blender_server_to_conversion(self) -> None:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "side_view_elevation_degrees": 20.0,
                    "validation_taa_samples": 8,
                }
            }
        )
        blender_server = MagicMock()
        router = AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=MagicMock(),
            cfg=cfg,
            blender_server=blender_server,
            collision_client=MagicMock(),
        )
        item = AssetItem(
            description="storage cabinet",
            short_name="cabinet",
            dimensions=[1.0, 0.5, 1.4],
            object_type=ObjectType.FURNITURE,
            strategies=["code_articulated"],
        )
        temp_dir = Path(tempfile.mkdtemp())
        try:
            image_path = temp_dir / "cabinet.png"
            output_dir = temp_dir / "code_object"
            mesh_path = output_dir / "mesh" / "cabinet.glb"
            urdf_path = output_dir / "cabinet.urdf"
            result_bundle = CodeObjectGenerationResult(
                output_dir=output_dir,
                mesh_path=mesh_path,
                object_plan_path=None,
                code_dir=None,
                pipeline_result_path=None,
                full_object_render_path=None,
                urdf_path=urdf_path,
            )
            runner = MagicMock()
            runner.generate_articulated_from_image.return_value = result_bundle

            with (
                patch.object(
                    router,
                    "_prepare_code_object_inputs",
                    return_value=(image_path, output_dir, None),
                ),
                patch(
                    "scenecode.agent_utils.asset_router.router.convert_generated_articulated_urdf",
                    return_value=SimpleNamespace(
                        sdf_path=output_dir / "cabinet.sdf",
                        analysis_path=output_dir / "analysis.json",
                    ),
                ) as mock_convert,
            ):
                result = router._generate_code_articulated_geometry(
                    item=item,
                    code_object_runner=runner,
                    image_generator=MagicMock(),
                    images_dir=temp_dir / "images",
                    code_object_dir=temp_dir / "code_object",
                    style_context="Modern style",
                )

            self.assertIsInstance(result, CodeArticulatedGeometry)
            mock_convert.assert_called_once()
            self.assertIs(mock_convert.call_args.kwargs["blender_server"], blender_server)
        finally:
            shutil.rmtree(temp_dir)

    def test_code_articulated_strategy_validates_packaged_sdf_mesh(self) -> None:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "side_view_elevation_degrees": 20.0,
                    "validation_taa_samples": 8,
                }
            }
        )
        router = AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=MagicMock(),
            cfg=cfg,
        )
        item = AssetItem(
            description="storage cabinet",
            short_name="cabinet",
            dimensions=[1.0, 0.5, 1.4],
            object_type=ObjectType.FURNITURE,
            strategies=["code_articulated"],
        )
        temp_dir = Path(tempfile.mkdtemp())
        try:
            validation_meshes = []
            articulated_results = []
            for idx in range(2):
                validation_mesh = temp_dir / f"sdf_{idx}" / "mesh" / f"cabinet_{idx}.glb"
                validation_mesh.parent.mkdir(parents=True, exist_ok=True)
                validation_mesh.write_bytes(b"glb")
                validation_meshes.append(validation_mesh)
                articulated_results.append(
                    CodeArticulatedGeometry(
                        urdf_path=temp_dir / f"cabinet_{idx}.urdf",
                        item=item,
                        geometry_path=temp_dir / f"rigid_{idx}.glb",
                        sdf_path=temp_dir / f"sdf_{idx}" / f"cabinet_{idx}.sdf",
                        analysis_path=temp_dir / f"sdf_{idx}" / "analysis.json",
                        validation_mesh_path=validation_mesh,
                    )
                )

            with (
                patch.object(
                    router,
                    "_generate_code_articulated_geometry",
                    side_effect=articulated_results,
                ) as mock_generate,
                patch.object(
                    router,
                    "validate_asset",
                    side_effect=[
                        ValidationResult(
                            is_acceptable=False,
                            reason="bad articulation",
                            suggestions=["retry"],
                        ),
                        ValidationResult(
                            is_acceptable=True,
                            reason="good articulation",
                        ),
                    ],
                ) as mock_validate,
            ):
                result = router._try_code_articulated_strategy(
                    item=item,
                    max_retries=3,
                    code_object_runner=MagicMock(),
                    image_generator=MagicMock(),
                    images_dir=temp_dir / "images",
                    code_object_dir=temp_dir / "code_object",
                    debug_dir=temp_dir / "debug",
                    style_context="Modern style",
                )

            self.assertIs(result, articulated_results[1])
            self.assertEqual(mock_generate.call_count, 2)
            self.assertEqual(mock_validate.call_count, 2)
            self.assertEqual(
                [call.kwargs["mesh_path"] for call in mock_validate.call_args_list],
                validation_meshes,
            )
            self.assertEqual(
                [call.kwargs["output_dir"] for call in mock_validate.call_args_list],
                [temp_dir / "debug" / "cabinet_code_articulated_validation"] * 2,
            )
        finally:
            shutil.rmtree(temp_dir)

    def test_code_articulated_missing_validation_mesh_retries_until_final_bypass(self) -> None:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "side_view_elevation_degrees": 20.0,
                    "validation_taa_samples": 8,
                }
            }
        )
        router = AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=MagicMock(),
            cfg=cfg,
        )
        item = AssetItem(
            description="storage cabinet",
            short_name="cabinet",
            dimensions=[1.0, 0.5, 1.4],
            object_type=ObjectType.FURNITURE,
            strategies=["code_articulated"],
        )
        temp_dir = Path(tempfile.mkdtemp())
        try:
            missing_mesh = CodeArticulatedGeometry(
                urdf_path=temp_dir / "cabinet_0.urdf",
                item=item,
                geometry_path=temp_dir / "rigid_0.glb",
            )
            final_pending = CodeArticulatedGeometry(
                urdf_path=temp_dir / "cabinet_1.urdf",
                item=item,
                geometry_path=temp_dir / "rigid_1.glb",
            )

            with (
                patch.object(
                    router,
                    "_generate_code_articulated_geometry",
                    side_effect=[missing_mesh, final_pending],
                ) as mock_generate,
                patch.object(router, "validate_asset") as mock_validate,
            ):
                result = router._try_code_articulated_strategy(
                    item=item,
                    max_retries=2,
                    code_object_runner=MagicMock(),
                    image_generator=MagicMock(),
                    images_dir=temp_dir / "images",
                    code_object_dir=temp_dir / "code_object",
                    debug_dir=temp_dir / "debug",
                    style_context="Modern style",
                )

            self.assertIs(result, final_pending)
            self.assertEqual(mock_generate.call_count, 2)
            mock_validate.assert_not_called()
        finally:
            shutil.rmtree(temp_dir)

    def test_code_articulated_strategy_no_movable_falls_back_to_rigid_validation(self) -> None:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "side_view_elevation_degrees": 20.0,
                    "validation_taa_samples": 8,
                }
            }
        )
        router = AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=MagicMock(),
            cfg=cfg,
        )
        item = AssetItem(
            description="plain cabinet",
            short_name="cabinet",
            dimensions=[1.0, 0.5, 1.0],
            object_type=ObjectType.FURNITURE,
            strategies=["code_articulated"],
        )
        temp_dir = Path(tempfile.mkdtemp())
        try:
            rigid = GeneratedGeometry(
                geometry_path=temp_dir / "cabinet.glb",
                item=item,
                asset_source="code_generated",
            )
            with (
                patch.object(
                    router,
                    "_generate_code_articulated_geometry",
                    return_value=rigid,
                ) as mock_generate,
                patch.object(
                    router,
                    "validate_asset",
                    return_value=ValidationResult(
                        is_acceptable=True,
                        reason="looks good",
                    ),
                ) as mock_validate,
            ):
                result = router._try_code_articulated_strategy(
                    item=item,
                    max_retries=2,
                    code_object_runner=MagicMock(),
                    image_generator=MagicMock(),
                    images_dir=temp_dir / "images",
                    code_object_dir=temp_dir / "code_object",
                    debug_dir=temp_dir / "debug",
                    style_context="Modern style",
                )

            self.assertIs(result, rigid)
            mock_generate.assert_called_once()
            mock_validate.assert_called_once()
        finally:
            shutil.rmtree(temp_dir)

    def test_wall_art_profile_preprocesses_crop_and_uses_config_override(self) -> None:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "side_view_elevation_degrees": 20.0,
                    "validation_taa_samples": 8,
                    "code_object": {
                        "wall_art_config_path": "/tmp/config_WallArt.yaml",
                        "wall_art_crop": {
                            "sam3_checkpoint": "/tmp/sam3.pt",
                            "frame_prompts": ["framed artwork", "picture frame"],
                            "threshold": 0.5,
                            "frame_inset_ratio": 0.08,
                        },
                    },
                    "sam3d": {
                        "sam3_checkpoint": "/tmp/default_sam3.pt",
                    },
                }
            }
        )
        router = AssetRouter(
            agent_type=AgentType.WALL_MOUNTED,
            vlm_service=MagicMock(),
            cfg=cfg,
        )
        item = AssetItem(
            description="framed landscape painting",
            short_name="painting",
            dimensions=[0.8, 0.6, 0.05],
            object_type=ObjectType.WALL_MOUNTED,
            strategies=["code_generated"],
            code_object_profile="wall_art",
        )
        temp_dir = Path(tempfile.mkdtemp())
        try:
            code_object_output_dir = temp_dir / "candidate"
            result_bundle = CodeObjectGenerationResult(
                output_dir=code_object_output_dir,
                mesh_path=code_object_output_dir / "mesh" / "painting.glb",
                object_plan_path=None,
                code_dir=None,
                pipeline_result_path=None,
                full_object_render_path=None,
            )
            runner = MagicMock()
            runner.generate_from_image.return_value = result_bundle
            image_generator = MagicMock()

            with patch(
                "scenecode.agent_utils.wall_art_preprocess.prepare_wall_art_reference_assets"
            ) as mock_preprocess:
                result = router._generate_code_object_geometry(
                    item=item,
                    code_object_runner=runner,
                    image_generator=image_generator,
                    images_dir=temp_dir / "images",
                    code_object_dir=temp_dir / "code_object",
                    style_context="Gallery wall",
                )

            self.assertIsNotNone(result)
            call_kwargs = runner.generate_from_image.call_args.kwargs
            self.assertEqual(call_kwargs["image_path"], result.image_path)
            self.assertEqual(
                call_kwargs["config_path_override"],
                "/tmp/config_WallArt.yaml",
            )
            mock_preprocess.assert_called_once()
            self.assertEqual(
                mock_preprocess.call_args.kwargs["image_path"],
                result.image_path,
            )
            self.assertEqual(
                mock_preprocess.call_args.kwargs["output_dir"],
                call_kwargs["output_dir"],
            )
        finally:
            shutil.rmtree(temp_dir)


    def test_simple_object_profile_uses_simplemanip_config_without_wall_art_preprocess(
        self,
    ) -> None:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "side_view_elevation_degrees": 20.0,
                    "validation_taa_samples": 8,
                    "code_object": {
                        "simple_object_config_path": "/tmp/config_simplemanip.yaml",
                    },
                }
            }
        )
        router = AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=MagicMock(),
            cfg=cfg,
        )
        item = AssetItem(
            description="simple decorative bowl",
            short_name="bowl",
            dimensions=[0.3, 0.3, 0.2],
            object_type=ObjectType.FURNITURE,
            strategies=["code_generated"],
            code_object_profile="SimpleObject",
        )
        temp_dir = Path(tempfile.mkdtemp())
        try:
            code_object_output_dir = temp_dir / "candidate"
            result_bundle = CodeObjectGenerationResult(
                output_dir=code_object_output_dir,
                mesh_path=code_object_output_dir / "mesh" / "bowl.glb",
                object_plan_path=None,
                code_dir=None,
                pipeline_result_path=None,
                full_object_render_path=None,
            )
            runner = MagicMock()
            runner.generate_from_image.return_value = result_bundle
            image_generator = MagicMock()

            with patch(
                "scenecode.agent_utils.wall_art_preprocess.prepare_wall_art_reference_assets"
            ) as mock_preprocess:
                result = router._generate_code_object_geometry(
                    item=item,
                    code_object_runner=runner,
                    image_generator=image_generator,
                    images_dir=temp_dir / "images",
                    code_object_dir=temp_dir / "code_object",
                    style_context="Minimalist style",
                )

            self.assertIsNotNone(result)
            call_kwargs = runner.generate_from_image.call_args.kwargs
            self.assertEqual(call_kwargs["image_path"], result.image_path)
            self.assertEqual(
                call_kwargs["config_path_override"],
                "/tmp/config_simplemanip.yaml",
            )
            mock_preprocess.assert_not_called()
        finally:
            shutil.rmtree(temp_dir)

    def test_manipuland_profile_uses_structmanip_config_without_wall_art_preprocess(
        self,
    ) -> None:
        cfg = OmegaConf.create(
            {
                "asset_manager": {
                    "side_view_elevation_degrees": 20.0,
                    "validation_taa_samples": 8,
                    "code_object": {
                        "manipuland_config_path": "/tmp/config_structmanip.yaml",
                    },
                }
            }
        )
        router = AssetRouter(
            agent_type=AgentType.MANIPULAND,
            vlm_service=MagicMock(),
            cfg=cfg,
        )
        item = AssetItem(
            description="small handled mug",
            short_name="mug",
            dimensions=[0.12, 0.09, 0.11],
            object_type=ObjectType.MANIPULAND,
            strategies=["code_generated"],
            code_object_profile="manipuland",
        )
        temp_dir = Path(tempfile.mkdtemp())
        try:
            code_object_output_dir = temp_dir / "candidate"
            result_bundle = CodeObjectGenerationResult(
                output_dir=code_object_output_dir,
                mesh_path=code_object_output_dir / "mesh" / "mug.glb",
                object_plan_path=None,
                code_dir=None,
                pipeline_result_path=None,
                full_object_render_path=None,
            )
            runner = MagicMock()
            runner.generate_from_image.return_value = result_bundle
            image_generator = MagicMock()

            with patch(
                "scenecode.agent_utils.wall_art_preprocess.prepare_wall_art_reference_assets"
            ) as mock_preprocess:
                result = router._generate_code_object_geometry(
                    item=item,
                    code_object_runner=runner,
                    image_generator=image_generator,
                    images_dir=temp_dir / "images",
                    code_object_dir=temp_dir / "code_object",
                    style_context="Functional style",
                )

            self.assertIsNotNone(result)
            call_kwargs = runner.generate_from_image.call_args.kwargs
            self.assertEqual(call_kwargs["image_path"], result.image_path)
            self.assertEqual(
                call_kwargs["config_path_override"],
                "/tmp/config_structmanip.yaml",
            )
            mock_preprocess.assert_not_called()
        finally:
            shutil.rmtree(temp_dir)


class TestWallArtPreprocess(unittest.TestCase):
    def test_prepare_wall_art_reference_assets_falls_back_to_original_image(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        try:
            image_path = temp_dir / "reference.png"
            Image.new("RGB", (8, 6), color=(12, 34, 56)).save(image_path)

            result = prepare_wall_art_reference_assets(
                image_path=image_path,
                output_dir=temp_dir / "candidate",
                sam3_checkpoint=temp_dir / "missing_sam3.pt",
            )

            self.assertTrue(result.used_fallback)
            self.assertTrue(result.picture_crop_path.exists())
            self.assertEqual(Image.open(result.picture_crop_path).size, (8, 6))
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["used_fallback"])
            self.assertEqual(
                summary["output_files"]["picture_crop"],
                str(result.picture_crop_path),
            )
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
