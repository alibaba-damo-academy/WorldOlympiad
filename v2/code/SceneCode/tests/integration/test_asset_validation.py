import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

from pathlib import Path
from unittest.mock import patch

from openai import OpenAI
from omegaconf import OmegaConf

from scenecode.agent_utils.asset_router import AssetRouter
from scenecode.agent_utils.blender import BlenderServer
from scenecode.agent_utils.room import AgentType
from scenecode.agent_utils.vlm_service import VLMService
from scenecode.prompts import AssetRouterPrompts, prompt_manager
from tests.integration.common import has_openai_key

ENV_MESH_PATH = "SCENECODE_VALIDATION_MESH_PATH"
ENV_DESCRIPTION = "SCENECODE_VALIDATION_DESCRIPTION"
ENV_EXPECTED_ACCEPTABLE = "SCENECODE_VALIDATION_EXPECTED_ACCEPTABLE"
ENV_OUTPUT_DIR = "SCENECODE_VALIDATION_OUTPUT_DIR"
ENV_MODEL = "SCENECODE_VALIDATION_MODEL"
ENV_USE_LENIENT = "SCENECODE_VALIDATION_USE_LENIENT"


def _default_mesh_path() -> Path:
    return (
        Path(__file__).parent.parent
        / "test_data/realistic_scene/generated_assets/sdf/office_chair_1761578426/office_chair.gltf"
    )


def _resolve_path(path_str: str | None, default: Path) -> Path:
    raw_path = path_str or str(default)
    return Path(raw_path).expanduser().resolve()


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "pass"}:
        return True
    if normalized in {"0", "false", "no", "n", "fail"}:
        return False

    raise ValueError(
        f"Unsupported boolean value '{value}'. "
        "Use one of: true/false/1/0/yes/no."
    )


def _build_test_config(model_override: str | None = None):
    global_config_path = (
        Path(__file__).parent.parent.parent / "configs/config.yaml"
    )
    config_path = (
        Path(__file__).parent.parent.parent
        / "configs/furniture_agent/base_furniture_agent.yaml"
    )
    global_config = OmegaConf.load(global_config_path)
    base_config = OmegaConf.load(config_path)
    overrides = OmegaConf.create(
        {
            "openai": {
                "api_base": os.getenv("OPENAI_API_BASE")
                or global_config.openai.api_base,
                "service_tier": global_config.openai.service_tier,
                "model": model_override or base_config.openai.model,
            }
        }
    )
    return OmegaConf.merge(base_config, overrides)


def _create_vlm_service(cfg):
    service_tier = getattr(cfg.openai, "service_tier", None)
    api_base = getattr(cfg.openai, "api_base", None)

    try:
        return VLMService(service_tier=service_tier, api_base=api_base)
    except TypeError as exc:
        if "api_base" not in str(exc):
            raise
        service = VLMService(service_tier=service_tier)
        if api_base:
            service.client = OpenAI(base_url=api_base)
        return service


class TestAssetValidationIntegration(unittest.TestCase):
    """Functional test for AssetRouter.validate_asset with custom inputs."""

    @classmethod
    def setUpClass(cls):
        if not has_openai_key():
            raise unittest.SkipTest("Requires OPENAI_API_KEY")

        cls.mesh_path = _resolve_path(
            os.getenv(ENV_MESH_PATH), default=_default_mesh_path()
        )
        cls.description = os.getenv(ENV_DESCRIPTION, "office chair")
        cls.expected_acceptable = _parse_optional_bool(
            os.getenv(ENV_EXPECTED_ACCEPTABLE)
        )
        cls.use_lenient = _parse_optional_bool(os.getenv(ENV_USE_LENIENT)) or False
        cls.output_dir_override = os.getenv(ENV_OUTPUT_DIR)
        cls.model_override = os.getenv(ENV_MODEL)

        if not cls.mesh_path.exists():
            raise unittest.SkipTest(f"Mesh file not found: {cls.mesh_path}")

        cls.cfg = _build_test_config(model_override=cls.model_override)
        cls.vlm_service = _create_vlm_service(cls.cfg)

        cls.output_root = (
            Path(cls.output_dir_override).expanduser().resolve()
            if cls.output_dir_override
            else Path(tempfile.mkdtemp(prefix="asset_validation_test_"))
        )
        cls.output_root.mkdir(parents=True, exist_ok=True)
        cls.cleanup_output_root = cls.output_dir_override is None

        cls.server = BlenderServer(
            port_range=(8060, 8090),
            server_startup_delay=0.1,
            port_cleanup_delay=0.1,
        )
        try:
            cls.server.start()
            cls.server.wait_until_ready()
        except Exception as exc:
            if cls.server.is_running():
                cls.server.stop()
                time.sleep(1)
            raise unittest.SkipTest(f"BlenderServer unavailable: {exc}")

        cls.router = AssetRouter(
            agent_type=AgentType.FURNITURE,
            vlm_service=cls.vlm_service,
            cfg=cls.cfg,
            blender_server=cls.server,
        )

    @classmethod
    def tearDownClass(cls):
        server = getattr(cls, "server", None)
        if server is not None and server.is_running():
            server.stop()
            time.sleep(1)

        if getattr(cls, "cleanup_output_root", False):
            shutil.rmtree(getattr(cls, "output_root", Path(".")), ignore_errors=True)

    def test_validate_asset_with_custom_inputs(self):
        validation_dir = self.output_root / "validation_outputs"
        expected_prompts_dir = (
            Path(__file__).resolve().parents[2] / "scenecode/prompts/data"
        )
        self.assertTrue(
            expected_prompts_dir.exists(),
            f"Expected prompts directory not found: {expected_prompts_dir}",
        )
        original_prompts_dir = prompt_manager.prompts_dir
        prompt_manager.prompts_dir = expected_prompts_dir
        original_get_prompt = prompt_manager.get_prompt

        def _get_prompt_with_debug_print(*args, **kwargs):
            rendered_prompt = original_get_prompt(*args, **kwargs)
            prompt_name = kwargs.get("prompt_name")
            if prompt_name in {
                AssetRouterPrompts.ASSET_VALIDATION,
                AssetRouterPrompts.ASSET_VALIDATION_LENIENT,
            }:
                prompt_path = (
                    prompt_manager.prompts_dir / f"{prompt_name.value}.yaml"
                ).resolve()
                print(
                    "\n=== Loaded asset validation prompt (from test hook) ===\n"
                    f"prompt_name: {prompt_name.value}\n"
                    f"prompt_path: {prompt_path}\n"
                    f"{rendered_prompt}\n"
                    "=== End asset validation prompt ===\n",
                    flush=True,
                )
            return rendered_prompt

        try:
            with patch(
                "scenecode.agent_utils.asset_router.router.prompt_manager.get_prompt",
                side_effect=_get_prompt_with_debug_print,
            ):
                result = self.router.validate_asset(
                    mesh_path=self.mesh_path,
                    description=self.description,
                    output_dir=validation_dir,
                    use_lenient=self.use_lenient,
                )
        finally:
            prompt_manager.prompts_dir = original_prompts_dir

        self.assertIsInstance(result.is_acceptable, bool)
        self.assertTrue(result.reason.strip())
        self.assertIsInstance(result.suggestions, list)
        self.assertTrue(validation_dir.exists())

        rendered_images = sorted(validation_dir.glob("*.png"))
        self.assertGreaterEqual(
            len(rendered_images),
            4,
            f"Expected validation renders in {validation_dir}",
        )

        summary = {
            "mesh_path": str(self.mesh_path),
            "description": self.description,
            "use_lenient": self.use_lenient,
            "model": self.cfg.openai.model,
            "is_acceptable": result.is_acceptable,
            "reason": result.reason,
            "suggestions": result.suggestions,
            "rendered_images": [str(path) for path in rendered_images],
        }
        (self.output_root / "validation_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

        if self.expected_acceptable is not None:
            self.assertEqual(
                result.is_acceptable,
                self.expected_acceptable,
                json.dumps(summary, indent=2),
            )


def _apply_cli_overrides(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(
        description="Run functional asset validation test with custom mesh input."
    )
    parser.add_argument("--mesh-path", help=f"Mesh path override for {ENV_MESH_PATH}")
    parser.add_argument(
        "--description",
        help=f"Description override for {ENV_DESCRIPTION}",
    )
    parser.add_argument(
        "--expected-acceptable",
        choices=["true", "false"],
        help=(
            "Optional assertion for validation result. "
            "If omitted, the test only checks the interface contract."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=f"Persist debug output to this directory via {ENV_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model",
        help=f"Optional model override via {ENV_MODEL}",
    )
    parser.add_argument(
        "--use-lenient",
        action="store_true",
        help=f"Use lenient validation prompt via {ENV_USE_LENIENT}=true",
    )

    args, remaining = parser.parse_known_args(argv[1:])

    if args.mesh_path:
        os.environ[ENV_MESH_PATH] = args.mesh_path
    if args.description:
        os.environ[ENV_DESCRIPTION] = args.description
    if args.expected_acceptable:
        os.environ[ENV_EXPECTED_ACCEPTABLE] = args.expected_acceptable
    if args.output_dir:
        os.environ[ENV_OUTPUT_DIR] = args.output_dir
    if args.model:
        os.environ[ENV_MODEL] = args.model
    if args.use_lenient:
        os.environ[ENV_USE_LENIENT] = "true"

    return [argv[0], *remaining]


if __name__ == "__main__":
    sys.argv = _apply_cli_overrides(sys.argv)
    unittest.main()
