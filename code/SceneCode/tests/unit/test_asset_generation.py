import os
import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from scenecode.agent_utils.geometry_generation_server.geometry_generation import (
    generate_geometry_from_image,
)
from scenecode.agent_utils.image_generation import (
    FallbackImageGenerator,
    FluxKleinImageGenerator,
    OpenAIImageGenerator,
    create_image_generator,
    _extract_and_save_openai_image,
)
from scenecode.prompts.registry import ImageGenerationPrompts

# Mock hy3dgen modules for CI where Hunyuan3D-2 is not installed.
# This allows the @patch decorators to work even when the real modules don't exist.
try:
    import hy3dgen.rembg  # noqa: F401
    import hy3dgen.shapegen  # noqa: F401
    import hy3dgen.texgen  # noqa: F401
except ImportError:
    # Only mock if real modules don't exist (CI environment).
    import sys

    # Create structured mocks with proper hierarchy.
    hy3dgen = MagicMock()
    hy3dgen.rembg = MagicMock()
    hy3dgen.shapegen = MagicMock()
    hy3dgen.shapegen.pipelines = MagicMock()
    hy3dgen.texgen = MagicMock()

    # Create the classes on the submodules.
    hy3dgen.rembg.BackgroundRemover = MagicMock()
    hy3dgen.shapegen.Hunyuan3DDiTFlowMatchingPipeline = MagicMock()
    hy3dgen.shapegen.FaceReducer = MagicMock()
    hy3dgen.shapegen.pipelines.export_to_trimesh = MagicMock()
    hy3dgen.texgen.Hunyuan3DPaintPipeline = MagicMock()

    # Add to sys.modules with proper structure.
    sys.modules["hy3dgen"] = hy3dgen
    sys.modules["hy3dgen.rembg"] = hy3dgen.rembg
    sys.modules["hy3dgen.shapegen"] = hy3dgen.shapegen
    sys.modules["hy3dgen.shapegen.pipelines"] = hy3dgen.shapegen.pipelines
    sys.modules["hy3dgen.texgen"] = hy3dgen.texgen


class TestOpenAIImageGenerator(unittest.TestCase):
    """Test the OpenAIImageGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Mock OpenAI client.
        self.mock_client = MagicMock()
        self.generator = OpenAIImageGenerator(client=self.mock_client)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.client, self.mock_client)
        self.assertEqual(self.generator.image_quality, "low")

    def test_initialization_with_quality(self):
        """Test generator initialization with custom quality."""
        generator = OpenAIImageGenerator(client=self.mock_client, quality="high")
        self.assertEqual(generator.image_quality, "high")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("scenecode.utils.openai.OpenAI")
    def test_initialization_with_api_base(self, mock_openai_class):
        """Generator uses configured api_base when creating its own client."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        generator = OpenAIImageGenerator(
            quality="high",
            api_base="https://api.openai.com/v1",
        )

        self.assertEqual(generator.client, mock_client)
        mock_openai_class.assert_called_once_with(
            base_url="https://api.openai.com/v1"
        )

    @patch("scenecode.agent_utils.image_generation.OpenAIImageGenerator")
    def test_factory_passes_api_base_to_openai_generator(self, mock_generator_class):
        """Factory passes api_base through to OpenAI image generator."""
        config = MagicMock()
        config.openai.quality = "medium"

        create_image_generator(
            backend="openai",
            config=config,
            api_base="https://api.openai.com/v1",
        )

        mock_generator_class.assert_called_once_with(
            quality="medium",
            api_base="https://api.openai.com/v1",
        )

    @patch("scenecode.agent_utils.image_generation.FluxKleinImageGenerator")
    @patch("scenecode.agent_utils.image_generation.OpenAIImageGenerator")
    def test_factory_wraps_openai_with_flux_fallback_when_configured(
        self, mock_openai_generator_class, mock_flux_generator_class
    ):
        """Factory returns a fallback wrapper for OpenAI -> FLUX."""
        config = MagicMock()
        config.openai.quality = "medium"
        config.fallback_backend = "flux-klein"
        config.flux_klein = MagicMock()

        primary_generator = MagicMock()
        fallback_generator = MagicMock()
        mock_openai_generator_class.return_value = primary_generator
        mock_flux_generator_class.return_value = fallback_generator

        generator = create_image_generator(
            backend="openai",
            config=config,
            api_base="https://api.openai.com/v1",
        )

        self.assertIsInstance(generator, FallbackImageGenerator)
        self.assertIs(generator.primary_generator, primary_generator)
        self.assertIs(generator.fallback_generator, fallback_generator)

    def test_fallback_generator_used_when_primary_generation_fails(self):
        """Runtime generation failures fall back from OpenAI to FLUX."""
        primary_generator = MagicMock()
        fallback_generator = MagicMock()
        primary_generator.generate_images.side_effect = RuntimeError("OpenAI failure")

        generator = FallbackImageGenerator(
            primary_backend="openai",
            primary_generator=primary_generator,
            fallback_backend="flux-klein",
            fallback_generator=fallback_generator,
        )

        generator.generate_images(
            style_prompt="Modern minimalist",
            object_descriptions=["A modern chair"],
            output_paths=[self.temp_path / "chair.png"],
        )

        primary_generator.generate_images.assert_called_once()
        fallback_generator.generate_images.assert_called_once()

    @patch("scenecode.agent_utils.image_generation.FluxKleinImageGenerator")
    @patch("scenecode.agent_utils.image_generation.OpenAIImageGenerator")
    def test_factory_uses_flux_when_openai_initialization_fails(
        self, mock_openai_generator_class, mock_flux_generator_class
    ):
        """Initialization failures also fall back to FLUX."""
        config = MagicMock()
        config.openai.quality = "medium"
        config.fallback_backend = "flux-klein"
        config.flux_klein = MagicMock()

        fallback_generator = MagicMock()
        mock_openai_generator_class.side_effect = ValueError("missing OPENAI_API_KEY")
        mock_flux_generator_class.return_value = fallback_generator

        generator = create_image_generator(backend="openai", config=config)

        self.assertIs(generator, fallback_generator)

    @patch("scenecode.agent_utils.image_generation._extract_and_save_openai_image")
    def test_generate_single_image(self, mock_extract):
        """Test single image generation."""
        # Mock response.
        mock_response = MagicMock()
        mock_response.data = [MagicMock(b64_json="base64data")]
        self.mock_client.images.generate.return_value = mock_response

        # Generate image.
        output_path = self.temp_path / "test_image.png"
        self.generator.generate_images(
            style_prompt="Modern minimalist",
            object_descriptions=["A modern chair"],
            output_paths=[output_path],
        )

        # Verify OpenAI API was called correctly.
        self.mock_client.images.generate.assert_called_once()
        call_kwargs = self.mock_client.images.generate.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-image-1.5")
        self.assertEqual(call_kwargs["size"], "1024x1024")
        self.assertEqual(call_kwargs["quality"], "low")
        self.assertEqual(call_kwargs["background"], "opaque")

        # Verify image extraction happened.
        mock_extract.assert_called_once()

    @patch("scenecode.agent_utils.image_generation._extract_and_save_openai_image")
    def test_generate_single_image_uses_rendered_prompt_contract(self, mock_extract):
        """Test the exact OpenAI Images API contract for a single GPT image call."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(b64_json="base64data")]
        self.mock_client.images.generate.return_value = mock_response

        self.generator.prompt_manager = MagicMock()
        self.generator.prompt_manager.get_prompt.return_value = "rendered prompt"

        output_path = self.temp_path / "chair.png"
        self.generator.generate_images(
            style_prompt="Warm Scandinavian",
            object_descriptions=["Oak dining chair"],
            output_paths=[output_path],
        )

        self.generator.prompt_manager.get_prompt.assert_called_once()
        prompt_call_args = self.generator.prompt_manager.get_prompt.call_args
        self.assertEqual(
            prompt_call_args.args[0], ImageGenerationPrompts.ASSET_IMAGE_INITIAL
        )
        self.assertEqual(prompt_call_args.kwargs["description"], "Oak dining chair")
        self.assertEqual(prompt_call_args.kwargs["style_prompt"], "Warm Scandinavian")

        self.mock_client.images.generate.assert_called_once_with(
            model="gpt-image-1.5",
            prompt="rendered prompt",
            size="1024x1024",
            n=1,
            output_format="png",
            quality="low",
            background="opaque",
            moderation="low",
        )
        mock_extract.assert_called_once_with(
            response=mock_response,
            output_path=output_path,
            description="Oak dining chair",
        )

    @patch("scenecode.agent_utils.image_generation._extract_and_save_openai_image")
    def test_generate_single_image_passes_explicit_size_override(self, mock_extract):
        """Test that explicit size overrides are forwarded to gpt-image-1.5."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(b64_json="base64data")]
        self.mock_client.images.generate.return_value = mock_response

        output_path = self.temp_path / "lamp.png"
        self.generator.generate_images(
            style_prompt="Minimalist",
            object_descriptions=["Desk lamp"],
            output_paths=[output_path],
            size="1024x1792",
        )

        call_kwargs = self.mock_client.images.generate.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gpt-image-1.5")
        self.assertEqual(call_kwargs["size"], "1024x1792")
        self.assertEqual(call_kwargs["output_format"], "png")
        self.assertEqual(call_kwargs["moderation"], "low")
        mock_extract.assert_called_once()

    @patch("scenecode.agent_utils.image_generation._extract_and_save_openai_image")
    def test_generate_multiple_images(self, mock_extract):
        """Test generating multiple images in parallel."""
        # Mock responses.
        mock_response1 = MagicMock()
        mock_response1.data = [MagicMock(b64_json="base64data1")]
        mock_response2 = MagicMock()
        mock_response2.data = [MagicMock(b64_json="base64data2")]
        self.mock_client.images.generate.side_effect = [mock_response1, mock_response2]

        # Generate images.
        descriptions = ["A modern chair", "A modern table"]
        output_paths = [self.temp_path / "chair.png", self.temp_path / "table.png"]

        self.generator.generate_images(
            style_prompt="Modern minimalist",
            object_descriptions=descriptions,
            output_paths=output_paths,
        )

        # Verify API was called correct number of times.
        self.assertEqual(self.mock_client.images.generate.call_count, 2)
        self.assertEqual(mock_extract.call_count, 2)

    def test_generate_images_mismatched_lengths(self):
        """Test error when descriptions and output paths have different lengths."""
        with self.assertRaises(ValueError) as context:
            self.generator.generate_images(
                style_prompt="Modern",
                object_descriptions=["chair", "table"],
                output_paths=[self.temp_path / "chair.png"],  # Only one path.
            )
        self.assertIn("Number of descriptions must match", str(context.exception))

    def test_generate_images_failure_propagates(self):
        """Test that exceptions from image generation propagate (fail-fast)."""
        # Mock API to raise an error.
        self.mock_client.images.generate.side_effect = RuntimeError(
            "Image generation API error"
        )

        descriptions = ["Item 1", "Item 2"]
        output_paths = [self.temp_path / "item1.png", self.temp_path / "item2.png"]

        # Verify that the original exception propagates.
        with self.assertRaises(RuntimeError) as context:
            self.generator.generate_images(
                style_prompt="Test style",
                object_descriptions=descriptions,
                output_paths=output_paths,
            )

        self.assertIn("Image generation API error", str(context.exception))


class TestFluxKleinImageGenerator(unittest.TestCase):
    """Test the FluxKleinImageGenerator class."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.python_executable = self.temp_path / "python"
        self.model_path = self.temp_path / "flux-model"
        self.worker_script = self.temp_path / "flux_klein_worker.py"
        self.reference_image_path = self.temp_path / "reference.png"

        self.python_executable.write_text("")
        self.model_path.write_text("")
        self.worker_script.write_text("")
        self.reference_image_path.write_text("")

        self.config = OmegaConf.create(
            {
                "python_executable": str(self.python_executable),
                "model_path": str(self.model_path),
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
                "max_sequence_length": 256,
                "seed": 41,
            }
        )
        self.generator = FluxKleinImageGenerator(config=self.config)
        self.generator.worker_script = self.worker_script

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_factory_creates_flux_klein_generator(self):
        config = OmegaConf.create(
            {
                "flux_klein": {
                    "python_executable": str(self.python_executable),
                    "model_path": str(self.model_path),
                    "width": 1024,
                    "height": 1024,
                    "num_inference_steps": 4,
                    "guidance_scale": 1.0,
                    "max_sequence_length": 256,
                    "seed": 41,
                }
            }
        )

        generator = create_image_generator(backend="flux-klein", config=config)

        self.assertIsInstance(generator, FluxKleinImageGenerator)

    @patch("scenecode.agent_utils.image_generation.subprocess.run")
    def test_generate_single_image_runs_flux_worker_with_rendered_prompt(
        self, mock_run
    ):
        output_path = self.temp_path / "chair.png"
        self.generator.prompt_manager = MagicMock()
        self.generator.prompt_manager.get_prompt.return_value = "rendered prompt"

        def run_side_effect(cmd, capture_output, text, check):
            output_path.write_bytes(b"png")
            return MagicMock(returncode=0, stdout="ok", stderr="")

        mock_run.side_effect = run_side_effect

        self.generator.generate_images(
            style_prompt="Warm Scandinavian",
            object_descriptions=["Oak dining chair"],
            output_paths=[output_path],
            size="640x480",
        )

        self.generator.prompt_manager.get_prompt.assert_called_once_with(
            ImageGenerationPrompts.ASSET_IMAGE_INITIAL,
            description="Oak dining chair",
            style_prompt="Warm Scandinavian",
        )
        cmd = mock_run.call_args.args[0]
        self.assertEqual(cmd[0], str(self.python_executable))
        self.assertEqual(cmd[1], str(self.worker_script))
        self.assertIn("--mode", cmd)
        self.assertIn("generate", cmd)
        self.assertIn("rendered prompt", cmd)
        self.assertIn(str(output_path), cmd)
        self.assertIn("640", cmd)
        self.assertIn("480", cmd)

    @patch("scenecode.agent_utils.image_generation.subprocess.run")
    def test_generate_furniture_context_image_runs_edit_command(self, mock_run):
        output_path = self.temp_path / "edited.png"
        self.generator.prompt_manager = MagicMock()
        self.generator.prompt_manager.get_prompt.return_value = "edit prompt"

        def run_side_effect(cmd, capture_output, text, check):
            output_path.write_bytes(b"png")
            return MagicMock(returncode=0, stdout="ok", stderr="")

        mock_run.side_effect = run_side_effect

        result = self.generator.generate_furniture_context_image(
            reference_image_path=self.reference_image_path,
            scene_description="Bedroom",
            width_m=3.0,
            length_m=4.0,
            output_path=output_path,
        )

        self.assertEqual(result, output_path)
        cmd = mock_run.call_args.args[0]
        self.assertIn("edit", cmd)
        self.assertIn("--reference-image-path", cmd)
        self.assertIn(str(self.reference_image_path), cmd)
        self.assertIn("1024", cmd)

    @patch("scenecode.agent_utils.image_generation.subprocess.run")
    def test_generate_images_rejects_invalid_size_override(self, mock_run):
        with self.assertRaises(ValueError) as context:
            self.generator.generate_images(
                style_prompt="Style",
                object_descriptions=["Chair"],
                output_paths=[self.temp_path / "chair.png"],
                size="bad-size",
            )

        self.assertIn("WIDTHxHEIGHT", str(context.exception))
        mock_run.assert_not_called()

    @patch("scenecode.agent_utils.image_generation.subprocess.run")
    def test_generate_images_fails_fast_on_subprocess_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="out", stderr="err")

        with self.assertRaises(RuntimeError) as context:
            self.generator.generate_images(
                style_prompt="Style",
                object_descriptions=["Chair"],
                output_paths=[self.temp_path / "chair.png"],
            )

        self.assertIn("exit code 1", str(context.exception))
        self.assertIn("stderr", str(context.exception))

    @patch("scenecode.agent_utils.image_generation.subprocess.run")
    def test_generate_images_fails_when_output_is_missing(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        with self.assertRaises(FileNotFoundError) as context:
            self.generator.generate_images(
                style_prompt="Style",
                object_descriptions=["Chair"],
                output_paths=[self.temp_path / "chair.png"],
            )

        self.assertIn("did not create the output image", str(context.exception))


class TestGeometryGeneration(unittest.TestCase):
    """Test the geometry generation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("hy3dgen.rembg.BackgroundRemover")
    @patch(
        "scenecode.agent_utils.geometry_generation_server.geometry_generation.Hunyuan3DPipelineManager"
    )
    @patch("hy3dgen.shapegen.pipelines.export_to_trimesh")
    @patch(
        "scenecode.agent_utils.geometry_generation_server.geometry_generation.Image"
    )
    def test_generate_geometry_from_image_success(
        self,
        mock_image,
        mock_export_to_trimesh,
        mock_pipeline_manager_class,
        mock_bg_remover_class,
    ):
        """Test successful geometry generation from image with pipeline caching."""
        # Mock image loading and processing.
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img

        # Mock pipeline manager and pipelines.
        mock_shape_pipeline = MagicMock()
        mock_pipeline_output = MagicMock()
        mock_shape_pipeline.return_value = mock_pipeline_output

        mock_texture_pipeline = MagicMock()
        mock_face_reducer = MagicMock()
        mock_background_remover = MagicMock()

        mock_mesh = MagicMock()
        mock_face_reducer.return_value = mock_mesh
        mock_texture_pipeline.return_value = mock_mesh
        mock_background_remover.return_value = mock_img
        mock_bg_remover_class.return_value = mock_background_remover

        # Configure pipeline manager to return mocked pipelines.
        mock_pipeline_manager_class.get_pipelines.return_value = (
            mock_shape_pipeline,
            mock_texture_pipeline,
            mock_face_reducer,
            mock_background_remover,
        )

        # Mock export_to_trimesh.
        mock_export_to_trimesh.return_value = [mock_mesh]

        # Test paths.
        image_path = self.temp_path / "test_image.png"
        output_path = self.temp_path / "test_output.glb"
        debug_folder = self.temp_path / "debug"
        debug_folder.mkdir()

        # Call function with pipeline caching enabled.
        generate_geometry_from_image(
            image_path=image_path,
            output_path=output_path,
            debug_folder=debug_folder,
            use_pipeline_caching=True,
        )

        # Verify pipeline manager was used with correct config.
        mock_pipeline_manager_class.get_pipelines.assert_called_once_with(
            use_mini=False
        )

        # Verify image processing.
        mock_image.open.assert_called_once_with(image_path)
        mock_background_remover.assert_called_once_with(mock_img)

        # Verify shape generation pipeline.
        mock_shape_pipeline.assert_called_once()

        # Verify mesh processing pipeline.
        mock_export_to_trimesh.assert_called_once_with(mock_pipeline_output)
        mock_face_reducer.assert_called_once_with(mock_mesh)

        # Verify texture generation.
        mock_texture_pipeline.assert_called_once_with(mock_mesh, image=mock_img)

        # Verify final export.
        mock_mesh.export.assert_called_once_with(output_path)

        # Verify debug image was saved.
        mock_img.save.assert_called_once()

    @patch("hy3dgen.shapegen.FaceReducer")
    @patch("hy3dgen.shapegen.Hunyuan3DDiTFlowMatchingPipeline")
    @patch("hy3dgen.texgen.Hunyuan3DPaintPipeline")
    @patch("hy3dgen.rembg.BackgroundRemover")
    @patch("hy3dgen.shapegen.pipelines.export_to_trimesh")
    @patch(
        "scenecode.agent_utils.geometry_generation_server.geometry_generation.Image"
    )
    def test_generate_geometry_from_image_without_debug(
        self,
        mock_image,
        mock_export_to_trimesh,
        mock_bg_remover_class,
        mock_tex_pipeline_class,
        mock_shape_pipeline_class,
        mock_face_reducer_class,
    ):
        """Test geometry generation without debug folder (no caching)."""
        # Mock image loading and processing.
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img

        # Mock background remover.
        mock_bg_remover = MagicMock()
        mock_bg_remover.return_value = mock_img
        mock_bg_remover_class.return_value = mock_bg_remover

        # Mock shape pipeline.
        mock_shape_pipeline = MagicMock()
        mock_shape_pipeline.enable_flashvdm = MagicMock()
        mock_pipeline_output = MagicMock()
        mock_shape_pipeline.return_value = mock_pipeline_output
        mock_shape_pipeline_class.from_pretrained.return_value = mock_shape_pipeline

        # Mock export_to_trimesh.
        mock_mesh = MagicMock()
        mock_export_to_trimesh.return_value = [mock_mesh]

        # Mock face reducer.
        mock_face_reducer = MagicMock()
        mock_face_reducer.return_value = mock_mesh
        mock_face_reducer_class.return_value = mock_face_reducer

        # Mock texture pipeline.
        mock_tex_pipeline = MagicMock()
        mock_tex_pipeline.return_value = mock_mesh
        mock_tex_pipeline_class.from_pretrained.return_value = mock_tex_pipeline

        # Test paths.
        image_path = self.temp_path / "test_image.png"
        output_path = self.temp_path / "test_output.glb"

        # Call function without debug folder and without caching.
        generate_geometry_from_image(
            image_path=image_path,
            output_path=output_path,
            debug_folder=None,
            use_pipeline_caching=False,
        )

        # Verify debug image was not saved.
        mock_img.save.assert_not_called()

        # Verify pipelines were initialized fresh (not cached).
        mock_shape_pipeline_class.from_pretrained.assert_called_once_with(
            "tencent/Hunyuan3D-2", subfolder="hunyuan3d-dit-v2-0-turbo"
        )
        mock_tex_pipeline_class.from_pretrained.assert_called_once_with(
            "tencent/Hunyuan3D-2"
        )

    @patch("hy3dgen.rembg.BackgroundRemover")
    @patch(
        "scenecode.agent_utils.geometry_generation_server.geometry_generation.Hunyuan3DPipelineManager"
    )
    @patch("hy3dgen.shapegen.pipelines.export_to_trimesh")
    @patch(
        "scenecode.agent_utils.geometry_generation_server.geometry_generation.Image"
    )
    def test_generate_geometry_from_image_with_mini_model(
        self,
        mock_image,
        mock_export_to_trimesh,
        mock_pipeline_manager_class,
        mock_bg_remover_class,
    ):
        """Test geometry generation with mini model and pipeline caching."""
        # Mock image loading and processing.
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img

        # Mock pipeline manager and pipelines.
        mock_shape_pipeline = MagicMock()
        mock_pipeline_output = MagicMock()
        mock_shape_pipeline.return_value = mock_pipeline_output

        mock_texture_pipeline = MagicMock()
        mock_face_reducer = MagicMock()
        mock_background_remover = MagicMock()

        mock_mesh = MagicMock()
        mock_face_reducer.return_value = mock_mesh
        mock_texture_pipeline.return_value = mock_mesh
        mock_background_remover.return_value = mock_img
        mock_bg_remover_class.return_value = mock_background_remover

        # Configure pipeline manager to return mocked pipelines.
        mock_pipeline_manager_class.get_pipelines.return_value = (
            mock_shape_pipeline,
            mock_texture_pipeline,
            mock_face_reducer,
            mock_background_remover,
        )

        # Mock export_to_trimesh.
        mock_export_to_trimesh.return_value = [mock_mesh]

        # Test paths.
        image_path = self.temp_path / "test_image.png"
        output_path = self.temp_path / "test_output.glb"

        # Call function with mini model and caching enabled.
        generate_geometry_from_image(
            image_path=image_path,
            output_path=output_path,
            use_mini=True,
            use_pipeline_caching=True,
        )

        # Verify pipeline manager was called with mini model config.
        mock_pipeline_manager_class.get_pipelines.assert_called_once_with(use_mini=True)

        # Verify mesh processing pipeline.
        mock_export_to_trimesh.assert_called_once_with(mock_pipeline_output)
        mock_face_reducer.assert_called_once_with(mock_mesh)

        # Verify final export.
        mock_mesh.export.assert_called_once_with(output_path)


class TestAssetConversion(unittest.TestCase):
    """Test the asset conversion functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)


class TestExtractAndSaveImage(unittest.TestCase):
    """Test the _extract_and_save_openai_image utility function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("builtins.open", create=True)
    @patch("base64.b64decode")
    def test_extract_and_save_image_success(self, mock_b64decode, mock_open):
        """Test successful image extraction and saving."""
        # Mock response with image data (Images API format).
        mock_data = MagicMock()
        mock_data.b64_json = "base64_image_data"

        mock_response = MagicMock()
        mock_response.data = [mock_data]

        # Mock base64 decode.
        mock_b64decode.return_value = b"decoded_image_data"

        # Mock file operations.
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Call function.
        output_path = self.temp_path / "test.png"
        _extract_and_save_openai_image(mock_response, output_path, "test chair")

        # Check base64 decode was called.
        mock_b64decode.assert_called_once_with("base64_image_data")

        # Check file was opened and written.
        mock_open.assert_called_once_with(output_path, "wb")
        mock_file.write.assert_called_once_with(b"decoded_image_data")

    def test_extract_and_save_image_no_data(self):
        """Test error when no image data in response."""
        # Mock response with no data.
        mock_response = MagicMock()
        mock_response.data = []

        # Call function and expect error.
        output_path = self.temp_path / "test.png"
        with self.assertRaises(ValueError) as context:
            _extract_and_save_openai_image(mock_response, output_path, "test chair")

        self.assertIn(
            "No image data returned from OpenAI for test chair", str(context.exception)
        )

    def test_extract_and_save_image_no_b64_json(self):
        """Test error when b64_json is None."""
        # Mock response with None b64_json.
        mock_data = MagicMock()
        mock_data.b64_json = None

        mock_response = MagicMock()
        mock_response.data = [mock_data]

        # Call function and expect error.
        output_path = self.temp_path / "test.png"
        with self.assertRaises(ValueError) as context:
            _extract_and_save_openai_image(mock_response, output_path, "test chair")

        self.assertIn(
            "No base64 image data in response for test chair", str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
