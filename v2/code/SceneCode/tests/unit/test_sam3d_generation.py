import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from PIL import Image

# Mock SAM3 and SAM 3D Objects modules for CI where they're not installed.
try:
    import sam3  # noqa: F401
except ImportError:
    import sys

    # Create structured mocks.
    sam3 = MagicMock()
    sam3.build_sam3 = MagicMock()
    sam3.build_sam3.build_sam3_video_predictor = MagicMock()

    sys.modules["sam3"] = sam3
    sys.modules["sam3.build_sam3"] = sam3.build_sam3

try:
    import sam3d_pipeline  # noqa: F401
except ImportError:
    import sys

    sam3d_pipeline = MagicMock()
    sam3d_pipeline.SAM3DPipeline = MagicMock()

    sys.modules["sam3d_pipeline"] = sam3d_pipeline

from scenecode.agent_utils.geometry_generation_server.geometry_generation import (
    generate_geometry_from_image,
)
from scenecode.agent_utils.geometry_generation_server.sam3d_pipeline_manager import (
    SAM3DPipelineManager,
    generate_3d_from_mask,
    generate_mask,
)


class TestSAM3DPipelineManager(unittest.TestCase):
    """Test the SAM3DPipelineManager singleton class."""

    def setUp(self):
        """Reset the singleton before each test."""
        # Reset singleton instance.
        SAM3DPipelineManager._instance = None
        SAM3DPipelineManager._sam3_model = None
        SAM3DPipelineManager._sam3d_pipeline = None
        SAM3DPipelineManager._current_config = None

    @patch(
        "scenecode.agent_utils.geometry_generation_server.sam3d_pipeline_manager.torch"
    )
    def test_singleton_pattern(self, mock_torch):
        """Test that SAM3DPipelineManager follows singleton pattern."""
        instance1 = SAM3DPipelineManager()
        instance2 = SAM3DPipelineManager()

        self.assertIs(instance1, instance2)

    @patch(
        "scenecode.agent_utils.geometry_generation_server.sam3d_pipeline_manager.torch"
    )
    def test_are_pipelines_loaded(self, mock_torch):
        """Test checking if pipelines are loaded."""
        # Initially, pipelines should not be loaded.
        self.assertFalse(SAM3DPipelineManager.are_pipelines_loaded())


class TestGenerateMask(unittest.TestCase):
    """Test the generate_mask function."""

    def test_foreground_mode(self):
        """Test mask generation in foreground mode.

        Tests that:
        1. Background mask is inverted to get foreground
        2. Edge-connected regions are removed (artifact cleanup)
        3. SAM3 API is called correctly
        """
        mock_processor = MagicMock()

        # 5x5 mask: background (True) everywhere except center 3x3 region.
        # After inversion, center 3x3 becomes foreground.
        # Edge removal keeps center since it doesn't touch edges.
        mock_masks = np.array(
            [
                [
                    [True, True, True, True, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, True, True, True, True],
                ]
            ],
            dtype=bool,
        )
        mock_scores = np.array([0.9])

        mock_processor.set_image.return_value = {}
        mock_processor.set_text_prompt.return_value = {
            "masks": mock_masks,
            "scores": mock_scores,
        }

        test_image = Image.new("RGB", (100, 100), color=(255, 0, 0))

        mask = generate_mask(
            image=test_image,
            sam3_processor=mock_processor,
            mode="foreground",
            threshold=0.5,
        )

        # After inversion: center 3x3 is 1, edges are 0.
        # Edge removal keeps center (doesn't touch boundary).
        self.assertEqual(mask.shape, (5, 5))
        expected = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(mask, expected)

        # Verify API calls.
        mock_processor.set_image.assert_called_once()
        mock_processor.set_text_prompt.assert_called_once()
        call_kwargs = mock_processor.set_text_prompt.call_args[1]
        self.assertEqual(call_kwargs.get("prompt"), "background")


class TestGenerate3DFromMask(unittest.TestCase):
    """Test the generate_3d_from_mask function."""

    def test_mesh_generation(self):
        """Test 3D mesh generation from mask.

        The pipeline returns a dict with {"glb": mesh} and we call
        output["glb"].export(path).
        """
        # Create mock SAM3D pipeline.
        mock_pipeline = MagicMock()
        mock_glb = MagicMock()
        # Pipeline returns dict with "glb" key.
        mock_pipeline.run.return_value = {"glb": mock_glb}

        # Create test image and mask.
        test_image = Image.new("RGB", (100, 100), color=(255, 0, 0))
        test_mask = np.ones((100, 100), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.glb"

            # Generate 3D mesh.
            generate_3d_from_mask(
                image=test_image,
                mask=test_mask,
                sam3d_pipeline=mock_pipeline,
                output_path=output_path,
            )

            # Verify pipeline.run was called with expected args.
            mock_pipeline.run.assert_called_once()
            call_args = mock_pipeline.run.call_args
            # First arg is RGBA image.
            rgba_image = call_args[0][0]
            self.assertEqual(rgba_image.shape[-1], 4)  # RGBA has 4 channels.

            # Verify glb.export was called with string path.
            mock_glb.export.assert_called_once_with(str(output_path))


class TestGeometryGeneration(unittest.TestCase):
    """Test the geometry generation routing."""

    @patch(
        "scenecode.agent_utils.geometry_generation_server.geometry_generation.generate_with_sam3d"
    )
    def test_backend_routing_sam3d(self, mock_sam3d):
        """Test that backend routing correctly calls SAM3D backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test.png"
            output_path = temp_path / "output.glb"

            # Create test image.
            test_image = Image.new("RGB", (100, 100), color=(255, 0, 0))
            test_image.save(image_path)

            sam3d_config = {
                "sam3_checkpoint": temp_path / "sam3.pt",
                "sam3d_checkpoint": temp_path / "sam3d.ckpt",
                "mode": "foreground",
                "text_prompt": None,
                "threshold": 0.5,
            }

            # Call with sam3d backend.
            generate_geometry_from_image(
                image_path=image_path,
                output_path=output_path,
                backend="sam3d",
                sam3d_config=sam3d_config,
            )

            # Verify SAM3D backend was called with correct parameters.
            mock_sam3d.assert_called_once()
            call_kwargs = mock_sam3d.call_args[1]
            self.assertEqual(
                call_kwargs["sam3_checkpoint"], sam3d_config["sam3_checkpoint"]
            )
            self.assertEqual(
                call_kwargs["sam3d_checkpoint"], sam3d_config["sam3d_checkpoint"]
            )
            self.assertEqual(call_kwargs["mode"], sam3d_config["mode"])

    def test_backend_routing_sam3d_without_config_raises_error(self):
        """Test that SAM3D backend without config raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test.png"
            output_path = temp_path / "output.glb"

            # Create test image.
            test_image = Image.new("RGB", (100, 100), color=(255, 0, 0))
            test_image.save(image_path)

            with self.assertRaises(ValueError) as context:
                generate_geometry_from_image(
                    image_path=image_path,
                    output_path=output_path,
                    backend="sam3d",
                    sam3d_config=None,
                )

            self.assertIn("sam3d_config is required", str(context.exception))

    def test_invalid_backend_raises_error(self):
        """Test that invalid backend raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "test.png"
            output_path = temp_path / "output.glb"

            # Create test image.
            test_image = Image.new("RGB", (100, 100), color=(255, 0, 0))
            test_image.save(image_path)

            with self.assertRaises(ValueError) as context:
                generate_geometry_from_image(
                    image_path=image_path,
                    output_path=output_path,
                    backend="invalid_backend",
                )

            self.assertIn("Unknown backend", str(context.exception))


if __name__ == "__main__":
    unittest.main()
