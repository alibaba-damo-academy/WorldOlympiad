import argparse
import sys
import tempfile
import types
import unittest

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

for _key in [k for k in sys.modules if k == "diffusers" or k.startswith("diffusers.")]:
    del sys.modules[_key]

_diffusers_stub = types.ModuleType("diffusers")


class _Flux2KleinPipeline:  # pragma: no cover - import shim for tests
    pass


_diffusers_stub.Flux2KleinPipeline = _Flux2KleinPipeline
sys.modules["diffusers"] = _diffusers_stub

from scenecode.agent_utils import flux_klein_worker


class TestFluxKleinWorker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.temp_dir.name) / "output.png"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _build_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            mode="generate",
            prompt="test prompt",
            output_path=str(self.output_path),
            reference_image_path=None,
            model_path="/model",
            width=512,
            height=512,
            num_inference_steps=4,
            guidance_scale=1.0,
            max_sequence_length=256,
            seed=41,
        )

    def test_main_runs_cleanup_on_success(self):
        args = self._build_args()
        fake_image = MagicMock()
        fake_pipe = MagicMock(return_value=SimpleNamespace(images=[fake_image]))

        with patch.object(flux_klein_worker, "_configure_logging"), patch.object(
            flux_klein_worker, "parse_args", return_value=args
        ), patch.object(
            flux_klein_worker, "build_pipeline", return_value=(fake_pipe, "cpu")
        ), patch.object(
            flux_klein_worker, "cleanup_worker_resources"
        ) as mock_cleanup, patch.object(
            flux_klein_worker, "describe_gpu_snapshot", return_value="gpu snapshot"
        ), patch.object(
            flux_klein_worker.torch, "inference_mode", return_value=nullcontext()
        ):
            flux_klein_worker.main()

        fake_image.save.assert_called_once_with(self.output_path)
        mock_cleanup.assert_called_once()
        self.assertIs(mock_cleanup.call_args.kwargs["pipe"], fake_pipe)
        self.assertIs(mock_cleanup.call_args.kwargs["generated_image"], fake_image)
        self.assertIsNone(mock_cleanup.call_args.kwargs["reference_image"])

    def test_main_runs_cleanup_on_exception(self):
        args = self._build_args()
        fake_pipe = MagicMock(side_effect=RuntimeError("boom"))

        with patch.object(flux_klein_worker, "_configure_logging"), patch.object(
            flux_klein_worker, "parse_args", return_value=args
        ), patch.object(
            flux_klein_worker, "build_pipeline", return_value=(fake_pipe, "cpu")
        ), patch.object(
            flux_klein_worker, "cleanup_worker_resources"
        ) as mock_cleanup, patch.object(
            flux_klein_worker, "describe_gpu_snapshot", return_value="gpu snapshot"
        ), patch.object(
            flux_klein_worker.torch, "inference_mode", return_value=nullcontext()
        ):
            with self.assertRaises(RuntimeError):
                flux_klein_worker.main()

        mock_cleanup.assert_called_once()
        self.assertIs(mock_cleanup.call_args.kwargs["pipe"], fake_pipe)
        self.assertIsNone(mock_cleanup.call_args.kwargs["generated_image"])
