import sys
import tempfile
import types
import unittest

from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# --- Module stubs to avoid heavy/broken transitive dependencies ---

if "omegaconf" not in sys.modules:
    _omegaconf_stub = types.ModuleType("omegaconf")

    class _DictConfig(dict):  # pragma: no cover
        pass

    _omegaconf_stub.DictConfig = _DictConfig
    sys.modules["omegaconf"] = _omegaconf_stub

for _key in [k for k in sys.modules if k == "openai" or k.startswith("openai.")]:
    del sys.modules[_key]

_openai_stub = types.ModuleType("openai")
_openai_stub.__spec__ = ModuleSpec("openai", None)


class _OpenAI:  # pragma: no cover
    pass


class _AsyncOpenAI:  # pragma: no cover
    pass


_openai_stub.OpenAI = _OpenAI
_openai_stub.AsyncOpenAI = _AsyncOpenAI
sys.modules["openai"] = _openai_stub

for _key in [k for k in sys.modules if k == "agents" or k.startswith("agents.")]:
    del sys.modules[_key]

_agents_stub = types.ModuleType("agents")


class _RunConfig:  # pragma: no cover
    pass


def _set_default_openai_client(*_a, **_kw):  # pragma: no cover
    pass


_agents_stub.RunConfig = _RunConfig
_agents_stub.set_default_openai_client = _set_default_openai_client
sys.modules["agents"] = _agents_stub

if "google" not in sys.modules:
    _google_stub = types.ModuleType("google")
    _genai_stub = types.ModuleType("google.genai")
    _genai_types_stub = types.ModuleType("google.genai.types")
    _google_stub.genai = _genai_stub
    _genai_stub.types = _genai_types_stub
    sys.modules["google"] = _google_stub
    sys.modules["google.genai"] = _genai_stub
    sys.modules["google.genai.types"] = _genai_types_stub

from scenecode.agent_utils.image_generation import FluxKleinImageGenerator


class ConfigStub(dict):
    def __getattr__(self, name):
        return self[name]


class TestFluxKleinImageGenerator(unittest.TestCase):
    def test_run_worker_includes_parent_gpu_snapshot_on_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            python_executable = temp_path / "python"
            python_executable.write_text("", encoding="utf-8")
            model_path = temp_path / "model"
            model_path.write_text("", encoding="utf-8")

            generator = FluxKleinImageGenerator(
                ConfigStub(
                    python_executable=str(python_executable),
                    model_path=str(model_path),
                    width=512,
                    height=512,
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    max_sequence_length=256,
                    seed=41,
                )
            )

            output_path = temp_path / "output.png"
            with patch(
                "scenecode.agent_utils.image_generation.describe_gpu_snapshot",
                return_value="GPU snapshot [parent]",
            ), patch(
                "scenecode.agent_utils.image_generation.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=1,
                    stdout="worker stdout",
                    stderr="worker stderr",
                ),
            ):
                with self.assertRaises(RuntimeError) as context:
                    generator._run_worker(
                        mode="generate",
                        prompt="hello",
                        output_path=output_path,
                        width=512,
                        height=512,
                    )

            message = str(context.exception)
            self.assertIn("worker stdout", message)
            self.assertIn("worker stderr", message)
            self.assertIn("GPU snapshot [parent]", message)
