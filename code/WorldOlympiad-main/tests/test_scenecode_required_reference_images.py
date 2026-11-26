import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

RUNTIME = Path(os.environ.get("SCENECODE_RUNTIME_ROOT", "/tmp/SceneCode-main-worldolympiad"))
ROUTER = RUNTIME / "scenecode/agent_utils/asset_router/router.py"
if not ROUTER.is_file():
    pytest.skip("requires staged SceneCode runtime", allow_module_level=True)


@pytest.fixture(params=[("_generate_code_object_geometry", "generate_from_image", "generate_from_text"),
                        ("_generate_code_articulated_geometry", "generate_articulated_from_image", "generate_articulated_from_text")])
def method(request, monkeypatch):
    from autoresearch import scenecode_image_evidence as evidence
    name, image_method, text_method = request.param
    tree = ast.parse(ROUTER.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"os": os, "console_logger": logging.getLogger("required-reference-test")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), str(ROUTER), "exec"), namespace)
    result = SimpleNamespace(status="no_movable_parts")
    runner = SimpleNamespace(**{image_method: Mock(return_value=result), text_method: Mock(return_value=result)})
    owner = SimpleNamespace(_prepare_code_object_inputs=Mock(), _build_code_generated_geometry=Mock(return_value="geometry"))
    inputs, outputs = Mock(), Mock()
    monkeypatch.setattr(evidence, "record_image_input", inputs)
    monkeypatch.setattr(evidence, "record_image_result", outputs)
    monkeypatch.setattr(evidence, "prepare_articulated_image_result", lambda _output, value: value)
    def call(image):
        owner._prepare_code_object_inputs.return_value = (image, Path("/task/output"), None)
        return namespace[name](owner, item=SimpleNamespace(description="authored object", dimensions=[.1,.1,.1], short_name="authored_object"),
                               code_object_runner=runner, image_generator=object(),
                               images_dir=Path("/task/images"), code_object_dir=Path("/task/code"))
    return SimpleNamespace(call=call, runner=runner, image_method=image_method, text_method=text_method,
                           result=result, inputs=inputs, outputs=outputs)


def test_required_missing_image_never_enters_text_fallback(method, monkeypatch):
    monkeypatch.setenv("SCENECODE_REQUIRE_REFERENCE_IMAGE", "1")
    with pytest.raises(RuntimeError, match="required SceneCode asset reference image"):
        method.call(None)
    getattr(method.runner, method.text_method).assert_not_called()
    getattr(method.runner, method.image_method).assert_not_called()
    method.inputs.assert_not_called()
    method.outputs.assert_not_called()


def test_image_path_records_correct_method_and_result_once(method, monkeypatch):
    monkeypatch.setenv("SCENECODE_REQUIRE_REFERENCE_IMAGE", "1")
    image = Path("/task/images/object.png")
    assert method.call(image) == "geometry"
    getattr(method.runner, method.image_method).assert_called_once()
    getattr(method.runner, method.text_method).assert_not_called()
    method.inputs.assert_called_once()
    assert method.inputs.call_args.args[:2] == (Path("/task/output"), image)
    assert method.inputs.call_args.kwargs["method"] == method.image_method
    method.outputs.assert_called_once_with(Path("/task/output"), method.result)


def test_nonrequired_text_route_is_unchanged(method, monkeypatch):
    monkeypatch.delenv("SCENECODE_REQUIRE_REFERENCE_IMAGE", raising=False)
    assert method.call(None) == "geometry"
    getattr(method.runner, method.text_method).assert_called_once()
    getattr(method.runner, method.image_method).assert_not_called()
    method.inputs.assert_not_called()
    method.outputs.assert_not_called()
