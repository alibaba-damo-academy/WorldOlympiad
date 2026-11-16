import ast
from pathlib import Path
import re
import subprocess

import pytest

PATCH = Path(__file__).resolve().parents[1] / "deployment/ziplab5090/patches/scenecode_thin_covering_shape_words.patch"
UPSTREAM = '"""Thin covering generator for procedural flat textured surfaces.\n\nThis module provides utilities for generating thin textured meshes\nwith PBR materials. Supports floor coverings (rugs, mats), wall coverings\n(posters, prints), and surface coverings (tablecloths, placemats).\nThese objects are purely decorative (no collision geometry) and are\nstatic (welded to world).\n"""\n\nimport logging\nimport math\nimport xml.etree.ElementTree as ET\n\nfrom pathlib import Path\n\nimport numpy as np\n\nfrom scenecode.utils.gltf_generation import (\n    create_glb_from_mesh_data,\n    zup_to_yup_transform,\n)\nfrom scenecode.utils.material import Material\n\nconsole_logger = logging.getLogger(__name__)\n\n\ndef infer_thin_covering_shape(description: str) -> str:\n    """Infer thin covering shape from description using keyword matching.\n\n    Args:\n        description: Thin covering description text.\n\n    Returns:\n        "circular" if description contains round/circular keywords,\n        "rectangular" otherwise.\n    """\n    desc_lower = description.lower()\n    if "round" in desc_lower or "circular" in desc_lower:\n        return "circular"\n    return "rectangular"\n\n\ndef create_rectangular_thin_covering_glb(\n    width: float,\n    second_dim: float,\n    thickness: float,\n    material_folder: Path,\n    output_path: Path,\n    texture_scale: float | None,\n    is_wall: bool = False,\n) -> Path:\n    pass\n'


@pytest.fixture
def infer_shape(tmp_path):
    target = tmp_path / "scenecode/agent_utils/thin_covering_generator.py"
    target.parent.mkdir(parents=True)
    target.write_text(UPSTREAM)
    subprocess.run(["patch", "--batch", "--forward", "-p1", "-d", str(tmp_path), "-i", str(PATCH)],
                   check=True, capture_output=True, text=True)
    tree = ast.parse(target.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "infer_thin_covering_shape")
    namespace = {"re": re}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(target), "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("description,expected", [
    ("Thin rigid rectangular slab placemat with gently rounded corners", "rectangular"),
    ("Natural cork rectangular table placemat, 30 by 22 cm, rounded corners", "rectangular"),
    ("A square mat surrounded by a dark border", "rectangular"),
    ("A round cork placemat", "circular"),
    ("CIRCULAR table mat", "circular"),
    ("A round placemat on a rectangular table", "circular"),
    ("A flat mat on a background", "rectangular"),
    ("A softly rounded corner mat", "rectangular"),
])
def test_shape_uses_whole_words_without_changing_round_shapes(infer_shape, description, expected):
    assert infer_shape(description) == expected
