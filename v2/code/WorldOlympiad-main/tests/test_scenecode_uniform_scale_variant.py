from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "derive_scenecode_uniform_scale_variant.py"
SPEC = importlib.util.spec_from_file_location("derive_scenecode_uniform_scale_variant", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_uniform_scale_updates_geometry_mass_and_inertia(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "cup.gltf").write_text("{}")
    (source / "collision.obj").write_text("mesh")
    (source / "visual_variant_manifest.json").write_text("{}")
    (source / "cup.sdf").write_text(
        """<sdf><model name='cup'><link name='base'>
        <inertial><mass>0.32</mass><pose>0.01 0 0.04 0 0 0</pose><inertia>
        <ixx>0.001</ixx><iyy>0.002</iyy><izz>0.003</izz>
        <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <visual><geometry><mesh><uri>cup.gltf</uri></mesh></geometry></visual>
        <collision><geometry><mesh><uri>collision.obj</uri><scale>1 1 1</scale></mesh></geometry></collision>
        </link></model></sdf>"""
    )
    output = tmp_path / "scaled"
    result = MODULE.derive_uniform_scale_variant(
        source / "cup.sdf", output, variant_id="cup_scale_080", scale=0.8
    )
    root = ET.parse(result).getroot()
    assert float(root.findtext(".//mass")) == pytest.approx(0.32 * 0.8**3)
    assert float(root.findtext(".//ixx")) == pytest.approx(0.001 * 0.8**5)
    assert [float(x) for x in root.findtext(".//visual/geometry/mesh/scale").split()] == [0.8] * 3
    assert [float(x) for x in root.findtext(".//collision/geometry/mesh/scale").split()] == [0.8] * 3
    manifest = json.loads((output / "physical_variant_manifest.json").read_text())
    assert manifest["visual_collision_scale_coherent"] is True
    assert manifest["changed_files"] == ["cup.sdf"]
    assert not (output / "visual_variant_manifest.json").exists()
