import os
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from autoresearch.compiler import _prepare_mesh, gltf_y_up_to_worldir_z_up_transform
from autoresearch.models import Geometry


@pytest.mark.skipif(not os.environ.get("BLENDER_BIN"), reason="requires the registered Blender runtime")
def test_material_preserving_blender_mesh_keeps_asymmetric_source_coordinates(tmp_path):
    mesh = trimesh.creation.box([.2, .3, .4])
    mesh.apply_translation([-.7, 1.2, .6])
    source = tmp_path / "source"
    source.mkdir()
    files = trimesh.Scene(mesh).export(file_type="gltf")
    for name, data in files.items():
        (source / name).write_bytes(data)
    gltf = next(source.glob("*.gltf"))
    assets = tmp_path / "converted"
    assets.mkdir()
    result = _prepare_mesh(Geometry(name="asymmetric", kind="mesh", path=str(gltf)),
                           assets, visual=True, repairs=[], attempt=1)
    expected = trimesh.load(gltf, force="scene").to_geometry()
    expected.apply_transform(gltf_y_up_to_worldir_z_up_transform())
    actual = trimesh.load(assets / result, force="scene").to_geometry()
    assert np.allclose(actual.bounds, expected.bounds, atol=1e-6)
    assert cKDTree(expected.vertices).query(actual.vertices)[0].max() < 1e-6
