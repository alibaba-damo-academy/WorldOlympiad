import json
from pathlib import Path
import numpy as np
import pytest
import trimesh
from autoresearch.models import BodyIR, Geometry, LinkIR
from scripts.build_cup_to_tray_worldir import _portable_body


def test_portable_worldir_keeps_separate_same_named_gltf_buffers(tmp_path):
    destination = tmp_path / "portable"
    packed = []
    sources = {}
    for idx, extent in enumerate((.1, .4)):
        folder = tmp_path / str(idx)
        folder.mkdir()
        exported = trimesh.Scene(trimesh.creation.box([extent, .2, .3])).export(file_type="gltf")
        for name, data in exported.items():
            (folder/name).write_bytes(data)
        source = folder / "model.gltf"
        body = BodyIR(name=f"body_{idx}", semantic_category="block", links=[LinkIR(name="base", visuals=[
            Geometry(name="surface", kind="mesh", path=str(source))])])
        result = _portable_body(body, destination, sources)
        mesh_path = destination / result.links[0].visuals[0].path
        doc = json.loads(mesh_path.read_text())
        assert all((mesh_path.parent / row["uri"]).is_file() for row in doc["buffers"])
        mesh = trimesh.load(mesh_path, force="scene").to_geometry()
        assert np.ptp(mesh.vertices, axis=0) == pytest.approx([extent, .2, .3], abs=1e-7)
        packed.append(mesh_path)
    assert packed[0].parent != packed[1].parent
