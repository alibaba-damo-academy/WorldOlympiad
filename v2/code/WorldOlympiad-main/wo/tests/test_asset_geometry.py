import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("bpy")
from wo.tasks.asset_geometry import measure_asset


@pytest.fixture
def native_asset(tmp_path):
    import trimesh
    mesh = tmp_path / "body.glb"
    shape = trimesh.creation.box(extents=[.4, .3, .2])
    shape.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    shape.export(mesh)
    sdf = tmp_path / "asset.sdf"
    sdf.write_text("""<sdf version="1.7"><model name="asset"><pose>9 8 7 0 0 0</pose>
<link name="body"><pose>1 0 0 0 0 0</pose><visual name="v"><pose>0.1 0 0 0 0 0</pose>
<geometry><mesh><uri>body.glb</uri><scale>2 2 2</scale></mesh></geometry>
</visual></link></model></sdf>""")
    return sdf, mesh


def test_native_scaled_mesh_model_frame_preserves_source(native_asset):
    sdf, mesh = native_asset
    before = [p.read_bytes() for p in [sdf, mesh]]
    data = measure_asset(sdf)
    body = data["links"][0]
    assert body["name"] == "body"
    assert np.allclose(body["min_m"], [.7, -.3, -.2])
    assert np.allclose(body["max_m"], [1.5, .3, .2])
    assert np.allclose(body["size_m"], [.8, .6, .4])
    assert not data["grasp_collision_execution_tested"]
    assert data["sdf_sha256"] == hashlib.sha256(before[0]).hexdigest()
    assert before == [p.read_bytes() for p in [sdf, mesh]]


def test_cli_returns_parseable_json(native_asset):
    sdf, _ = native_asset
    result = subprocess.run([sys.executable, "-m", "wo.tasks.asset_geometry", "--sdf", str(sdf)],
        cwd=Path(__file__).resolve().parents[2], env=dict(os.environ),
        text=True, capture_output=True, timeout=30, check=True)
    data = json.loads(result.stdout)
    assert data["sdf"] == str(sdf)
    assert np.allclose(data["links"][0]["size_m"], [.8, .6, .4])


def test_missing_visual_measurement_is_error(tmp_path):
    sdf = tmp_path / "no_visual.sdf"
    sdf.write_text('<sdf version="1.7"><model name="empty"><link name="body"/></model></sdf>')
    with pytest.raises((ValueError, RuntimeError)):
        measure_asset(sdf)


def test_unsupported_visual_shape_is_not_silently_omitted(tmp_path):
    sdf = tmp_path / "primitive.sdf"
    sdf.write_text('<sdf version="1.7"><model name="box"><link name="body"><visual name="v"><geometry><box><size>1 1 1</size></box></geometry></visual></link></model></sdf>')
    with pytest.raises(ValueError, match="no partial measurement"):
        measure_asset(sdf)
