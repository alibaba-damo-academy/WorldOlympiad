from pathlib import Path
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation
import autoresearch.robotics_hanging_geometry as module


@pytest.mark.parametrize("split_vertices",[False,True])
def test_aperture_and_sparse_shaft_are_measured_from_complete_geometry(monkeypatch,split_vertices):
    body=trimesh.creation.annulus(r_min=.035,r_max=.039,height=.1,sections=32)
    body.apply_translation([0,0,.05])
    ring=trimesh.creation.torus(major_radius=.027,minor_radius=.003,major_sections=32,minor_sections=12)
    matrix=np.eye(4);matrix[:3,:3]=Rotation.from_euler("x",90,degrees=True).as_matrix()
    matrix[:3,3]=[-.063,0,.061];ring.apply_transform(matrix)
    mug=trimesh.util.concatenate([body,ring])
    if split_vertices:
        vertices=mug.triangles.reshape(-1,3)
        mug=trimesh.Trimesh(vertices=vertices,faces=np.arange(len(vertices)).reshape(-1,3),process=False)
    base=trimesh.creation.box([.16,.12,.025]);base.apply_translation([0,0,.0125])
    post=trimesh.creation.box([.018,.018,.155]);post.apply_translation([0,0,.1025])
    # Only two end rings of vertices: no vertices occur at the shaft midpoint.
    shaft=trimesh.creation.cylinder(radius=.006,height=.06,sections=16)
    matrix=np.eye(4);matrix[:3,:3]=Rotation.from_euler("y",90,degrees=True).as_matrix()
    matrix[:3,3]=[-.039,0,.145];shaft.apply_transform(matrix)
    rack=trimesh.util.concatenate([base,post,shaft])
    monkeypatch.setattr(module,"visual_mesh",lambda p:mug if p.name=="mug" else rack)
    result=module.measure_assets(Path("mug"),Path("rack"))
    assert result["handle_inner_radius"]==pytest.approx(.024,abs=1e-5)
    assert result["ring_center"]==pytest.approx([-.063,0,.061],abs=1e-5)
    assert result["hook_bar_z"]==pytest.approx(.145,abs=1e-5)
    assert result["hook_radius"]==pytest.approx(.006,abs=1e-5)
    assert result["hook_tip_z"]==pytest.approx(.151,abs=1e-5)
