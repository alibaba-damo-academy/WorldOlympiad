from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh
from autoresearch.scenecode_mesh_export_receipt import record_blender_y_up_export

from autoresearch.scenecode_mesh_frame_conversion import (
    Y_UP_TO_Z_UP, bind_visual_frame, verified_mesh_basis,
)


def meshes(tmp_path, *, gltf_rotation=None):
    source = trimesh.creation.box(extents=(.23, .41, .73))
    source.apply_translation([.13, .47, -.19])
    obj = tmp_path / "source.obj"
    obj.write_text(trimesh.exchange.obj.export_obj(source, digits=6))
    converted = source.copy()
    if gltf_rotation is not None:
        transform = np.eye(4)
        transform[:3, :3] = gltf_rotation
        converted.apply_transform(transform)
    gltf = tmp_path / "destination.glb"
    gltf.write_bytes(converted.export(file_type="glb"))
    return obj, gltf


@pytest.mark.parametrize("baked,expected", [
    (None, "undo_gltf_up_axis"), (Y_UP_TO_Z_UP.T, "identity"),
])
def test_distinguishes_same_raw_axes_from_already_converted_mesh(tmp_path, baked, expected):
    obj, gltf = meshes(tmp_path, gltf_rotation=baked)
    receipt = verified_mesh_basis(obj, gltf)
    assert receipt["basis_name"] == expected
    assert receipt["passed"]


def test_composes_origin_and_preserves_translation(tmp_path):
    obj, gltf = meshes(tmp_path)
    urdf = ET.fromstring(f'<visual><geometry><mesh filename="{obj}"/></geometry></visual>')
    sdf = ET.fromstring(f'<visual><pose>.1 .2 .3 .7 -.2 1.1</pose><geometry><mesh><uri>{gltf}</uri></mesh></geometry></visual>')
    bind_visual_frame(urdf, sdf, tmp_path)
    pose = np.fromstring(sdf.findtext("pose"), sep=" ")
    np.testing.assert_array_equal(pose[:3], [.1, .2, .3])
    actual = Rotation.from_euler("xyz", pose[3:]).as_matrix() @ Y_UP_TO_Z_UP
    np.testing.assert_allclose(actual, Rotation.from_euler("xyz", [.7, -.2, 1.1]).as_matrix(), atol=1e-12)


@pytest.mark.parametrize("change", ["translation", "scale", "triangles"])
def test_rejects_geometry_change_instead_of_fitting_it(tmp_path, change):
    obj, gltf = meshes(tmp_path)
    mesh = trimesh.load(str(gltf), force="mesh", process=False)
    if change == "translation":
        mesh.apply_translation([0, .1, 0])
    elif change == "scale":
        mesh.apply_scale(.9)
    else:
        mesh.update_faces(np.arange(len(mesh.faces) - 1))
    gltf.write_bytes(mesh.export(file_type="glb"))
    with pytest.raises(ValueError, match="mesh"):
        verified_mesh_basis(obj, gltf)


def test_ambiguous_switched_mesh_fails_and_identical_declared_mesh_is_safe(tmp_path):
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    obj = tmp_path / "cube.obj"
    gltf = tmp_path / "cube.glb"
    obj.write_text(mesh.export(file_type="obj"))
    gltf.write_bytes(mesh.export(file_type="glb"))
    with pytest.raises(ValueError, match="ambiguous"):
        verified_mesh_basis(obj, gltf)
    assert verified_mesh_basis(obj, obj)["basis_name"] == "identity"


def test_actual_exporter_axes_resolve_symmetry_and_stale_receipts_fail(tmp_path):
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    obj, gltf = tmp_path / "cube.obj", tmp_path / "cube.gltf"
    obj.write_text(mesh.export(file_type="obj"))
    for name, data in mesh.export(file_type="gltf").items():
        target = gltf if name.endswith(".gltf") else tmp_path / name
        target.write_bytes(data)
    for path in (obj, gltf):
        record_blender_y_up_export(path)
    assert verified_mesh_basis(obj, gltf)["basis_name"] == "verified_exporter_axes"
    obj.write_text(obj.read_text() + "\n# source changed\n")
    with pytest.raises(ValueError, match="receipt"):
        verified_mesh_basis(obj, gltf)


def test_unhandled_declared_scale_is_not_silently_ignored(tmp_path):
    obj, gltf = meshes(tmp_path)
    urdf = ET.fromstring(f'<visual><geometry><mesh filename="{obj}" scale="2 2 2"/></geometry></visual>')
    sdf = ET.fromstring(f'<visual><pose>0 0 0 0 0 0</pose><geometry><mesh><uri>{gltf}</uri></mesh></geometry></visual>')
    with pytest.raises(ValueError, match="physics conversion"):
        bind_visual_frame(urdf, sdf, tmp_path)
