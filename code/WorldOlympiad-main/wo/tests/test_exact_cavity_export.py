import hashlib
import xml.etree.ElementTree as ET
import trimesh
from wo.sim.prismatic_collision import preserve_prismatic_cavities


def test_shared_sdf_export_keeps_visual_and_publishes_exact_collision(tmp_path):
    mesh = trimesh.creation.annulus(r_min=.0275, r_max=.04, height=.05, sections=32)
    visual = tmp_path / "visual.obj"
    mesh.export(visual)
    before = hashlib.sha256(visual.read_bytes()).hexdigest()
    sdf = tmp_path / "asset.sdf"
    sdf.write_text(f'<sdf version="1.6"><model name="asset"><link name="body">'
        f'<visual name="visual"><geometry><mesh><uri>{visual}</uri></mesh></geometry></visual>'
        f'<collision name="old"><geometry><mesh><uri>{visual}</uri></mesh></geometry></collision>'
        '</link></model></sdf>')
    receipt = preserve_prismatic_cavities(sdf, sdf)
    assert receipt and receipt["policy"] == "exact_vertical_triangle_prisms"
    assert hashlib.sha256(visual.read_bytes()).hexdigest() == before
    root = ET.parse(sdf)
    assert root.findtext('.//visual/geometry/mesh/uri') == str(visual)
    assert len(root.findall('.//collision')) == receipt["pieces"]
    assert abs(receipt["partition_volume_m3"] - receipt["visual_volume_m3"]) < 1e-10
    assert preserve_prismatic_cavities(sdf, sdf) == receipt


def test_concave_shell_is_not_convexified(tmp_path):
    from wo.sim.concave_collision import preserve_concave_shell
    mesh = trimesh.creation.annulus(r_min=.0275, r_max=.04, height=.05, sections=16)
    visual = tmp_path / 'shell.obj'; mesh.export(visual)
    sdf = tmp_path / 'shell.sdf'
    sdf.write_text(f'<sdf version="1.6"><model name="asset"><link name="body">'
                   f'<visual name="visual"><geometry><mesh><uri>{visual}</uri></mesh></geometry></visual>'
                   f'<collision name="old"><geometry><mesh><uri>{visual}</uri></mesh></geometry></collision>'
                   '</link></model></sdf>')
    receipt = preserve_concave_shell(sdf, sdf)
    assert receipt and receipt['policy'] == 'face_preserving_convex_shell_decomposition'
    assert receipt['pieces'] == len(mesh.faces)
    assert len(ET.parse(sdf).findall('.//collision')) == len(mesh.faces)


def test_concave_shell_with_exporter_seams_is_preserved(tmp_path):
    from wo.sim.concave_collision import preserve_concave_shell
    mesh = trimesh.creation.annulus(r_min=.0275, r_max=.04, height=.05, sections=16)
    # Simulate a source exporter that duplicates coincident seam vertices.
    duplicated = trimesh.Trimesh(
        vertices=mesh.vertices[mesh.faces].reshape(-1, 3),
        faces=__import__('numpy').arange(len(mesh.faces) * 3).reshape(-1, 3),
        process=False,
    )
    visual = tmp_path / 'seamed.obj'; duplicated.export(visual)
    sdf = tmp_path / 'seamed.sdf'
    sdf.write_text(f'<sdf version="1.6"><model name="asset"><link name="body">'
                   f'<visual name="visual"><geometry><mesh><uri>{visual}</uri></mesh></geometry></visual>'
                   f'<collision name="old"><geometry><mesh><uri>{visual}</uri></mesh></geometry></collision>'
                   '</link></model></sdf>')
    receipt = preserve_concave_shell(sdf, sdf)
    assert receipt and receipt['pieces'] == len(duplicated.faces)
