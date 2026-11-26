"""Focused checks for the fixed-joint multi-link -> single-rigid projection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np
import trimesh
import xml.etree.ElementTree as ET

from autoresearch.scenecode_mesh_projection import validate_projection_receipt

from autoresearch.scenecode_rigid_projection import (
    _GLTF_Y_UP_TO_Z_UP_BASIS,
    _rotate_inertial,
    parse_locked_sdf,
    project_and_verify,
    project_single_rigid,
)


def test_gltf_basis_rotates_com_and_inertia_together():
    got = _rotate_inertial({"mass": 1.0, "com": [0.0, 0.0, 0.12],
                            "ixx": 1.0, "iyy": 2.0, "izz": 3.0,
                            "ixy": 0.1, "ixz": 0.2, "iyz": 0.3},
                           _GLTF_Y_UP_TO_Z_UP_BASIS)
    assert got["com"] == pytest.approx([0.0, -0.12, 0.0])
    assert got["ixx"] == pytest.approx(1.0)
    assert got["iyy"] == pytest.approx(3.0)
    assert got["izz"] == pytest.approx(2.0)


CUBE_OBJ = """\
v -0.01 -0.01 -0.01
v 0.01 -0.01 -0.01
v 0.01 0.01 -0.01
v -0.01 0.01 -0.01
v -0.01 -0.01 0.01
v 0.01 -0.01 0.01
v 0.01 0.01 0.01
v -0.01 0.01 0.01
f 1 2 3
f 1 3 4
f 5 7 6
f 5 8 7
f 1 5 6
f 1 6 2
f 2 6 7
f 2 7 3
f 3 7 8
f 3 8 4
f 4 8 5
f 4 5 1
"""


def _write_pair(tmp_path: Path, *, second_link_pose: str = "0 0 0.05 0 0 0"):
    mesh = tmp_path / "part.obj"
    mesh.write_text(CUBE_OBJ, encoding="utf-8")

    def model(link_tag: str, pose: str) -> str:
        return f"""\
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="demo">
    <link name="base_link">
      <inertial><mass>0.30</mass><pose>0 0 0 0 0 0</pose>
        <inertia><ixx>1e-4</ixx><iyy>1e-4</iyy><izz>1e-4</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <{link_tag}><pose>0 0 0 0 0 0</pose><geometry><mesh>
        <uri>{mesh}</uri><scale>1 1 1</scale></mesh></geometry></{link_tag}>
    </link>
    <link name="lid">
      <pose>{pose}</pose>
      <inertial><mass>0.10</mass><pose>0 0 0 0 0 0</pose>
        <inertia><ixx>1e-5</ixx><iyy>1e-5</iyy><izz>1e-5</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <{link_tag}><pose>0 0 0 0 0 0</pose><geometry><mesh>
        <uri>{mesh}</uri><scale>1 1 1</scale></mesh></geometry></{link_tag}>
    </link>
    <joint name="lid_joint" type="fixed"><parent>base_link</parent><child>lid</child></joint>
  </model>
</sdf>
"""

    visual = tmp_path / "visual_locked.sdf"
    visual.write_text(model("visual", second_link_pose), encoding="utf-8")
    physics = tmp_path / "physics_locked.sdf"
    physics.write_text(model("collision", second_link_pose), encoding="utf-8")
    return visual, physics


def test_parse_keeps_every_declared_link_and_mesh(tmp_path: Path) -> None:
    visual, _ = _write_pair(tmp_path)
    links = parse_locked_sdf(visual)
    assert [link["name"] for link in links] == ["base_link", "lid"]
    assert links[1]["pose"][2, 3] == pytest.approx(0.05)
    assert len(links[0]["visuals"]) == 1


def test_projection_yields_one_passing_rigid_body(tmp_path: Path) -> None:
    visual, physics = _write_pair(tmp_path)
    receipt = project_and_verify(
        visual_sdf=visual,
        physics_sdf=physics,
        output_dir=tmp_path / "asset",
        name="demo",
    )
    assert receipt["passed"] is True
    assert receipt["source_link_count"] == {"visual": 2, "physics": 2}
    assert receipt["mesh_count"] == {"visual": 2, "collision": 2}
    assert receipt["verification"]["num_joints"] == 0
    assert receipt["verification"]["base_collision_shapes"] == 2
    assert receipt["combined_inertial"]["mass"] == pytest.approx(0.40)
    written = json.loads((tmp_path / "asset" / "projection_receipt.json").read_text())
    assert written["urdf_sha256"] == receipt["urdf_sha256"]
    urdf = (tmp_path / "asset" / "mobility.urdf").read_text()
    assert urdf.count("<link ") == 1
    assert urdf.count("<joint") == 0


def test_native_rigid_scene_parser_preserves_primitive_geometry(tmp_path: Path) -> None:
    source = tmp_path / "primitives.sdf"
    source.write_text(
        """<sdf version="1.9"><model name="primitives"><link name="body">
        <visual name="box"><geometry><box><size>0.1 0.2 0.3</size></box></geometry></visual>
        <collision name="cylinder"><geometry><cylinder><radius>0.02</radius><length>0.1</length></cylinder></geometry></collision>
        <collision name="sphere"><geometry><sphere><radius>0.03</radius></sphere></geometry></collision>
        </link></model></sdf>""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="resolved local mesh URIs"):
        parse_locked_sdf(source)
    parsed = parse_locked_sdf(source, allow_primitives=True)

    assert parsed[0]["visuals"][0]["kind"] == "box"
    assert parsed[0]["visuals"][0]["size"] == [0.1, 0.2, 0.3]
    assert [item["kind"] for item in parsed[0]["collisions"]] == [
        "cylinder", "sphere",
    ]


def test_projection_refuses_an_unresolved_mesh(tmp_path: Path) -> None:
    visual, physics = _write_pair(tmp_path)
    text = physics.read_text().replace("part.obj", "missing.obj")
    physics.write_text(text, encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        project_and_verify(
            visual_sdf=visual,
            physics_sdf=physics,
            output_dir=tmp_path / "asset2",
            name="demo",
        )


def _write_gltf_visual_pair(tmp_path: Path):
    """Published visual is localized; its old authoring sibling is not."""

    asset = tmp_path / "assetdir"
    (asset / "visual").mkdir(parents=True)
    (asset / "blender_output" / "parts").mkdir(parents=True)
    sibling = asset / "blender_output" / "parts" / "part.obj"
    stale = trimesh.creation.box(extents=(.02, .02, .02))
    stale.apply_translation([0, .79, -.24])
    stale.export(sibling)
    declared = asset / "visual" / "part_visual.gltf"
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[:3, 3] = [.01, -.02, .03]
    scene.add_geometry(trimesh.creation.box(extents=(.02, .02, .02)), transform=transform)
    for name, content in scene.export(file_type="gltf").items():
        destination = declared if name.endswith(".gltf") else declared.parent / name
        destination.write_bytes(content)
    collision_mesh = asset / "part_collision.obj"
    collision_mesh.write_text(CUBE_OBJ, encoding="utf-8")

    def model(tag: str, uri: Path) -> str:
        return f"""\
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="demo">
    <link name="base_link">
      <inertial><mass>0.30</mass><pose>0 0 0 0 0 0</pose>
        <inertia><ixx>1e-4</ixx><iyy>1e-4</iyy><izz>1e-4</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <{tag}><pose>0 0 0 0 0 0</pose><geometry><mesh>
        <uri>{uri}</uri><scale>1 1 1</scale></mesh></geometry></{tag}>
    </link>
  </model>
</sdf>
"""
    visual = asset / "visual_locked.sdf"
    visual.write_text(model("visual", declared), encoding="utf-8")
    physics = asset / "physics_locked.sdf"
    physics.write_text(model("collision", collision_mesh), encoding="utf-8")
    return visual, physics


def test_projection_preserves_published_frame_instead_of_stale_sibling(tmp_path: Path) -> None:
    visual, physics = _write_gltf_visual_pair(tmp_path)
    output = tmp_path / "projected"
    # The visual is glTF/Y-up, but the physics SDF owns a native Z-up COM.
    # This is the production projection boundary: visual conversion must not
    # rotate already-authored physics mass properties.
    physics.write_text(physics.read_text().replace(
        '<pose>0 0 0 0 0 0</pose>\n        <inertia>',
        '<pose>0.001 0.002 0.03 0 0 0</pose>\n        <inertia>',
        1), encoding="utf-8")
    receipt = project_single_rigid(
        visual_sdf=visual,
        physics_sdf=physics,
        output_dir=output,
        name="demo",
    )
    assert receipt["visual_axis_contract"] == "gltf_y_up_to_sapien_z_up_rx90"
    import xml.etree.ElementTree as ET
    urdf = ET.parse(output / "mobility.urdf").getroot()
    assert [float(v) for v in urdf.find("link/inertial/origin").attrib["xyz"].split()] == pytest.approx([0.001, 0.002, 0.03])
    assert receipt["visual_elements_with_gltf_y_up_basis"] == 1
    import xml.etree.ElementTree as ET

    urdf = ET.parse(output / "mobility.urdf").getroot()
    visual_origin = urdf.find("link/visual/origin")
    assert visual_origin is not None
    roll, pitch, yaw = (float(v) for v in visual_origin.attrib["rpy"].split())
    assert roll == pytest.approx(1.5707963, abs=1e-6)
    assert pitch == pytest.approx(0.0, abs=1e-6)
    assert yaw == pytest.approx(0.0, abs=1e-6)
    mesh = trimesh.load(urdf.find("link/visual/geometry/mesh").get("filename"), force="mesh")
    assert np.allclose(mesh.bounds, [[0, -.03, .02], [.02, -.01, .04]], atol=1e-8)
    proof = receipt["geometry_fidelity"]["visuals"][0]
    assert proof["conversion"] == "declared scene graph to OBJ"
    assert proof["geometry_checks"][0]["max_triangle_corner_error_m"] < 1e-7
    assert not any("blender_output" in path for path in receipt["input_file_sha256"])


def test_projection_keeps_a_native_z_up_visual_unrotated(tmp_path: Path) -> None:
    visual, physics = _write_pair(tmp_path)  # declares an .obj directly
    output = tmp_path / "projected_native"
    receipt = project_single_rigid(
        visual_sdf=visual,
        physics_sdf=physics,
        output_dir=output,
        name="demo",
    )
    assert receipt["visual_elements_with_gltf_y_up_basis"] == 0
    import xml.etree.ElementTree as ET

    urdf = ET.parse(output / "mobility.urdf").getroot()
    roll, pitch, yaw = (
        float(v) for v in urdf.find("link/visual/origin").attrib["rpy"].split()
    )
    assert (roll, pitch, yaw) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)


def test_broken_declared_mesh_cannot_fall_back_to_an_authoring_sibling(tmp_path):
    visual, physics = _write_gltf_visual_pair(tmp_path)
    (visual.parent / "visual/part_visual.gltf").write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        project_single_rigid(visual_sdf=visual, physics_sdf=physics,
                             output_dir=tmp_path / "broken", name="demo")


def test_relative_frame_chain_and_default_mesh_scale_are_preserved(tmp_path):
    visual, _ = _write_pair(tmp_path)
    tree = ET.parse(visual)
    model = tree.getroot().find("model")
    base = model.find("link[@name='base_link']")
    ET.SubElement(base, "pose").text = "1 2 3 0 0 1.5707963267948966"
    lid = model.find("link[@name='lid']/pose")
    lid.set("relative_to", "base_link")
    lid.text = "1 0 0 0 0 0"
    for mesh in model.findall("link/visual/geometry/mesh"):
        mesh.remove(mesh.find("scale"))
    tree.write(visual)
    links = parse_locked_sdf(visual)
    assert np.allclose(links[1]["pose"][:3, 3], [1, 3, 3])
    assert links[1]["visuals"][0]["scale"] == [1, 1, 1]


def test_nonfixed_joint_cannot_be_silently_frozen(tmp_path):
    visual, _ = _write_pair(tmp_path)
    visual.write_text(visual.read_text().replace('type="fixed"', 'type="revolute"'))
    with pytest.raises(ValueError, match="every source joint"):
        parse_locked_sdf(visual)


def test_cyclic_sdf_frames_are_rejected(tmp_path):
    visual, _ = _write_pair(tmp_path)
    tree = ET.parse(visual)
    base = tree.getroot().find("model/link[@name='base_link']")
    ET.SubElement(base, "pose", relative_to="lid").text = "0 0 0 0 0 0"
    tree.getroot().find("model/link[@name='lid']/pose").set("relative_to", "base_link")
    tree.write(visual)
    with pytest.raises(ValueError, match="cyclic or unresolved"):
        parse_locked_sdf(visual)


def test_projection_fidelity_and_dependency_binding_are_required(tmp_path):
    visual, physics = _write_gltf_visual_pair(tmp_path)
    output = tmp_path / "projection"
    receipt = project_and_verify(visual_sdf=visual, physics_sdf=physics,
                                 output_dir=output, name="demo")
    path = output / "projection_receipt.json"
    assert validate_projection_receipt(path)["passed"]
    native = next(output.glob("visual_meshes/*/*/mesh.obj"))
    native.write_text(native.read_text() + "\nv 0 0 9\n")
    with pytest.raises(ValueError, match="dependency changed"):
        validate_projection_receipt(path)
    receipt.pop("geometry_fidelity")
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="no verified"):
        validate_projection_receipt(path)
