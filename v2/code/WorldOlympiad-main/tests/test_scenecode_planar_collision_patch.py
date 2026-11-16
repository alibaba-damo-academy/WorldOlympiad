import ast
import logging
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest

PATCH = Path(__file__).resolve().parents[1] / "deployment/ziplab5090/patches/scenecode_task_planar_collision_origin.patch"
UPSTREAM = 'def generate_thin_covering_sdf(\n    visual_mesh_path: Path,\n    output_path: Path,\n    model_name: str | None = None,\n    collision_dims: tuple[float, float, float] | None = None,\n    collision_shape: str = "rectangular",\n) -> Path:\n    """Generate Drake SDF file for a static thin covering.\n\n    For floor/manipuland thin coverings (rugs, tablecloths), no collision\n    geometry is included - they are purely decorative. For wall thin coverings\n    (paintings, posters), collision geometry is added so Drake can detect\n    furniture collisions.\n\n    Args:\n        visual_mesh_path: Path to the thin covering GLTF mesh file.\n        output_path: Path where SDF file will be saved.\n        model_name: Optional name for the model (defaults to mesh stem).\n        collision_dims: Optional (width, depth, height) for collision primitive.\n            If provided, adds collision geometry. Used for wall coverings.\n        collision_shape: Shape of collision primitive ("rectangular" or "circular").\n            Only used when collision_dims is provided.\n\n    Returns:\n        Path to the generated SDF file.\n\n    Raises:\n        FileNotFoundError: If visual mesh file doesn\'t exist.\n    """\n    if not visual_mesh_path.exists():\n        raise FileNotFoundError(f"Visual mesh not found: {visual_mesh_path}")\n\n    model_name = model_name or visual_mesh_path.stem\n\n    console_logger.debug(f"Generating thin covering SDF for \'{model_name}\'")\n\n    # Create SDF XML structure.\n    sdf = ET.Element("sdf", version="1.7")\n    model = ET.SubElement(sdf, "model", name=model_name)\n\n    # Add single link.\n    link = ET.SubElement(model, "link", name="base_link")\n\n    # Visual geometry (external mesh reference).\n    visual = ET.SubElement(link, "visual", name="visual")\n    visual_geom = ET.SubElement(visual, "geometry")\n    visual_mesh_elem = ET.SubElement(visual_geom, "mesh")\n\n    # Use relative URI for mesh (assumes mesh is in same directory as SDF).\n    mesh_filename = visual_mesh_path.name\n    visual_uri = ET.SubElement(visual_mesh_elem, "uri")\n    visual_uri.text = mesh_filename\n\n    # Add collision geometry if dimensions provided (for wall thin coverings).\n    # Wall covering visual mesh bounds (from create_rectangular_thin_covering_glb):\n    #   X: -width/2 to +width/2 (centered)\n    #   Y: 0 to thickness (back at Y=0 against wall)\n    #   Z: 0 to height (bottom at Z=0)\n    # Collision primitive must be offset to match these bounds.\n    if collision_dims is not None:\n        width, depth, height = collision_dims\n        collision = ET.SubElement(link, "collision", name="collision")\n\n        # Offset collision to align with visual mesh.\n        # Visual mesh: Y from 0 to depth, Z from 0 to height.\n        # Collision primitive is centered, so offset by (depth/2, height/2).\n        collision_pose = ET.SubElement(collision, "pose")\n\n        collision_geom = ET.SubElement(collision, "geometry")\n\n        if collision_shape == "circular":\n            # Cylinder for circular wall coverings (round paintings/mirrors).\n            # Default SDF cylinder axis is Z. For wall covering, axis should be Y\n            # (thin direction perpendicular to wall). Rotate 90° around X.\n            roll = math.pi / 2  # Rotate Z-axis to Y-axis.\n            collision_pose.text = f"0 {depth / 2:.6f} {height / 2:.6f} {roll:.6f} 0 0"\n\n            cylinder = ET.SubElement(collision_geom, "cylinder")\n            radius_elem = ET.SubElement(cylinder, "radius")\n            radius_elem.text = f"{width / 2:.6f}"\n            length_elem = ET.SubElement(cylinder, "length")\n            length_elem.text = f"{depth:.6f}"  # Thickness along rotated axis.\n            console_logger.debug(\n                f"Added cylinder collision: r={width / 2:.3f}m, len={depth:.3f}m"\n            )\n        else:\n            # Box for rectangular wall coverings.\n            collision_pose.text = f"0 {depth / 2:.6f} {height / 2:.6f} 0 0 0"\n\n            box = ET.SubElement(collision_geom, "box")\n            size_elem = ET.SubElement(box, "size")\n            size_elem.text = f"{width:.6f} {depth:.6f} {height:.6f}"\n            console_logger.debug(\n                f"Added box collision: {width:.3f}m x {depth:.3f}m x {height:.3f}m"\n            )\n\n    # Format XML with indentation.\n    ET.indent(sdf, space="  ", level=0)\n\n    # Create ElementTree and write to file.\n    tree = ET.ElementTree(sdf)\n\n    # Ensure output directory exists.\n    output_path.parent.mkdir(parents=True, exist_ok=True)\n\n    # Write with XML declaration.\n    tree.write(output_path, encoding="utf-8", xml_declaration=True)\n\n    console_logger.info(f"Generated thin covering SDF: {output_path}")\n\n    return output_path\n'


@pytest.fixture
def builder(tmp_path):
    source = tmp_path / "scenecode/agent_utils/thin_covering_generator.py"
    source.parent.mkdir(parents=True)
    source.write_text(UPSTREAM)
    whole = PATCH.read_text()
    start = whole.index("--- a/scenecode/agent_utils/thin_covering_generator.py")
    patch = tmp_path / "generator.patch"; patch.write_text(whole[start:])
    subprocess.run(["patch", "--batch", "--forward", "-p1", "-d", str(tmp_path), "-i", str(patch)],
                   check=True, capture_output=True, text=True)
    namespace = {"Path": Path, "ET": ET, "math": math, "console_logger": logging.getLogger(__name__)}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    return namespace["generate_thin_covering_sdf"]


@pytest.mark.parametrize("shape", ["rectangular", "circular"])
@pytest.mark.parametrize("wall", [False, True])
def test_collision_origin_and_axis_follow_surface_orientation(builder, tmp_path, shape, wall):
    visual = tmp_path / "surface.gltf"; visual.write_text("unchanged visual")
    dims = (.3, .002, .22) if wall else (.3, .22, .002)
    sdf = builder(visual, tmp_path / "surface.sdf", collision_dims=dims, collision_shape=shape, is_wall=wall)
    root = ET.parse(sdf).getroot(); collision = root.find("model/link/collision")
    pose = np.fromstring(collision.findtext("pose"), sep=" ")
    np.testing.assert_allclose(pose[:3], [0, dims[1]/2 if wall else 0, dims[2]/2], atol=1e-6)
    if shape == "circular":
        assert pose[3] == pytest.approx(math.pi/2 if wall else 0, abs=1e-6)
        assert float(collision.findtext("geometry/cylinder/length")) == dims[1 if wall else 2]
        assert float(collision.findtext("geometry/cylinder/radius")) == pytest.approx(.15 if wall else .11)
    else:
        np.testing.assert_allclose(np.fromstring(collision.findtext("geometry/box/size"), sep=" "), dims)
    assert visual.read_text() == "unchanged visual"
    assert root.find("model/link/inertial") is None  # Existing mass protocol stays unchanged.


@pytest.mark.parametrize("shape", ["rectangular", "circular"])
def test_legacy_default_wall_output_is_unchanged(builder, tmp_path, shape):
    visual = tmp_path / "surface.gltf"; visual.touch()
    a = builder(visual, tmp_path / "a.sdf", collision_dims=(.3,.002,.22), collision_shape=shape)
    b = builder(visual, tmp_path / "b.sdf", collision_dims=(.3,.002,.22), collision_shape=shape, is_wall=True)
    assert a.read_bytes() == b.read_bytes()


def test_decorative_covering_still_has_no_collision(builder, tmp_path):
    visual = tmp_path / "surface.gltf"; visual.touch()
    sdf = builder(visual, tmp_path / "a.sdf", is_wall=False)
    assert ET.parse(sdf).find("model/link/collision") is None
