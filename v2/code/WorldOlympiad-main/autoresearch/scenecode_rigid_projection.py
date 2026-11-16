"""Project a fixed-joint multi-link SceneCode model into one rigid native body.

SceneCode emits a manipuland as an SDF; after ``sdf_rigid_joint_lock`` every
declared joint is ``fixed``, but the model still declares several links.  The
RoboTwin native planner can only grasp *one* rigid body, and a fixed-joint
articulation is not a single rigid body (PyBullet merges fixed joints on load,
SAPIEN does not).  This module performs the missing deterministic projection:

* visuals are taken from the derived *visual* SDF, collisions from the derived
  *physics* SDF, and every element is composed into the model frame through its
  link pose;
* the per-link inertials are combined into one rigid-body inertial with the
  parallel-axis theorem;
* the result is a single-link ``mobility.urdf`` plus a RoboTwin ``model_data``
  and a receipt that records source digests, mesh inventories and a PyBullet
  load check (one body, zero joints, expected collision count and mass).

Nothing here invents geometry: every mesh and every inertial value comes from
the derived SDFs.  A projection without a passed PyBullet check is reported as
``passed: false`` and must not be consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from html import escape

import numpy as np
from scipy.spatial.transform import Rotation

from .scenecode_mesh_projection import (
    convert_declared_visual, mesh_dependencies, _scene_meshes,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _floats(text: str | None, count: int, default: float = 0.0) -> list[float]:
    values = [default] * count if text is None else [float(value) for value in text.split()]
    if len(values) != count or not np.isfinite(values).all():
        raise ValueError(f"expected {count} finite SDF values")
    return values


def _pose_matrix(text: str | None) -> np.ndarray:
    x, y, z, roll, pitch, yaw = _floats(text, 6)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    matrix[:3, 3] = [x, y, z]
    return matrix


def _rpy_xyz(matrix: np.ndarray) -> tuple[list[float], list[float]]:
    xyz = matrix[:3, 3].tolist()
    rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz").tolist()
    return xyz, rpy


def parse_locked_sdf(path: Path, *, allow_primitives: bool = False) -> list[dict]:
    """Read one locked (all-fixed) SDF model without merging anything."""

    root = ET.parse(path).getroot()
    model = root.find("model")
    if model is None:
        raise ValueError(f"SDF has no <model>: {path}")
    if model.findall("model"):
        raise ValueError("nested SDF models need explicit flattening before rigid projection")
    if any(joint.get("type") != "fixed" for joint in model.findall("joint")):
        raise ValueError("rigid projection requires every source joint to be fixed")
    frames = {}
    for element in [*model.findall("link"), *model.findall("frame"), *model.findall("joint")]:
        name = element.get("name")
        if not name or name in frames:
            raise ValueError("SDF projection frames must have unique names")
        pose = element.find("pose")
        default_frame = element.findtext("child") if element.tag == "joint" else \
            element.get("attached_to", "__model__") if element.tag == "frame" else "__model__"
        frames[name] = ((pose.get("relative_to") or default_frame) if pose is not None else default_frame,
                        _pose_matrix(None if pose is None else pose.text))
    resolved = {"__model__": np.eye(4), "world": np.linalg.inv(_pose_matrix(model.findtext("pose")))}
    active = set()
    def resolve(name):
        name = name.removeprefix(f"{model.get('name')}::")
        if name in resolved:
            return resolved[name]
        if name in active or name not in frames:
            raise ValueError(f"cyclic or unresolved SDF frame: {name}")
        active.add(name)
        parent, local = frames[name]
        resolved[name] = resolve(parent) @ local
        active.remove(name)
        return resolved[name]
    def local_pose(element, link_name):
        pose = element.find("pose")
        parent = (pose.get("relative_to") or link_name) if pose is not None else link_name
        return np.linalg.inv(resolve(link_name)) @ resolve(parent) @ _pose_matrix(None if pose is None else pose.text)
    def mesh_record(element, link_name):
        geometry = element.find("geometry")
        mesh_node = None if geometry is None else geometry.find("mesh")
        if mesh_node is not None:
            uri = mesh_node.findtext("uri")
            if not uri or "://" in uri:
                raise ValueError("rigid projection requires resolved local mesh URIs")
            mesh = Path(uri)
            if not mesh.is_absolute():
                mesh = Path(path).parent / mesh
            scale = _floats(mesh_node.findtext("scale"), 3, 1.0)
            if min(scale) <= 0:
                raise ValueError("SDF mesh scale must be positive")
            return {"name": element.get("name"), "kind": "mesh",
                    "uri": str(mesh.resolve()), "scale": scale,
                    "pose": local_pose(element, link_name)}
        if not allow_primitives or geometry is None:
            raise ValueError("rigid projection requires resolved local mesh URIs")
        for kind in ("box", "cylinder", "sphere"):
            primitive = geometry.find(kind)
            if primitive is None:
                continue
            values = {
                key: _floats(primitive.findtext(key), count)
                for key, count in {
                    "size": 3,
                    "radius": 1,
                    "length": 1,
                }.items()
                if primitive.find(key) is not None
            }
            return {"name": element.get("name"), "kind": kind,
                    **values, "pose": local_pose(element, link_name)}
        raise ValueError("unsupported SDF geometry in native rigid scene")
    links: list[dict] = []
    for link in model.findall("link"):
        inertial = link.find("inertial")
        inertia = [0.0] * 6
        mass = 0.0
        com = np.eye(4)
        if inertial is not None:
            mass = float(inertial.findtext("mass") or 0.0)
            com = local_pose(inertial, link.get("name"))
            block = inertial.find("inertia")
            if block is not None:
                inertia = [
                    float(block.findtext(key) or 0.0)
                    for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                ]
        visuals, collisions = [], []
        for element in link.findall("visual"):
            visuals.append(mesh_record(element, link.get("name")))
        for element in link.findall("collision"):
            collisions.append(mesh_record(element, link.get("name")))
        links.append(
            {
                "name": link.get("name"),
                "pose": resolve(link.get("name")),
                "mass": mass,
                "com": com,
                "inertia": inertia,
                "visuals": visuals,
                "collisions": collisions,
            }
        )
    return links


def _rotate_inertial(inertial: dict, rotation: np.ndarray) -> dict:
    """Express a combined inertial in a new rigid-link frame."""
    r = np.asarray(rotation, dtype=float)[:3, :3]
    com = r @ np.asarray(inertial["com"], dtype=float)
    matrix = np.array([[inertial["ixx"], inertial["ixy"], inertial["ixz"]],
                       [inertial["ixy"], inertial["iyy"], inertial["iyz"]],
                       [inertial["ixz"], inertial["iyz"], inertial["izz"]]])
    matrix = r @ matrix @ r.T
    return {**inertial, "com": com.tolist(),
            "ixx": float(matrix[0, 0]), "iyy": float(matrix[1, 1]),
            "izz": float(matrix[2, 2]), "ixy": float(matrix[0, 1]),
            "ixz": float(matrix[0, 2]), "iyz": float(matrix[1, 2])}


def _combine_inertial(links: list[dict]) -> dict:
    """Combine per-link rigid inertials into one body inertial."""

    total_mass = float(sum(link["mass"] for link in links))
    if total_mass <= 0:
        raise ValueError("projected model has no positive mass")
    weighted = np.zeros(3)
    for link in links:
        com = (link["pose"] @ link["com"])[:3, 3]
        weighted += link["mass"] * com
    body_com = weighted / total_mass

    inertia = np.zeros((3, 3))
    for link in links:
        local = np.array(
            [
                [link["inertia"][0], link["inertia"][3], link["inertia"][4]],
                [link["inertia"][3], link["inertia"][1], link["inertia"][5]],
                [link["inertia"][4], link["inertia"][5], link["inertia"][2]],
            ]
        )
        rotation = (link["pose"] @ link["com"])[:3, :3]
        com = (link["pose"] @ link["com"])[:3, 3]
        offset = com - body_com
        inertia += rotation @ local @ rotation.T
        inertia += link["mass"] * (
            float(offset @ offset) * np.eye(3) - np.outer(offset, offset)
        )
    return {
        "mass": total_mass,
        "com": body_com.tolist(),
        "ixx": float(inertia[0, 0]),
        "iyy": float(inertia[1, 1]),
        "izz": float(inertia[2, 2]),
        "ixy": float(inertia[0, 1]),
        "ixz": float(inertia[0, 2]),
        "iyz": float(inertia[1, 2]),
    }


#: Published SceneCode glTF visuals are Y-up; collision OBJs are Z-up.
#: Conversion preserves the declared scene's coordinates, and the element pose
#: applies this basis exactly once. Authoring siblings are never substituted.
_GLTF_Y_UP_TO_Z_UP_BASIS = np.eye(4)
_GLTF_Y_UP_TO_Z_UP_BASIS[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()


def _mesh_element(tag: str, uri: str, scale: list[float], pose: np.ndarray) -> str:
    xyz, rpy = _rpy_xyz(pose)
    xyz_text = " ".join(f"{value:.9f}" for value in xyz)
    rpy_text = " ".join(f"{value:.9f}" for value in rpy)
    scale_text = " ".join(f"{value:.9f}" for value in scale)
    return (
        f"    <{tag}>\n"
        f"      <origin xyz=\"{xyz_text}\" rpy=\"{rpy_text}\"/>\n"
        f"      <geometry>\n"
        f"        <mesh filename=\"{escape(uri, quote=True)}\" scale=\"{scale_text}\"/>\n"
        f"      </geometry>\n"
        f"    </{tag}>\n"
    )


def project_single_rigid(
    *,
    visual_sdf: Path,
    physics_sdf: Path,
    output_dir: Path,
    name: str,
    link_name: str = "base_link",
) -> dict:
    """Emit a single-link URDF for one locked multi-link model."""

    visual_sdf, physics_sdf, output_dir = Path(visual_sdf).resolve(), Path(physics_sdf).resolve(), Path(output_dir).resolve()
    visual_links = parse_locked_sdf(visual_sdf)
    physics_links = parse_locked_sdf(physics_sdf)
    if not visual_links or not physics_links:
        raise ValueError("projection needs both a visual and a physics SDF")

    visual_elements = []
    conversion_checks = []
    input_digests = {str(path): sha256_file(path) for path in (visual_sdf, physics_sdf)}
    output_digests = {}
    gltf_basis_applied = 0
    for link in visual_links:
        for element in link["visuals"]:
            converted, proof = convert_declared_visual(element["uri"], output_dir=output_dir,
                                                       scale=np.asarray(element["scale"]))
            conversion_checks.append({"link": link["name"], "visual": element["name"], **proof})
            input_digests.update(proof["input_file_sha256"])
            output_digests.update(proof["output_file_sha256"])
            pose = link["pose"] @ element["pose"]
            gltf_y_up_basis = converted[0]["y_up"]
            if gltf_y_up_basis:
                pose = pose @ _GLTF_Y_UP_TO_Z_UP_BASIS
                gltf_basis_applied += 1
            for resolved in converted:
                visual_elements.append(
                {
                    **element,
                    "uri": resolved["uri"],
                    "declared_uri": element["uri"],
                    "pose": pose,
                    "link": link["name"],
                    "gltf_y_up_basis_applied": gltf_y_up_basis,
                }
            )
    collision_elements = []
    for link in physics_links:
        for element in link["collisions"]:
            input_digests.update(mesh_dependencies(Path(element["uri"])))
            collision_elements.append(
                {
                    **element,
                    "pose": link["pose"] @ element["pose"],
                    "link": link["name"],
                }
            )
    for element in visual_elements + collision_elements:
        if not element["uri"] or not Path(element["uri"]).is_file():
            raise FileNotFoundError(
                f"projected mesh does not resolve: {element.get('uri')!r}"
            )
    if not visual_elements or not collision_elements:
        raise ValueError("projection needs at least one visual and one collision mesh")

    # Mass properties belong to the physics SDF's native link frame.  A glTF
    # visual is converted only as geometry; its Y-up basis must not reinterpret
    # an already-authored physics COM/inertia (including when visual_sdf and
    # physics_sdf happen to be the same file).  Raw mesh-derived mass data must
    # be converted by the authoring exporter before it is serialized into SDF.
    inertial = _combine_inertial(physics_links)
    com_pose = np.eye(4)
    com_pose[:3, 3] = inertial["com"]
    com_xyz, com_rpy = _rpy_xyz(com_pose)

    lines = [f'<?xml version="1.0" ?>', f'<robot name="{name}">', f'  <link name="{link_name}">']
    lines.append("    <inertial>")
    lines.append("      <origin xyz=\"%s\" rpy=\"%s\"/>" % (
        " ".join(f"{v:.9f}" for v in com_xyz),
        " ".join(f"{v:.9f}" for v in com_rpy),
    ))
    lines.append("      <mass value=\"%.9f\"/>" % inertial["mass"])
    lines.append(
        "      <inertia ixx=\"%.9e\" ixy=\"%.9e\" ixz=\"%.9e\" iyy=\"%.9e\" iyz=\"%.9e\" izz=\"%.9e\"/>"
        % (
            inertial["ixx"], inertial["ixy"], inertial["ixz"],
            inertial["iyy"], inertial["iyz"], inertial["izz"],
        )
    )
    lines.append("    </inertial>")
    for element in visual_elements:
        lines.append(_mesh_element("visual", element["uri"], element["scale"], element["pose"]).rstrip("\n"))
    for element in collision_elements:
        lines.append(_mesh_element("collision", element["uri"], element["scale"], element["pose"]).rstrip("\n"))
    lines.append("  </link>")
    lines.append("</robot>")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = output_dir / "mobility.urdf"
    urdf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    extent_points = []
    for element in visual_elements + collision_elements:
        for _, mesh in _scene_meshes(Path(element["uri"])):
            points = np.asarray(mesh.vertices) * element["scale"]
            posed = points @ element["pose"][:3, :3].T + element["pose"][:3, 3]
            extent_points.extend([posed.min(axis=0), posed.max(axis=0)])
    corners = np.array(extent_points)
    extents = (corners.max(axis=0) - corners.min(axis=0)).tolist()

    receipt = {
        "schema_version": "scenecode_rigid_projection.v1",
        "name": name,
        "link_name": link_name,
        "visual_sdf": str(visual_sdf),
        "visual_sdf_sha256": sha256_file(Path(visual_sdf)),
        "physics_sdf": str(physics_sdf),
        "physics_sdf_sha256": sha256_file(Path(physics_sdf)),
        "urdf": str(urdf_path),
        "urdf_sha256": sha256_file(urdf_path),
        "source_link_count": {
            "visual": len(visual_links),
            "physics": len(physics_links),
        },
        "mesh_count": {
            "visual": len(visual_elements),
            "collision": len(collision_elements),
        },
        "visual_axis_contract": "gltf_y_up_to_sapien_z_up_rx90",
        "visual_elements_with_gltf_y_up_basis": gltf_basis_applied,
        "combined_inertial": inertial,
        "declared_extents_m": extents,
        "model_bounds_m": [corners.min(axis=0).tolist(), corners.max(axis=0).tolist()],
        "input_file_sha256": input_digests,
        "output_file_sha256": output_digests,
        "geometry_fidelity": {"method": "declared_source_mesh_scene_graph",
                              "passed": all(check["passed"] for check in conversion_checks),
                              "visuals": conversion_checks},
        "passed": False,
        "verification": None,
    }
    return receipt


def verify_pybullet(urdf_path: Path, receipt: dict) -> dict:
    """Load the projected URDF in PyBullet and check the single-body invariant."""

    import pybullet as p

    client = p.connect(p.DIRECT)
    try:
        body = p.loadURDF(str(urdf_path), useFixedBase=False, flags=p.URDF_USE_INERTIA_FROM_FILE)
        joints = p.getNumJoints(body)
        collisions = p.getCollisionShapeData(body, -1)
        dynamics = p.getDynamicsInfo(body, -1)
        mass = float(dynamics[0])
        result = {
            "body_count": 1,
            "num_joints": int(joints),
            "base_collision_shapes": len(collisions),
            "base_mass_kg": mass,
            "passed": bool(
                joints == 0
                and len(collisions) == receipt["mesh_count"]["collision"]
                and abs(mass - receipt["combined_inertial"]["mass"]) < 1e-6
            ),
        }
    finally:
        p.disconnect(client)
    return result


def project_and_verify(
    *,
    visual_sdf: Path,
    physics_sdf: Path,
    output_dir: Path,
    name: str,
    link_name: str = "base_link",
) -> dict:
    receipt = project_single_rigid(
        visual_sdf=visual_sdf,
        physics_sdf=physics_sdf,
        output_dir=output_dir,
        name=name,
        link_name=link_name,
    )
    receipt["verification"] = verify_pybullet(Path(receipt["urdf"]), receipt)
    receipt["passed"] = bool(receipt["verification"]["passed"] and receipt["geometry_fidelity"]["passed"])
    (Path(output_dir) / "projection_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-sdf", type=Path, required=True)
    parser.add_argument("--physics-sdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--link-name", default="base_link")
    args = parser.parse_args()
    receipt = project_and_verify(
        visual_sdf=args.visual_sdf,
        physics_sdf=args.physics_sdf,
        output_dir=args.output_dir,
        name=args.name,
        link_name=args.link_name,
    )
    print(json.dumps({"passed": receipt["passed"], "verification": receipt["verification"]}, indent=2))


if __name__ == "__main__":
    main()
