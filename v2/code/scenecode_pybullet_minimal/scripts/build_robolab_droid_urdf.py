#!/usr/bin/env python3
"""Build a portable Franka+Robotiq URDF matching RoboLab's DROID asset.

RoboLab publishes its benchmark robot as a flattened USD.  PyBullet and
Drake's URDF path cannot consume that articulation directly, so this builder
combines the bundled Franka arm with a pinned PickNik Robotiq 2F-85
description.  The fixed mount and public joint names are taken from the
pinned RoboLab USD; the generated manifest records every source and transform.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET


ROBOLAB_COMMIT = "f72d39445ab699c9cfd9bf551e8d21b6075293b9"
ROBOLAB_USD_SHA256 = "f555695465687548a1bd31b5e3f30385182d476a67c17080b7820ad0ef747e41"
PICKNIK_COMMIT = "2c047340aeb2440f7a60e429264221aab9658707"
XACRO = "http://wiki.ros.org/xacro"

REMOVED_PANDA_LINKS = {
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
    "panda_grasptarget",
}
REMOVED_PANDA_JOINTS = {
    "panda_hand_joint",
    "panda_finger_joint1",
    "panda_finger_joint2",
    "panda_grasptarget_hand",
}
NAME_MAP = {
    "robotiq_85_base_link": "base_link",
    "robotiq_85_left_knuckle_link": "left_outer_knuckle",
    "robotiq_85_right_knuckle_link": "right_outer_knuckle",
    "robotiq_85_left_finger_link": "left_outer_finger",
    "robotiq_85_right_finger_link": "right_outer_finger",
    "robotiq_85_left_inner_knuckle_link": "left_inner_knuckle",
    "robotiq_85_right_inner_knuckle_link": "right_inner_knuckle",
    "robotiq_85_left_finger_tip_link": "left_inner_finger",
    "robotiq_85_right_finger_tip_link": "right_inner_finger",
    "robotiq_85_base_joint": "panda_hand_joint",
    "robotiq_85_left_knuckle_joint": "finger_joint",
    "robotiq_85_right_knuckle_joint": "right_outer_knuckle_joint",
    "robotiq_85_left_finger_joint": "left_outer_finger_fixed_joint",
    "robotiq_85_right_finger_joint": "right_outer_finger_fixed_joint",
    "robotiq_85_left_inner_knuckle_joint": "left_inner_finger_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint": "right_inner_finger_knuckle_joint",
    "robotiq_85_left_finger_tip_joint": "left_inner_finger_joint",
    "robotiq_85_right_finger_tip_joint": "right_inner_finger_joint",
}
MIMIC_MULTIPLIERS = {
    "finger_joint": 1.0,
    "right_outer_knuckle_joint": -1.0,
    "left_inner_finger_knuckle_joint": 1.0,
    "right_inner_finger_knuckle_joint": -1.0,
    "left_inner_finger_joint": -1.0,
    "right_inner_finger_joint": 1.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _substitute(node: ET.Element) -> None:
    for key, value in list(node.attrib.items()):
        value = value.replace("${prefix}", "").replace("${parent}", "panda_link8")
        node.set(key, NAME_MAP.get(value, value))
    if node.text:
        node.text = node.text.replace("${prefix}", "").replace(
            "${parent}", "panda_link8"
        )
    for child in node:
        _substitute(child)


def _remove_descendants(node: ET.Element, local_name: str) -> None:
    for parent in node.iter():
        for child in list(parent):
            if _local_name(child.tag) == local_name:
                parent.remove(child)


def _stl_to_obj(source: Path, destination: Path) -> None:
    """Convert one triangular STL to deterministic OBJ without changing geometry."""

    data = source.read_bytes()
    triangles: list[tuple[tuple[float, float, float], ...]] = []
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if 84 + triangle_count * 50 == len(data):
            for index in range(triangle_count):
                values = struct.unpack_from("<12fH", data, 84 + index * 50)
                triangles.append(
                    (
                        (values[3], values[4], values[5]),
                        (values[6], values[7], values[8]),
                        (values[9], values[10], values[11]),
                    )
                )
    if not triangles:
        vertices = []
        for line in data.decode("ascii").splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                vertices.append(tuple(float(value) for value in parts[1:]))
        if not vertices or len(vertices) % 3:
            raise ValueError(f"unsupported or malformed STL: {source}")
        triangles = [tuple(vertices[index : index + 3]) for index in range(0, len(vertices), 3)]

    unique_vertices: dict[tuple[float, float, float], int] = {}
    faces: list[tuple[int, int, int]] = []
    for triangle in triangles:
        face = []
        for vertex in triangle:
            if vertex not in unique_vertices:
                unique_vertices[vertex] = len(unique_vertices) + 1
            face.append(unique_vertices[vertex])
        faces.append(tuple(face))
    lines = ["# Deterministic conversion of " + str(source)]
    lines.extend(
        "v " + " ".join(f"{value:.9g}" for value in vertex)
        for vertex in unique_vertices
    )
    lines.extend("f " + " ".join(str(index) for index in face) for face in faces)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="ascii")


def _convert_collada_meshes(
    jobs: dict[Path, Path],
    blender_executable: Path,
    converter_script: Path,
) -> list[dict]:
    if not jobs:
        return []
    if not blender_executable.is_file():
        raise FileNotFoundError(f"Blender executable is missing: {blender_executable}")
    if not converter_script.is_file():
        raise FileNotFoundError(f"COLLADA converter is missing: {converter_script}")
    command = [
        str(blender_executable),
        "-b",
        "--python",
        str(converter_script),
        "--",
    ]
    for source, destination in sorted(jobs.items()):
        command.extend((str(source), str(destination)))
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    marker = "ROBOLAB_COLLADA_CONVERSION="
    records = [
        json.loads(line.removeprefix(marker))
        for line in completed.stdout.splitlines()
        if line.startswith(marker)
    ]
    if len(records) != 1 or len(records[0]) != len(jobs):
        raise RuntimeError(
            "Blender COLLADA conversion did not publish the expected audit records"
        )
    for record in records[0]:
        destination = Path(record["destination"])
        if not destination.is_file() or destination.suffix.lower() != ".obj":
            raise RuntimeError(f"Blender did not produce a valid OBJ: {destination}")
        if _sha256(destination) != record["destination_sha256"]:
            raise RuntimeError(f"OBJ changed after Blender conversion: {destination}")
    return records[0]


def _rewrite_meshes(
    root: ET.Element,
    panda_urdf: Path,
    robotiq_repo: Path,
    output: Path,
    blender_executable: Path,
    converter_script: Path,
) -> tuple[list[Path], list[Path], list[dict]]:
    sources: list[Path] = []
    generated: list[Path] = []
    collada_jobs: dict[Path, Path] = {}
    panda_mesh_root = panda_urdf.parent / "meshes"
    robotiq_root = robotiq_repo / "robotiq_description"
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename") or ""
        if filename.startswith("package://meshes/"):
            source = panda_mesh_root / filename.removeprefix("package://meshes/")
        elif filename.startswith("package://robotiq_description/"):
            source = robotiq_root / filename.removeprefix(
                "package://robotiq_description/"
            )
        else:
            source = Path(filename)
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"URDF mesh is missing: {source}")
        sources.append(source)
        if source.suffix.lower() == ".stl":
            destination = output / "collision_meshes" / f"{source.stem}.obj"
            _stl_to_obj(source, destination)
            mesh.set("filename", str(destination))
            generated.append(destination)
        elif source.suffix.lower() == ".dae":
            destination = output / "visual_meshes" / f"{source.stem}.obj"
            previous = collada_jobs.setdefault(source, destination)
            if previous != destination:
                raise ValueError(f"COLLADA destination collision for {source}")
            mesh.set("filename", str(destination))
        else:
            mesh.set("filename", str(source))
    visual_records = _convert_collada_meshes(
        collada_jobs, blender_executable, converter_script
    )
    return sorted(set(sources)), sorted(set(generated)), visual_records


def _validate_names(root: ET.Element) -> None:
    links = [node.get("name") for node in root.findall("link")]
    joints = [node.get("name") for node in root.findall("joint")]
    if None in links or len(links) != len(set(links)):
        raise ValueError("generated URDF has missing or duplicate link names")
    if None in joints or len(joints) != len(set(joints)):
        raise ValueError("generated URDF has missing or duplicate joint names")
    required_links = {
        "panda_link0",
        "panda_link8",
        "base_link",
        "left_inner_finger",
        "right_inner_finger",
    }
    required_joints = {f"panda_joint{index}" for index in range(1, 8)} | set(
        MIMIC_MULTIPLIERS
    )
    if missing := required_links - set(links):
        raise ValueError(f"generated URDF is missing links: {sorted(missing)}")
    if missing := required_joints - set(joints):
        raise ValueError(f"generated URDF is missing joints: {sorted(missing)}")
    if any(node.tag.startswith(f"{{{XACRO}}}") for node in root.iter()):
        raise ValueError("generated URDF still contains xacro nodes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panda-urdf", type=Path, required=True)
    parser.add_argument("--picknik-repo", type=Path, required=True)
    parser.add_argument("--robolab-usd", type=Path, required=True)
    parser.add_argument("--blender-executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    panda_urdf = args.panda_urdf.resolve()
    picknik_repo = args.picknik_repo.resolve()
    robolab_usd = args.robolab_usd.resolve()
    blender_executable = args.blender_executable.resolve()
    converter_script = Path(__file__).with_name(
        "convert_collada_to_obj_blender.py"
    ).resolve()
    output = args.output_dir.resolve()
    for path in (panda_urdf, robolab_usd):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not picknik_repo.is_dir():
        raise FileNotFoundError(picknik_repo)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite robot asset: {output}")
    if _git_revision(picknik_repo) != PICKNIK_COMMIT:
        raise ValueError(f"PickNik source must be pinned to {PICKNIK_COMMIT}")
    if _sha256(robolab_usd) != ROBOLAB_USD_SHA256:
        raise ValueError("RoboLab DROID USD differs from its pinned Git-LFS object")

    panda_tree = ET.parse(panda_urdf)
    robot = panda_tree.getroot()
    if robot.tag != "robot":
        raise ValueError("Franka source is not a URDF robot")
    for child in list(robot):
        name = child.get("name")
        if (child.tag == "link" and name in REMOVED_PANDA_LINKS) or (
            child.tag == "joint" and name in REMOVED_PANDA_JOINTS
        ):
            robot.remove(child)

    macro_path = (
        picknik_repo
        / "robotiq_description/urdf/robotiq_2f_85_macro.urdf.xacro"
    )
    macro_root = ET.parse(macro_path).getroot()
    macro = next(
        (
            node
            for node in macro_root
            if node.tag == f"{{{XACRO}}}macro"
            and node.get("name") == "robotiq_gripper"
        ),
        None,
    )
    if macro is None:
        raise ValueError("pinned PickNik source has no robotiq_gripper macro")
    for source in macro:
        if _local_name(source.tag) not in {"link", "joint"}:
            continue
        node = copy.deepcopy(source)
        _substitute(node)
        _remove_descendants(node, "surface")
        _remove_descendants(node, "mimic")
        robot.append(node)

    mount = next(
        joint
        for joint in robot.findall("joint")
        if joint.get("name") == "panda_hand_joint"
    )
    _remove_descendants(mount, "insert_block")
    for origin in mount.findall("origin"):
        mount.remove(origin)
    mount.append(
        ET.Element(
            "origin",
            {
                "xyz": "0 0 0.018174022",
                "rpy": f"0 0 {-math.pi / 2.0:.16g}",
            },
        )
    )

    output.mkdir(parents=True)
    mesh_sources, generated_collision_meshes, visual_conversion_records = _rewrite_meshes(
        robot,
        panda_urdf,
        picknik_repo,
        output,
        blender_executable,
        converter_script,
    )
    _validate_names(robot)
    ET.indent(panda_tree, space="  ")
    urdf_path = output / "franka_robotiq_2f_85.urdf"
    panda_tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    shutil.copy2(picknik_repo / "LICENSE", output / "PICKNIK_LICENSE")
    manifest = {
        "schema_version": "robolab_droid_portable_urdf.v2",
        "robot_embodiment": "robolab_droid",
        "generated_urdf": str(urdf_path),
        "generated_urdf_sha256": _sha256(urdf_path),
        "panda_urdf": str(panda_urdf),
        "panda_urdf_sha256": _sha256(panda_urdf),
        "picknik_repo": str(picknik_repo),
        "picknik_commit": PICKNIK_COMMIT,
        "picknik_macro": str(macro_path),
        "picknik_macro_sha256": _sha256(macro_path),
        "robolab_commit": ROBOLAB_COMMIT,
        "robolab_usd": str(robolab_usd),
        "robolab_usd_sha256": ROBOLAB_USD_SHA256,
        "mount": {
            "parent": "panda_link8",
            "child": "base_link",
            "xyz_m": [0.0, 0.0, 0.018174022],
            "rpy_rad": [0.0, 0.0, -math.pi / 2.0],
            "source": "RoboLab USD fixed joint plus PickNik-to-RoboLab base-axis registration",
            "maximum_published_knuckle_pivot_residual_m": 0.000242006,
        },
        "gripper": {
            "open_driver_rad": 0.0,
            "closed_driver_rad": math.pi / 4.0,
            "explicit_pybullet_mimic_multipliers": MIMIC_MULTIPLIERS,
        },
        "mesh_sources": [
            {"path": str(path), "sha256": _sha256(path)} for path in mesh_sources
        ],
        "generated_collision_meshes": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in generated_collision_meshes
        ],
        "visual_conversion": {
            "authority": "official SceneCode Blender runtime",
            "blender_executable": str(blender_executable),
            "converter_script": str(converter_script),
            "converter_script_sha256": _sha256(converter_script),
            "coordinate_contract": "metres, source Z-up preserved via OBJ -Y-forward/Z-up",
            "records": visual_conversion_records,
        },
    }
    manifest_path = output / "robot_asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
