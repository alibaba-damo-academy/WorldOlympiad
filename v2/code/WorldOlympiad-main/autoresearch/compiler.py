"""Deterministic WorldIR -> self-contained PyBullet URDF compiler."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file, write_json
from .materials import CATEGORY_COLORS
from .math3d import compose
from .models import BodyIR, Geometry, Inertial, LinkIR, Pose, RepairRecord, WorldIR


def _fmt(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _quat_to_rpy(quat: tuple[float, float, float, float]) -> tuple[float, float, float]:
    x, y, z, w = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return roll, pitch, math.atan2(siny_cosp, cosy_cosp)


def _origin(parent: ET.Element, pose: Pose) -> None:
    ET.SubElement(parent, "origin", xyz=_fmt(pose.xyz), rpy=_fmt(_quat_to_rpy(pose.quat_xyzw)))


def _geometry(parent: ET.Element, geometry: Geometry, mesh_path: str | None = None) -> None:
    node = ET.SubElement(parent, "geometry")
    if geometry.kind == "mesh":
        ET.SubElement(node, "mesh", filename=mesh_path or str(geometry.path), scale=_fmt(geometry.scale))
    elif geometry.kind == "box":
        ET.SubElement(node, "box", size=_fmt(geometry.size or (0, 0, 0)))
    elif geometry.kind == "sphere":
        ET.SubElement(node, "sphere", radius=f"{geometry.radius:.9g}")
    else:
        ET.SubElement(node, "cylinder", radius=f"{geometry.radius:.9g}", length=f"{geometry.length:.9g}")


def _default_inertial(link: LinkIR, category: str) -> Inertial:
    mass_by_category = {"ball": 0.25, "cup": 0.35, "slider": 0.3, "block": 0.18}
    mass = next((value for key, value in mass_by_category.items() if key in category), 1.0)
    size = (0.2, 0.2, 0.2)
    for geometry in link.collisions:
        if geometry.kind == "box" and geometry.size:
            size = geometry.size
            break
        if geometry.kind == "sphere" and geometry.radius:
            radius = geometry.radius
            inertia = 0.4 * mass * radius * radius
            return Inertial(mass=mass, inertia=(inertia, inertia, inertia, 0, 0, 0))
    x, y, z = size
    return Inertial(
        mass=mass,
        inertia=(
            mass * (y * y + z * z) / 12,
            mass * (x * x + z * z) / 12,
            mass * (x * x + y * y) / 12,
            0,
            0,
            0,
        ),
    )


def _principalize(inertial: Inertial) -> tuple[Inertial, Pose, bool]:
    ixx, iyy, izz, ixy, ixz, iyz = inertial.inertia
    matrix = np.asarray([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)
    values, vectors = np.linalg.eigh(matrix)
    # Inertia has units of mass * length squared: a fixed SI cutoff rejects
    # legitimate small links after scaling. Test definiteness relative to the
    # tensor's own scale, retaining (not clamping) measured eigenvalues.
    scale = float(np.max(np.abs(values)))
    if not np.isfinite(values).all() or float(values.min()) <= np.finfo(np.float64).eps * 32 * scale:
        raise ValueError(f"inertia is not positive definite: eigenvalues={values.tolist()}")
    if np.linalg.det(vectors) < 0:
        vectors[:, 0] *= -1
    trace = float(np.trace(matrix))
    offdiag = matrix - np.diag(np.diag(matrix))
    changed = float(np.linalg.norm(offdiag)) > max(1e-10, trace * 1e-8)
    # Rotation matrix -> quaternion xyzw.
    try:
        from scipy.spatial.transform import Rotation

        quat = tuple(float(value) for value in Rotation.from_matrix(vectors).as_quat())
    except ImportError:
        quat = (0.0, 0.0, 0.0, 1.0)
        if changed:
            raise RuntimeError("scipy is required to principalize a non-diagonal inertia")
    result = Inertial(
        mass=inertial.mass,
        com=inertial.com,
        inertia=(float(values[0]), float(values[1]), float(values[2]), 0.0, 0.0, 0.0),
    )
    return result, Pose(xyz=inertial.com, quat_xyzw=quat), changed


def _prepare_mesh(
    geometry: Geometry,
    asset_dir: Path,
    *,
    visual: bool,
    repairs: list[RepairRecord],
    attempt: int,
) -> str:
    source = Path(str(geometry.path)).resolve()
    suffix = source.suffix.lower()
    destination = asset_dir / f"{geometry.name}{'.obj' if suffix in {'.gltf', '.glb'} else suffix}"
    if suffix in {".gltf", ".glb"}:
        blender = (os.environ.get("BLENDER_BIN") or shutil.which("blender")) if visual else None
        converter = Path(__file__).resolve().parents[1] / "scripts" / "blender_gltf_to_obj.py"
        blender_error: str | None = None
        if blender and converter.is_file():
            try:
                result = subprocess.run(
                    [
                        blender,
                        "--background",
                        "--python",
                        str(converter),
                        "--",
                        "--input",
                        str(source),
                        "--output",
                        str(destination),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                companion_names = sorted(
                    path.name
                    for path in asset_dir.glob(f"{destination.stem}*")
                    if path != destination
                )
                repairs.append(
                    RepairRecord(
                        attempt=attempt,
                        operation="convert_visual_mesh_preserve_material",
                        target=str(source),
                        detail=(
                            "converted glTF to a Z-up OBJ/MTL/texture bundle with Blender; "
                            f"companions={companion_names}"
                        ),
                    )
                )
            except Exception as exc:
                blender_error = str(exc)
        if destination.exists() and destination.stat().st_size > 0:
            return destination.name
        try:
            import trimesh

            loaded = trimesh.load(source, force="scene")
            # glTF is Y-up, while WorldIR, the SceneCode collision OBJs and
            # Isaac are Z-up. Blender performs this during Z-up OBJ export;
            # the headless fallback must do the same.
            loaded.apply_transform(gltf_y_up_to_worldir_z_up_transform())
            mesh = loaded.dump(concatenate=True) if hasattr(loaded, "dump") else loaded
            mesh.export(destination, file_type="obj")
            repairs.append(
                RepairRecord(
                    attempt=attempt,
                    operation="convert_visual_mesh",
                    target=str(source),
                    detail=(
                        f"converted {suffix} Y-up to Z-up OBJ with deterministic category-color fallback"
                        + (f" after Blender material conversion failed: {blender_error}" if blender_error else "")
                    ),
                )
            )
        except Exception as exc:
            raise RuntimeError(f"failed to convert mesh {source}: {exc}") from exc
    else:
        shutil.copy2(source, destination)
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"mesh preparation produced no output: {destination}")
    return destination.name


def gltf_y_up_to_worldir_z_up_transform() -> np.ndarray:
    """Return the right-handed +90-degree X rotation from glTF to WorldIR."""

    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _write_body_urdf(body: BodyIR, output_dir: Path, repairs: list[RepairRecord]) -> tuple[Path, dict[str, int]]:
    body_dir = output_dir / body.name
    asset_dir = body_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    robot = ET.Element("robot", name=body.name)
    mesh_cache: dict[tuple[str, bool], str] = {}
    counts = {"visual": 0, "collision": 0}
    link_names = {link.name for link in body.links}

    for link in body.links:
        link_node = ET.SubElement(robot, "link", name=link.name)
        inertial = link.inertial
        if inertial is None:
            inertial = _default_inertial(link, body.semantic_category)
            repairs.append(RepairRecord(attempt=2, operation="inject_inertial", target=f"{body.name}/{link.name}", detail="used deterministic category prior"))
        inertial, inertial_pose, changed = _principalize(inertial)
        if changed:
            repairs.append(RepairRecord(attempt=2, operation="principalize_inertia", target=f"{body.name}/{link.name}", detail="diagonalized the symmetric inertia tensor"))
        inertial_node = ET.SubElement(link_node, "inertial")
        _origin(inertial_node, inertial_pose)
        ET.SubElement(inertial_node, "mass", value=f"{inertial.mass:.9g}")
        ixx, iyy, izz, ixy, ixz, iyz = inertial.inertia
        ET.SubElement(inertial_node, "inertia", ixx=f"{ixx:.9g}", iyy=f"{iyy:.9g}", izz=f"{izz:.9g}", ixy=f"{ixy:.9g}", ixz=f"{ixz:.9g}", iyz=f"{iyz:.9g}")
        for visual in link.visuals:
            node = ET.SubElement(link_node, "visual", name=visual.name)
            _origin(node, visual.pose)
            mesh_path = None
            if visual.kind == "mesh":
                cache_key = (str(visual.path), True)
                if cache_key not in mesh_cache:
                    mesh_cache[cache_key] = _prepare_mesh(
                        visual, asset_dir, visual=True, repairs=repairs, attempt=1
                    )
                mesh_path = mesh_cache[cache_key]
                mesh_path = f"assets/{mesh_path}"
            _geometry(node, visual, mesh_path)
            rgba = visual.rgba or CATEGORY_COLORS.get(body.semantic_category, (0.55, 0.55, 0.55, 1.0))
            material = ET.SubElement(node, "material", name=f"{body.name}_{visual.name}_material")
            ET.SubElement(material, "color", rgba=_fmt(rgba))
            counts["visual"] += 1
        for collision in link.collisions:
            node = ET.SubElement(link_node, "collision", name=collision.name)
            _origin(node, collision.pose)
            mesh_path = None
            if collision.kind == "mesh":
                cache_key = (str(collision.path), False)
                if cache_key not in mesh_cache:
                    mesh_cache[cache_key] = _prepare_mesh(collision, asset_dir, visual=False, repairs=repairs, attempt=1)
                mesh_path = f"assets/{mesh_cache[cache_key]}"
            _geometry(node, collision, mesh_path)
            counts["collision"] += 1

    joints = list(body.joints)
    child_links = {joint.child for joint in joints}
    root_name = next((link.name for link in body.links if link.name not in child_links), body.links[0].name)
    # SDF permits disconnected links with poses; URDF does not. Attach them deterministically.
    for link in body.links:
        if link.name != root_name and link.name not in child_links:
            from .models import JointIR

            joints.append(JointIR(name=f"auto_fixed_{link.name}", type="fixed", parent=root_name, child=link.name, pose=link.pose))
            repairs.append(RepairRecord(attempt=2, operation="connect_sdf_link", target=f"{body.name}/{link.name}", detail=f"added fixed joint to {root_name}"))
    for joint in joints:
        if joint.parent not in link_names or joint.child not in link_names:
            raise ValueError(f"invalid joint topology in body {body.name}: {joint.name}")
        node = ET.SubElement(robot, "joint", name=joint.name, type=joint.type)
        ET.SubElement(node, "parent", link=joint.parent)
        ET.SubElement(node, "child", link=joint.child)
        _origin(node, joint.pose)
        if joint.type != "fixed":
            ET.SubElement(node, "axis", xyz=_fmt(joint.axis))
            if joint.type != "continuous":
                if joint.lower is None or joint.upper is None or joint.lower >= joint.upper:
                    raise ValueError(f"joint {joint.name} requires ordered finite limits")
                ET.SubElement(node, "limit", lower=f"{joint.lower:.9g}", upper=f"{joint.upper:.9g}", effort=f"{joint.effort:.9g}", velocity=f"{joint.velocity:.9g}")
            ET.SubElement(node, "dynamics", damping=f"{joint.damping:.9g}")

    path = body_dir / f"{body.name}.urdf"
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)
    return path, counts


def _package_precompiled_urdf(
    body: BodyIR,
    output_dir: Path,
    repairs: list[RepairRecord],
) -> tuple[Path, dict[str, int]]:
    """Copy a complete audited URDF directory for complex released robots."""

    source = Path(str(body.metadata["precompiled_urdf_source"])).resolve()
    if not source.is_file() or source.suffix.lower() != ".urdf":
        raise FileNotFoundError(f"invalid precompiled URDF source: {source}")
    body_dir = output_dir / body.name
    package_dir = body_dir / source.parent.name
    if package_dir.exists():
        raise FileExistsError(f"precompiled URDF package already exists: {package_dir}")
    shutil.copytree(source.parent, package_dir)
    packaged = package_dir / source.name
    if not packaged.is_file():
        raise RuntimeError(f"precompiled URDF copy is incomplete: {packaged}")
    missing_meshes = [
        str(geometry.path)
        for link in body.links
        for geometry in (*link.visuals, *link.collisions)
        if geometry.kind == "mesh" and not Path(str(geometry.path)).is_file()
    ]
    if missing_meshes:
        raise FileNotFoundError(f"precompiled robot references missing meshes: {missing_meshes}")
    tool_name = body.metadata.get("simready_tool_link")
    if tool_name:
        tool_link = next((link for link in body.links if link.name == tool_name), None)
        if tool_link is None or len(tool_link.visuals) != 1 or len(tool_link.collisions) != 1:
            raise ValueError("SimReady robot tool must have exactly one visual and collision")
        tree = ET.parse(packaged)
        robot_node = tree.getroot()
        link_node = ET.SubElement(robot_node, "link", name=tool_link.name)
        inertial = tool_link.inertial or _default_inertial(tool_link, body.semantic_category)
        inertial_node = ET.SubElement(link_node, "inertial")
        _origin(inertial_node, Pose(xyz=inertial.com))
        ET.SubElement(inertial_node, "mass", value=f"{inertial.mass:.9g}")
        ixx, iyy, izz, ixy, ixz, iyz = inertial.inertia
        ET.SubElement(inertial_node, "inertia", ixx=f"{ixx:.9g}", iyy=f"{iyy:.9g}", izz=f"{izz:.9g}", ixy=f"{ixy:.9g}", ixz=f"{ixz:.9g}", iyz=f"{iyz:.9g}")
        visual_node = ET.SubElement(link_node, "visual", name=tool_link.visuals[0].name)
        _origin(visual_node, tool_link.visuals[0].pose)
        _geometry(visual_node, tool_link.visuals[0])
        material_node = ET.SubElement(visual_node, "material", name="worldolympiad_tool_material")
        ET.SubElement(material_node, "color", rgba=_fmt(tool_link.visuals[0].rgba or (0.12, 0.12, 0.14, 1.0)))
        collision_node = ET.SubElement(link_node, "collision", name=tool_link.collisions[0].name)
        _origin(collision_node, tool_link.collisions[0].pose)
        _geometry(collision_node, tool_link.collisions[0])
        joint_node = ET.SubElement(robot_node, "joint", name="worldolympiad_pusher_joint", type="fixed")
        ET.SubElement(joint_node, "parent", link=str(body.metadata["simready_tool_parent"]))
        ET.SubElement(joint_node, "child", link=str(tool_name))
        _origin(joint_node, Pose(xyz=tuple(body.metadata["simready_tool_pose_xyz"])))
        ET.indent(robot_node, space="  ")
        tree.write(packaged, encoding="utf-8", xml_declaration=True)
        repairs.append(
            RepairRecord(
                attempt=1,
                operation="append_simready_contact_tool",
                target=body.name,
                detail=f"added audited fixed contact tool link {tool_name}",
            )
        )
    counts = {
        "visual": sum(len(link.visuals) for link in body.links),
        "collision": sum(len(link.collisions) for link in body.links),
    }
    repairs.append(
        RepairRecord(
            attempt=1,
            operation="package_complete_precompiled_urdf",
            target=body.name,
            detail=(
                f"copied released articulated asset directory {source.parent.name}; "
                f"links={len(body.links)}, joints={len(body.joints)}, geometry={counts}"
            ),
        )
    )
    return packaged, counts


def compile_world(world: WorldIR, output_dir: str | Path) -> Path:
    """Compile every body and fail if any declared geometry disappears."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    repairs: list[RepairRecord] = []
    manifest: dict[str, Any] = {
        "schema_version": "pybullet_bundle.v1",
        "world_hash": world.content_hash(),
        "bodies": {},
        "repairs": [],
    }
    for body in world.bodies:
        implicit_joint_poses = body.metadata.get(
            "simready_sdf_implicit_joint_pose_names", []
        )
        if implicit_joint_poses:
            repairs.append(
                RepairRecord(
                    attempt=2,
                    operation="resolve_sdf_implicit_joint_pose",
                    target=body.name,
                    detail=(
                        "transferred child-link poses to URDF joint origins for "
                        f"SDF joints with omitted pose: {implicit_joint_poses}"
                    ),
                )
            )
        if body.metadata.get("simready_visual_scale_policy") == "remove_baked_scenecode_visual_scale":
            repairs.append(
                RepairRecord(
                    attempt=1,
                    operation="remove_baked_visual_mesh_scale",
                    target=body.name,
                    detail=(
                        "removed stale SDF visual scale after raw GLTF dimensions "
                        "matched the authoritative SceneCode bbox"
                    ),
                )
            )
        restore_factor = body.metadata.get(
            "simready_restore_original_asset_scale_factor"
        )
        if restore_factor is not None and not math.isclose(
            float(restore_factor), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            repairs.append(
                RepairRecord(
                    attempt=3,
                    operation="restore_generated_asset_scale",
                    target=body.name,
                    detail=(
                        "restored pre-placement generated dimensions by factor "
                        f"{float(restore_factor):.12g} from SceneCode cumulative "
                        f"scale {float(body.metadata['simready_source_scenecode_scale_factor']):.12g}"
                    ),
                )
            )
        task_scale = body.metadata.get("simready_task_size_scale_factor")
        if task_scale is not None and not math.isclose(
            float(task_scale), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            repairs.append(
                RepairRecord(
                    attempt=3,
                    operation="normalize_task_actor_size",
                    target=body.name,
                    detail=(
                        f"uniformly scaled by {float(task_scale):.12g} under "
                        f"{body.metadata['simready_size_profile_id']}; "
                        f"source_extent_m={body.metadata['simready_source_extent_m']}, "
                        f"sized_extent_m={body.metadata['simready_task_sized_extent_m']}"
                    ),
                )
            )
        if body.metadata.get("precompiled_urdf_source"):
            urdf, counts = _package_precompiled_urdf(body, destination, repairs)
        else:
            urdf, counts = _write_body_urdf(body, destination, repairs)
        expected = {
            "visual": sum(len(link.visuals) for link in body.links),
            "collision": sum(len(link.collisions) for link in body.links),
        }
        if counts != expected:
            raise RuntimeError(f"compiler dropped geometry for {body.name}: expected={expected}, actual={counts}")
        effective_pose = compose(body.pose, body.internal_model_pose)
        manifest["bodies"][body.name] = {
            "urdf_path": str(urdf),
            "urdf_sha256": sha256_file(urdf),
            "pose": effective_pose.model_dump(mode="json"),
            "static": body.static,
            "target": body.target,
            "semantic_category": body.semantic_category,
            "support": body.support,
            "material": body.material.model_dump(mode="json"),
            "metadata": body.metadata,
            "source_path": body.source_path,
            "source_sha256": body.source_sha256,
            "geometry_counts": counts,
            "bundle_files": [
                {
                    "path": path.relative_to(destination).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(urdf.parent.rglob("*"))
                if path.is_file()
            ],
        }
    manifest["repairs"] = [repair.model_dump(mode="json") for repair in repairs]
    path = write_json(destination / "pybullet_bundle.json", manifest)
    write_json(destination / "compile_report.json", {
        "status": "repaired_pass" if repairs else "pass",
        "world_hash": world.content_hash(),
        "repairs": manifest["repairs"],
        "bundle_sha256": sha256_file(path),
    })
    return path
