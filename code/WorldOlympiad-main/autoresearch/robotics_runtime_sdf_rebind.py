"""Resolve published reference collision meshes for the native WAM adapter.

Older v6 bundles archive merged physics SDFs whose mesh URIs still name the
materializer's deleted cache.  The model runtime must read the bundle's hashed
meshes, never revive that cache or trust a mutable SceneCode generation path.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from .io import write_json
from .robotics_views import sha256_file
from .sdf_rigid_joint_lock import rigidify_sdf_joints
from .scenecode_rigid_projection import (
    parse_locked_sdf, _combine_inertial, _pose_matrix, _rpy_xyz,
)


def _inside(root: Path, value: Path) -> Path:
    if value.is_symlink():
        raise ValueError(f"reference asset may not be a symlink: {value}")
    path = value.resolve()
    path.relative_to(root)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _source_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("prepared reference asset must use a bundle-relative path")
    return _inside(root, root / value)


def _flatten_locked_sdf(source: Path, output: Path) -> dict:
    """Compose a verified fixed-link SDF into one rigid Isaac body."""
    parsed = parse_locked_sdf(source, allow_primitives=True)
    tree = ET.parse(source)
    root = tree.getroot()
    model = root.find("model")
    if model is None or len(parsed) != len(model.findall("link")):
        raise ValueError("locked SDF link projection is incomplete")
    inertial = _combine_inertial(parsed)
    original_links = list(model.findall("link"))
    for item in [*model.findall("link"), *model.findall("joint")]:
        model.remove(item)
    link = ET.SubElement(model, "link", name="rigid_subject")
    mass = ET.SubElement(link, "inertial")
    ET.SubElement(mass, "mass").text = f"{inertial['mass']:.12g}"
    ET.SubElement(mass, "pose").text = " ".join(
        f"{value:.12g}" for value in (*inertial["com"], 0.0, 0.0, 0.0)
    )
    inertia = ET.SubElement(mass, "inertia")
    for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
        ET.SubElement(inertia, key).text = f"{inertial[key]:.12g}"
    collision_count = 0
    visual_count = 0
    for record, old_link in zip(parsed, original_links, strict=True):
        for kind in ("visual", "collision"):
            originals = old_link.findall(kind)
            measured_items = record[f"{kind}s"]
            if len(originals) != len(measured_items):
                raise ValueError(f"locked SDF {kind} inventory changed")
            for index, (old, measured) in enumerate(zip(
                originals, measured_items, strict=True
            )):
                element = deepcopy(old)
                element.set("name", f"{record['name']}_{kind}_{index}")
                pose = record["pose"] @ measured["pose"]
                xyz, rpy = _rpy_xyz(pose)
                pose_node = element.find("pose")
                if pose_node is None:
                    pose_node = ET.Element("pose")
                    element.insert(0, pose_node)
                pose_node.attrib.clear()
                pose_node.text = " ".join(f"{value:.12g}" for value in (*xyz, *rpy))
                uri = element.find("geometry/mesh/uri")
                if uri is not None:
                    uri.text = str(measured["uri"])
                link.append(element)
                if kind == "collision":
                    collision_count += 1
                else:
                    visual_count += 1
    if collision_count == 0 or visual_count == 0:
        raise ValueError("rigid subject lacks visual or collision geometry")
    tree.write(output, encoding="unicode", xml_declaration=True)
    return {
        "schema_version": "robotics_isaac_rigid_sdf_projection.v1",
        "source_sdf_sha256": sha256_file(source),
        "output_sdf_sha256": sha256_file(output),
        "source_link_count": len(parsed),
        "output_link_count": 1,
        "collision_count": collision_count,
        "visual_count": visual_count,
        "combined_mass": inertial["mass"],
    }


def bind_runtime_physics_sdfs(
    reference_manifest: Path, prepared: dict, output_dir: Path,
    bundle_digests: dict[str, str],
    *, allow_subject_joint_lock: bool = False,
) -> dict[str, Path]:
    """Write version-bound runtime SDFs with URIs into the immutable bundle."""
    reference_manifest = Path(reference_manifest).resolve()
    root = reference_manifest.parent
    output_dir = Path(output_dir).resolve()
    if root in output_dir.parents or root == output_dir:
        raise ValueError("runtime derivatives must stay outside the reference bundle")
    if not prepared:
        raise ValueError("DROID reference needs at least one prepared physics asset")
    invalid_keys = [
        key for key in prepared
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", key)
    ]
    if invalid_keys:
        raise ValueError(f"prepared asset keys are not safe file names: {invalid_keys}")

    # The original SDF's own relative URIs are the strongest mapping from a
    # merged collision name to an archived mesh.  A bundle-wide name lookup
    # covers bridge-generated mesh names only when their content is unique.
    mesh_index: dict[str, list[Path]] = {}
    for relative, expected in bundle_digests.items():
        path = _inside(root, root / relative)
        if path.suffix.lower() not in {".obj", ".stl", ".dae", ".ply"}:
            continue
        if sha256_file(path) != expected:
            raise ValueError(f"archived mesh digest changed: {relative}")
        mesh_index.setdefault(path.name, []).append(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    derived: dict[str, Path] = {}
    records = []
    for role, record in sorted(prepared.items()):
        if record.get('physics_urdf'):
            raise ValueError(f'{role}: authoritative expert URDF requires the URDF '
                             'runtime importer; source SDF physics substitution is forbidden')
        source = _source_path(root, record["source_sdf"])
        physics = _source_path(root, record.get("physics_sdf") or record["source_sdf"])
        source_meshes: dict[str, Path] = {}
        for node in ET.parse(source).findall(".//mesh/uri"):
            raw = str(node.text or "")
            if not raw:
                raise ValueError("source SDF has an empty mesh URI")
            candidate = _inside(root, source.parent / raw)
            if candidate.name in source_meshes and source_meshes[candidate.name] != candidate:
                raise ValueError("source SDF uses ambiguous mesh basenames")
            source_meshes[candidate.name] = candidate
        tree = ET.parse(physics)
        bindings = []
        for node in tree.findall(".//mesh/uri"):
            old = str(node.text or "")
            name = Path(old).name
            # New bundles preserve the physics SDF's own mesh paths. Prefer
            # those role-local bindings to a potentially ambiguous basename.
            direct = Path(old)
            if not direct.is_absolute():
                direct = physics.parent / direct
            candidate = None
            try:
                relative_direct = direct.resolve().relative_to(root).as_posix()
            except ValueError:
                relative_direct = None
            if relative_direct in bundle_digests:
                candidate = _inside(root, direct)
                if sha256_file(candidate) != bundle_digests[relative_direct]:
                    raise ValueError(f'archived mesh digest changed: {relative_direct}')
            if candidate is None:
                candidate = source_meshes.get(name)
            if candidate is None:
                choices = mesh_index.get(name) or []
                digests = {sha256_file(path) for path in choices}
                if len(digests) != 1:
                    raise ValueError(f"merged {role} mesh lacks one archived match: {name}")
                candidate = sorted(choices)[0]
            relative = candidate.relative_to(root).as_posix()
            expected = bundle_digests.get(relative)
            if expected != sha256_file(candidate):
                raise ValueError(f"merged {role} mesh is not bound to bundle: {relative}")
            node.text = str(candidate)
            bindings.append({
                "merged_uri": old, "bundle_mesh": relative, "sha256": expected,
            })
        visuals = []
        visual_source = source
        source_tree = ET.parse(source)
        source_moving_joints = [
            item.get("name") for item in source_tree.getroot().findall(".//joint")
            if item.get("type") != "fixed"
        ]
        if source_moving_joints:
            if not allow_subject_joint_lock:
                raise ValueError("source visual articulation is not a rigid task asset")
            visual_source = output_dir / f"{role}_source_locked.sdf"
            rigidify_sdf_joints(
                source, visual_source, joints=source_moving_joints,
                provenance={"reference_manifest_sha256": sha256_file(reference_manifest),
                            "purpose": "native_rigid_runtime_visual_projection"},
            )
        resolved_visual_source = output_dir / f"{role}_source_resolved.sdf"
        visual_tree = ET.parse(visual_source)
        for uri in visual_tree.getroot().findall(".//mesh/uri"):
            raw = str(uri.text or "")
            candidate_path = Path(raw)
            if not candidate_path.is_absolute():
                candidate_path = source.parent / candidate_path
            candidate = _inside(root, candidate_path)
            relative = candidate.relative_to(root).as_posix()
            expected = bundle_digests.get(relative)
            if expected != sha256_file(candidate):
                raise ValueError(f"source visual is not bound to reference bundle: {relative}")
            uri.text = str(candidate)
        visual_tree.write(resolved_visual_source, encoding="unicode", xml_declaration=True)
        visual_source = resolved_visual_source
        parsed_visuals = parse_locked_sdf(visual_source, allow_primitives=True)
        xml_visual_links = ET.parse(visual_source).getroot().findall(".//link")
        if len(parsed_visuals) != len(xml_visual_links):
            raise ValueError("visual link inventory changed during rigid projection")
        physics_model = tree.getroot().find("model")
        if physics_model is None:
            raise ValueError("merged physics SDF lacks a model")
        physics_links = physics_model.findall("link")
        child_names = {item.findtext("child") for item in physics_model.findall("joint")}
        physics_roots = [item for item in physics_links
                         if item.get("name") not in child_names]
        if len(physics_roots) != 1:
            raise ValueError("merged physics SDF has no unique root link")
        target_link = physics_roots[0]
        root_inverse = np.linalg.inv(_pose_matrix(target_link.findtext("pose")))
        for measured_link, source_link in zip(parsed_visuals, xml_visual_links, strict=True):
            declared = source_link.findall("visual")
            if len(declared) != len(measured_link["visuals"]):
                raise ValueError("source visual inventory differs from parsed geometry")
            for original, measured in zip(declared, measured_link["visuals"], strict=True):
                copied = deepcopy(original)
                pose = root_inverse @ measured_link["pose"] @ measured["pose"]
                xyz, rpy = _rpy_xyz(pose)
                pose_node = copied.find("pose")
                if pose_node is None:
                    pose_node = ET.Element("pose")
                    copied.insert(0, pose_node)
                pose_node.attrib.clear()
                pose_node.text = " ".join(f"{value:.12g}" for value in (*xyz, *rpy))
                for uri in copied.findall(".//mesh/uri"):
                    candidate = _inside(root, Path(str(measured["uri"])))
                    relative = candidate.relative_to(root).as_posix()
                    expected = bundle_digests.get(relative)
                    if expected != sha256_file(candidate):
                        raise ValueError("source visual is not bound to reference bundle")
                    uri.text = str(candidate)
                    visuals.append({"bundle_mesh": relative, "sha256": expected,
                                    "source_link": measured_link["name"]})
                target_link.append(copied)
        target = output_dir / f"{role}.sdf"
        tree.write(target, encoding="unicode", xml_declaration=True)
        projection = None
        if len(tree.getroot().findall(".//link")) > 1:
            if not allow_subject_joint_lock:
                raise ValueError(f"multi-link asset {role!r} needs a rigid-task contract")
            joints = [
                item.get("name") for item in tree.getroot().findall(".//joint")
                if item.get("type") != "fixed"
            ]
            locked = output_dir / f"{role}_locked.sdf"
            lock_receipt = rigidify_sdf_joints(
                target, locked, joints=joints,
                provenance={"reference_manifest_sha256": sha256_file(reference_manifest),
                            "purpose": "native_rigid_runtime_unmanipulated_internal_joint"},
            ) if joints else None
            flattened = output_dir / f"{role}_rigid.sdf"
            projection = _flatten_locked_sdf(locked if joints else target, flattened)
            projection["joint_lock"] = lock_receipt
            target = flattened
        derived[role] = target
        records.append({
            "role": role,
            "source_sdf_sha256": sha256_file(source),
            "merged_physics_sdf_sha256": sha256_file(physics),
            "runtime_sdf": target.name,
            "runtime_sdf_sha256": sha256_file(target),
            "mesh_bindings": bindings,
            "visual_bindings": visuals,
            "rigid_projection": projection,
        })
    write_json(output_dir / "asset_binding_receipt.json", {
        "schema_version": "robotics_runtime_sdf_rebind.v1",
        "reference_manifest_sha256": sha256_file(reference_manifest),
        "assets": records,
    })
    return derived
