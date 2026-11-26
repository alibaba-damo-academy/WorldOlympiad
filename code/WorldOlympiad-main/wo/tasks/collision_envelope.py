"""Conform undersized mesh-collision envelopes to contracted visual bounds.

SceneCode's convex decomposition can omit a small part of the source mesh's
outer envelope.  For assets with an explicit size contract, this module applies
one link-frame affine correction to all collision meshes when (and only when)
their complete envelope is inside the already-correct visual envelope.  It
never edits visuals, dimensions, cavity predicates, or an overhanging collision.
The output meshes and certificate hash updates are recorded for audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from .geometry_contract import collision_bounds, visual_bounds


def _resolve_mesh(sdf_path: Path, raw: str) -> Path:
    path = Path(raw.removeprefix("file://"))
    return path if path.is_absolute() else sdf_path.parent / path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract_mesh(mesh_path: Path, collision_node: ET.Element, geometry: ET.Element,
                   source_center: np.ndarray, target_center: np.ndarray,
                   factors: np.ndarray, index: int) -> Path:
    """Write a collision mesh with a link-frame affine applied to its vertices."""
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=float)
    scale_node = geometry.find("scale")
    mesh_scale = (np.asarray([float(x) for x in scale_node.text.split()])
                  if scale_node is not None else np.ones(3))
    if mesh_scale.shape != (3,) or not np.isfinite(mesh_scale).all() or np.any(mesh_scale == 0):
        raise ValueError(f"invalid collision mesh scale in {mesh_path}")
    pose = collision_node.find("pose")
    translation = np.zeros(3)
    rotation = np.eye(3)
    if pose is not None:
        if pose.get("relative_to"):
            raise ValueError("cannot conform collision with relative_to pose")
        values = [float(x) for x in (pose.text or "").split()]
        if len(values) not in (3, 6):
            raise ValueError("collision pose must have 3 or 6 values")
        translation = np.asarray(values[:3])
        if len(values) == 6:
            rotation = trimesh.transformations.euler_matrix(*values[3:])[:3, :3]
    world_vertices = (vertices * mesh_scale) @ rotation.T + translation
    world_vertices = target_center + (world_vertices - source_center) * factors
    local_vertices = (world_vertices - translation) @ rotation
    mesh.vertices = local_vertices / mesh_scale
    output = mesh_path.with_name(f"{mesh_path.stem}_contracted_{index:03d}.obj")
    mesh.export(output)
    return output


def conform_collision_envelopes(spec, scene, generation_dir: Path) -> dict:
    """Correct only wholly-inside collision envelopes for explicit size contracts.

    Visual geometry must already satisfy the authored dimensions.  Collision
    geometry that overhangs or is offset outside the visual envelope is left
    untouched so the authoritative validator continues to fail closed.
    """
    generation_dir = Path(generation_dir)
    certificate_path = generation_dir / "task_inventory_certificate.json"
    changed = []
    for declaration in spec.objects:
        expected = declaration.geometry_constraints.get("overall_dimensions_m")
        if expected is None:
            continue
        asset = scene.support if declaration.object_id == scene.support.object_id else scene.objects[declaration.object_id]
        sdf_path = Path(asset.sdf_path)
        visual = visual_bounds(sdf_path)
        visual_size = np.diff(visual, axis=0)[0] * asset.scale
        expected = np.asarray(expected, dtype=float)
        if expected.shape != (3,) or not np.all(np.isfinite(expected)) or np.any(expected <= 0):
            raise ValueError("overall_dimensions_m must be three finite positive sizes")
        if not np.all(np.abs(visual_size - expected) <= 0.0005):
            continue
        collision = collision_bounds(sdf_path)
        collision_size = np.diff(collision, axis=0)[0]
        visual_local_size = np.diff(visual, axis=0)[0]
        if np.any(collision_size <= 0) or np.any(visual_local_size <= 0):
            continue
        # Do not silently shrink collisions: only repair a decomposition wholly
        # contained by the visual AABB, including its center offset.
        if np.any(collision[0] < visual[0] - 1e-7) or np.any(collision[1] > visual[1] + 1e-7):
            continue
        already_aligned = (np.all(np.abs(collision_size - visual_local_size) <= 0.0005 / asset.scale)
                           and np.all(np.abs(collision.mean(axis=0) - visual.mean(axis=0))
                                      <= 0.0005 / asset.scale))
        if already_aligned:
            continue
        factors = visual_local_size / collision_size
        source_center = collision.mean(axis=0)
        target_center = visual.mean(axis=0)
        tree = ET.parse(sdf_path)
        link = tree.find(".//link")
        if link is None:
            continue
        collisions = link.findall("collision")
        if not collisions:
            continue
        mesh_records = []
        created_outputs = []
        for index, collision_node in enumerate(collisions):
            geometry = collision_node.find("geometry/mesh")
            if geometry is None or geometry.find("uri") is None:
                mesh_records = []
                break
            source_mesh = _resolve_mesh(sdf_path, geometry.findtext("uri", ""))
            if not source_mesh.is_file():
                mesh_records = []
                break
            pose = collision_node.find("pose")
            pose_copy = None if pose is None else ET.fromstring(ET.tostring(pose))
            output = _contract_mesh(source_mesh, collision_node, geometry,
                                    source_center, target_center, factors, index)
            created_outputs.append(output)
            geometry.find("uri").text = str(output.resolve())
            scale_node = geometry.find("scale")
            # _contract_mesh inverted the original pose and scale, so preserve
            # both SDF transforms while swapping in the corrected mesh.
            mesh_records.append({"source": str(source_mesh), "output": str(output),
                                 "sha256": _digest(output),
                                 "pose": None if pose_copy is None else pose_copy.text,
                                 "source_scale": None if scale_node is None else scale_node.text})
        if not mesh_records or len(mesh_records) != len(collisions):
            # No partial rewrite should be published for an unsupported asset.
            for output in created_outputs:
                output.unlink(missing_ok=True)
            continue
        tree.write(sdf_path, encoding="utf-8", xml_declaration=True)
        changed.append({"object_id": declaration.object_id, "sdf_path": str(sdf_path),
                        "old_bounds_m": collision.tolist(), "visual_bounds_m": visual.tolist(),
                        "factors": factors.tolist(), "collision_meshes": mesh_records,
                        "sdf_sha256": _digest(sdf_path)})
    if not changed:
        return {"schema_version": "wo_collision_envelope_adjustment.v1", "adjustments": []}

    manifest_path = generation_dir / "generation_manifest.json"
    if not certificate_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("collision adjustment requires current inventory certificate and generation manifest")
    certificate = json.loads(certificate_path.read_text())
    changed_sdfs = {Path(record["sdf_path"]).resolve(): record["sdf_sha256"] for record in changed}
    updated_paths = set()
    for match in certificate.get("matches", []):
        path = Path(match.get("sdf_path", ""))
        if not path.is_absolute():
            path = generation_dir / path
        resolved_path = path.resolve()
        digest = changed_sdfs.get(resolved_path)
        if digest:
            match["sdf_sha256"] = digest
            updated_paths.add(resolved_path)
    if updated_paths != set(changed_sdfs):
        raise ValueError("modified collision assets are not all bound by the task inventory certificate")
    certificate_path.write_text(json.dumps(certificate, indent=2) + "\n")
    manifest = json.loads(manifest_path.read_text())
    manifest["task_inventory_certificate_sha256"] = _digest(certificate_path)
    adjustments_path = generation_dir / "collision_envelope_adjustments.json"
    adjustments_path.write_text(json.dumps({
        "schema_version": "wo_collision_envelope_adjustment.v1",
        "adjustments": changed,
    }, indent=2) + "\n")
    manifest["collision_envelope_adjustments"] = {
        "path": adjustments_path.name, "sha256": _digest(adjustments_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report = {"schema_version": "wo_collision_envelope_adjustment.v1", "adjustments": changed}
    return report
