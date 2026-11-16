"""Verify the mesh basis at the URDF-to-SDF format boundary.

The declared URDF mesh is authoritative here. This is distinct from native
projection of an already published SDF, whose declared SDF visual is authority.
Drake maps glTF Y-up into Z-up; copying an OBJ origin while substituting glTF
therefore needs a verified basis composition, not a guessed rotation removal.
"""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from autoresearch.scenecode_mesh_projection import _scene_meshes, file_digest, mesh_dependencies
from autoresearch.scenecode_mesh_export_receipt import export_geometry_hashes


Y_UP_TO_Z_UP = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def _corners(path: Path) -> np.ndarray:
    return np.concatenate([np.asarray(mesh.triangles).reshape(-1, 3)
                           for _, mesh in _scene_meshes(path)])


def _native_basis(path: Path) -> np.ndarray:
    return Y_UP_TO_Z_UP if path.suffix.lower() in {".gltf", ".glb"} else np.eye(3)


def _export_basis(path: Path):
    receipt_path = path.with_suffix(path.suffix + ".frame.json")
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    basis = np.asarray(receipt.get("blender_to_mesh_basis"))
    if (receipt.get("schema_version") != "scenecode_blender_mesh_export.v1"
            or receipt.get("mesh_sha256") != file_digest(path)
            or receipt.get("geometry_file_sha256") != export_geometry_hashes(path)
            or receipt.get("applied_modifiers") is not True
            or basis.shape != (3, 3) or not np.array_equal(basis, Y_UP_TO_Z_UP.T)):
        raise ValueError("mesh export frame receipt is stale or unsupported")
    return basis, receipt_path


def verified_mesh_basis(source: Path, destination: Path) -> dict:
    """Accept only known coordinate bases, never fit translations or scales."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    source_points = _corners(source) @ _native_basis(source).T
    destination_points = _corners(destination) @ _native_basis(destination).T
    if len(source_points) != len(destination_points):
        raise ValueError("mesh format conversion changed triangle count")
    if not np.isfinite(source_points).all() or not np.isfinite(destination_points).all():
        raise ValueError("mesh format conversion contains nonfinite geometry")
    # SceneCode's OBJ exporter writes six decimals; glTF stores float32.
    tolerance = max(2e-6, float(np.abs(source_points).max()) * 2e-7)
    tree = cKDTree(source_points)
    candidates = []
    provenance = [_export_basis(path) for path in (source, destination)]
    choices = (("identity", np.eye(3)),) if source == destination else (
        ("identity", np.eye(3)), ("undo_gltf_up_axis", Y_UP_TO_Z_UP.T),
        ("apply_gltf_up_axis", Y_UP_TO_Z_UP))
    if source != destination and all(value is not None for value in provenance):
        source_export, destination_export = [item[0] for item in provenance]
        basis = (_native_basis(source) @ source_export
                 @ np.linalg.inv(_native_basis(destination) @ destination_export))
        choices = (("verified_exporter_axes", basis),)
    for name, basis in choices:
        points = destination_points @ basis.T
        error = max(float(tree.query(points)[0].max()),
                    float(cKDTree(points).query(source_points)[0].max()))
        if error <= tolerance:
            candidates.append((name, basis, error))
    if len(candidates) != 1:
        raise ValueError("mesh basis is ambiguous or differs beyond a known up-axis conversion")
    name, basis, error = candidates[0]
    return {"schema_version": "scenecode_mesh_frame.v1", "passed": True,
            "source_mesh": str(source), "destination_mesh": str(destination),
            "basis_name": name, "basis_matrix": basis.tolist(),
            "triangle_count": len(source_points) // 3,
            "max_corner_error": error, "format_precision_tolerance": tolerance,
            "input_file_sha256": mesh_dependencies(source) | mesh_dependencies(destination)
                | {str(value[1]): file_digest(value[1]) for value in provenance if value is not None}}


def bind_visual_frame(urdf_visual: ET.Element, sdf_visual: ET.Element,
                      urdf_dir: Path) -> dict | None:
    """Compose a verified format basis with the original visual's local pose."""
    source_mesh = urdf_visual.find("geometry/mesh")
    destination_uri = sdf_visual.findtext("geometry/mesh/uri")
    if source_mesh is None or not destination_uri:
        return None
    source_uri = source_mesh.get("filename")
    if not source_uri:
        raise ValueError("URDF visual has no declared source mesh")
    mesh_scale = np.fromstring(source_mesh.get("scale", "1 1 1"), sep=" ")
    if (mesh_scale.shape != (3,) or not np.isfinite(mesh_scale).all()
            or not np.allclose(mesh_scale, 1.0)):
        raise ValueError("non-unit URDF mesh scale requires a complete visual and physics conversion")
    receipt = verified_mesh_basis(Path(urdf_dir) / source_uri,
                                  Path(urdf_dir) / destination_uri)
    pose = sdf_visual.find("pose")
    if pose is None:
        pose = ET.SubElement(sdf_visual, "pose")
    values = np.fromstring(pose.text or "0 0 0 0 0 0", sep=" ")
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("invalid SDF visual pose")
    original_rotation = Rotation.from_euler("xyz", values[3:]).as_matrix()
    rotation = original_rotation @ np.asarray(receipt["basis_matrix"])
    rpy = Rotation.from_matrix(rotation).as_euler("xyz")
    rpy[np.abs(rpy) < 1e-12] = 0
    pose.text = " ".join(format(float(v), ".17g") for v in [*values[:3], *rpy])
    receipt["original_pose"] = values.tolist()
    receipt["converted_pose"] = [*values[:3].tolist(), *rpy.tolist()]
    sdf_visual.append(ET.Comment(" WORLDOLYMPIAD_MESH_FRAME " + json.dumps(receipt, sort_keys=True) + " "))
    return receipt
