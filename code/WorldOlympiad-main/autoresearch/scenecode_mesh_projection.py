"""Convert declared SceneCode visuals without substituting a different frame.

Blender authoring meshes may still use assembly coordinates after the published
glTF has been recentered on a joint. Only the declared mesh and its scene graph
are authoritative. The conversion checks actual geometry and freezes all files
it reads or creates; body count alone is not a visual-fidelity check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex

import numpy as np
from scipy.spatial import cKDTree
import trimesh


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mesh_dependencies(path: Path) -> dict[str, str]:
    """Record local source dependencies, including glTF buffers and textures."""
    path = Path(path).resolve()
    files = {path}
    if path.suffix.lower() == ".gltf":
        data = json.loads(path.read_text())
        for entry in data.get("buffers", []) + data.get("images", []):
            uri = entry.get("uri", "")
            if not uri or uri.startswith("data:"):
                continue
            if "://" in uri:
                raise ValueError("native asset projection requires local mesh dependencies")
            from urllib.parse import unquote
            files.add((path.parent / unquote(uri)).resolve())
    elif path.suffix.lower() == ".obj":
        for line in path.read_text(errors="strict").splitlines():
            tokens = shlex.split(line, comments=True)
            if tokens and tokens[0] == "mtllib":
                for value in tokens[1:]:
                    material = (path.parent / value).resolve()
                    files.add(material)
                    for row in material.read_text().splitlines():
                        fields = shlex.split(row, comments=True)
                        if fields and fields[0].lower().startswith(("map_", "bump")):
                            files.add((material.parent / fields[-1]).resolve())
    return {str(file): file_digest(file) for file in sorted(files)}


def _scene_meshes(path: Path):
    scene = trimesh.load(str(path), force="scene", process=False)
    if not scene.graph.nodes_geometry:
        raise ValueError(f"declared visual has no triangle geometry: {path}")
    for node in sorted(scene.graph.nodes_geometry):
        matrix, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError("native visual conversion requires triangle meshes")
        mesh.apply_transform(matrix)
        yield str(node), mesh


def _geometry_comparison(source, converted, scale) -> dict:
    """Compare all triangle corners after export, in physical source units."""
    source_points = np.asarray(source.triangles).reshape(-1, 3) * scale
    converted_points = np.asarray(converted.triangles).reshape(-1, 3) * scale
    if not len(source_points) or not len(converted_points):
        raise ValueError("empty geometry cannot certify a native mesh conversion")
    error = max(float(cKDTree(source_points).query(converted_points)[0].max()),
                float(cKDTree(converted_points).query(source_points)[0].max()))
    tolerance = max(1e-7, float(np.ptp(source_points, axis=0).max()) * 1e-7)
    return {
        "source_triangle_count": len(source.faces),
        "converted_triangle_count": len(converted.faces),
        "max_triangle_corner_error_m": error,
        "tolerance_m": tolerance,
        "source_bounds_m": [source_points.min(axis=0).tolist(), source_points.max(axis=0).tolist()],
        "converted_bounds_m": [converted_points.min(axis=0).tolist(), converted_points.max(axis=0).tolist()],
        "passed": bool(len(source.faces) == len(converted.faces) and error <= tolerance),
    }


def convert_declared_visual(uri: str, *, output_dir: Path, scale: np.ndarray) -> tuple[list[dict], dict]:
    """Export the declared glTF/GLB's actual scene geometry to native OBJ files.

    No same-named authoring sibling is considered. Scene node transforms are
    baked exactly once; the caller composes the SDF link/element frames and the
    declared SceneCode Y-up basis exactly once.
    """
    source = Path(uri).resolve()
    inputs = mesh_dependencies(source)
    meshes = list(_scene_meshes(source))
    outputs = []
    if source.suffix.lower() not in {".gltf", ".glb"}:
        return [{"uri": str(source), "y_up": False}], {
            "declared_uri": str(source), "conversion": "native mesh retained",
            "input_file_sha256": inputs, "output_file_sha256": {}, "passed": True,
        }
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(output_dir).resolve() / "visual_meshes" / identity
    directory.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for index, (node, mesh) in enumerate(meshes):
        # Separate instance directories prevent material/texture name collisions.
        instance_dir = directory / str(index)
        instance_dir.mkdir(exist_ok=True)
        path = instance_dir / "mesh.obj"
        obj, resources = trimesh.exchange.obj.export_obj(
            mesh, return_texture=True, digits=10, mtl_name="mesh.mtl",
        )
        path.write_text(obj)
        for name, payload in (resources or {}).items():
            destination = (instance_dir / name).resolve()
            if instance_dir not in destination.parents:
                raise ValueError("converted material escaped its asset directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload.encode() if isinstance(payload, str) else payload)
        converted = trimesh.load(str(path), force="mesh", process=False)
        comparison = _geometry_comparison(mesh, converted, np.asarray(scale, dtype=float))
        if not comparison["passed"]:
            raise ValueError(f"native visual conversion changed declared geometry: {source}, node={node}")
        comparisons.append({"node": node, "native_mesh": str(path), **comparison})
        outputs.append({"uri": str(path), "y_up": True})
    created = {str(file): file_digest(file) for file in sorted(directory.rglob("*")) if file.is_file()}
    return outputs, {
        "declared_uri": str(source), "conversion": "declared scene graph to OBJ",
        "input_file_sha256": inputs, "output_file_sha256": created,
        "geometry_checks": comparisons, "passed": all(c["passed"] for c in comparisons),
    }


def validate_projection_receipt(path: Path) -> dict:
    """Reject stale/unverified generated assets before native reference or reuse."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if (data.get("schema_version") != "scenecode_rigid_projection.v1"
            or data.get("passed") is not True
            or data.get("geometry_fidelity", {}).get("passed") is not True
            or data.get("geometry_fidelity", {}).get("method") != "declared_source_mesh_scene_graph"):
        raise ValueError("native projection has no verified declared-source geometry fidelity")
    bindings = data.get("input_file_sha256", {}) | data.get("output_file_sha256", {})
    if not bindings:
        raise ValueError("native projection has no frozen mesh dependencies")
    for name, digest in bindings.items():
        if file_digest(Path(name)) != digest:
            raise ValueError(f"native projection dependency changed: {name}")
    urdf = Path(data["urdf"])
    if not urdf.is_absolute():
        urdf = path.parent / urdf
    if file_digest(urdf) != data["urdf_sha256"]:
        raise ValueError("native projection URDF changed after validation")
    return data
