"""Read-only material inspection and dependency binding for identity renders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digests(repository: Path) -> dict[str, str]:
    return {str(repository / name): sha256(repository / name) for name in (
        "scripts/render_scenecode_asset_identity.py",
        "scripts/render_scenecode_identity_blender.py",
        "autoresearch/scenecode_identity_lighting.py",
        "autoresearch/scenecode_identity_render_contract.py",
    )}


def resolve_uri(parent: Path, uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"} or parsed.netloc:
        raise ValueError(f"identity mesh requires a local file URI: {uri}")
    path = Path(unquote(parsed.path if parsed.scheme else uri))
    return (path if path.is_absolute() else parent / path).resolve()


def read_gltf(path: Path) -> dict:
    if path.suffix.lower() == ".gltf":
        return json.loads(path.read_text())
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError(f"truncated GLB: {path}")
    magic, version, length, size, kind = struct.unpack_from("<4sIIII", raw)
    if magic != b"glTF" or version != 2 or length != len(raw) or kind != 0x4E4F534A or size > len(raw) - 20:
        raise ValueError(f"invalid GLB header: {path}")
    return json.loads(raw[20:20 + size].decode().rstrip(" \x00"))


def asset_material_profile(sdf_path: Path) -> dict:
    """Inspect declared files without copying, editing, or guessing siblings."""
    sdf_path = sdf_path.resolve()
    dependencies = {str(sdf_path): sha256(sdf_path)}
    extensions: set[str] = set()
    for node in ET.parse(sdf_path).findall(".//visual/geometry/mesh/uri"):
        mesh = resolve_uri(sdf_path.parent, (node.text or "").strip())
        dependencies[str(mesh)] = sha256(mesh)
        if mesh.suffix.lower() not in {".gltf", ".glb"}:
            continue
        data = read_gltf(mesh)
        for item in [*data.get("buffers", []), *data.get("images", [])]:
            uri = item.get("uri")
            if uri and not uri.startswith("data:"):
                dependency = resolve_uri(mesh.parent, uri)
                dependencies[str(dependency)] = sha256(dependency)
        for material in data.get("materials", []):
            extensions.update(name for name in material.get("extensions", {})
                              if name.startswith("KHR_materials_"))
    return {"schema_version": "scenecode_identity_material_profile.v1",
            "material_extensions": sorted(extensions),
            "requires_native_material_renderer": bool(extensions),
            "dependencies": dependencies}


def validate_dependencies(dependencies: dict[str, str]) -> None:
    if not dependencies:
        raise ValueError("identity render has no bound source dependencies")
    for name, digest in dependencies.items():
        path = Path(name)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"identity render dependency changed: {name}")


def validate_render_sources(report: dict, repository: Path) -> None:
    if report.get("render_source_digests") != source_digests(repository):
        raise ValueError("identity render predates the current renderer implementation")
    for object_id, result in report.get("results", {}).items():
        if result.get("observable_material_copy"):
            raise ValueError(f"identity render modified material appearance: {object_id}")
        profile = result.get("material_profile") or {}
        validate_dependencies(profile.get("dependencies") or {})
        current = asset_material_profile(Path(result["sdf_path"]))
        if profile != current:
            raise ValueError(f"identity render material profile changed: {object_id}")
        if profile["requires_native_material_renderer"] and result.get("effective_renderer") != "blender_native_cpu":
            raise ValueError(f"identity renderer cannot certify declared material: {object_id}")
