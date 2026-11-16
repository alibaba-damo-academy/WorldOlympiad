"""Independent file and glTF-structure admission for SceneCode visual revisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_AUDIT_METADATA = {".visual_revision_verdict.json", "visual_revision_verdict.json"}


def _files(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): _sha(path)
            for path in sorted(root.rglob("*")) if path.is_file()
            and path.name not in {"visual_variant_manifest.json", *_AUDIT_METADATA}}


def _resource_evidence(value: object) -> dict[str, str]:
    """Return protected-resource hashes, excluding the published audit record."""
    if not isinstance(value, dict):
        return {}
    return {str(name): str(digest) for name, digest in value.items()
            if Path(str(name)).name not in _AUDIT_METADATA}


def _audit_evidence(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(name): str(digest) for name, digest in value.items()
            if Path(str(name)).name in _AUDIT_METADATA}


def _audit_files(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): _sha(path)
            for path in sorted(root.rglob("*")) if path.is_file()
            and path.name in _AUDIT_METADATA}


def _glb_chunks(path: Path) -> tuple[dict, list[tuple[int, int, bytes]]]:
    """Parse a GLB and return JSON plus every non-JSON chunk verbatim.

    JSON is the only part a material revision may edit.  In particular, reading
    only the JSON chunk is insufficient: vertex/index data lives in BIN chunks.
    """
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"glTF":
        raise ValueError(f"invalid GLB header: {path}")
    version, total = struct.unpack_from("<II", data, 4)
    if version != 2 or total != len(data):
        raise ValueError(f"invalid GLB version or length: {path}")
    offset, json_value, protected = 12, None, []
    while offset < total:
        if offset + 8 > total:
            raise ValueError(f"truncated GLB chunk header: {path}")
        length, chunk_type = struct.unpack_from("<II", data, offset)
        end = offset + 8 + length
        if end > total:
            raise ValueError(f"truncated GLB chunk: {path}")
        chunk = data[offset + 8:end]
        if chunk_type == 0x4E4F534A:
            if json_value is not None:
                raise ValueError(f"GLB has multiple JSON chunks: {path}")
            try:
                json_value = json.loads(chunk.decode("utf-8").rstrip(" \\t\\r\\n\\0"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid GLB JSON chunk: {path}") from exc
        else:
            protected.append((length, chunk_type, chunk))
        offset = end
    if offset != total or json_value is None:
        raise ValueError(f"GLB must contain exactly one JSON chunk: {path}")
    return json_value, protected


def _gltf_json(path: Path) -> dict:
    """Read the JSON scene from either textual glTF or structurally valid GLB."""
    if path.suffix.lower() != ".glb":
        return json.loads(path.read_text())
    return _glb_chunks(path)[0]


def verify_visual_revision(*, manifest_path: Path, output: Path,
                           expected_source_sdf: Path | None = None,
                           source_sdf_override: Path | None = None,
                           write_output: bool = True) -> dict:
    manifest_path, output = manifest_path.resolve(), output.resolve()
    variant = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    declared_source = Path(manifest["source_sdf"])
    if not declared_source.is_absolute():
        # Packed references may name the certified inventory from the bundle
        # root, while legacy manifests resolve paths from the variant itself.
        candidates = (manifest_path.parent / declared_source,
                      manifest_path.parent.parent / declared_source)
        declared_source = next((candidate for candidate in candidates
                                if candidate.is_file()), candidates[0])
    declared_source = declared_source.resolve()
    source_sdf = (Path(source_sdf_override).resolve() if source_sdf_override is not None
                  else declared_source)
    if expected_source_sdf is not None and source_sdf != Path(expected_source_sdf).resolve():
        raise ValueError("visual revision is not bound to the selected certified SDF")
    # The certified SDF selected by the resolver may be a staged copy in a
    # reference output.  Its sibling directory is not necessarily the
    # authoritative resource inventory named by the manifest.  Compare mesh,
    # glTF and node resources with that bound inventory while using source_sdf
    # itself for the authoritative SDF graph/collision declarations.
    source = (declared_source.parent if source_sdf_override is not None
              else source_sdf.parent)
    source_files = _files(source)
    variant_files = _files(variant)
    if (manifest.get("schema_version") != "scenecode_visual_variant.v1"
            or manifest.get("physics_geometry_unchanged") is not True
            or manifest.get("gltf_geometry_and_other_materials_unchanged") is not True
            or not source_sdf.is_file()
            or _resource_evidence(manifest.get("source_file_sha256")) != source_files
            or _resource_evidence(manifest.get("output_file_sha256")) != variant_files
            or (_audit_evidence(manifest.get("source_file_sha256"))
                and _audit_evidence(manifest.get("source_file_sha256")) != _audit_files(source))
            or (_audit_evidence(manifest.get("output_file_sha256"))
                and _audit_evidence(manifest.get("output_file_sha256")) != _audit_files(variant))):
        raise ValueError("visual revision file evidence changed")
    changed = sorted(name for name, digest in variant_files.items()
                     if source_files.get(name) != digest)
    if (changed != sorted(manifest.get("changed_files") or [])
            or not changed or any(Path(name).suffix.lower() not in {".gltf", ".glb"} for name in changed)
            or _sha(source_sdf) != _sha(variant / source_sdf.name)):
        raise ValueError("visual revision changed non-material source geometry")
    tree = ET.parse(source_sdf).getroot()
    declared = {str(Path(uri.text).as_posix())
                for link in tree.findall(".//link")
                for uri in link.findall("visual/geometry/mesh/uri")
                if uri.text}
    collisions = {str(Path(uri.text).as_posix())
                  for uri in tree.findall(".//collision/geometry/mesh/uri")
                  if uri.text}
    if not set(changed) <= declared or set(changed) & collisions:
        raise ValueError("visual revision changed an unselected or collision mesh")
    allowed_pbr = {"baseColorFactor", "baseColorTexture", "metallicFactor",
                   "roughnessFactor", "metallicRoughnessTexture"}
    for relative in changed:
        before_path, after_path = source / relative, variant / relative
        before = _gltf_json(before_path)
        after = _gltf_json(after_path)
        if relative.lower().endswith(".glb"):
            # JSON padding may change with a material edit, but all binary
            # resources (including their order, lengths and bytes) are
            # geometry-bearing protected data.
            _, before_chunks = _glb_chunks(before_path)
            _, after_chunks = _glb_chunks(after_path)
            if before_chunks != after_chunks:
                raise ValueError("visual revision changed GLB binary geometry/resource data")
        if ({k: v for k, v in before.items() if k != "materials"}
                != {k: v for k, v in after.items() if k != "materials"}
                or len(before.get("materials") or []) != len(after.get("materials") or [])):
            raise ValueError("visual revision changed glTF geometry or material count")
        for old, new in zip(before["materials"], after["materials"], strict=True):
            if ({k: v for k, v in old.items() if k != "pbrMetallicRoughness"}
                    != {k: v for k, v in new.items() if k != "pbrMetallicRoughness"}):
                raise ValueError("visual revision changed non-PBR material fields")
            old_pbr = old.get("pbrMetallicRoughness") or {}
            new_pbr = new.get("pbrMetallicRoughness") or {}
            if ({k: v for k, v in old_pbr.items() if k not in allowed_pbr}
                    != {k: v for k, v in new_pbr.items() if k not in allowed_pbr}):
                raise ValueError("visual revision changed a protected PBR field")
    verdict = {
        "schema_version": "worldolympiad_visual_revision_verdict.v1",
        "passed": True,
        "source_sdf_sha256": _sha(source_sdf),
        "variant_sdf_sha256": _sha(variant / source_sdf.name),
        "visual_variant_manifest_sha256": _sha(manifest_path),
        "changed_files": changed,
        "collision_and_geometry_unchanged": True,
        "visual_review": "required_in_formal_reference_context",
    }
    if write_output:
        if output.exists():
            if json.loads(output.read_text()) != verdict:
                raise ValueError("published visual revision verdict changed")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sdf", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_visual_revision(
        manifest_path=args.manifest, output=args.output,
        expected_source_sdf=args.expected_source_sdf,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
