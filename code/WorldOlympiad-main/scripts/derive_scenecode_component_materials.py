"""Derive a visual-only, multi-component SceneCode material revision.

The source SDF, collision meshes and glTF geometry remain byte-identical.
Only named material values in distinct component glTF files may change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): _sha(path)
            for path in sorted(root.rglob("*")) if path.is_file()
            and path.name != "visual_variant_manifest.json"}


def derive(*, source_sdf: Path, plan_json: Path, output_dir: Path) -> Path:
    source_sdf, plan_json, output_dir = (
        Path(value).resolve() for value in (source_sdf, plan_json, output_dir))
    if output_dir.exists():
        raise FileExistsError("visual revision output already exists")
    plan = json.loads(plan_json.read_text())
    edits = plan.get("edits")
    if (plan.get("schema_version") != "worldolympiad_component_material_plan.v1"
            or not isinstance(edits, list) or not edits):
        raise ValueError("component material plan is invalid")
    root = source_sdf.parent
    if not source_sdf.is_file() or root in output_dir.parents:
        raise ValueError("source SDF is missing or output is inside its source")
    source_hashes = _files(root)
    tree = ET.parse(source_sdf).getroot()
    links = {link.get("name"): link for link in tree.findall(".//link")}
    selected: dict[str, list[tuple[int, dict]]] = {}
    for edit in edits:
        link = links.get(edit.get("link_name"))
        if link is None or len(link.findall("visual")) != 1:
            raise ValueError("material edit must select one visual on a declared link")
        uri = link.findtext("visual/geometry/mesh/uri")
        if not uri:
            raise ValueError("selected visual lacks a mesh")
        relative = Path(uri)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".gltf":
            raise ValueError("material edit requires a local textual glTF")
        source_gltf = (root / relative).resolve()
        source_gltf.relative_to(root)
        if not source_gltf.is_file():
            raise FileNotFoundError(source_gltf)
        rgba = edit.get("base_color_rgba")
        metallic = edit.get("metallic_factor")
        roughness = edit.get("roughness_factor")
        if (not isinstance(rgba, list) or len(rgba) != 4
                or not all(isinstance(x, (float, int)) and math.isfinite(x)
                           and 0 <= x <= 1 for x in rgba)
                or any(not isinstance(x, (float, int)) or not math.isfinite(x)
                       or not 0 <= x <= 1 for x in (metallic, roughness))):
            raise ValueError("material values must be finite RGBA, metallic and roughness")
        payload = json.loads(source_gltf.read_text())
        materials = payload.get("materials") or []
        material_name = edit.get("material_name")
        if material_name is None and len(materials) == 1:
            material_index = 0
        else:
            matches = [index for index, material in enumerate(materials)
                       if material.get("name") == material_name]
            if len(matches) != 1:
                raise ValueError("selected component glTF needs one unambiguous material")
            material_index = matches[0]
        changes = selected.setdefault(relative.as_posix(), [])
        if any(index == material_index for index, _ in changes):
            raise ValueError("material edits cannot repeat one component material")
        changes.append((material_index, edit))
    for collision in tree.findall(".//collision/geometry/mesh/uri"):
        if (collision.text or "").strip() in selected:
            raise ValueError("a selected visual is also a collision mesh")
    shutil.copytree(root, output_dir, symlinks=False)
    for relative, edits_for_file in selected.items():
        source_payload = json.loads((root / relative).read_text())
        output_payload = json.loads((output_dir / relative).read_text())
        for material_index, edit in edits_for_file:
            pbr = output_payload["materials"][material_index].setdefault(
                "pbrMetallicRoughness", {})
            pbr.pop("baseColorTexture", None)
            pbr.pop("metallicRoughnessTexture", None)
            pbr["baseColorFactor"] = [float(x) for x in edit["base_color_rgba"]]
            pbr["metallicFactor"] = float(edit["metallic_factor"])
            pbr["roughnessFactor"] = float(edit["roughness_factor"])
        if ({k: v for k, v in source_payload.items() if k != "materials"}
                != {k: v for k, v in output_payload.items() if k != "materials"}):
            raise RuntimeError("component material edit changed glTF geometry")
        (output_dir / relative).write_text(json.dumps(
            output_payload, sort_keys=True, separators=(",", ":")) + "\n")
    output_hashes = _files(output_dir)
    changed = sorted(name for name in output_hashes
                     if source_hashes.get(name) != output_hashes[name])
    if changed != sorted(selected) or _files(root) != source_hashes:
        raise RuntimeError("visual revision changed undeclared source or output files")
    manifest = {
        "schema_version": "scenecode_visual_variant.v1",
        "variant_id": str(plan.get("variant_id")),
        "source_sdf": str(source_sdf),
        "source_file_sha256": source_hashes,
        "output_file_sha256": output_hashes,
        "changed_files": changed,
        "component_material_plan_sha256": _sha(plan_json),
        "component_edits": edits,
        "gltf_geometry_and_other_materials_unchanged": True,
        "physics_geometry_unchanged": True,
        "provenance_claim": "visual-only material revision of certified SceneCode asset",
    }
    (output_dir / "visual_variant_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output_dir / source_sdf.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sdf", type=Path, required=True)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(derive(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
