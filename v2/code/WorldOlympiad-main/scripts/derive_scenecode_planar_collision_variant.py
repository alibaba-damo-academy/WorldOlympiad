#!/usr/bin/env python3
"""Add a bounds-matched box collision to a collisionless planar SceneCode asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "physical_variant_manifest.json"
    }


def derive_planar_collision_variant(
    source_sdf: Path,
    output_dir: Path,
    *,
    variant_id: str,
    size_xyz_m: tuple[float, float, float],
) -> Path:
    """Copy an asset and add one centered box collision matching visual bounds."""

    source_sdf = source_sdf.resolve()
    output_dir = output_dir.resolve()
    if not source_sdf.is_file():
        raise FileNotFoundError(source_sdf)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite physical variant: {output_dir}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", variant_id):
        raise ValueError("unsafe physical variant id")
    if len(size_xyz_m) != 3 or any(
        not math.isfinite(value) or value <= 0.0 or value > 5.0
        for value in size_xyz_m
    ):
        raise ValueError("planar collision dimensions must be finite positive xyz")
    if size_xyz_m[2] > 0.10:
        raise ValueError("planar collision variant requires a thin vertical extent")

    source_root = source_sdf.parent
    source_hashes = _files(source_root)
    shutil.copytree(source_root, output_dir)
    (output_dir / "visual_variant_manifest.json").unlink(missing_ok=True)
    (output_dir / "physical_variant_manifest.json").unlink(missing_ok=True)
    output_sdf = output_dir / source_sdf.name
    tree = ET.parse(output_sdf)
    model = tree.getroot().find("model")
    if model is None:
        raise ValueError("SceneCode SDF has no model")
    if any(list(node) for node in model.findall(".//collision/geometry")):
        raise ValueError("planar collision variant requires a collisionless source")
    links = model.findall("link")
    if len(links) != 1:
        raise ValueError("planar collision variant requires exactly one model link")

    link = links[0]
    collision = ET.SubElement(
        link, "collision", {"name": "worldolympiad_planar_region_collision"}
    )
    pose = ET.SubElement(collision, "pose")
    pose.text = f"0 0 {size_xyz_m[2] / 2.0:.12g} 0 0 0"
    geometry = ET.SubElement(collision, "geometry")
    box = ET.SubElement(geometry, "box")
    size = ET.SubElement(box, "size")
    size.text = " ".join(f"{value:.12g}" for value in size_xyz_m)

    ET.indent(tree, space="  ")
    tree.write(output_sdf, encoding="utf-8", xml_declaration=True)
    output_hashes = _files(output_dir)
    changed = sorted(
        relative
        for relative, digest in output_hashes.items()
        if source_hashes.get(relative) != digest
    )
    if changed != [source_sdf.name]:
        raise RuntimeError(f"planar collision invariant failed: changed={changed}")

    manifest = output_dir / "physical_variant_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "scenecode_physical_variant.v2",
                "variant_id": variant_id,
                "source_sdf": str(source_sdf),
                "transformation": {
                    "kind": "add_planar_box_collision",
                    "size_xyz_m": list(size_xyz_m),
                    "pose_xyz_rpy": [0.0, 0.0, size_xyz_m[2] / 2.0, 0.0, 0.0, 0.0],
                },
                "source_file_sha256": source_hashes,
                "output_file_sha256": output_hashes,
                "changed_files": changed,
                "visual_geometry_unchanged": True,
                "collision_geometry_matches_visual_bounds": True,
                "provenance_claim": (
                    "bounds-matched physical derivative of a collisionless "
                    "registered SceneCode planar task region"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_sdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--size-xyz-m", type=float, nargs=3, required=True)
    args = parser.parse_args()
    print(
        derive_planar_collision_variant(
            args.source_sdf,
            args.output_dir,
            variant_id=args.variant_id,
            size_xyz_m=tuple(args.size_xyz_m),
        )
    )


if __name__ == "__main__":
    main()
