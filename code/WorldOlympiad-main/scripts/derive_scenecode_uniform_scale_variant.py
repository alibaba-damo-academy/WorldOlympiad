#!/usr/bin/env python3
"""Derive a uniformly scaled physical asset from a SceneCode SDF bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "physical_variant_manifest.json"
    }


def _scaled_triplet(text: str | None, scale: float) -> str:
    values = [float(value) for value in (text or "0 0 0").split()]
    if len(values) != 3:
        raise ValueError(f"expected xyz triplet, got {text!r}")
    return " ".join(f"{value * scale:.12g}" for value in values)


def _scaled_pose(text: str | None, scale: float) -> str:
    """Scale the translation of an ``xyz rpy`` pose and keep the rotation."""

    values = [float(value) for value in (text or "0 0 0").split()]
    if len(values) not in {3, 6}:
        raise ValueError(f"expected an xyz or xyz-rpy pose, got {text!r}")
    return " ".join(
        [
            *(f"{value * scale:.12g}" for value in values[:3]),
            *(f"{value:.12g}" for value in values[3:]),
        ]
    )


def _scale_primitive_geometry(model: ET.Element, scale: float) -> int:
    """Scale non-mesh primitives, which carry their size in the SDF itself."""

    scaled = 0
    for box in model.findall(".//geometry/box"):
        size = box.find("size")
        if size is not None and size.text:
            size.text = _scaled_triplet(size.text, scale)
            scaled += 1
    for cylinder in model.findall(".//geometry/cylinder"):
        for tag in ("radius", "length"):
            node = cylinder.find(tag)
            if node is not None and node.text:
                node.text = f"{float(node.text) * scale:.12g}"
                scaled += 1
    for sphere in model.findall(".//geometry/sphere"):
        radius = sphere.find("radius")
        if radius is not None and radius.text:
            radius.text = f"{float(radius.text) * scale:.12g}"
            scaled += 1
    return scaled


def _scale_linkage(model: ET.Element, scale: float) -> tuple[int, int]:
    """Move every link and joint offset with the geometry it positions.

    A jointed SceneCode asset stores its assembly offsets in the link poses (and
    in the joint poses of nested models).  Scaling only the visual and collision
    poses of a jointed asset shrinks each part in place and silently desynchronises
    the articulation, so the link and joint translations must scale too.
    """

    links = 0
    joints = 0
    for link in model.findall("link"):
        pose = link.find("pose")
        if pose is not None and pose.text:
            pose.text = _scaled_pose(pose.text, scale)
            links += 1
    for joint in model.findall("joint"):
        pose = joint.find("pose")
        if pose is not None and pose.text:
            pose.text = _scaled_pose(pose.text, scale)
            joints += 1
    return links, joints


def derive_uniform_scale_variant(
    source_sdf: Path,
    output_dir: Path,
    *,
    variant_id: str,
    scale: float,
) -> Path:
    source_sdf = source_sdf.resolve()
    output_dir = output_dir.resolve()
    if not source_sdf.is_file():
        raise FileNotFoundError(source_sdf)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite physical variant: {output_dir}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", variant_id):
        raise ValueError("unsafe physical variant id")
    if not 0.1 <= scale <= 2.0:
        raise ValueError("uniform scale must lie in [0.1, 2.0]")

    source_root = source_sdf.parent
    source_hashes = _files(source_root)
    shutil.copytree(source_root, output_dir)
    # A manifest copied from an earlier visual-only derivative no longer
    # describes this directory once physical geometry is uniformly scaled.
    (output_dir / "visual_variant_manifest.json").unlink(missing_ok=True)
    (output_dir / "physical_variant_manifest.json").unlink(missing_ok=True)
    output_sdf = output_dir / source_sdf.name
    tree = ET.parse(output_sdf)
    model = tree.getroot().find("model")
    if model is None:
        raise ValueError("SceneCode SDF has no model")

    mesh_count = 0
    for mesh in model.findall(".//geometry/mesh"):
        scale_node = mesh.find("scale")
        existing = [1.0, 1.0, 1.0]
        if scale_node is None:
            scale_node = ET.SubElement(mesh, "scale")
        elif scale_node.text:
            existing = [float(value) for value in scale_node.text.split()]
            if len(existing) != 3:
                raise ValueError("mesh scale must contain three values")
        scale_node.text = " ".join(f"{value * scale:.12g}" for value in existing)
        mesh_count += 1
    primitive_count = _scale_primitive_geometry(model, scale)
    link_pose_count, joint_pose_count = _scale_linkage(model, scale)
    if mesh_count == 0 and primitive_count == 0:
        raise ValueError("physical scale variant requires actual geometry")

    for link in model.findall("link"):
        for pose in link.findall("./visual/pose") + link.findall("./collision/pose"):
            values = [float(value) for value in (pose.text or "").split()]
            if len(values) == 6:
                pose.text = " ".join(
                    [*(f"{value * scale:.12g}" for value in values[:3]),
                     *(f"{value:.12g}" for value in values[3:])]
                )
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = inertial.find("mass")
        if mass is not None and mass.text:
            mass.text = f"{float(mass.text) * scale**3:.12g}"
        pose = inertial.find("pose")
        if pose is not None and pose.text:
            values = [float(value) for value in pose.text.split()]
            if len(values) == 6:
                pose.text = " ".join(
                    [*(f"{value * scale:.12g}" for value in values[:3]),
                     *(f"{value:.12g}" for value in values[3:])]
                )
        inertia = inertial.find("inertia")
        if inertia is not None:
            for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
                node = inertia.find(key)
                if node is not None and node.text:
                    node.text = f"{float(node.text) * scale**5:.12g}"

    ET.indent(tree, space="  ")
    tree.write(output_sdf, encoding="utf-8", xml_declaration=True)
    output_hashes = _files(output_dir)
    changed = sorted(
        relative
        for relative, digest in output_hashes.items()
        if source_hashes.get(relative) != digest
    )
    if changed != [source_sdf.name]:
        raise RuntimeError(f"uniform-scale invariant failed: changed={changed}")
    manifest = output_dir / "physical_variant_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "scenecode_physical_scale_variant.v1",
                "variant_id": variant_id,
                "source_sdf": str(source_sdf),
                "uniform_scale": scale,
                "mass_scale": scale**3,
                "inertia_scale": scale**5,
                "scaled_mesh_count": mesh_count,
                "scaled_primitive_count": primitive_count,
                "scaled_link_pose_count": link_pose_count,
                "scaled_joint_pose_count": joint_pose_count,
                "articulation_scale_coherent": True,
                "source_file_sha256": source_hashes,
                "output_file_sha256": output_hashes,
                "changed_files": changed,
                "visual_collision_scale_coherent": True,
                "provenance_claim": "uniform physical derivative of the registered SceneCode asset",
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
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    print(
        derive_uniform_scale_variant(
            args.source_sdf,
            args.output_dir,
            variant_id=args.variant_id,
            scale=args.scale,
        )
    )


if __name__ == "__main__":
    main()
