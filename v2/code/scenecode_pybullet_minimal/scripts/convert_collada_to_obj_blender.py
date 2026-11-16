#!/usr/bin/env python3
"""Convert COLLADA meshes to material-preserving OBJ files inside Blender.

This script is executed by Blender, not by the bridge Python environment.  It
keeps the source's Z-up, metre-based coordinates unchanged so that the visual
geometry remains registered with the URDF collision geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh_bounds(objects: list[bpy.types.Object]) -> dict[str, list[float]]:
    points = [
        obj.matrix_world @ vertex.co
        for obj in objects
        for vertex in obj.data.vertices
    ]
    if not points:
        raise RuntimeError("COLLADA import produced no mesh vertices")
    return {
        "minimum_m": [min(point[axis] for point in points) for axis in range(3)],
        "maximum_m": [max(point[axis] for point in points) for axis in range(3)],
    }


def _convert(source: Path, destination: Path) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.wm.collada_import(
        filepath=str(source),
        import_units=False,
        fix_orientation=False,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender failed to import {source}: {result}")
    mesh_objects = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "MESH"),
        key=lambda obj: obj.name,
    )
    if not mesh_objects:
        raise RuntimeError(f"COLLADA has no mesh objects: {source}")
    bounds = _mesh_bounds(mesh_objects)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.wm.obj_export(
        filepath=str(destination),
        export_selected_objects=True,
        forward_axis="NEGATIVE_Y",
        up_axis="Z",
        global_scale=1.0,
        apply_modifiers=True,
        export_uv=True,
        export_normals=True,
        export_materials=True,
        path_mode="RELATIVE",
        export_triangulated_mesh=True,
    )
    if "FINISHED" not in result or not destination.is_file():
        raise RuntimeError(f"Blender failed to export {destination}: {result}")
    material = destination.with_suffix(".mtl")
    return {
        "source": str(source),
        "source_sha256": _sha256(source),
        "destination": str(destination),
        "destination_sha256": _sha256(destination),
        "material": str(material) if material.is_file() else None,
        "material_sha256": _sha256(material) if material.is_file() else None,
        "mesh_object_count": len(mesh_objects),
        "world_coordinate_bounds": bounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pairs",
        nargs="+",
        help="alternating source.dae and destination.obj paths",
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if len(args.pairs) % 2:
        raise ValueError("expected alternating COLLADA source and OBJ destination paths")
    records = []
    for index in range(0, len(args.pairs), 2):
        source = Path(args.pairs[index]).resolve()
        destination = Path(args.pairs[index + 1]).resolve()
        if not source.is_file() or source.suffix.lower() != ".dae":
            raise FileNotFoundError(f"invalid COLLADA source: {source}")
        if destination.suffix.lower() != ".obj":
            raise ValueError(f"OBJ destination required: {destination}")
        records.append(_convert(source, destination))
    print("ROBOLAB_COLLADA_CONVERSION=" + json.dumps(records, sort_keys=True))


if __name__ == "__main__":
    main()
