"""Convert a glTF visual to a Z-up OBJ bundle while preserving base textures."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import bpy


def _materialize_images(destination: Path, prefix: str) -> list[Path]:
    outputs: list[Path] = []
    ignored = {"Render Result", "Viewer Node"}
    for index, image in enumerate(item for item in bpy.data.images if item.name not in ignored):
        suffix = Path(image.filepath).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            suffix = ".png"
        path = destination / f"{prefix}_texture_{index:02d}{suffix}"
        if image.packed_file is not None:
            path.write_bytes(image.packed_file.data)
        else:
            source = Path(bpy.path.abspath(image.filepath))
            if not source.is_file():
                continue
            shutil.copy2(source, path)
        image.filepath = str(path)
        try:
            image.reload()
        except RuntimeError:
            pass
        outputs.append(path)
    return outputs


def _make_mtl_paths_local(mtl: Path, textures: list[Path]) -> None:
    if not mtl.exists() or not textures:
        return
    names = {path.name for path in textures}
    output: list[str] = []
    for line in mtl.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0].lower().startswith("map_") and Path(parts[-1]).name in names:
            parts[-1] = Path(parts[-1]).name
            line = " ".join(parts)
        output.append(line)
    mtl.write_text("\n".join(output) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    # Blender leaves its own arguments before a standalone ``--`` marker.
    argv = __import__("sys").argv
    args = parser.parse_args(argv[argv.index("--") + 1 :] if "--" in argv else argv[1:])

    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [item for item in bpy.context.scene.objects if item.type == "MESH"]
    if not meshes or not any(len(item.data.vertices) for item in meshes):
        raise RuntimeError(f"no mesh vertices imported from {source}")
    textures = _materialize_images(output.parent, output.stem)
    # glTF imports into Blender's Z-up coordinates.  Exporting OBJ explicitly
    # as Z-up keeps it aligned with WorldIR and PyBullet.
    bpy.ops.wm.obj_export(
        filepath=str(output),
        export_materials=True,
        export_pbr_extensions=True,
        path_mode="RELATIVE",
        export_triangulated_mesh=True,
        # Blender's OBJ exporter uses +Y as the source forward axis. -Y
        # rotates the complete imported geometry 180 degrees around Z.
        forward_axis="Y",
        up_axis="Z",
    )
    mtl = output.with_suffix(".mtl")
    _make_mtl_paths_local(mtl, textures)
    print(
        json.dumps(
            {
                "input": str(source),
                "output": str(output),
                "mesh_objects": len(meshes),
                "textures": [path.name for path in textures],
                "material": mtl.name if mtl.exists() else None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
