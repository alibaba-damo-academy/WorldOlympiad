"""Preserve a declared glTF scene's surfaces and textures in a native USD asset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    source, output = args.input.resolve(), args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not meshes or not any(len(obj.data.polygons) for obj in meshes):
        raise ValueError(f'no visual geometry: {source}')
    for obj in bpy.context.scene.objects:
        obj.select_set(obj.type == 'MESH')
    # Both Blender imports and WorldIR use Z-up. Export the imported scene
    # graph once, retaining individual materials, UVs and base-color textures.
    bpy.ops.wm.usd_export(
        filepath=str(output), selected_objects_only=True,
        export_animation=False, export_materials=True,
        generate_preview_surface=True, export_textures=True,
        relative_paths=True, root_prim_path='/Asset',
        export_cameras=False, export_lights=False,
    )
    print(json.dumps({'input': str(source), 'output': str(output),
                      'meshes': len(meshes),
                      'materials': len(bpy.data.materials)}))


if __name__ == '__main__':
    main()
