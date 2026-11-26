"""Render a declared SDF visual assembly in an isolated Blender CPU process.

The caller supplies resolved visual transforms, the original mesh paths, and the
calibrated colour-faithful lighting plan owned by
``autoresearch.scenecode_identity_lighting``.  No asset is exported or
rewritten; imported materials retain their properties and no display-referred
filmic look is applied, so the rendered value tracks the declared material.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def main() -> None:
    import bpy
    from mathutils import Matrix, Vector

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    plan = json.loads(args.plan.read_text())
    lighting = plan["lighting"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 8
    scene.render.resolution_x = plan["width"]
    scene.render.resolution_y = plan["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    # Colour-faithful material display: the scene-referred Standard transform
    # plus the calibrated neutral environment.  A filmic look would make the
    # rendered value a function of the display transform instead of the asset.
    scene.view_settings.view_transform = lighting["view_transform"]
    scene.view_settings.look = lighting["look"]
    scene.view_settings.exposure = float(lighting["exposure"])
    scene.view_settings.gamma = 1
    scene.world = bpy.data.worlds.new("Neutral identity world")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (*lighting["world_color"], 1)
    background.inputs["Strength"].default_value = float(lighting["world_strength"])

    meshes = []
    for visual in plan["visuals"]:
        before = set(bpy.data.objects)
        path = Path(visual["mesh_path"])
        if path.suffix.lower() in {".glb", ".gltf"}:
            bpy.ops.import_scene.gltf(filepath=str(path))
        elif path.suffix.lower() == ".obj":
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            raise ValueError(f"unsupported native identity mesh: {path}")
        created = set(bpy.data.objects) - before
        transform = Matrix(visual["transform"]) @ Matrix.Diagonal((*visual["scale"], 1.0))
        # The glTF importer already converts Y-up geometry into Blender Z-up.
        # Apply the SDF frame exactly once to imported top-level objects.
        for obj in created:
            if obj.parent not in created:
                obj.matrix_world = transform @ obj.matrix_world
            if obj.type == "MESH":
                meshes.append(obj)
            elif obj.type in {"CAMERA", "LIGHT"}:
                bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    if not corners:
        raise ValueError("native identity assembly has no visible geometry")
    low = Vector([min(p[i] for p in corners) for i in range(3)])
    high = Vector([max(p[i] for p in corners) for i in range(3)])
    center = (low + high) / 2
    radius = max((high - low).length / 2, 0.001)
    distance = max(radius / math.sin(math.radians(21)) * 1.3, 0.25)
    for name, offset, energy in (
        (item[0], tuple(item[1]), float(item[2])) for item in lighting["area_lights"]
    ):
        light = bpy.data.lights.new(name, "AREA")
        light.energy = energy * radius ** 2
        light.shape = "DISK"
        light.size = radius * 3
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = center + Vector(offset) * radius
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()
    camera = bpy.data.cameras.new("Identity camera")
    camera.type = "PERSP"
    camera.sensor_fit = "VERTICAL"
    camera.angle = math.radians(42)
    camera.clip_start = max(radius * 0.001, 0.0001)
    camera.clip_end = max(distance * 10, 10)
    camera_object = bpy.data.objects.new("Identity camera", camera)
    scene.collection.objects.link(camera_object)
    scene.camera = camera_object
    records = []
    for index, (yaw, elevation) in enumerate(plan["views"]):
        y, e = math.radians(yaw), math.radians(elevation)
        direction = Vector((math.cos(e) * math.cos(y), math.cos(e) * math.sin(y), math.sin(e)))
        camera_object.location = center + distance * direction
        camera_object.rotation_euler = (center - camera_object.location).to_track_quat("-Z", "Y").to_euler()
        image_path = Path(plan["output_dir"]) / f'{plan["object_id"]}_view{index}.png'
        scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)
        records.append({"index": index, "yaw_deg": yaw, "elevation_deg": elevation,
                        "eye": list(camera_object.location), "image": str(image_path)})
    receipt = {"renderer": "blender_native_cpu", "blender_version": bpy.app.version_string,
               "source_materials_modified": False, "views": records,
               "bounds_low": list(low), "bounds_high": list(high),
               "lighting": lighting,
               "render_settings": {"engine": "CYCLES", "device": "CPU", "samples": 24,
                                   "view_transform": lighting["view_transform"],
                                   "exposure": float(lighting["exposure"])}}
    (Path(plan["output_dir"]) / "native_material_render.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
