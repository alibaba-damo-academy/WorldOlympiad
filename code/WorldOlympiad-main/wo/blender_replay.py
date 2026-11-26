"""Blender replay renderer (run with the SceneCode Python that provides ``bpy``).

    python blender_replay.py PLAN.json OUT_DIR SETTINGS.py

Imports every visual mesh once, then per frame only sets node and camera
transforms and renders each view.  Lighting/world/engine settings are the
official SceneCode ones (point light, regular world, robotics render settings).
Writes OUT_DIR/<view>/frame_00000.png ... and OUT_DIR/<view>.mp4.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import bpy
import mathutils

plan_path, out_dir, settings = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
FFMPEG = os.environ.get("WO_FFMPEG", "ffmpeg")
plan = json.loads(plan_path.read_text())
out_dir.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
try:
    from scenecode.agent_utils.blender.render_settings import setup_regular_world
    setup_regular_world()
except Exception as exc:  # pragma: no cover - fallback outside the SceneCode runtime
    print("world setup fallback:", exc)
    world = bpy.data.worlds.new("World")
    scene.world = world
light_data = bpy.data.lights.new(name="DefaultLight", type="POINT")
light_data.energy = 1000.0
light = bpy.data.objects.new(name="DefaultLight", object_data=light_data)
scene.collection.objects.link(light)
light.location = (4.0, 1.0, 6.0)


def M(flat):
    return mathutils.Matrix([flat[0:4], flat[4:8], flat[8:12], flat[12:16]])


nodes = {}
for name in plan["nodes"]:
    empty = bpy.data.objects.new(name.replace("/", "__"), None)
    scene.collection.objects.link(empty)
    nodes[name] = empty

cache = {}
for mesh in plan["meshes"]:
    path = mesh["file"]
    before = set(bpy.data.objects)
    suffix = Path(path).suffix.lower()
    if suffix in (".gltf", ".glb"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif suffix == ".obj":
        # URDF/SDF meshes are already in the link frame (z-up)
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    elif suffix == ".stl":
        bpy.ops.wm.stl_import(filepath=path, forward_axis="Y", up_axis="Z")
    elif suffix == ".dae":
        bpy.ops.wm.collada_import(filepath=path)
    else:
        print("skip unsupported mesh", path)
        continue
    new = [o for o in bpy.data.objects if o not in before]
    local = M([v for row in mesh["link_T_visual"] for v in row]) @ mathutils.Matrix.Diagonal(
        (*mesh["scale"], 1.0))
    parent = nodes[mesh["node"]]
    for obj in new:
        if obj.parent is None:
            basis = obj.matrix_basis.copy()
            obj.parent = parent
            obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            obj.matrix_basis = local @ basis

for primitive in plan.get("primitives", []):
    if primitive["kind"] != "box":
        raise ValueError(f"Unsupported replay primitive: {primitive['kind']}")
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.object
    obj.name = primitive["node"].replace("/", "__")
    obj.parent = nodes[primitive["node"]]
    obj.scale = primitive["size"]
    material = bpy.data.materials.new(obj.name)
    material.diffuse_color = primitive["rgba"]
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = primitive["rgba"]
    shader.inputs["Roughness"].default_value = 0.6
    obj.data.materials.append(material)

exec(compile(settings.read_text(), str(settings), "exec"), {"bpy": bpy}, {})
# The container CPU quota is far below the host CPU count.
scene.render.threads_mode = "FIXED"
scene.render.threads = int(os.environ.get("WO_RENDER_THREADS", "4"))
cam_data = bpy.data.cameras.new("cam")
cam_data.sensor_fit = "VERTICAL"
cam_data.clip_start = 0.01
cam_data.clip_end = 100.0
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
scene.render.image_settings.file_format = "PNG"

DEPTH_VIEWS = [v for v in os.environ.get("WO_DEPTH_VIEWS", "exterior_left").split(",") if v in plan["views"]]


def frame0_depth_and_calibration(view):
    """Metric z-depth of frame 0 by ray casting the visual meshes, plus OpenCV calibration."""
    import numpy as np

    w, h, fov = plan["views"][view]
    T = np.array(plan["cameras"][0][view]).reshape(4, 4)  # Blender/OpenGL camera-to-world
    fy = (h / 2) / np.tan(np.radians(fov) / 2)
    K = [[fy, 0.0, w / 2], [0.0, fy, h / 2], [0.0, 0.0, 1.0]]
    world_cam_cv = T @ np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV axes: x right, y down, z forward
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = mathutils.Vector(T[:3, 3])
    depth = np.zeros((int(h), int(w)), np.float32)
    R = world_cam_cv[:3, :3]
    for v in range(int(h)):
        for u in range(int(w)):
            d_cam = np.array([(u + 0.5 - w / 2) / fy, (v + 0.5 - h / 2) / fy, 1.0])
            hit, loc, *_ = scene.ray_cast(depsgraph, origin, mathutils.Vector(R @ d_cam))
            if hit:
                depth[v, u] = float((R.T @ (np.array(loc) - T[:3, 3]))[2])
    np.save(out_dir / f"depth_{view}_00000.npy", depth)
    (out_dir / f"calibration_{view}.json").write_text(json.dumps(
        {"head_camera": {"intrinsics": K, "extrinsics": world_cam_cv.tolist()}}, indent=1))


t0 = time.time()
for i, (frame, cams) in enumerate(zip(plan["frames"], plan["cameras"])):
    for name, flat in frame.items():
        nodes[name].matrix_world = M(flat)
    if i == 0 and os.environ.get("WO_FRAME0_ONLY") == "1":
        for view in DEPTH_VIEWS:
            frame0_depth_and_calibration(view)
        print("frame-0 calibration written", flush=True)
        os._exit(0)  # bpy teardown can hang after the leak detector runs
    for view, flat in cams.items():
        w, h, fov = plan["views"][view]
        scene.render.resolution_x, scene.render.resolution_y = int(w), int(h)
        cam_data.angle_y = fov * 3.141592653589793 / 180.0
        cam.matrix_world = M(flat)
        scene.render.filepath = str(out_dir / view / f"frame_{i:05d}.png")
        bpy.ops.render.render(write_still=True)
    if i == 0:
        for view in DEPTH_VIEWS:
            frame0_depth_and_calibration(view)
        if os.environ.get("WO_FRAME0_ONLY") == "1":
            print("frame-0 calibration written")
            sys.exit(0)
    if i % 25 == 0:
        print(f"frame {i}/{len(plan['frames'])} {time.time() - t0:.1f}s", flush=True)

for view in plan["views"]:
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(plan["fps"]),
                    "-i", str(out_dir / view / "frame_%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "16", str(out_dir / f"{view}.mp4")], check=True)
print("done", time.time() - t0, flush=True)
os._exit(0)  # bpy teardown can hang after the leak detector runs
