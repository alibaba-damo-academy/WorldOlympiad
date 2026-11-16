"""Blender-side renderer for SceneCode assets animated by PyBullet link poses."""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector


def _numbers(value: str | None, count: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if not value:
        return default
    parsed = tuple(float(item) for item in value.split())
    if len(parsed) != count:
        raise ValueError(f"expected {count} values, got {value!r}")
    return parsed


def _origin_matrix(node: ET.Element | None) -> Matrix:
    if node is None:
        return Matrix.Identity(4)
    xyz = _numbers(node.get("xyz"), 3, (0.0, 0.0, 0.0))
    rpy = _numbers(node.get("rpy"), 3, (0.0, 0.0, 0.0))
    return Matrix.Translation(Vector(xyz)) @ Euler(rpy, "XYZ").to_matrix().to_4x4()


def _scale_matrix(scale: tuple[float, float, float]) -> Matrix:
    matrix = Matrix.Identity(4)
    matrix[0][0], matrix[1][1], matrix[2][2] = scale
    return matrix


def _material(name: str, rgba: tuple[float, float, float, float]):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = rgba
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = 0.48
        bsdf.inputs["Metallic"].default_value = 0.0
    return material


def _resolve_mesh(filename: str, urdf_path: Path) -> Path:
    if filename.startswith("package://"):
        relative = filename[len("package://") :]
        package, _, rest = relative.partition("/")
        candidates = [urdf_path.parent / package / rest, urdf_path.parent / rest]
    else:
        path = Path(filename)
        candidates = [path] if path.is_absolute() else [urdf_path.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve {filename!r} from {urdf_path}")


def _import_mesh(path: Path, *, preserve_obj_axes: bool) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        if preserve_obj_axes:
            # Released robot packages author mesh vertices directly in URDF
            # link coordinates.  Applying Blender's generic Y-up conversion
            # would rotate each link visual away from its kinematic frame.
            bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
        else:
            # Match Bullet's convention for SceneCode's generated Wavefront
            # exports, which retain a Y-up object frame.
            bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".stl":
        bpy.ops.wm.stl_import(filepath=str(path))
    else:
        raise RuntimeError(f"unsupported Blender replay mesh format: {path}")
    return [item for item in bpy.data.objects if item not in before]


def _primitive(geometry: ET.Element) -> bpy.types.Object:
    child = list(geometry)[0]
    if child.tag == "box":
        size = _numbers(child.get("size"), 3, (1.0, 1.0, 1.0))
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.object
        obj.scale = size
    elif child.tag == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24)
        obj = bpy.context.object
        obj.scale = (float(child.get("radius", "1")),) * 3
    elif child.tag == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=1.0, depth=1.0)
        obj = bpy.context.object
        obj.scale = (
            float(child.get("radius", "1")),
            float(child.get("radius", "1")),
            float(child.get("length", "1")),
        )
    else:
        raise RuntimeError(f"not a primitive geometry: {child.tag}")
    bpy.context.view_layer.update()
    return obj


def _rgba_for_visual(
    visual: ET.Element,
    global_materials: dict[str, tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    material = visual.find("material")
    if material is None:
        return (0.55, 0.55, 0.55, 1.0)
    color = material.find("color")
    if color is not None:
        return _numbers(color.get("rgba"), 4, (0.55, 0.55, 0.55, 1.0))
    return global_materials.get(material.get("name", ""), (0.55, 0.55, 0.55, 1.0))


def _load_body(
    body_name: str,
    body_manifest: dict,
    urdf_path: Path,
    link_roots: dict[str, bpy.types.Object],
    missing_visuals: list[str],
) -> int:
    robot = ET.parse(urdf_path).getroot()
    global_materials = {}
    for material in robot.findall("material"):
        color = material.find("color")
        if material.get("name") and color is not None:
            global_materials[material.get("name")] = _numbers(
                color.get("rgba"), 4, (0.55, 0.55, 0.55, 1.0)
            )
    visual_count = 0
    metadata = body_manifest.get("metadata", {})
    force_urdf_material = bool(metadata.get("simready_visual_rgba"))
    preserve_obj_axes = bool(
        metadata.get("precompiled_urdf_source")
        or metadata.get("visual_mesh_coordinate_frame") == "worldir_z_up"
        or metadata.get("simready_analytic_frame_policy")
        == "geometry_centered_link_frame"
    )
    for link in robot.findall("link"):
        link_name = link.get("name")
        key = f"{body_name}::{link_name}"
        root = bpy.data.objects.new(f"PB::{key}", None)
        root.empty_display_type = "PLAIN_AXES"
        bpy.context.scene.collection.objects.link(root)
        link_roots[key] = root
        for visual_index, visual in enumerate(link.findall("visual")):
            visual_root = bpy.data.objects.new(f"VIS::{key}::{visual_index}", None)
            bpy.context.scene.collection.objects.link(visual_root)
            visual_root.parent = root
            visual_root.matrix_parent_inverse = Matrix.Identity(4)
            geometry = visual.find("geometry")
            if geometry is None or not list(geometry):
                missing_visuals.append(f"{key}:{visual_index}:missing_geometry")
                continue
            child = list(geometry)[0]
            scale = (1.0, 1.0, 1.0)
            try:
                if child.tag == "mesh":
                    scale = _numbers(child.get("scale"), 3, (1.0, 1.0, 1.0))
                    imported = _import_mesh(
                        _resolve_mesh(child.get("filename", ""), urdf_path),
                        preserve_obj_axes=preserve_obj_axes,
                    )
                else:
                    imported = [_primitive(geometry)]
            except Exception as exc:
                missing_visuals.append(f"{key}:{visual_index}:{exc}")
                continue
            visual_root.matrix_basis = _origin_matrix(visual.find("origin")) @ _scale_matrix(scale)
            rgba = _rgba_for_visual(visual, global_materials)
            fallback = _material(f"Fallback::{body_name}::{visual_index}", rgba)
            for obj in imported:
                original = obj.matrix_world.copy()
                obj.parent = visual_root
                obj.matrix_parent_inverse = Matrix.Identity(4)
                obj.matrix_basis = original
                obj.name = f"GEO::{body_name}::{link_name}::{visual_index}::{obj.name}"
                obj["worldolympiad_body"] = body_name
                obj["worldolympiad_link"] = link_name
                if obj.type == "MESH":
                    if force_urdf_material:
                        obj.data.materials.clear()
                        obj.data.materials.append(fallback)
                    elif not obj.data.materials:
                        obj.data.materials.append(fallback)
                    for polygon in obj.data.polygons:
                        polygon.use_smooth = False
                    visual_count += 1
    return visual_count


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _setup_camera(calibration: dict, width: int, height: int) -> bpy.types.Object:
    data = bpy.data.cameras.new("FormalCamera")
    camera = bpy.data.objects.new("FormalCamera", data)
    bpy.context.scene.collection.objects.link(camera)
    cv = np.asarray(calibration["camera_to_world_opencv"], dtype=np.float64)
    conversion = np.diag([1.0, -1.0, -1.0, 1.0])
    camera.matrix_world = Matrix((cv @ conversion).tolist())
    data.type = "PERSP"
    data.angle_y = math.radians(float(calibration["fov_y_deg"]))
    data.clip_start = float(calibration.get("near_m", 0.02))
    data.clip_end = float(calibration.get("far_m", 20.0))
    data.sensor_fit = "VERTICAL"
    bpy.context.scene.camera = camera
    return camera


def _body_bounds() -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, list[Vector]] = {}
    for obj in bpy.context.scene.objects:
        body = obj.get("worldolympiad_body")
        if body is None or obj.type != "MESH":
            continue
        grouped.setdefault(str(body), []).extend(
            obj.matrix_world @ Vector(corner) for corner in obj.bound_box
        )
    result = {}
    for body, corners in grouped.items():
        result[body] = {
            "minimum": [min(float(point[axis]) for point in corners) for axis in range(3)],
            "maximum": [max(float(point[axis]) for point in corners) for axis in range(3)],
        }
    return result


def _area_light(name: str, location: Vector, target: Vector, energy: float, size: float, color):
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy, data.shape, data.size, data.color = energy, "DISK", size, color
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    _look_at(obj, target)


def _setup_lighting(camera: bpy.types.Object, calibration: dict) -> None:
    target = Vector(calibration["look_at"])
    eye = Vector(calibration["eye"])
    view = eye - target
    horizontal = Vector((view.x, view.y, 0.0))
    if horizontal.length < 1e-6:
        horizontal = Vector((1.0, 0.0, 0.0))
    horizontal.normalize()
    side = Vector((-horizontal.y, horizontal.x, 0.0))
    _area_light("Key_Area", target + horizontal * 1.5 + side * 1.2 + Vector((0, 0, 2.4)), target, 220, 2.5, (1.0, 0.82, 0.68))
    _area_light("Fill_Area", target - horizontal * 0.6 - side * 1.8 + Vector((0, 0, 1.5)), target, 90, 2.0, (0.68, 0.82, 1.0))
    _area_light("Rim_Area", target - horizontal * 1.8 + side * 0.4 + Vector((0, 0, 2.1)), target, 140, 1.8, (1.0, 0.95, 0.82))
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.055, 0.065, 0.085, 1.0)
    background.inputs["Strength"].default_value = 0.10


def _render_instance_mask(scene, path: Path, selected_body: str | None) -> None:
    previous_path = scene.render.filepath
    previous_mode = scene.render.image_settings.color_mode
    previous_view_transform = scene.view_settings.view_transform
    previous_look = scene.view_settings.look
    previous_exposure = scene.view_settings.exposure
    previous_gamma = scene.view_settings.gamma
    previous_dither = scene.render.dither_intensity
    material = bpy.data.materials.get("ReplayMaskWhite") or bpy.data.materials.new(
        "ReplayMaskWhite"
    )
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 100.0
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    selected_object_names = [
        obj.name
        for obj in scene.objects
        if obj.type == "MESH" and obj.get("worldolympiad_body") == selected_body
    ]
    print(
        f"BLENDER_REPLAY_MASK {selected_body} selected={len(selected_object_names)}",
        flush=True,
    )
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.dither_intensity = 0.0
    previous_override = scene.view_layers[0].material_override
    previous_visibility = {
        obj.name: obj.hide_render for obj in scene.objects if obj.type == "MESH"
    }
    for obj in scene.objects:
        if obj.type == "MESH":
            obj.hide_render = not bool(
                selected_body is not None
                and obj.get("worldolympiad_body") == selected_body
            )
            obj.update_tag()
    scene.view_layers[0].material_override = material
    world_background = scene.world.node_tree.nodes.get("Background")
    previous_background_color = world_background.inputs["Color"].default_value[:]
    previous_background_strength = world_background.inputs["Strength"].default_value
    world_background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    world_background.inputs["Strength"].default_value = 0.0
    bpy.context.view_layer.update()
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    scene.view_layers[0].material_override = previous_override
    for obj_name, hide_render in previous_visibility.items():
        obj = bpy.data.objects[obj_name]
        obj.hide_render = hide_render
        obj.update_tag()
    world_background.inputs["Color"].default_value = previous_background_color
    world_background.inputs["Strength"].default_value = previous_background_strength
    bpy.context.view_layer.update()
    scene.render.filepath = previous_path
    scene.render.image_settings.color_mode = previous_mode
    scene.view_settings.view_transform = previous_view_transform
    scene.view_settings.look = previous_look
    scene.view_settings.exposure = previous_exposure
    scene.view_settings.gamma = previous_gamma
    scene.render.dither_intensity = previous_dither


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bundle = json.loads(Path(config["bundle_path"]).read_text(encoding="utf-8"))
    calibration = json.loads(Path(config["calibration_path"]).read_text(encoding="utf-8"))
    with np.load(config["trajectory_path"], allow_pickle=False) as payload:
        timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
        trajectory_names = [str(item) for item in payload["link_names"].tolist()]
        positions = np.asarray(payload["link_positions"], dtype=np.float64)
        orientations = np.asarray(payload["link_orientations_xyzw"], dtype=np.float64)
    if positions.shape[:2] != (len(timestamps), len(trajectory_names)):
        raise RuntimeError("link trajectory position shape does not match names/timestamps")
    if orientations.shape[:2] != (len(timestamps), len(trajectory_names)):
        raise RuntimeError("link trajectory orientation shape does not match names/timestamps")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    link_roots: dict[str, bpy.types.Object] = {}
    missing_visuals: list[str] = []
    body_visual_counts: dict[str, int] = {}
    for body_name, body in bundle["bodies"].items():
        body_visual_counts[body_name] = _load_body(
            body_name, body, Path(body["urdf_path"]), link_roots, missing_visuals
        )
    target_name = config["target_name"]
    robot_name = config.get("robot_name")
    for obj in bpy.context.scene.objects:
        body_name = obj.get("worldolympiad_body")
        obj.pass_index = 1 if body_name == target_name else (2 if body_name == robot_name else 0)
    missing_links = sorted(set(trajectory_names) - set(link_roots))
    extra_links = sorted(set(link_roots) - set(trajectory_names))
    if missing_links:
        raise RuntimeError(f"trajectory links have no visual scene nodes: {missing_links}")

    fps = int(config["fps"])
    frame_count = int(config["expected_frames"])
    frame_times = np.arange(frame_count, dtype=np.float64) / fps + timestamps[0]
    sample_indices = np.searchsorted(timestamps, frame_times, side="left")
    sample_indices = np.clip(sample_indices, 0, len(timestamps) - 1)
    previous = np.maximum(sample_indices - 1, 0)
    choose_previous = np.abs(timestamps[previous] - frame_times) < np.abs(
        timestamps[sample_indices] - frame_times
    )
    sample_indices[choose_previous] = previous[choose_previous]
    name_to_index = {name: index for index, name in enumerate(trajectory_names)}
    for frame_offset, sample_index in enumerate(sample_indices):
        frame_number = frame_offset + 1
        for name, root in link_roots.items():
            if name not in name_to_index:
                continue
            index = name_to_index[name]
            root.location = positions[sample_index, index].tolist()
            x, y, z, w = orientations[sample_index, index]
            root.rotation_mode = "QUATERNION"
            root.rotation_quaternion = (w, x, y, z)
            root.keyframe_insert(data_path="location", frame=frame_number)
            root.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)
    for action in bpy.data.actions:
        for curve in action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"

    camera = _setup_camera(calibration, int(config["width"]), int(config["height"]))
    _setup_lighting(camera, calibration)
    scene = bpy.context.scene
    scene.render.engine = config["render_engine"]
    scene.render.resolution_x = int(config["width"])
    scene.render.resolution_y = int(config["height"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.fps = fps
    scene.frame_start, scene.frame_end = 1, frame_count
    scene.render.image_settings.color_depth = "8"
    scene.render.use_file_extension = True
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.view_settings.look = "AgX - Medium High Contrast"
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = 64
    masks_dir = Path(config["masks_dir"]) if config.get("masks_dir") else None

    frames_dir = Path(config["frames_dir"])
    for frame_offset in range(frame_count):
        frame_number = frame_offset + 1
        scene.frame_set(frame_number)
        scene.render.filepath = str(frames_dir / f"{frame_offset:04d}.png")
        bpy.ops.render.render(write_still=True)
        if masks_dir is not None:
            _render_instance_mask(
                scene, masks_dir / f"target_{frame_offset:04d}.png", target_name
            )
            _render_instance_mask(
                scene, masks_dir / f"robot_{frame_offset:04d}.png", robot_name
            )
        if frame_offset % 10 == 0 or frame_number == frame_count:
            print(f"BLENDER_REPLAY_FRAME {frame_number}/{frame_count}", flush=True)
    scene.frame_set(1)
    bpy.context.view_layer.update()
    camera_forward = -(camera.matrix_world.to_3x3().col[2])
    report = {
        "schema_version": "blender_replay_script_report.v1",
        "render_engine": scene.render.engine,
        "blender_version": bpy.app.version_string,
        "trajectory_link_count": len(trajectory_names),
        "render_link_count": len(link_roots),
        "missing_trajectory_links": missing_links,
        "extra_render_links": extra_links,
        "missing_visuals": missing_visuals,
        "body_visual_counts": body_visual_counts,
        "target_visual_count": body_visual_counts.get(config["target_name"], 0),
        "instance_mask_mode": (
            "amodal_selected_body_silhouette" if masks_dir is not None else None
        ),
        "frame_count": frame_count,
        "sample_indices": sample_indices.tolist(),
        "camera_location": list(camera.matrix_world.translation),
        "camera_forward": list(camera_forward),
        "camera_look_at": calibration["look_at"],
        "body_bounds_at_first_frame": _body_bounds(),
    }
    report_path = args.config.parent / "blender_script_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"BLENDER_REPLAY_REPORT {report_path}", flush=True)


if __name__ == "__main__":
    main()
