"""
Headless Blender socket server (addon-free).

This script runs inside Blender background mode and exposes a small JSON-over-TCP
command server compatible with `BlenderMCPClient`:
- request:  {"type": "<command>", "params": {...}}\n
- response: {"status": "success", "result": ...}\n
           or {"status": "error", "message": "..."}\n
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import tempfile
import selectors
import socket
import traceback
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import bpy
import mathutils


class BlenderHeadlessServer:
    """Minimal command server running in Blender background mode."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9876):
        self.host = host
        self.port = int(port)
        self.running = False
        self.selector = selectors.DefaultSelector()
        self.listen_sock: Optional[socket.socket] = None

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listen_sock.bind((self.host, self.port))
        self.listen_sock.listen(16)
        self.listen_sock.setblocking(False)
        self.selector.register(self.listen_sock, selectors.EVENT_READ, data=None)
        self.running = True
        print(f"[headless-server] listening on {self.host}:{self.port}", flush=True)

    def stop(self) -> None:
        self.running = False
        if self.listen_sock is not None:
            try:
                self.selector.unregister(self.listen_sock)
            except Exception:
                pass
            try:
                self.listen_sock.close()
            except Exception:
                pass
            self.listen_sock = None
        print("[headless-server] stopped", flush=True)

    def _close_client(self, sock: socket.socket) -> None:
        try:
            self.selector.unregister(sock)
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass

    def run_forever(self) -> None:
        self.start()
        try:
            while self.running:
                events = self.selector.select(timeout=0.2)
                for key, mask in events:
                    if key.data is None:
                        self._accept_client()
                    else:
                        self._service_client(key, mask)
        finally:
            for key in list(self.selector.get_map().values()):
                sock = key.fileobj
                if isinstance(sock, socket.socket):
                    self._close_client(sock)
            self.stop()

    def _accept_client(self) -> None:
        assert self.listen_sock is not None
        conn, addr = self.listen_sock.accept()
        conn.setblocking(False)
        data = SimpleNamespace(addr=addr, inb=bytearray())
        self.selector.register(conn, selectors.EVENT_READ, data=data)

    def _send_json(self, sock: socket.socket, payload: Dict[str, Any]) -> None:
        raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        sock.sendall(raw)

    def _service_client(self, key: selectors.SelectorKey, mask: int) -> None:
        sock = key.fileobj
        data = key.data
        if not isinstance(sock, socket.socket):
            return
        if not (mask & selectors.EVENT_READ):
            return
        try:
            chunk = sock.recv(65536)
        except Exception:
            self._close_client(sock)
            return
        if not chunk:
            self._close_client(sock)
            return

        data.inb.extend(chunk)
        while b"\n" in data.inb:
            raw_line, rest = data.inb.split(b"\n", 1)
            data.inb = bytearray(rest)
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                command = json.loads(line)
            except json.JSONDecodeError:
                self._send_json(sock, {"status": "error", "message": "Invalid JSON"})
                continue

            response = self.execute_command(command)
            try:
                self._send_json(sock, response)
            except Exception:
                self._close_client(sock)
                return

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        cmd_type = command.get("type")
        params = dict(command.get("params", {}) or {})
        # compatibility: some clients send "code" at top level
        if cmd_type == "execute_code" and "code" not in params and "code" in command:
            params["code"] = command.get("code")

        handlers = {
            "execute_code": self.execute_code,
            "get_scene_info": self.get_scene_info,
            "get_object_info": self.get_object_info,
            "get_scene_bounds": self.get_scene_bounds,
            "list_scene_object_names": self.list_scene_object_names,
            "hide_objects_except": self.hide_objects_except,
            "show_all_objects": self.show_all_objects,
            "set_camera_position": self.set_camera_position,
            "setup_lighting": self.setup_lighting,
            "render_scene": self.render_scene,
            "export_scene_to_obj": self.export_scene_to_obj,
            "export_scene_to_gltf": self.export_scene_to_gltf,
            "export_scene_to_glb": self.export_scene_to_glb,
            "export_object_to_obj_with_baked_materials": self.export_object_to_obj_with_baked_materials,
            "export_object_to_gltf_with_baked_materials": self.export_object_to_gltf_with_baked_materials,
            "shutdown_server": self.shutdown_server,
        }

        handler = handlers.get(cmd_type)
        if handler is None:
            return {"status": "error", "message": f"Unknown command type: {cmd_type}"}

        try:
            result = handler(**params)
            return {"status": "success", "result": result}
        except Exception as exc:
            traceback.print_exc()
            return {"status": "error", "message": str(exc)}

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def shutdown_server(self) -> Dict[str, Any]:
        self.running = False
        return {"message": "Shutting down server"}

    def execute_code(self, code: str) -> Dict[str, Any]:
        namespace = {"bpy": bpy, "mathutils": mathutils}
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(code, namespace)
        return {"executed": True, "result": buf.getvalue()}

    def get_scene_info(self) -> Dict[str, Any]:
        scene = bpy.context.scene
        objects = []
        for obj in scene.objects:
            objects.append(
                {
                    "name": obj.name,
                    "type": obj.type,
                    "location": [float(obj.location.x), float(obj.location.y), float(obj.location.z)],
                    "rotation": [
                        float(obj.rotation_euler.x),
                        float(obj.rotation_euler.y),
                        float(obj.rotation_euler.z),
                    ],
                    "scale": [float(obj.scale.x), float(obj.scale.y), float(obj.scale.z)],
                    "visible": bool(obj.visible_get()),
                }
            )
        return {
            "name": scene.name,
            "object_count": len(scene.objects),
            "objects": objects,
            "materials_count": len(bpy.data.materials),
        }

    @staticmethod
    def _mesh_aabb(obj: bpy.types.Object) -> Optional[Tuple[List[float], List[float]]]:
        if obj.type != "MESH":
            return None
        corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        if not corners:
            return None
        min_corner = mathutils.Vector(
            (
                min(c.x for c in corners),
                min(c.y for c in corners),
                min(c.z for c in corners),
            )
        )
        max_corner = mathutils.Vector(
            (
                max(c.x for c in corners),
                max(c.y for c in corners),
                max(c.z for c in corners),
            )
        )
        return (
            [float(min_corner.x), float(min_corner.y), float(min_corner.z)],
            [float(max_corner.x), float(max_corner.y), float(max_corner.z)],
        )

    def get_object_info(self, name: str) -> Dict[str, Any]:
        obj = bpy.data.objects.get(name)
        if obj is None:
            raise ValueError(f"Object not found: {name}")

        materials = [slot.material.name for slot in obj.material_slots if slot.material]
        info: Dict[str, Any] = {
            "name": obj.name,
            "type": obj.type,
            "location": [float(obj.location.x), float(obj.location.y), float(obj.location.z)],
            "rotation": [
                float(obj.rotation_euler.x),
                float(obj.rotation_euler.y),
                float(obj.rotation_euler.z),
            ],
            "scale": [float(obj.scale.x), float(obj.scale.y), float(obj.scale.z)],
            "visible": bool(obj.visible_get()),
            "materials": materials,
        }

        aabb = self._mesh_aabb(obj)
        if aabb is not None:
            min_corner, max_corner = aabb
            info["bounds"] = {"min": min_corner, "max": max_corner}
            info["world_bounding_box"] = [min_corner, max_corner]
            if obj.data is not None:
                info["mesh"] = {
                    "vertices": len(obj.data.vertices),
                    "edges": len(obj.data.edges),
                    "polygons": len(obj.data.polygons),
                }
        return info

    def get_scene_bounds(self) -> Optional[Dict[str, List[float]]]:
        min_corner: Optional[mathutils.Vector] = None
        max_corner: Optional[mathutils.Vector] = None
        for obj in bpy.context.scene.objects:
            if obj.type != "MESH" or obj.hide_render or not obj.visible_get():
                continue
            aabb = self._mesh_aabb(obj)
            if aabb is None:
                continue
            lo, hi = aabb
            if min_corner is None:
                min_corner = mathutils.Vector(lo)
                max_corner = mathutils.Vector(hi)
            else:
                min_corner.x = min(min_corner.x, lo[0])
                min_corner.y = min(min_corner.y, lo[1])
                min_corner.z = min(min_corner.z, lo[2])
                max_corner.x = max(max_corner.x, hi[0])
                max_corner.y = max(max_corner.y, hi[1])
                max_corner.z = max(max_corner.z, hi[2])

        if min_corner is None or max_corner is None:
            return None

        center = (min_corner + max_corner) / 2.0
        size = max_corner - min_corner
        return {
            "min": [float(min_corner.x), float(min_corner.y), float(min_corner.z)],
            "max": [float(max_corner.x), float(max_corner.y), float(max_corner.z)],
            "center": [float(center.x), float(center.y), float(center.z)],
            "size": [float(size.x), float(size.y), float(size.z)],
        }

    def list_scene_object_names(self) -> List[Dict[str, str]]:
        return [{"name": obj.name, "type": obj.type} for obj in bpy.context.scene.objects]

    def hide_objects_except(self, keep_visible_names: List[str]) -> Dict[str, Any]:
        keep_set = set(keep_visible_names or [])
        for obj in bpy.context.scene.objects:
            if obj.type in ("CAMERA", "LIGHT"):
                continue
            obj.hide_render = obj.name not in keep_set
        return {"kept_visible": list(keep_set)}

    def show_all_objects(self) -> Dict[str, Any]:
        for obj in bpy.context.scene.objects:
            obj.hide_render = False
        return {"updated": True}

    def set_camera_position(self, position: List[float], look_at: List[float] = None) -> Dict[str, Any]:
        look_at = look_at or [0.0, 0.0, 0.0]
        scene = bpy.context.scene
        camera = scene.camera
        if camera is None or camera.type != "CAMERA":
            cameras = [obj for obj in bpy.data.objects if obj.type == "CAMERA"]
            if cameras:
                camera = cameras[0]
            else:
                cam_data = bpy.data.cameras.new(name="Camera")
                camera = bpy.data.objects.new("Camera", cam_data)
                scene.collection.objects.link(camera)
            scene.camera = camera

        camera.location = mathutils.Vector(tuple(position))
        target = mathutils.Vector(tuple(look_at))
        direction = target - camera.location
        if direction.length < 1e-8:
            direction = mathutils.Vector((0.0, 0.0, -1.0))
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

        return {
            "camera_name": camera.name,
            "position": [float(v) for v in camera.location],
            "look_at": [float(v) for v in target],
        }

    def setup_lighting(self, light_type: str = "MULTI_POINT", intensity: float = 1.0) -> Dict[str, Any]:
        # remove existing lights
        for obj in list(bpy.data.objects):
            if obj.type == "LIGHT":
                bpy.data.objects.remove(obj, do_unlink=True)

        lt = (light_type or "MULTI_POINT").upper()
        if lt == "SIMPLE":
            configs = [
                {"name": "Sun", "type": "SUN", "position": (4.0, -4.0, 6.0), "energy": 4.0 * intensity},
            ]
        else:
            # default multi-point rig
            configs = [
                {
                    "name": "Key_Area",
                    "type": "AREA",
                    "position": (2.4, -2.8, 2.6),
                    "energy": 140.0 * intensity,
                    "size": 3.2,
                    "color": (1.0, 0.98, 0.95),
                },
                {
                    "name": "Fill_Soft",
                    "type": "AREA",
                    "position": (-2.6, -2.2, 2.0),
                    "energy": 65.0 * intensity,
                    "size": 4.8,
                    "color": (0.95, 0.97, 1.0),
                },
                {
                    "name": "Rim_Light",
                    "type": "SPOT",
                    "position": (-2.0, 3.0, 3.2),
                    "energy": 90.0 * intensity,
                    "color": (1.0, 1.0, 1.0),
                },
            ]

        lights_created: List[str] = []
        for cfg in configs:
            light_data = bpy.data.lights.new(name=cfg["name"], type=cfg["type"])
            light_data.energy = float(cfg["energy"])
            if "color" in cfg:
                light_data.color = cfg["color"]
            if cfg["type"] == "AREA" and "size" in cfg:
                light_data.size = float(cfg["size"])
            light_obj = bpy.data.objects.new(cfg["name"], light_data)
            bpy.context.scene.collection.objects.link(light_obj)
            light_obj.location = cfg["position"]
            direction = mathutils.Vector((0.0, 0.0, 0.0)) - light_obj.location
            if direction.length >= 1e-8:
                light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
            lights_created.append(cfg["name"])

        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get("Background")
        if bg_node is not None:
            bg_node.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)
            bg_node.inputs["Strength"].default_value = 0.3

        return {"success": True, "light_type": lt, "lights_created": lights_created}

    def render_scene(
        self,
        output_path: str,
        resolution: List[int] = None,
        samples: int = 128,
        engine: str = "CYCLES",
    ) -> Dict[str, Any]:
        scene = bpy.context.scene
        resolution = resolution or [336, 336]
        width, height = int(resolution[0]), int(resolution[1])
        out_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if scene.camera is None:
            self.set_camera_position([2.0, -2.0, 1.5], [0.0, 0.0, 0.0])

        eng = (engine or "CYCLES").upper()
        if eng == "CYCLES":
            scene.render.engine = "CYCLES"
            scene.cycles.samples = int(samples)
            scene.cycles.device = "GPU" if bpy.context.preferences.addons.get("cycles") else "CPU"
        elif eng in {"EEVEE", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}:
            scene.render.engine = "BLENDER_EEVEE"
            if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
                scene.eevee.taa_render_samples = int(samples)
        else:
            scene.render.engine = eng

        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.resolution_percentage = 100
        scene.render.filepath = out_path

        ext = os.path.splitext(out_path)[1].lower()
        format_map = {
            ".png": "PNG",
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".exr": "OPEN_EXR",
            ".tiff": "TIFF",
            ".tif": "TIFF",
            ".bmp": "BMP",
        }
        scene.render.image_settings.file_format = format_map.get(ext, "PNG")
        if scene.render.image_settings.file_format == "PNG":
            scene.render.film_transparent = False

        bpy.ops.render.render(write_still=True)
        return {
            "success": True,
            "output_path": out_path,
            "resolution": [width, height],
            "samples": int(samples),
            "engine": scene.render.engine,
        }

    @staticmethod
    def _sanitize_asset_stem(name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "")
        return sanitized.strip("._") or "asset"

    @staticmethod
    def _get_mesh_objects(use_selection: bool = False) -> List[bpy.types.Object]:
        scene = bpy.context.scene
        if use_selection:
            return [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
        return [obj for obj in scene.objects if obj.type == "MESH"]

    @staticmethod
    def _ensure_object_uv(
        obj: bpy.types.Object,
        view_layer: bpy.types.ViewLayer,
    ) -> None:
        if obj.type != "MESH" or obj.data.uv_layers:
            return
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project()
        bpy.ops.object.mode_set(mode="OBJECT")

    @staticmethod
    def _save_baked_image(image: bpy.types.Image, output_path: str) -> str:
        abs_output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
        image.filepath_raw = abs_output_path
        image.file_format = "PNG"
        image.save()
        image.source = "FILE"
        image.filepath = abs_output_path
        image.reload()
        return abs_output_path

    @staticmethod
    def _flatten_baked_texture_paths(baked_objects: List[Dict[str, Any]]) -> List[str]:
        texture_paths: List[str] = []
        seen: set[str] = set()
        for baked_object in baked_objects:
            for slot_info in baked_object.get("material_slots", []):
                for texture_path in slot_info.get("baked_images", {}).values():
                    if texture_path and texture_path not in seen:
                        texture_paths.append(texture_path)
                        seen.add(texture_path)
        return texture_paths

    @staticmethod
    def _restore_original_materials(original_materials_map: Dict[str, list[Any]]) -> None:
        for obj_name, original_mats in original_materials_map.items():
            obj = bpy.data.objects.get(obj_name)
            if obj is None:
                continue
            for index, mat in enumerate(original_mats):
                if index < len(obj.material_slots):
                    obj.material_slots[index].material = mat

    @staticmethod
    def _update_mtl_texture_reference(
        lines: List[str],
        directive: str,
        relative_path: str,
    ) -> List[str]:
        prefix = f"{directive} "
        filtered = [
            line
            for line in lines
            if not line.lstrip().startswith(prefix)
        ]
        filtered.append(f"{directive} {relative_path}\n")
        return filtered

    def _patch_mtl_with_baked_textures(
        self,
        mtl_path: str,
        baked_objects: List[Dict[str, Any]],
    ) -> None:
        if not os.path.exists(mtl_path):
            raise FileNotFoundError(f"MTL file not found: {mtl_path}")

        material_to_images: Dict[str, Dict[str, str]] = {}
        for baked_object in baked_objects:
            for slot_info in baked_object.get("material_slots", []):
                material_name = slot_info.get("material_name")
                baked_images = {
                    key: value
                    for key, value in slot_info.get("baked_images", {}).items()
                    if value
                }
                if material_name and baked_images:
                    material_to_images[material_name] = baked_images

        if not material_to_images:
            return

        directive_map = {
            "base_color": "map_Kd",
            "roughness": "map_Pr",
            "metallic": "map_Pm",
            "normal": "map_Bump",
        }

        with open(mtl_path, "r", encoding="utf-8") as handle:
            original_lines = handle.readlines()

        patched_lines: List[str] = []
        current_material: Optional[str] = None
        current_block: List[str] = []

        def flush_current_block() -> None:
            nonlocal current_material, current_block
            if current_material is not None and current_material in material_to_images:
                baked_images = material_to_images[current_material]
                for pass_name, directive in directive_map.items():
                    image_path = baked_images.get(pass_name)
                    if image_path:
                        relative_path = os.path.relpath(
                            image_path,
                            start=os.path.dirname(mtl_path),
                        ).replace("\\", "/")
                        current_block[:] = self._update_mtl_texture_reference(
                            current_block,
                            directive,
                            relative_path,
                        )
            patched_lines.extend(current_block)
            current_material = None
            current_block = []

        for line in original_lines:
            if line.startswith("newmtl "):
                flush_current_block()
                current_material = line.strip().split(" ", 1)[1]
            current_block.append(line)

        flush_current_block()

        with open(mtl_path, "w", encoding="utf-8") as handle:
            handle.writelines(patched_lines)

    def _bake_materials_for_export(
        self,
        output_dir: str,
        asset_stem: str,
        resolution: int = 1024,
        use_selection: bool = False,
        passes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Bake mesh materials into external textures for export pipelines."""
        scene = bpy.context.scene
        view_layer = bpy.context.view_layer
        original_engine = scene.render.engine
        original_samples = getattr(scene.cycles, "samples", None)
        original_active = view_layer.objects.active
        original_selected = list(bpy.context.selected_objects)
        original_materials_map: Dict[str, list[Any]] = {}
        baked_objects: List[Dict[str, Any]] = []
        requested_passes = set(passes or ["base_color", "roughness", "normal", "metallic"])
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)

        if hasattr(bpy.ops, "object") and bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")

        scene.render.engine = "CYCLES"
        if hasattr(scene, "cycles"):
            scene.cycles.samples = 16

        objs = self._get_mesh_objects(use_selection=use_selection)
        if not objs:
            raise ValueError("No mesh objects available for baking")

        def _clear_input_links(socket) -> None:
            while socket.is_linked:
                links.remove(socket.links[0])

        try:
            for obj in objs:
                original_materials_map[obj.name] = [
                    slot.material for slot in obj.material_slots
                ]
                baked_object_info = {
                    "object_name": obj.name,
                    "material_slots": [],
                }

                self._ensure_object_uv(obj, view_layer)

                multi_slot = len(obj.material_slots) > 1
                for slot_index, slot in enumerate(obj.material_slots):
                    slot_info: Dict[str, Any] = {
                        "slot_index": slot_index,
                        "material_name": slot.material.name if slot.material else None,
                        "baked_images": {},
                    }
                    if not slot.material or not slot.material.use_nodes:
                        baked_object_info["material_slots"].append(slot_info)
                        continue

                    slot.material = slot.material.copy()
                    mat = slot.material
                    slot_info["material_name"] = mat.name
                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links

                    bsdf = next(
                        (n for n in nodes if n.type == "BSDF_PRINCIPLED"), None
                    )
                    out_node = next(
                        (n for n in nodes if n.type == "OUTPUT_MATERIAL"), None
                    )
                    if not bsdf or not out_node:
                        baked_object_info["material_slots"].append(slot_info)
                        continue

                    object_prefix = asset_stem
                    if len(objs) > 1:
                        object_prefix = f"{asset_stem}_{self._sanitize_asset_stem(obj.name)}"
                    suffix_prefix = object_prefix
                    if multi_slot:
                        suffix_prefix = f"{object_prefix}_mat{slot_index}"

                    baked_nodes: Dict[str, bpy.types.ShaderNodeTexImage] = {}
                    baked_images: Dict[str, bpy.types.Image] = {}

                    def _new_bake_node(pass_name: str, label: str, *, non_color: bool) -> bpy.types.ShaderNodeTexImage:
                        image = bpy.data.images.new(
                            name=f"{suffix_prefix}_{label}",
                            width=resolution,
                            height=resolution,
                        )
                        node = nodes.new("ShaderNodeTexImage")
                        node.image = image
                        node.name = f"Bake_{label}"
                        if non_color:
                            node.image.colorspace_settings.name = "Non-Color"
                        baked_images[pass_name] = image
                        baked_nodes[pass_name] = node
                        return node

                    node_color = (
                        _new_bake_node("base_color", "BaseColor", non_color=False)
                        if "base_color" in requested_passes
                        else None
                    )
                    node_metallic = (
                        _new_bake_node("metallic", "Metallic", non_color=True)
                        if "metallic" in requested_passes
                        else None
                    )
                    node_roughness = (
                        _new_bake_node("roughness", "Roughness", non_color=True)
                        if "roughness" in requested_passes
                        else None
                    )
                    node_normal = (
                        _new_bake_node("normal", "Normal", non_color=True)
                        if "normal" in requested_passes
                        else None
                    )

                    emit_node = nodes.new("ShaderNodeEmission")

                    bpy.ops.object.select_all(action="DESELECT")
                    obj.select_set(True)
                    view_layer.objects.active = obj

                    def bake_pass_via_emit(input_name: str, target_node, image) -> None:
                        in_socket = bsdf.inputs[input_name]
                        _clear_input_links(emit_node.inputs["Color"])
                        _clear_input_links(out_node.inputs["Surface"])
                        if in_socket.is_linked:
                            source_socket = in_socket.links[0].from_socket
                            links.new(source_socket, emit_node.inputs["Color"])
                        else:
                            value = in_socket.default_value
                            if isinstance(value, (float, int)):
                                emit_node.inputs["Color"].default_value = (
                                    value,
                                    value,
                                    value,
                                    1.0,
                                )
                            else:
                                emit_node.inputs["Color"].default_value = value

                        links.new(emit_node.outputs["Emission"], out_node.inputs["Surface"])

                        for node in nodes:
                            node.select = False
                        target_node.select = True
                        nodes.active = target_node

                        bpy.ops.object.bake(type="EMIT")
                        pass_stem = {
                            "Base Color": "basecolor",
                            "Metallic": "metallic",
                            "Roughness": "roughness",
                        }[input_name]
                        saved_path = self._save_baked_image(
                            image,
                            os.path.join(abs_output_dir, f"{suffix_prefix}_{pass_stem}.png"),
                        )
                        slot_info["baked_images"][pass_stem.replace("basecolor", "base_color")] = saved_path

                    if node_color is not None:
                        bake_pass_via_emit("Base Color", node_color, baked_images["base_color"])
                    if node_metallic is not None:
                        bake_pass_via_emit("Metallic", node_metallic, baked_images["metallic"])
                    if node_roughness is not None:
                        bake_pass_via_emit("Roughness", node_roughness, baked_images["roughness"])

                    _clear_input_links(out_node.inputs["Surface"])
                    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
                    if node_normal is not None:
                        for node in nodes:
                            node.select = False
                        node_normal.select = True
                        nodes.active = node_normal
                        bpy.ops.object.bake(type="NORMAL")
                        saved_normal_path = self._save_baked_image(
                            baked_images["normal"],
                            os.path.join(abs_output_dir, f"{suffix_prefix}_normal.png"),
                        )
                        slot_info["baked_images"]["normal"] = saved_normal_path

                    keep_nodes = {
                        bsdf,
                        out_node,
                        *(node for node in [node_color, node_metallic, node_roughness, node_normal] if node is not None),
                    }
                    for node in list(nodes):
                        if node not in keep_nodes:
                            nodes.remove(node)

                    _clear_input_links(bsdf.inputs["Base Color"])
                    _clear_input_links(bsdf.inputs["Metallic"])
                    _clear_input_links(bsdf.inputs["Roughness"])
                    _clear_input_links(bsdf.inputs["Normal"])
                    _clear_input_links(out_node.inputs["Surface"])

                    if node_color is not None:
                        links.new(node_color.outputs["Color"], bsdf.inputs["Base Color"])
                    if node_metallic is not None:
                        links.new(node_metallic.outputs["Color"], bsdf.inputs["Metallic"])
                    if node_roughness is not None:
                        links.new(node_roughness.outputs["Color"], bsdf.inputs["Roughness"])
                    if node_normal is not None:
                        normal_map = nodes.new("ShaderNodeNormalMap")
                        links.new(node_normal.outputs["Color"], normal_map.inputs["Color"])
                        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
                    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
                    baked_object_info["material_slots"].append(slot_info)
                baked_objects.append(baked_object_info)
        finally:
            scene.render.engine = original_engine
            if original_samples is not None and hasattr(scene, "cycles"):
                scene.cycles.samples = original_samples
            bpy.ops.object.select_all(action="DESELECT")
            for obj in original_selected:
                if obj and obj.name in bpy.data.objects:
                    obj.select_set(True)
            if original_active and original_active.name in bpy.data.objects:
                view_layer.objects.active = bpy.data.objects[original_active.name]

        return {
            "original_materials_map": original_materials_map,
            "baked_objects": baked_objects,
            "output_dir": abs_output_dir,
        }

    def _bake_materials_for_glb_export(
        self,
        resolution: int = 1024,
        use_selection: bool = False,
    ) -> Dict[str, list[Any]]:
        """Bake mesh materials for legacy GLB export."""
        result = self._bake_materials_for_export(
            output_dir=os.path.join(tempfile.gettempdir(), "code_object_glb_bakes"),
            asset_stem="glb_export",
            resolution=resolution,
            use_selection=use_selection,
        )
        return result["original_materials_map"]

    @staticmethod
    def _restore_export_selection(
        original_selected: List[bpy.types.Object],
        original_active: Optional[bpy.types.Object],
        view_layer: bpy.types.ViewLayer,
    ) -> None:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in original_selected:
            if obj and obj.name in bpy.data.objects:
                bpy.data.objects[obj.name].select_set(True)
        if original_active and original_active.name in bpy.data.objects:
            view_layer.objects.active = bpy.data.objects[original_active.name]

    @staticmethod
    def _prepare_export_selection(
        object_name: Optional[str],
        use_selection: bool,
    ) -> Tuple[List[bpy.types.Object], Optional[bpy.types.Object], bool]:
        view_layer = bpy.context.view_layer
        original_selected = list(bpy.context.selected_objects)
        original_active = view_layer.objects.active
        effective_use_selection = bool(use_selection)

        if object_name is not None:
            obj = bpy.data.objects.get(object_name)
            if obj is None:
                raise ValueError(f"Object not found: {object_name}")
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            view_layer.objects.active = obj
            effective_use_selection = True
        elif effective_use_selection and not bpy.context.selected_objects:
            raise ValueError("No selected objects to export")

        return original_selected, original_active, effective_use_selection

    def export_scene_to_obj(
        self,
        output_path: str,
        use_selection: bool = False,
        export_apply: bool = False,
        export_materials: bool = True,
        object_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        out_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.splitext(out_path)[1].lower() != ".obj":
            raise ValueError("output_path must end with .obj")

        if hasattr(bpy.ops, "object") and bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")

        view_layer = bpy.context.view_layer
        original_selected, original_active, effective_use_selection = (
            self._prepare_export_selection(object_name, use_selection)
        )
        try:
            if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_export"):
                bpy.ops.wm.obj_export(
                    filepath=out_path,
                    export_selected_objects=bool(effective_use_selection),
                    export_materials=bool(export_materials),
                    forward_axis="NEGATIVE_Z",
                    up_axis="Y",
                )
            else:
                bpy.ops.export_scene.obj(
                    filepath=out_path,
                    use_selection=bool(effective_use_selection),
                    use_materials=bool(export_materials),
                    axis_forward="-Z",
                    axis_up="Y",
                )
        finally:
            self._restore_export_selection(original_selected, original_active, view_layer)

        return {
            "success": True,
            "output_path": out_path,
            "export_format": "OBJ",
            "use_selection": bool(effective_use_selection),
            "export_apply": bool(export_apply),
            "export_materials": bool(export_materials),
            "object_name": object_name,
        }

    def export_scene_to_gltf(
        self,
        output_path: str,
        use_selection: bool = False,
        export_apply: bool = False,
        export_format: str = "GLTF_SEPARATE",
        object_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        out_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        format_name = str(export_format or "GLTF_SEPARATE").upper()
        expected_ext = ".glb" if format_name == "GLB" else ".gltf"
        if os.path.splitext(out_path)[1].lower() != expected_ext:
            raise ValueError(f"output_path must end with {expected_ext}")

        if hasattr(bpy.ops, "object") and bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")

        view_layer = bpy.context.view_layer
        original_selected, original_active, effective_use_selection = (
            self._prepare_export_selection(object_name, use_selection)
        )
        try:
            bpy.ops.export_scene.gltf(
                filepath=out_path,
                export_format=format_name,
                use_selection=bool(effective_use_selection),
                export_apply=bool(export_apply),
                export_cameras=False,
                export_lights=False,
            )
        finally:
            self._restore_export_selection(original_selected, original_active, view_layer)

        return {
            "success": True,
            "output_path": out_path,
            "export_format": format_name,
            "use_selection": bool(effective_use_selection),
            "export_apply": bool(export_apply),
            "object_name": object_name,
        }

    def export_object_to_obj_with_baked_materials(
        self,
        output_path: str,
        object_name: str,
        bake_textures: bool = True,
        bake_resolution: int = 1024,
        texture_output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        out_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.splitext(out_path)[1].lower() != ".obj":
            raise ValueError("output_path must end with .obj")

        if hasattr(bpy.ops, "object") and bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")

        view_layer = bpy.context.view_layer
        original_selected, original_active, effective_use_selection = (
            self._prepare_export_selection(object_name, True)
        )
        bake_result: Dict[str, Any] = {
            "original_materials_map": {},
            "baked_objects": [],
        }
        try:
            if bake_textures:
                bake_result = self._bake_materials_for_export(
                    output_dir=texture_output_dir or os.path.dirname(out_path),
                    asset_stem=self._sanitize_asset_stem(os.path.splitext(os.path.basename(out_path))[0]),
                    resolution=int(bake_resolution),
                    use_selection=bool(effective_use_selection),
                )

            if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_export"):
                bpy.ops.wm.obj_export(
                    filepath=out_path,
                    export_selected_objects=True,
                    export_materials=True,
                    forward_axis="NEGATIVE_Z",
                    up_axis="Y",
                )
            else:
                bpy.ops.export_scene.obj(
                    filepath=out_path,
                    use_selection=True,
                    use_materials=True,
                    axis_forward="-Z",
                    axis_up="Y",
                )
        finally:
            if bake_textures and bake_result["original_materials_map"]:
                self._restore_original_materials(bake_result["original_materials_map"])
            self._restore_export_selection(original_selected, original_active, view_layer)

        mtl_path = os.path.splitext(out_path)[0] + ".mtl"
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"OBJ export failed: {out_path}")
        if not os.path.exists(mtl_path):
            raise FileNotFoundError(f"OBJ material export failed: {mtl_path}")
        if bake_textures:
            self._patch_mtl_with_baked_textures(mtl_path, bake_result["baked_objects"])

        baked_texture_paths = self._flatten_baked_texture_paths(
            bake_result["baked_objects"]
        )
        return {
            "success": True,
            "output_path": out_path,
            "mtl_path": mtl_path,
            "object_name": object_name,
            "baked_textures": baked_texture_paths,
            "baked_objects": bake_result["baked_objects"],
            "bake_resolution": int(bake_resolution),
            "material_slot_count": len(
                bake_result["baked_objects"][0]["material_slots"]
            ) if bake_result["baked_objects"] else 0,
            "baked_passes": ["base_color", "roughness", "normal", "metallic"],
        }

    def export_object_to_gltf_with_baked_materials(
        self,
        output_path: str,
        object_name: str,
        bake_textures: bool = True,
        bake_resolution: int = 1024,
        texture_output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        out_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.splitext(out_path)[1].lower() != ".gltf":
            raise ValueError("output_path must end with .gltf")

        if hasattr(bpy.ops, "object") and bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")

        view_layer = bpy.context.view_layer
        original_selected, original_active, effective_use_selection = (
            self._prepare_export_selection(object_name, True)
        )
        bake_result: Dict[str, Any] = {
            "original_materials_map": {},
            "baked_objects": [],
        }
        try:
            if bake_textures:
                bake_result = self._bake_materials_for_export(
                    output_dir=texture_output_dir or os.path.dirname(out_path),
                    asset_stem=self._sanitize_asset_stem(os.path.splitext(os.path.basename(out_path))[0]),
                    resolution=int(bake_resolution),
                    use_selection=bool(effective_use_selection),
                )

            bpy.ops.export_scene.gltf(
                filepath=out_path,
                export_format="GLTF_SEPARATE",
                use_selection=True,
                export_apply=False,
                export_cameras=False,
                export_lights=False,
                export_yup=True,
            )
        finally:
            if bake_textures and bake_result["original_materials_map"]:
                self._restore_original_materials(bake_result["original_materials_map"])
            self._restore_export_selection(original_selected, original_active, view_layer)

        if not os.path.exists(out_path):
            raise FileNotFoundError(f"GLTF export failed: {out_path}")
        bin_path = os.path.splitext(out_path)[0] + ".bin"
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"GLTF binary export failed: {bin_path}")

        baked_texture_paths = self._flatten_baked_texture_paths(
            bake_result["baked_objects"]
        )
        return {
            "success": True,
            "output_path": out_path,
            "bin_path": bin_path,
            "object_name": object_name,
            "baked_textures": baked_texture_paths,
            "baked_objects": bake_result["baked_objects"],
            "bake_resolution": int(bake_resolution),
            "material_slot_count": len(
                bake_result["baked_objects"][0]["material_slots"]
            ) if bake_result["baked_objects"] else 0,
            "baked_passes": ["base_color", "roughness", "normal", "metallic"],
        }

    def export_scene_to_glb(
        self,
        output_path: str,
        use_selection: bool = False,
        export_apply: bool = False,
        bake_textures: bool = True,
        bake_resolution: int = 1024,
    ) -> Dict[str, Any]:
        out_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.splitext(out_path)[1].lower() != ".glb":
            raise ValueError("output_path must end with .glb")

        if hasattr(bpy.ops, "object") and bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")

        bake_result: Dict[str, Any] = {
            "original_materials_map": {},
            "baked_objects": [],
        }
        try:
            if bake_textures:
                bake_result = self._bake_materials_for_export(
                    output_dir=os.path.join(
                        tempfile.gettempdir(),
                        "code_object_glb_bakes",
                        self._sanitize_asset_stem(os.path.splitext(os.path.basename(out_path))[0]),
                    ),
                    asset_stem=self._sanitize_asset_stem(os.path.splitext(os.path.basename(out_path))[0]),
                    resolution=int(bake_resolution),
                    use_selection=bool(use_selection),
                )

            bpy.ops.export_scene.gltf(
                filepath=out_path,
                export_format="GLB",
                use_selection=bool(use_selection),
                export_apply=bool(export_apply),
                export_cameras=False,
                export_lights=False,
            )
        finally:
            if bake_textures and bake_result["original_materials_map"]:
                self._restore_original_materials(bake_result["original_materials_map"])

        return {
            "success": True,
            "output_path": out_path,
            "export_format": "GLB",
            "use_selection": bool(use_selection),
            "export_apply": bool(export_apply),
            "baked_textures": bool(bake_textures),
            "bake_resolution": int(bake_resolution),
        }


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blender headless socket server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    return parser.parse_args(argv)


def main() -> None:
    # Blender passes script args after `--`
    argv = []
    if "--" in os.sys.argv:
        argv = os.sys.argv[os.sys.argv.index("--") + 1 :]
    args = _parse_args(argv)
    server = BlenderHeadlessServer(host=args.host, port=args.port)
    server.run_forever()


if __name__ == "__main__":
    main()
