"""
Blender MCP Client
连接 Blender MCP Server，发送执行命令
"""

import json
import asyncio
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

from .request_lock import acquire_blender_request_lock


@dataclass
class MCPResponse:
    """MCP 响应"""
    success: bool
    result: Any = None
    error: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error
        }


class BlenderMCPClient:
    """
    Blender MCP 客户端
    
    连接 Blender MCP Server，提供以下功能：
    - 执行 Python 代码
    - 获取场景信息
    - 渲染图像
    - 操作对象
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9876,
        timeout: float = 300.0
    ):
        """
        初始化客户端
        
        Args:
            host: MCP Server 地址
            port: MCP Server 端口
            timeout: 请求超时时间（秒）
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)
        self._reader = None
        self._writer = None
        self._connected = False
        self._request_id = 0
        
    async def connect(self) -> bool:
        """
        连接到 MCP Server
        
        Returns:
            是否连接成功
        """
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout
            )
            self._connected = True
            self.logger.info(f"Connected to Blender MCP Server at {self.host}:{self.port}")
            return True
        except asyncio.TimeoutError:
            self.logger.error(f"Connection timeout to {self.host}:{self.port}")
            return False
        except ConnectionRefusedError:
            self.logger.error(f"Connection refused to {self.host}:{self.port}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self) -> None:
        """断开连接"""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._connected = False
        self._reader = None
        self._writer = None
        self.logger.info("Disconnected from Blender MCP Server")
    
    async def _send_command(self, cmd_type: str, params: Dict[str, Any] = None) -> MCPResponse:
        """发送命令并等待响应"""
        if not self._connected:
            return MCPResponse(success=False, error="Not connected")

        with acquire_blender_request_lock(f"code_object.mcp.{cmd_type}"):
            self._request_id += 1
            request = {
                "type": cmd_type,
                "params": params or {}
            }

            try:
                # 发送请求
                data = json.dumps(request).encode() + b'\n'
                self.logger.debug(f"Sending: {request}")
                self._writer.write(data)
                await self._writer.drain()

                # 兼容两种服务端返回方式：
                # 1) 单行 JSON + '\n'
                # 2) 无换行，直接返回完整 JSON
                self.logger.debug("Waiting for response...")
                response = None
                buffer = bytearray()
                while True:
                    chunk = await asyncio.wait_for(self._reader.read(4096), timeout=self.timeout)
                    if not chunk:
                        break
                    buffer.extend(chunk)

                    # 优先按整包 JSON 解析（无换行场景）
                    text_full = buffer.decode(errors="replace").strip()
                    if text_full:
                        try:
                            response = json.loads(text_full)
                            break
                        except json.JSONDecodeError:
                            pass

                    # 再按首行 JSON 解析（有换行场景）
                    if b"\n" in buffer:
                        first_line = buffer.split(b"\n", 1)[0].decode(errors="replace").strip()
                        if first_line:
                            response = json.loads(first_line)
                            break

                if response is None:
                    return MCPResponse(success=False, error="Empty or invalid response")

                # 兼容 status/success 两种格式
                success = response.get("success", False)
                if "status" in response:
                    success = response["status"] == "success"

                return MCPResponse(
                    success=success,
                    result=response.get("result"),
                    error=response.get("error") or response.get("message")
                )

            except asyncio.TimeoutError:
                self.logger.error(f"Command timed out: {cmd_type}")
                return MCPResponse(success=False, error="Timeout")
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON response: {e}")
                return MCPResponse(success=False, error="Invalid JSON response")
            except Exception as e:
                self.logger.error(f"Command failed: {e}")
                return MCPResponse(success=False, error=str(e))


    async def execute(self, action: str, params: Dict[str, Any] = None) -> Any:
        """
        执行操作
        
        Args:
            action: 操作名称
            params: 参数
            
        Returns:
            执行结果
        """
        response = await self._send_command(action, params)
        
        if not response.success:
            raise RuntimeError(f"MCP request failed: {response.error}")
        
        return response.result
    
    # ==================== 高级 API ====================
    
    async def execute_code(self, code: str) -> Dict[str, Any]:
        """
        执行 Python 代码。
        请求格式：{"type": "execute_code", "params": {"code": code}}，与 Blender addon 一致。
        """
        response = await self._send_command("execute_code", {"code": code})
        
        return {
            "success": response.success,
            "result": response.result,
            "error": response.error
        }
    
    async def get_scene_info(self) -> Dict[str, Any]:
        """
        获取场景信息
        
        Returns:
            场景信息字典
        """
        return await self.execute("get_scene_info", {})
    
    async def get_object_info(self, name: str) -> Dict[str, Any]:
        """
        获取对象详细信息（对应 addon.py 的 get_object_info）
        
        Args:
            name: 对象名称
            
        Returns:
            对象信息，包含 location, rotation, scale, materials, mesh 等
        """
        return await self.execute("get_object_info", {"name": name})

    async def get_scene_bounds(self) -> Optional[Dict[str, Any]]:
        """
        获取当前场景中所有网格物体的整体包围盒（世界坐标 AABB）。
        用于部件特写渲染时计算相机位置。
        
        Returns:
            dict 含 min, max, center, size（均为 [x,y,z]）；无网格时返回 None
        """
        result = await self.execute("get_scene_bounds", {})
        return result if isinstance(result, dict) and result else None

    async def list_scene_object_names(self) -> List[Dict[str, str]]:
        """
        返回场景中所有对象的 name 与 type，用于特写时筛选当前部件对象。
        Returns:
            [{"name": str, "type": str}, ...]
        """
        result = await self.execute("list_scene_object_names", {})
        return result if isinstance(result, list) else []

    async def hide_objects_except(self, keep_visible_names: List[str]) -> None:
        """
        渲染时只保留指定名称的对象可见，其余对象 hide_render=True。
        特写渲染前调用，渲染后需调用 show_all_objects 恢复。
        """
        await self.execute("hide_objects_except", {"keep_visible_names": keep_visible_names})

    async def show_all_objects(self) -> None:
        """恢复所有对象参与渲染（hide_render=False）。特写渲染完成后调用。"""
        await self.execute("show_all_objects", {})

    async def list_objects(self) -> List[str]:
        """
        列出场景中的所有对象
        
        Returns:
            对象名称列表
        """
        scene_info = await self.get_scene_info()
        return [obj["name"] for obj in scene_info.get("objects", [])]
    
    async def create_object(
        self,
        object_type: str,
        name: str = None,
        location: tuple = (0, 0, 0),
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建对象（通过执行代码实现）
        
        Args:
            object_type: 对象类型（cube, sphere, cylinder, cone, torus, plane）
            name: 对象名称
            location: 位置
            
        Returns:
            创建结果
        """
        type_map = {
            "cube": "bpy.ops.mesh.primitive_cube_add",
            "sphere": "bpy.ops.mesh.primitive_uv_sphere_add",
            "cylinder": "bpy.ops.mesh.primitive_cylinder_add",
            "cone": "bpy.ops.mesh.primitive_cone_add",
            "torus": "bpy.ops.mesh.primitive_torus_add",
            "plane": "bpy.ops.mesh.primitive_plane_add",
            "monkey": "bpy.ops.mesh.primitive_monkey_add",
        }
        
        op = type_map.get(object_type.lower())
        if not op:
            raise ValueError(f"Unknown object type: {object_type}")
        
        code = f"{op}(location={location})\n"
        if name:
            code += f"bpy.context.active_object.name = '{name}'\n"
        code += "print(bpy.context.active_object.name)"
        
        return await self.execute_code(code)
    
    async def delete_object(self, object_name: str) -> bool:
        """
        删除对象（通过执行代码实现）
        
        Args:
            object_name: 对象名称
            
        Returns:
            是否成功
        """
        code = f"""
obj = bpy.data.objects.get('{object_name}')
if obj:
    bpy.data.objects.remove(obj, do_unlink=True)
    print('deleted')
else:
    print('not found')
"""
        result = await self.execute_code(code)
        return "deleted" in result.get("result", "")
    
    async def transform_object(
        self,
        object_name: str,
        location: tuple = None,
        rotation: tuple = None,
        scale: tuple = None
    ) -> Dict[str, Any]:
        """
        变换对象（通过执行代码实现）
        
        Args:
            object_name: 对象名称
            location: 新位置
            rotation: 新旋转（欧拉角，弧度）
            scale: 新缩放
        """
        code = f"obj = bpy.data.objects.get('{object_name}')\n"
        code += "if obj:\n"
        if location:
            code += f"    obj.location = {location}\n"
        if rotation:
            code += f"    obj.rotation_euler = {rotation}\n"
        if scale:
            code += f"    obj.scale = {scale}\n"
        code += "    print('transformed')\n"
        code += "else:\n"
        code += "    print('not found')\n"
        
        return await self.execute_code(code)
    
    async def add_modifier(
        self,
        object_name: str,
        modifier_type: str,
        modifier_name: str = None,
        **settings
    ) -> Dict[str, Any]:
        """
        添加修改器
        
        Args:
            object_name: 对象名称
            modifier_type: 修改器类型（BEVEL, SUBSURF 等）
            modifier_name: 修改器名称
            **settings: 修改器设置
            
        Returns:
            操作结果
        """
        params = {
            "object_name": object_name,
            "modifier_type": modifier_type,
            "settings": settings
        }
        if modifier_name:
            params["modifier_name"] = modifier_name
            
        return await self.execute("add_modifier", params)
    
    async def render_scene(
        self,
        output_path: str,
        resolution: tuple = (336, 336),
        samples: int = 128
    ) -> Dict[str, Any]:
        """
        渲染场景
        
        Args:
            output_path: 输出路径
            resolution: 分辨率 (width, height)
            samples: 采样数
            
        Returns:
            渲染结果
        """
        return await self.execute("render_scene", {
            "output_path": output_path,
            "resolution": list(resolution),
            "samples": samples
        })

    async def export_scene_to_glb(
        self,
        output_path: str,
        use_selection: bool = False,
        export_apply: bool = False,
        bake_textures: bool = True,
        bake_resolution: int = 1024,
    ) -> Dict[str, Any]:
        """
        导出当前 Blender 场景为 .glb 文件。

        Args:
            output_path: 输出 .glb 路径
            use_selection: 是否仅导出当前选中对象
            export_apply: 是否应用变换到导出结果
            bake_textures: 导出前是否先烘焙材质贴图
            bake_resolution: 烘焙贴图分辨率

        Returns:
            导出结果
        """
        return await self.execute("export_scene_to_glb", {
            "output_path": output_path,
            "use_selection": bool(use_selection),
            "export_apply": bool(export_apply),
            "bake_textures": bool(bake_textures),
            "bake_resolution": int(bake_resolution),
        })

    async def export_scene_to_obj(
        self,
        output_path: str,
        use_selection: bool = False,
        export_apply: bool = False,
        export_materials: bool = True,
        object_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """导出当前 Blender 场景为 .obj 文件。"""
        return await self.execute("export_scene_to_obj", {
            "output_path": output_path,
            "use_selection": bool(use_selection),
            "export_apply": bool(export_apply),
            "export_materials": bool(export_materials),
            "object_name": object_name,
        })

    async def export_scene_to_gltf(
        self,
        output_path: str,
        use_selection: bool = False,
        export_apply: bool = False,
        export_format: str = "GLTF_SEPARATE",
        object_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """导出当前 Blender 场景为 .gltf/.glb 文件。"""
        return await self.execute("export_scene_to_gltf", {
            "output_path": output_path,
            "use_selection": bool(use_selection),
            "export_apply": bool(export_apply),
            "export_format": export_format,
            "object_name": object_name,
        })

    async def export_object_to_obj_with_baked_materials(
        self,
        output_path: str,
        object_name: str,
        bake_textures: bool = True,
        bake_resolution: int = 1024,
        texture_output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """导出单个对象为带 baked 贴图的 OBJ/MTL 资产包。"""
        return await self.execute("export_object_to_obj_with_baked_materials", {
            "output_path": output_path,
            "object_name": object_name,
            "bake_textures": bool(bake_textures),
            "bake_resolution": int(bake_resolution),
            "texture_output_dir": texture_output_dir,
        })

    async def export_object_to_gltf_with_baked_materials(
        self,
        output_path: str,
        object_name: str,
        bake_textures: bool = True,
        bake_resolution: int = 1024,
        texture_output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """导出单个对象为带 baked 贴图的 GLTF_SEPARATE 资产包。"""
        return await self.execute("export_object_to_gltf_with_baked_materials", {
            "output_path": output_path,
            "object_name": object_name,
            "bake_textures": bool(bake_textures),
            "bake_resolution": int(bake_resolution),
            "texture_output_dir": texture_output_dir,
        })

    @staticmethod
    def _build_v3_full_object_render_code(
        output_path: str,
        resolution: tuple = (512, 512),
        samples: int = 128,
        engine: str = "CYCLES",
        camera_location: tuple = (1.4, -1.4, 1.0),
        target_location: tuple = (0.0, 0.0, 0.5),
    ) -> str:
        """构建与 render_batch_v3 对齐的 full-object 渲染脚本。"""
        width = int(resolution[0])
        height = int(resolution[1])
        sample_count = int(samples)
        engine_name = str(engine or "CYCLES").upper()
        if engine_name == "EEVEE":
            engine_candidates = ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"]
        elif engine_name in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}:
            engine_candidates = [engine_name]
        else:
            engine_candidates = ["CYCLES"]
        return (
            "import bpy, mathutils, os\n"
            f"render_path = {output_path!r}\n"
            f"camera_location = {tuple(camera_location)!r}\n"
            f"target_location = {tuple(target_location)!r}\n"
            f"resolution = ({width}, {height})\n"
            f"samples = {sample_count}\n"
            f"engine_candidates = {engine_candidates!r}\n"
            "scene = bpy.context.scene\n"
            "\n"
            "def normalize_scene_objects():\n"
            "    min_x = min_y = min_z = float('inf')\n"
            "    max_x = max_y = max_z = float('-inf')\n"
            "    mesh_objects = []\n"
            "    for obj in scene.objects:\n"
            "        if obj.type != 'MESH':\n"
            "            continue\n"
            "        if obj.hide_get() or obj.hide_viewport:\n"
            "            continue\n"
            "        mesh_objects.append(obj)\n"
            "        for corner in obj.bound_box:\n"
            "            world_corner = obj.matrix_world @ mathutils.Vector(corner)\n"
            "            x, y, z = world_corner\n"
            "            min_x = min(min_x, x)\n"
            "            min_y = min(min_y, y)\n"
            "            min_z = min(min_z, z)\n"
            "            max_x = max(max_x, x)\n"
            "            max_y = max(max_y, y)\n"
            "            max_z = max(max_z, z)\n"
            "    if not mesh_objects:\n"
            "        return False\n"
            "    dx = max_x - min_x\n"
            "    dy = max_y - min_y\n"
            "    dz = max_z - min_z\n"
            "    longest = max(dx, dy, dz, 1e-6)\n"
            "    scale = 1.0 / longest\n"
            "    center = mathutils.Vector((\n"
            "        0.5 * (min_x + max_x),\n"
            "        0.5 * (min_y + max_y),\n"
            "        0.5 * (min_z + max_z),\n"
            "    ))\n"
            "    desired_center = mathutils.Vector(target_location)\n"
            "    for obj in mesh_objects:\n"
            "        mesh = obj.data\n"
            "        mw = obj.matrix_world\n"
            "        mw_inv = mw.inverted()\n"
            "        for v in mesh.vertices:\n"
            "            world_co = mw @ v.co\n"
            "            world_co = (world_co - center) * scale + desired_center\n"
            "            v.co = mw_inv @ world_co\n"
            "        mesh.update()\n"
            "    return True\n"
            "\n"
            "if hasattr(bpy.ops, 'object') and bpy.ops.object.mode_set.poll():\n"
            "    bpy.ops.object.mode_set(mode='OBJECT')\n"
            "\n"
            "if not normalize_scene_objects():\n"
            "    raise RuntimeError('No visible mesh objects found for V3 render')\n"
            "\n"
            "os.makedirs(os.path.dirname(render_path), exist_ok=True)\n"
            "\n"
            "for obj in list(bpy.data.objects):\n"
            "    if obj.type in {'CAMERA', 'LIGHT'}:\n"
            "        bpy.data.objects.remove(obj, do_unlink=True)\n"
            "\n"
            "cam_data = bpy.data.cameras.new(name='Camera_Auto')\n"
            "cam_obj = bpy.data.objects.new('Camera_Auto', cam_data)\n"
            "bpy.context.scene.collection.objects.link(cam_obj)\n"
            "scene.camera = cam_obj\n"
            "cam_obj.location = camera_location\n"
            "direction = mathutils.Vector(target_location) - cam_obj.location\n"
            "if direction.length >= 1e-8:\n"
            "    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()\n"
            "\n"
            "configs = [\n"
            "    {\n"
            "        'name': 'Key_Area',\n"
            "        'type': 'AREA',\n"
            "        'position': (2.4, -2.8, 2.6),\n"
            "        'energy': 140.0,\n"
            "        'size': 3.2,\n"
            "        'color': (1.0, 0.98, 0.95),\n"
            "    },\n"
            "    {\n"
            "        'name': 'Fill_Soft',\n"
            "        'type': 'AREA',\n"
            "        'position': (-2.6, -2.2, 2.0),\n"
            "        'energy': 65.0,\n"
            "        'size': 4.8,\n"
            "        'color': (0.95, 0.97, 1.0),\n"
            "    },\n"
            "    {\n"
            "        'name': 'Rim_Light',\n"
            "        'type': 'SPOT',\n"
            "        'position': (-2.0, 3.0, 3.2),\n"
            "        'energy': 90.0,\n"
            "        'color': (1.0, 1.0, 1.0),\n"
            "    },\n"
            "]\n"
            "for cfg in configs:\n"
            "    light_data = bpy.data.lights.new(name=cfg['name'], type=cfg['type'])\n"
            "    light_data.energy = float(cfg['energy'])\n"
            "    if 'color' in cfg:\n"
            "        light_data.color = cfg['color']\n"
            "    if cfg['type'] == 'AREA' and 'size' in cfg:\n"
            "        light_data.size = float(cfg['size'])\n"
            "    if cfg['type'] == 'SPOT':\n"
            "        light_data.spot_size = 0.9\n"
            "        light_data.spot_blend = 0.3\n"
            "    light_obj = bpy.data.objects.new(cfg['name'], light_data)\n"
            "    bpy.context.scene.collection.objects.link(light_obj)\n"
            "    light_obj.location = cfg['position']\n"
            "    light_direction = mathutils.Vector((0.0, 0.0, 0.0)) - light_obj.location\n"
            "    if light_direction.length >= 1e-8:\n"
            "        light_obj.rotation_euler = light_direction.to_track_quat('-Z', 'Y').to_euler()\n"
            "\n"
            "world = scene.world\n"
            "if world is None:\n"
            "    world = bpy.data.worlds.new('World')\n"
            "    scene.world = world\n"
            "world.use_nodes = True\n"
            "bg_node = world.node_tree.nodes.get('Background')\n"
            "if bg_node is not None:\n"
            "    bg_node.inputs['Color'].default_value = (0.05, 0.05, 0.05, 1.0)\n"
            "    bg_node.inputs['Strength'].default_value = 0.3\n"
            "\n"
            "scene.render.resolution_x = int(resolution[0])\n"
            "scene.render.resolution_y = int(resolution[1])\n"
            "scene.render.resolution_percentage = 100\n"
            "selected_engine = None\n"
            "for candidate in engine_candidates:\n"
            "    try:\n"
            "        scene.render.engine = candidate\n"
            "        selected_engine = candidate\n"
            "        break\n"
            "    except TypeError:\n"
            "        continue\n"
            "if selected_engine is None:\n"
            "    raise RuntimeError(f'Unsupported render engine candidates: {engine_candidates}')\n"
            "if selected_engine == 'CYCLES':\n"
            "    scene.cycles.samples = int(samples)\n"
            "    try:\n"
            "        bpy.context.preferences.addons['cycles']\n"
            "        scene.cycles.device = 'GPU'\n"
            "    except Exception:\n"
            "        scene.cycles.device = 'CPU'\n"
            "else:\n"
            "    eevee_settings = getattr(scene, 'eevee', None)\n"
            "    if eevee_settings is not None:\n"
            "        if hasattr(eevee_settings, 'taa_render_samples'):\n"
            "            eevee_settings.taa_render_samples = int(samples)\n"
            "        elif hasattr(eevee_settings, 'samples'):\n"
            "            eevee_settings.samples = int(samples)\n"
            "scene.render.filepath = render_path\n"
            "ext = os.path.splitext(render_path)[1].lower()\n"
            "format_map = {\n"
            "    '.png': 'PNG',\n"
            "    '.jpg': 'JPEG',\n"
            "    '.jpeg': 'JPEG',\n"
            "    '.exr': 'OPEN_EXR',\n"
            "    '.tiff': 'TIFF',\n"
            "    '.tif': 'TIFF',\n"
            "    '.bmp': 'BMP',\n"
            "}\n"
            "scene.render.image_settings.file_format = format_map.get(ext, 'PNG')\n"
            "if scene.render.image_settings.file_format == 'PNG':\n"
            "    scene.render.film_transparent = False\n"
            "bpy.ops.render.render(write_still=True)\n"
            "print(render_path)\n"
        )

    async def render_full_object_v3(
        self,
        output_path: str,
        resolution: tuple = (512, 512),
        samples: int = 128,
        engine: str = "CYCLES",
        camera_location: tuple = (1.4, -1.4, 1.0),
        target_location: tuple = (0.0, 0.0, 0.5),
    ) -> Dict[str, Any]:
        """以 render_batch_v3 的风格执行 full-object 渲染。"""
        code = self._build_v3_full_object_render_code(
            output_path=output_path,
            resolution=resolution,
            samples=samples,
            engine=engine,
            camera_location=camera_location,
            target_location=target_location,
        )
        return await self.execute_code(code)
    
    async def render_object(
        self,
        object_name: str,
        output_path: str,
        resolution: tuple = (336, 336),
        samples: int = 64,
        isolate: bool = True
    ) -> Dict[str, Any]:
        """
        渲染单个对象
        
        Args:
            object_name: 对象名称
            output_path: 输出路径
            resolution: 分辨率
            samples: 采样数
            isolate: 是否隔离渲染（隐藏其他对象）
            
        Returns:
            渲染结果
        """
        return await self.execute("render_object", {
            "name": object_name,
            "output_path": output_path,
            "resolution": list(resolution),
            "samples": samples,
            "isolate": isolate
        })
    
    async def set_camera_position(
        self,
        position: tuple,
        look_at: tuple = (0, 0, 0)
    ) -> Dict[str, Any]:
        """
        设置相机位置
        
        Args:
            position: 相机位置
            look_at: 观察目标点
            
        Returns:
            操作结果
        """
        return await self.execute("set_camera_position", {
            "position": list(position),
            "look_at": list(look_at)
        })
    
    async def clear_scene(self, keep_camera: bool = True, keep_light: bool = True) -> Dict[str, Any]:
        """
        清空场景（通过执行代码实现）
        
        Args:
            keep_camera: 是否保留相机
            keep_light: 是否保留灯光
        """
        code = """
import bpy
for obj in list(bpy.data.objects):
"""
        if keep_camera and keep_light:
            code += "    if obj.type not in {'CAMERA', 'LIGHT'}:\n"
        elif keep_camera:
            code += "    if obj.type != 'CAMERA':\n"
        elif keep_light:
            code += "    if obj.type != 'LIGHT':\n"
        else:
            code += "    if True:\n"
        
        code += """        bpy.data.objects.remove(obj, do_unlink=True)
print('scene cleared')
"""
        return await self.execute_code(code)
    
    async def save_blend_file(self, filepath: str) -> Dict[str, Any]:
        """
        保存 .blend 文件（通过执行代码实现）
        
        Args:
            filepath: 保存路径
        """
        code = f"bpy.ops.wm.save_as_mainfile(filepath=r'{filepath}')\nprint('saved')"
        return await self.execute_code(code) 
    
    async def load_blend_file(self, filepath: str) -> Dict[str, Any]:
        """
        加载 .blend 文件（通过执行代码实现）
        
        Args:
            filepath: 文件路径
        """
        code = f"bpy.ops.wm.open_mainfile(filepath=r'{filepath}')\nprint('loaded')"
        return await self.execute_code(code)
    
    # ==================== 上下文管理器 ====================
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.disconnect()
        return False


class MockBlenderMCPClient(BlenderMCPClient):
    """
    Mock Blender MCP Client
    用于测试，不需要实际的 Blender MCP Server
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mock_objects = {}
        self._mock_scene = {
            "name": "Scene",
            "object_count": 0,
            "objects": [],
            "materials_count": 0
        }
    
    async def connect(self) -> bool:
        """模拟连接"""
        self._connected = True
        self.logger.info("Mock client connected")
        return True
    
    async def disconnect(self) -> None:
        """模拟断开"""
        self._connected = False
        self.logger.info("Mock client disconnected")
    
    async def _send_command(self, cmd_type: str, params: Dict[str, Any] = None) -> MCPResponse:
        """发送命令并等待响应"""
        if not self._connected:
            return MCPResponse(success=False, error="Not connected")
        
        self._request_id += 1
        request = {
            "id": self._request_id,
            "type": cmd_type,  # 确认 Server 确实在看 "type" 字段
            "params": params or {}
        }
        
        try:
            # 发送请求
            data = json.dumps(request).encode() + b'\n'
            self.logger.debug(f"Sending: {request}")
            self._writer.write(data)
            await self._writer.drain()
            
            # 读取响应 - 修改为非 readline 方式
            self.logger.debug("Waiting for response...")
            
            # 方法改进：读取数据块并尝试解析 JSON
            response_data = b""
            try:
                # 读取第一块数据
                chunk = await asyncio.wait_for(
                    self._reader.read(4096),  # 读取最大 4KB，通常足够容纳响应
                    timeout=self.timeout
                )
                response_data += chunk
                
                # 尝试解析
                text = response_data.decode().strip()
                response = json.loads(text)
                
            except json.JSONDecodeError:
                # 如果 JSON 不完整，可能需要继续读取（这里简化处理，假设一次发完）
                self.logger.error(f"Invalid JSON received: {response_data}")
                return MCPResponse(success=False, error="Invalid JSON response")
            except asyncio.TimeoutError:
                self.logger.error(f"Command timed out: {cmd_type}")
                return MCPResponse(success=False, error="Timeout")
            
            self.logger.debug(f"Received: {response}")
            
            # 兼容处理：Server 返回的是 status: error 还是 success: false
            success = response.get("success", False)
            if "status" in response:
                success = response["status"] == "success"
                
            return MCPResponse(
                success=success,
                result=response.get("result"),
                error=response.get("error") or response.get("message") # 兼容 error/message 字段
            )
            
        except Exception as e:
            self.logger.error(f"Command failed: {e}")
            return MCPResponse(success=False, error=str(e))
