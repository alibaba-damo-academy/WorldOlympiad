"""
Agent Pipeline
主入口，协调各个 Agent 组件完成完整的 3D 物体生成流程
"""

import ast
import os
import json
import asyncio
import logging
import re
import shutil
from typing import Any, Dict, Optional, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import yaml

from .schemas import ObjectPlan
from .agents import (
    PlannerAgent,
    PlannerCheckerAgent,
    PartConstructorAgent,
)
from .blender import BlenderMCPClient, BlenderBackendManager
from .utils.llm_client import create_llm_client, LLMClient, TokenUsageTracker
from .utils.code_naming import get_parts_package_name


@dataclass
class PipelineResult:
    """Pipeline 执行结果"""
    success: bool
    object_name: str
    output_dir: str
    object_plan: Optional[ObjectPlan] = None
    stages_completed: List[str] = field(default_factory=list)
    stages_failed: List[str] = field(default_factory=list)
    total_time: float = 0.0
    error: Optional[str] = None
    mesh_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "object_name": self.object_name,
            "output_dir": self.output_dir,
            "object_plan": self.object_plan.to_dict() if self.object_plan else None,
            "stages_completed": self.stages_completed,
            "stages_failed": self.stages_failed,
            "total_time": self.total_time,
            "error": self.error,
            "mesh_path": self.mesh_path
        }


class Pipeline:
    """
    Agent Pipeline
    
    完整的处理流程：
    1. Planner: 分析图片或文本，生成 ObjectPlan
    2. Planner Checker: 检查 ObjectPlan 的有效性
    3. Part Constructor: 为每个部件生成代码并执行（仅 generation）
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
        use_mock: bool = False,
        port: Optional[int] = None,
    ):
        """
        初始化 Pipeline
        
        Args:
            config_path: 配置文件路径
            llm_client: LLM 客户端（如果提供，将作为所有阶段的默认客户端）
            use_mock: 是否使用 Mock 模式（不需要实际的 Blender）
            port: Blender MCP 端口（优先级：args > mcp_config > default 9876）
        """
        self.config = self._load_config(config_path)
        self.use_mock = use_mock
        self.logger = self._setup_logger()
        self._port = port

        # Token 用量统计（在 _init_llm_client 中创建并注入到 client）
        self._usage_tracker = None

        # 初始化 LLM 客户端（分别为 planner 和 constructor 阶段）
        if llm_client is not None:
            # 如果外部提供了客户端，所有阶段共用
            self.llm_client = llm_client
            self.planner_llm_client = llm_client
            self.constructor_llm_client = llm_client
        elif not use_mock:
            # 初始化不同阶段的 LLM 客户端
            self.llm_client = self._init_llm_client()  # 默认客户端（回退用）
            self.planner_llm_client = self._init_llm_client(config_key="planner_llm")
            self.constructor_llm_client = self._init_llm_client(config_key="constructor_llm")
        else:
            self.llm_client = None
            self.planner_llm_client = None
            self.constructor_llm_client = None
            self.logger.info("Mock mode: LLM clients not initialized")

        # Blender 客户端
        self.blender_client = None
        self.blender_backend_manager = None
        
        # 初始化 Agents
        self._init_agents(config_path)

    def _init_llm_client(self, config_key: str = "llm") -> Optional[LLMClient]:
        """
        根据配置初始化 LLM 客户端
        
        Args:
            config_key: 配置键名，可选值:
                - "llm": 默认 LLM 配置
                - "planner_llm": Planner 阶段专用配置
                - "constructor_llm": Constructor 阶段专用配置
        
        Returns:
            LLMClient 实例
        """
        model_config = self.config.get("model", {}).get(config_key, {})
        
        # 如果指定的配置不存在，回退到默认 llm 配置
        if not model_config and config_key != "llm":
            self.logger.warning(f"Config '{config_key}' not found, falling back to default 'llm' config")
            model_config = self.config.get("model", {}).get("llm", {})

        provider = model_config.get("provider", "openai")
        model_name = model_config.get("model_name")
        api_base = model_config.get("api_base")
        temperature = model_config.get("temperature", 0.7)

        self.logger.info(f"Initializing LLM client [{config_key}]: provider={provider}, model={model_name}")

        log_dir = self.config.get("paths", {}).get("log_dir", "./logs")
        # 只在第一次初始化时创建 usage_tracker
        if self._usage_tracker is None:
            self._usage_tracker = TokenUsageTracker(log_dir=log_dir, logger_name="Pipeline.LLM")

        try:
            kwargs = {}
            if model_name:
                kwargs["model"] = model_name
            if api_base:
                kwargs["api_base"] = api_base
            if temperature:
                kwargs["temperature"] = temperature

            client = create_llm_client(
                provider=provider,
                usage_tracker=self._usage_tracker,
                **kwargs
            )
            self.logger.info(f"LLM client [{config_key}] initialized successfully (token usage tracking enabled)")
            return client

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client [{config_key}]: {e}")
            raise RuntimeError(f"LLM client [{config_key}] initialization failed: {e}")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置"""
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "paths": {
                "output_dir": "./dataset_out",
                "temp_dir": "./temp",
                "log_dir": "./logs"
            },
            "pipeline": {
                "enable_planner_check": True,
                "enable_part_construct": True,
                "enable_glb_export": True,
                "save_intermediate": True
            },
            "blender": {
                "mcp_server": {
                    "host": "127.0.0.1",
                    "port": 9876
                },
                "runtime": {
                    "command": "blender",
                    "startup_timeout": 45.0,
                    "stop_timeout": 10.0
                }
            },
            "logging": {
                "level": "INFO"
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("Pipeline")
        # Prevent duplicate output through the root logger.
        logger.propagate = False

        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        logger.setLevel(level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # 文件处理器
            log_dir = self.config.get("paths", {}).get("log_dir", "./logs")
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _init_agents(self, config_path: Optional[str]) -> None:
        """初始化所有 Agents，使用不同阶段的 LLM 客户端"""
        # 使用 Mock 版本或正式版本
        if self.use_mock:
            from .agents.planner import MockPlannerAgent
            from .agents.part_constructor import MockPartConstructorAgent
            
            # Planner 阶段使用 planner_llm_client
            self.planner = MockPlannerAgent(config_path, self.planner_llm_client)
            self.planner_checker = PlannerCheckerAgent(config_path, self.planner_llm_client)
            
            # Constructor 阶段使用 constructor_llm_client（仅 generation）
            self.part_constructor = MockPartConstructorAgent(
                config_path, self.constructor_llm_client
            )
        else:
            # Planner 阶段使用 planner_llm_client（模型 A）
            self.planner = PlannerAgent(config_path, self.planner_llm_client)
            self.planner_checker = PlannerCheckerAgent(config_path, self.planner_llm_client)
            
            # Constructor 阶段使用 constructor_llm_client（模型 B），仅 generation
            self.part_constructor = PartConstructorAgent(
                config_path, self.constructor_llm_client
            )
    
    async def _connect_blender(self) -> bool:
        """启动并连接 Blender headless backend（每次 pipeline 独立生命周期）。"""
        if self.use_mock:
            from .blender.mcp_client import MockBlenderMCPClient
            self.blender_client = MockBlenderMCPClient()
            connected = await self.blender_client.connect()
        else:
            mcp_config = self.config.get("blender", {}).get("mcp_server", {})
            runtime_config = self.config.get("blender", {}).get("runtime", {})
            host = mcp_config.get("host", "127.0.0.1")
            # port 优先级: args > mcp_config > default
            port = self._port if self._port is not None else int(mcp_config.get("port", 9876))
            blender_command = runtime_config.get("command", "blender")
            startup_timeout = float(runtime_config.get("startup_timeout", 45.0))
            stop_timeout = float(runtime_config.get("stop_timeout", 10.0))
            server_script = runtime_config.get("server_script")
            backend_log_file = runtime_config.get("backend_log_file")

            self.blender_backend_manager = BlenderBackendManager(
                blender_command=blender_command,
                host=host,
                port=port,
                server_script=server_script,
                startup_timeout=startup_timeout,
                stop_timeout=stop_timeout,
                log_file=backend_log_file,
            )
            await self.blender_backend_manager.start()

            self.blender_client = BlenderMCPClient(host=host, port=port)
            connected = await self.blender_client.connect()
            if not connected:
                await self.blender_backend_manager.stop()
                self.blender_backend_manager = None
                return False
        
        if connected:
            # 设置各 Agent 的 blender_client
            self.planner.blender_client = self.blender_client
            self.planner_checker.blender_client = self.blender_client
            self.part_constructor.blender_client = self.blender_client
        
        return connected
    
    async def _disconnect_blender(self) -> None:
        """断开连接并关闭本次 pipeline 启动的 Blender backend。"""
        if self.blender_client:
            try:
                await self.blender_client.disconnect()
            except Exception as e:
                self.logger.warning("Failed to disconnect blender client: %s", e)
            finally:
                self.blender_client = None
        if self.blender_backend_manager:
            try:
                await self.blender_backend_manager.stop()
            except Exception as e:
                self.logger.warning("Failed to stop blender backend: %s", e)
            finally:
                self.blender_backend_manager = None
    
    def _validate_generation_input(
        self,
        image_path: Optional[str] = None,
        text_input: Optional[str] = None
    ) -> str:
        """完整生成模式下，校验输入必须二选一。"""
        has_image = bool(image_path and str(image_path).strip())
        has_text = bool(text_input and str(text_input).strip())
        if has_image == has_text:
            raise ValueError("Full generation requires exactly one of image_path or text_input")
        return "image" if has_image else "text"

    def _extract_text_object_name(self, text_input: Optional[str]) -> str:
        """从文本中提取输出目录前缀名。"""
        if hasattr(self, "planner") and hasattr(self.planner, "_extract_object_name_from_text"):
            return self.planner._extract_object_name_from_text(text_input)
        return "object"

    def _prepare_output_dir(
        self,
        image_path: Optional[str] = None,
        text_input: Optional[str] = None
    ) -> str:
        """准备输出目录（支持 image 或 text 输入）。"""
        input_mode = self._validate_generation_input(image_path=image_path, text_input=text_input)

        if input_mode == "image":
            input_name = Path(image_path).stem
        else:
            input_name = self._extract_text_object_name(text_input)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{input_name}_{timestamp}"
        
        base_output_dir = self.config.get("paths", {}).get("output_dir", "./dataset_out")
        output_dir = os.path.join(base_output_dir, output_name)
        
        # 创建目录结构
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "code"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "renders"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "renders", "parts"), exist_ok=True)
        
        if input_mode == "image":
            shutil.copy(image_path, os.path.join(output_dir, "image.png"))
        else:
            with open(os.path.join(output_dir, "input_text.txt"), "w", encoding="utf-8") as f:
                f.write((text_input or "").strip())
        
        return output_dir
    
    def _check_existing_plan(self, output_dir: str) -> Optional[ObjectPlan]:
        """检查是否存在已生成的 ObjectPlan"""
        plan_path = os.path.join(output_dir, "ObjectPlan.json")
        if os.path.exists(plan_path):
            self.logger.info(f"Found existing ObjectPlan at {plan_path}, loading...")
            try:
                return ObjectPlan.load(plan_path)
            except Exception as e:
                self.logger.warning(f"Failed to load existing plan: {e}, will regenerate")
        return None

    def _check_existing_construction(self, output_dir: str, object_plan: ObjectPlan) -> Optional[Dict]:
        """检查是否存在已完成的构建结果"""
        code_dir = os.path.join(output_dir, "code", get_parts_package_name(object_plan.name))
        if os.path.exists(code_dir):
            # 检查所有部件的代码文件是否存在
            existing_parts = []
            for part in object_plan.parts:
                part_file = os.path.join(code_dir, f"{part.name}.py")
                if os.path.exists(part_file):
                    existing_parts.append(part.name)
            
            if len(existing_parts) == len(object_plan.parts):
                self.logger.info(f"Found existing construction for all {len(existing_parts)} parts, skipping...")
                return {"construct_success": True, "part_results": existing_parts, "skipped": True}
        return None

    def _fix_main_block(self, code: str) -> str:
        """将 if __name__ == '__main__' 修改为 if True，便于在 Blender 中 exec 执行"""
        pattern = r'if\s+__name__\s*==\s*["\']__main__["\']\s*:'
        return re.sub(pattern, 'if True:', code)

    def _get_ordered_part_script_paths(self, code_dir: str, object_name: str) -> List[Path]:
        """按总脚本 import 顺序解析需要执行的部件脚本。"""
        main_path = Path(code_dir) / f"{object_name}.py"
        if not main_path.exists():
            raise FileNotFoundError(f"Main script not found: {main_path}")

        source = main_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(main_path))

        parts_pkg: Optional[str] = None
        ordered_modules: List[str] = []
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level != 0 or not node.module or not node.module.startswith("parts_"):
                continue

            module_parts = node.module.split(".")
            if len(module_parts) != 2:
                raise RuntimeError(
                    f"Unexpected parts import format in {main_path}: {node.module}"
                )

            pkg_name, part_module = module_parts
            if parts_pkg is None:
                parts_pkg = pkg_name
            elif parts_pkg != pkg_name:
                raise RuntimeError(
                    f"Multiple parts packages found in {main_path}: "
                    f"{sorted({parts_pkg, pkg_name})}"
                )
            ordered_modules.append(part_module)

        if not ordered_modules or parts_pkg is None:
            raise RuntimeError(f"No parts_* imports found in main script: {main_path}")

        parts_dir = Path(code_dir) / parts_pkg
        if not parts_dir.exists():
            raise FileNotFoundError(f"Parts directory not found: {parts_dir}")

        ordered_paths: List[Path] = []
        for module_name in ordered_modules:
            script_path = parts_dir / f"{module_name}.py"
            if not script_path.exists():
                raise FileNotFoundError(f"Part script not found: {script_path}")
            ordered_paths.append(script_path)
        return ordered_paths

    def _build_ordered_part_execution_code(
        self,
        *,
        code_dir: str,
        ordered_part_scripts: List[Path],
    ) -> str:
        """构造按顺序执行部件脚本的 Blender 代码。"""
        code_dir_abs = os.path.abspath(code_dir)
        snippets = [
            "import sys",
            f"if r\"{code_dir_abs}\" not in sys.path:",
            f"    sys.path.insert(0, r\"{code_dir_abs}\")",
        ]
        for script_path in ordered_part_scripts:
            script_path_abs = script_path.resolve()
            script_dir_abs = str(script_path_abs.parent)
            script_code = self._fix_main_block(
                script_path_abs.read_text(encoding="utf-8")
            )
            snippets.extend(
                [
                    f"if r\"{script_dir_abs}\" not in sys.path:",
                    f"    sys.path.insert(0, r\"{script_dir_abs}\")",
                    f"__file__ = r\"{script_path_abs}\"",
                    script_code,
                ]
            )
        return "\n".join(snippets)

    async def _ensure_scene_has_full_object(self, output_dir: str, object_plan: ObjectPlan) -> bool:
        """
        确保 Blender 场景中已构建完整物体（用于使用 cache 或仅 check 时场景为空的情况）。
        按总脚本 import 顺序执行各个部件脚本，在场景中创建全部部件。
        """
        code_dir = os.path.join(output_dir, "code")
        try:
            ordered_part_scripts = self._get_ordered_part_script_paths(
                code_dir, object_plan.name
            )
            code_to_run = self._build_ordered_part_execution_code(
                code_dir=code_dir,
                ordered_part_scripts=ordered_part_scripts,
            )
            exec_result = await self.blender_client.execute_code(code_to_run)
            if not exec_result.get("success", True):
                self.logger.error(
                    "Ordered part execution failed: %s",
                    exec_result.get("error", "unknown"),
                )
                return False
            self.logger.info(
                "Full object built in scene from %d ordered part scripts",
                len(ordered_part_scripts),
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to build full object in scene: {e}")
            return False

    async def _clear_blender_scene(self) -> None:
        """清空 Blender 当前场景中的所有对象。"""
        if self.blender_client is None:
            return
        clear_scene_code = """
import bpy
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
print('scene cleared')
"""
        await self.blender_client.execute_code(clear_scene_code)

    async def _rebuild_scene_from_saved_code(self, output_dir: str, object_plan: ObjectPlan) -> bool:
        """从保存的主脚本重建场景，确保导出结果与最终代码一致。"""
        await self._clear_blender_scene()
        return await self._ensure_scene_has_full_object(output_dir, object_plan)

    async def _export_full_object_glb(
        self,
        output_dir: str,
        object_plan: ObjectPlan,
        mesh_output_path: Optional[str] = None,
    ) -> Optional[str]:
        """将完整物体导出为单文件 .glb。"""
        if self.blender_client is None:
            return None

        scene_ready = await self._rebuild_scene_from_saved_code(output_dir, object_plan)
        if not scene_ready:
            return None

        export_dir = os.path.join(output_dir, "mesh")
        os.makedirs(export_dir, exist_ok=True)
        output_path = os.path.abspath(
            mesh_output_path or os.path.join(export_dir, f"{object_plan.name}.glb")
        )
        try:
            result = await self.blender_client.export_scene_to_glb(
                output_path=output_path,
                use_selection=False,
                export_apply=False,
            )
            if result and os.path.exists(output_path):
                self.logger.info("Full object exported to GLB: %s", output_path)
                return output_path
        except Exception as e:
            self.logger.error("Failed to export GLB: %s", e)
        return None

    async def _render_full_object_and_save(
        self,
        output_dir: str,
        object_plan: ObjectPlan,
        need_build_scene: bool
    ) -> Optional[str]:
        """
        渲染整个物体并保存到 output_dir/renders/full_object.png。
        为避免上游阶段残留单个 part，渲染前总是清空并重建完整场景。
        """
        if self.blender_client is None:
            return None

        # Always rebuild the full object scene before rendering. Earlier stages
        # such as per-part export leave Blender containing only the last part.
        scene_ready = await self._rebuild_scene_from_saved_code(output_dir, object_plan)
        if not scene_ready:
            return None

        render_dir = os.path.join(output_dir, "renders")
        os.makedirs(render_dir, exist_ok=True)
        output_path = os.path.abspath(os.path.join(render_dir, "full_object.png"))
        try:
            render_config = self.config.get("blender", {}).get("render", {})
            scene_config = self.config.get("blender", {}).get("scene", {})
            resolution = (
                render_config.get("resolution_x", 512),
                render_config.get("resolution_y", 512)
            )
            samples = render_config.get("samples", 128)
            camera_position = tuple(scene_config.get("camera_position", [1.4, -1.4, 1.0]))
            result = await self.blender_client.render_full_object_v3(
                output_path=output_path,
                resolution=resolution,
                samples=samples,
                camera_location=camera_position,
                target_location=(0.0, 0.0, 0.5),
            )
            if result and (result.get("success", True) if isinstance(result, dict) else True) and os.path.exists(output_path):
                self.logger.info(f"Full object rendered and saved to: {output_path}")
                return output_path
        except Exception as e:
            self.logger.error(f"Failed to render full object: {e}")
        return None

    async def run(
        self,
        image_path: Optional[str] = None,
        text_input: Optional[str] = None,
        output_dir: Optional[str] = None,
        skip_stages: List[str] = None,
        mesh_output_path: Optional[str] = None,
    ) -> PipelineResult:
        """
        执行完整 Pipeline
        
        Args:
            image_path: 输入图片路径（完整模式下与 text_input 二选一）
            text_input: 输入文本描述（完整模式下与 image_path 二选一）
            output_dir: 输出目录（可选，默认自动生成）
            skip_stages: 跳过的阶段列表
            
        Returns:
            PipelineResult
        """
        start_time = datetime.now()
        skip_stages = skip_stages or []
        stages_completed = []
        stages_failed = []
        
        self.logger.info("=" * 60)
        try:
            input_mode = self._validate_generation_input(image_path=image_path, text_input=text_input)
        except ValueError as e:
            return PipelineResult(
                success=False,
                object_name="",
                output_dir=output_dir or "",
                error=str(e)
            )
        if input_mode == "image":
            self.logger.info(f"Starting Pipeline for image: {image_path}")
        else:
            self.logger.info("Starting Pipeline for text input")
        self.logger.info("=" * 60)
        
        # 检查输入
        if input_mode == "image" and not os.path.exists(image_path):
            return PipelineResult(
                success=False,
                object_name="",
                output_dir="",
                error=f"Image not found: {image_path}"
            )
        
        # 准备输出目录
        if output_dir is None:
            output_dir = self._prepare_output_dir(image_path=image_path, text_input=text_input)
        else:
            os.makedirs(output_dir, exist_ok=True)
            if input_mode == "image":
                shutil.copy(image_path, os.path.join(output_dir, "image.png"))
            else:
                with open(os.path.join(output_dir, "input_text.txt"), "w", encoding="utf-8") as f:
                    f.write((text_input or "").strip())
        
        self.logger.info(f"Output directory: {output_dir}")
        
        # 连接 Blender
        self.logger.info("Connecting to Blender MCP Server...")
        if not await self._connect_blender():
            raise RuntimeError("Failed to connect to Blender headless backend")
        
        # 每次 pipeline 启动前清空 Blender 场景（包括相机和灯光）
        self.logger.info("Clearing Blender scene...")
        await self._clear_blender_scene()
        self.logger.info("Blender scene cleared")

        object_plan = None
        mesh_path = None
        
        try:
            # ==================== Stage 1: Planning ====================
            self.logger.info("\n" + "="*40)
            self.logger.info("Stage 1: Planning")
            self.logger.info("="*40)

            object_plan = self._check_existing_plan(output_dir)
            if "planning" not in skip_stages:
                
                if object_plan is None:
                    try:
                        plan_path = os.path.join(output_dir, "ObjectPlan.json")
                        object_plan = await self.planner.run(
                            image_path=image_path,
                            text_input=text_input,
                            output_path=plan_path
                        )
                        stages_completed.append("planning")
                        self.logger.info(f"Planning completed: {len(object_plan.parts)} parts identified")
                    except Exception as e:
                        self.logger.error(f"Planning failed: {e}")
                        stages_failed.append("planning")
                        raise
            else:
                self.logger.info("Skipping planning stage")
            if os.path.exists(os.path.join(output_dir, "code", f"{object_plan.name}.py")):
                existing_main_path = os.path.join(output_dir, "code", f"{object_plan.name}.py")
                self.logger.info(
                    f"Skipping construction because object code already exists: {existing_main_path}"
                )
                return PipelineResult(
                    success=True,
                    object_name=object_plan.name,
                    output_dir=output_dir,
                    error="Skipping construction because part code already exists"
                )
            
            # ==================== Stage 2: Plan Check ====================
            if self.config.get("pipeline", {}).get("enable_planner_check", True):
                self.logger.info("\n" + "="*40)
                self.logger.info("Stage 2: Plan Check")
                self.logger.info("="*40)
                
                if "plan_check" not in skip_stages and object_plan:
                    try:
                        object_plan, check_history = await self.planner_checker.check_and_fix(
                            plan=object_plan,
                            image_path=image_path,
                            text_input=text_input,
                            planner_agent=self.planner
                        )
                        
                        # 保存更新后的 plan
                        object_plan.save(os.path.join(output_dir, "ObjectPlan.json"))
                        
                        # 保存检查历史 并非实时保存，max_retry 后保存全部内容
                        if self.config.get("pipeline", {}).get("save_intermediate", True):
                            history_path = os.path.join(output_dir, "plan_check_history.json")
                            with open(history_path, 'a', encoding='utf-8') as f:
                                json.dump([h.to_dict() for h in check_history], f, indent=2, ensure_ascii=False)
                        
                        stages_completed.append("plan_check")
                        self.logger.info("Plan check completed")
                    except Exception as e:
                        self.logger.error(f"Plan check failed: {e}")
                        stages_failed.append("plan_check")
                else:
                    self.logger.info("Skipping plan check stage")
            
            # ==================== Stage 3: Part Construction & Check ====================
            # 先构造全部部件，再统一检查
            self.logger.info("\n" + "="*40)
            self.logger.info("Stage 3: Part Construction & Check")
            self.logger.info("="*40)
            
            construction_result = None
            if "construction" not in skip_stages and object_plan:
                cached_result = self._check_existing_construction(output_dir, object_plan)

                pipeline_cfg = self.config.get("pipeline", {})
                enable_part_construct = pipeline_cfg.get("enable_part_construct", True)
                use_cache = cached_result and enable_part_construct

                if not use_cache:
                    try:
                        code_dir = os.path.join(output_dir, "code")
                        render_dir = os.path.join(output_dir, "renders", "parts")
                        
                        construction_result = await self.part_constructor.run(
                            object_plan=object_plan,
                            image_path=image_path,
                            text_input=text_input,
                            output_dir=code_dir,
                            render_output_dir=render_dir,
                            use_llm=self.llm_client is not None,
                            execute=self.blender_client is not None,
                            construct_parts=enable_part_construct
                        )
                        
                        construct_success = construction_result.get("construct_success", True)
                        
                        if construct_success or not enable_part_construct:
                            stages_completed.append("construction")
                            self.logger.info(f"Construction completed: {len(construction_result['part_results'])} parts")
                        else:
                            stages_failed.append("construction")
                            self.logger.warning(f"Construction partially failed: {construction_result.get('failed_parts', [])}")
                        
                    except Exception as e:
                        self.logger.error(f"Construction failed: {e}")
                        stages_failed.append("construction")
                else:
                    construction_result = cached_result
                    stages_completed.append("construction")
                    self.logger.info("Using cached construction, skipping Part Constructor run")
            else:
                self.logger.info("Skipping construction stage")

            # ==================== Render full object ====================
            # 所有 part 生成完毕后，渲染整个物体并保存
            if (
                object_plan is not None
                and self.blender_client is not None
                and "construction" not in skip_stages
                and construction_result is not None
                and "render_full_object" not in skip_stages
            ):
                pipeline_cfg = self.config.get("pipeline", {})
                use_cache = (
                    construction_result.get("skipped") is True
                    and pipeline_cfg.get("enable_part_construct", True)
                )
                need_build_scene = use_cache
                try:
                    render_path = await self._render_full_object_and_save(
                        output_dir, object_plan, need_build_scene=need_build_scene
                    )
                    if render_path:
                        stages_completed.append("render_full_object")
                except Exception as e:
                    self.logger.warning(f"Render full object failed (non-fatal): {e}")
            
            # ==================== Export GLB ====================
            if (
                object_plan is not None
                and self.blender_client is not None
                and self.config.get("pipeline", {}).get("enable_glb_export", True)
                and "export_glb" not in skip_stages
            ):
                self.logger.info("\n" + "=" * 40)
                self.logger.info("Stage 4: Export GLB")
                self.logger.info("=" * 40)
                mesh_path = await self._export_full_object_glb(
                    output_dir=output_dir,
                    object_plan=object_plan,
                    mesh_output_path=mesh_output_path,
                )
                if mesh_path:
                    stages_completed.append("export_glb")
                else:
                    stages_failed.append("export_glb")

            # ==================== Complete ====================
            total_time = (datetime.now() - start_time).total_seconds()
            
            success = len(stages_failed) == 0 and len(stages_completed) > 0
            
            result = PipelineResult(
                success=success,
                object_name=object_plan.name if object_plan else "",
                output_dir=output_dir,
                object_plan=object_plan,
                stages_completed=stages_completed,
                stages_failed=stages_failed,
                total_time=total_time,
                mesh_path=mesh_path
            )
            
            # 保存最终结果
            result_path = os.path.join(output_dir, "pipeline_result.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            
            self.logger.info("\n" + "="*60)
            self.logger.info(f"Pipeline completed in {total_time:.2f} seconds")
            self.logger.info(f"Success: {success}")
            self.logger.info(f"Stages completed: {stages_completed}")
            self.logger.info(f"Stages failed: {stages_failed}")
            self.logger.info(f"Output: {output_dir}")
            if self._usage_tracker:
                summary = self._usage_tracker.get_summary()
                self.logger.info(
                    "Token usage: total_input=%d total_output=%d total=%d calls=%d",
                    summary["total_input_tokens"],
                    summary["total_output_tokens"],
                    summary["total_tokens"],
                    summary["call_count"],
                )
            self.logger.info("="*60)

            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {e}")
            total_time = (datetime.now() - start_time).total_seconds()
            
            return PipelineResult(
                success=False,
                object_name=object_plan.name if object_plan else "",
                output_dir=output_dir,
                object_plan=object_plan,
                stages_completed=stages_completed,
                stages_failed=stages_failed,
                total_time=total_time,
                error=str(e),
                mesh_path=mesh_path
            )
            
        finally:
            # 断开 Blender 连接
            await self._disconnect_blender()


async def run_single(args) -> int:
    """处理单张图片"""
    import traceback
    
    try:
        # 打印当前脚本的绝对路径，便于确认实际运行的入口文件
        try:
            print(f"[INFO] pipeline_batch file path: {os.path.abspath(__file__)}")
        except Exception:
            pass
        
        has_image = bool(args.image)
        has_text_jsonl = bool(getattr(args, "text_jsonl", None))
        if has_image == has_text_jsonl:
            print("[ERROR] single mode requires exactly one of <image> or --text-jsonl")
            return 1

        # 创建 Pipeline
        print("[DEBUG] Creating Pipeline...")
        pipeline = Pipeline(
            config_path=args.config,
            use_mock=args.mock,
            port=getattr(args, "port", None),
        )
        print("[DEBUG] Pipeline created successfully")

        image_path = args.image if has_image else None
        text_input = None
        if has_text_jsonl:
            if not os.path.exists(args.text_jsonl):
                print(f"[ERROR] JSONL file not found: {args.text_jsonl}")
                return 1

            text_records = []
            with open(args.text_jsonl, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] Invalid JSON in {args.text_jsonl}:{line_no}: {e}")
                        return 1
                    text_desc = str(item.get("text_description", "")).strip()
                    if text_desc:
                        text_records.append(text_desc)

            if not text_records:
                print(f"[ERROR] No valid text_description found in {args.text_jsonl}")
                return 1
            if len(text_records) > 1:
                print(
                    "[ERROR] single mode with --text-jsonl expects exactly one valid record; "
                    "use batch mode for multiple records"
                )
                return 1
            text_input = text_records[0]
        
        # 执行
        print("[DEBUG] Running Pipeline...")
        result = await pipeline.run(
            image_path=image_path,
            text_input=text_input,
            output_dir=args.output,
            skip_stages=args.skip
        )
        print(f"[DEBUG] Pipeline finished: {result.success}")
        
        return 0 if result.success else 1
        
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        traceback.print_exc()
        return 1


async def run_batch(args) -> int:
    """批量处理（图片目录或文本 JSONL，二选一）"""
    import traceback
    import glob
    
    base_output_dir = args.output_dir
    has_image_dir = bool(args.image_dir)
    has_text_jsonl = bool(getattr(args, "text_jsonl", None))
    if has_image_dir == has_text_jsonl:
        print("[ERROR] batch mode requires exactly one of --image-dir or --text-jsonl")
        return 1

    # 创建 Pipeline（复用同一个实例）
    print("[DEBUG] Creating Pipeline...")
    pipeline = Pipeline(
        config_path=args.config,
        use_mock=args.mock,
        port=getattr(args, "port", None),
    )
    print("[DEBUG] Pipeline created successfully")

    # 统计结果（根据输入模式设置 total）
    results = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "details": []
    }

    if has_image_dir:
        image_dir = args.image_dir
        # 支持的图片格式
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']

        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))

        # 排序以保证处理顺序一致
        image_files.sort()

        if not image_files:
            print(f"[ERROR] No image files found in {image_dir}")
            return 1

        results["total"] = len(image_files)
        print(f"[INFO] Found {len(image_files)} images to process")
        print(f"[INFO] Output base directory: {base_output_dir}")
        print("="*60)

        # 逐个处理图片
        for idx, image_path in enumerate(image_files, 1):
            image_name = Path(image_path).stem
            output_dir = os.path.join(base_output_dir, image_name)

            print(f"\n{'='*60}")
            print(f"[{idx}/{len(image_files)}] Processing: {image_name}")
            print(f"  Input: {image_path}")
            print(f"  Output: {output_dir}")
            print("="*60)

            try:
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(os.path.join(output_dir, "code"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "renders"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "renders", "parts"), exist_ok=True)

                shutil.copy(image_path, os.path.join(output_dir, "image.png"))

                result = await pipeline.run(
                    image_path=image_path,
                    output_dir=output_dir,
                    skip_stages=args.skip
                )

                if result.success:
                    results["success"] += 1
                    print(f"[SUCCESS] {image_name} completed successfully")
                else:
                    results["failed"] += 1
                    print(f"[FAILED] {image_name} failed: {result.error}")

                results["details"].append({
                    "input_mode": "image",
                    "image_name": image_name,
                    "image_path": image_path,
                    "output_dir": output_dir,
                    "success": result.success,
                    "error": result.error,
                    "stages_completed": result.stages_completed,
                    "stages_failed": result.stages_failed,
                    "total_time": result.total_time
                })

            except Exception as e:
                results["failed"] += 1
                error_msg = str(e)
                print(f"[ERROR] {image_name} failed with exception: {error_msg}")
                traceback.print_exc()

                results["details"].append({
                    "input_mode": "image",
                    "image_name": image_name,
                    "image_path": image_path,
                    "output_dir": output_dir,
                    "success": False,
                    "error": error_msg,
                    "stages_completed": [],
                    "stages_failed": ["exception"],
                    "total_time": 0
                })
    else:
        text_jsonl = args.text_jsonl
        if not os.path.exists(text_jsonl):
            print(f"[ERROR] JSONL file not found: {text_jsonl}")
            return 1

        text_records = []
        with open(text_jsonl, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Invalid JSON in {text_jsonl}:{line_no}: {e}")
                    return 1

                text_desc = str(item.get("text_description", "")).strip()
                if not text_desc:
                    continue

                item_id = str(item.get("id", f"item_{line_no:04d}")).strip() or f"item_{line_no:04d}"
                text_records.append({
                    "id": item_id,
                    "text_description": text_desc,
                })

        if not text_records:
            print(f"[ERROR] No valid text_description found in {text_jsonl}")
            return 1

        results["total"] = len(text_records)
        print(f"[INFO] Found {len(text_records)} text records to process")
        print(f"[INFO] Output base directory: {base_output_dir}")
        print("="*60)

        for idx, item in enumerate(text_records, 1):
            item_id = item["id"]
            text_input = item["text_description"]
            output_dir = os.path.join(base_output_dir, item_id)

            print(f"\n{'='*60}")
            print(f"[{idx}/{len(text_records)}] Processing text id: {item_id}")
            print(f"  Input JSONL: {text_jsonl}")
            print(f"  Output: {output_dir}")
            print("="*60)

            try:
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(os.path.join(output_dir, "code"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "renders"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "renders", "parts"), exist_ok=True)

                with open(os.path.join(output_dir, "input_text.txt"), "w", encoding="utf-8") as f:
                    f.write(text_input)

                result = await pipeline.run(
                    text_input=text_input,
                    output_dir=output_dir,
                    skip_stages=args.skip
                )

                if result.success:
                    results["success"] += 1
                    print(f"[SUCCESS] {item_id} completed successfully")
                else:
                    results["failed"] += 1
                    print(f"[FAILED] {item_id} failed: {result.error}")

                results["details"].append({
                    "input_mode": "text",
                    "text_id": item_id,
                    "output_dir": output_dir,
                    "success": result.success,
                    "error": result.error,
                    "stages_completed": result.stages_completed,
                    "stages_failed": result.stages_failed,
                    "total_time": result.total_time
                })

            except Exception as e:
                results["failed"] += 1
                error_msg = str(e)
                print(f"[ERROR] {item_id} failed with exception: {error_msg}")
                traceback.print_exc()

                results["details"].append({
                    "input_mode": "text",
                    "text_id": item_id,
                    "output_dir": output_dir,
                    "success": False,
                    "error": error_msg,
                    "stages_completed": [],
                    "stages_failed": ["exception"],
                    "total_time": 0
                })
    
    # 打印汇总结果
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total images: {results['total']}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['success']/results['total']*100:.1f}%")
    print("="*60)
    
    # 保存批量处理结果
    batch_result_path = os.path.join(base_output_dir, "batch_results.json")
    with open(batch_result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Batch results saved to: {batch_result_path}")
    
    return 0 if results["failed"] == 0 else 1


async def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Pipeline for 3D Furniture Generation")
    
    # 创建子命令
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # 单图处理命令
    single_parser = subparsers.add_parser("single", help="Process a single image")
    single_parser.add_argument("image", nargs="?", default=None, help="Input image path")
    single_parser.add_argument("--text-jsonl", help="JSONL file for text input（与 image 二选一）")
    single_parser.add_argument("--output", "-o", help="Output directory")
    single_parser.add_argument("--config", "-c", help="Config file path")
    single_parser.add_argument("--port", type=int, default=None, help="Blender MCP port (overrides config)")
    single_parser.add_argument("--mock", action="store_true", help="Use mock mode (no Blender required)")
    single_parser.add_argument("--skip", nargs="*", default=[], 
                              help="Stages to skip: planning, plan_check, construction")
    
    # 批量处理命令
    batch_parser = subparsers.add_parser("batch", help="Process all images in a directory")
    batch_parser.add_argument("--image-dir", "-i", help="Input images directory（与 --text-jsonl 二选一）")
    batch_parser.add_argument("--text-jsonl", help="Input text JSONL file（与 --image-dir 二选一）")
    batch_parser.add_argument("--output-dir", "-o", required=True, help="Base output directory")
    batch_parser.add_argument("--config", "-c", help="Config file path")
    batch_parser.add_argument("--port", type=int, default=None, help="Blender MCP port (overrides config)")
    batch_parser.add_argument("--mock", action="store_true", help="Use mock mode (no Blender required)")
    batch_parser.add_argument("--skip", nargs="*", default=[], 
                             help="Stages to skip: planning, plan_check, construction")
    
    # 为了向后兼容，如果直接提供输入路径，使用单图处理模式
    parser.add_argument("legacy_image", nargs="?", help="Input image path (for backward compatibility)")
    parser.add_argument("--text-jsonl", help="JSONL file for text input（与 image 二选一）")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--port", type=int, default=None, help="Blender MCP port (overrides config)")
    parser.add_argument("--mock", action="store_true", help="Use mock mode (no Blender required)")
    parser.add_argument("--skip", nargs="*", default=[],
                       help="Stages to skip: planning, plan_check, construction")
    
    args = parser.parse_args()
    
    if args.command == "batch":
        return await run_batch(args)
    elif args.command == "single":
        return await run_single(args)
    elif args.legacy_image:
        if args.text_jsonl:
            print("[ERROR] legacy mode requires exactly one of <legacy_image> or --text-jsonl")
            return 1
        args.image = args.legacy_image
        return await run_single(args)
    elif args.text_jsonl:
        args.image = None
        return await run_single(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
