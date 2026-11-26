"""
Part Constructor Agent
根据 PartPlan 与输入参考（图片或文本）生成 Blender Python 代码并执行，仅负责生成与执行
"""

import os
import base64
import json
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime

from .base_agent import BaseAgent
from ..schemas import (
    ObjectPlan, PartPlan, ActionTrace, BlenderAction,
)
from ..utils.code_naming import get_parts_package_name, get_object_builder_func_name


class PartConstructorAgent(BaseAgent):
    """
    Part Constructor Agent
    
    功能：
    - 根据 PartPlan 生成 Blender Python 代码
    - 执行代码创建部件
    - 保存操作轨迹
    """
    MOVABLE_KEYWORDS = (
        "drawer", "drawers", "door", "doors", "sliding_door", "slide_door", "sliding"
    )
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        llm_client: Optional[Any] = None,
        blender_client: Optional[Any] = None,
    ):
        super().__init__(config_path, llm_client, blender_client)
        prompts_cfg = self.config.get("prompts", {})
        self.prompt_name = prompts_cfg.get("constructor", "constructor")
        self.prompt_template = self._load_prompt(self.prompt_name)
        self.logger.info("Using constructor prompt: %s", self.prompt_name)
                
        # 从配置获取参数
        constructor_config = self.config.get("agents", {}).get("part_constructor", {})
        self.max_retry = constructor_config.get("max_retry", 5)
        self.save_action_trace = constructor_config.get("save_action_trace", True)
        
    def _encode_image(self, image_path: str) -> str:
        """将图片编码为 base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_image_url(self, image_path: str) -> str:
        """获取图片的 data URL"""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        mime_type = mime_types.get(ext, "image/png")
        base64_image = self._encode_image(image_path)
        return f"data:{mime_type};base64,{base64_image}"

    def _is_movable_part(self, part: PartPlan) -> bool:
        """判断部件是否可动，兼容旧 plan（无字段时按关键词推断）。"""
        if getattr(part, "is_movable", False) or getattr(part, "must_be_independent", False):
            return True
        name = str(getattr(part, "name", "")).lower()
        part_type = str(getattr(part, "part_type", "")).lower()
        combined = f"{name} {part_type}"
        return any(k in combined for k in self.MOVABLE_KEYWORDS)

    def _has_instances(self, part: PartPlan) -> bool:
        """判断部件是否使用 instances 模式。"""
        return bool(getattr(part, "instances", []))

    def _build_generation_constraints(self, part: PartPlan) -> str:
        """按部件语义生成补充约束，减少可动件连体错误。"""
        if not self._is_movable_part(part):
            return ""

        if self._has_instances(part):
            instances_json = json.dumps(part.instances, indent=2, ensure_ascii=False)
            func_name = f"create_{part.name}_instances"
            return (
                f"Additional hard constraints (movable part instance group):\n"
                f"1. This part has multiple fully identical instances and must use `instances` mode.\n"
                f"2. Function name must be {func_name}(), returning dict {{instance_name: obj}}.\n"
                f"3. First build one prototype object at the first instance position, including all sub-structures (front_panel/walls/bottom_panel, etc.), then merge into one object and set origin.\n"
                f"4. Create the remaining copies via obj.copy() + obj.data.copy() (independent meshes). Materials may be shared.\n"
                f"5. Each copy must have unique `name` and `data.name`, and use its own position.\n"
                f"6. ARRAY/MIRROR modifiers are forbidden.\n"
                f"\ninstances list:\n```json\n{instances_json}\n```\n"
            )

        return (
            "Additional hard constraints (movable part):\n"
            "1. This part must be an independently movable object. Do not generate instances with ARRAY/MIRROR modifiers.\n"
            "2. If this part is made of front_panel/left_wall/right_wall/back_wall/bottom_panel, build sub-structures first, then merge into one final object.\n"
            "3. Function should return a single object, not an object list.\n"
        )

    def _validate_generated_code_for_part(self, part: PartPlan, code: str) -> Optional[str]:
        """
        对生成代码做快速语义守卫，防止可动件被 Array/Mirror 或列表返回破坏独立性。
        返回 None 表示通过，否则返回错误信息。
        """
        if not self._is_movable_part(part):
            return None

        import re
        if re.search(r"modifiers\.new\([^)]*['\"]ARRAY['\"]", code, flags=re.IGNORECASE):
            return (
                f"Part {part.name} is movable, but the generated code uses ARRAY (forbidden). "
                "Create independent object instances instead of array duplication."
            )
        if re.search(r"modifiers\.new\([^)]*['\"]MIRROR['\"]", code, flags=re.IGNORECASE):
            return (
                f"Part {part.name} is movable, but the generated code uses MIRROR (forbidden). "
                "Create explicit independent objects instead of mirrored movable parts."
            )
        # instances 模式返回 dict 是合法的，仅对非 instances 的单件检查 return all_objects
        if not self._has_instances(part):
            if re.search(r"return\s+all_objects\b", code, flags=re.IGNORECASE):
                return (
                    f"Part {part.name} is movable, but the function returns an object list. "
                    "Merge drawer sub-structures into one final object and return that single object."
                )
        return None
    
    async def _generate_code_with_llm(
        self,
        part: PartPlan,
        image_path: Optional[str] = None,
        text_input: Optional[str] = None,
        context: str = ""
    ) -> str:
        """使用 LLM 生成代码"""
        if self.llm_client is None:
            # 没有 LLM，抛出异常
            raise ValueError("LLM client is not available. Cannot generate code.")

        input_mode = self._validate_single_input(image_path=image_path, text_input=text_input)
        messages = [
            {
                "role": "system",
                "content": self.prompt_template
            },
        ]

        if input_mode == "image":
            user_content: Any = [
                {"type": "image_url", "image_url": {"url": self._get_image_url(image_path)}},
                {
                    "type": "text",
                    "text": f"""Generate Blender Python code from the following part plan (reference input: image):

PartPlan:
```json
{json.dumps(part.to_dict(), indent=2, ensure_ascii=False)}
```

{context}

Generate complete executable Python code to create this part in Blender.
Code requirements:
1. Follow coding conventions.
2. Include necessary comments.
3. Handle possible runtime issues.
4. Set positions and rotations correctly.
5. Create and assign material/texture according to `PartPlan.material`.

Return code only, without extra explanation."""
                }
            ]
        else:
            user_content = [
                {
                    "type": "text",
                    "text": f"""Generate Blender Python code from the following part plan (reference input: text description):

Original text requirement:
{text_input}

PartPlan:
```json
{json.dumps(part.to_dict(), indent=2, ensure_ascii=False)}
```

{context}

Generate complete executable Python code to create this part in Blender.
Code requirements:
1. Follow coding conventions.
2. Include necessary comments.
3. Handle possible runtime issues.
4. Set positions and rotations correctly.
5. Create and assign material/texture according to `PartPlan.material`.

Return code only, without extra explanation."""
                }
            ]
        messages.append({"role": "user", "content": user_content})
        
        response = await self._call_llm(messages)
        
        # 提取代码块
        import re
        code_pattern = r'```(?:python)?\s*([\s\S]*?)\s*```'
        matches = re.findall(code_pattern, response)
        
        if matches:
            return matches[0]
        else:
            # 如果没有代码块，假设整个响应就是代码
            return response
    def _fix_main_block(self, code: str) -> str:
        """将 if __name__ == '__main__' 修改为 if True"""
        import re
        
        # 匹配 if __name__ == "__main__": 或 if __name__ == '__main__':
        pattern = r'if\s+__name__\s*==\s*["\']__main__["\']\s*:'
        
        # 替换为 if True:
        fixed_code = re.sub(pattern, 'if True:', code)
        
        return fixed_code

    async def _execute_code(self, code: str) -> Dict[str, Any]:
        """在 Blender 中执行代码"""
        if self.blender_client is None:
            self.logger.warning("Blender client not available, skipping execution")
            return {"success": False, "error": "Blender client not available"}

        # 修复代码：移除或替换 if __name__ == "__main__" 块
        code = self._fix_main_block(code)

        try:
            result = await self._call_blender("execute_code", {"code": code})
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _generate_code_fix_from_error(
        self,
        part: PartPlan,
        code: str,
        error: str,
        image_path: Optional[str] = None,
        text_input: Optional[str] = None,
    ) -> str:
        """
        根据 Blender 执行错误信息，使用 LLM 修正代码。
        返回修正后的完整代码。
        """
        if self.llm_client is None:
            raise ValueError("LLM client is not available. Cannot fix code from error.")

        input_mode = self._validate_single_input(image_path=image_path, text_input=text_input)

        context = f"""The following code failed in Blender, together with Blender's error message.
Fix the code according to the error (for example API usage, undefined variables, indentation, bpy calls), and output the **full** corrected executable Python code.
Do not output only a patch snippet. You must output the complete corrected code.

Current code:
```python
{code}
```

Execution error:
{error}

Return only the corrected full Python code (you may wrap it in ```python ... ```)."""

        if self._is_movable_part(part):
            if self._has_instances(part):
                instances_json = json.dumps(part.instances, indent=2, ensure_ascii=False)
                context += f"""

Additional hard constraints (movable part instance group):
1. Do not use ARRAY/MIRROR modifiers for movable parts.
2. Function name must be create_{part.name}_instances(), returning dict {{instance_name: obj}}.
3. Build a prototype at the first instance position, merge sub-structures into one object, and set origin.
4. Create remaining copies with obj.copy() + obj.data.copy(), and assign unique names and positions.

instances list:
```json
{instances_json}
```"""
            else:
                context += """

Additional hard constraints (movable part):
1. Do not use ARRAY/MIRROR modifiers for movable parts.
2. Keep each movable instance as an independent object, never a single array-combined object.
3. If front/walls/bottom sub-structures exist, merge them into one final object and return it."""

        if input_mode == "text":
            context = f"""Original text requirement:
{text_input}

{context}"""

        messages = [
            {"role": "system", "content": self.prompt_template},
            {"role": "user", "content": context},
        ]

        if input_mode == "image" and image_path and os.path.isfile(image_path):
            image_url = self._get_image_url(image_path)
            messages[-1]["content"] = [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": context},
            ]

        response = await self._call_llm(messages)
        import re
        code_pattern = r"```(?:python)?\s*([\s\S]*?)\s*```"
        matches = re.findall(code_pattern, response)
        if matches:
            return matches[0]
        return response

    async def construct_part(
        self,
        part: PartPlan,
        image_path: Optional[str],
        output_dir: str,
        text_input: Optional[str] = None,
        render_output_dir: str = None,
        use_llm: bool = True,
        execute: bool = True,
    ) -> Dict[str, Any]:
        """
        构造单个部件。
        
        Args:
            part: PartPlan 对象
            image_path: 原始图片路径（与 text_input 二选一）
            output_dir: 代码输出目录
            text_input: 原始文本输入（与 image_path 二选一）
            render_output_dir: 渲染输出目录（保留参数兼容，未使用）
            use_llm: 是否使用 LLM 生成代码
            execute: 是否执行代码
            
        Returns:
            构造结果字典
        """
        has_instances = self._has_instances(part)
        if has_instances:
            inst_names = [inst["name"] for inst in part.instances]
            self.logger.info(f"Constructing instance group: {part.name} -> {inst_names}")
        else:
            self.logger.info(f"Constructing part: {part.name}")
        self._validate_single_input(image_path=image_path, text_input=text_input)
        
        # 确保渲染输出目录
        if render_output_dir is None:
            render_output_dir = os.path.join(os.path.dirname(output_dir), "renders", "parts")
        os.makedirs(render_output_dir, exist_ok=True)
        
        # 生成代码
        if use_llm:
            generation_constraints = self._build_generation_constraints(part)
            code = await self._generate_code_with_llm(
                part,
                image_path=image_path,
                text_input=text_input,
                context=generation_constraints,
            )
        else:
            code = self._generate_code_template(part)
        
        # 保存代码
        code_path = os.path.join(output_dir, f"{part.name}.py")
        os.makedirs(output_dir, exist_ok=True)
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        self.logger.info(f"Code saved to: {code_path}")
        
        # 执行代码（失败时根据错误信息用 LLM 修正并重试）
        execution_result = None
        execution_attempts = 0
        max_execution_retries = max(1, self.max_retry)

        if execute:
            while execution_attempts < max_execution_retries:
                execution_attempts += 1
                if execution_attempts > 1:
                    # 根据错误信息用 LLM 修正代码
                    self.logger.info(
                        f"Execution failed, fixing code from error (attempt {execution_attempts}/{max_execution_retries})..."
                    )
                    if self.llm_client is None:
                        self.logger.warning("LLM not available for fix, skipping retry")
                        break
                    try:
                        code = await self._generate_code_fix_from_error(
                            part,
                            code,
                            execution_result.get("error", "Unknown error"),
                            image_path=image_path,
                            text_input=text_input
                        )
                        with open(code_path, "w", encoding="utf-8") as f:
                            f.write(code)
                        self.logger.info(f"Fixed code saved to: {code_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to generate fix from error: {e}")
                        break

                self.logger.info(f"Executing code for {part.name} (attempt {execution_attempts})...")
                code_guard_error = self._validate_generated_code_for_part(part, code)
                if code_guard_error:
                    execution_result = {"success": False, "error": code_guard_error}
                    self.logger.error(
                        f"Attempt {execution_attempts} blocked by code guard: {code_guard_error}"
                    )
                    continue
                execution_result = await self._execute_code(code)
                if execution_result["success"]:
                    self.logger.info(f"Part {part.name} created successfully")
                    break
                self.logger.error(
                    f"Attempt {execution_attempts} failed: {execution_result.get('error')}"
                )

            if not execution_result["success"]:
                raise RuntimeError(
                    f"Failed to create part {part.name} after {execution_attempts} attempt(s): {execution_result.get('error')}"
                )

        # 记录操作轨迹
        actions = [
            BlenderAction(
                action_type="generate_code",
                parameters={"part_plan": part.to_dict()},
                result="success",
                timestamp=datetime.now().isoformat()
            ),
            BlenderAction(
                action_type="execute_code",
                parameters={"code_path": code_path},
                result="success" if (execution_result and execution_result["success"]) else "failed",
                timestamp=datetime.now().isoformat()
            )
        ]
        
        action_trace = ActionTrace(
            part_name=part.name,
            actions=actions,
            final_code=code,
            iterations=1
        )
        
        # 保存操作轨迹
        if self.save_action_trace:
            trace_path = os.path.join(output_dir, f"{part.name}_trace.json")
            action_trace.save(trace_path)
        
        result = {
            "part_name": part.name,
            "code_path": code_path,
            "code": code,
            "execution_result": execution_result,
            "action_trace": action_trace,
        }
        if has_instances:
            result["instance_names"] = [inst["name"] for inst in part.instances]
        return result
    
    async def run(
        self,
        object_plan: ObjectPlan,
        image_path: Optional[str],
        output_dir: str,
        text_input: Optional[str] = None,
        render_output_dir: str = None,
        use_llm: bool = True,
        execute: bool = True,
        construct_parts: bool = True
    ) -> Dict[str, Any]:
        """
        构造所有部件（仅生成与执行，不做 part check）。
        当 construct_parts=False 时跳过构造，仅从已有部件代码组装 results 并生成主文件。
        
        Args:
            object_plan: ObjectPlan 对象
            image_path: 原始图片路径（与 text_input 二选一）
            output_dir: 输出目录
            text_input: 原始文本输入（与 image_path 二选一）
            render_output_dir: 渲染输出目录
            use_llm: 是否使用 LLM 生成代码
            execute: 是否执行代码
            construct_parts: 是否执行部件构造（False 时仅加载已有代码并生成主文件）
            
        Returns:
            构造结果字典
        """
        self.logger.info(f"Starting construction for: {object_plan.name}")
        self._validate_single_input(image_path=image_path, text_input=text_input)
        
        # 创建输出目录（使用合法包名，避免 import 语法错误）
        parts_package_name = get_parts_package_name(object_plan.name)
        parts_dir = os.path.join(output_dir, parts_package_name)
        os.makedirs(parts_dir, exist_ok=True)
        
        # 渲染输出目录
        if render_output_dir is None:
            render_output_dir = os.path.join(os.path.dirname(output_dir), "renders", "parts")
        os.makedirs(render_output_dir, exist_ok=True)
        
        # 按优先级排序部件
        sorted_parts = object_plan.get_parts_sorted_by_priority()
        
        results = []
        failed_parts = []
        
        total_parts = len(sorted_parts)
        
        if construct_parts:
            for idx, part in enumerate(sorted_parts):
                self.logger.info(f"\n--- Part {idx + 1}/{total_parts}: {part.name} ---")
                
                code_path = os.path.join(parts_dir, f"{part.name}.py")
                if os.path.exists(code_path):
                    self.logger.info(f"Part code already exists, skipping: {code_path}")
                    try:
                        with open(code_path, "r", encoding="utf-8") as f:
                            code = f.read()
                        results.append({
                            "part_name": part.name,
                            "code_path": code_path,
                            "code": code,
                            "execution_result": {"success": True, "skipped": True},
                            "action_trace": None,
                            "error": None
                        })
                    except Exception as e:
                        self.logger.error(f"Error loading existing code for {part.name}: {e}")
                        failed_parts.append(part.name)
                        results.append({"part_name": part.name, "error": str(e)})
                    continue
                
                try:
                    result = await self.construct_part(
                        part=part,
                        image_path=image_path,
                        text_input=text_input,
                        output_dir=parts_dir,
                        render_output_dir=render_output_dir,
                        use_llm=use_llm,
                        execute=execute,
                    )
                    results.append(result)
                    
                    if execute and result.get("execution_result") and not result["execution_result"]["success"]:
                        failed_parts.append(part.name)
                        
                except Exception as e:
                    self.logger.error(f"Error constructing {part.name}: {e}")
                    failed_parts.append(part.name)
                    results.append({
                        "part_name": part.name,
                        "error": str(e)
                    })
        else:
            # 跳过构造：从已有代码文件组装 results，仅生成主文件
            self.logger.info(f"Skipping construction (construct_parts=False), loading existing code for {total_parts} parts")
            for idx, part in enumerate(sorted_parts):
                code_path = os.path.join(parts_dir, f"{part.name}.py")
                if not os.path.exists(code_path):
                    self.logger.warning(f"Part code not found: {code_path}")
                    failed_parts.append(part.name)
                    results.append({"part_name": part.name, "code_path": code_path, "error": "code file not found"})
                    continue
                try:
                    with open(code_path, "r", encoding="utf-8") as f:
                        code = f.read()
                except Exception as e:
                    failed_parts.append(part.name)
                    results.append({"part_name": part.name, "code_path": code_path, "error": str(e)})
                    continue
                results.append({
                    "part_name": part.name,
                    "code_path": code_path,
                    "code": code,
                    "execution_result": {"success": True, "skipped": True},
                    "action_trace": None,
                    "error": None
                })
        
        # 生成主文件（组合所有部件）
        main_code = self._generate_main_file(object_plan, parts_package_name)
        main_path = os.path.join(output_dir, f"{object_plan.name}.py")
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(main_code)
        
        construct_success = len(failed_parts) == 0
        
        self.logger.info("\n=== Construction Summary ===")
        self.logger.info(f"Total parts: {total_parts}")
        self.logger.info(f"Construction success: {total_parts - len(failed_parts)}/{total_parts}")
        if failed_parts:
            self.logger.warning(f"Failed parts: {failed_parts}")
        
        return {
            "object_name": object_plan.name,
            "output_dir": output_dir,
            "parts_dir": parts_dir,
            "render_output_dir": render_output_dir,
            "main_file": main_path,
            "part_results": results,
            "failed_parts": failed_parts,
            "construct_success": construct_success,
            "success": construct_success
        }
    
    def _generate_main_file(self, object_plan: ObjectPlan, parts_package_name: str) -> str:
        """生成主文件，组合所有部件"""
        parts = object_plan.get_parts_sorted_by_priority()
        builder_func_name = get_object_builder_func_name(object_plan.name)
        
        import_lines = []
        create_lines = []
        for p in parts:
            if getattr(p, "instances", []):
                import_lines.append(
                    f"from {parts_package_name}.{p.name} import create_{p.name}_instances"
                )
                create_lines.append(
                    f'parts.update(create_{p.name}_instances())'
                )
            else:
                import_lines.append(
                    f"from {parts_package_name}.{p.name} import create_{p.name}"
                )
                create_lines.append(
                    f'parts["{p.name}"] = create_{p.name}()'
                )
        
        imports = "\n".join(import_lines)
        creates = "\n    ".join(create_lines)
        
        code = f'''# {object_plan.name}
# {object_plan.description}
# Generated by Part Constructor Agent
# Category: {object_plan.category}
# Total parts: {len(parts)}

import bpy
import sys
import os

# 添加部件目录到路径
sys.path.insert(0, os.path.dirname(__file__))

{imports}


def {builder_func_name}():
    """
    创建完整的 {object_plan.category}
    
    Dimensions: {object_plan.total_dimensions.width} x {object_plan.total_dimensions.depth} x {object_plan.total_dimensions.height}
    Style: {object_plan.style}
    """
    
    parts = {{}}
    
    # 按优先级创建部件
    {creates}
    
    return parts


def clear_scene():
    """清空场景"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


if __name__ == "__main__":
    # 可选：清空场景
    # clear_scene()
    
    # 创建物体
    parts = {builder_func_name}()
    print(f"Created {{len(parts)}} parts for {object_plan.name}")
'''
        return code


class MockPartConstructorAgent(PartConstructorAgent):
    """用于测试的 Mock Part Constructor"""
    
    async def _call_llm(self, messages: list, **kwargs) -> str:
        """返回模板代码"""
        return "```python\n# Mock code\nimport bpy\nprint('Mock')\n```"
    
    async def _execute_code(self, code: str) -> Dict[str, Any]:
        """模拟执行"""
        return {"success": True, "result": "Mock execution"}
    
    async def construct_part(
        self,
        part: PartPlan,
        image_path: Optional[str],
        output_dir: str,
        text_input: Optional[str] = None,
        render_output_dir: str = None,
        use_llm: bool = True,
        execute: bool = True,
    ) -> Dict[str, Any]:
        """Mock 版本：跳过实际的 LLM 调用"""
        self.logger.info(f"[Mock] Constructing part: {part.name}")
        
        # 保存 Mock 代码
        code = f"# Mock code for {part.name}\nimport bpy\nprint('Mock {part.name}')\n"
        code_path = os.path.join(output_dir, f"{part.name}.py")
        os.makedirs(output_dir, exist_ok=True)
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        return {
            "part_name": part.name,
            "code_path": code_path,
            "code": code,
            "execution_result": {"success": True, "result": "Mock execution"},
            "action_trace": None,
        }
