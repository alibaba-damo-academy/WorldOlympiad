"""
Planner Agent
从输入图片或文本分析物体，生成 ObjectPlan
"""

import os
import json
import base64
from typing import Any, Dict, Optional, List
from pathlib import Path

from .base_agent import BaseAgent
from ..schemas import ObjectPlan


class PlannerAgent(BaseAgent):
    """
    Planner Agent
    
    功能：
    - 分析输入图片或文本，识别物体类别
    - 拆解部件，确定每个部件的形状、尺寸、位置
    - 生成 ObjectPlan
    """
    MOVABLE_KEYWORDS = (
        "drawer", "drawers", "door", "doors", "sliding_door", "slide_door", "sliding"
    )
    FORBIDDEN_MODIFIERS_FOR_MOVABLE = ("array", "mirror")
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        llm_client: Optional[Any] = None,
        blender_client: Optional[Any] = None
    ):
        super().__init__(config_path, llm_client, blender_client)
        prompts_cfg = self.config.get("prompts", {})
        self.prompt_name = prompts_cfg.get("planner", "planner")
        self.prompt_template = self._load_prompt(self.prompt_name)
        self.logger.info("Using planner prompt: %s", self.prompt_name)
        
        # 从配置获取参数
        planner_config = self.config.get("agents", {}).get("planner", {})
        self.max_parts = planner_config.get("max_parts", 20)
        self.min_parts = planner_config.get("min_parts", 2)
        
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
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(ext, "image/png")
        base64_image = self._encode_image(image_path)
        return f"data:{mime_type};base64,{base64_image}"
    
    def _build_messages(
        self,
        image_path: Optional[str] = None,
        text_input: Optional[str] = None,
        additional_context: str = ""
    ) -> List[Dict]:
        """构建 LLM 消息"""
        input_mode = self._validate_single_input(image_path=image_path, text_input=text_input)
        
        system_message = {
            "role": "system",
            "content": self.prompt_template
        }

        common_requirements = f"""Requirements:
1. Identify the object category (chair/table/stool/etc.).
2. Decompose all visible or expected parts.
3. Estimate shape and dimensions for each part.
4. Define positional and hierarchical relationships between parts.
5. Assign reasonable build priority.
6. Provide `material` information for each part.
7. For movable parts such as drawers/doors/sliding doors, set `is_movable=true` and `must_be_independent=true`.
8. Movable parts must not use array/mirror to represent multiple instances.
9. If movable parts are fully identical (same shape/material/sub_parts, only position differs), merge into one PartPlan using `instances` with each instance name and position.
10. If a movable part has unique decoration or dimension differences, keep it as a separate PartPlan.
11. Fixed repeated structures may use array/mirror.

Output ObjectPlan strictly in JSON format.

{additional_context}"""

        if input_mode == "image":
            image_url = self._get_image_url(image_path)
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                },
                {
                    "type": "text",
                    "text": f"""Analyze this furniture image and generate a detailed ObjectPlan.

{common_requirements}"""
                }
            ]
        else:
            user_content = [
                {
                    "type": "text",
                    "text": f"""Generate a detailed ObjectPlan from the following furniture text description.

Text description:
{text_input}

{common_requirements}"""
                }
            ]
        
        user_message = {
            "role": "user",
            "content": user_content
        }
        
        return [system_message, user_message]
    
    def _validate_plan(self, plan_dict: Dict[str, Any]) -> List[str]:
        """
        验证 plan 的基本有效性
        
        Returns:
            错误列表，空列表表示验证通过
        """
        errors = []
        
        # 检查必需字段
        required_fields = ["category", "parts"]
        for field in required_fields:
            if field not in plan_dict:
                errors.append(f"Missing required field: {field}")
        
        # # 检查类别
        # if "category" in plan_dict:
        #     try:
        #         ObjectCategory(plan_dict["category"])
        #     except ValueError:
        #         errors.append(f"Invalid category: {plan_dict['category']}")
        
        # 检查部件
        parts = plan_dict.get("parts", [])
        # if len(parts) < self.min_parts:
        #     errors.append(f"Too few parts: {len(parts)} < {self.min_parts}")
        # if len(parts) > self.max_parts:
        #     errors.append(f"Too many parts: {len(parts)} > {self.max_parts}")
        
        # 检查部件名称唯一性
        part_names = [p.get("name") for p in parts]
        if len(part_names) != len(set(part_names)):
            errors.append("Duplicate part names detected")
        
        return errors
    
    def _post_process_plan(self, plan_dict: Dict[str, Any]) -> Dict[str, Any]:
        """后处理 plan，修复常见问题"""
        material_hints = plan_dict.get("material_hints", {})
        
        # 确保所有部件都有必需字段
        for part in plan_dict.get("parts", []):
            # 设置默认 position
            if "position" not in part:
                part["position"] = {"x": 0, "y": 0, "z": 0}
            
            # 设置默认 rotation
            if "rotation" not in part:
                part["rotation"] = {"x": 0, "y": 0, "z": 0}
            
            # 检查是否有 shape 字段
            if "shape" not in part:
                raise ValueError(f"Missing required field 'shape' in part: {part.get('name', 'unknown')}")
            
            # 设置默认 priority
            if "priority" not in part:
                part["priority"] = 1

            # 推断并修正可动件语义字段
            self._normalize_movable_flags(part)

            # 设置默认 material/texture
            if "material" not in part:
                hint = material_hints.get(part.get("name")) or material_hints.get(part.get("part_type"))
                if hint:
                    part["material"] = {"type": hint, "texture": hint}
                else:
                    part["material"] = {
                        "type": "generic",
                        "base_color": [0.8, 0.8, 0.8],
                        "roughness": 0.5,
                        "metallic": 0.0,
                        "texture": "none"
                    }

            # 子部件默认继承静态属性；若业务后续需要可动子部件，可由上游显式标注
            for sp in part.get("sub_parts", []):
                sp.setdefault("is_movable", False)
                sp.setdefault("must_be_independent", False)
        
        # 确保有 total_dimensions
        if "total_dimensions" not in plan_dict:
            # 从部件估算整体尺寸
            max_x = max_y = max_z = 0
            for part in plan_dict.get("parts", []):
                pos = part.get("position", {})
                dim = part.get("shape", {}).get("dimensions", {})
                max_x = max(max_x, abs(pos.get("x", 0)) + dim.get("width", 0) / 2)
                max_y = max(max_y, abs(pos.get("y", 0)) + dim.get("depth", 0) / 2)
                max_z = max(max_z, pos.get("z", 0) + dim.get("height", 0) / 2)
            
            plan_dict["total_dimensions"] = {
                "width": max_x * 2,
                "depth": max_y * 2,
                "height": max_z
            }
        
        return plan_dict

    def _normalize_movable_flags(self, part: Dict[str, Any]) -> None:
        """标准化可动件字段，并移除可动件上的禁用修饰器。"""
        def _as_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y"}
            return bool(value)

        name = str(part.get("name", "")).lower()
        part_type = str(part.get("part_type", "")).lower()
        combined = f"{name} {part_type}"
        inferred_movable = any(k in combined for k in self.MOVABLE_KEYWORDS)

        is_movable = part.get("is_movable")
        if is_movable is None:
            is_movable = inferred_movable
        else:
            is_movable = _as_bool(is_movable)
        part["is_movable"] = is_movable

        must_be_independent = part.get("must_be_independent")
        if must_be_independent is None:
            must_be_independent = is_movable
        else:
            must_be_independent = _as_bool(must_be_independent)
        if is_movable:
            must_be_independent = True
        part["must_be_independent"] = must_be_independent

        # 校验 instances 字段（确保每个 instance 有 name 和 position）
        instances = part.get("instances", [])
        if instances:
            for inst in instances:
                if "name" not in inst or "position" not in inst:
                    self.logger.warning(
                        "Instance in part '%s' missing 'name' or 'position': %s",
                        part.get("name", "unknown"), inst,
                    )

        # 可动件禁止 array/mirror（固定件仍可用）
        if is_movable:
            shape = part.get("shape", {})
            modifiers = shape.get("modifiers", [])
            if isinstance(modifiers, list):
                filtered = []
                removed = []
                for mod in modifiers:
                    mod_str = str(mod)
                    mod_lower = mod_str.lower()
                    if any(key in mod_lower for key in self.FORBIDDEN_MODIFIERS_FOR_MOVABLE):
                        removed.append(mod_str)
                    else:
                        filtered.append(mod)
                if removed:
                    self.logger.warning(
                        "Removed forbidden modifiers from movable part '%s': %s",
                        part.get("name", "unknown"),
                        removed,
                    )
                shape["modifiers"] = filtered
    
    async def run(
        self,
        image_path: Optional[str] = None,
        output_path: Optional[str] = None,
        additional_context: str = "",
        text_input: Optional[str] = None
    ) -> ObjectPlan:
        """
        执行规划
        
        Args:
            image_path: 输入图片路径（与 text_input 二选一）
            output_path: 输出 JSON 路径（可选）
            additional_context: 额外的上下文信息
            text_input: 输入文本描述（与 image_path 二选一）
            
        Returns:
            ObjectPlan 对象
        """
        input_mode = self._validate_single_input(image_path=image_path, text_input=text_input)
        self.logger.info(f"Starting planning with {input_mode} input")

        if input_mode == "image":
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
        
        # 构建消息
        messages = self._build_messages(
            image_path=image_path,
            text_input=text_input,
            additional_context=additional_context
        )
        
        # 调用 LLM
        self.logger.info("Calling LLM for object analysis...")
        response = await self._call_llm(messages)
        
        # 解析响应
        self.logger.info("Parsing LLM response...")
        plan_dict = self._parse_json_response(response)
        
        # 优先用输入来源生成稳定 name
        if input_mode == "image":
            object_name = os.path.basename(image_path).split(".")[0]
        else:
            object_name = self._extract_object_name_from_text(text_input)
            if object_name == "object":
                # 文本无法提取简名时，回退到类别字段（如 chair/table）
                object_name = self._sanitize_object_name(plan_dict.get("category", "object"), fallback="object")
        plan_dict["name"] = self._sanitize_object_name(object_name, fallback="object")

        # 验证 plan 的基本有效性
        errors = self._validate_plan(plan_dict)
        if errors:
            self.logger.warning(f"Plan validation warnings: {errors}")
            # 可以选择抛出异常或尝试修复
        
        # 后处理
        plan_dict = self._post_process_plan(plan_dict)
        
        # 转换为 ObjectPlan 对象
        object_plan = ObjectPlan.from_dict(plan_dict)
        
        # 保存结果
        if output_path:
            self.logger.info(f"Saving plan to: {output_path}")
            object_plan.save(output_path)
        
        self.logger.info(f"Planning completed. Found {len(object_plan.parts)} parts.")
        return object_plan
    
    async def run_with_retry(
        self,
        image_path: Optional[str] = None,
        output_path: Optional[str] = None,
        max_retry: int = 3,
        text_input: Optional[str] = None,
        additional_context: str = ""
    ) -> ObjectPlan:
        """
        带重试的规划执行
        
        Args:
            image_path: 输入图片路径（与 text_input 二选一）
            output_path: 输出 JSON 路径
            max_retry: 最大重试次数
            text_input: 输入文本描述（与 image_path 二选一）
            additional_context: 额外的上下文信息
            
        Returns:
            ObjectPlan 对象
        """
        last_error = None
        
        for attempt in range(max_retry):
            try:
                self.logger.info(f"Planning attempt {attempt + 1}/{max_retry}")
                return await self.run(
                    image_path=image_path,
                    output_path=output_path,
                    additional_context=additional_context,
                    text_input=text_input
                )
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retry - 1:
                    self.logger.info("Retrying with additional guidance...")
        
        raise RuntimeError(f"Planning failed after {max_retry} attempts. Last error: {last_error}")

    def _build_revision_messages(
        self,
        original_plan: ObjectPlan,
        correction_context: str,
        image_path: Optional[str] = None,
        text_input: Optional[str] = None,
    ) -> List[Dict]:
        """
        构建 LLM 消息用于在已有 ObjectPlan 基础上进行修正。
        
        输入：
        - 原始 ObjectPlan（JSON）
        - correction_context：来自 PlannerChecker 的问题与修改建议
        - 参考图片或参考文本（二选一，可选）
        
        要求：
        - 以原始 ObjectPlan 为基础做「最小必要修改」
        - 保留已正确的结构，仅修正存在问题的部分
        - 输出完整、可直接使用的 ObjectPlan JSON
        """
        # 允许不提供 image/text，只基于 ObjectPlan+correction_context 修正
        input_mode = None
        if image_path or text_input:
            input_mode = self._validate_single_input(image_path=image_path, text_input=text_input)

        system_message = {
            "role": "system",
            "content": '''**Revision mode (ObjectPlan -> revised ObjectPlan)**:
- When you receive:
    - an existing `ObjectPlan` JSON,
    - `correction_context` (issues and fix suggestions from Planner Checker),
    - and optional reference image or reference text,
- you are in **revision mode** and must follow these rules:
    - **Do not** fully re-plan from scratch based on image/text.
    - Use the current `ObjectPlan` as the baseline and apply only necessary changes.
    - Keep already-correct structures, fields, and values.
    - Only fix the fields or parts explicitly identified as problematic (for example missing parts, wrong dimensions, unreasonable is_movable/instances).
    - Output a **complete and self-consistent** `ObjectPlan` JSON that can be used directly by downstream construction.
- If a field is not mentioned in `correction_context` and is already reasonable, do not change it casually.'''}

        plan_json = json.dumps(original_plan.to_dict(), indent=2, ensure_ascii=False)

        if input_mode == "image":
            image_url = self._get_image_url(image_path)
            user_content: List[Dict[str, Any]] = [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {
                    "type": "text",
                    "text": f"""You are now in **revision mode** (ObjectPlan -> revised ObjectPlan).

Apply minimal necessary changes to the existing ObjectPlan based on:
1. Original reference image
2. Current ObjectPlan (JSON)
3. `correction_context` from the checking stage (issues and fix suggestions)

Strict output requirements:
- Use the current ObjectPlan as baseline; modify only problematic fields/parts.
- Preserve already-correct structures and fields.
- Output a complete and self-consistent ObjectPlan JSON.

Current ObjectPlan:
```json
{plan_json}
```

correction_context:
{correction_context}

Output ObjectPlan strictly in JSON format.
"""
                },
            ]
        elif input_mode == "text":
            user_content = [
                {
                    "type": "text",
                    "text": f"""You are now in **revision mode** (ObjectPlan -> revised ObjectPlan). Do not fully re-plan from the text description.

Original text description:
{text_input}

Apply minimal necessary changes to the existing ObjectPlan based on:
1. Original text description
2. Current ObjectPlan (JSON)
3. `correction_context` from the checking stage (issues and fix suggestions)

Strict output requirements:
- Use the current ObjectPlan as baseline; modify only problematic fields/parts.
- Preserve already-correct structures and fields.
- Output a complete and self-consistent ObjectPlan JSON.

Current ObjectPlan:
```json
{plan_json}
```

correction_context:
{correction_context}

Output ObjectPlan strictly in JSON format.
"""
                }
            ]
        else:
            # 没有额外参考，只基于 plan + correction_context 修正
            raise ValueError("No reference provided")

        user_message = {
            "role": "user",
            "content": user_content,
        }

        return [system_message, user_message]

    async def revise_plan(
        self,
        original_plan: ObjectPlan,
        correction_context: str,
        image_path: Optional[str] = None,
        text_input: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> ObjectPlan:
        """
        在「已有 ObjectPlan」基础上进行修正，而不是重新从图片/文本完整规划。
        
        Args:
            original_plan: 当前的 ObjectPlan
            correction_context: 来自 PlannerChecker 的问题与修改建议文本
            image_path: 参考图片（可选）
            text_input: 参考文本描述（可选）
            output_path: 修正后 ObjectPlan 的保存路径（可选）
        """
        self.logger.info("Starting plan revision based on existing ObjectPlan and correction_context")

        messages = self._build_revision_messages(
            original_plan=original_plan,
            correction_context=correction_context,
            image_path=image_path,
            text_input=text_input,
        )

        # 调用 LLM 进行修正
        self.logger.info("Calling LLM for plan revision...")
        response = await self._call_llm(messages)

        # 解析响应
        self.logger.info("Parsing LLM revision response...")
        plan_dict = self._parse_json_response(response)

        # 如果未显式给出 name，则继承原 plan 的 name
        if "name" not in plan_dict or not plan_dict.get("name"):
            plan_dict["name"] = original_plan.name

        # 若未给出 category/total_dimensions 等关键字段，可从原 plan 继承
        for key in ("category", "total_dimensions", "style", "material_hints"):
            if key not in plan_dict and hasattr(original_plan, key):
                try:
                    plan_dict[key] = getattr(original_plan, key)
                except Exception:
                    continue

        # 验证修正后 plan
        errors = self._validate_plan(plan_dict)
        if errors:
            self.logger.warning(f"Revised plan validation warnings: {errors}")

        # 后处理并转换为 ObjectPlan
        plan_dict = self._post_process_plan(plan_dict)
        object_plan = ObjectPlan.from_dict(plan_dict)

        if output_path:
            self.logger.info(f"Saving revised plan to: {output_path}")
            object_plan.save(output_path)

        self.logger.info(f"Plan revision completed. Final parts count: {len(object_plan.parts)}")
        return object_plan


# 用于测试的简化版本（不需要 LLM）
class MockPlannerAgent(PlannerAgent):
    """用于测试的 Mock Planner"""
    
    async def _call_llm(self, messages: list, **kwargs) -> str:
        """返回模拟响应"""
        mock_response = {
            "category": "chair",
            "description": "A simple test chair",
            "parts": [
                {
                    "name": "seat",
                    "part_type": "seat",
                    "shape": {
                        "base_shape": "cube",
                        "dimensions": {"width": 0.45, "depth": 0.45, "height": 0.04},
                        "modifiers": ["bevel"],
                        "description": "Square seat"
                    },
                    "position": {"x": 0, "y": 0, "z": 0.45},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                    "is_symmetric": False,
                    "description": "Main seat",
                    "priority": 0,
                    "material": {
                        "type": "wood",
                        "base_color": [0.6, 0.4, 0.25],
                        "roughness": 0.6,
                        "metallic": 0.0,
                        "texture": "wood_grain"
                    }
                },
                {
                    "name": "leg_01",
                    "part_type": "leg",
                    "shape": {
                        "base_shape": "cube",
                        "dimensions": {"width": 0.04, "depth": 0.04, "height": 0.45},
                        "modifiers": [],
                        "description": "Square leg"
                    },
                    "position": {"x": 0.18, "y": 0.18, "z": 0.225},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                    "is_symmetric": True,
                    "symmetric_axis": "x",
                    "description": "Front right leg",
                    "priority": 1,
                    "material": {
                        "type": "wood",
                        "base_color": [0.5, 0.35, 0.2],
                        "roughness": 0.7,
                        "metallic": 0.0,
                        "texture": "wood_grain"
                    }
                }
            ],
            "total_dimensions": {"width": 0.45, "depth": 0.45, "height": 0.9},
            "style": "minimalist",
            "material_hints": {"seat": "wood", "legs": "wood"}
        }
        
        import json
        return f"```json\n{json.dumps(mock_response, indent=2)}\n```"
