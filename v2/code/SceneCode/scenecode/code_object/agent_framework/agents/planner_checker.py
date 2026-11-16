"""
Planner Checker Agent
检查 ObjectPlan 的有效性，提供修正建议
"""

import base64
import re
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass

from .base_agent import BaseAgent
from ..schemas import ObjectPlan


@dataclass
class CheckIssue:
    """检查问题"""
    severity: str  # error, warning, suggestion
    category: str  # 检查类别
    description: str
    suggestion: str
    affected_parts: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "suggestion": self.suggestion,
            "affected_parts": self.affected_parts or []
        }


@dataclass
class CheckResult:
    """检查结果"""
    is_valid: bool
    score: int  # 0-100
    issues: List[CheckIssue]
    corrections: Dict[str, Any]
    summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "score": self.score,
            "issues": [i.to_dict() for i in self.issues],
            "corrections": self.corrections,
            "summary": self.summary
        }


class PlannerCheckerAgent(BaseAgent):
    """
    Planner Checker Agent
    
    功能：
    - 检查 ObjectPlan 的完整性
    - 检查部件命名规范
    - 检查空间关系合理性
    - 检查尺寸比例
    - 提供修正建议
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
        self.prompt_name = prompts_cfg.get("checker", "checker")
        self.prompt_template = self._load_prompt(self.prompt_name)
        self.logger.info("Using planner checker prompt: %s", self.prompt_name)
        
        # 从配置获取参数
        checker_config = self.config.get("agents", {}).get("planner_checker", {})
        self.max_retry = checker_config.get("max_retry", 3)
        
        # 部件模板
        self.part_templates = self.config.get("part_templates", {})

    def _planner_checker_prompt(self) -> str:
        """
        从 checker.md 中提取 Planner Checker 段落。
        若分隔失败，回退到完整模板。
        """
        parts = self.prompt_template.split("---")
        if len(parts) >= 2:
            return parts[1].strip()
        return self.prompt_template
        
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

    
    def _check_parts_completeness(self, plan: ObjectPlan) -> List[CheckIssue]:
        """检查部件完整性"""
        issues = []
        
        # 获取类别对应的部件模板
        category = plan.category
        template = self.part_templates.get(category, {})
        required_types = template.get("required", [])
        
        # 获取所有部件类型
        part_types = set(p.part_type for p in plan.parts)
        
        # 检查必需部件
        for req_type in required_types:
            # 特殊处理 legs（可能是 leg）
            type_variants = [req_type, req_type.rstrip('s'), req_type + 's']
            if not any(t in part_types for t in type_variants):
                issues.append(CheckIssue(
                    severity="error",
                    category="Part completeness",
                    description=f"Missing required part type: {req_type}",
                    suggestion=f"Add part(s) of type `{req_type}`"
                ))
        
        # 检查是否有重复名称
        names = [p.name for p in plan.parts]
        duplicates = set([n for n in names if names.count(n) > 1])
        if duplicates:
            issues.append(CheckIssue(
                severity="error",
                category="Part completeness",
                description=f"Duplicate part names found: {duplicates}",
                suggestion="Ensure each part has a unique name",
                affected_parts=list(duplicates)
            ))
        
        return issues
    
    # def _check_naming_convention(self, plan: ObjectPlan) -> List[CheckIssue]:
    #     """
    #       检查命名规范(optional)
    #       只需要读取输入 image 的 name 即可
    #     """
    #     issues = []
        
    #     import re
    #     naming_pattern = re.compile(r'^[a-z][a-z0-9_]*$')
        
    #     for part in plan.parts:
    #         if not naming_pattern.match(part.name):
    #             issues.append(CheckIssue(
    #                 severity="warning",
    #                 category="命名检查",
    #                 description=f"部件名称不符合规范: {part.name}",
    #                 suggestion="命名应使用小写字母、数字和下划线，以字母开头",
    #                 affected_parts=[part.name]
    #             ))
        
    #     return issues
    
    def _check_dimensions(self, plan: ObjectPlan) -> List[CheckIssue]:
        """检查尺寸合理性"""
        issues = []
        
        # 标准尺寸参考（米）
        # 非必要；类别不在standard_dimensions中，仍可正常运行
        standard_dimensions = {
            "chair": {"min_height": 0.4, "max_height": 1.2, "min_width": 0.3, "max_width": 1.2},
            "table": {"min_height": 0.4, "max_height": 1.2, "min_width": 0.5, "max_width": 1.2},
            "stool": {"min_height": 0.3, "max_height": 1.2, "min_width": 0.2, "max_width": 1.2},
        }
        
        category = plan.category
        if category in standard_dimensions:
            std = standard_dimensions[category]
            total = plan.total_dimensions
            
            if total.height < std["min_height"] or total.height > std["max_height"]:
                issues.append(CheckIssue(
                    severity="warning",
                    category="Dimension check",
                    description=f"Object height {total.height}m is outside typical range [{std['min_height']}, {std['max_height']}]",
                    suggestion="Review whether the overall height is reasonable"
                ))
            
            if total.width < std["min_width"] or total.width > std["max_width"]:
                issues.append(CheckIssue(
                    severity="warning",
                    category="Dimension check",
                    description=f"Object width {total.width}m is outside typical range [{std['min_width']}, {std['max_width']}]",
                    suggestion="Review whether the overall width is reasonable"
                ))
        
        # 检查各部件尺寸
        for part in plan.parts:
            dims = part.shape.dimensions
            # 检查是否有零或负值
            if dims.width <= 0 or dims.depth <= 0 or dims.height <= 0:
                issues.append(CheckIssue(
                    severity="error",
                    category="Dimension check",
                    description=f"Part {part.name} has invalid dimensions",
                    suggestion="Dimensions must be positive numbers",
                    affected_parts=[part.name]
                ))
            
            # 检查尺寸是否过小
            min_size = 0.001  # 1mm
            if dims.width < min_size or dims.depth < min_size or dims.height < min_size:
                issues.append(CheckIssue(
                    severity="warning",
                    category="Dimension check",
                    description=f"Part {part.name} is too small",
                    suggestion=f"Recommended minimum dimension is {min_size}m",
                    affected_parts=[part.name]
                ))
        
        return issues
    
    def _check_priority(self, plan: ObjectPlan) -> List[CheckIssue]:
        """检查构建优先级"""
        issues = []
        
        # 检查是否有负优先级
        for part in plan.parts:
            if part.priority < 0:
                issues.append(CheckIssue(
                    severity="error",
                    category="Priority check",
                    description=f"Part {part.name} has negative priority",
                    suggestion="Use a non-negative integer priority",
                    affected_parts=[part.name]
                ))
        
        return issues

    def _is_movable_part(self, part: Any) -> bool:
        """判断部件是否可动，兼容旧 plan（无 is_movable 字段时用关键词推断）。"""
        if getattr(part, "is_movable", False) or getattr(part, "must_be_independent", False):
            return True
        name = str(getattr(part, "name", "")).lower()
        part_type = str(getattr(part, "part_type", "")).lower()
        combined = f"{name} {part_type}"
        return any(k in combined for k in self.MOVABLE_KEYWORDS)

    def _check_movable_independence(self, plan: ObjectPlan) -> List[CheckIssue]:
        """检查可动件是否独立、是否误用 array/mirror。"""
        issues: List[CheckIssue] = []

        for part in plan.parts:
            if not self._is_movable_part(part):
                continue

            part_name = part.name
            has_instances = bool(getattr(part, "instances", []))

            must_be_independent = bool(getattr(part, "must_be_independent", False))
            if not must_be_independent:
                issues.append(CheckIssue(
                    severity="warning",
                    category="Movable-part independence",
                    description=f"Movable part {part_name} does not explicitly set must_be_independent=true",
                    suggestion="Set must_be_independent=true for movable parts to avoid merged modeling",
                    affected_parts=[part_name]
                ))

            modifiers = [str(m).lower() for m in getattr(part.shape, "modifiers", [])]
            forbidden_mods = [
                m for m in modifiers
                if any(k in m for k in self.FORBIDDEN_MODIFIERS_FOR_MOVABLE)
            ]
            if forbidden_mods:
                issues.append(CheckIssue(
                    severity="error",
                    category="Movable-part independence",
                    description=f"Movable part {part_name} uses forbidden modifiers: {forbidden_mods}",
                    suggestion="Movable parts must not use array/mirror; split into independent part objects",
                    affected_parts=[part_name]
                ))

            # 校验 instances 字段
            if has_instances:
                instances = part.instances
                inst_names = set()
                for inst in instances:
                    if "name" not in inst or "position" not in inst:
                        issues.append(CheckIssue(
                            severity="error",
                            category="Movable-part independence",
                            description=f"Instances entry for movable part {part_name} is missing `name` or `position`",
                            suggestion="Each instance must include `name` and `position`",
                            affected_parts=[part_name]
                        ))
                    inst_name = inst.get("name", "")
                    if inst_name in inst_names:
                        issues.append(CheckIssue(
                            severity="error",
                            category="Movable-part independence",
                            description=f"Duplicate instance name in movable part {part_name}: {inst_name}",
                            suggestion="Each instance `name` must be unique",
                            affected_parts=[part_name]
                        ))
                    inst_names.add(inst_name)
                continue  # instances 模式合法，跳过组对象检查

            # 无 instances 时，组名可动件（如 drawers_left / doors）通常表示尚未拆成独立实例
            group_like = (
                bool(re.search(r"(drawers|doors|_group|_set)", part_name.lower()))
                or bool(re.search(r"(drawers|doors)", str(part.part_type).lower()))
            )
            indexed = bool(re.search(r"_\d+$", part_name))
            if group_like and not indexed:
                issues.append(CheckIssue(
                    severity="error",
                    category="Movable-part independence",
                    description=f"Movable part {part_name} appears to be a grouped object and is not split into independent instances",
                    suggestion="Split this movable group into independent parts, or use `instances` with each replica name and position",
                    affected_parts=[part_name]
                ))

        return issues
    
    def _calculate_score(self, issues: List[CheckIssue]) -> int:
        """计算检查分数"""
        score = 100
        
        for issue in issues:
            if issue.severity == "error":
                score -= 20
            elif issue.severity == "warning":
                score -= 5
            elif issue.severity == "suggestion":
                score -= 1
        
        return max(0, score)
    
    def _rule_based_check(self, plan: ObjectPlan) -> Tuple[List[CheckIssue], Dict[str, Any]]:
        """基于规则的检查"""
        all_issues = []
        
        # 执行各项检查
        # all_issues.extend(self._check_category(plan))
        all_issues.extend(self._check_parts_completeness(plan))
        # all_issues.extend(self._check_naming_convention(plan))
        all_issues.extend(self._check_dimensions(plan))
        all_issues.extend(self._check_priority(plan))
        all_issues.extend(self._check_movable_independence(plan))
        
        # 生成修正建议
        corrections = {}
        
        return all_issues, corrections
    
    async def _llm_based_check(
        self,
        plan: ObjectPlan,
        image_path: Optional[str] = None,
        text_input: Optional[str] = None
    ) -> Tuple[List[CheckIssue], Dict[str, Any]]:
        """基于 LLM 的检查"""
        if self.llm_client is None:
            return [], {}

        input_mode = self._validate_single_input(image_path=image_path, text_input=text_input)

        messages = [
            {
                "role": "system",
                "content": self._planner_checker_prompt()
            },
        ]

        if input_mode == "image":
            image_url = self._get_image_url(image_path)
            user_content: Any = [
                {"type": "image_url", "image_url": {"url": image_url}},
                {
                    "type": "text",
                    "text": f"""Check whether the following ObjectPlan correctly describes the furniture in the image.

ObjectPlan:
```json
{plan.to_json()}
```

Output the check result strictly in the required format."""
                }
            ]
        else:
            user_content = [
                {
                    "type": "text",
                    "text": f"""Check whether the following ObjectPlan correctly describes the furniture in the text requirements.

Original text requirement:
{text_input}

ObjectPlan:
```json
{plan.to_json()}
```

Output the check result strictly in the required format."""
                }
            ]
        messages.append({"role": "user", "content": user_content})
        
        response = await self._call_llm(messages)
        result = self._parse_json_response(response)
        
        # 解析 LLM 返回的问题
        issues = []
        for issue_dict in result.get("issues", []):
            issues.append(CheckIssue(
                severity=issue_dict.get("severity", "warning"),
                category=issue_dict.get("category", "LLM check"),
                description=issue_dict.get("description", ""),
                suggestion=issue_dict.get("suggestion", "")
            ))
        
        corrections = result.get("corrections", {})
        
        return issues, corrections
    
    async def run(
        self,
        plan: ObjectPlan,
        image_path: Optional[str] = None,
        text_input: Optional[str] = None,
        use_llm: bool = True
    ) -> CheckResult:
        """
        执行检查
        
        Args:
            plan: ObjectPlan 对象
            image_path: 原始输入图片路径（与 text_input 二选一）
            text_input: 原始输入文本描述（与 image_path 二选一）
            use_llm: 是否使用 LLM 进行额外检查
            
        Returns:
            CheckResult 对象
        """
        self.logger.info(f"Starting plan check for: {plan.name}")
        self._validate_single_input(image_path=image_path, text_input=text_input)
        
        all_issues = []
        all_corrections = {}
        
        # 规则检查
        self.logger.info("Running rule-based checks...")
        rule_issues, rule_corrections = self._rule_based_check(plan)
        all_issues.extend(rule_issues)
        all_corrections.update(rule_corrections)
        
        # LLM 检查
        if use_llm and self.llm_client is not None:
            self.logger.info("Running LLM-based checks...")
            try:
                llm_issues, llm_corrections = await self._llm_based_check(
                    plan,
                    image_path=image_path,
                    text_input=text_input
                )
                all_issues.extend(llm_issues)
                all_corrections.update(llm_corrections)
            except Exception as e:
                self.logger.warning(f"LLM check failed: {e}")
        
        # 计算分数
        score = self._calculate_score(all_issues)
        
        # 判断是否有效
        has_errors = any(i.severity == "error" for i in all_issues)
        # is_valid = not has_errors and score >= 60
        is_valid = not has_errors 
        
        # 生成摘要
        error_count = sum(1 for i in all_issues if i.severity == "error")
        warning_count = sum(1 for i in all_issues if i.severity == "warning")
        summary = f"Check completed: {error_count} errors, {warning_count} warnings, score {score}/100"
        
        result = CheckResult(
            is_valid=is_valid,
            score=score,
            issues=all_issues,
            corrections=all_corrections,
            summary=summary
        )
        
        self.logger.info(summary)
        return result
    
    async def check_and_fix(
        self,
        plan: ObjectPlan,
        image_path: Optional[str] = None,
        text_input: Optional[str] = None,
        planner_agent: Any = None,
        max_retry: int = None
    ) -> Tuple[ObjectPlan, List[CheckResult]]:
        """
        检查并修复 plan
        
        Args:
            plan: ObjectPlan 对象
            image_path: 原始图片路径（与 text_input 二选一）
            text_input: 原始输入文本（与 image_path 二选一）
            planner_agent: Planner Agent 实例（用于基于 ObjectPlan 的修正）
            max_retry: 最大重试次数
            
        Returns:
            (修正后的 ObjectPlan, 检查历史)
        """
        self._validate_single_input(image_path=image_path, text_input=text_input)
        max_retry = max_retry or self.max_retry
        check_history = []
        current_plan = plan
        
        for attempt in range(max_retry):
            self.logger.info(f"Check attempt {attempt + 1}/{max_retry}")
            
            # 检查
            result = await self.run(
                current_plan,
                image_path=image_path,
                text_input=text_input
            )
            check_history.append(result)
            
            # 如果通过，返回
            if result.is_valid:
                self.logger.info("Plan check passed!")
                return current_plan, check_history
            
            # 如果有修正建议，应用修正（基于现有 ObjectPlan 做最小修改）
            if result.corrections and planner_agent:
                self.logger.info("Applying corrections via ObjectPlan revision (no full re-planning)...")
                
                # 构建修正上下文（仅包含摘要与问题列表，原始 JSON 由 Planner 侧注入）
                correction_context = f"""
The previous plan has the following issues:
{result.summary}

Detailed issues:
{[i.to_dict() for i in result.issues if i.severity in ['error', 'warning']]}

Suggested modifications:
{result.corrections}

The LLM must minimally modify only incorrect fields or parts while keeping other correct parts unchanged.
"""
                try:
                    # 使用 PlannerAgent 的修正接口：ObjectPlan → 修正后 ObjectPlan
                    current_plan = await planner_agent.revise_plan(
                        original_plan=current_plan,
                        correction_context=correction_context,
                        image_path=image_path,
                        text_input=text_input,
                    )
                except Exception as e:
                    self.logger.error(f"Plan revision failed: {e}, will retry...")
                    continue
            else:
                # 没有修正建议或没有 planner_agent，无法继续修复
                self.logger.warning("No corrections available or no planner agent provided")
                break
        
        self.logger.warning(f"Plan check did not pass after {max_retry} attempts")
        return current_plan, check_history
