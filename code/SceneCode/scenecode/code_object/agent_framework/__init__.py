"""
Agent Framework for 3D Furniture Generation
用于生成高质量 3D 家具对象及对应代码的 Agent Framework
"""

from .schemas import ObjectPlan, PartPlan, ObjectCategory
from .agents import (
    BaseAgent,
    PlannerAgent,
    PlannerCheckerAgent,
    PartConstructorAgent,
)
from .blender import BlenderMCPClient

__version__ = "0.1.0"

__all__ = [
    # Pipeline
    "Pipeline",
    "PipelineResult",
    "CodeArticulatedPipeline",
    "CodeArticulatedPipelineResult",
    
    # Schemas
    "ObjectPlan",
    "PartPlan",
    "ObjectCategory",
    
    # Agents
    "BaseAgent",
    "PlannerAgent",
    "PlannerCheckerAgent",
    "PartConstructorAgent",
    
    # Blender
    "BlenderMCPClient",
]


def __getattr__(name: str):
    """懒加载 pipeline，避免 python -m agent_framework.pipeline 时触发 RuntimeWarning。"""
    if name == "Pipeline":
        from .pipeline import Pipeline
        return Pipeline
    if name == "PipelineResult":
        from .pipeline import PipelineResult
        return PipelineResult
    if name == "CodeArticulatedPipeline":
        from .code_articulated_pipeline import CodeArticulatedPipeline
        return CodeArticulatedPipeline
    if name == "CodeArticulatedPipelineResult":
        from .code_articulated_pipeline import CodeArticulatedPipelineResult
        return CodeArticulatedPipelineResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
