"""Articulated wrapper pipeline for Code_Object outputs.

This pipeline reuses the existing image/text-conditioned Code_Object pipeline to
produce ObjectPlan, code, renders, and final mesh exports. It then runs the
URDF generation pipeline when the ObjectPlan contains movable parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from .pipeline import Pipeline, PipelineResult
    from .urdf_pipeline_v1 import URDFPipeline
except ImportError:  # pragma: no cover - script mode fallback
    from pipeline import Pipeline, PipelineResult
    from urdf_pipeline_v1 import URDFPipeline


@dataclass
class CodeArticulatedPipelineResult:
    """Result bundle for articulated Code_Object generation."""

    success: bool
    status: str
    object_name: str
    output_dir: str
    mesh_path: str | None = None
    object_plan_path: str | None = None
    code_dir: str | None = None
    pipeline_result_path: str | None = None
    full_object_render_path: str | None = None
    parts_output_dir: str | None = None
    urdf_path: str | None = None
    stages_completed: list[str] = field(default_factory=list)
    stages_failed: list[str] = field(default_factory=list)
    error: str | None = None
    base_pipeline_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "object_name": self.object_name,
            "output_dir": self.output_dir,
            "mesh_path": self.mesh_path,
            "object_plan_path": self.object_plan_path,
            "code_dir": self.code_dir,
            "pipeline_result_path": self.pipeline_result_path,
            "full_object_render_path": self.full_object_render_path,
            "parts_output_dir": self.parts_output_dir,
            "urdf_path": self.urdf_path,
            "stages_completed": self.stages_completed,
            "stages_failed": self.stages_failed,
            "error": self.error,
            "base_pipeline_result": self.base_pipeline_result,
        }


class CodeArticulatedPipeline:
    """Wrapper around the existing Code_Object pipeline plus URDF generation."""

    def __init__(
        self,
        config_path: str | None = None,
        llm_client=None,
        use_mock: bool = False,
        port: int | None = None,
        urdf_prompt_path: str | None = None,
        urdf_llm_config_key: str = "constructor_llm",
    ) -> None:
        self.pipeline = Pipeline(
            config_path=config_path,
            llm_client=llm_client,
            use_mock=use_mock,
            port=port,
        )
        self.urdf_pipeline = URDFPipeline(
            config_path=config_path,
            prompt_path=urdf_prompt_path,
            llm_config_key=urdf_llm_config_key,
            llm_client=llm_client,
            use_mock=use_mock,
        )

    async def run(
        self,
        image_path: str | None = None,
        text_input: str | None = None,
        output_dir: str | None = None,
        skip_stages: list[str] | None = None,
        mesh_output_path: str | None = None,
    ) -> CodeArticulatedPipelineResult:
        """Run the base Code_Object pipeline, then generate URDF if needed."""
        base_result: PipelineResult = await self.pipeline.run(
            image_path=image_path,
            text_input=text_input,
            output_dir=output_dir,
            skip_stages=skip_stages,
            mesh_output_path=mesh_output_path,
        )

        resolved_output_dir = base_result.output_dir or (output_dir or "")
        object_dir = Path(resolved_output_dir) if resolved_output_dir else None

        object_plan_path = None
        code_dir = None
        pipeline_result_path = None
        full_object_render_path = None
        parts_output_dir = None
        if object_dir is not None:
            candidate = object_dir / "ObjectPlan.json"
            object_plan_path = str(candidate) if candidate.exists() else None

            candidate = object_dir / "code"
            code_dir = str(candidate) if candidate.exists() else None

            candidate = object_dir / "pipeline_result.json"
            pipeline_result_path = str(candidate) if candidate.exists() else None

            candidate = object_dir / "renders" / "full_object.png"
            full_object_render_path = str(candidate) if candidate.exists() else None

            candidate = object_dir / "blender_output" / "parts"
            parts_output_dir = str(candidate) if candidate.exists() else None

        if not base_result.success:
            return CodeArticulatedPipelineResult(
                success=False,
                status="failed",
                object_name=base_result.object_name,
                output_dir=resolved_output_dir,
                mesh_path=base_result.mesh_path,
                object_plan_path=object_plan_path,
                code_dir=code_dir,
                pipeline_result_path=pipeline_result_path,
                full_object_render_path=full_object_render_path,
                parts_output_dir=parts_output_dir,
                stages_completed=list(base_result.stages_completed),
                stages_failed=list(base_result.stages_failed),
                error=base_result.error,
                base_pipeline_result=base_result.to_dict(),
            )

        if object_dir is None:
            return CodeArticulatedPipelineResult(
                success=False,
                status="failed",
                object_name=base_result.object_name,
                output_dir="",
                mesh_path=base_result.mesh_path,
                object_plan_path=object_plan_path,
                code_dir=code_dir,
                pipeline_result_path=pipeline_result_path,
                full_object_render_path=full_object_render_path,
                parts_output_dir=parts_output_dir,
                stages_completed=list(base_result.stages_completed),
                stages_failed=list(base_result.stages_failed),
                error="Base pipeline succeeded without an output directory",
                base_pipeline_result=base_result.to_dict(),
            )

        urdf_result = await self.urdf_pipeline.process_object_dir(object_dir)
        urdf_path: str | None = None
        status = "failed"
        success = False
        error = None

        if urdf_result.status == "generated":
            status = "generated"
            success = True
            urdf_path = urdf_result.output_path
        elif urdf_result.status == "skipped":
            if urdf_result.reason == "No part with is_movable=true in ObjectPlan.":
                status = "no_movable_parts"
                success = True
            elif urdf_result.reason == "URDF file already exists.":
                status = "generated"
                success = True
                urdf_path = urdf_result.output_path
            else:
                error = urdf_result.reason or "URDF generation skipped unexpectedly"
        else:
            error = urdf_result.error or "URDF generation failed"

        stages_completed = list(base_result.stages_completed)
        stages_failed = list(base_result.stages_failed)
        if success and status == "generated":
            stages_completed.append("urdf_generation")
        elif status == "no_movable_parts":
            stages_completed.append("movability_check")
        else:
            stages_failed.append("urdf_generation")

        return CodeArticulatedPipelineResult(
            success=success,
            status=status,
            object_name=base_result.object_name,
            output_dir=resolved_output_dir,
            mesh_path=base_result.mesh_path,
            object_plan_path=object_plan_path,
            code_dir=code_dir,
            pipeline_result_path=pipeline_result_path,
            full_object_render_path=full_object_render_path,
            parts_output_dir=parts_output_dir,
            urdf_path=urdf_path,
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            error=error,
            base_pipeline_result=base_result.to_dict(),
        )
