"""Helpers for generating assets with the vendored Code_Object pipeline."""

from __future__ import annotations

import asyncio
import importlib
import logging
import random
import threading

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from scenecode.agent_utils.mesh_utils import convert_gltf_to_glb
from scenecode.utils.network_utils import is_port_available

console_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodeObjectGenerationResult:
    """Result bundle returned by the Code_Object pipeline."""

    output_dir: Path
    mesh_path: Path
    object_plan_path: Path | None
    code_dir: Path | None
    pipeline_result_path: Path | None
    full_object_render_path: Path | None
    status: str = "generated"
    urdf_path: Path | None = None


@dataclass(frozen=True)
class _CodeObjectResumeState:
    """Cached-output state used to resume a failed Code_Object run."""

    has_existing_plan: bool
    has_existing_parts: bool
    skip_stages: tuple[str, ...]


class CodeObjectRunner:
    """Thin adapter that invokes Code_Object's pipeline from SceneCode."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._host = "127.0.0.1"
        self._package_root = Path(__file__).resolve().parents[1]
        self._repo_root = self._package_root.parent

        code_object_cfg = cfg.asset_manager.get("code_object", {})
        self._code_object_root = self._resolve_code_object_root(
            code_object_cfg.get("root")
        )
        configured_path = code_object_cfg.get("config_path")
        self._config_path = self._resolve_config_path(configured_path)

        port_range = code_object_cfg.get("blender_mcp_port_range", [9900, 9999])
        if len(port_range) != 2:
            raise ValueError(
                "asset_manager.code_object.blender_mcp_port_range must contain "
                "exactly two integers"
            )
        self._port_range = (int(port_range[0]), int(port_range[1]))
        self._max_concurrent_runs = int(code_object_cfg.get("max_concurrent_runs", 2))
        if self._max_concurrent_runs < 1:
            raise ValueError("asset_manager.code_object.max_concurrent_runs must be >= 1")
        self._max_attempts = int(code_object_cfg.get("max_attempts", 3))
        if self._max_attempts < 1:
            raise ValueError("asset_manager.code_object.max_attempts must be >= 1")

        self._reserved_ports: set[int] = set()
        self._reservation_lock = threading.Lock()
        self._concurrency_semaphore = threading.BoundedSemaphore(
            self._max_concurrent_runs
        )

    def generate_from_image(
        self,
        *,
        image_path: Path,
        output_dir: Path,
        config_path_override: str | Path | None = None,
    ) -> CodeObjectGenerationResult:
        """Run the Code_Object pipeline for a single reference image."""
        if not image_path.exists():
            raise FileNotFoundError(f"Code_Object input image not found: {image_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        active_config_path = (
            self._resolve_config_path(str(config_path_override))
            if config_path_override is not None
            else self._config_path
        )
        port = self._reserve_port()
        attempt_summaries: list[str] = []
        try:
            pipeline_cls = self._load_pipeline_class()
            for attempt in range(1, self._max_attempts + 1):
                resume_state = self._detect_resume_state(output_dir)
                console_logger.info(
                    "Code_Object object attempt %d/%d for %s "
                    "(has_existing_plan=%s, has_existing_parts=%s, skip_stages=%s)",
                    attempt,
                    self._max_attempts,
                    image_path,
                    resume_state.has_existing_plan,
                    resume_state.has_existing_parts,
                    list(resume_state.skip_stages),
                )

                pipeline = pipeline_cls(
                    config_path=str(active_config_path),
                    use_mock=False,
                    port=port,
                )
                result = asyncio.run(
                    pipeline.run(
                        image_path=str(image_path),
                        output_dir=str(output_dir),
                        skip_stages=list(resume_state.skip_stages),
                    )
                )

                if getattr(result, "success", False):
                    mesh_path = self._resolve_mesh_path(
                        result=result, output_dir=output_dir
                    )
                    if mesh_path is not None and mesh_path.exists():
                        if attempt > 1:
                            console_logger.info(
                                "Code_Object internal retry succeeded on attempt %d/%d "
                                "for %s",
                                attempt,
                                self._max_attempts,
                                image_path,
                            )
                        return self._build_generation_result(
                            output_dir=output_dir,
                            mesh_path=mesh_path,
                        )

                    failure_summary = (
                        "pipeline reported success but did not produce a GLB export"
                    )
                else:
                    failure_summary = self._summarize_pipeline_failure(result)

                attempt_summaries.append(
                    f"attempt {attempt}/{self._max_attempts}: {failure_summary}"
                )
                console_logger.warning(
                    "Code_Object attempt %d/%d failed for %s: %s",
                    attempt,
                    self._max_attempts,
                    image_path,
                    failure_summary,
                )

                if attempt < self._max_attempts:
                    console_logger.info(
                        "Retrying Code_Object for %s with the same output directory "
                        "to reuse any existing plan/parts",
                        image_path,
                    )
        finally:
            self._release_port(port)

        console_logger.warning(
            "Code_Object internal retries exhausted for %s after %d attempt(s); "
            "allowing the caller to continue with outer fallback logic",
            image_path,
            self._max_attempts,
        )
        raise RuntimeError(
            f"Code_Object pipeline failed for {image_path} after "
            f"{self._max_attempts} attempt(s): {'; '.join(attempt_summaries)}"
        )

    def generate_articulated_from_image(
        self,
        *,
        image_path: Path,
        output_dir: Path,
        config_path_override: str | Path | None = None,
    ) -> CodeObjectGenerationResult:
        """Run the articulated Code_Object pipeline for a single reference image."""
        if not image_path.exists():
            raise FileNotFoundError(f"Code_Object input image not found: {image_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        active_config_path = (
            self._resolve_config_path(str(config_path_override))
            if config_path_override is not None
            else self._config_path
        )
        port = self._reserve_port()
        attempt_summaries: list[str] = []
        try:
            pipeline_cls = self._load_pipeline_class("CodeArticulatedPipeline")
            for attempt in range(1, self._max_attempts + 1):
                resume_state = self._detect_resume_state(output_dir)
                console_logger.info(
                    "Code_Object articulated attempt %d/%d for %s "
                    "(has_existing_plan=%s, has_existing_parts=%s, skip_stages=%s)",
                    attempt,
                    self._max_attempts,
                    image_path,
                    resume_state.has_existing_plan,
                    resume_state.has_existing_parts,
                    list(resume_state.skip_stages),
                )

                pipeline = pipeline_cls(
                    config_path=str(active_config_path),
                    use_mock=False,
                    port=port,
                )
                result = asyncio.run(
                    pipeline.run(
                        image_path=str(image_path),
                        output_dir=str(output_dir),
                        skip_stages=list(resume_state.skip_stages),
                    )
                )

                if getattr(result, "success", False):
                    mesh_path = self._resolve_mesh_path(
                        result=result, output_dir=output_dir
                    )
                    status = getattr(result, "status", "generated")
                    urdf_path_raw = getattr(result, "urdf_path", None)
                    urdf_path = Path(urdf_path_raw) if urdf_path_raw else None
                    if mesh_path is not None and mesh_path.exists():
                        if status == "generated" and (urdf_path is None or not urdf_path.exists()):
                            failure_summary = (
                                "articulated pipeline reported success but did not produce a URDF export"
                            )
                        else:
                            if attempt > 1:
                                console_logger.info(
                                    "Code_Object articulated retry succeeded on attempt %d/%d "
                                    "for %s",
                                    attempt,
                                    self._max_attempts,
                                    image_path,
                                )
                            return self._build_generation_result(
                                output_dir=output_dir,
                                mesh_path=mesh_path,
                                status=status,
                                urdf_path=urdf_path,
                            )
                    else:
                        failure_summary = (
                            "articulated pipeline reported success but did not produce a GLB export"
                        )
                else:
                    failure_summary = self._summarize_pipeline_failure(result)

                attempt_summaries.append(
                    f"attempt {attempt}/{self._max_attempts}: {failure_summary}"
                )
                console_logger.warning(
                    "Code_Object articulated attempt %d/%d failed for %s: %s",
                    attempt,
                    self._max_attempts,
                    image_path,
                    failure_summary,
                )

                if attempt < self._max_attempts:
                    console_logger.info(
                        "Retrying articulated Code_Object for %s with the same output directory "
                        "to reuse any existing plan/parts",
                        image_path,
                    )
        finally:
            self._release_port(port)

        console_logger.warning(
            "Code_Object articulated retries exhausted for %s after %d attempt(s); "
            "allowing the caller to continue with outer fallback logic",
            image_path,
            self._max_attempts,
        )
        raise RuntimeError(
            f"Code_Object articulated pipeline failed for {image_path} after "
            f"{self._max_attempts} attempt(s): {'; '.join(attempt_summaries)}"
        )

    def _resolve_code_object_root(self, configured_root: str | None) -> Path:
        """Resolve the vendored Code_Object root directory."""
        candidates: list[Path] = []
        if configured_root:
            raw_root = Path(configured_root)
            if raw_root.is_absolute():
                candidates.append(raw_root)
            else:
                candidates.extend(
                    [
                        self._repo_root / raw_root,
                        self._package_root / raw_root,
                    ]
                )

        candidates.append(self._package_root / "code_object")

        for candidate in candidates:
            if (candidate / "agent_framework").exists():
                return candidate.resolve()

        raise FileNotFoundError(
            "Could not resolve the vendored Code_Object root. Tried: "
            + ", ".join(str(path) for path in candidates)
        )

    def _resolve_config_path(self, configured_path: str | None) -> Path:
        """Resolve a Code_Object config path against SceneCode package locations."""
        candidates: list[Path] = []
        if configured_path:
            raw_path = Path(configured_path)
            if raw_path.is_absolute():
                candidates.append(raw_path)
            else:
                legacy_parts = raw_path.parts
                if legacy_parts and legacy_parts[0] == "Code_Object":
                    candidates.append(self._code_object_root / Path(*legacy_parts[1:]))
                candidates.extend(
                    [
                        self._repo_root / raw_path,
                        self._code_object_root / raw_path,
                        self._code_object_root / "agent_framework" / raw_path,
                    ]
                )

        candidates.append(self._code_object_root / "agent_framework" / "config.yaml")

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(
            "Could not resolve a valid Code_Object config path. Tried: "
            + ", ".join(str(path) for path in candidates)
        )

    def _load_pipeline_class(self, class_name: str = "Pipeline"):
        """Import Code_Object pipeline classes lazily to keep startup light."""
        module = importlib.import_module("scenecode.code_object.agent_framework")
        return getattr(module, class_name)

    def _reserve_port(self) -> int:
        """Reserve a Blender MCP port for one Code_Object pipeline run."""
        self._concurrency_semaphore.acquire()
        try:
            with self._reservation_lock:
                ports = list(range(self._port_range[0], self._port_range[1] + 1))
                random.shuffle(ports)
                for port in ports:
                    if port in self._reserved_ports:
                        continue
                    if not is_port_available(self._host, port):
                        continue
                    self._reserved_ports.add(port)
                    return port
        except Exception:
            self._concurrency_semaphore.release()
            raise

        self._concurrency_semaphore.release()
        raise RuntimeError(
            "No available Blender MCP ports for Code_Object in range "
            f"{self._port_range}"
        )

    def _release_port(self, port: int) -> None:
        """Release a previously reserved port."""
        with self._reservation_lock:
            self._reserved_ports.discard(port)
        self._concurrency_semaphore.release()

    def _detect_resume_state(self, output_dir: Path) -> _CodeObjectResumeState:
        """Inspect cached outputs to decide which pipeline stages can be skipped."""
        object_plan_path = output_dir / "ObjectPlan.json"
        code_dir = output_dir / "code"
        part_files = [
            part_file
            for parts_dir in code_dir.glob("parts_*")
            if parts_dir.is_dir()
            for part_file in parts_dir.glob("*.py")
            if part_file.is_file() and part_file.name != "__init__.py"
        ]
        has_existing_parts = bool(part_files)
        skip_stages = ("plan_check",) if has_existing_parts else ()
        return _CodeObjectResumeState(
            has_existing_plan=object_plan_path.exists(),
            has_existing_parts=has_existing_parts,
            skip_stages=skip_stages,
        )

    def _summarize_pipeline_failure(self, result) -> str:
        """Create a concise failure summary for retry logging."""
        summary_parts: list[str] = []
        stages_failed = getattr(result, "stages_failed", None)
        if stages_failed:
            summary_parts.append(f"stages_failed={stages_failed}")
        error = getattr(result, "error", None)
        if error:
            summary_parts.append(f"error={error}")
        if not summary_parts:
            summary_parts.append("pipeline returned unsuccessful result")
        return ", ".join(summary_parts)

    def _build_generation_result(
        self,
        *,
        output_dir: Path,
        mesh_path: Path,
        status: str = "generated",
        urdf_path: Path | None = None,
    ) -> CodeObjectGenerationResult:
        """Build the result bundle from the shared output directory."""
        object_plan_path = output_dir / "ObjectPlan.json"
        code_dir = output_dir / "code"
        pipeline_result_path = output_dir / "pipeline_result.json"
        full_object_render_path = output_dir / "renders" / "full_object.png"

        return CodeObjectGenerationResult(
            output_dir=output_dir,
            mesh_path=mesh_path,
            object_plan_path=object_plan_path if object_plan_path.exists() else None,
            code_dir=code_dir if code_dir.exists() else None,
            pipeline_result_path=(
                pipeline_result_path if pipeline_result_path.exists() else None
            ),
            full_object_render_path=(
                full_object_render_path if full_object_render_path.exists() else None
            ),
            status=status,
            urdf_path=urdf_path if urdf_path is not None and urdf_path.exists() else None,
        )

    def _resolve_mesh_path(self, result, output_dir: Path) -> Path | None:
        """Resolve the final GLB path from the pipeline result or output directory."""
        pipeline_path = (
            getattr(result, "mesh_path", None)
            or getattr(result, "glb_path", None)
            or getattr(result, "gltf_path", None)
        )
        if pipeline_path:
            candidate = Path(pipeline_path)
            if not candidate.is_absolute():
                candidate = output_dir / candidate
            return self._ensure_glb_export(candidate)

        for mesh_dir in (output_dir / "mesh", output_dir / "gltf"):
            if not mesh_dir.exists():
                continue

            glb_files = sorted(mesh_dir.glob("*.glb"))
            if len(glb_files) == 1:
                return glb_files[0]

            if len(glb_files) > 1:
                console_logger.warning(
                    "Multiple Code_Object GLB files found in %s, using %s",
                    mesh_dir,
                    glb_files[0],
                )
                return glb_files[0]

            gltf_files = sorted(mesh_dir.glob("*.gltf"))
            if len(gltf_files) == 1:
                return self._ensure_glb_export(gltf_files[0])

            if len(gltf_files) > 1:
                console_logger.warning(
                    "Multiple Code_Object GLTF files found in %s, using %s",
                    mesh_dir,
                    gltf_files[0],
                )
                return self._ensure_glb_export(gltf_files[0])

        return None

    def _ensure_glb_export(self, mesh_path: Path) -> Path:
        """Convert Code_Object GLTF exports to GLB so materials stay bundled."""
        suffix = mesh_path.suffix.lower()
        if suffix == ".glb":
            return mesh_path
        if suffix != ".gltf":
            raise ValueError(f"Unsupported Code_Object mesh format: {mesh_path}")

        glb_path = mesh_path.with_suffix(".glb")
        if glb_path.exists():
            return glb_path

        console_logger.info(
            "Converting Code_Object export to GLB for downstream use: %s -> %s",
            mesh_path,
            glb_path,
        )
        return convert_gltf_to_glb(mesh_path, glb_path)
