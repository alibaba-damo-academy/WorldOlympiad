"""
URDF generation pipeline.

Inputs:
1. ObjectPlan JSON
2. Target folder path (single object folder or batch parent folder)
3. URDF prompt template path

Output:
- <object_dir>/<object_name>.urdf
"""

import argparse
import asyncio
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from .utils.llm_client import LLMClient, TokenUsageTracker, create_llm_client
except ImportError:  # pragma: no cover - script mode fallback
    from utils.llm_client import LLMClient, TokenUsageTracker, create_llm_client


DEFAULT_PLAN_CANDIDATES = ("ObjectPlan.json", "objectPlan.json")


@dataclass
class URDFGenerationResult:
    """Single object URDF generation result."""

    status: str  # generated | skipped | failed
    object_dir: str
    object_name: str = ""
    plan_path: str = ""
    output_path: str = ""
    obj_files: List[str] = field(default_factory=list)
    reason: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "object_dir": self.object_dir,
            "object_name": self.object_name,
            "plan_path": self.plan_path,
            "output_path": self.output_path,
            "obj_files": self.obj_files,
            "reason": self.reason,
            "error": self.error,
        }


class URDFPipeline:
    """URDF generation pipeline with single/batch modes."""

    def __init__(
        self,
        *,
        config_path: Optional[str] = None,
        prompt_path: Optional[str] = None,
        llm_config_key: str = "constructor_llm",
        llm_client: Optional[LLMClient] = None,
        use_mock: bool = False,
    ):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.use_mock = use_mock
        self.prompt_path = Path(prompt_path) if prompt_path else (Path(__file__).parent / "prompts" / "urdf.md")
        self.prompt_template = self._load_prompt_template(self.prompt_path)
        self.llm_config_key = llm_config_key
        self._usage_tracker: Optional[TokenUsageTracker] = None

        if llm_client is not None:
            self.llm_client = llm_client
        elif not use_mock:
            self.llm_client = self._init_llm_client(config_key=llm_config_key)
        else:
            self.llm_client = None
            self.logger.info("Mock mode enabled, LLM client is not initialized.")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        if config_path is None:
            config_path = str(Path(__file__).parent / "config.yaml")
        cfg_path = Path(config_path)
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {
            "model": {"llm": {"provider": "openai", "model_name": "gpt-4o-mini"}},
            "paths": {"log_dir": "./logs"},
            "logging": {"level": "INFO"},
        }

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("URDFPipeline")
        # Prevent duplicate output through the root logger.
        logger.propagate = False
        level_name = (self.config.get("logging", {}) or {}).get("level", "INFO")
        level = getattr(logging, str(level_name).upper(), logging.INFO)
        logger.setLevel(level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _load_prompt_template(self, prompt_path: Path) -> str:
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        with prompt_path.open("r", encoding="utf-8") as f:
            return f.read()

    def _init_llm_client(self, config_key: str) -> LLMClient:
        model_cfg = (self.config.get("model", {}) or {}).get(config_key, {})
        if not model_cfg:
            # Fallback chain
            if config_key != "constructor_llm":
                model_cfg = (self.config.get("model", {}) or {}).get("constructor_llm", {})
            if not model_cfg:
                model_cfg = (self.config.get("model", {}) or {}).get("llm", {})

        provider = model_cfg.get("provider", "openai")
        model_name = model_cfg.get("model_name")
        api_base = model_cfg.get("api_base")
        temperature = model_cfg.get("temperature", 0.2)

        log_dir = (self.config.get("paths", {}) or {}).get("log_dir", "./logs")
        if self._usage_tracker is None:
            self._usage_tracker = TokenUsageTracker(log_dir=log_dir, logger_name="URDFPipeline.LLM")

        kwargs: Dict[str, Any] = {"usage_tracker": self._usage_tracker}
        if model_name:
            kwargs["model"] = model_name
        if api_base:
            kwargs["api_base"] = api_base
        if temperature is not None:
            kwargs["temperature"] = temperature

        self.logger.info(
            "Initializing URDF LLM client: config_key=%s provider=%s model=%s",
            config_key,
            provider,
            model_name,
        )
        return create_llm_client(provider=provider, **kwargs)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @staticmethod
    def _sanitize_identifier(name: str, fallback: str = "object") -> str:
        text = (name or "").strip().lower()
        text = text.encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"[^a-z0-9_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if not text:
            text = fallback
        if text[0].isdigit():
            text = f"obj_{text}"
        return text

    def _resolve_plan_path(self, object_dir: Path, object_plan_hint: Optional[str]) -> Path:
        if object_plan_hint:
            hint_path = Path(object_plan_hint)
            if hint_path.is_absolute() and hint_path.exists():
                return hint_path

            candidate = object_dir / object_plan_hint
            if candidate.exists():
                return candidate

            if hint_path.exists():
                return hint_path

            raise FileNotFoundError(
                f"ObjectPlan path not found. hint={object_plan_hint}, object_dir={object_dir}"
            )

        for name in DEFAULT_PLAN_CANDIDATES:
            path = object_dir / name
            if path.exists():
                return path

        raise FileNotFoundError(
            f"No ObjectPlan file found in {object_dir}. "
            f"Checked: {', '.join(DEFAULT_PLAN_CANDIDATES)}"
        )

    def _load_plan_json(self, plan_path: Path) -> Dict[str, Any]:
        with plan_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _part_has_movable(self, part: Dict[str, Any]) -> bool:
        if self._as_bool(part.get("is_movable", False)):
            return True
        for child in part.get("sub_parts", []) or []:
            if isinstance(child, dict) and self._part_has_movable(child):
                return True
        return False

    def _plan_requires_urdf(self, plan: Dict[str, Any]) -> bool:
        for part in (plan.get("parts") or []):
            if isinstance(part, dict) and self._part_has_movable(part):
                return True
        return False

    def _collect_obj_files(self, object_dir: Path) -> List[str]:
        parts_dir = object_dir / "blender_output" / "parts"
        if not parts_dir.exists() or not parts_dir.is_dir():
            raise FileNotFoundError(f"Parts directory not found: {parts_dir}")

        obj_files = sorted([p.name for p in parts_dir.glob("*.obj") if p.is_file()])
        if not obj_files:
            raise FileNotFoundError(f"No .obj files found in: {parts_dir}")
        return obj_files

    def _build_messages(
        self,
        *,
        object_dir: Path,
        object_name: str,
        plan_dict: Dict[str, Any],
        obj_files: List[str],
        output_filename: str,
    ) -> List[Dict[str, Any]]:
        plan_json = json.dumps(plan_dict, ensure_ascii=False, indent=2)
        obj_lines = "\n".join([f"- {f}" for f in obj_files])
        user_text = f"""Generate URDF from the following inputs.

Target URDF filename: {output_filename}

ObjectPlan JSON:
```json
{plan_json}
```

Available OBJ files under `blender_output/parts`:
{obj_lines}

Hard output constraints:
1. Return only URDF XML content (no markdown fence, no explanation).
2. The root must be `<robot name="{object_name}">`.
3. Every OBJ file in the list must map to one link.
4. Mesh path must use `blender_output/parts/<exact_obj_filename>`.
5. Keep XML well-formed and valid.
"""

        return [
            {"role": "system", "content": self.prompt_template},
            {"role": "user", "content": user_text},
        ]

    async def _generate_urdf_with_llm(self, messages: List[Dict[str, Any]]) -> str:
        if self.llm_client is None:
            raise RuntimeError("LLM client is not initialized.")

        response = await self.llm_client.chat(messages)
        if isinstance(response, dict):
            content = response.get("content", "")
        else:
            content = response

        if not content or not str(content).strip():
            raise ValueError("Empty response from LLM for URDF generation.")
        return self._extract_urdf_text(str(content))

    def _extract_urdf_text(self, content: str) -> str:
        text = content.strip()

        fence_match = re.search(r"```(?:xml)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()

        if "<?xml" in text:
            text = text[text.find("<?xml") :].strip()
        elif "<robot" in text:
            text = text[text.find("<robot") :].strip()
            text = f'<?xml version="1.0" ?>\n{text}'

        end_idx = text.rfind("</robot>")
        if end_idx != -1:
            text = text[: end_idx + len("</robot>")].strip()

        return text

    def _validate_urdf(self, urdf_text: str) -> None:
        if not urdf_text.strip():
            raise ValueError("Generated URDF is empty.")

        xml_body = re.sub(r"^\s*<\?xml[^>]*\?>", "", urdf_text, count=1).strip()
        try:
            root = ET.fromstring(xml_body)
        except ET.ParseError as e:
            # Include the generated URDF string in the error message so the user can inspect it
            raise ValueError(f"URDF XML parse error: {e}\n\n--- Invalid URDF Content ---\n{urdf_text}\n----------------------------") from e

        if root.tag != "robot":
            raise ValueError(f"URDF root tag must be <robot>, got <{root.tag}>")

    def _generate_mock_urdf(self, object_name: str, obj_files: List[str]) -> str:
        lines = [
            '<?xml version="1.0" ?>',
            f'<robot name="{object_name}">',
            '  <link name="base_link"/>',
            "",
        ]

        for filename in obj_files:
            link_name = self._sanitize_identifier(Path(filename).stem, fallback="part")
            mesh_path = f"blender_output/parts/{filename}"
            lines.extend(
                [
                    f'  <!-- {link_name}: fixed (mock) -->',
                    f'  <link name="{link_name}">',
                    "    <visual>",
                    '      <origin xyz="0 0 0" rpy="1.5707963 0 0"/>',
                    "      <geometry>",
                    f'        <mesh filename="{mesh_path}"/>',
                    "      </geometry>",
                    "    </visual>",
                    "    <collision>",
                    '      <origin xyz="0 0 0" rpy="1.5707963 0 0"/>',
                    "      <geometry>",
                    f'        <mesh filename="{mesh_path}"/>',
                    "      </geometry>",
                    "    </collision>",
                    "  </link>",
                    f'  <joint name="joint_{link_name}" type="fixed">',
                    '    <parent link="base_link"/>',
                    f'    <child link="{link_name}"/>',
                    '    <origin xyz="0 0 0" rpy="0 0 0"/>',
                    "  </joint>",
                    "",
                ]
            )

        lines.append("</robot>")
        return "\n".join(lines) + "\n"

    async def process_object_dir(
        self,
        object_dir: Path,
        *,
        object_plan_hint: Optional[str] = None,
        dry_run: bool = False,
    ) -> URDFGenerationResult:
        object_dir = object_dir.resolve()
        if not object_dir.exists() or not object_dir.is_dir():
            return URDFGenerationResult(
                status="failed",
                object_dir=str(object_dir),
                error=f"Object directory does not exist: {object_dir}",
            )

        try:
            plan_path = self._resolve_plan_path(object_dir, object_plan_hint)
            plan_dict = self._load_plan_json(plan_path)
        except Exception as e:
            return URDFGenerationResult(
                status="failed",
                object_dir=str(object_dir),
                error=str(e),
            )

        raw_object_name = str(plan_dict.get("name", "")).strip() or object_dir.name
        object_name = self._sanitize_identifier(raw_object_name, fallback=self._sanitize_identifier(object_dir.name))
        output_path = object_dir / f"{object_name}.urdf"

        if output_path.exists() and not dry_run:
            return URDFGenerationResult(
                status="skipped",
                object_dir=str(object_dir),
                object_name=object_name,
                plan_path=str(plan_path),
                output_path=str(output_path),
                reason="URDF file already exists.",
            )

        if not self._plan_requires_urdf(plan_dict):
            return URDFGenerationResult(
                status="skipped",
                object_dir=str(object_dir),
                object_name=object_name,
                plan_path=str(plan_path),
                output_path=str(output_path),
                reason="No part with is_movable=true in ObjectPlan.",
            )

        try:
            obj_files = self._collect_obj_files(object_dir)
        except Exception as e:
            return URDFGenerationResult(
                status="failed",
                object_dir=str(object_dir),
                object_name=object_name,
                plan_path=str(plan_path),
                output_path=str(output_path),
                error=str(e),
            )

        try:
            if self.use_mock:
                urdf_text = self._generate_mock_urdf(object_name, obj_files)
            else:
                messages = self._build_messages(
                    object_dir=object_dir,
                    object_name=object_name,
                    plan_dict=plan_dict,
                    obj_files=obj_files,
                    output_filename=output_path.name,
                )
                urdf_text = await self._generate_urdf_with_llm(messages)

            self._validate_urdf(urdf_text)

            if not dry_run:
                with output_path.open("w", encoding="utf-8") as f:
                    f.write(urdf_text.rstrip() + "\n")

            return URDFGenerationResult(
                status="generated",
                object_dir=str(object_dir),
                object_name=object_name,
                plan_path=str(plan_path),
                output_path=str(output_path),
                obj_files=obj_files,
            )
        except Exception as e:
            return URDFGenerationResult(
                status="failed",
                object_dir=str(object_dir),
                object_name=object_name,
                plan_path=str(plan_path),
                output_path=str(output_path),
                obj_files=obj_files,
                error=str(e),
            )

    async def run_batch(
        self,
        batch_dir: Path,
        *,
        object_plan_hint: Optional[str] = None,
        result_json_path: Optional[Path] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        batch_dir = batch_dir.resolve()
        if not batch_dir.exists() or not batch_dir.is_dir():
            raise FileNotFoundError(f"Batch directory does not exist: {batch_dir}")

        object_dirs = []
        for child in sorted(batch_dir.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if object_plan_hint:
                candidate = child / object_plan_hint
                if candidate.exists():
                    object_dirs.append(child)
                    continue
                if Path(object_plan_hint).is_absolute() and Path(object_plan_hint).exists():
                    object_dirs.append(child)
                    continue
            else:
                if any((child / name).exists() for name in DEFAULT_PLAN_CANDIDATES):
                    object_dirs.append(child)

        if not object_dirs:
            raise RuntimeError(
                f"No candidate object folders found in {batch_dir}. "
                f"Expected ObjectPlan file in immediate subfolders."
            )

        summary: Dict[str, Any] = {
            "batch_dir": str(batch_dir),
            "total": len(object_dirs),
            "generated": 0,
            "skipped": 0,
            "failed": 0,
            "details": [],
            "timestamp": datetime.now().isoformat(),
        }

        for idx, object_dir in enumerate(object_dirs, start=1):
            self.logger.info("[%d/%d] Processing %s", idx, len(object_dirs), object_dir.name)
            result = await self.process_object_dir(
                object_dir,
                object_plan_hint=object_plan_hint,
                dry_run=dry_run,
            )
            summary[result.status] += 1
            summary["details"].append(result.to_dict())

            if result.status == "generated":
                self.logger.info("Generated: %s", result.output_path)
            elif result.status == "skipped":
                self.logger.info("Skipped: %s (%s)", result.object_dir, result.reason)
            else:
                self.logger.error("Failed: %s (%s)", result.object_dir, result.error)

        if result_json_path is None:
            result_json_path = batch_dir / "urdf_batch_results.json"
        if not dry_run:
            with result_json_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.logger.info("Batch summary saved to: %s", result_json_path)
        else:
            self.logger.info("Dry-run mode: batch summary file is not written.")

        return summary


async def _run_single(args: argparse.Namespace) -> int:
    pipeline = URDFPipeline(
        config_path=args.config,
        prompt_path=args.prompt,
        llm_config_key=args.llm_config_key,
        use_mock=args.mock,
    )
    result = await pipeline.process_object_dir(
        Path(args.object_dir),
        object_plan_hint=args.object_plan,
        dry_run=args.dry_run,
    )

    if result.status == "generated":
        print(f"[GENERATED] {result.output_path}")
        return 0
    if result.status == "skipped":
        print(f"[SKIPPED] {result.object_dir} - {result.reason}")
        return 0
    print(f"[FAILED] {result.object_dir} - {result.error}")
    return 1


async def _run_batch(args: argparse.Namespace) -> int:
    pipeline = URDFPipeline(
        config_path=args.config,
        prompt_path=args.prompt,
        llm_config_key=args.llm_config_key,
        use_mock=args.mock,
    )
    summary = await pipeline.run_batch(
        Path(args.batch_dir),
        object_plan_hint=args.object_plan,
        result_json_path=Path(args.result_json) if args.result_json else None,
        dry_run=args.dry_run,
    )
    print(
        f"[SUMMARY] total={summary['total']} "
        f"generated={summary['generated']} skipped={summary['skipped']} failed={summary['failed']}"
    )
    return 0 if summary["failed"] == 0 else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="URDF generation pipeline (single/batch).")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    def add_shared_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--object-plan",
            default=None,
            help=(
                "ObjectPlan path hint. "
                "Single mode: absolute path or path relative to object dir. "
                "Batch mode: file name/path relative to each object dir."
            ),
        )
        p.add_argument(
            "--prompt",
            default=str(Path(__file__).parent / "prompts" / "urdf.md"),
            help="Prompt template path for URDF generation.",
        )
        p.add_argument("--config", "-c", default=None, help="Config file path.")
        p.add_argument(
            "--llm-config-key",
            default="constructor_llm",
            help="Model config key in YAML (default: constructor_llm).",
        )
        p.add_argument("--mock", action="store_true", help="Use mock URDF generator (no LLM calls).")
        p.add_argument("--dry-run", action="store_true", help="Validate and print result without writing files.")

    single = subparsers.add_parser("single", help="Generate URDF for one object directory.")
    single.add_argument("object_dir", help="Object directory path.")
    add_shared_args(single)

    batch = subparsers.add_parser("batch", help="Generate URDF for object directories in a parent folder.")
    batch.add_argument("batch_dir", help="Batch parent directory path.")
    batch.add_argument("--result-json", default=None, help="Batch summary output JSON path.")
    add_shared_args(batch)

    return parser


async def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "single":
        return await _run_single(args)
    if args.command == "batch":
        return await _run_batch(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
