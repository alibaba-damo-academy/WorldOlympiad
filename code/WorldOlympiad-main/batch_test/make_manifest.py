#!/usr/bin/env python3
"""Create a JSONL manifest for batch scoring directories."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from batch_test.test_bench import PIPELINE_BY_ALIAS, PIPELINE_SPECS, sanitize_filename_part
except Exception:
    @dataclass(frozen=True)
    class PipelineSpec:
        name: str
        output_prefix: str
        aliases: tuple[str, ...]

    PIPELINE_SPECS = (
        PipelineSpec("cosmos-predict", "cosmos", ("cosmos", "cosmos-predict", "cosmos_predict", "cosmos-predict2p5", "cosmos_predict2p5")),
        PipelineSpec("hunyuan-gamecraft", "hunyuan_gamecraft", ("hunyuan-gamecraft", "hunyuan_gamecraft", "gamecraft")),
        PipelineSpec("hunyuan-worldplay", "hunyuan_worldplay", ("hunyuan-worldplay", "hunyuan_worldplay", "worldplay")),
        PipelineSpec("lingbot-world", "lingbot_world", ("lingbot-world", "lingbot_world", "lingbot")),
        PipelineSpec("longlive", "longlive", ("longlive",)),
        PipelineSpec("matrix-game2", "matrix_game2", ("matrix-game2", "matrix_game2", "matrix", "matrix-game-2", "matrix_game_2")),
        PipelineSpec("rolling-forcing", "rolling_forcing", ("rolling-forcing", "rolling_forcing")),
        PipelineSpec("wow", "wow", ("wow",)),
        PipelineSpec("yume1p5", "yume1p5", ("yume1p5", "yume-1p5", "yume_1p5", "yume")),
    )
    PIPELINE_BY_ALIAS = {
        alias: spec
        for spec in PIPELINE_SPECS
        for alias in (spec.name, *spec.aliases)
    }

    def sanitize_filename_part(value: str) -> str:
        import os
        import re

        value = value.strip().replace(os.sep, "_")
        if os.altsep:
            value = value.replace(os.altsep, "_")
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
        value = value.strip("._-")
        return value or "sample"


def find_one(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


from continuous_action.current_metric import CURRENT_MANIFEST_NAME
ACTIVE_CAE_MANIFEST = CURRENT_MANIFEST_NAME
ACTIVE_CAE_MANIFESTS = (CURRENT_MANIFEST_NAME,)


def resolve_cae_manifest(case_dir: Path, pattern: str | None) -> tuple[Path | None, str | None]:
    """Discover only the current CAE sidecar, never retired versions."""
    if pattern is None:
        return None, None
    if pattern != CURRENT_MANIFEST_NAME:
        return None, "only the current continuous_action_manifest.json is supported"
    case_path = case_dir / "worldolympiad_case.json"
    if case_path.is_file():
        try:
            declared = json.loads(case_path.read_text()).get("continuous_action_manifest")
        except (OSError, ValueError) as exc:
            return None, f"invalid worldolympiad_case.json: {exc}"
        if declared is not None and declared != CURRENT_MANIFEST_NAME:
            return None, "retired or invalid CAE pointer; regenerate the current manifest"
    candidate = case_dir / CURRENT_MANIFEST_NAME
    if candidate.is_file():
        if candidate.resolve().parent != case_dir.resolve():
            return None, "CAE manifest must resolve inside the case directory"
        return candidate, None
    return None, None


def build_item(
    case_dir: Path,
    *,
    output_name: str | None,
    output_name_template: str,
    output_prefix: str | None,
    gen_pattern: str,
    ref_pattern: str,
    chunk_pattern: str | None,
    cae_pattern: str | None,
    cae_model: str | None,
    pipeline: str | None,
    version_cae_output: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    gen_video = find_one(case_dir, gen_pattern)
    ref_video = find_one(case_dir, ref_pattern)
    prompt_json = case_dir / "prompt.json"
    chunk_json = None
    if gen_video is not None:
        exact_chunk_json = case_dir / f"{gen_video.stem}_chunk_timestamps.json"
        if exact_chunk_json.exists():
            chunk_json = exact_chunk_json
    if chunk_json is None and chunk_pattern:
        chunk_json = find_one(case_dir, chunk_pattern)
    cae_manifest, cae_error = resolve_cae_manifest(case_dir, cae_pattern)

    missing = []
    if not gen_video:
        missing.append(f"generated video matching {gen_pattern!r}")
    if not ref_video:
        missing.append(f"reference video matching {ref_pattern!r}")
    if not prompt_json.exists():
        missing.append("prompt.json")
    if not chunk_json:
        missing.append(
            f"chunk timestamps matching {chunk_pattern!r}"
            if chunk_pattern
            else "chunk timestamps matching generated video stem"
        )
    if cae_error:
        missing.append(cae_error)
    if missing:
        return None, "; ".join(missing)

    score_output_name = output_name or output_name_template.format(
        id=case_dir.name,
        pipeline=pipeline or "",
        prefix=output_prefix or "",
    )
    if cae_manifest is not None and version_cae_output:
        try:
            cae_payload = json.loads(cae_manifest.read_text(encoding="utf-8"))
            metric_version = str(cae_payload["metric_version"])
        except (KeyError, OSError, json.JSONDecodeError) as exc:
            return None, f"cannot version score output from CAE manifest: {exc}"
        from continuous_action.current_metric import CURRENT_METRIC_VERSION
        if metric_version != CURRENT_METRIC_VERSION:
            return None, "retired CAE metric"
        suffix = "_cae"
        output_path = Path(score_output_name)
        if not output_path.stem.endswith(suffix):
            score_output_name = str(
                output_path.with_name(
                    f"{output_path.stem}{suffix}{output_path.suffix}"
                )
            )
    item = {
        "id": case_dir.name,
        "video": str(gen_video),
        "gt_video": str(ref_video),
        "prompt_json": str(prompt_json),
        "chunk_json": str(chunk_json),
        "output": str(case_dir / score_output_name),
    }
    if pipeline:
        item["pipeline"] = pipeline
    if output_prefix:
        item["pipeline_output_prefix"] = output_prefix
    if cae_manifest is not None:
        item["cae_manifest"] = str(cae_manifest)
        if cae_model:
            item["cae_model"] = cae_model
    return item, None


def resolve_pipeline(pipeline: str | None, pipeline_output_prefix: str | None) -> tuple[str | None, str | None]:
    if not pipeline and not pipeline_output_prefix:
        return None, None
    if pipeline:
        try:
            spec = PIPELINE_BY_ALIAS[pipeline]
        except KeyError as exc:
            valid = ", ".join(sorted(PIPELINE_BY_ALIAS))
            raise ValueError(f"Unknown pipeline alias {pipeline!r}. Valid aliases: {valid}") from exc
        pipeline_name = spec.name
        output_prefix = pipeline_output_prefix or spec.output_prefix
    else:
        pipeline_name = None
        output_prefix = pipeline_output_prefix
    return pipeline_name, sanitize_filename_part(output_prefix)


def default_patterns(output_prefix: str | None, gen_pattern: str | None, chunk_pattern: str | None) -> tuple[str, str | None]:
    if gen_pattern:
        resolved_gen_pattern = gen_pattern
    elif output_prefix:
        resolved_gen_pattern = f"{output_prefix}_gen_*.mp4"
    else:
        resolved_gen_pattern = "matrix_game2_gen_*.mp4"

    if chunk_pattern:
        resolved_chunk_pattern = chunk_pattern
    elif output_prefix:
        resolved_chunk_pattern = f"{output_prefix}_gen_*_chunk_timestamps.json"
    elif gen_pattern:
        resolved_chunk_pattern = None
    else:
        resolved_chunk_pattern = "matrix_game2_gen_*_chunk_timestamps.json"
    return resolved_gen_pattern, resolved_chunk_pattern


def default_output_name_template(output_prefix: str | None, output_name_template: str | None) -> str:
    if output_name_template:
        return output_name_template
    if output_prefix:
        return "{prefix}_judge_{id}.json"
    return "score_physical_interaction_3d.json"


def print_pipeline_table() -> None:
    print("Supported pipelines from batch_test/test_bench.py:")
    for spec in PIPELINE_SPECS:
        aliases = ", ".join(spec.aliases)
        print(f"  {spec.name}: output_prefix={spec.output_prefix}; aliases={aliases}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create batch JSONL manifest from case directories")
    parser.add_argument("--root", help="Directory containing per-video case directories")
    parser.add_argument("--output", help="Output JSONL manifest path")
    parser.add_argument("--list-pipelines", action="store_true", help="Print pipeline aliases from batch_test/test_bench.py and exit")
    parser.add_argument(
        "--pipeline",
        choices=sorted(PIPELINE_BY_ALIAS),
        help=(
            "Pipeline alias from batch_test/test_bench.py / test_bench_pool_runner.py, "
            "e.g. cosmos-predict, longlive, matrix-game2. The alias resolves to the pipeline output_prefix."
        ),
    )
    parser.add_argument(
        "--pipeline-output-prefix",
        help=(
            "Override the resolved output prefix, matching test_bench.py --pipeline-output-prefix. "
            "Use this if generation used a custom prefix."
        ),
    )
    parser.add_argument(
        "--gen-pattern",
        help="Generated video glob inside each case directory. Overrides --pipeline. Example: 'cosmos_predict_*.mp4'.",
    )
    parser.add_argument("--ref-pattern", default="ref_*.mp4", help="Reference video glob inside each case directory")
    parser.add_argument(
        "--chunk-pattern",
        help=(
            "Chunk timestamp glob inside each case directory. If omitted, the script first tries "
            "<generated-video-stem>_chunk_timestamps.json."
        ),
    )
    parser.add_argument(
        "--cae-pattern",
        default=ACTIVE_CAE_MANIFEST,
        help=(
            "Optional active-formal CAE manifest inside each case directory. In default mode, "
            "worldolympiad_case.json must point to the exact v13 sidecar. Pass an explicit "
            "non-v13 pattern only for historical inspection. Missing files are marked not_applicable."
        ),
    )
    parser.add_argument(
        "--cae-model",
        help="Optional prediction key selected from matching CAE manifests",
    )
    parser.add_argument(
        "--output-name",
        help=(
            "Static per-case score JSON filename. If set, every case uses this exact filename. "
            "Usually prefer --output-name-template for pipeline-specific output names."
        ),
    )
    parser.add_argument(
        "--output-name-template",
        help=(
            "Per-case score JSON filename template. Available fields: {id}, {pipeline}, {prefix}. "
            "Defaults to {prefix}_judge_{id}.json when a pipeline/output prefix is set."
        ),
    )
    parser.add_argument(
        "--print-skipped",
        action="store_true",
        help="Print skipped case directories and the missing inputs.",
    )
    args = parser.parse_args()

    if args.list_pipelines:
        print_pipeline_table()
        return
    if not args.root:
        parser.error("--root is required unless --list-pipelines is used")
    if not args.output:
        parser.error("--output is required unless --list-pipelines is used")

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    pipeline_name, output_prefix = resolve_pipeline(args.pipeline, args.pipeline_output_prefix)
    gen_pattern, chunk_pattern = default_patterns(output_prefix, args.gen_pattern, args.chunk_pattern)
    output_name_template = default_output_name_template(output_prefix, args.output_name_template)

    items = []
    skipped = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        item, reason = build_item(
            case_dir,
            output_name=args.output_name,
            output_name_template=output_name_template,
            output_prefix=output_prefix,
            gen_pattern=gen_pattern,
            ref_pattern=args.ref_pattern,
            chunk_pattern=chunk_pattern,
            cae_pattern=args.cae_pattern,
            cae_model=args.cae_model,
            pipeline=pipeline_name,
            version_cae_output=(
                args.output_name is None and args.output_name_template is None
            ),
        )
        if item is None:
            skipped.append((str(case_dir), reason))
            continue
        items.append(item)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} item(s) to {output_path}")
    if args.pipeline:
        print(f"Pipeline: {pipeline_name}")
    if output_prefix:
        print(f"Pipeline output prefix: {output_prefix}")
    print(f"Generated video pattern: {gen_pattern}")
    print(f"Chunk timestamp pattern: {chunk_pattern or '<generated-video-stem>_chunk_timestamps.json'}")
    print(f"Optional CAE manifest pattern: {args.cae_pattern or '<disabled>'}")
    output_display = args.output_name or output_name_template
    if args.output_name is None and args.output_name_template is None:
        output_display += " (resolved CAE cases append their metric suffix)"
    print(f"Per-case score output name: {output_display}")
    if skipped:
        print(f"Skipped {len(skipped)} incomplete case directorie(s).")
        if args.print_skipped:
            for case_dir, reason in skipped:
                print(f"  {case_dir}: {reason}")


if __name__ == "__main__":
    main()
