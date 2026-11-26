#!/usr/bin/env python3
"""Summarize WorldEval batch scores from a scheduler summary JSONL file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


SCORE_FIELDS = {
    "three_d": ("three_d", "final_score_normalized"),
    "interaction": ("interaction", "score"),
    "combined": ("combined_score",),
    "worldolympiad2": ("worldolympiad2_score",),
    "continuous_action": ("continuous_action", "score"),
    "continuous_action_path": ("continuous_action", "path_score"),
    "continuous_action_velocity": ("continuous_action", "velocity_score"),
    "clip_interaction": ("clip_interaction", "summary", "semantic_adherence"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            row["_summary_line"] = line_number
            rows.append(row)
    return rows


def nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def empty_physical_counts() -> dict[str, int]:
    return {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "irrelevant": 0,
        "unknown": 0,
    }


def add_physical_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def physical_question_counts(physical: dict[str, Any]) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    counts = empty_physical_counts()
    by_question: dict[str, dict[str, int]] = {}
    results = physical.get("results")
    if not isinstance(results, list):
        counts["unknown"] += 1
        return counts, by_question

    for index, item in enumerate(results):
        if not isinstance(item, dict):
            counts["unknown"] += 1
            continue

        question_id = str(item.get("question_id") or f"question_{index}")
        question_counts = by_question.setdefault(question_id, empty_physical_counts())
        related = item.get("related")
        compliant = item.get("compliant")

        counts["total"] += 1
        question_counts["total"] += 1

        if related is False:
            bucket = "irrelevant"
        elif related is True and compliant is True:
            bucket = "correct"
        elif related is True and compliant is False:
            bucket = "incorrect"
        else:
            bucket = "unknown"

        counts[bucket] += 1
        question_counts[bucket] += 1
    return counts, by_question


def read_score_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing_output"
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except json.JSONDecodeError as exc:
        return None, f"invalid_json: {exc}"
    except OSError as exc:
        return None, f"read_error: {exc}"


def collect_case_rows(summary_rows: list[dict[str, Any]], *, include_non_ok: bool) -> list[dict[str, Any]]:
    case_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        status = str(row.get("status") or "")
        output = row.get("output")
        case = {
            "id": row.get("id") or row.get("_summary_line"),
            "status": status,
            "exit_code": row.get("exit_code"),
            "output": output,
            "error": None,
        }

        if status != "ok" and not include_non_ok:
            case["error"] = "not_ok"
            case_rows.append(case)
            continue
        if not output:
            case["error"] = "missing_output_path"
            case_rows.append(case)
            continue

        score_data, error = read_score_file(Path(str(output)))
        if error:
            case["error"] = error
            case_rows.append(case)
            continue

        assert score_data is not None
        for name, path in SCORE_FIELDS.items():
            case[f"{name}_score"] = as_float(nested_get(score_data, path))

        physical = score_data.get("physical")
        if isinstance(physical, dict):
            counts, by_question = physical_question_counts(physical)
            case["physical_total_question_count"] = counts["total"]
            case["physical_correct_question_count"] = counts["correct"]
            case["physical_incorrect_question_count"] = counts["incorrect"]
            case["physical_irrelevant_question_count"] = counts["irrelevant"]
            case["physical_unknown_question_count"] = counts["unknown"]
            case["_physical_by_question"] = by_question

        three_d = score_data.get("three_d")
        if isinstance(three_d, dict):
            case["three_d_raw_score"] = as_float(three_d.get("final_score_raw"))

        interaction = score_data.get("interaction")
        if isinstance(interaction, dict):
            case["interaction_raw_score"] = as_float(interaction.get("overall_raw"))

        continuous_action = score_data.get("continuous_action")
        if isinstance(continuous_action, dict):
            case["continuous_action_status"] = continuous_action.get("status")
            case["continuous_action_model"] = continuous_action.get("selected_model")
            case["continuous_action_benchmark_status"] = continuous_action.get(
                "benchmark_status"
            )
            evaluation = continuous_action.get("evaluation")
            if not isinstance(evaluation, dict):
                evaluation = {}
            metric_config = evaluation.get("metric_config")
            if not isinstance(metric_config, dict):
                metric_config = {}
            case["continuous_action_metric_version"] = (
                continuous_action.get("metric_version")
                or evaluation.get("metric_version")
                or metric_config.get("metric_version")
            )
            case["continuous_action_schema_version"] = (
                continuous_action.get("schema_version")
                or evaluation.get("schema_version")
            )
            case["continuous_action_metric_config_sha256"] = (
                continuous_action.get("metric_config_sha256")
                or evaluation.get("metric_config_sha256")
            )

        case_rows.append(case)
    return case_rows


def aggregate(case_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    for row in case_rows:
        status = str(row.get("status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
        error = row.get("error")
        if error:
            error_counts[str(error)] = error_counts.get(str(error), 0) + 1

    score_summary = {}
    for name in SCORE_FIELDS:
        values = [
            value
            for value in (as_float(row.get(f"{name}_score")) for row in case_rows)
            if value is not None
        ]
        score_summary[name] = score_stats(values)

    cae_status_counts: dict[str, int] = {}
    cae_metric_version_counts: dict[str, int] = {}
    cae_scores_by_metric_version: dict[str, list[float]] = {}
    for row in case_rows:
        status = str(row.get("continuous_action_status") or "missing")
        cae_status_counts[status] = cae_status_counts.get(status, 0) + 1
        metric_version = str(row.get("continuous_action_metric_version") or "missing")
        cae_metric_version_counts[metric_version] = (
            cae_metric_version_counts.get(metric_version, 0) + 1
        )
        score = as_float(row.get("continuous_action_score"))
        if score is not None:
            cae_scores_by_metric_version.setdefault(metric_version, []).append(score)

    physical_counts = empty_physical_counts()
    physical_by_question: dict[str, dict[str, int]] = {}
    for row in case_rows:
        add_physical_counts(
            physical_counts,
            {
                "total": int(row.get("physical_total_question_count") or 0),
                "correct": int(row.get("physical_correct_question_count") or 0),
                "incorrect": int(row.get("physical_incorrect_question_count") or 0),
                "irrelevant": int(row.get("physical_irrelevant_question_count") or 0),
                "unknown": int(row.get("physical_unknown_question_count") or 0),
            },
        )
        by_question = row.get("_physical_by_question")
        if not isinstance(by_question, dict):
            continue
        for question_id, counts in by_question.items():
            if not isinstance(counts, dict):
                continue
            target = physical_by_question.setdefault(str(question_id), empty_physical_counts())
            add_physical_counts(target, counts)

    return {
        "summary_items": len(summary_rows),
        "status_counts": status_counts,
        "error_counts": error_counts,
        "physical_questions": {
            "overall": physical_counts,
            "by_question": dict(sorted(physical_by_question.items())),
        },
        "scores": score_summary,
        "continuous_action_status_counts": cae_status_counts,
        "continuous_action_metric_version_counts": cae_metric_version_counts,
        "continuous_action_scores_by_metric_version": {
            version: score_stats(values)
            for version, values in sorted(cae_scores_by_metric_version.items())
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "status",
        "exit_code",
        "error",
        "combined_score",
        "worldolympiad2_score",
        "continuous_action_score",
        "continuous_action_path_score",
        "continuous_action_velocity_score",
        "continuous_action_status",
        "continuous_action_model",
        "continuous_action_benchmark_status",
        "continuous_action_metric_version",
        "continuous_action_schema_version",
        "continuous_action_metric_config_sha256",
        "three_d_score",
        "three_d_raw_score",
        "interaction_score",
        "interaction_raw_score",
        "clip_interaction_score",
        "physical_total_question_count",
        "physical_correct_question_count",
        "physical_incorrect_question_count",
        "physical_irrelevant_question_count",
        "physical_unknown_question_count",
        "output",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_report(result: dict[str, Any]) -> None:
    print(f"Items in scheduler summary: {result['summary_items']}")
    print(f"Status counts: {json.dumps(result['status_counts'], ensure_ascii=False, sort_keys=True)}")
    if result["error_counts"]:
        print(f"Errors: {json.dumps(result['error_counts'], ensure_ascii=False, sort_keys=True)}")
    print(
        "Continuous action applicability: "
        f"{json.dumps(result['continuous_action_status_counts'], ensure_ascii=False, sort_keys=True)}"
    )
    print(
        "Continuous action metric versions: "
        f"{json.dumps(result['continuous_action_metric_version_counts'], ensure_ascii=False, sort_keys=True)}"
    )
    if len(result["continuous_action_scores_by_metric_version"]) > 1:
        print(
            "CAUTION: CAE scores from different metric versions are reported separately; "
            "do not compare the pooled CAE statistic as one benchmark."
        )

    physical = result["physical_questions"]["overall"]
    print("\nPhysical questions:")
    print(
        "  "
        f"total={physical['total']} "
        f"correct={physical['correct']} "
        f"incorrect={physical['incorrect']} "
        f"irrelevant={physical['irrelevant']} "
        f"unknown={physical['unknown']}"
    )

    print("\nScores (0-1 unless noted):")
    for name in (
        "worldolympiad2",
        "combined",
        "continuous_action",
        "continuous_action_path",
        "continuous_action_velocity",
        "three_d",
        "interaction",
        "clip_interaction",
    ):
        stats = result["scores"][name]
        label = name.replace("_", " ")
        if stats["count"] == 0:
            print(f"  {label}: n=0")
            continue
        print(
            f"  {label}: "
            f"n={stats['count']} mean={stats['mean']:.6f} "
            f"median={stats['median']:.6f} min={stats['min']:.6f} max={stats['max']:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read a batch_scheduler summary JSONL and aggregate scores from the "
            "per-case judge JSON files referenced by each row's output field."
        )
    )
    parser.add_argument("summary", help="Path to batch_scheduler summary.jsonl")
    parser.add_argument("--output-json", help="Optional path for aggregate summary JSON")
    parser.add_argument("--output-csv", help="Optional path for per-case CSV")
    parser.add_argument(
        "--include-non-ok",
        action="store_true",
        help="Try to read score files even for rows whose scheduler status is not ok.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary_rows = load_jsonl(summary_path)
    case_rows = collect_case_rows(summary_rows, include_non_ok=args.include_non_ok)
    result = aggregate(case_rows, summary_rows)
    result["source_summary"] = str(summary_path)

    print_report(result)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote aggregate JSON: {output_json}")

    if args.output_csv:
        output_csv = Path(args.output_csv)
        write_csv(output_csv, case_rows)
        print(f"Wrote per-case CSV: {output_csv}")


if __name__ == "__main__":
    main()
