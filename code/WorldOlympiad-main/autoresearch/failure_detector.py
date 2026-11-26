"""Turn official WorldOlympiad scores into an auditable one-pass data-gap report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .io import write_json


DATA_RECIPES = {
    "physical": {
        "gap": "physical evolution without a reliable sub-dimension diagnosis",
        "data": "balanced mechanics, thermal-transition, and material-response videos with state labels",
        "vary": ["phenomenon family", "material", "time scale", "viewpoint"],
    },
    "physical.mechanics": {
        "gap": "rigid-body motion and contact physics",
        "data": "controlled drop, rolling, sliding, collision, tipping, and stacking scenes",
        "vary": ["mass", "friction", "restitution", "speed", "contact complexity"],
    },
    "physical.thermotics": {
        "gap": "phase and thermal-state transitions",
        "data": "melting, freezing, evaporation, condensation, and sublimation videos",
        "vary": ["material", "temperature trend", "transition rate", "occlusion"],
    },
    "physical.material": {
        "gap": "material-specific appearance and response",
        "data": "deformation, hardness, dissolution, combustion, and color-mixing videos",
        "vary": ["material class", "force", "duration", "viewpoint"],
    },
    "geometry": {
        "gap": "3D structure and camera-consistent appearance",
        "data": "multi-view static scenes and camera-motion sequences with controlled occlusion",
        "vary": ["camera path", "view baseline", "occlusion", "scene depth", "texture frequency"],
    },
    "interaction": {
        "gap": "instruction following over time",
        "data": "multi-chunk action sequences with explicit transitions and long-horizon goals",
        "vary": ["action order", "chunk duration", "transition", "distractor", "horizon"],
    },
    "continuous_action.path": {
        "gap": "task-relevant object trajectory shape",
        "data": "paired reference/imagination motions with direction, curvature, and endpoint variation",
        "vary": ["direction", "path curvature", "endpoint", "obstacle", "camera"],
    },
    "continuous_action.velocity": {
        "gap": "motion speed and timing",
        "data": "paired motions with controlled speed, acceleration, pause, and impact timing",
        "vary": ["speed", "acceleration", "trigger time", "pause", "contact timing"],
    },
}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _case_scores(score: dict[str, Any]) -> dict[str, float | None]:
    physical = score.get("physical") if isinstance(score.get("physical"), dict) else {}
    interaction = score.get("interaction") if isinstance(score.get("interaction"), dict) else {}
    geometry = score.get("three_d") if isinstance(score.get("three_d"), dict) else {}
    action = (
        score.get("continuous_action")
        if isinstance(score.get("continuous_action"), dict)
        else {}
    )
    dimensions = (
        physical.get("dimension_scores")
        if isinstance(physical.get("dimension_scores"), dict)
        else {}
    )
    applicable_action = action.get("status") == "applicable"
    return {
        "physical": _number(physical.get("score")),
        "physical.mechanics": _number(dimensions.get("mechanics")),
        "physical.thermotics": _number(dimensions.get("thermotics")),
        "physical.material": _number(dimensions.get("material")),
        "geometry": _number(geometry.get("final_score_normalized")),
        "interaction": _number(interaction.get("score")),
        "continuous_action.path": _number(action.get("path_score"))
        if applicable_action
        else None,
        "continuous_action.velocity": _number(action.get("velocity_score"))
        if applicable_action
        else None,
    }


def detect_failures(
    score_paths: list[str | Path], *, diagnostic_threshold: float = 0.7
) -> dict[str, Any]:
    """Rank observed weaknesses and map them to bounded data recommendations."""

    if not 0.0 <= diagnostic_threshold <= 1.0:
        raise ValueError("diagnostic_threshold must be in [0, 1]")
    cases = []
    evidence: dict[str, list[tuple[str, float]]] = {}
    for raw_path in score_paths:
        path = Path(raw_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores = _case_scores(payload)
        case_id = path.parent.name
        cases.append({"case_id": case_id, "score_path": str(path), "scores": scores})
        for dimension, value in scores.items():
            if value is not None:
                evidence.setdefault(dimension, []).append((case_id, value))

    findings = []
    for dimension, rows in evidence.items():
        average = mean(value for _, value in rows)
        if average >= diagnostic_threshold:
            continue
        recipe_key = dimension
        if dimension == "physical" and any(
            key in evidence
            for key in ("physical.mechanics", "physical.thermotics", "physical.material")
        ):
            # Prefer official sub-dimension evidence when it exists.
            continue
        recipe = DATA_RECIPES[recipe_key]
        findings.append(
            {
                "dimension": dimension,
                "mean_score": round(average, 6),
                "diagnostic_threshold": diagnostic_threshold,
                "severity": round(diagnostic_threshold - average, 6),
                "evidence": [
                    {"case_id": case_id, "score": score}
                    for case_id, score in sorted(rows, key=lambda row: row[1])
                ],
                "data_gap": recipe["gap"],
                "recommended_data": recipe["data"],
                "coverage_axes": recipe["vary"],
            }
        )
    findings.sort(key=lambda item: (-item["severity"], item["dimension"]))

    unobserved = [
        dimension for dimension in DATA_RECIPES if dimension not in evidence
    ]
    return {
        "schema_version": "worldolympiad_failure_report.v1",
        "report_scope": "one_pass_diagnosis_and_data_recommendation",
        "automation_boundary": {
            "generates_new_data": False,
            "trains_or_finetunes_model": False,
            "starts_research_outer_loop": False,
        },
        "diagnostic_threshold": diagnostic_threshold,
        "case_count": len(cases),
        "cases": cases,
        "findings": findings,
        "unobserved_dimensions": unobserved,
        "summary": {
            "finding_count": len(findings),
            "highest_priority_gap": findings[0]["dimension"] if findings else None,
            "note": "Thresholds are diagnostic policy, not official WorldOlympiad pass/fail gates.",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the deterministic JSON diagnosis as a concise human report."""

    lines = [
        "# World Model Failure / Data Gap Report",
        "",
        f"评测 case 数：{report['case_count']}。诊断阈值：{report['diagnostic_threshold']:.2f}。",
        "该阈值只用于缺陷排序，不是 WorldOlympiad 官方 pass/fail 门槛。",
        "",
        "## 缺陷排序",
        "",
    ]
    findings = report["findings"]
    if findings:
        lines.extend(
            [
                "| 优先级 | 维度 | 平均分 | 数据缺口 | 建议数据 |",
                "|---:|---|---:|---|---|",
            ]
        )
        for priority, item in enumerate(findings, start=1):
            clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {priority} | `{clean(item['dimension'])}` | {item['mean_score']:.3f} | "
                f"{clean(item['data_gap'])} | {clean(item['recommended_data'])} |"
            )
    else:
        lines.append("没有已观测维度低于当前诊断策略。")

    lines.extend(["", "## 证据与覆盖建议", ""])
    for item in findings:
        cases = ", ".join(
            f"{row['case_id']}={row['score']:.3f}" for row in item["evidence"]
        )
        axes = "、".join(item["coverage_axes"])
        lines.extend(
            [
                f"### {item['dimension']}",
                "",
                f"- 证据：{cases}",
                f"- 建议覆盖轴：{axes}",
                f"- 建议数据：{item['recommended_data']}",
                "",
            ]
        )
    if report["unobserved_dimensions"]:
        lines.extend(
            [
                "## 未观测维度",
                "",
                "以下维度没有有效分数，因此不判为模型失败："
                + "、".join(report["unobserved_dimensions"]),
                "",
            ]
        )
    lines.extend(
        [
            "## 自动化边界",
            "",
            "本报告只诊断缺陷并建议数据方向；不会生成数据、训练或微调模型，也不会启动研究外环。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score", nargs="+")
    parser.add_argument("--diagnostic-threshold", type=float, default=0.7)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output")
    args = parser.parse_args()
    report = detect_failures(
        args.score, diagnostic_threshold=args.diagnostic_threshold
    )
    output = write_json(args.output, report)
    if args.markdown_output:
        markdown = Path(args.markdown_output).resolve()
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(report), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
