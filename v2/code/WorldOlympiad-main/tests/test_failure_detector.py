from __future__ import annotations

import json
from pathlib import Path

from autoresearch.failure_detector import detect_failures, render_markdown


def _score(path: Path, *, geometry: float, interaction: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "physical": {
                    "score": 0.8,
                    "dimension_scores": {
                        "mechanics": 0.55,
                        "thermotics": None,
                        "material": None,
                    },
                },
                "three_d": {"final_score_normalized": geometry},
                "interaction": {"score": interaction},
                "continuous_action": {
                    "status": "applicable",
                    "path_score": 0.5,
                    "velocity_score": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_failure_detector_ranks_gaps_and_stops_before_training(tmp_path: Path) -> None:
    paths = [
        _score(tmp_path / "ball" / "score.json", geometry=0.4, interaction=0.85),
        _score(tmp_path / "cup" / "score.json", geometry=0.5, interaction=0.8),
    ]
    report = detect_failures(paths, diagnostic_threshold=0.7)
    dimensions = [item["dimension"] for item in report["findings"]]
    assert dimensions[0] == "geometry"
    assert "physical.mechanics" in dimensions
    assert "continuous_action.path" in dimensions
    assert "interaction" not in dimensions
    assert "physical.thermotics" in report["unobserved_dimensions"]
    assert report["automation_boundary"] == {
        "generates_new_data": False,
        "trains_or_finetunes_model": False,
        "starts_research_outer_loop": False,
    }
    markdown = render_markdown(report)
    assert "# World Model Failure / Data Gap Report" in markdown
    assert "不会生成数据、训练或微调模型" in markdown
