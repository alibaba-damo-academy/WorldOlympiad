import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoresearch.robotics_reference_materializer import (
    _certify_published_graph_compatibility,
)


def test_old_reference_reuses_only_unchanged_task_and_physically_met_constraint(
    tmp_path: Path,
) -> None:
    final = tmp_path / "bundle"
    provenance = final / "agentic_provenance"
    provenance.mkdir(parents=True)
    task = tmp_path / "task.json"
    task.write_text('{"instruction":"Place the kettle upright."}')
    (final / "task_instruction.json").write_bytes(task.read_bytes())
    old = {
        "case_id": "case", "schema_version": "plan.v1", "status": "static_plan_validated",
        "requested_wams": ["cosmos3_nano_droid"],
        "controller_program": [{"operator": "place", "constraints": {}}],
        "success_program": [{"predicate": "on_region"}],
    }
    new = {
        **old,
        "controller_program": [{"operator": "place", "capability_status": "native_registered",
                                "constraints": {"terminal_upright": True,
                                                "terminal_upright_axis_world": [0, 0, 1],
                                                "terminal_upright_min_cosine": 0.99}}],
        "success_program": [{"predicate": "on_region", "evaluator_status": "registered",
                             "terminal_upright": True,
                             "terminal_upright_axis_world": [0, 0, 1],
                             "terminal_upright_min_cosine": 0.99}],
    }
    old_path = provenance / "task_graph.json"
    new_path = tmp_path / "new_graph.json"
    old_path.write_text(json.dumps(old))
    new_path.write_text(json.dumps(new))
    trajectory = {"frames": [{"cup": {
        "orientation_xyzw": [0, 0, 0, 1]
    }}]}
    (final / "trajectory.json").write_text(json.dumps(trajectory))
    source = {"plan_path": new_path, "task_path": task,
              "task": SimpleNamespace(case_id="case"),
              "terminal_alignment_axis_local": [0, 0, 1]}
    receipt = _certify_published_graph_compatibility(
        final=final, source=source, output_root=tmp_path, manifest_sha256="a" * 64
    )
    assert json.loads(receipt.read_text())["added_upright_evidence"][
        "terminal_upright_cosine"] == pytest.approx(1.0)
    new["controller_program"][0]["operator"] = "strike"
    new_path.write_text(json.dumps(new))
    with pytest.raises(ValueError, match="changes controller_program semantics"):
        _certify_published_graph_compatibility(
            final=final, source=source, output_root=tmp_path,
            manifest_sha256="a" * 64,
        )
