"""Declined asset recovery attempts must remain auditable."""
import ast
from pathlib import Path


def test_declined_recoveries_are_recorded_without_recertification():
    path = Path(__file__).resolve().parents[2] / "autoresearch/robotics_scenecode_case_generator.py"
    tree = ast.parse(path.read_text())
    generate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_case")
    for report in ("recovery_report", "separation_report"):
        append = f"recovery_reports.append({report})"
        guards = [n for n in ast.walk(generate) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == f"{report}['changed']"]
        assert len(guards) == 1
        assert not any(ast.unparse(n) == append for n in guards[0].body)
        assert any(isinstance(n, ast.Expr) and ast.unparse(n) == append
                   for n in ast.walk(generate))
        assert "certify_task_inventory" in ast.unparse(guards[0])


def test_inventory_failure_can_arbitrate_complete_vs_partial_recovery():
    path = Path(__file__).resolve().parents[2] / "autoresearch/robotics_scenecode_case_generator.py"
    source = path.read_text()
    assert "prior.get(\"recoverable_candidate\", \"\")" in source
    assert "Selected evidence-backed partial recovery" in source
    assert "recovery.rename(archived)" in source
