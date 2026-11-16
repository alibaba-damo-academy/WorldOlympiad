from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "deployment/ziplab5090/patches/scenecode_lossless_manipuland_requirements.patch"


def test_lossless_manipuland_patch_is_well_formed_unified_diff():
    """Reject malformed hunk counts before runtime staging reaches SceneCode."""
    checked = subprocess.run(
        ["git", "apply", "--numstat", str(PATCH)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert checked.stdout.strip().endswith(
        "12\t0\tscenecode/prompts/data/manipuland_agent/analyze_furniture_for_placement.yaml"
    )


def test_lossless_patch_avoids_brittle_downstream_checklist_context(tmp_path):
    """The required semantic rules must apply despite unrelated upstream prompt edits."""
    source = tmp_path / "scenecode/prompts/data/manipuland_agent/analyze_furniture_for_placement.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n" * 39
        + "  3. Mark items as REQUIRED vs optional for each furniture piece\n"
        '  4. Extract style constraints that apply to all surfaces (e.g., "minimalist", "cluttered")\n'
        "\n"
        "  **Extraction checklist:**\n"
        '  1. **Explicit object mentions**: "desk with laptop" → laptop REQUIRED on desk\n'
        '  2. **Distribution across furniture**: "laptop on one table, books on the other" →\n'
        "     assign laptop to table_0, books to table_1 (or reason about which fits better)\n"
        "  **New upstream section:** unrelated additions\n"
        "  **Constraint Levels:**\n"
    )
    applied = subprocess.run(
        ["patch", "--forward", "--batch", "-p1", "-d", str(tmp_path)],
        input=PATCH.read_text(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert applied.returncode == 0, applied.stdout + applied.stderr
    result = source.read_text()
    assert "**Lossless requirement forwarding:**" in result
    assert "Copy every cross-object geometry sentence" in result
