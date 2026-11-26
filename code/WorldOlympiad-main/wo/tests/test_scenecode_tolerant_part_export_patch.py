from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "deployment/ziplab5090/patches/scenecode_tolerant_part_export.patch"


def test_tolerant_part_export_patch_applies_to_current_construction_flow(tmp_path):
    """Keep partial construction recoverable while hard exceptions remain failures."""
    target = tmp_path / "scenecode/code_object/agent_framework/pipeline.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        """                        if construct_success or not enable_part_construct:
                            stages_completed.append("construction")
                            self.logger.info(f"Construction completed: {len(construction_result['part_results'])} parts")
                        else:
                            stages_failed.append("construction")
                            self.logger.warning(f"Construction partially failed: {construction_result.get('failed_parts', [])}")
                        
                    except Exception as e:
                        self.logger.error(f"Construction failed: {e}")
                        stages_failed.append("construction")
"""
    )

    applied = subprocess.run(
        ["patch", "--forward", "--batch", "-p1", "-d", str(tmp_path)],
        input=PATCH.read_text(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert applied.returncode == 0, applied.stdout + applied.stderr
    result = target.read_text()
    assert '"Construction incomplete; exporting the assembled parts' in result
    assert 'stages_failed.append("construction")' in result  # thrown exceptions still fail
    assert 'stages_failed.append("construction")' not in result.split("except Exception", 1)[0]


def test_tolerant_part_export_patch_is_valid_unified_diff():
    checked = subprocess.run(
        ["git", "apply", "--numstat", str(PATCH)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
