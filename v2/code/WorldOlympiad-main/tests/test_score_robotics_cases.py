from pathlib import Path


def test_unbound_score_is_archived_before_recomputation():
    text = Path('scripts/score_robotics_cases.py').read_text()
    assert 'def _archive_uncommitted_score' in text
    assert 'if receipt.exists()' in text
    assert 'if score.exists():' in text
    assert 'shutil.move(str(score)' in text
    assert 'score.unlink()' not in text
