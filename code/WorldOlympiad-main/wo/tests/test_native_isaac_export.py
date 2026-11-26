from pathlib import Path
import pytest
from wo.models.native_isaac_export import observation_paths


def test_native_evidence_requires_every_executed_frame(tmp_path):
    for i in (0, 2):
        (tmp_path / f'step_{i:06d}_left_image.png').write_bytes(b'rgb')
    with pytest.raises(FileNotFoundError, match='step_000001'):
        observation_paths(tmp_path, 3, 'left')
    (tmp_path / 'step_000001_left_image.png').write_bytes(b'rgb')
    assert [p.name for p in observation_paths(tmp_path, 3, 'left')] == [
        f'step_{i:06d}_left_image.png' for i in range(3)]


def test_gr00t_records_after_execution_and_uses_native_export():
    root = Path(__file__).resolve().parents[2]
    runner = (root / 'scripts/run_cosmos3_worldir_isaac_closed_loop.py').read_text()
    assert "frames.append(frame)\n            if ARGS.policy_model == 'gr00t':" in runner
    adapter = (root / 'wo/models/gr00t.py').read_text()
    assert 'wo.models.native_isaac_export' in adapter
    assert 'c._export(' not in adapter
