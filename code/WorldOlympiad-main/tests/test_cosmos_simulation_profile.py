import json
from pathlib import Path

import pytest

from autoresearch.cosmos_simulation_profile import read_simulation_profile


def test_default_profile_matches_original_robolab_clock_and_gpu():
    path = Path(__file__).resolve().parents[1]/'deployment/seetacloud/robolab_simulation_profile.json'
    values = read_simulation_profile(path)
    assert values['device'] == 'cuda:0'
    assert values['dt'] * values['render_interval'] == pytest.approx(1/15)
    assert values['physx']['max_position_iteration_count'] == 32


@pytest.mark.parametrize('field,value', [('dt', 1/160), ('render_interval', 6)])
def test_profile_rejects_accidental_twenty_hz_control(tmp_path, field, value):
    data = {'sim': {'device': 'cuda:0', 'dt': 1/120, 'render_interval': 8}}
    data['sim'][field] = value
    path = tmp_path/'profile.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='15 Hz'):
        read_simulation_profile(path)
