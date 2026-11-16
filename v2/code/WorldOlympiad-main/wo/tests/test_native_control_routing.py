import pytest
from wo.agent.stages import STAGES, MODELS, CONTROL_MODELS


@pytest.fixture(autouse=True)
def experimental_adapter(monkeypatch):
    monkeypatch.setitem(CONTROL_MODELS, 'experimental', 'wo.models.experimental')


def test_fastwam_native_review_enables_bound_matrix():
    args = dict(run='bench_v1', case='bottle', model='fastwam', window_s=60, seed=0)
    STAGES['episode'].validate(args)
    cmd = STAGES['episode'].command(args, 7)
    assert MODELS['fastwam'] == 'wo.models.fastwam'
    assert 'wo.models.fastwam' in cmd
    assert 'wo.agent.episode_stage' in cmd


def test_experimental_native_control_is_bound_and_not_matrix_registered():
    args = dict(run='bench_v1', case='bottle', model='experimental', control_only=True, window_s=60, seed=0)
    STAGES['episode'].validate(args)
    cmd = STAGES['episode'].command(args, 7)
    assert 'experimental' not in MODELS
    assert 'wo.models.experimental' in cmd
    assert 'wo.agent.episode_stage' in cmd  # retains all-reference gate and binding
    assert cmd[cmd.index('--window-s') + 1] == '60'


@pytest.mark.parametrize('flag', [None, False, 'true', 1])
def test_experimental_adapter_cannot_be_scheduled_as_matrix(flag):
    args = dict(run='bench_v1', case='bottle', model='experimental', control_only=flag)
    with pytest.raises(ValueError, match='control_only'):
        STAGES['episode'].validate(args)
    with pytest.raises(ValueError, match='control_only'):
        STAGES['episode'].command(args, 7)
