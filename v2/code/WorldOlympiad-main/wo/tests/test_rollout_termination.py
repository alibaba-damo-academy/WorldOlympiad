from wo.agent.stages import _episode_window_args
from wo.models.rollout_termination import should_terminate_rollout


def test_unbounded_rollout_continues_until_native_success():
    assert not should_terminate_rollout(
        elapsed_s=10_000.0, time_limit_s=None, success_at=None, stop_on_success=True)
    assert should_terminate_rollout(
        elapsed_s=87.9, time_limit_s=None, success_at=87.876, stop_on_success=True)


def test_fixed_window_control_still_ends_at_its_deadline():
    assert not should_terminate_rollout(
        elapsed_s=59.9, time_limit_s=60.0, success_at=None, stop_on_success=False)
    assert should_terminate_rollout(
        elapsed_s=60.0, time_limit_s=60.0, success_at=None, stop_on_success=False)


def test_fixed_window_mode_does_not_end_early_on_success():
    assert not should_terminate_rollout(
        elapsed_s=30.0, time_limit_s=60.0, success_at=29.0, stop_on_success=False)


def test_fastwam_stage_omits_legacy_window_unless_explicitly_fixed():
    assert _episode_window_args({"model": "fastwam", "window_s": 60}) == []
    assert _episode_window_args({"model": "fastwam", "window_s": 60,
                                 "fixed_window": True}) == ["--window-s", "60"]
    assert _episode_window_args({"model": "cosmos3", "window_s": 60}) == ["--window-s", "60"]


def test_native_failure_stops_at_original_action_budget_without_time_cap():
    from wo.models.rollout_termination import rollout_end_reason
    args = dict(elapsed_s=420.,time_limit_s=None,success_at=None,stop_on_success=True,action_limit=1000)
    assert rollout_end_reason(**args,executed_actions=999) is None
    assert rollout_end_reason(**args,executed_actions=1000) == 'native_action_limit'
    args['success_at'] = 419.
    assert rollout_end_reason(**args,executed_actions=1000) == 'native_task_success'
