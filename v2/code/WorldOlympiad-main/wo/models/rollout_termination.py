"""Pure episode termination rules shared by native rollout adapters."""
from __future__ import annotations


def rollout_end_reason(*, elapsed_s: float, time_limit_s: float | None,
                       success_at: float | None, stop_on_success: bool,
                       executed_actions: int = 0, action_limit: int | None = None) -> str | None:
    if stop_on_success and success_at is not None:
        return "native_task_success"
    if time_limit_s is not None and elapsed_s >= time_limit_s:
        return "evaluation_window_complete"
    if action_limit is not None and executed_actions >= action_limit:
        return "native_action_limit"
    return None


def should_terminate_rollout(**kwargs) -> bool:
    """Stop on native success, configured time, or the original native action budget."""
    return rollout_end_reason(**kwargs) is not None
