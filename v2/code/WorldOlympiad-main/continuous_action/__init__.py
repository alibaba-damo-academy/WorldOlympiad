"""Continuous Action Evaluation for WorldOlympiad 2.0.

The public API operates on action trajectories inferred from videos. Under the
formal imagination protocol, both GT and WAM future video pass through
Dream2Flow; policy action-head tensors are explicitly excluded.
"""

from .metrics import ContinuousActionScore, score_action_trajectories
from .schema import ActionTrajectory

__all__ = ["ActionTrajectory", "ContinuousActionScore", "score_action_trajectories"]
