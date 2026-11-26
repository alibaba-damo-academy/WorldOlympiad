import numpy as np

from wo.expert.rollout import resolve_target


class _World:
    support_id = "table"
    bodies = {"book": 1, "table": 2}


def test_reorient_without_target_uses_support():
    class Goal:
        target = None
    class Step:
        goal = Goal()
        target = None
    assert resolve_target(_World(), Step()) == "table"


def test_invalid_target_falls_back_to_support():
    class Goal:
        target = "missing"
    class Step:
        goal = Goal()
        target = "missing"
    assert resolve_target(_World(), Step()) == "table"
