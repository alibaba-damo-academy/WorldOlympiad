"""Lower pedestal search is bounded and cannot displace ordinary candidates."""
from wo.expert.embodiment import placement_candidates


def test_lower_mounts_are_last_and_separately_bounded():
    canonical = [(0, (0, 0, 1), 0)]
    preserve = [(0, (1, 0, 1), 0)]
    lower = [(0, (1, 0, h), 0) for h in (.85, .70, .55)]
    rows = list(placement_candidates(canonical, preserve, budget=2, lower_mounts=lower))
    assert rows[:2] == [(False, *canonical[0]), (True, *preserve[0])]
    assert rows[2:] == [(True, *row) for row in lower[:2]]


def test_lower_mount_duplicates_do_not_consume_budget():
    row = (0, (1, 0, .8), 0)
    other = (0, (1, 0, .6), 0)
    assert list(placement_candidates([], [], budget=2, lower_mounts=[row, row, other])) == [
        (True, *row), (True, *other)]


def test_without_lower_mounts_original_search_unchanged():
    row = (0, (1, 0, 1), 0)
    assert list(placement_candidates([row])) == [(False, *row)]
