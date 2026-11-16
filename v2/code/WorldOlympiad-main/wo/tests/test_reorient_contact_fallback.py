from types import SimpleNamespace

import pytest

from wo.expert.rollout import allows_overhead_release, needs_tilted_contact_search


@pytest.mark.parametrize('reason', ['release_ik', 'release_collision:table', 'release_descent_path'])
def test_reorientation_release_failure_changes_contact_frame(reason):
    evidence = {'rejections': {reason: 1}, 'approach_failures': []}
    assert needs_tilted_contact_search(SimpleNamespace(primitive='reorient'), evidence)
    assert not needs_tilted_contact_search(SimpleNamespace(primitive='transport'), evidence)


def test_unrelated_failure_does_not_trigger_contact_search():
    step = SimpleNamespace(primitive='reorient')
    assert not needs_tilted_contact_search(step, {'rejections': {'open_finger_penetration': 1}})
    assert not needs_tilted_contact_search(step, {'rejections': {'release_ik': 0}})


def test_inside_goal_allows_collision_checked_overhead_drop():
    assert allows_overhead_release(SimpleNamespace(
        primitive='transport', goal=SimpleNamespace(kind='inside')))
    assert not allows_overhead_release(SimpleNamespace(
        primitive='transport', goal=SimpleNamespace(kind='on_region')))


def test_existing_top_approach_trigger_preserved():
    assert needs_tilted_contact_search(SimpleNamespace(primitive='transport'), {
        'approach_failures': [{'grasp_kind': 'top', 'reason': 'collision'}]})
