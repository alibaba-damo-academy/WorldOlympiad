import pytest

from autoresearch.cosmos_contact_support import contact_support_evidence


def test_normal_rule_and_ranking_match_shared_support_below():
    # Two shapes of the same target accumulate; strong lateral and downward
    # contacts must not count as support, even if net resultant points up.
    ev = contact_support_evidence(
        [2, 3, 100, 100, 4],
        [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, -1], [0, 0, 1]],
        [1, 3, 1], [0, 1, 4], ['target', 'target', 'table'])
    assert ev['supported_by'] == 'target'
    assert ev['upward_load_by_body_n'] == {'target': 5, 'table': 4}
    assert ev['touching_bodies'] == ['table', 'target']


def test_unoccupied_buffer_padding_is_not_evidence():
    ev = contact_support_evidence([1, float('nan')],
        [[0, 0, 1], [float('nan')]*3], [1], [0], ['target'])
    assert ev['supported_by'] == 'target'


def test_target_touch_does_not_imply_target_support():
    ev = contact_support_evidence([20, 2], [[1, 0, 0], [0, 0, 1]],
                                  [1, 1], [0, 1], ['target', 'table'])
    assert 'target' in ev['touching_bodies']
    assert ev['supported_by'] == 'table'


def test_contact_normal_threshold_and_zero_force_floor():
    ev = contact_support_evidence([100, 0], [[0, 0, .5], [0, 0, 1]],
                                  [1, 1], [0, 1], ['table', 'target'])
    assert ev['supported_by'] == 'target'
    assert ev['upward_load_by_body_n'] == {'target': 1e-6}
    empty = contact_support_evidence([], [], [0], [0], ['target'])
    assert empty['supported_by'] is None
    assert empty['touching_bodies'] == []


@pytest.mark.parametrize('forces,normals,counts,starts,names', [
    ([1], [[0, 0, 1]], [2], [0], ['target']),
    ([float('nan')], [[0, 0, 1]], [1], [0], ['target']),
    ([1], [[0, 0, float('inf')]], [1], [0], ['target']),
    ([1], [[0, 0, 1]], [-1], [0], ['target']),
    ([1], [[0, 0, 1]], [1], [0], []),
    ([1], [[0, 0, 1]], [1], [.5], ['target']),
])
def test_invalid_or_incomplete_contact_buffers_fail_closed(forces, normals, counts, starts, names):
    with pytest.raises(ValueError):
        contact_support_evidence(forces, normals, counts, starts, names)
