import numpy as np
from wo.expert.kin import Pose
from wo.expert.push import _plan_push_entry, _top_down, _needs_push_entry
from wo.expert.kin import tcp_to_ee


def test_entry_reuse_requires_orientation_not_only_position():
    start = Pose(np.array([.1, .2, .75]), _top_down(0))
    desired = tcp_to_ee(start)
    assert not _needs_push_entry(desired, start)
    # Same EE origin, previous certified orthogonal wrist frame.
    actual = Pose(desired.p.copy(), _top_down(np.pi / 2))
    assert _needs_push_entry(actual, start)
    assert _needs_push_entry(Pose(desired.p, _top_down(np.pi)), start)
    assert _needs_push_entry(desired.moved([.03, 0, 0]), start)
    assert not _needs_push_entry(desired.moved([.001, 0, 0]), start)



class EntryKin:
    def __init__(self, always_fail=False):
        self.ignore = {'existing'}
        self.always_fail = always_fail
        self.strokes = 0

    def ik(self, pose, seed, accept_solution=None):
        assert self.ignore == {'existing'}
        if accept_solution is not None and not accept_solution(seed):
            return None
        return seed

    def in_collision(self, q, gripper):
        return None

    def plan(self, q, qa, gripper):
        assert self.ignore == {'existing'}
        return [q, qa]

    def cartesian_path(self, q, start, end, **kwargs):
        assert kwargs['check'] is True
        if 'diagnostics' not in kwargs:
            assert self.ignore == {'existing', 'subject'}
            return [q, q]
        assert self.ignore == {'existing', 'subject', 'target'}
        self.strokes += 1
        if self.always_fail or self.strokes == 1:
            kwargs['diagnostics']['reason'] = 'ik'
            return None
        kwargs['diagnostics']['reason'] = 'success'
        return [q, q]


def test_entry_flips_equivalent_axis_only_after_checked_stroke_failure():
    kin = EntryKin()
    start = Pose(np.array([0., 0., .75]), _top_down(0))
    end = start.moved([.04, 0, 0])
    evidence = {}
    result = _plan_push_entry(kin, np.zeros(7), start, end,
                              'subject', 'target', evidence, 4)
    assert result is not None
    a, b = result[:2]
    np.testing.assert_allclose(a.p, start.p)
    np.testing.assert_allclose(b.p, end.p)
    np.testing.assert_allclose(a.R[:, 2], start.R[:, 2])
    np.testing.assert_allclose(a.R[:, 0], -start.R[:, 0])
    assert evidence['entry_frames'] == [{'segment': 4, 'flipped': True}]
    assert kin.ignore == {'existing'}


def test_entry_never_accepts_rejected_strokes():
    kin = EntryKin(True)
    pose = Pose(np.zeros(3), _top_down(0))
    assert _plan_push_entry(kin, np.zeros(7), pose, pose,
                            'subject', 'target', {}, 0) is None
    assert kin.strokes == 48


class HighApproachKin(EntryKin):
    def ik(self, pose, seed, accept_solution=None):
        if pose.p[2] > 1.:
            return None
        return super().ik(pose, seed, accept_solution)


def test_lower_approach_after_exhaustion_retains_checks():
    kin = HighApproachKin()
    kin.strokes = 1
    start = Pose(np.array([.2, .1, .75]), _top_down(0))
    evidence = {}
    result = _plan_push_entry(kin, np.zeros(7), start, start.moved([.04, 0, 0]),
                              'subject', 'target', evidence, 10)
    assert result is not None
    assert evidence['entry_frames'][0]['approach_rise_m'] == .08
    assert len(evidence['entry_checks']) == 17
    assert all(x['reason'] == 'above_ik' for x in evidence['entry_checks'][:16])
    assert evidence['entry_checks'][-1]['reason'] == 'success'
    np.testing.assert_allclose(result[0].p, start.p)
    assert kin.ignore == {'existing'}
    assert kin.ignore == {'existing'}


class SingularEntryKin(EntryKin):
    def cartesian_path(self, q, start, end, **kwargs):
        result = super().cartesian_path(q, start, end, **kwargs)
        if 'diagnostics' in kwargs and self.strokes <= 2:
            kwargs['diagnostics']['reason'] = 'ik'
            return None
        return result


def test_entry_orthogonal_vertical_frame_remains_fully_checked():
    kin = SingularEntryKin()
    start = Pose(np.array([.2, .1, .75]), _top_down(0))
    end = start.moved([.04, 0, 0])
    evidence = {}
    result = _plan_push_entry(kin, np.zeros(7), start, end,
                              'subject', 'target', evidence, 4)
    assert result is not None
    a, b = result[:2]
    np.testing.assert_allclose(a.p, start.p)
    np.testing.assert_allclose(b.p, end.p)
    np.testing.assert_allclose(a.R[:, 2], start.R[:, 2])
    assert abs(np.dot(a.R[:, 0], start.R[:, 0])) < 1e-10
    assert evidence['entry_frames'][0]['wrist_rotation_rad'] == np.pi / 2
    assert kin.ignore == {'existing'}


class InnerWorkspaceKin(EntryKin):
    def cartesian_path(self, q, start, end, **kwargs):
        result = super().cartesian_path(q, start, end, **kwargs)
        if 'diagnostics' in kwargs and self.strokes <= 4:
            kwargs['diagnostics']['reason'] = 'ik'
            return None
        return result


def test_tilted_pusher_is_fallback_and_keeps_contact_line_and_checks():
    kin = InnerWorkspaceKin()
    start = Pose(np.array([-.2, .04, .75]), _top_down(0))
    end = start.moved([.04, -.03, 0])
    evidence = {}
    result = _plan_push_entry(kin, np.zeros(7), start, end,
                              'subject', 'target', evidence, 4)
    assert result is not None
    a, b = result[:2]
    np.testing.assert_allclose(a.p, start.p)
    np.testing.assert_allclose(b.p, end.p)
    np.testing.assert_allclose(a.R.T @ a.R, np.eye(3), atol=1e-14)
    assert kin.strokes == 5
    assert evidence['entry_frames'][0]['wrist_tilt_rad'] == np.deg2rad(-25)
    assert kin.ignore == {'existing'}


class BlockedFirstBranchKin(EntryKin):
    def ik(self, pose, seed, accept_solution=None):
        if accept_solution is None:
            return np.zeros(7)
        # First branch still fails descent; second branch must be checked too.
        assert not accept_solution(np.zeros(7))
        alternative = np.ones(7)
        return alternative if accept_solution(alternative) else None

    def cartesian_path(self, q, start, end, **kwargs):
        if np.all(q == 0):
            assert self.ignore == {'existing', 'subject'}
            return None
        return super().cartesian_path(q, start, end, **kwargs)


def test_entry_exhausts_branches_only_after_all_first_branches_fail():
    kin = BlockedFirstBranchKin()
    kin.strokes = 1
    start = Pose(np.array([.2, .1, .75]), _top_down(0))
    result = _plan_push_entry(kin, np.zeros(7), start, start.moved([.04, 0, 0]),
                              'subject', 'target', {}, 10)
    assert result is not None
    np.testing.assert_equal(result[2], np.ones(7))
    assert kin.ignore == {'existing'}
