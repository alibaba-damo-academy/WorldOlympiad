import numpy as np
import pytest
from wo.expert.kin import Pose, TCP_OFFSET, aperture_tcp_offset, tcp_to_ee

class MovingPads:
    robot, cid, fingers = 1, 0, [2, 3]
    def __init__(self):
        self.f = 0.23
        self.p = self
        self.R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        self.origin = np.array([0.3, -0.2, 0.7])
    def gripper_fraction(self): return self.f
    def set_gripper_state(self, f): self.f = f
    def ee_pose(self):
        from wo.expert.kin import mat_to_quat
        return self.origin, mat_to_quat(self.R)
    def getLinkState(self, robot, link, **kw):
        local = np.array([(1 if link == 2 else -1)*(0.067-0.04*self.f), 0, 0.098+0.02*self.f])
        return [None]*4 + [self.origin + self.R @ local]

def test_tcp_delta_is_in_ee_frame_and_state_restored():
    w = MovingPads()
    offset = aperture_tcp_offset(w, 0.6)
    assert offset == pytest.approx([0, 0, 0.162])
    assert w.f == 0.23
    tcp = Pose(np.array([0.1, 0.2, 0.3]), w.R)
    ee = tcp_to_ee(tcp, offset)
    assert ee.p + ee.R @ offset == pytest.approx(tcp.p)

def test_open_calibration_unchanged():
    assert aperture_tcp_offset(None, 0) == pytest.approx(TCP_OFFSET)

def test_measurement_failure_restores_state():
    w = MovingPads()
    def fail(*a, **kw): raise RuntimeError('frame unavailable')
    w.getLinkState = fail
    with pytest.raises(RuntimeError): aperture_tcp_offset(w, 0.6)
    assert w.f == 0.23
