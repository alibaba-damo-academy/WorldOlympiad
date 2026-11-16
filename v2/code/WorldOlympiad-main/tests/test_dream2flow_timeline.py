from __future__ import annotations

import unittest

import numpy as np

from scripts.run_dream2flow_reference import _append_terminal_hold


class Dream2FlowTimelineTest(unittest.TestCase):
    def test_terminal_hold_does_not_retime_final_motion_sample(self) -> None:
        joints = np.arange(6, dtype=np.float64).reshape(3, 2)
        ee = np.arange(21, dtype=np.float64).reshape(3, 7)
        indices = np.asarray([0, 1, 2], dtype=np.int64)

        out_joints, out_ee, out_indices, timestamps = _append_terminal_hold(
            joints,
            ee,
            indices,
            source_fps=10.0,
            source_frame_count=3,
        )

        np.testing.assert_allclose(timestamps, [0.0, 0.1, 0.2, 0.3])
        np.testing.assert_array_equal(out_indices, [0, 1, 2, 2])
        np.testing.assert_array_equal(out_joints[-1], out_joints[-2])
        np.testing.assert_array_equal(out_ee[-1], out_ee[-2])


if __name__ == "__main__":
    unittest.main()
