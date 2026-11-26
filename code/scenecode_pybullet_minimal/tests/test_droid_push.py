from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scenecode_pybullet_bridge.droid_push import (
    PUSH_PHASE_ACTION_COUNTS,
    DroidPushInputs,
    push_waypoints_from_aabbs,
    run_droid_push,
)


class DroidPushContractTest(unittest.TestCase):
    def test_waypoints_follow_runtime_subject_target_corridor(self) -> None:
        result = push_waypoints_from_aabbs(
            subject_aabb=((0.08, -0.02, 0.90), (0.12, 0.02, 1.00)),
            target_aabb=((-0.14, 0.08, 0.90), (-0.06, 0.16, 0.91)),
        )
        direction = np.asarray(result["direction_xy"])
        np.testing.assert_allclose(np.linalg.norm(direction), 1.0)
        self.assertGreater(
            float(np.dot(np.asarray(result["finish_xy"]) - np.asarray(result["contact_xy"]), direction)),
            0.1,
        )
        self.assertAlmostEqual(float(result["push_base_z_m"]), 1.093632138)

    def test_timeline_has_no_external_target_duration(self) -> None:
        self.assertEqual(sum(PUSH_PHASE_ACTION_COUNTS.values()), 233)
        self.assertEqual(PUSH_PHASE_ACTION_COUNTS["push"], 60)

    def test_invalid_friction_fails_before_loading_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "friction"):
                run_droid_push(
                    DroidPushInputs(
                        table_sdf=root / "table.sdf",
                        subject_sdf=root / "subject.sdf",
                        target_sdf=root / "target.sdf",
                        panda_urdf=root / "robot.urdf",
                        output_dir=root / "output",
                        subject_lateral_friction=2.1,
                    )
                )


if __name__ == "__main__":
    unittest.main()
