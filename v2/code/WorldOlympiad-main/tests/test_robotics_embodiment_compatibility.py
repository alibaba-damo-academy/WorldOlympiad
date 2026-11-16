from __future__ import annotations

import unittest

from autoresearch.robotics_embodiment_compatibility import (
    validate_parallel_gripper_cylinder,
)


class ParallelGripperCompatibilityTest(unittest.TestCase):
    def test_body_sized_proxy_preserves_closing_travel(self) -> None:
        result = validate_parallel_gripper_cylinder(
            radius_m=0.0365,
            max_aperture_m=0.085,
            minimum_closing_travel_m=0.008,
        )
        self.assertEqual(result["status"], "pass")
        self.assertAlmostEqual(result["available_closing_travel_m"], 0.012)

    def test_zero_margin_bounding_radius_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "available_closing_travel=0.000000"):
            validate_parallel_gripper_cylinder(
                radius_m=0.0425,
                max_aperture_m=0.085,
                minimum_closing_travel_m=0.008,
            )


if __name__ == "__main__":
    unittest.main()
