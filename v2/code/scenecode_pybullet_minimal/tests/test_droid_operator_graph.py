from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scenecode_pybullet_bridge.droid_operator_graph import (
    DroidGraphObject,
    DroidOperatorGraphInputs,
    DroidPickTerminalStep,
    _validate_inputs,
)


class DroidOperatorGraphContractTest(unittest.TestCase):
    def _objects(self, root: Path) -> tuple[DroidGraphObject, ...]:
        return (
            DroidGraphObject("alpha", root / "a.sdf", (-0.2, -0.1), "manipuland"),
            DroidGraphObject("beta", root / "b.sdf", (-0.1, -0.1), "manipuland"),
            DroidGraphObject("zone_one", root / "z1.sdf", (0.2, 0.1), "destination"),
            DroidGraphObject("zone_two", root / "z2.sdf", (0.1, 0.1), "destination"),
        )

    def _step(self, subject: str, target: str) -> DroidPickTerminalStep:
        return DroidPickTerminalStep(
            subject_id=subject,
            target_id=target,
            terminal_operator="place",
            success_predicate="in_container",
            grasp_base_offset_z_m=0.18,
            hover_base_offset_z_m=0.36,
            lift_base_offset_z_m=0.42,
        )

    def test_arbitrary_two_step_graph_is_admitted_from_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = DroidOperatorGraphInputs(
                table_sdf=root / "table.sdf",
                panda_urdf=root / "robot.urdf",
                output_dir=root / "out",
                objects=self._objects(root),
                steps=(
                    self._step("alpha", "zone_one"),
                    self._step("beta", "zone_two"),
                ),
            )
            self.assertEqual(set(_validate_inputs(inputs)), {"alpha", "beta", "zone_one", "zone_two"})

    def test_graph_allows_one_subject_in_multiple_compiled_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = DroidOperatorGraphInputs(
                table_sdf=root / "table.sdf",
                panda_urdf=root / "robot.urdf",
                output_dir=root / "out",
                objects=self._objects(root),
                steps=(
                    self._step("alpha", "zone_one"),
                    self._step("alpha", "zone_two"),
                ),
            )
            self.assertEqual(set(_validate_inputs(inputs)), {
                "alpha", "beta", "zone_one", "zone_two"
            })

    def test_terminal_operator_must_match_its_physical_predicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = DroidPickTerminalStep(
                subject_id="alpha",
                target_id="zone_one",
                terminal_operator="insert",
                success_predicate="in_container",
                grasp_base_offset_z_m=0.18,
                hover_base_offset_z_m=0.36,
                lift_base_offset_z_m=0.42,
            )
            inputs = DroidOperatorGraphInputs(
                table_sdf=root / "table.sdf",
                panda_urdf=root / "robot.urdf",
                output_dir=root / "out",
                objects=self._objects(root),
                steps=(bad, self._step("beta", "zone_two")),
            )
            with self.assertRaisesRegex(ValueError, "predicate disagree"):
                _validate_inputs(inputs)


if __name__ == "__main__":
    unittest.main()
