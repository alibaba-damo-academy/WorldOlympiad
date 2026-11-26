from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scenecode_pybullet_bridge import (
    PolicySceneInputs,
    PolicySceneObject,
    PyBulletDroidPolicyScene,
)


class PolicySceneObjectContractTest(unittest.TestCase):
    def _inputs(
        self, root: Path, additional: tuple[PolicySceneObject, ...]
    ) -> PolicySceneInputs:
        return PolicySceneInputs(
            table_sdf=root / "table.sdf",
            cup_sdf=root / "subject.sdf",
            tray_sdf=root / "target.sdf",
            panda_urdf=root / "robot.urdf",
            output_dir=root / "output",
            subject_id="subject_a",
            target_id="target_a",
            additional_objects=additional,
        )

    def test_additional_object_ids_cannot_alias_an_active_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = PolicySceneObject(
                object_id="subject_a",
                sdf_path=root / "other.sdf",
                center_xy=(0.1, 0.1),
                role="manipuland",
            )
            with self.assertRaisesRegex(ValueError, "ids must be unique"):
                PyBulletDroidPolicyScene(self._inputs(root, (duplicate,)))

    def test_manipulands_cannot_be_declared_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = PolicySceneObject(
                object_id="subject_b",
                sdf_path=root / "other.sdf",
                center_xy=(0.1, 0.1),
                role="manipuland",
                fixed=True,
            )
            with self.assertRaisesRegex(ValueError, "manipuland may not be fixed"):
                PyBulletDroidPolicyScene(self._inputs(root, (invalid,)))

    def test_object_table_is_data_driven(self) -> None:
        item = PolicySceneObject(
            object_id="destination_b",
            sdf_path=Path("destination.sdf"),
            center_xy=(0.2, -0.1),
            role="destination",
            collision_enabled=False,
            fixed=True,
        )
        inputs = PolicySceneInputs(
            table_sdf=Path("table.sdf"),
            cup_sdf=Path("subject.sdf"),
            tray_sdf=Path("target.sdf"),
            panda_urdf=Path("robot.urdf"),
            output_dir=Path("output"),
            subject_id="subject_a",
            target_id="destination_a",
            additional_objects=(item,),
        )
        self.assertEqual(inputs.additional_objects[0].object_id, "destination_b")
        self.assertEqual(inputs.additional_objects[0].role, "destination")


if __name__ == "__main__":
    unittest.main()
