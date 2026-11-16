from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

from scenecode_pybullet_bridge import ArticulationBinding, ArticulatedSceneInputs
from scenecode_pybullet_bridge.articulated import flatten_sdf_link_poses
from scenecode_pybullet_bridge.articulated_contact import _joint_seeds
from scenecode_pybullet_bridge.sdf import prepare_physics_sdf


class ArticulatedContractTest(unittest.TestCase):
    def test_default_robot_placement_translates_with_host(self) -> None:
        original = ArticulatedSceneInputs(table_sdf=None, host_sdf=Path('host.sdf'),
            panda_urdf=Path('robot.urdf'), output_dir=Path('output'), binding=self._binding())
        moved = replace(original, host_center_xy=(1.2, .55), robot_base_xy=None)
        self.assertAlmostEqual(moved.robot_base_xy[0] - original.robot_base_xy[0], 1.2)
        self.assertAlmostEqual(moved.robot_base_xy[1] - original.robot_base_xy[1], .54)
        explicit = replace(moved, robot_base_xy=(2., 3.))
        self.assertEqual(explicit.robot_base_xy, (2., 3.))
        self.assertTrue(original.automatic_robot_placement)
        self.assertFalse(explicit.automatic_robot_placement)
        yaw_only = replace(original, robot_base_xy=None, robot_base_yaw=.3)
        self.assertFalse(yaw_only.automatic_robot_placement)

    def _binding(self) -> ArticulationBinding:
        return ArticulationBinding(
            case_id="case",
            component_id="drawer",
            host_object_id="cabinet",
            host_sdf="cabinet.sdf",
            host_sdf_sha256="a" * 64,
            joint_name="drawer_joint",
            joint_type="prismatic",
            parent_link="body",
            child_link="drawer",
            axis_xyz=(0.0, -1.0, 0.0),
            lower_limit=0.0,
            upper_limit=0.3,
            closed_position=0.0,
            open_position=0.3,
            travel=0.3,
            open_delta_xyz=(0.0, -0.3, 0.0),
        )

    def test_binding_rejects_degenerate_travel(self) -> None:
        binding = self._binding()
        invalid = binding.__class__(**{**binding.__dict__, "open_delta_xyz": (0.0, 0.0, 0.0)})
        with self.assertRaisesRegex(ValueError, "open delta"):
            invalid.validate()

    def test_close_requires_closed_predicate(self) -> None:
        with self.assertRaisesRegex(ValueError, "operator and success predicate"):
            ArticulatedSceneInputs(
                table_sdf=Path("table.sdf"),
                host_sdf=Path("host.sdf"),
                panda_urdf=Path("robot.urdf"),
                output_dir=Path("out"),
                binding=self._binding(),
                operator="close",
                success_predicate="joint_at_open_limit",
            ).validate()

    def test_binding_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binding.json"
            path.write_text(json.dumps(self._binding().__dict__), encoding="utf-8")
            restored = ArticulationBinding.from_json(path)
            self.assertEqual(restored.component_id, "drawer")
            self.assertEqual(restored.open_position, 0.3)

    def test_flatten_sdf_link_poses_resolves_nested_parent_frames(self) -> None:
        """PyBullet resolves link poses against the model root, not the parent."""

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "host.sdf"
            source.write_text(_nested_link_sdf(), encoding="utf-8")
            artifact = prepare_physics_sdf(source, Path(directory) / "physics.sdf")
            flattened = flatten_sdf_link_poses(artifact)
            root = ET.parse(flattened.physics_sdf).getroot()
            links = {link.get("name"): link for link in root.iter() if link.tag.rsplit("}", 1)[-1] == "link"}
            poses = {
                name: next(
                    node for node in link if node.tag.rsplit("}", 1)[-1] == "pose"
                )
                for name, link in links.items()
            }
            self.assertEqual([float(v) for v in poses["handle"].text.split()[:3]], [0.3, 0.0, 1.0])
            self.assertEqual([float(v) for v in poses["door"].text.split()[:3]], [0.0, 0.0, 1.0])
            for node in poses.values():
                self.assertIsNone(node.get("relative_to"))

    def test_contact_planner_seeds_are_deterministic_and_bounded(self) -> None:
        lower = np.array([-1.0] * 7)
        upper = np.array([1.0] * 7)
        seeds = _joint_seeds(lower, upper, 8)
        self.assertEqual(len(seeds), 8)
        for seed in seeds:
            self.assertTrue(np.all(seed >= lower))
            self.assertTrue(np.all(seed <= upper))
        again = _joint_seeds(lower, upper, 8)
        for first, second in zip(seeds, again):
            np.testing.assert_allclose(first, second)

    def test_grasp_standoff_and_actuation_count_are_validated(self) -> None:
        original = ArticulatedSceneInputs(
            table_sdf=None, host_sdf=Path("host.sdf"), panda_urdf=Path("robot.urdf"),
            output_dir=Path("output"), binding=self._binding(),
        )
        with self.assertRaisesRegex(ValueError, "grasp standoff"):
            replace(original, grasp_standoff_m=0.5).validate()
        with self.assertRaisesRegex(ValueError, "grasp lead"):
            replace(original, grasp_lead_rad=0.5).validate()
        with self.assertRaisesRegex(ValueError, "arm force scale"):
            replace(original, arm_force_scale=0.5).validate()
        with self.assertRaisesRegex(ValueError, "actuation step count"):
            replace(original, actuate_steps=5).validate()
        slower = replace(original, actuate_steps=180)
        self.assertEqual(dict(slower.action_counts)["actuate"], 180)
        self.assertEqual(dict(original.action_counts)["actuate"], 90)


def _nested_link_sdf() -> str:
    return """<?xml version="1.0"?>
<sdf version="1.7"><model name="fixture">
  <link name="base"><collision name="base_collision"><geometry><box><size>0.1 0.1 0.1</size></box></geometry></collision></link>
  <link name="door"><pose relative_to="base">0 0 1 0 0 0</pose><collision name="door_collision"><geometry><box><size>0.1 0.1 0.5</size></box></geometry></collision></link>
  <link name="handle"><pose relative_to="door">0.3 0 0 0 0 0</pose><collision name="handle_collision"><geometry><box><size>0.02 0.02 0.02</size></box></geometry></collision></link>
  <joint name="hinge" type="revolute"><parent>base</parent><child>door</child><axis><xyz>0 0 1</xyz></axis></joint>
  <joint name="grip" type="fixed"><parent>door</parent><child>handle</child></joint>
</model></sdf>
"""


if __name__ == "__main__":
    unittest.main()
