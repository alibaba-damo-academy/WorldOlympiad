"""Unit tests for convert_urdf_to_sdf using real URDF examples.

What this test does:
- Calls `convert_urdf_to_sdf` for two real assets (`Box_01`, `Bucket_01`).
- Verifies the converter can generate valid SDF files from those URDF inputs.

Input:
- URDF files under
  `tests/test_data/urdf_examples`
  (`Box_01/box_01.urdf` and `Bucket_01/bucket_01.urdf`).
- Minimal per-link physics values required by `convert_urdf_to_sdf`.

Output:
- Generated SDF files next to each input URDF file.
- Assertions on SDF structure and consistency:
  root/version, model existence, link/joint counts, link/joint names,
  and inertial fields for each link.
"""

import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path
from unittest.mock import patch

from scenecode.agent_utils.urdf_to_sdf import (
    LinkPhysics,
    URDFParseResult,
    convert_urdf_to_sdf,
    merge_link_visual_meshes_for_sdf,
    parse_urdf,
)


EXAMPLES_ROOT = Path(
    "tests/test_data/urdf_examples"
)


class TestConvertUrdfToSdfExamples(unittest.TestCase):
    """Test convert_urdf_to_sdf() with real URDF examples."""

    CASES = (
        ("Box_01", "box_01.urdf"),
        ("Bucket_01", "bucket_01.urdf"),
    )

    @classmethod
    def setUpClass(cls):
        if not EXAMPLES_ROOT.exists():
            raise unittest.SkipTest(f"URDF examples directory not found: {EXAMPLES_ROOT}")

        missing = []
        for folder_name, urdf_name in cls.CASES:
            urdf_path = EXAMPLES_ROOT / folder_name / urdf_name
            if not urdf_path.exists():
                missing.append(str(urdf_path))

        if missing:
            raise unittest.SkipTest(
                "Missing URDF example files:\n" + "\n".join(missing)
            )

    def _build_minimal_link_physics(
        self, urdf_result: URDFParseResult
    ) -> dict[str, LinkPhysics]:
        """Provide required physics for links with geometry."""
        physics: dict[str, LinkPhysics] = {}
        for link_name, link_elem in urdf_result.links.items():
            has_geometry = bool(link_elem.findall("visual") or link_elem.findall("collision"))
            if has_geometry:
                physics[link_name] = LinkPhysics(
                    mass=1.0,
                    inertia_ixx=0.1,
                    inertia_iyy=0.1,
                    inertia_izz=0.1,
                    inertia_ixy=0.0,
                    inertia_ixz=0.0,
                    inertia_iyz=0.0,
                    center_of_mass=(0.0, 0.0, 0.0),
                )
        return physics

    def test_merge_link_visual_meshes_for_sdf_preserves_origin_rotation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            obj_path = tmp_path / 'drawer.obj'
            obj_path.write_text('o drawer\n', encoding='utf-8')
            sdf_dir = tmp_path / 'out'
            visual_xml = ET.fromstring(
                """<link name="drawer">
                    <visual>
                        <origin xyz="0 0.03 -0.685" rpy="1.5707963 0 0"/>
                        <geometry><mesh filename="drawer.obj"/></geometry>
                    </visual>
                </link>"""
            )

            with patch('scenecode.agent_utils.urdf_to_sdf.merge_objs_to_gltf') as mock_merge:
                output_path = merge_link_visual_meshes_for_sdf(
                    urdf_link=visual_xml,
                    urdf_dir=tmp_path,
                    sdf_dir=sdf_dir,
                    link_name='drawer_01',
                )

        self.assertEqual(output_path, sdf_dir / 'visual' / 'drawer_01_visual.gltf')
        mock_merge.assert_called_once()
        merged_entry = mock_merge.call_args.kwargs['obj_paths_with_offsets'][0]
        self.assertEqual(merged_entry[0], obj_path)
        self.assertEqual(merged_entry[1], (0.0, 0.03, -0.685))
        self.assertEqual(merged_entry[2], (1.5707963, 0.0, 0.0))

    def test_convert_urdf_examples(self):
        """Converts Box_01 and Bucket_01 URDFs and validates SDF structure."""
        for folder_name, urdf_name in self.CASES:
            with self.subTest(example=folder_name):
                urdf_path = EXAMPLES_ROOT / folder_name / urdf_name
                urdf_result = parse_urdf(urdf_path)
                link_physics = self._build_minimal_link_physics(urdf_result)

                output_path = urdf_path.with_suffix(".sdf")
                result_path = convert_urdf_to_sdf(
                    urdf_path=urdf_path,
                    output_path=output_path,
                    link_physics=link_physics,
                    repair_missing_meshes=True,
                )

                self.assertEqual(result_path, output_path)
                self.assertTrue(output_path.exists())

                tree = ET.parse(output_path)
                root = tree.getroot()
                self.assertEqual(root.tag, "sdf")
                self.assertEqual(root.get("version"), "1.7")

                model = root.find("model")
                self.assertIsNotNone(model)

                sdf_links = model.findall("link")
                sdf_joints = model.findall("joint")
                self.assertEqual(len(sdf_links), len(urdf_result.links))
                self.assertEqual(len(sdf_joints), len(urdf_result.joints))

                sdf_link_names = {elem.get("name") for elem in sdf_links}
                urdf_link_names = set(urdf_result.links.keys())
                self.assertEqual(sdf_link_names, urdf_link_names)

                sdf_joint_names = {elem.get("name") for elem in sdf_joints}
                urdf_joint_names = set(urdf_result.joints.keys())
                self.assertEqual(sdf_joint_names, urdf_joint_names)

                for link_elem in sdf_links:
                    inertial = link_elem.find("inertial")
                    self.assertIsNotNone(inertial)
                    self.assertIsNotNone(inertial.find("mass"))
                    inertia = inertial.find("inertia")
                    self.assertIsNotNone(inertia)
                    self.assertIsNotNone(inertia.find("ixx"))
                    self.assertIsNotNone(inertia.find("iyy"))
                    self.assertIsNotNone(inertia.find("izz"))


if __name__ == "__main__":
    unittest.main()
