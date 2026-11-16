import tempfile
import unittest

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh

from scenecode.agent_utils.sdf_mesh_utils import (
    SDFParseError,
    _parse_mesh_scale,
    combine_sdf_meshes_at_joint_angles,
)


class TestSDFMeshUtils(unittest.TestCase):
    def _write_unit_box_mesh(self, temp_dir: Path) -> Path:
        mesh_path = temp_dir / "unit_box.glb"
        trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(mesh_path)
        return mesh_path

    def _write_sdf(
        self,
        temp_dir: Path,
        mesh_path: Path,
        scale_text: str | None = None,
        link_pose_text: str | None = None,
    ) -> Path:
        scale_xml = f"<scale>{scale_text}</scale>" if scale_text is not None else ""
        link_pose_xml = (
            f"<pose>{link_pose_text}</pose>" if link_pose_text is not None else ""
        )
        sdf_path = temp_dir / "scaled_visual.sdf"
        sdf_path.write_text(
            f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="scaled_visual">
    <link name="box_link">
      {link_pose_xml}
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>1.0</ixx>
          <iyy>1.0</iyy>
          <izz>1.0</izz>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyz>0.0</iyz>
        </inertia>
      </inertial>
      <visual name="box_visual">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <mesh>
            <uri>{mesh_path.name}</uri>
            {scale_xml}
          </mesh>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
""",
            encoding="utf-8",
        )
        return sdf_path

    def test_combine_sdf_meshes_applies_visual_mesh_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            mesh_path = self._write_unit_box_mesh(temp_dir)
            sdf_path = self._write_sdf(temp_dir, mesh_path, "2 3 4")

            combined = combine_sdf_meshes_at_joint_angles(sdf_path)

            np.testing.assert_allclose(
                combined.bounds[1] - combined.bounds[0],
                np.array([2.0, 3.0, 4.0]),
                atol=1e-6,
            )

    def test_combine_sdf_meshes_defaults_missing_visual_mesh_scale_to_unit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            mesh_path = self._write_unit_box_mesh(temp_dir)
            sdf_path = self._write_sdf(temp_dir, mesh_path)

            combined = combine_sdf_meshes_at_joint_angles(sdf_path)

            np.testing.assert_allclose(
                combined.bounds[1] - combined.bounds[0],
                np.array([1.0, 1.0, 1.0]),
                atol=1e-6,
            )

    def test_combine_sdf_meshes_applies_zup_link_pose_to_yup_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            mesh_path = self._write_unit_box_mesh(temp_dir)
            sdf_path = self._write_sdf(
                temp_dir,
                mesh_path,
                link_pose_text="0 0 2 0 0 0",
            )

            combined = combine_sdf_meshes_at_joint_angles(sdf_path)

            np.testing.assert_allclose(
                combined.bounds.mean(axis=0),
                np.array([0.0, 2.0, 0.0]),
                atol=1e-6,
            )

    def test_parse_mesh_scale_rejects_malformed_scale(self) -> None:
        mesh_elem = ET.fromstring("<mesh><scale>2 3</scale></mesh>")

        with self.assertRaises(SDFParseError):
            _parse_mesh_scale(mesh_elem, {})


if __name__ == "__main__":
    unittest.main()
