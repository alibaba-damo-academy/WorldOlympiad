import xml.etree.ElementTree as ET

import numpy as np
import pytest

from autoresearch.scenecode_code_object_urdf_frames import urdf_closed_link_frames


def mechanism(kind, limit=""):
    return ET.fromstring(f"""<robot name="fixture">
      <link name="base"/><link name="panel"/><link name="knob"/>
      <joint name="panel_mount" type="fixed">
        <parent link="base"/><child link="panel"/>
        <origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
      </joint>
      <joint name="knob_rotation" type="{kind}">
        <parent link="panel"/><child link="knob"/>
        <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>{limit}
      </joint>
    </robot>""")


@pytest.mark.parametrize("kind", ["fixed", "revolute", "prismatic", "continuous"])
def test_zero_coordinate_frames_preserve_rotated_parent_and_joint_xml(kind):
    limit = '<limit lower="-0.5" upper="0.7"/>' if kind in {"revolute", "prismatic"} else ""
    root = mechanism(kind, limit)
    before = ET.tostring(root)
    frames = urdf_closed_link_frames(root)
    np.testing.assert_allclose(frames["knob"][:3, 3], [1, 2.2, 3.1], atol=1e-12)
    np.testing.assert_allclose(frames["knob"][:3, :3],
                               [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert ET.tostring(root) == before


def test_continuous_effort_velocity_limit_has_no_position_bounds():
    root = mechanism("continuous", '<limit effort="2" velocity="1"/>')
    np.testing.assert_allclose(urdf_closed_link_frames(root)["knob"][:3, 3],
                               [1, 2.2, 3.1], atol=1e-12)


def test_bounded_joint_must_still_include_zero_coordinate():
    root = mechanism("revolute", '<limit lower="0.1" upper="0.7"/>')
    with pytest.raises(ValueError, match="zero-coordinate closed pose"):
        urdf_closed_link_frames(root)


def test_unsupported_multidof_joint_reports_type_and_name():
    with pytest.raises(ValueError, match="floating.*knob_rotation"):
        urdf_closed_link_frames(mechanism("floating"))
