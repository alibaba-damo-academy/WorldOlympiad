import copy
from pathlib import Path

import pytest

from autoresearch.robotics_cosmos_registration import register_droid_frame_zero


def test_frame_zero_registration_checks_real_robot_fk(tmp_path: Path) -> None:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("""<robot name="minimal">
      <link name="root"/><link name="base_link"/>
      <joint name="joint1" type="revolute">
        <parent link="root"/><child link="base_link"/>
        <origin xyz="0 0 1" rpy="0 0 0"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" effort="1" velocity="1"/>
      </joint>
    </robot>""")
    trajectory = {
        "controller": {"robot_embodiment": "robolab_droid"},
        "frames": [{"frame": 0, "robot": {
            "joint_names": ["joint1"], "joint_positions": [0.0],
            "base_position": [0.0, 0.0, 0.0],
            "base_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "end_effector_position": [0.0, 0.0, 1.0],
        }}],
    }
    result = register_droid_frame_zero(trajectory, urdf)
    assert result["target_link_name"] == "base_link"
    assert result["fk_residual_m"] < 1.0e-4
    assert result["initial_joint_positions"] == {"joint1": 0.0}
    changed = copy.deepcopy(trajectory)
    changed["frames"][0]["robot"]["end_effector_position"][2] = 1.1
    with pytest.raises(ValueError, match="FK differs"):
        register_droid_frame_zero(changed, urdf)
