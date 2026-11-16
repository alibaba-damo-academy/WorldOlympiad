import numpy as np
import pytest
from autoresearch.pybullet_frame_contract import measured_base_link_pose, require_root_link_measurement, reset_base_link_pose
from autoresearch.models import BodyIR, LinkIR


def test_real_bullet_root_pose_undoes_rotated_inertial_frame(tmp_path):
    p = pytest.importorskip("pybullet")
    path = tmp_path / "offset.urdf"
    path.write_text('''<robot name="offset"><link name="root"><inertial>
    <origin xyz="0.08 -0.05 0.378" rpy="0.2 -0.1 0.3"/><mass value="8"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.3"/>
    </inertial></link></robot>''')
    client = p.connect(p.DIRECT)
    try:
        position = [1.2, -0.4, 0.007]
        orientation = p.getQuaternionFromEuler([0.1, -0.2, 0.7])
        body = p.loadURDF(str(path), position, orientation)
        receipt = measured_base_link_pose(p, body)
        assert receipt["pose_frame"] == "link"
        assert np.allclose(receipt["position"], position, atol=2e-6)
        assert abs(np.dot(receipt["orientation_xyzw"], orientation)) == pytest.approx(1, abs=2e-6)
        assert np.linalg.norm(np.asarray(receipt["base_com_position"]) - position) > .3
        moved = [-.3, .8, .012]
        moved_rotation = p.getQuaternionFromEuler([-.2, .15, -.4])
        reset_base_link_pose(p, body, moved, moved_rotation, physicsClientId=client)
        after = measured_base_link_pose(p, body)
        assert np.allclose(after["position"], moved, atol=2e-6)
        assert abs(np.dot(after["orientation_xyzw"], moved_rotation)) == pytest.approx(1, abs=2e-6)
    finally:
        p.disconnect(client)


def test_unlabelled_root_measurement_is_rejected():
    body = BodyIR(name="host", semantic_category="cabinet", links=[LinkIR(name="root")])
    with pytest.raises(ValueError, match="root pose is ambiguous"):
        require_root_link_measurement(body, "root", {"position": [0, 0, .378]})
    require_root_link_measurement(body, "root", {"pose_frame": "link"})
