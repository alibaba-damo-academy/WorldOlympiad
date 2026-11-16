"""Import contracts only: no claim of native simulator execution."""
import math
import numpy as np
import pytest
from autoresearch.models import Pose
from autoresearch.urdf_importer import import_urdf_body
from autoresearch.worldir_fixed_joint_collapse import collapse_fixed_joint_tree


INERTIAL = '<inertial><origin xyz="0.01 0 0" rpy="0 0 1.5707963267948966"/><mass value="0.2"/><inertia ixx="1" iyy="2" izz="3"/></inertial>'

def source(tmp_path, bad=None):
    links = ''.join(f'<link name="{name}">{INERTIAL}<collision><origin xyz="0.02 0 0"/><geometry><box size="0.04 0.03 0.02"/></geometry></collision></link>' for name in ['cap', 'fixed', 'base'])
    joints = '''<joint name="fixed_joint" type="fixed"><parent link="base"/><child link="fixed"/><origin xyz="0.1 0 0" rpy="0 0 1.5707963267948966"/></joint>
    <joint name="slide" type="prismatic"><parent link="fixed"/><child link="cap"/><origin xyz="0.03 0 0"/><axis xyz="1 0 0"/><limit lower="0" upper="0.04" effort="3" velocity="0.1"/></joint>'''
    xml = f'<robot name="test">{links}{joints}</robot>'
    if bad == 'missing_mass': xml = xml.replace(INERTIAL, '', 1)
    if bad == 'zero_mass': xml = xml.replace('value="0.2"', 'value="0"', 1)
    if bad == 'unknown': xml = xml.replace('parent link="fixed"', 'parent link="unknown"')
    if bad == 'cycle': xml = xml.replace('parent link="base"', 'parent link="cap"')
    if bad == 'duplicate': xml = xml.replace('name="cap"', 'name="base"', 1)
    path = tmp_path / 'object.urdf'; path.write_text(xml)
    return path


def test_floating_import_preserves_mechanism_and_model_frames(tmp_path):
    body = import_urdf_body(source(tmp_path), body_name='object', pose=Pose(xyz=(4, 5, 6)), static=False, semantic_category='bottle')
    assert not body.static and not body.metadata['fixed_base']
    assert 'end_effector_link_name' not in body.metadata
    links = {link.name: link for link in body.links}
    np.testing.assert_allclose(links['base'].pose.xyz, (0, 0, 0))
    np.testing.assert_allclose(links['cap'].pose.xyz, (0.1, 0.03, 0), atol=1e-12)
    np.testing.assert_allclose(links['cap'].inertial.inertia[:3], (2, 1, 3), atol=1e-12)
    assert links['cap'].collisions[0].pose.xyz == (0.02, 0, 0)
    assert body.joints[1].pose == links['cap'].pose
    assert body.joints[1].axis == (1, 0, 0)
    assert body.joints[1].upper == 0.04
    collapse_fixed_joint_tree(body)
    assert len(body.joints) == 1 and body.joints[0].type == 'prismatic'
    assert {link.name for link in body.links} == {'base', 'cap'}
    assert sum(link.inertial.mass for link in body.links) == pytest.approx(0.6)


@pytest.mark.parametrize('bad', ['missing_mass', 'zero_mass', 'unknown', 'cycle', 'duplicate'])
def test_dynamic_import_fails_closed(tmp_path, bad):
    with pytest.raises(ValueError):
        import_urdf_body(source(tmp_path, bad), body_name='object', pose=Pose(), static=False)


def test_legacy_robot_defaults_stay_fixed(tmp_path):
    body = import_urdf_body(source(tmp_path), body_name='robot', pose=Pose())
    assert body.static and body.semantic_category == 'robot_arm'
    assert body.metadata['end_effector_link_name'] == 'panda_grasptarget'
