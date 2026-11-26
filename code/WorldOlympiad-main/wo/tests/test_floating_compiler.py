import json
import pytest
from autoresearch.isaac_compiler import prepare_isaac_bundle
from autoresearch.models import WorldIR, BodyIR, LinkIR, JointIR, Inertial


def world(joint_type='prismatic', measured=True):
    inertia = Inertial(mass=0.2, inertia=(0.001, 0.002, 0.003, 0, 0, 0))
    return WorldIR(scene_id='mechanism', prompt='floating', bodies=[BodyIR(
        name='bottle', semantic_category='bottle', links=[
            LinkIR(name='cap', inertial=inertia),
            LinkIR(name='base', inertial=inertia if measured else None)],
        joints=[JointIR(name='cap_slide', type=joint_type, parent='base', child='cap',
                        lower=0, upper=0.04)])])


def test_prepare_keeps_movable_joint_and_all_link_measurements(tmp_path):
    path = prepare_isaac_bundle(world(), tmp_path)
    plan = json.loads(path.read_text())
    body = plan['bodies'][0]
    assert not body['static']
    assert body['joints'][0]['type'] == 'prismatic'
    assert body['joints'][0]['upper'] == 0.04
    assert {link['name']: link['inertial']['mass'] for link in body['links']} == {'base': 0.2, 'cap': 0.2}


@pytest.mark.parametrize('kind', ['revolute', 'continuous'])
def test_prepare_accepts_other_movable_joint_types(tmp_path, kind):
    path = prepare_isaac_bundle(world(kind), tmp_path)
    assert json.loads(path.read_text())['bodies'][0]['joints'][0]['type'] == kind


def test_missing_measured_mass_fails_closed(tmp_path):
    with pytest.raises(ValueError, match='measured positive'):
        prepare_isaac_bundle(world(measured=False), tmp_path)


def test_uncollapsed_fixed_subtree_fails_closed(tmp_path):
    with pytest.raises(ValueError, match='movable tree'):
        prepare_isaac_bundle(world('fixed'), tmp_path)
