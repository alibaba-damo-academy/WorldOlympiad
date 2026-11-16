import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from autoresearch.cosmos_task_graph_predicates import local_mesh


def fixture():
    box = {'kind': 'box', 'size': [.1, .1, .1]}
    body = {'static': False, 'joints': [{'name': 'slide'}], 'links': [
        {'name': 'base', 'pose': {'xyz': [0, 0, 0]}, 'collisions': [box]},
        {'name': 'cap', 'pose': {'xyz': [.2, 0, 0]}, 'collisions': [box]}]}
    r = Rotation.from_euler('z', np.pi/2)
    p = np.array([1., 2., 3.])
    state = {'position': p, 'orientation_xyzw': r.as_quat(), 'link_states': {
        'base': {'position': p, 'orientation_xyzw': r.as_quat()},
        'cap': {'position': p+r.apply([.7, 0, 0]), 'orientation_xyzw': r.as_quat()}}}
    return body, state


def test_measured_child_displacement_not_zero_joint_assembly(tmp_path):
    body, state = fixture()
    mesh = local_mesh(body, tmp_path, role='collisions', state=state)
    np.testing.assert_allclose(mesh.bounds, [[-.05, -.05, -.05], [.75, .05, .05]], atol=1e-12)
    authored = local_mesh(body, tmp_path, role='collisions')
    assert authored.bounds[1, 0] == pytest.approx(.25)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'nonfinite'])
def test_live_link_measurements_fail_closed(tmp_path, mutation):
    body, state = fixture()
    if mutation == 'missing':
        del state['link_states']['cap']
    elif mutation == 'extra':
        state['link_states']['other'] = state['link_states']['base']
    else:
        state['link_states']['cap']['position'] = [np.nan, 0, 0]
    with pytest.raises(ValueError):
        local_mesh(body, tmp_path, role='collisions', state=state)
