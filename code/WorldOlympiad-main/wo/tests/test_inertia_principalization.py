import numpy as np
import pytest
from autoresearch.compiler import _principalize
from autoresearch.models import Inertial


@pytest.mark.parametrize('scale', [1., 1e-10, 1e-20])
def test_small_positive_tensor_retains_measured_inertia(scale):
    tensor = np.array([[2., .3, 0.], [.3, 3., .2], [0., .2, 4.]]) * scale
    result, pose, changed = _principalize(Inertial(mass=.001, inertia=(
        tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2])))
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_quat(pose.quat_xyzw).as_matrix()
    reconstructed = rotation @ np.diag(result.inertia[:3]) @ rotation.T
    np.testing.assert_allclose(reconstructed / scale, tensor / scale, rtol=1e-14, atol=1e-14)
    assert result.mass == .001


@pytest.mark.parametrize('values', [(0., 1e-10, 1e-10), (-1e-12, 1e-10, 1e-10), (0., 0., 0.)])
def test_nonpositive_tensor_rejected(values):
    with pytest.raises(ValueError, match='positive definite'):
        _principalize(Inertial(mass=.001, inertia=(*values, 0., 0., 0.)))


def test_floating_compiler_accepts_small_measured_link(tmp_path):
    import json
    from autoresearch.isaac_compiler import prepare_isaac_bundle
    from wo.tests.test_floating_compiler import world
    scene = world()
    small = (1.88242854866e-10, 1.88242854866e-10, 2.48041753624e-10)
    scene.bodies[0].links[0].inertial = Inertial(mass=.001, inertia=(*small, 0., 0., 0.))
    plan = json.loads(prepare_isaac_bundle(scene, tmp_path).read_text())
    link = plan['bodies'][0]['links'][0]
    np.testing.assert_allclose(link['inertial']['inertia'][:3], small, rtol=1e-14, atol=0.)
