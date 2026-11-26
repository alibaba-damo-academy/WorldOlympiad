from types import SimpleNamespace
import sys
import pytest
import numpy as np

class Array(np.ndarray):
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return np.asarray(self)

def tensor(value):
    return np.asarray(value).view(Array)

torch = SimpleNamespace(tensor=tensor, zeros=lambda *shape: tensor(np.zeros(shape)),
    isfinite=np.isfinite, cat=lambda values, dim: tensor(np.concatenate(values, axis=dim)),
    linalg=SimpleNamespace(vector_norm=np.linalg.norm))

@pytest.fixture(autouse=True)
def fake_tensor_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', torch)
from autoresearch.cosmos_contact_support import LinkContactSensors, read_contact_support


class Sensor:
    def __init__(self, force, normal=(0., 0., 1.)):
        self.data = SimpleNamespace(force_matrix_w=torch.tensor([[[[0., 0., force]]]]))
        self.calls = []
        self.contact_physx_view = SimpleNamespace(get_contact_data=lambda dt: (
            torch.tensor([abs(force), 0.]), torch.zeros(2, 3),
            torch.tensor([normal, (0., 0., 0.)]), torch.zeros(2),
            torch.tensor([[1]]), torch.tensor([[0]])))

    def update(self, dt, **kw):
        self.calls.append((dt, kw))


def test_factory_binds_real_links_only(monkeypatch):
    from autoresearch.cosmos_contact_support import create_link_contact_sensors
    rigid_api = object()
    monkeypatch.setitem(sys.modules, 'pxr', SimpleNamespace(
        UsdPhysics=SimpleNamespace(RigidBodyAPI=rigid_api)))
    reports, configs = [], []
    def prim(path):
        return SimpleNamespace(IsValid=lambda: True,
            HasAPI=lambda api: api is rigid_api and path != '/parent')
    stage = SimpleNamespace(GetPrimAtPath=prim)
    report = SimpleNamespace(Apply=lambda p: SimpleNamespace(CreateThresholdAttr=lambda:
        SimpleNamespace(Set=lambda value: reports.append(value))))
    def factory(config):
        configs.append(config)
        return Sensor(1.)
    kwargs = dict(sensor_factory=factory, config_factory=SimpleNamespace, report_api=report)
    with pytest.raises(ValueError, match='authored rigid link'):
        create_link_contact_sensors(stage, ['/a', '/parent'], ['/table'], **kwargs)
    assert not reports and not configs
    group = create_link_contact_sensors(stage, ['/a', '/b'], ['/table', '/robot'],
                                       track_contact_points=True, **kwargs)
    assert len(group.sensors) == 2
    assert [cfg.prim_path for cfg in configs] == ['/a', '/b']
    assert all(cfg.filter_prim_paths_expr == ['/table', '/robot'] for cfg in configs)
    assert all(cfg.track_contact_points for cfg in configs)
    assert reports == [0., 0.]


def test_opposed_link_forces_do_not_cancel():
    a, b = Sensor(3.), Sensor(-3.)
    group = LinkContactSensors([a, b])
    assert torch.linalg.vector_norm(group.data.force_matrix_w) > 4.
    group.update(.01, force_recompute=True)
    assert a.calls == b.calls == [(.01, {'force_recompute': True})]


def test_support_aggregates_normals_not_resultants():
    group = LinkContactSensors([Sensor(3.), Sensor(-3., (0., 0., -1.))])
    evidence = read_contact_support(group, ['table'], .01)
    assert evidence['upward_load_by_body_n'] == {'table': 3.}
    assert evidence['supported_by'] == 'table'
    assert evidence['touching_bodies'] == ['table']


def test_all_supporting_links_contribute():
    evidence = read_contact_support(LinkContactSensors([Sensor(3.), Sensor(2.)]), ['table'], .01)
    assert evidence['upward_load_by_body_n'] == {'table': 5.}


def test_missing_measurement_fails_closed():
    a = Sensor(1.)
    a.data.force_matrix_w = None
    with pytest.raises(ValueError, match='missing measured'):
        _ = LinkContactSensors([a]).data
    with pytest.raises(ValueError):
        LinkContactSensors([])


def test_nonfinite_and_many_body_views_refused():
    a = Sensor(float('nan'))
    with pytest.raises(ValueError, match='nonfinite'):
        _ = LinkContactSensors([a]).data
    a.data.force_matrix_w = torch.zeros(1, 2, 1, 3)
    with pytest.raises(ValueError, match='exactly one'):
        _ = LinkContactSensors([a]).data
