from types import SimpleNamespace
import numpy as np
import pytest
from autoresearch.floating_runtime import FloatingObject, floating_runtime_binding
from wo.tests.test_floating_state import fixture
from autoresearch.floating_runtime import create_scene_object, runtime_rigid_paths, runtime_object_state
from scripts.build_task_graph_worldir import initial_joint_metadata


def test_dispatch_rigid_single_link_and_preserve_rigid_state_reader():
    body = {'prim_name': 'box', 'links': [{}], 'static': False}
    asset = create_scene_object(body, rigid_factory=lambda path: ('rigid', path))
    assert asset == ('rigid', '/World/WorldIR/Scene/box')
    assert runtime_rigid_paths(body, asset) == ['/World/WorldIR/Scene/box']
    assert runtime_object_state(asset, lambda a: {'rigid': a}) == {'rigid': asset}


def test_dispatch_floating_requires_measured_provenance_and_does_not_call_rigid():
    body, _ = body_fixture()
    def forbid(path):
        raise AssertionError('floating parent cannot be rigid')
    with pytest.raises(ValueError, match='evidence'):
        create_scene_object(body, rigid_factory=forbid)
    body['metadata'] = initial_joint_metadata({'initial_joint_positions': {'slide': .2},
        'initial_joint_state_source': 'trajectory.json:frames[0].objects.b.joint_positions'})
    result = create_scene_object(body, rigid_factory=forbid,
        floating_factory=lambda b, p: (b, p))
    assert result == (body, {'slide': .2})
    body['metadata']['initial_joint_positions'] = {}
    with pytest.raises(ValueError, match='every joint'):
        create_scene_object(body, rigid_factory=forbid)


def test_initial_measurement_metadata_missing_is_not_zero_and_invalid_fails():
    assert initial_joint_metadata({}) == {}
    with pytest.raises(ValueError): initial_joint_metadata({'initial_joint_positions': {}})
    with pytest.raises(ValueError):
        initial_joint_metadata({'initial_joint_positions': {'j': float('nan')},
                                'initial_joint_state_source': 'measured'})


def test_runner_wires_dispatch_live_state_and_link_contacts_before_reset():
    # Isaac entry point cannot be imported by CPU pytest (starts SimulationApp).
    # Check the integration itself, in addition to mocked factory behavior.
    import ast
    from pathlib import Path
    tree = ast.parse(Path('scripts/run_cosmos3_worldir_isaac_closed_loop.py').read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    named = [node.func.id for node in calls if isinstance(node.func, ast.Name)]
    assert 'create_scene_object' in named
    assert 'runtime_rigid_paths' in named
    assert 'runtime_object_state' in named
    assert named.count('create_link_contact_sensors') == 3
    # The runner must not retain direct parent-based RigidObject construction.
    assert 'RigidObject' not in named
    assert any(isinstance(node.func, ast.Attribute) and node.func.attr == 'initialize'
               for node in calls)


class Tensor:
    dtype = 'unused'
    def __init__(self, value): self.value = np.asarray(value)
    def __getitem__(self, key): return Tensor(self.value[key])
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


def body_fixture():
    body, args = fixture()
    body.update(name='cup', prim_name='cup')
    body['joints'][0].update(lower=0., upper=.3)
    return body, args


def test_binding_uses_topological_root_all_links_and_explicit_initial_state():
    body, _ = body_fixture()
    binding = floating_runtime_binding(body, {'slide': .2})
    assert binding['root_path'] == '/World/WorldIR/Scene/cup/base'
    assert binding['rigid_link_paths'] == ['/World/WorldIR/Scene/cup/cap', '/World/WorldIR/Scene/cup/base']
    assert binding['initial_joint_positions'] == {'slide': .2}


@pytest.mark.parametrize('positions', [{}, {'other': 0}, {'slide': float('nan')}, {'slide': float('inf')}])
def test_initial_state_fail_closed(positions):
    body, _ = body_fixture()
    with pytest.raises(ValueError): floating_runtime_binding(body, positions)


@pytest.mark.parametrize('value,bound', [(-.0001367248832958126, 'lower'), (.4, 'upper')])
def test_measured_solver_excursion_is_preserved_not_clamped(value, bound):
    import copy
    body, _ = body_fixture()
    original = copy.deepcopy(body)
    binding = floating_runtime_binding(body, {'slide': value})
    assert binding['initial_joint_positions'] == {'slide': value}
    diagnostic = binding['initial_joint_limit_excursions']['slide']
    assert diagnostic['bound'] == bound
    assert diagnostic['measured'] == value
    assert diagnostic['delta'] == value - diagnostic['limit']
    assert body == original  # no physics-limit widening or geometry rewrite


def test_runtime_reads_live_link_frames_and_live_joints_not_root_com():
    body, args = body_fixture()
    data = SimpleNamespace(**{key: Tensor([args[value]]) for key, value in {
        'body_link_pos_w': 'positions', 'body_link_quat_w': 'quaternions_wxyz',
        'body_link_lin_vel_w': 'linear_velocities', 'body_link_ang_vel_w': 'angular_velocities',
        'joint_pos': 'joint_positions', 'joint_vel': 'joint_velocities'}.items()})
    # No root_pos_w or COM fields exist: reading those must not be possible.
    updates = []
    asset = SimpleNamespace(data=data, body_names=args['link_names'],
        joint_names=args['joint_names'], update=updates.append)
    wrapper = FloatingObject(body, asset, {'slide': .2})
    wrapper.update(.01)
    state = wrapper.measured_state()
    assert updates == [.01]
    assert state['position'] == pytest.approx([4, 5, 6])
    assert state['link_states']['cap']['position'] == [4, 5, 8]
    assert state['joint_states']['slide']['position'] == .2
    from autoresearch.cosmos_scene_state import live_object_records
    measured = runtime_object_state(wrapper, lambda a: pytest.fail('not a rigid asset'))
    records = live_object_records({'b': {}}, [SimpleNamespace(
        trajectory_object_name='b', body_name='cup', dynamic=True)], {'cup': measured})
    assert records['b']['link_states'] == state['link_states']
    assert records['b']['joint_positions'] == {'slide': .2}
    assert records['b']['measurement_source'] == 'native_floating_articulation_link_frames'
    measured['link_states']['cap']['position'][0] = 100
    assert records['b']['link_states']['cap']['position'][0] == 4
    asset.joint_names = ['wrong']
    with pytest.raises(ValueError): wrapper.measured_state()


def test_factory_no_spawn_drives_or_anchor_and_explicit_initial_joint_cfg(monkeypatch):
    import sys
    from autoresearch.floating_runtime import create_floating_object
    class Config(SimpleNamespace):
        InitialStateCfg = SimpleNamespace
    configs = []
    def articulation(cfg):
        configs.append(cfg)
        return SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'isaaclab', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'isaaclab.assets', SimpleNamespace(
        Articulation=articulation, ArticulationCfg=Config))
    body, _ = body_fixture()
    wrapped = create_floating_object(body, {'slide': .2})
    assert wrapped.body is body
    cfg = configs[0]
    assert cfg.prim_path == '/World/WorldIR/Scene/cup'
    assert cfg.spawn is None and cfg.actuators == {}
    assert cfg.init_state.joint_pos == {'slide': .2}
    assert cfg.init_state.joint_vel == {'.*': 0.}
