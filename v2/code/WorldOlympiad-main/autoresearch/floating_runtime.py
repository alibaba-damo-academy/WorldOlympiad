"""Isaac floating asset bridge with measured object-frame rigid compatibility.

No Isaac imports at module load; policy and simulator environments can share the
contract tests. Callers must provide authoritative initial joint positions and
must install contact reporting on rigid link paths, never the non-rigid parent.
"""
from types import SimpleNamespace
import numpy as np
from .floating_state import measured_floating_state
from .isaac_articulation import floating_tree_root
from .isaac_compiler import _safe_name


def floating_runtime_binding(body, initial_joint_positions, scene_path='/World/WorldIR/Scene'):
    root = floating_tree_root(body)
    links = {link['name']: link['prim_name'] for link in body['links']}
    joints = {joint['name']: joint for joint in body['joints']}
    if set(initial_joint_positions) != set(joints):
        raise ValueError('authoritative initial floating joint states are required for every joint')
    values = {}
    excursions = {}
    for name, joint in joints.items():
        value = float(initial_joint_positions[name])
        if not np.isfinite(value):
            raise ValueError('nonfinite initial floating joint state')
        # This is a measured-state replay, not an authored configuration
        # validator. Constraint solvers can leave real settled joints beyond a
        # limit. Preserve that evidence exactly: never clamp it or widen the
        # compiled physics limits. Record the excursion for runtime diagnostics.
        if joint['type'] != 'continuous':
            for key, op in [('lower', lambda x, y: x < y), ('upper', lambda x, y: x > y)]:
                if joint.get(key) is not None:
                    limit = float(joint[key])
                    if not np.isfinite(limit):
                        raise ValueError('nonfinite declared floating joint limit')
                    if op(value, limit):
                        excursions[name] = {'bound': key, 'limit': limit,
                                            'measured': value, 'delta': value - limit}
        values[_safe_name(name)] = value
    if len(values) != len(joints):
        raise ValueError('floating joint names alias after sanitization')
    parent = f"{scene_path}/{body['prim_name']}"
    return {'prim_path': parent, 'root_path': f'{parent}/{links[root]}',
            'rigid_link_paths': [f'{parent}/{links[name]}' for name in links],
            'initial_joint_positions': values,
            'initial_joint_limit_excursions': excursions}


class FloatingObject:
    """Wrap an Isaac Articulation; expose object-frame state, not root-link/COM.

    The underlying Articulation has no actuators. Its independently measured
    link poses and joint values are retained in measured_state for predicates.
    """
    def __init__(self, body, asset, initial_joint_positions):
        self.body = body
        self.asset = asset
        self.binding = floating_runtime_binding(body, initial_joint_positions)

    @property
    def device(self):
        return self.asset.device

    def initialize(self):
        import torch
        names = list(self.asset.joint_names)
        values = self.binding['initial_joint_positions']
        if set(names) != set(values) or len(names) != len(values):
            raise ValueError('live floating joints differ from compiled initial state')
        positions = torch.tensor([[values[name] for name in names]],
                                 device=self.device, dtype=self.asset.data.joint_pos.dtype)
        self.asset.write_joint_state_to_sim(positions, torch.zeros_like(positions))
        # No position target: passive mechanisms must stay passive.
        self.asset.update(0.)

    def update(self, dt):
        self.asset.update(dt)

    def measured_state(self):
        data = self.asset.data
        def cpu(value):
            return value[0].detach().cpu().numpy()
        state = measured_floating_state(self.body,
            link_names=list(self.asset.body_names),
            positions=cpu(data.body_link_pos_w),
            quaternions_wxyz=cpu(data.body_link_quat_w),
            linear_velocities=cpu(data.body_link_lin_vel_w),
            angular_velocities=cpu(data.body_link_ang_vel_w),
            joint_names=list(self.asset.joint_names),
            joint_positions=cpu(data.joint_pos), joint_velocities=cpu(data.joint_vel))
        from copy import deepcopy
        state['initial_joint_limit_excursions'] = deepcopy(
            self.binding['initial_joint_limit_excursions'])
        return state

    @property
    def data(self):
        import torch
        state = self.measured_state()
        def tensor(value):
            return torch.tensor([value], dtype=self.asset.data.body_link_pos_w.dtype,
                                device=self.device)
        q = state['orientation_xyzw']
        return SimpleNamespace(root_link_pos_w=tensor(state['position']),
            root_link_quat_w=tensor([q[3], q[0], q[1], q[2]]),
            root_link_lin_vel_w=tensor(state['linear_velocity']),
            root_link_ang_vel_w=tensor(state['angular_velocity']))


def create_scene_object(body, *, rigid_factory=None, floating_factory=None):
    """Dispatch compiled physics without treating a non-rigid parent as rigid.

    Floating trees require bound measured frame-zero joints and provenance;
    legacy absent measurements block initialization rather than imply zero.
    """
    parent = '/World/WorldIR/Scene/' + body['prim_name']
    if body.get('static'):
        raise ValueError('static bodies do not have dynamic runtime assets')
    if len(body['links']) > 1:
        metadata = body.get('metadata') or {}
        if not metadata.get('initial_joint_state_source'):
            raise ValueError('floating object lacks authoritative initial joint evidence')
        positions = metadata.get('initial_joint_positions')
        if not isinstance(positions, dict):
            raise ValueError('floating object lacks measured initial joints')
        floating_runtime_binding(body, positions)
        return (floating_factory or create_floating_object)(body, positions)
    if rigid_factory is None:
        from isaaclab.assets import RigidObject, RigidObjectCfg
        rigid_factory = lambda path: RigidObject(RigidObjectCfg(prim_path=path, spawn=None))
    return rigid_factory(parent)


def runtime_rigid_paths(body, asset):
    if isinstance(asset, FloatingObject):
        return list(asset.binding['rigid_link_paths'])
    if len(body['links']) != 1 or body.get('static'):
        raise ValueError('runtime rigid paths require a dispatched dynamic asset')
    return ['/World/WorldIR/Scene/' + body['prim_name']]


def runtime_object_state(asset, rigid_reader):
    if isinstance(asset, FloatingObject):
        state = asset.measured_state()
        state['link_position'] = state['position']
        state['link_orientation_xyzw'] = state['orientation_xyzw']
        return state
    return rigid_reader(asset)


def create_floating_object(body, initial_joint_positions):
    from isaaclab.assets import Articulation, ArticulationCfg
    binding = floating_runtime_binding(body, initial_joint_positions)
    # Articulation locates the root below this parent. Do not override authored
    # floating physics or add drives/world anchors via a spawn configuration.
    asset = Articulation(ArticulationCfg(prim_path=binding['prim_path'], spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=binding['initial_joint_positions'], joint_vel={'.*': 0.}),
        actuators={}))
    return FloatingObject(body, asset, initial_joint_positions)
