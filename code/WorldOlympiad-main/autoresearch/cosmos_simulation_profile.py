"""Reuse an exported RoboLab simulation configuration without hidden overrides."""
from __future__ import annotations

import json
from pathlib import Path


def read_simulation_profile(path: str | Path) -> dict:
    profile = json.loads(Path(path).read_text())
    values = profile['sim']
    if abs(float(values['dt']) - 1 / 120) > 1e-10 or values['render_interval'] != 8:
        raise ValueError('RoboLab profile must preserve 120 Hz physics and 15 Hz control')
    if not isinstance(values.get('device'), str):
        raise ValueError('RoboLab profile must declare its simulation device')
    material_function = values.get('physics_material', {}).get('func')
    if material_function not in (None, 'isaaclab.sim.spawners.materials.physics_materials:spawn_rigid_body_material'):
        raise ValueError('unsupported custom physics material function in RoboLab profile')
    return values


def build_simulation_config(sim_utils, path: str | Path):
    """Call after AppLauncher initialized Isaac; retain every exported setting."""
    values = read_simulation_profile(path)
    material = values.pop('physics_material')
    material.pop('func', None)
    values['physics_material'] = sim_utils.RigidBodyMaterialCfg(**material)
    values['physx'] = sim_utils.PhysxCfg(**values['physx'])
    values['render'] = sim_utils.RenderCfg(**values['render'])
    return sim_utils.SimulationCfg(**values)
