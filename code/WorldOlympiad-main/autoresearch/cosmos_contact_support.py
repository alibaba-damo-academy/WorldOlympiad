"""Native support evidence from PhysX per-contact normal-force buffers.

Normal vectors point from the filtered body toward the sensor body. Match
wo.predicates.support_below, including its normal.z > 0.5 rule; never infer
support from geometry or an aggregate force vector.
"""
import numpy as np


def create_link_contact_sensors(stage, rigid_link_paths, filter_paths, *,
                                track_contact_points=False, sensor_factory=None,
                                config_factory=None, report_api=None):
    """Bind each real rigid link separately, preserving the same filter order."""
    paths = list(rigid_link_paths)
    filters = list(filter_paths)
    if not paths or len(set(paths)) != len(paths) or not filters:
        raise ValueError('unique rigid link paths and nonempty contact filters required')
    if sensor_factory is None or config_factory is None or report_api is None:
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from pxr import PhysxSchema
        sensor_factory = sensor_factory or ContactSensor
        config_factory = config_factory or ContactSensorCfg
        report_api = report_api or PhysxSchema.PhysxContactReportAPI
    from pxr import UsdPhysics
    prims = [stage.GetPrimAtPath(path) for path in paths]
    # Validate all paths before installing any APIs. Non-rigid parents cannot
    # emit valid contact reports for a floating articulation.
    if any(not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims):
        raise ValueError('contact sensor path must be an authored rigid link')
    sensors = []
    for path, prim in zip(paths, prims):
        report_api.Apply(prim).CreateThresholdAttr().Set(0.)
        sensors.append(sensor_factory(config_factory(prim_path=path,
            filter_prim_paths_expr=filters, track_contact_points=track_contact_points,
            max_contact_data_count_per_prim=max(1024, 64 * len(filters)))))
    return LinkContactSensors(sensors)


class LinkContactSensors:
    """One filtered sensor per rigid link, never a many-body filtered view.

    Keep force rows separate: opposite link forces must not cancel when
    deciding contact. Support loads are aggregated only after reading each
    link's measured per-contact normals (see read_contact_support).
    """
    def __init__(self, sensors):
        self.sensors = tuple(sensors)
        if not self.sensors:
            raise ValueError('contact group requires at least one rigid link sensor')

    def update(self, dt, **kwargs):
        for sensor in self.sensors:
            sensor.update(dt, **kwargs)

    @property
    def data(self):
        from types import SimpleNamespace
        import torch
        matrices = [sensor.data.force_matrix_w for sensor in self.sensors]
        if any(value is None for value in matrices):
            raise ValueError('missing measured link contact force matrix')
        if any(not torch.isfinite(value).all() for value in matrices):
            raise ValueError('nonfinite measured link contact force matrix')
        if any(value.ndim != 4 or value.shape[1] != 1 for value in matrices):
            raise ValueError('filtered contact sensor must represent exactly one rigid link')
        if len({(value.shape[0], value.shape[2], value.shape[3]) for value in matrices}) != 1:
            raise ValueError('link contact filters differ')
        return SimpleNamespace(force_matrix_w=torch.cat(matrices, dim=1))


def read_contact_support(sensor, filter_names, dt):
    """Read terminal PhysX contact buffers, refusing possible saturation."""
    if isinstance(sensor, LinkContactSensors):
        evidence = [read_contact_support(child, filter_names, dt) for child in sensor.sensors]
        loads, touching = {}, set()
        for item in evidence:
            touching.update(item['touching_bodies'])
            for name, value in item['upward_load_by_body_n'].items():
                loads[name] = loads.get(name, 0.) + value
        return {'touching_bodies': sorted(touching), 'upward_load_by_body_n': loads,
                'supported_by': max(loads, key=loads.get) if loads else None,
                'authority': 'physx_per_contact_normal_force'}
    data = sensor.contact_physx_view.get_contact_data(dt=dt)
    forces, _, normals, _, counts, starts = [x.detach().cpu().numpy() for x in data]
    if int(np.asarray(counts).sum()) >= np.asarray(forces).size:
        raise ValueError('native support contact buffer saturated')
    return contact_support_evidence(forces, normals, counts, starts, filter_names)


def contact_support_evidence(forces, normals, counts, starts, filter_names):
    forces = np.asarray(forces, dtype=float).reshape(-1)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    counts = np.asarray(counts).reshape(-1)
    starts = np.asarray(starts).reshape(-1)
    if len(counts) != len(filter_names) or len(starts) != len(counts):
        raise ValueError('contact filters are not bound one-to-one to names')
    loads, touching = {}, set()
    for name, count, start in zip(filter_names, counts, starts):
        if int(count) != count or int(start) != start or count < 0 or start < 0:
            raise ValueError('invalid native contact buffer range')
        count, start = int(count), int(start)
        if start + count > min(len(forces), len(normals)):
            raise ValueError('truncated native contact buffer')
        if not (np.isfinite(forces[start:start+count]).all()
                and np.isfinite(normals[start:start+count]).all()):
            raise ValueError('nonfinite native contact data')
        if count:
            touching.add(name)
        for force, normal in zip(forces[start:start+count], normals[start:start+count]):
            if normal[2] > 0.5:
                loads[name] = loads.get(name, 0.0) + max(float(force), 1e-6) * float(normal[2])
    return {'touching_bodies': sorted(touching), 'upward_load_by_body_n': loads,
            'supported_by': max(loads, key=loads.get) if loads else None,
            'authority': 'physx_per_contact_normal_force'}
