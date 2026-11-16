"""Content-bound actual expert rigid dynamics for the shared RoboTwin importer."""
import json
from pathlib import Path
import numpy as np
from wo.expert.initialization_check import load_bound_supplement, sha


def validate_rigid(row):
    for key, size in [('inertia_diagonal_kg_m2', 3), ('inertial_position', 3),
                      ('inertial_orientation_xyzw', 4)]:
        v = np.asarray(row[key], dtype=float)
        if v.shape != (size,) or not np.isfinite(v).all():
            raise ValueError('invalid measured dynamics ' + key)
    mass = float(row['mass_kg'])
    if not np.isfinite(mass) or mass <= 0 or np.any(np.asarray(row['inertia_diagonal_kg_m2']) <= 0):
        raise ValueError('nonpositive measured rigid dynamics')
    if abs(np.linalg.norm(row['inertial_orientation_xyzw']) - 1) > 1e-6:
        raise ValueError('nonunit measured inertial orientation')
    return row


def bind(ref, out, objects):
    ref, out = Path(ref), Path(out)
    candidates = []
    for path in sorted((ref.parent.parent / 'expert_checks' / ref.name).glob('*/initialization.json')):
        if json.loads(path.read_text()).get('measured_dynamics'):
            candidates.append(path)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError('ambiguous measured dynamics evidence')
    source = candidates[0]
    raw = source.read_bytes()
    data = load_bound_supplement(ref, source)
    if json.loads(raw) != data or source.read_bytes() != raw:
        raise ValueError('measured dynamics evidence changed during binding')
    archive = out / 'lingbot/initialization_dynamics.json'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and archive.read_bytes() != raw:
        raise ValueError('immutable dynamics archive changed')
    archive.write_bytes(raw)
    for obj in objects:
        if obj['fixed']:
            continue
        links = data['measured_dynamics'][obj['object_id']]
        if obj.get('urdf') or len(links) != 1:
            raise ValueError('measured multi-link dynamics consumer pending')
        obj['measured_dynamics'] = validate_rigid(next(iter(links.values())))
        obj['mass'] = obj['measured_dynamics']['mass_kg']
    return {'archive': str(archive.resolve()), 'sha256': sha(archive),
            'source': str(source.resolve()), 'source_sha256': sha(source),
            'protocol': data['dynamics_protocol']}


def verify(plan):
    binding = plan.get('dynamics_evidence')
    if binding is None:
        return
    archive = Path(binding['archive'])
    if sha(archive) != binding['sha256']:
        raise ValueError('native measured dynamics archive changed')
    data = json.loads(archive.read_text())
    for obj in plan['objects']:
        if obj['fixed']:
            continue
        links = data['measured_dynamics'][obj['object_id']]
        if len(links) != 1 or obj.get('urdf'):
            raise ValueError('unsupported measured dynamics representation')
        expected = validate_rigid(next(iter(links.values())))
        if obj.get('measured_dynamics') != expected:
            raise ValueError('native measured dynamics plan mismatch')


def apply(builder, row, pose_factory):
    row = validate_rigid(row)
    x, y, z, w = row['inertial_orientation_xyzw']
    builder.set_mass_and_inertia(row['mass_kg'],
        pose_factory(p=row['inertial_position'], q=[w, x, y, z]),
        row['inertia_diagonal_kg_m2'])


def _bindable_single_actor_rows(rows):
    """Validate rows that were flattened into one native static actor.

    RoboTwin's table/support import can flatten a multi-link static URDF into
    one actor.  There is no per-link inertia or material target in that
    representation, so accepting arbitrary rows would silently lose physical
    evidence.  Static rows are safe only when they agree on the contact
    material; dynamic or conflicting rows remain an explicit import error.
    """
    if not rows:
        raise ValueError('native actor has no reference dynamics links')
    values = []
    for row in rows.values():
        if float(row.get('mass_kg', 0.0)) != 0.0:
            raise ValueError('dynamic multi-link reference flattened into one native actor')
        values.append((float(row['lateral_friction']), float(row['restitution'])))
    if any(v != values[0] for v in values[1:]):
        raise ValueError('flattened static links have conflicting contact materials')
    return next(iter(rows))


def apply_loaded(scene, handle, rows, pose_factory):
    """Bind every imported link's measured inertia and collision material."""
    from sapien import physx
    if hasattr(handle, 'get_links'):
        entities = {link.entity.get_name(): link.entity for link in handle.get_links()}
        if set(entities) != set(rows):
            raise ValueError('native/reference dynamics link identities differ')
    else:
        # A single-link actor is also exposed without get_links by SAPIEN.
        # Bind its measured dynamics normally. Only collapse multiple
        # reference links when they are genuinely static and share material;
        # dynamic multi-link evidence must not be discarded.
        if len(rows) == 1:
            name = next(iter(rows))
        else:
            name = _bindable_single_actor_rows(rows)
            first = rows[name]
            rows = {name: first}
        entities = {name: handle}
    report = {}
    for name, ent in entities.items():
        row = rows[name]
        components = [c for c in ent.get_components() if isinstance(c, physx.PhysxRigidBaseComponent)]
        if len(components) != 1:
            raise ValueError('missing unambiguous native physical component')
        component = components[0]
        if row['mass_kg'] > 0:
            validate_rigid(row)
            if not all(hasattr(component, method) for method in ('set_mass','set_inertia','set_cmass_local_pose')):
                raise ValueError('dynamic reference link imported as a static native actor')
            x,y,z,w = row['inertial_orientation_xyzw']
            component.set_cmass_local_pose(pose_factory(p=row['inertial_position'],q=[w,x,y,z]))
            component.set_mass(row['mass_kg'])
            component.set_inertia(row['inertia_diagonal_kg_m2'])
        friction, restitution = float(row['lateral_friction']), float(row['restitution'])
        if not np.isfinite([friction,restitution]).all() or friction < 0 or not 0 <= restitution <= 1:
            raise ValueError('invalid reference contact parameters')
        material = scene.create_physical_material(friction,friction,restitution)
        shapes = component.get_collision_shapes()
        for shape in shapes:
            shape.set_physical_material(material)
        actual_materials = [{'static_friction':float(s.get_physical_material().static_friction),
                             'dynamic_friction':float(s.get_physical_material().dynamic_friction),
                             'restitution':float(s.get_physical_material().restitution)} for s in shapes]
        if any(abs(m['static_friction']-friction)>1e-6 or abs(m['dynamic_friction']-friction)>1e-6
               or abs(m['restitution']-restitution)>1e-6 for m in actual_materials):
            raise ValueError('native contact material binding failed')
        report[name] = {'reference':row,'collision_materials':actual_materials,
                        'unsupported_contact_terms':{key:row[key] for key in ('rolling_friction','spinning_friction') if row.get(key)},
                        'mass_kg':float(component.get_mass()) if hasattr(component,'get_mass') else None,
                        'inertia_diagonal_kg_m2':component.get_inertia().tolist() if hasattr(component,'get_inertia') else None,
                        'inertial_pose':{'p':component.get_cmass_local_pose().p.tolist(),
                                         'q_wxyz':component.get_cmass_local_pose().q.tolist()} if hasattr(component,'get_cmass_local_pose') else None}
        if row['mass_kg'] > 0 and (abs(report[name]['mass_kg']-row['mass_kg']) > 1e-6 or
            not np.allclose(report[name]['inertia_diagonal_kg_m2'],row['inertia_diagonal_kg_m2'],rtol=1e-5,atol=1e-9)):
            raise ValueError('native mass/inertia binding failed')
        if row['mass_kg'] > 0:
            local = report[name]['inertial_pose']
            q_actual = np.asarray(local['q_wxyz'])[[1,2,3,0]]
            if not np.allclose(local['p'],row['inertial_position'],atol=1e-7,rtol=1e-5) or abs(np.dot(q_actual,row['inertial_orientation_xyzw'])) < 1-1e-6:
                raise ValueError('native inertial frame binding failed')
    return report
