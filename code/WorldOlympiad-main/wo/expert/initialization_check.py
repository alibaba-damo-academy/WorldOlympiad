"""Isolated measured initialization replay of an archived expert, never a reference rewrite."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from wo.scene import SceneInputs, SceneObject
from wo.sim.world import World, PreparedBody, DROID_HOME


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scene(data):
    def obj(row):
        row = dict(row)
        row['sdf_path'] = Path(row['sdf_path'])
        return SceneObject(**row)
    return SceneInputs(case_id=data['case_id'], support=obj(data['support']),
                       objects={k: obj(v) for k, v in data['objects'].items()},
                       robot_urdf=Path(data['robot_urdf']),
                       robot_base_position=data['robot_base_position'],
                       robot_base_yaw=data['robot_base_yaw'], source=data.get('source', {}),
                       articulations=data.get('articulations', {}))


def pose_compatibility(expected, actual):
    if set(expected) != set(actual):
        raise ValueError('initialization object identity mismatch')
    evidence = {}
    for oid, old in expected.items():
        new = actual[oid]
        p0, p1 = np.asarray(old['position']), np.asarray(new['position'])
        q0, q1 = np.asarray(old['orientation_xyzw']), np.asarray(new['orientation_xyzw'])
        if not all(np.isfinite(v).all() for v in (p0, p1, q0, q1)):
            raise ValueError('nonfinite initialization pose')
        if any(abs(np.linalg.norm(q) - 1) > 1e-5 for q in (q0, q1)):
            raise ValueError('nonunit initialization orientation')
        distance = float(np.linalg.norm(p0 - p1))
        angle = float(2 * np.arccos(np.clip(abs(np.dot(q0, q1)), 0, 1)))
        evidence[oid] = {'position_delta_m': distance, 'orientation_delta_rad': angle,
                         'compatible': distance <= 1e-5 and angle <= 1e-4}
    return evidence


def asset_closure(path, manifest):
    path = Path(path).resolve()
    if str(path) in manifest:
        return
    manifest[str(path)] = sha(path)
    if path.suffix.lower() not in {'.urdf', '.sdf', '.xml'}:
        return
    root = ET.parse(path).getroot()
    for node in root.iter():
        value = node.get('filename') or (node.text.strip() if node.tag == 'uri' and node.text else None)
        if value:
            if '://' in value:
                raise ValueError(f'unresolved asset URI: {value}')
            asset_closure(path.parent / value, manifest)


def measured_dynamics(world):
    """Read actual post-override simulator parameters, indexed by physical link name.

    Inertia is expressed in each link's measured local inertial frame, not in
    the mesh frame. Consumers must preserve that frame rather than reuse an
    authored mass/inertia or recompute an approximation.
    """
    p, c = world.p, world.cid
    result = {}
    for oid, body in world.bodies.items():
        links = {}
        for index in range(-1, p.getNumJoints(body, physicsClientId=c)):
            name = (p.getBodyInfo(body, physicsClientId=c)[0] if index == -1 else
                    p.getJointInfo(body, index, physicsClientId=c)[12]).decode()
            if not name or name in links:
                raise ValueError('ambiguous measured dynamics link identity')
            d = p.getDynamicsInfo(body, index, physicsClientId=c)
            row = dict(mass_kg=float(d[0]), lateral_friction=float(d[1]),
                       inertia_diagonal_kg_m2=list(d[2]),
                       inertial_position=list(d[3]), inertial_orientation_xyzw=list(d[4]),
                       restitution=float(d[5]), rolling_friction=float(d[6]),
                       spinning_friction=float(d[7]))
            numbers = [row['mass_kg'], row['lateral_friction'], *d[2], *d[3], *d[4],
                       row['restitution'], row['rolling_friction'], row['spinning_friction']]
            if not np.isfinite(numbers).all() or row['mass_kg'] < 0 or any(v < 0 for v in d[2]):
                raise ValueError('invalid measured dynamics')
            links[name] = row
        result[oid] = links
    return result


def run(ref, out, task):
    ref, out, task = Path(ref).resolve(), Path(out).resolve(), Path(task).resolve()
    if out == ref or ref in out.parents:
        raise ValueError('supplement must not write within canonical reference')
    out.mkdir(parents=True, exist_ok=False)
    cert = json.loads((ref / 'certificate.json').read_text())
    if not cert.get('success') or sha(task) != cert['task_sha256']:
        raise ValueError('canonical task/reference binding mismatch')
    scene = load_scene(json.loads((ref / 'scene_inputs.json').read_text()))
    manifest = {str(ref / name): sha(ref / name) for name in
                ('certificate.json', 'scene_inputs.json', 'trajectory.json')}
    manifest[str(task)] = sha(task)
    prepared = {}
    for obj in [scene.support, *scene.objects.values()]:
        key = ref / 'physics_sdf' / f'{obj.object_id}_s{obj.scale:.3f}'
        path = Path((key / '.path').read_text().strip())
        # Never regenerate physics from SDF or choose a globbed variant.
        if ref not in path.resolve().parents:
            raise ValueError('prepared physics is not canonical-reference owned')
        prepared[obj.object_id] = PreparedBody(obj.object_id, path, {})
        manifest[str(key / '.path')] = sha(key / '.path')
        asset_closure(path, manifest)
    asset_closure(scene.robot_urdf, manifest)
    world = World(scene, prepared, mode='dynamic')
    try:
        world.step_physics(240)
        world.record(commanded=[*DROID_HOME, 0.0])
        frame = world.frames[0]
        dynamics = measured_dynamics(world)
    finally:
        world.close()
    original = json.loads((ref / 'trajectory.json').read_text())['frames'][0]
    comparison = pose_compatibility(original['objects'], frame['objects'])
    # Recheck every source after simulation; an in-flight reference rewrite is invalid.
    if any(sha(path) != digest for path, digest in manifest.items()):
        raise ValueError('source evidence changed during initialization replay')
    result = {'schema_version': 'wo_expert_initialization_check.v1',
              'canonical_reference': str(ref), 'source_sha256': manifest,
              'protocol': 'exact archived SceneInputs/PreparedBody; dynamic PyBullet; 240 settling steps',
              'pose_comparison': comparison,
              'runtime_evidence': {'initialization_code_sha256': sha(__file__),
                                   'world_code_sha256': sha(Path(__file__).parents[1] / 'sim/world.py'),
                                   'pybullet_api_version': world.p.getAPIVersion()},
              'compatible': all(row['compatible'] for row in comparison.values()),
              'measured_frame0': frame,
              'measured_dynamics': dynamics,
              'dynamics_protocol': 'post-settling getDynamicsInfo for every base/child link; local inertial frame',
              'task_success_claim': False}
    (out / 'initialization.json').write_text(json.dumps(result, indent=2))
    return result


def load_bound_supplement(ref, evidence):
    """Validate an isolated measurement against unchanged canonical inputs."""
    ref, evidence = Path(ref).resolve(), Path(evidence).resolve()
    data = json.loads(evidence.read_text())
    if (data.get('schema_version') != 'wo_expert_initialization_check.v1'
            or Path(data.get('canonical_reference', '')).resolve() != ref
            or data.get('compatible') is not True
            or data.get('task_success_claim') is not False):
        raise ValueError('invalid initialization supplement binding')
    manifest = data.get('source_sha256', {})
    required = {str(ref / name) for name in
                ('certificate.json', 'scene_inputs.json', 'trajectory.json')}
    scene = load_scene(json.loads((ref / 'scene_inputs.json').read_text()))
    closure = {}
    for obj in [scene.support, *scene.objects.values()]:
        marker = ref / 'physics_sdf' / f'{obj.object_id}_s{obj.scale:.3f}' / '.path'
        closure[str(marker)] = sha(marker)
        asset_closure(Path(marker.read_text().strip()), closure)
    asset_closure(scene.robot_urdf, closure)
    cert = json.loads((ref / 'certificate.json').read_text())
    required.add(str(Path(cert['task']).resolve()))
    required.update(closure)
    if not required.issubset(manifest) or any(sha(p) != h for p, h in manifest.items()):
        raise ValueError('initialization supplement source changed or incomplete')
    if any(manifest[p] != h for p, h in closure.items()):
        raise ValueError('initialization physics closure mismatch')
    original = json.loads((ref / 'trajectory.json').read_text())['frames'][0]
    frame = data['measured_frame0']
    comparison = pose_compatibility(original['objects'], frame['objects'])
    if not all(row['compatible'] for row in comparison.values()):
        raise ValueError('initialization supplement pose mismatch')
    for oid, row in frame['objects'].items():
        values = row.get('joint_positions')
        if not isinstance(values, dict) or any(
                not isinstance(k, str) or not k or not np.isfinite(float(v))
                for k, v in values.items()):
            raise ValueError('missing or invalid measured passive joints')
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--task', type=Path, required=True)
    args = ap.parse_args()
    result = run(args.ref, args.out, args.task)
    if not result['compatible']:
        raise SystemExit('canonical initialization replay incompatible; evidence not publishable')


if __name__ == '__main__':
    main()
