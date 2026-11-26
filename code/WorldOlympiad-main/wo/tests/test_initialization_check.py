import json
import pytest
from wo.expert.initialization_check import load_scene, pose_compatibility, asset_closure, sha
from wo.agent.stages import _expert_check_command, _expert_check_dir, _check_expert_check


@pytest.mark.parametrize('mass', [.123, 600.0, 0.0])
def test_actual_dynamics_not_authored_or_recomputed(mass):
    from types import SimpleNamespace
    from wo.expert.initialization_check import measured_dynamics
    class Physics:
        def getNumJoints(self, *a, **kw): return 1
        def getBodyInfo(self, *a, **kw): return (b'root', b'body')
        def getJointInfo(self, *a, **kw): return [None] * 12 + [b'child']
        def getDynamicsInfo(self, body, link, **kw):
            return (mass if link == -1 else .02, .8, (.01, .02, .03),
                    (.1, .2, .3), (0, 0, 0, 1), 0, .0005, .002)
    rows = measured_dynamics(SimpleNamespace(p=Physics(), cid=7, bodies={'cube': 2}))['cube']
    assert set(rows) == {'root', 'child'}
    assert rows['root']['mass_kg'] == mass
    assert rows['root']['inertia_diagonal_kg_m2'] == [.01, .02, .03]
    assert rows['root']['inertial_position'] == [.1, .2, .3]
    assert rows['child']['mass_kg'] == .02


def test_measured_dynamics_rejects_nonfinite():
    from types import SimpleNamespace
    from wo.expert.initialization_check import measured_dynamics
    class Physics:
        def getNumJoints(self, *a, **kw): return 0
        def getBodyInfo(self, *a, **kw): return (b'root', b'body')
        def getDynamicsInfo(self, *a, **kw):
            return (float('nan'), .8, (1, 1, 1), (0, 0, 0), (0, 0, 0, 1), 0, 0, 0)
    with pytest.raises(ValueError, match='invalid measured dynamics'):
        measured_dynamics(SimpleNamespace(p=Physics(), cid=7, bodies={'cube': 2}))


def test_scene_preserves_archived_inputs():
    obj = dict(object_id='bottle', role='subject', sdf_path='/old/bottle.sdf',
               position=[1, 2, 3], orientation_xyzw=[0, 0, 0, 1], fixed=False, scale=.288,
               initial_container='basket')
    data = dict(case_id='c', support={**obj, 'object_id': 'table'}, objects={'bottle': obj},
                robot_urdf='/robot.urdf', robot_base_position=[4, 5, 6], robot_base_yaw=1.2,
                source={'generation': '/old'}, articulations={'cap': {'host': 'bottle'}})
    assert load_scene(data).to_dict() == data


def test_pose_replay_rejects_millimeter_displacement_and_accepts_quaternion_sign():
    old = {'b': dict(position=[0, 0, 1], orientation_xyzw=[0, 0, 0, 1])}
    new = {'b': dict(position=[0, 0, 1], orientation_xyzw=[0, 0, 0, -1])}
    assert pose_compatibility(old, new)['b']['compatible']
    new['b']['position'][1] = .0053
    assert not pose_compatibility(old, new)['b']['compatible']
    with pytest.raises(ValueError, match='identity'):
        pose_compatibility(old, {})
    new['b']['position'][0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        pose_compatibility(old, new)


def test_closure_hashes_exact_referenced_assets(tmp_path):
    mesh = tmp_path / 'mesh.obj'
    mesh.write_text('mesh')
    urdf = tmp_path / 'body.urdf'
    urdf.write_text('<robot><mesh filename="mesh.obj"/></robot>')
    manifest = {}
    asset_closure(urdf, manifest)
    assert manifest == {str(urdf): sha(urdf), str(mesh): sha(mesh)}
    urdf.write_text('<robot><mesh filename="missing.obj"/></robot>')
    with pytest.raises(FileNotFoundError):
        asset_closure(urdf, {})


def test_stage_is_isolated_initialization_not_reference_or_task_success(tmp_path, monkeypatch):
    import wo.agent.stages as stages
    monkeypatch.setattr(stages, 'run_dir', lambda run: tmp_path)
    args = dict(run='bench_v1', case='c', canonical_initialization=True, evidence_id='init001')
    command = _expert_check_command(args, 1)
    assert command[2] == 'wo.expert.initialization_check'
    out = _expert_check_dir(args)
    assert out == tmp_path / 'expert_checks/c/init001'
    out.mkdir(parents=True)
    (out / 'initialization.json').write_text(json.dumps({'compatible': True}))
    assert _check_expert_check(args)['task_success_claim'] is False
    with pytest.raises(ValueError):
        _expert_check_dir({**args, 'evidence_id': '../reference'})
    normal = _expert_check_command(dict(run='bench_v1', case='c'), 1)
    assert normal[2] == 'wo.expert.run'


def test_supplement_rejects_forged_binding(tmp_path):
    from wo.expert.initialization_check import load_bound_supplement
    evidence = tmp_path / 'evidence.json'
    evidence.write_text(json.dumps(dict(schema_version='wo_expert_initialization_check.v1',
        canonical_reference=str(tmp_path / 'other'), compatible=True, task_success_claim=False)))
    with pytest.raises(ValueError, match='binding'):
        load_bound_supplement(tmp_path / 'ref', evidence)


def test_supplement_requires_physics_closure(tmp_path, monkeypatch):
    import wo.expert.initialization_check as check
    from types import SimpleNamespace
    ref = tmp_path / 'ref'
    ref.mkdir()
    task = tmp_path / 'task.json'
    task.write_text('{}')
    (ref / 'certificate.json').write_text(json.dumps({'task': str(task)}))
    (ref / 'scene_inputs.json').write_text('{}')
    obj = dict(position=[0, 0, 1], orientation_xyzw=[0, 0, 0, 1])
    (ref / 'trajectory.json').write_text(json.dumps({'frames': [{'objects': {'b': obj}}]}))
    robot = tmp_path / 'robot.urdf'
    robot.write_text('<robot/>')
    marker = ref / 'physics_sdf/b_s1.000/.path'
    marker.parent.mkdir(parents=True)
    marker.write_text(str(robot))
    monkeypatch.setattr(check, 'load_scene', lambda data: SimpleNamespace(
        support=SimpleNamespace(object_id='b', scale=1), objects={}, robot_urdf=robot))
    paths = [ref / n for n in ('certificate.json', 'scene_inputs.json', 'trajectory.json')]
    paths += [task, marker, robot]
    data = dict(schema_version='wo_expert_initialization_check.v1', canonical_reference=str(ref),
                compatible=True, task_success_claim=False,
                source_sha256={str(p): sha(p) for p in paths},
                measured_frame0={'objects': {'b': {**obj, 'joint_positions': {'cap': .001}}}})
    evidence = tmp_path / 'initialization.json'
    evidence.write_text(json.dumps(data))
    assert check.load_bound_supplement(ref, evidence)['compatible']
    data['measured_frame0']['objects']['b']['position'] = [.005, 0, 1]
    evidence.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='pose mismatch'):
        check.load_bound_supplement(ref, evidence)
    task.write_text('changed')
    with pytest.raises(ValueError, match='source changed'):
        check.load_bound_supplement(ref, evidence)
