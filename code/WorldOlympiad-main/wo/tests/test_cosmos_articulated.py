import json
import math
from pathlib import Path

import numpy as np
import pytest

from wo.models import cosmos
from wo.models import cosmos_articulated as art
from wo.spec import Articulation, Predicate, Step, TaskObject, TaskSpec
from wo.eval import cae_episodes

@pytest.mark.parametrize('residual', [-4.282647795972911e-06, 4e-6, 0.0])
def test_native_initial_position_uses_spawn_not_reference_solver_residual(residual):
    from autoresearch.articulated_native_task import JointTask
    q = art.native_initial_position(residual)
    assert q == 0.0
    task = JointTask('revolute', (0, 0, 1), (0, 0, 0), (0, 0, 0, 1),
                     (0, 0, 0), q, 1.2, 0, 1.5707963, .1, .02, 5, 900)
    assert task.closed_position == 0.0
    assert task.lower == 0 and task.upper == 1.5707963


@pytest.mark.parametrize('q', [.002, -.002, float('nan'), float('inf')])
def test_native_initial_position_rejects_non_equivalent_reference(q):
    with pytest.raises(art.ArticulationUnsupported):
        art.native_initial_position(q)


CERT_ART = {"cabinet_door": {"host": "cabinet_host", "joint_name": "hinge", "child_link": "door",
                             "lower": -1.5707963, "upper": 0.0, "q_closed": 0.0,
                             "q_open": -1.5707963, "joint_type": "revolute"}}


def _spec(*steps):
    return TaskSpec('cab', 'articulation', 'Pull the cabinet door handle until the door is wide open.',
                    (TaskObject('cabinet_host', 'cabinet', 'support'),), steps,
                    articulations=(Articulation('cabinet_door', 'cabinet_host', 'revolute'),))


def _open_step(**params):
    return Step('articulate', 'cabinet_door', Predicate('joint_open', 'cabinet_door', None, params))


def _ref(tmp_path, spec):
    task = tmp_path / 'task.json'
    task.write_text(json.dumps(spec.to_dict()))
    ref = tmp_path / 'ref'
    ref.mkdir()
    (ref / 'certificate.json').write_text(json.dumps(
        {'task': str(task), 'family': spec.family, 'articulations': CERT_ART}))
    return ref


def test_binding_and_goal_band_match_wo_predicate():
    b = art.binding(_spec(_open_step()), {"articulations": CERT_ART})
    assert b["component"] == "cabinet_door" and b["host"] == "cabinet_host"
    assert b["threshold"] == pytest.approx(0.60)
    target, tol = art.goal_band(b)
    # the runner accepts |q-target|<=tol: exactly fractions [0.6, 1.0] (+1e-4 rad)
    for frac, ok in ((0.0, False), (0.55, False), (0.61, True), (0.85, True), (1.0, True)):
        q = b["q_closed"] + frac * (b["q_open"] - b["q_closed"])
        assert (abs(q - target) <= tol) == ok == art.satisfied(art.fraction(q, b), b)
    graph = art.task_graph(_spec(_open_step()), b)
    assert graph["success_program"][0]["subject"] == "cabinet_door"
    assert graph["articulation_components"][0]["child_link"] == "door"
    assert art.binding(_spec(_open_step(threshold=0.8)), {"articulations": CERT_ART})["threshold"] == 0.8


def test_unrepresentable_articulation_is_unsupported():
    mixed = _spec(_open_step(), Step('transport', 'cabinet_host',
                                     Predicate('on', 'cabinet_host', 'cabinet_host')))
    with pytest.raises(art.ArticulationUnsupported):
        art.binding(mixed, {"articulations": CERT_ART})
    with pytest.raises(art.ArticulationUnsupported, match='lacks articulation'):
        art.binding(_spec(_open_step()), {"articulations": {}})


def test_run_episode_dispatches_articulate_to_articulated_runner(tmp_path, monkeypatch):
    ref = _ref(tmp_path, _spec(_open_step()))
    seen = {}
    monkeypatch.setattr(cosmos, 'run_articulated_episode',
                        lambda r, o, **kw: seen.update(ref=r, out=o, **kw) or {'runner': 'articulated'})
    monkeypatch.setattr(cosmos, 'build_bundle', lambda *a, **k: pytest.fail('rigid path used'))
    assert cosmos.run_episode(ref, tmp_path / 'out', window_s=60, seed=0)['runner'] == 'articulated'
    assert seen['window_s'] == 60 and seen['ref'] == ref


def test_gr00t_routes_joint_task_without_rigid_bundle(tmp_path, monkeypatch):
    from wo.models import gr00t
    ref = _ref(tmp_path, _spec(_open_step()))
    seen = {}
    monkeypatch.setattr(cosmos, 'run_articulated_episode',
                        lambda r, o, **kw: seen.update(ref=r, out=o, **kw) or {'runner': 'articulated'})
    monkeypatch.setattr(cosmos, 'build_bundle', lambda *a, **k: pytest.fail('rigid path used'))
    result = gr00t.run_episode(ref, tmp_path / 'out', window_s=60, seed=4, port=29999)
    assert result['runner'] == 'articulated'
    assert seen['policy_model'] == 'gr00t'
    assert seen['seed'] == 4 and seen['window_s'] == 60 and seen['port'] == 29999


def test_mixed_articulate_task_becomes_not_applicable(tmp_path, monkeypatch):
    import sys
    mixed = _spec(_open_step(), Step('transport', 'cabinet_host',
                                     Predicate('on', 'cabinet_host', 'cabinet_host')))
    ref = _ref(tmp_path, mixed)
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['cosmos', '--ref', str(ref), '--out', str(out)])
    cosmos.main()
    ep = json.loads((out / 'episode.json').read_text())
    assert ep['status'] == 'not_applicable' and 'exactly one articulate step' in ep['reason']


def _fake_runner_output(out: Path, qs, *, success=False, completion=None, released=True):
    cap = out / 'native_run/capture'
    cap.mkdir(parents=True)
    frames = []
    for i, q in enumerate(qs):
        completed = completion is not None and i >= completion
        frame_success = completed and (success or i < len(qs) - 1)
        terminal = {"task_success": frame_success,
                    "released": released, "stable_steps": 5 if frame_success else 0,
                    "completion_frame": completion if completed else None,
                    "done": i == len(qs) - 1,
                    "reason": "evaluation_window_complete" if i == len(qs) - 1 else None}
        frames.append({"frame": i, "articulated": {"position": q}, "terminal": terminal,
                       "child_link_position": [0., 0., .4], "child_link_xyzw": [0, 0, 0, 1],
                       "finger_contacts": [not released, not released],
                       "robot": {"joint_names": ["panda_joint1"], "joint_positions": [0.1],
                                 "base_position": [0, 0, 0], "base_xyzw": [0, 0, 0, 1],
                                 "ee_position": [0, 0, 1], "gripper": 0.0 if released else 0.8}})
    (cap / 'trajectory.json').write_text(json.dumps({"frames": frames}))
    (cap / 'result.json').write_text(json.dumps({
        "schema_version": "cosmos_articulated_policy_trial.v1", "status": "complete",
        "task_success": success, "completion_frame": completion, "terminal": frames[-1],
        "termination_reason": frames[-1]["terminal"]["reason"], "maximum_progress": 1.0,
        "frame_count": len(frames), "action_count": len(frames) - 1, "window_count": 1}))
    (cap / 'progress.json').write_text(json.dumps({"step": len(frames) - 1,
        "terminal": frames[-1]["terminal"], "windows": [
        {"index": 0, "conditioning_frame": 0, "executed_count": len(qs) - 1}]}))
    return cap


def _conversion_reference(tmp_path):
    ref = tmp_path / 'ref'
    ref.mkdir()
    host = {"link_position": [0, 0, 0], "link_orientation_xyzw": [0, 0, 0, 1]}
    (ref / 'trajectory.json').write_text(json.dumps({"frames": [{"objects": {"cabinet_host": host}}]}))
    b = art.binding(_spec(_open_step()), {"articulations": CERT_ART})
    return ref, b, host


def test_convert_uses_native_release_and_stability_completion(tmp_path):
    ref, b, host = _conversion_reference(tmp_path)
    qo = -1.5707963
    out = tmp_path / 'ep'
    _fake_runner_output(out, [0.0, 0.7 * qo, 0.5 * qo, 0.65 * qo, 0.9 * qo, 0.95 * qo],
                        success=True, completion=5)
    meta = art.convert(ref, out, b, policy_model='gr00t')
    assert meta['model'] == 'gr00t_n1.6_droid'
    assert meta['action_source'] == 'gr00t_n1.6_droid'
    assert meta["task_success"] is True and meta["completion_frame"] == 5
    assert meta["joint_band_diagnostic"]["continuous_to_end_from_frame"] == 3
    evidence = meta["completion_evidence"]["native_predicate_evidence"]
    assert evidence["measured_at_s"] == pytest.approx(5 / 15)
    assert evidence["native_monitor"]["released"] is True
    assert evidence["native_monitor"]["stable_steps"] == 5
    assert meta["max_joint_fraction"] == pytest.approx(0.95)
    traj = json.loads((out / 'native_run/simulation/trajectory.json').read_text())
    f = traj["frames"][-1]
    assert f["articulation_positions"]["cabinet_door"] == pytest.approx(0.95 * qo)
    assert f["objects"]["cabinet_host"] == host and f["robot"]["joint_names"] == ["panda_joint1"]


@pytest.mark.parametrize('released', [False, True], ids=['held', 'released_but_unstable'])
def test_convert_does_not_promote_joint_target_without_native_success(tmp_path, released):
    ref, b, _ = _conversion_reference(tmp_path)
    out = tmp_path / 'ep'
    _fake_runner_output(out, [0.0, -1.4, -1.5], released=released)
    meta = art.convert(ref, out, b)
    assert meta["task_success"] is False and meta["completion_frame"] is None
    assert meta["completion_evidence"] is None
    assert meta["joint_band_diagnostic"]["final_satisfied"] is True
    assert meta["joint_band_diagnostic"]["includes_release_and_stability"] is False
    assert meta["terminal_native_monitor"]["released"] is released
    assert meta["terminal_native_monitor"]["stable_steps"] == 0


def test_convert_preserves_native_first_completion_when_terminal_success_is_lost(tmp_path):
    ref, b, _ = _conversion_reference(tmp_path)
    out = tmp_path / 'ep'
    _fake_runner_output(out, [0.0, -1.4, -1.5, -0.4], success=False, completion=2)
    meta = art.convert(ref, out, b)
    assert meta["task_success"] is False
    assert meta["completion_frame"] == 2
    assert meta["completion_evidence"]["native_predicate_evidence"]["native_monitor"]["task_success"] is True
    assert meta["terminal_native_monitor"]["task_success"] is False


@pytest.mark.parametrize('mutation', ['missing_file', 'legacy_subset', 'wrong_count', 'monitor_disagrees'])
def test_convert_rejects_unbound_native_measurement_without_writing(tmp_path, mutation):
    ref, b, _ = _conversion_reference(tmp_path)
    out = tmp_path / 'ep'
    cap = _fake_runner_output(out, [0.0, -1.4, -1.5])
    result = cap / 'result.json'
    if mutation == 'missing_file':
        result.unlink()
    elif mutation == 'legacy_subset':
        result.write_text(json.dumps({"task_success": False, "maximum_progress": 1.0}))
    else:
        value = json.loads(result.read_text())
        value['frame_count' if mutation == 'wrong_count' else 'task_success'] = 999 if mutation == 'wrong_count' else True
        result.write_text(json.dumps(value))
    with pytest.raises(art.ArticulationMeasurementError):
        art.convert(ref, out, b)
    assert not (cap / 'capture_metadata.json').exists()
    assert not (out / 'native_run/simulation/trajectory.json').exists()


URDF = """<robot name="cab">
 <link name="body"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
 <link name="door"><inertial><origin xyz="0.5 0 0"/><mass value="1"/>
  <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
 <joint name="hinge" type="revolute"><parent link="body"/><child link="door"/>
  <origin xyz="0 0 0.4"/><axis xyz="0 0 1"/><limit lower="-1.6" upper="0" effort="1" velocity="1"/></joint>
</robot>"""


def test_moving_part_track_follows_joint(tmp_path):
    ref = tmp_path / 'ref'
    (ref / 'physics_sdf/cabinet_host_s1.000').mkdir(parents=True)
    (ref / 'physics_sdf/cabinet_host_s1.000/cab.urdf').write_text(URDF)
    host = {"link_position": [1, 0, 0], "link_orientation_xyzw": [0, 0, 0, 1]}
    qs = np.linspace(0, -math.pi / 2, 5)
    frames = [{"robot": {"base_position": [0, 0, 0], "base_orientation_xyzw": [0, 0, 0, 1],
                         "joint_names": ["panda_joint1"], "joint_positions": [0.0]},
               "objects": {"cabinet_host": host}, "articulation_positions": {"cabinet_door": float(q)}}
              for q in qs]
    traj = tmp_path / 'traj.json'
    traj.write_text(json.dumps({"frames": frames}))
    b = art.binding(_spec(_open_step()), {"articulations": CERT_ART})
    tracks = cae_episodes.entity_tracks(traj, ["cabinet_door"], [b], ref)
    pos = tracks["cabinet_door"].positions
    assert np.allclose(pos[0], [1.5, 0, 0.4], atol=1e-4)
    assert np.allclose(pos[-1], [1.0, -0.5, 0.4], atol=1e-4)
