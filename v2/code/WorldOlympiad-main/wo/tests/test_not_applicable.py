import json
import sys
from pathlib import Path

from wo.models import cosmos
from wo.eval import case, report


def test_unsupported_is_explicit_and_scores_without_video(tmp_path, monkeypatch):
    task = tmp_path / 'task.json'
    from wo.spec import TaskSpec, TaskObject, Step, Predicate
    spec = TaskSpec('x', 'pick_place', 'place x in bowl',
                    (TaskObject('x', 'red block', 'manipuland'),
                     TaskObject('bowl', 'blue bowl', 'destination')),
                    (Step('transport', 'x', Predicate('inside', 'x', 'bowl')),))
    task.write_text(json.dumps(spec.to_dict()))
    ref = tmp_path / 'ref'
    ref.mkdir()
    (ref / 'certificate.json').write_text(json.dumps({'task': str(task), 'family': spec.family}))
    out = tmp_path / 'out'
    def unsupported(*args, **kwargs):
        raise cosmos.Unsupported('native simulator cannot represent deformable bodies')
    monkeypatch.setattr(cosmos, 'run_episode', unsupported)
    monkeypatch.setattr(sys, 'argv', ['cosmos', '--ref', str(ref), '--out', str(out), '--seed', '1'])
    cosmos.main()
    ep = json.loads((out / 'episode.json').read_text())
    assert ep['status'] == 'not_applicable' and ep['task_success'] is None
    # An untyped adapter exception stays pending until its applicability is reviewed.
    pending = case.build_case(ref, out, out / 'case')
    assert case.score(pending)['status'] == 'pending'
    ep['applicability'] = {'kind': 'simulation_representation'}
    (out / 'episode.json').write_text(json.dumps(ep))
    scored = case.build_case(ref, out, out / 'case')
    assert case.score(scored)['status'] == 'not_applicable'
    row = report.episode_result(out)
    assert row['status'] == 'not_applicable' and row['reason'] == ep['reason']
    assert row['worldolympiad2_score'] is None
    # Archived action-only adapters used termination_reason rather than status.
    # Read these without rewriting reference-bound native evidence or requiring videos.
    ep.pop('status')
    ep['not_applicable_reason'] = ep.pop('reason')
    (out / 'episode.json').write_text(json.dumps(ep))
    legacy_case = case.build_case(ref, out, out / 'legacy_case')
    assert case.score(legacy_case)['reason'] == ep['not_applicable_reason']
    assert report.applicability_result(ep)['status'] == 'not_applicable'


def test_missing_measurement_is_pending_not_na():
    row = report.applicability_result({'termination_reason': 'not_applicable',
                                     'not_applicable_reason': 'native no measurement for covering'})
    assert row['status'] == 'pending'
    assert row['applicability_kind'] == 'implementation_gap'
    assert report.applicability_result({'termination_reason': 'not_applicable'})['status'] == 'pending'
    assert report.applicability_result({'termination_reason': 'timeout', 'videos': {}}) == {}
    assert report.applicability_result({'termination_reason': 'not_applicable',
        'not_applicable_reason': "native runner cannot represent primitive 'articulate'"})['status'] == 'pending'
