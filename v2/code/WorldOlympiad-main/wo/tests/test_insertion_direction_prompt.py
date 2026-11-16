import json

from wo.spec import Predicate, Step, TaskObject, TaskSpec
from wo.tasks.scene import legacy_task
from autoresearch.scenecode_object_requirements import object_requirements


def task(kind='inserted', axis=(1, 0, 0)):
    return TaskSpec(case_id='direction_test', family='insertion', instruction='Insert the key.',
        initial_state='The separate key lies beside an empty lock keyway.',
        objects=(TaskObject('key', 'key', 'manipuland', 'a flat metal key',
                            geometry_constraints={'overall_dimensions_m': [.06, .02, .003]}),
                 TaskObject('lock', 'lock', 'destination', 'a lock with an open keyway',
                            geometry_constraints={'clear_inner_opening_m': [.022, .005],
                                                  'cavity': {'depth_m': .04, 'axis': list(axis)}}),
                 TaskObject('table', 'table', 'support', 'a wood table')),
        steps=(Step('transport', 'key', Predicate(kind, 'key', 'lock'), target='lock',
                    constraints={'release': 'stable'}),))


def test_horizontal_insertion_keeps_typed_geometry_and_release_without_prose_recipe(monkeypatch):
    spec = task();before = spec.to_dict();compiled = legacy_task(spec)
    assert compiled['scene_brief'] == spec.initial_state
    assert compiled['objects'][1]['geometry_constraints'] == spec.objects[1].geometry_constraints
    assert compiled['action_program'][-1]['constraints'] == {'release': 'stable'}
    assert compiled['success_predicates'] == [{'kind': 'inserted', 'subject': 'key', 'object': 'lock'}]
    monkeypatch.setenv('SCENECODE_AUTHORED_TASK_REQUIREMENTS', json.dumps(compiled))
    monkeypatch.delenv('SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST', raising=False)
    text = object_requirements('', [.06, .02, .003], asset_name='key')
    compatibility = json.loads(next(line.split('=', 1)[1] for line in text.splitlines() if line.startswith('compatible_geometry=')))
    assert compatibility[0]['cavity'] == {'depth_m': .04, 'axis': [1, 0, 0]}
    assert compatibility[0]['clear_inner_opening_m'] == [.022, .005]
    assert 'success_predicates' not in text and spec.to_dict() == before


def test_vertical_geometry_retains_exact_authored_axis(monkeypatch):
    spec = task(axis=(0, 0, -1));compiled = legacy_task(spec)
    monkeypatch.setenv('SCENECODE_AUTHORED_TASK_REQUIREMENTS', json.dumps(compiled))
    monkeypatch.delenv('SCENECODE_CODE_OBJECT_REFERENCE_IMAGE_MANIFEST', raising=False)
    text = object_requirements('', None, asset_name='lock')
    geometry = json.loads(next(line.split('=', 1)[1] for line in text.splitlines() if line.startswith('geometry_constraints=')))
    assert geometry['cavity'] == {'depth_m': .04, 'axis': [0, 0, -1]}
    assert compiled['scene_brief'] == spec.initial_state


def test_other_families_do_not_receive_insertion_guidance():
    assert legacy_task(task(kind='on'))['scene_brief'] == task(kind='on').initial_state


def test_compacted_initial_state_does_not_repeat_the_original_brief():
    from dataclasses import replace
    from autoresearch.robotics_contracts import TaskInstructionSpec
    from autoresearch.robotics_scene_prompt_compiler import compile_task_prompt
    from autoresearch.scene_prompting import compile_scene_prompt, assess_scene_prompt
    spec = replace(task(), initial_state='The separate key lies beside an empty keyway. The front approach remains clear.')
    scene = compile_task_prompt(TaskInstructionSpec.model_validate(legacy_task(spec)))
    prose = compile_scene_prompt(scene)
    assert prose.count('separate key lies beside') == 1
    assert prose.count('front approach remains clear') == 1
    assert assess_scene_prompt(scene, prose).checks['scene_brief_preserved']
