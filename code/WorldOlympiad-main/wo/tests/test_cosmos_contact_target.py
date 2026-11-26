import ast
from pathlib import Path
import pytest
from autoresearch.cosmos_runtime_bodies import contact_target_body


def body(name, identity=None, static=True):
    return {'name': name, 'static': static,
            'metadata': {'task_object_id': identity or name}}


def graph(target='table', predicate='on_region'):
    return {'success_program': [{'object': target, 'predicate': predicate}]}


def test_declared_support_is_actual_contact_target_without_duplicate_body():
    table = body('wooden_support', 'table')
    plan = {'bodies': [body('cup', 'book', False), table]}
    assert contact_target_body(plan, graph()) is table
    assert len(plan['bodies']) == 2


def test_existing_destination_binding_is_unchanged():
    tray = body('tray', 'bin')
    assert contact_target_body({'bodies': [body('table'), tray]}, graph('bin', 'inside')) is tray


@pytest.mark.parametrize('bodies,g', [
    ([body('other')], graph()),
    ([body('table', static=False)], graph()),
    ([body('table')], graph(predicate='inside')),
    ([body('table')], {'success_program': []}),
    ([body('a', 'table'), body('b', 'table')], graph()),
])
def test_missing_ambiguous_dynamic_or_non_support_target_is_not_fabricated(bodies, g):
    with pytest.raises(ValueError):
        contact_target_body({'bodies': bodies}, g)


def test_runner_binds_actual_target_prim_and_initializes_legacy_slot():
    source = Path('scripts/run_cosmos3_worldir_isaac_closed_loop.py').read_text()
    ast.parse(source)
    assert 'stage, target_prim_path' in source
    assert 'RigidObjectCfg(prim_path=target_prim_path, spawn=None)' in source
    assert 'result.setdefault("tray", {})' in source
