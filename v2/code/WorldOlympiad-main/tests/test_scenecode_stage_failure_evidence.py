"""Run the staged pipeline's real control flow with injected stage outcomes."""
import ast
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def pipeline_types():
    path = Path(os.environ.get('SCENECODE_TEST_PIPELINE', '/tmp/SceneCode-main-worldolympiad/scenecode/code_object/agent_framework/pipeline.py'))
    if not path.is_file():
        pytest.skip('requires staged SceneCode pipeline')
    tree = ast.parse(path.read_text())
    result = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PipelineResult')
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Pipeline')
    owner.body = [n for n in owner.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'run']
    namespace = dict(globals(), ObjectPlan=object)
    exec(compile(ast.Module(body=[result, owner], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['Pipeline']


def make_pipeline(cls, failed_stage):
    p = cls()
    p.config = {'pipeline': {'enable_planner_check': failed_stage == 'plan_check'}}
    p.logger = logging.getLogger('failure-evidence-test')
    p.blender_client = object()
    p.llm_client = None
    p._usage_tracker = None
    plan = SimpleNamespace(name='part', parts=[object()], to_dict=lambda: {'name': 'part'})
    p._validate_generation_input = lambda **_: 'text'
    p._check_existing_plan = lambda *_: None if failed_stage == 'planning' else plan
    p._check_existing_construction = lambda *_: None
    p._connect_blender = AsyncMock(return_value=True)
    p._clear_blender_scene = AsyncMock()
    p._disconnect_blender = AsyncMock()
    p._disconnect_llm_clients = AsyncMock()
    p.planner = SimpleNamespace(run=AsyncMock(side_effect=ValueError('invalid authored plan')))
    p.planner_checker = SimpleNamespace(check_and_fix=AsyncMock(side_effect=ValueError('measured part extent mismatch')))
    p.part_constructor = SimpleNamespace(run=AsyncMock(return_value={'part_results': [], 'construct_success': True}))
    p._export_part_mesh_assets = AsyncMock(return_value='/verified/parts')
    p._render_full_object_and_save = AsyncMock(return_value=None)
    p._export_full_object_glb = AsyncMock(return_value='/verified/part.glb')
    if failed_stage == 'export_parts':
        p._export_part_mesh_assets.side_effect = RuntimeError("MCP request failed: No module named 'autoresearch'")
    elif failed_stage == 'construction':
        p.part_constructor.run.side_effect = RuntimeError('PART_BOUNDS_CONTRACT: wrong center')
    elif failed_stage == 'construction_partial':
        p.part_constructor.run.return_value = {'part_results': [], 'construct_success': False, 'failed_parts': ['handle']}
    elif failed_stage == 'export_glb':
        p._export_full_object_glb.return_value = None
    return p


@pytest.mark.parametrize('stage,detail', [
    ('export_parts', "No module named 'autoresearch'"),
    ('planning', 'invalid authored plan'),
    ('plan_check', 'measured part extent mismatch'),
    ('construction', 'PART_BOUNDS_CONTRACT'),
    ('construction_partial', 'handle'),
    ('export_glb', 'No valid GLB artifact'),
])
def test_failure_reason_reaches_saved_result(pipeline_types, tmp_path, stage, detail):
    p = make_pipeline(pipeline_types, stage)
    result = asyncio.run(p.run(text_input='an asset', output_dir=str(tmp_path)))
    saved = json.loads((tmp_path / 'pipeline_result.json').read_text())
    assert not result.success
    assert detail in result.error
    assert saved == result.to_dict()
    key = 'construction' if stage == 'construction_partial' else stage
    assert detail in saved['stage_errors'][key]
    p._disconnect_blender.assert_awaited_once()


def test_success_does_not_inherit_failure(pipeline_types, tmp_path):
    p = make_pipeline(pipeline_types, None)
    result = asyncio.run(p.run(text_input='an asset', output_dir=str(tmp_path)))
    assert result.success and result.error is None and result.stage_errors == {}


@pytest.mark.parametrize('passes', [True, False])
def test_complete_source_cache_is_reexecuted_before_export(pipeline_types, tmp_path, passes):
    p = make_pipeline(pipeline_types, None if passes else 'construction')
    p._check_existing_construction = lambda *_: {'construct_success': True, 'part_results': ['part'], 'skipped': True}
    result = asyncio.run(p.run(text_input='an asset', output_dir=str(tmp_path)))
    p.part_constructor.run.assert_awaited_once()
    assert result.success is passes
    if not passes:
        assert 'PART_BOUNDS_CONTRACT' in result.error
        # One failed part used to refuse the whole assembly, discarding every
        # part that had been built.  The assembled parts are exported and the
        # gap is recorded instead; downstream admission judges the asset.
        p._export_part_mesh_assets.assert_awaited()
