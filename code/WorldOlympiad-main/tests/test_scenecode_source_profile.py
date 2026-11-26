import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest
import yaml

from autoresearch.robotics_scenecode_case_generator import _code_object_profile_overrides
from autoresearch.scenecode_openai_stream import collect_responses_completion
from wo.tasks.scene import source_config_env


@pytest.fixture
def profile(tmp_path):
    task = tmp_path / 'tasks/case.json'
    task.parent.mkdir();task.write_text('{}')
    root = tmp_path / 'source_profiles/sol6'
    root.mkdir(parents=True)
    value = {'model': {role: {'provider': 'openai', 'model_name': 'gpt-6.1-sol'}
                       for role in ('llm', 'planner_llm', 'constructor_llm', 'vision')}}
    for filename in ('config.yaml', 'config_WallArt.yaml', 'config_simplemanip.yaml', 'config_structmanip.yaml'):
        (root / filename).write_text(yaml.safe_dump(value))
    return task, root


def test_default_source_configuration_is_unchanged(tmp_path):
    assert source_config_env(tmp_path/'tasks/case.json', None) == {}
    assert _code_object_profile_overrides(None) == []


def test_explicit_source_profile_selects_all_native_paths_and_author_roles(profile):
    task, root = profile
    selected = source_config_env(task, root)
    assert selected['SCENECODE_OPENAI_MODEL'] == 'gpt-6.1-sol'
    assert selected['SCENECODE_CODE_OBJECT_API'] == 'responses'
    assert selected['SCENECODE_CODE_OBJECT_STREAM'] == '1'
    assert selected['SCENECODE_CODE_OBJECT_READ_TIMEOUT'] == 'none'
    assert selected['SCENECODE_CODE_OBJECT_RESPONSES_STREAM'] == '0'
    overrides = _code_object_profile_overrides(str(root))
    assert len(overrides) == 16
    assert any('furniture_agent.asset_manager.code_object.config_path=' in line for line in overrides)
    assert any('manipuland_agent.asset_manager.code_object.manipuland_config_path=' in line for line in overrides)
    for filename in ('config.yaml', 'config_structmanip.yaml'):
        configuration = yaml.safe_load((root/filename).read_text())
        assert configuration['model']['planner_llm']['model_name'] == 'gpt-6.1-sol'
        assert configuration['model']['constructor_llm']['model_name'] == 'gpt-6.1-sol'


def test_profile_mismatch_or_outside_run_rejects_locally(profile, tmp_path):
    task, root = profile
    value = yaml.safe_load((root/'config_structmanip.yaml').read_text())
    value['model']['constructor_llm']['model_name'] = 'gpt-5.6-sol'
    (root/'config_structmanip.yaml').write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match='agree'):
        source_config_env(task, root)
    with pytest.raises(ValueError):
        source_config_env(task, tmp_path.parent)


def test_responses_read_deadline_override_reaches_actual_httpx_request(monkeypatch):
    from openai import AsyncOpenAI
    monkeypatch.setenv('SCENECODE_CODE_OBJECT_RESPONSES_STREAM', '0')
    monkeypatch.setenv('SCENECODE_CODE_OBJECT_READ_TIMEOUT', 'none')
    observed = []
    async def handler(request):
        observed.append(request.extensions['timeout'])
        body = json.loads(request.content)
        assert body['model'] == 'gpt-6.1-sol' and 'timeout' not in body
        return httpx.Response(200, json={'id':'resp_unit','object':'response','created_at':0,
            'model':'gpt-6.1-sol','status':'completed','output':[{'id':'msg_unit','type':'message',
            'role':'assistant','status':'completed','content':[{'type':'output_text','text':'ready','annotations':[]}]}],
            'usage':None})
    async def run():
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = AsyncOpenAI(api_key='unit-test', base_url='https://local-test.invalid/v1', http_client=http_client)
        try:
            result = await collect_responses_completion(client, {'model':'gpt-6.1-sol',
                'messages':[{'role':'user','content':'ready'}],'max_tokens':32})
            assert result.choices[0].message.content == 'ready'
        finally:
            await client.close()
    asyncio.run(run())
    assert observed == [{'connect':30,'read':None,'write':30,'pool':30}]


def test_registered_scene_carries_only_explicit_profile_argument(profile):
    from wo.agent.stages import STAGES
    _, root = profile
    base = STAGES['scene'].command({'run':'dev','case':'case'}, 1)
    assert '--source-config-root' not in base
    explicit = STAGES['scene'].command({'run':'dev','case':'case','source_config_root':str(root)}, 1)
    assert explicit == [*base, '--source-config-root', str(root)]
