import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('explicit_profile', [False, True])
def test_original_launcher_preserves_only_explicit_public_profile(tmp_path, explicit_profile):
    workspace = tmp_path/'workspace'
    repo = workspace/'code/WorldOlympiad-main'
    stage = repo/'deployment/ziplab5090/stage_scenecode_runtime.sh'
    stage.parent.mkdir(parents=True)
    # Helpers are checked out as 0644 on the shared volume; the launcher must
    # invoke them through bash rather than relying on executable mode bits.
    stage.write_text('#!/bin/sh\nexit 0\n')
    python = workspace/'envs/scenecode/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('''#!/usr/bin/env python3
import json,os,sys
if sys.argv[1:3] == ['-m','autoresearch.robotics_scenecode_case_generator']:
    keys=('SCENECODE_OPENAI_MODEL','SCENECODE_CODE_OBJECT_API','SCENECODE_CODE_OBJECT_STREAM','SCENECODE_CODE_OBJECT_RESPONSES_STREAM','SCENECODE_CODE_OBJECT_READ_TIMEOUT','SCENECODE_CODE_OBJECT_REASONING_EFFORT','SCENECODE_BWRAP_WRITABLE_PATHS','SCENECODE_INLINE_SINGLE_WORKER','CUDA_VISIBLE_DEVICES')
    from pathlib import Path
    Path(os.environ['PROBE_OUTPUT']).write_text(json.dumps({k:os.environ.get(k) for k in keys}))
''');python.chmod(0o755)
    autoresearch=workspace/'envs/autoresearch/bin/python';autoresearch.parent.mkdir(parents=True)
    autoresearch.write_text('#!/bin/sh\nexit 0\n');autoresearch.chmod(0o755)
    api_env=tmp_path/'api.env'
    api_env.write_text('OPENAI_API_KEY=unit_test_credential\nSCENECODE_OPENAI_MODEL=gpt-5.6-sol\nSCENECODE_CODE_OBJECT_READ_TIMEOUT=300\nSCENECODE_CODE_OBJECT_REASONING_EFFORT=high\n')
    api_env.chmod(0o600)
    result=tmp_path/'observed.json'
    env={k:v for k,v in os.environ.items() if not k.startswith(('SCENECODE_','WORLDOLYMPIAD_'))}
    env.update(WORLDOLYMPIAD_WORKSPACE=str(workspace),WORLD_REPO=str(repo),SCENECODE_ENV_FILE=str(api_env),PROBE_OUTPUT=str(result),WO_TESTED_MODEL_GPU='1')
    stat_stub=workspace/'bin/stat';stat_stub.parent.mkdir(parents=True)
    stat_stub.write_text('#!/bin/sh\n[ "$1" = "-c" ] && [ "$2" = "%a" ] || exit 2\nprintf "600\\n"\n')
    stat_stub.chmod(0o755)
    env['PATH']=str(stat_stub.parent)+os.pathsep+env.get('PATH','')
    if explicit_profile:
        env.update(SCENECODE_CODE_OBJECT_CONFIG_ROOT=str(tmp_path/'explicit'),SCENECODE_OPENAI_MODEL='gpt-6.1-sol',SCENECODE_CODE_OBJECT_API='responses',SCENECODE_CODE_OBJECT_STREAM='1',SCENECODE_CODE_OBJECT_RESPONSES_STREAM='0',SCENECODE_CODE_OBJECT_READ_TIMEOUT='none')
        env['SCENECODE_RENDER_GPU']='6'
    launcher=Path(__file__).resolve().parents[2]/'deployment/ziplab5090/run_scenecode_case.sh'
    subprocess.run(['bash',str(launcher),'batch','case'],env=env,check=True,capture_output=True,text=True)
    observed=json.loads(result.read_text())
    assert observed['SCENECODE_OPENAI_MODEL']==('gpt-6.1-sol' if explicit_profile else 'gpt-5.6-sol')
    assert observed['SCENECODE_CODE_OBJECT_READ_TIMEOUT']==('none' if explicit_profile else '300')
    assert observed['SCENECODE_CODE_OBJECT_API']=='responses'
    assert observed['SCENECODE_CODE_OBJECT_REASONING_EFFORT']==(None if explicit_profile else 'high')
    assert observed['SCENECODE_BWRAP_WRITABLE_PATHS']==os.pathsep.join((
        str(workspace/'tmp'),
        str(workspace/'outputs/robotics_scenecode_generations'),
    ))
    assert observed['CUDA_VISIBLE_DEVICES']==('6' if explicit_profile else '1')
    assert observed['SCENECODE_INLINE_SINGLE_WORKER']=='1'
    if explicit_profile:
        assert observed['SCENECODE_CODE_OBJECT_STREAM']=='1'
        assert observed['SCENECODE_CODE_OBJECT_RESPONSES_STREAM']=='0'


def test_launcher_invokes_helpers_with_bash():
    launcher=Path(__file__).resolve().parents[2]/'deployment/ziplab5090/run_scenecode_case.sh'
    text=launcher.read_text()
    assert 'bash "$world_repo/deployment/ziplab5090/bootstrap_scenecode.sh"' in text
    assert 'bash "$world_repo/deployment/ziplab5090/stage_scenecode_runtime.sh"' in text
    assert 'SCENECODE_BWRAP_WRITABLE_PATHS=' in text
    assert 'export CUDA_VISIBLE_DEVICES=$render_gpu' in text
