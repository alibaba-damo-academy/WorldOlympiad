#!/usr/bin/env python3
"""Score prepared native cases with local weights and one managed VLM service."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from contextlib import ExitStack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoresearch.robotics_resource_lock import exclusive_gpu_job
from continuous_action.official_plan import sha256


QWENVL_STARTUP_TIMEOUT_S = 600
QWENVL_READY_POLL_INTERVAL_S = 2


def run_cae(roots, workspace, compute_gpu_id, pair_validation=None):
    from scripts.run_official_dream2flow_case import run_case
    from autoresearch.full_track_report import build_report_v3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(compute_gpu_id)
    for root in roots:
        case, _video, _reference, _prompt, _chunks, hashes = inputs(root)
        receipt = json.loads((root/'three_track_receipt.json').read_text())
        if receipt['inputs'] != hashes or receipt['score_sha256'] != sha256(root/'score.json'):
            raise ValueError('three-track scoring receipt differs from the selected pair')
        # Applicability comes from the pre-inference case contract. A low CAE
        # score or a model task failure cannot change it.
        applicability = case['tracks'].get('continuous_action')
        if applicability not in ('pending_symmetric_dream2flow_manifest', 'pending_articulated_registration'):
            raise ValueError('this native runner requires declared rigid CAE applicability')
        if applicability == 'pending_articulated_registration' and pair_validation is None:
            raise ValueError('articulated CAE requires independent native pair validation')
        report = run_case(root, workspace, force_flow=False, pair_validation=pair_validation)
        plan = {'schema_version': 'worldolympiad_track_applicability.v1', 'cases': [{
            'case_id': case['case_id'], 'task_sha256': sha256(root/'worldolympiad_case.json'),
            'tracks': {name: True for name in ('physical','geometry','interaction','continuous_action')}}]}
        combined = build_report_v3([root], applicability_plan=plan, action_report=report)
        path = root/'three_track_report_v1.json'
        if path.exists() and json.loads(path.read_text()) != combined:
            raise FileExistsError('completed three-track report changed')
        if not path.exists(): path.write_text(json.dumps(combined,indent=2)+'\n')
        if combined['complete_count'] != 1: raise ValueError('native case scoring is still pending')
        print(f"CAE and all three tracks complete: {case['case_id']}",flush=True)


def inputs(root):
    case=json.loads((root/'worldolympiad_case.json').read_text())
    video=root/case['generated_video'];reference=root/case['reference_video']
    prompt=root/'prompt.json';chunks=root/(video.stem+'_chunk_timestamps.json')
    for path in (video,reference,prompt,chunks):
        if not path.is_file():raise FileNotFoundError(path)
    hashes={str(path.relative_to(root)):sha256(path) for path in (video,reference,prompt,chunks)}
    return case,video,reference,prompt,chunks,hashes


def _archive_uncommitted_score(root, score, receipt):
    if receipt.exists():
        return
    if score.exists():
        history=root/'.score_history'/str(time.time_ns());history.mkdir(parents=True,exist_ok=True)
        shutil.move(str(score),str(history/score.name))


def _score_command(repo,workspace,weights,python,case,video,reference,prompt,chunks,output,phase,*,local_vlm=False):
    backend='local' if local_vlm else 'qwenvl_server'
    command=[str(python),str(repo/'scripts/score_worldolympiad.py'),
        '--video',str(video),'--gt-video',str(reference),
        '--prompt',prompt.read_text(),'--prompt-json',str(prompt),'--chunk-json',str(chunks),
        '--vlm-backend',backend,'--vlm-model',str(weights/'QwenVL'),
        '--interaction-vlm-backend',backend,'--interaction-vlm-model',str(weights/'QwenVL'),
        '--three-d-model-name',str(weights/'da3'),'--three-d-vlm-backend',backend,
        '--three-d-scoring-model',str(weights/'QwenVL'),'--sam3-model',str(weights/'sam3/sam3.pt'),
        '--sam-device','cuda:0','--da-device','cuda:0','--num-workers','1',
        '--physical-batch-mode','dimension','--output',str(output)]
    if phase == 'physical':
        command += ['--skip-interaction','--skip-3d']
    elif phase == 'interaction':
        command += ['--skip-physical','--skip-3d','--run-clip-interaction',
                    '--clip-device','cuda:0','--clip-download-root',str(weights/'clip')]
    elif phase == 'geometry':
        command += ['--skip-physical','--skip-interaction','--offload-qwenvl-before-3d']
    elif phase == 'full':
        command += ['--run-clip-interaction','--clip-device','cuda:0',
                    '--clip-download-root',str(weights/'clip')]
    else:
        raise ValueError(f'unknown score phase: {phase}')
    return command


def _start_qwenvl(repo,workspace,weights,python,gpu_id,env,logpath):
    log=logpath.open('w')
    service=subprocess.Popen([str(python),str(repo/'batch_test/start_qwenvl_servers.py'),
        '--worldeval-root',str(repo),'--gpus',str(gpu_id),'--ports','8008','--model',str(weights/'QwenVL'),
        '--quiet'],stdout=log,stderr=subprocess.STDOUT,cwd=workspace,env=env)
    deadline = time.monotonic() + QWENVL_STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if service.poll() is not None:
            log.close()
            raise RuntimeError(f'VLM startup failed: {logpath}')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8008/health', timeout=2) as response:
                ready = response.status == 200 and json.load(response).get('model') == str(weights/'QwenVL')
            if ready:
                return service,log
        except (OSError, ValueError, TimeoutError):
            pass
        time.sleep(min(QWENVL_READY_POLL_INTERVAL_S, max(0, deadline - time.monotonic())))
    service.terminate()
    try: service.wait(timeout=30)
    except subprocess.TimeoutExpired: service.kill();service.wait()
    log.close()
    raise TimeoutError(f'VLM did not become ready: {logpath}')


def _stop_qwenvl(service,log):
    if service is None:return
    service.terminate()
    try:service.wait(timeout=30)
    except subprocess.TimeoutExpired:service.kill();service.wait()
    log.close()


def _require_idle_compute_gpu(gpu_id, phase):
    from wo.agent.config import (
        all_gpu_info, _score_gpu_reason, SPLIT_SCORE_MIN_FREE_MIB,
    )
    inventory = all_gpu_info()
    info = next((item for item in inventory if item['index'] == int(gpu_id)), None)
    reason = _score_gpu_reason(info, SPLIT_SCORE_MIN_FREE_MIB)
    if reason != 'eligible':
        raise RuntimeError(
            f'fresh scoring preflight blocked compute GPU {gpu_id} before {phase}: {reason}'
        )


def _run_phase(repo,workspace,weights,python,root,data,attempt,phase,env,*,local_vlm=False):
    case,video,reference,prompt,chunks,_hashes=data
    output=attempt/f'{phase}.json';logpath=attempt/f'{phase}.log'
    command=_score_command(repo,workspace,weights,python,case,video,reference,prompt,chunks,
                           output,phase,local_vlm=local_vlm)
    with logpath.open('w') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=workspace,env=env)
    return json.loads(output.read_text())


def _merge_split(parts,attempt):
    from scripts import score_video_physical_3d as scorer
    physical=parts['physical'].get('physical')
    interaction=parts['interaction'].get('interaction')
    geometry=parts['geometry'].get('three_d')
    summary=dict(parts['physical'])
    summary.update(physical=physical,interaction=interaction,
                   clip_interaction=parts['interaction'].get('clip_interaction'),
                   three_d=geometry)
    artifacts=dict(parts['physical'].get('artifacts') or {})
    for phase in ('interaction','geometry'):
        for key,value in (parts[phase].get('artifacts') or {}).items():
            if value is not None:artifacts[key]=value
    artifacts['split_score_parts']={key:str(attempt/f'{key}.json') for key in parts}
    summary['artifacts']=artifacts
    action=summary.get('continuous_action') or {'status':'not_applicable','reason':'no_cae_manifest'}
    summary['combined_score']=scorer.combine_scores(physical,interaction,geometry)
    summary['worldolympiad2_score'],summary['worldolympiad2_tracks']=scorer.combine_worldolympiad2_scores(
        physical,interaction,geometry,action)
    summary['score_policy']=scorer.SCORE_POLICY
    summary['track_results']=scorer.from_judge_summaries(
        physical,interaction,geometry,action)['track_results']
    return summary


def _commit_score(root,summary,hashes,case_id):
    score=root/'score.json';tmp=root/f'.score.{time.time_ns()}.tmp'
    tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    os.replace(tmp,score)
    receipt=root/'three_track_receipt.json';receipt_tmp=root/f'.receipt.{time.time_ns()}.tmp'
    receipt_tmp.write_text(json.dumps(dict(case_id=case_id,inputs=hashes,score_sha256=sha256(score),
        task_success_used_for_scoring=False),indent=2)+'\n')
    os.replace(receipt_tmp,receipt)


def run(roots,workspace,gpu_id,score_mode='full',compute_gpu_id=None):
    if score_mode not in {'full','split'}:raise ValueError('score_mode must be full or split')
    compute_gpu_id = str(gpu_id if compute_gpu_id is None else compute_gpu_id)
    gpu_id = str(gpu_id)
    from wo.agent.config import configured_gpu_indices
    allowed = set(configured_gpu_indices())
    if not allowed:
        raise ValueError('set WOV2_GPU_ALLOWLIST before scoring')
    if not gpu_id.isdigit() or int(gpu_id) not in allowed:
        raise ValueError('VLM GPU must be in WOV2_GPU_ALLOWLIST')
    if not compute_gpu_id.isdigit() or int(compute_gpu_id) not in allowed:
        raise ValueError('compute GPU must be in WOV2_GPU_ALLOWLIST')
    if score_mode == 'split' and gpu_id == compute_gpu_id:
        raise ValueError('split scoring needs distinct VLM-service and SAM/DA compute GPUs')
    repo=Path(__file__).resolve().parents[1];python=workspace/'envs/worldeval/bin/python'
    weights=workspace/'weights'
    if not weights.is_dir() and (repo/'weights').is_dir():weights=repo/'weights'
    vlm_env=os.environ.copy()
    vlm_env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                   CUDA_VISIBLE_DEVICES=gpu_id,QWENVL_SERVER_URL='http://127.0.0.1:8008')
    compute_env=os.environ.copy()
    compute_env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                       CUDA_VISIBLE_DEVICES=compute_gpu_id,WOV2_SCORE_COMPUTE_GPU=compute_gpu_id,
                       QWENVL_SERVER_URL='http://127.0.0.1:8008')
    pending=[]
    for root in roots:
        data=inputs(root);receipt=root/'three_track_receipt.json';score=root/'score.json'
        if receipt.exists():
            previous=json.loads(receipt.read_text())
            if previous['inputs']!=data[-1] or previous['score_sha256']!=sha256(score):
                raise ValueError(f'completed scoring inputs changed: {root}')
            continue
        _archive_uncommitted_score(root,score,receipt)
        pending.append((root,data))
    if not pending:return
    logroot=workspace/'outputs/evaluator_services';logroot.mkdir(parents=True,exist_ok=True)
    attempts={root:root/'score_stages'/f'{time.time_ns()}_{uuid.uuid4().hex[:8]}' for root,_ in pending}
    for path in attempts.values():path.mkdir(parents=True,exist_ok=False)
    service=service_log=None
    try:
        from wo.agent.config import (
            all_gpu_info, _score_gpu_reason, SPLIT_SCORE_MIN_FREE_MIB,
            FULL_SCORE_MIN_FREE_MIB,
        )
        inventory = all_gpu_info()
        required_free = FULL_SCORE_MIN_FREE_MIB if score_mode == 'full' else SPLIT_SCORE_MIN_FREE_MIB
        for selected in {int(gpu_id), int(compute_gpu_id)}:
            info = next((item for item in inventory if item['index'] == selected), None)
            reason = _score_gpu_reason(info, required_free)
            if reason != 'eligible':
                raise RuntimeError(f'fresh scoring preflight blocked GPU {selected}: {reason}')
        service,service_log=_start_qwenvl(repo,workspace,weights,python,gpu_id,vlm_env,
            logroot/f'robotics_qwenvl_{time.time_ns()}.log')
        if score_mode=='full':
            for root,data in pending:
                result=_run_phase(repo,workspace,weights,python,root,data,attempts[root],'full',compute_env)
                _commit_score(root,result,data[-1],data[0]['case_id'])
                print(f"Scored {data[0]['case_id']}",flush=True)
        else:
            parts_by_root={root:{} for root,_ in pending}
            for phase in ('physical','interaction'):
                for root,data in pending:
                    _require_idle_compute_gpu(compute_gpu_id, phase)
                    parts_by_root[root][phase]=_run_phase(
                        repo,workspace,weights,python,root,data,attempts[root],phase,compute_env)
            _stop_qwenvl(service,service_log);service=service_log=None
            for root,data in pending:
                _require_idle_compute_gpu(compute_gpu_id, 'geometry')
                parts_by_root[root]['geometry']=_run_phase(
                    repo,workspace,weights,python,root,data,attempts[root],'geometry',compute_env,local_vlm=True)
                summary=_merge_split(parts_by_root[root],attempts[root])
                _commit_score(root,summary,data[-1],data[0]['case_id'])
                print(f"Scored {data[0]['case_id']} in three sequential GPU phases",flush=True)
    finally:
        _stop_qwenvl(service,service_log)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root',type=Path,nargs='+',required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--gpu-id',default='0',help='physical GPU index for the VLM service')
    parser.add_argument('--compute-gpu-id',help='physical GPU index for SAM/DA/CLIP work')
    parser.add_argument('--score-mode',choices=('full','split'),default=os.environ.get('WO_SCORE_MODE','full'))
    parser.add_argument('--run-cae',action='store_true')
    parser.add_argument('--cae-only',action='store_true')
    parser.add_argument('--pair-validation',type=Path)
    parser.add_argument('--after-job',help='Wait for this durable native job before reading its case artifacts.')
    parser.add_argument('--reuse-source-case',type=Path,nargs='+')
    args=parser.parse_args()
    # --after-job waited on the retired olympiad_forge queue; dependency ordering
    # is now the job queue's `after` field, so this flag is refused rather than
    # silently ignored.  The case artifacts are validated below either way.
    if args.after_job:
        raise SystemExit('--after-job was removed with the olympiad_forge queue; '
                         'order jobs with wo.agent.jobs `after` instead')
    if args.reuse_source_case:
        from scripts.reuse_cosmos_cae_reference import prepare
        if len(args.reuse_source_case) != len(args.case_root):raise ValueError('CAE reuse sources must match case roots one-to-one')
        for source,root in zip(args.reuse_source_case,args.case_root,strict=True):
            if not (root/'cae_reference_reuse.json').exists():prepare(source,root)
    compute_gpu_id = args.gpu_id if args.compute_gpu_id is None else args.compute_gpu_id
    with ExitStack() as leases:
        for gpu in sorted({str(args.gpu_id), str(compute_gpu_id)}, key=int):
            leases.enter_context(exclusive_gpu_job(args.workspace/'outputs', gpu))
        roots=[root.resolve() for root in args.case_root]
        if not args.cae_only:
            run(roots,args.workspace.resolve(),args.gpu_id,args.score_mode,compute_gpu_id)
        if args.run_cae or args.cae_only:
            run_cae(roots,args.workspace.resolve(),compute_gpu_id,args.pair_validation)
