"""Physical tasks -> official SceneCode -> PyBullet -> Blender -> CAE frontend.

Reference-only artifacts never claim a model-comparison CAE score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n')
    return path


SCENE_PROMPT = '''A warm contemporary science demonstration room, 4.8 m by 4.0 m, has pale walls, a neutral floor, a central oak worktable 1.8 m wide and 1.0 m deep with a top 0.75 m above the floor, a closed sideboard along the far wall, a framed abstract print, and two simple pendant lights. On the worktable are one solid red sphere 0.12 m in diameter, one solid blue cube 0.12 m on each side, and one solid yellow rectangular block 0.10 m wide, 0.10 m deep and 0.24 m tall, each a separate opaque single-piece object with a matte surface. Initially, the three objects rest upright and apart on the central tabletop, with open space around them and a clear near edge; the background furniture remains behind the work area. Soft warm-white light, subtle wood grain and visible contact shadows make the colored objects distinct against the table while retaining the room context.'''


TASKS = {
    'free_fall': {'subject': 'ball', 'instruction': 'The red sphere is released from rest above the table.'},
    'projectile': {'subject': 'ball', 'instruction': 'The red sphere starts moving horizontally above the table under gravity.'},
    'rebound': {'subject': 'ball', 'instruction': 'The red sphere is released above the table with an elastic contact material.'},
    'sliding': {'subject': 'cube', 'instruction': 'The blue cube starts sliding across the horizontal table.'},
    'collision': {'subject': 'ball', 'instruction': 'The red sphere moves horizontally toward the stationary blue cube on the table.'},
    'support_tip': {'subject': 'tower', 'instruction': 'A short horizontal force acts near the upper side of the yellow tall block.'},
}
BASE_TASKS = tuple(TASKS)
TASKS.update({
    'ramp_roll': {'subject':'ball','instruction':'Release the teal ball at the upper end of the wooden slope.'},
    'domino_chain': {'subject':'tower','instruction':'Tip the first red block toward the remaining upright row.'},
    'collision_chain': {'subject':'ball','instruction':'Roll the red ball toward the blue and yellow balls.'},
    'shallow_containment': {'subject':'ball','instruction':'Release the purple ball above the empty shallow tray.'},
})


def generate(workspace, output, *, scene_prompt=None, task_record=None):
    from .generate_scene import _environment_with_optional_secret, _hydra_string_override
    from .robotics_resource_lock import exclusive_blender_job
    from .robotics_scenecode_case_generator import ROBOTICS_SCENECODE_OVERRIDES
    destination = output/'generation'
    scene_prompt = scene_prompt or SCENE_PROMPT
    selected_tasks = {task_record['id']: task_record} if task_record else {name:TASKS[name] for name in BASE_TASKS}
    house = destination/'scene_000/combined_house/house_state.json'
    prompt = output/'scene_prompt.txt'
    if prompt.exists() and prompt.read_text()!=scene_prompt:
        raise ValueError('scene prompt changed; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True); prompt.write_text(scene_prompt)
    write(output/'tasks.json', {'schema_version':'physical_reference_tasks.v1','tasks':selected_tasks,'scene_prompt_sha256':digest(prompt),'physics_backend':'pybullet'})
    if house.exists():
        print('Reuse completed SceneCode scene:', house, flush=True)
        return house
    source = Path(os.environ.get('SCENECODE_SOURCE_ROOT', workspace/'code/SceneCode-main')).resolve()
    env = _environment_with_optional_secret(workspace/'.secrets/scenecode.env')
    env['PYTHONPATH'] = str(workspace/'code/WorldOlympiad-main')+os.pathsep+str(source)
    model = env.get('SCENECODE_OPENAI_MODEL','gpt-5.6-sol')
    command=[str(workspace/'envs/scenecode/bin/python'),str(source/'main.py'),'+name=physical_reference_pybullet',
        'experiment.single_scene_id=0',_hydra_string_override('experiment.single_prompt',scene_prompt),
        'experiment.num_workers=1',f'hydra.run.dir={destination}',*ROBOTICS_SCENECODE_OVERRIDES]
    for agent in ('floor_plan','furniture','wall','ceiling','manipuland'):
        command.extend([f'{agent}_agent.openai.model={model}',f'{agent}_agent.session_memory.summarization_model={model}'])
    # Resume completed upstream stages instead of recreating a room after interruption.
    stages=[('manipuland','scene_after_ceiling_objects'),('ceiling','scene_after_wall_objects'),('wall','scene_after_furniture')]
    for stage, checkpoint in stages:
        if any(destination.glob(f'scene_*/room_*/scene_states/{checkpoint}/scene_state.json')):
            command.extend([f'experiment.pipeline.resume_from_path={destination}',f'experiment.pipeline.start_stage={stage}'])
            break
    with exclusive_blender_job(workspace/'outputs'):
        with (output/'generation.log').open('a') as log:
            try:
                subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            except subprocess.CalledProcessError:
                tail=(output/'generation.log').read_text(errors='replace')[-12000:]
                reason='provider_daily_quota_exhausted' if '日限额已用完' in tail else 'generation_process_error'
                write(output/'generation_manifest.json',dict(stage='blocked',reason=reason,scene_prompt_sha256=digest(prompt),house_state=None))
                raise
    if not house.is_file(): raise RuntimeError('SceneCode did not produce combined house state')
    write(output/'generation_manifest.json',{'source':str(source.resolve()),'house_state':str(house),'house_state_sha256':digest(house),'scene_prompt_sha256':digest(prompt),'task_ids':list(selected_tasks)})
    return house


def bind_generated_scene(output, house):
    """Convert a completed official scene and bind its requested object roles."""
    from .scenecode_importer import import_house_state
    world=import_house_state(house,scene_id='physical_reference_generated')
    labels={b.name:(' '.join([b.name,b.semantic_category,str(b.metadata.get('description',''))])).lower() for b in world.bodies}
    rules={'table':('worktable','workbench','table'),'ball':('sphere','ball'),'cube':('cube',),'tower':('tall','yellow')}
    bindings={}
    for role,words in rules.items():
        candidates=[b for b in world.bodies if any(word in labels[b.name] for word in words)]
        if len(candidates)!=1:raise ValueError(f'expected one {role} from SceneCode; found {[b.name for b in candidates]}')
        bindings[role]={'body':candidates[0].name,'grounding':{'table':'wooden table','ball':'red sphere','cube':'blue cube','tower':'yellow tall block'}[role]}
    path=output/'generated_world_ir.json';world.write_json(path)
    write(output/'generated_bindings.json',bindings)
    return path,bindings


# The released LingBot-World v2 adapter requires native 832x464 reference
# pixels; rendering at any other size is rejected before the model runs.
NATIVE_RENDER_WH = (832, 464)


def camera_at(center, *, elevated):
    focus=np.asarray(center)+[0,0,0.20 if elevated else 0.08]
    eye=focus+np.asarray([0.48,-1.05,0.62])
    forward=(focus-eye)/np.linalg.norm(focus-eye)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    transform=np.eye(4);transform[:3,:3]=np.stack([right,-up,forward],axis=1);transform[:3,3]=eye
    width,height=NATIVE_RENDER_WH;fov=50.;focal=height/(2*np.tan(np.deg2rad(fov)/2))
    return dict(eye=eye.tolist(),look_at=focus.tolist(),up=up.tolist(),camera_to_world_opencv=transform.tolist(),
        intrinsics=[[focal,0,width/2],[0,focal,height/2],[0,0,1]],width=width,height=height,fov_y_deg=fov,near_m=.02,far_m=20.)


def simulate_tasks(workspace,output,source_world,bindings,selected):
    import pybullet as p
    from .compiler import compile_world
    from .models import WorldIR, Pose
    from .runner import _base_link_pose, _render
    from .physical_apparatus import EXPERIMENTS, apparatus
    original=WorldIR.model_validate_json(source_world.read_text())
    by_name={b.name:b for b in original.bodies}
    fresh=source_world==output/'generated_world_ir.json'
    source_manifest=write(output/'scene_source.json',dict(source_world=str(source_world),source_sha256=digest(source_world),
        source_kind='task_generated_scenecode' if fresh else 'existing_scenecode_assets_rebound_to_new_pybullet_tasks',fresh_scene_generation=fresh,bindings=bindings))
    for task_id in selected:
        task=TASKS[task_id];root=output/'tasks'/task_id;ref=root/'reference'
        ref.mkdir(parents=True,exist_ok=True)
        if (root/'simulation.json').exists():
            previous=json.loads((root/'simulation.json').read_text())
            if previous['source_manifest_sha256']!=digest(source_manifest): raise ValueError('source binding changed')
            continue
        extended=task_id in EXPERIMENTS
        roles={'table':'table'} if extended else {'table':'table','subject':task['subject']}
        if task_id=='collision':roles['other']='cube'
        bodies=[]
        for name,role in roles.items():
            b=by_name[bindings[role]['body']].model_copy(deep=True)
            b.name=name;b.static=name=='table';b.target=name=='subject';b.support=None if name=='table' else 'table'
            b.metadata.pop('precompiled_urdf_source',None)
            # compiler.py exports SceneCode glTF visuals to Z-up OBJ.
            b.metadata['visual_mesh_coordinate_frame']='worldir_z_up'
            bodies.append(b)
        layout={};trigger={};grounding=None
        if extended:
            new_bodies,layout,trigger,grounding=apparatus(task_id,root)
            bodies.extend(new_bodies)
            write(root/'apparatus.json',dict(source='procedural_physical_fixture',fresh_scenecode_assets=False,layout_bottom_center_m=layout,trigger=trigger))
        # Preserve the actual SceneCode room, not an isolated table on a world background.
        selected_names={bindings[role]['body'] for role in roles.values()}
        for original_body in original.bodies:
            if original_body.name in selected_names or not original_body.static:
                continue
            if original_body.semantic_category in ('robot_arm','sorting_tray','push_target_zone'):
                continue
            background=original_body.model_copy(deep=True)
            background.name='background_'+background.name
            background.target=False
            background.support=None
            background.metadata['visual_mesh_coordinate_frame']='worldir_z_up'
            bodies.append(background)
        world=WorldIR(scene_id=task_id,prompt=task['instruction'],bodies=bodies,
            provenance=dict(source_world=str(source_world),source_sha256=digest(source_world),bindings=bindings))
        world.write_json(root/'world_ir.json')
        bundle_path=compile_world(world,root/'pybullet_bundle')
        bundle=json.loads(bundle_path.read_text());p.connect(p.DIRECT)
        try:
            p.setGravity(0,0,-9.81);p.setTimeStep(1/240)
            p.setPhysicsEngineParameter(numSolverIterations=100,deterministicOverlappingPairs=1)
            ids={}
            for name,item in bundle['bodies'].items():
                ids[name]=p.loadURDF(item['urdf_path'],basePosition=item['pose']['xyz'],baseOrientation=item['pose']['quat_xyzw'],useFixedBase=item['static'],flags=p.URDF_USE_INERTIA_FROM_FILE)
                for link in range(-1,p.getNumJoints(ids[name])):
                    p.changeDynamics(ids[name],link,lateralFriction=.3,linearDamping=0,angularDamping=0,restitution=.9 if task_id=='rebound' else .05)
            lower,upper=np.asarray(p.getAABB(ids['table']))
            center=(lower+upper)/2;center[2]=upper[2]
            for name in ids:
                if name in layout:
                    low,high=np.asarray(p.getAABB(ids[name]));oldpos,orientation=p.getBasePositionAndOrientation(ids[name])
                    target=center+np.asarray(layout[name]);target[2]+=(high[2]-low[2])/2+.001
                    p.resetBasePositionAndOrientation(ids[name],np.asarray(oldpos)+target-(low+high)/2,orientation)
                    for link in range(-1,p.getNumJoints(ids[name])):
                        p.changeDynamics(ids[name],link,lateralFriction=.5 if task_id=='domino_chain' else .3,rollingFriction=.001 if task_id=='ramp_roll' else .003,restitution=.8 if task_id=='collision_chain' else .12)
                    continue
                if bundle['bodies'][name]['static']:continue
                low,high=np.asarray(p.getAABB(ids[name]));oldpos,orientation=p.getBasePositionAndOrientation(ids[name])
                target=center.copy();target[2]+=(high[2]-low[2])/2+.003
                if name=='subject':
                    if task_id in ('free_fall','projectile','rebound'):target[2]+=.35
                    if task_id in ('projectile','sliding','collision'):target[0]-=.22
                if name=='other':target[0]+=.02
                newpos=np.asarray(oldpos)+target-(low+high)/2
                p.resetBasePositionAndOrientation(ids[name],newpos,orientation)
            if task_id in ('projectile','sliding','collision'):
                speed={'projectile':.65,'sliding':.65,'collision':1.2}[task_id]
                p.resetBaseVelocity(ids['subject'],linearVelocity=[speed,0,0])
            if trigger.get('kind')=='velocity':p.resetBaseVelocity(ids[trigger['body']],linearVelocity=trigger['velocity'])
            camera=camera_at(center,elevated=task_id in ('free_fall','projectile','rebound'))
            write(ref/'calibration.json',camera)
            write(ref/'camera_calibration_info.json',{'head_camera':{'intrinsics':camera['intrinsics'],'extrinsics':camera['camera_to_world_opencv']}})
            records=[]
            for name,body in ids.items():
                records.append((name+'::'+p.getBodyInfo(body)[0].decode(),body,-1))
                for j in range(p.getNumJoints(body)):records.append((name+'::'+p.getJointInfo(body,j)[12].decode(),body,j))
            names=list(ids);times=[];positions=[];orientations=[];linear=[];links_p=[];links_q=[];depths=[];masks=[];contacts=[]
            for step in range(961):
                if step%16==0:
                    times.append(step/240);positions.append([_base_link_pose(p,ids[n])[0] for n in names]);orientations.append([_base_link_pose(p,ids[n])[1] for n in names]);linear.append([p.getBaseVelocity(ids[n])[0] for n in names])
                    poses=[_base_link_pose(p,b) if j==-1 else p.getLinkState(b,j,computeForwardKinematics=True)[4:6] for _,b,j in records]
                    links_p.append([v[0] for v in poses]);links_q.append([v[1] for v in poses])
                    _,depth,seg=_render(p,camera,p.ER_TINY_RENDERER);depths.append(depth);masks.append(seg==ids['subject'])
                if step==960:break
                if trigger.get('kind')=='push' and trigger['start_s']<=step/240<trigger['end_s']:
                    body=ids[trigger['body']];low,high=np.asarray(p.getAABB(body));at=(low+high)/2
                    at[2]=low[2]+trigger['height_fraction']*(high[2]-low[2])
                    mass=p.getDynamicsInfo(body,-1)[0]
                    p.applyExternalForce(body,-1,[mass*trigger['force_per_mass'],0,0],at.tolist(),p.WORLD_FRAME)
                if task_id=='support_tip' and 60<=step<84:
                    low,high=np.asarray(p.getAABB(ids['subject']));at=(low+high)/2;at[2]=high[2]-.005
                    mass=sum(p.getDynamicsInfo(ids['subject'],j)[0] for j in range(-1,p.getNumJoints(ids['subject'])))
                    p.applyExternalForce(ids['subject'],-1,[mass*8,0,0],at.tolist(),p.WORLD_FRAME)
                p.stepSimulation()
                for c in p.getContactPoints():
                    if c[1] in ids.values() and c[2] in ids.values():contacts.append(dict(time_s=(step+1)/240,body_a=c[1],body_b=c[2],distance_m=c[8],normal_force_n=c[9]))
            np.savez_compressed(ref/'state_trajectory.npz',timestamps=np.asarray(times),body_names=np.asarray(names),positions=np.asarray(positions),orientations_xyzw=np.asarray(orientations),linear_velocities=np.asarray(linear),link_names=np.asarray([v[0] for v in records]),link_positions=np.asarray(links_p),link_orientations_xyzw=np.asarray(links_q))
            np.save(ref/'depth_frames.npy',np.asarray(depths,dtype=np.float32));np.save(ref/'initial_subject_mask.npy',masks[0])
            write(ref/'contacts.json',contacts)
            if not np.isfinite(positions).all():raise ValueError('non-finite PyBullet state')
            phrase=grounding or bindings[task['subject']]['grounding']
            instruction=task['instruction']
            replacements={'red sphere':bindings['ball']['grounding'],'blue cube':bindings['cube']['grounding'],'yellow tall block':bindings['tower']['grounding']}
            if not extended:
                for old,new in replacements.items():instruction=instruction.replace(old,new)
            ys,xs=np.nonzero(masks[0])
            if not len(xs):raise ValueError('target not visible at frame zero')
            write(ref/'scene_data.json',dict(instruction=instruction,object_name=phrase,canonical_object_name='subject',initial_component_region=dict(schema_version='initial_component_box.v1',frame=0,image_size_wh=[camera['width'],camera['height']],component_id='subject',bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],source_kind='pybullet_frame_zero_projection')))
            write(root/'simulation.json',dict(task_id=task_id,physics_backend='pybullet',physics_dt_s=1/240,fps=15,frames=len(times),duration_s=4.,source_manifest_sha256=digest(source_manifest),bundle_path=str(bundle_path),target_body_id=ids['subject'],trajectory_sha256=digest(ref/'state_trajectory.npz'),instruction=instruction,body_instance_map=ids,stage='reference_simulated'))
            print(task_id,'PyBullet complete',flush=True)
        finally:p.disconnect()


def render_tasks(workspace,output,selected):
    from .robotics_resource_lock import exclusive_blender_job
    from .media import write_h264_rgb
    import cv2
    repo=workspace/'code/WorldOlympiad-main'
    for task_id in selected:
        root=output/'tasks'/task_id;ref=root/'reference';sim=json.loads((root/'simulation.json').read_text())
        if (root/'render.json').exists():continue
        frames=ref/'rgb_frames';frames.mkdir(exist_ok=True)
        calibration=json.loads((ref/'calibration.json').read_text())
        width,height=int(calibration['width']),int(calibration['height'])
        if (width,height)!=NATIVE_RENDER_WH:raise ValueError(f'reference must render at native model pixels, got {width}x{height}')
        # The native wrapper re-renders the shared replay at model pixels and
        # audits real Blender projections against the declared intrinsics.
        calibration['camera_audit_path']=str(ref/'camera_audit.json')
        write(ref/'calibration.json',calibration)
        config=write(ref/'blender_config.json',dict(bundle_path=sim['bundle_path'],trajectory_path=str(ref/'state_trajectory.npz'),calibration_path=str(ref/'calibration.json'),frames_dir=str(frames),fps=15,width=width,height=height,target_name='subject',robot_name=None,expected_frames=61,render_engine='BLENDER_EEVEE_NEXT',masks_dir=None))
        with exclusive_blender_job(workspace/'outputs'),(root/'render.log').open('a') as log:
            subprocess.run([str(workspace/'runtime/blender-4.4.3-linux-x64/blender'),'--background','--factory-startup','--python',str(repo/'scripts/blender_physics_native_reference.py'),'--','--config',str(config)],stdout=log,stderr=subprocess.STDOUT,check=True)
        audit=json.loads((ref/'camera_audit.json').read_text())
        if (audit['width'],audit['height'])!=(width,height) or audit['pixel_projection_max_abs_error']>1e-3 or not audit['direct_render'] or audit['rgb_resized']:
            raise ValueError('native render camera audit failed')
        paths=sorted(frames.glob('*.png'))
        if len(paths)!=61:raise ValueError(f'expected 61 RGB frames, found {len(paths)}')
        rgb=[cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB) for p in paths]
        write_h264_rgb(ref/'rgb.mp4',rgb,15,crf=16)
        cv2.imwrite(str(ref/'camera_rgb.png'),cv2.cvtColor(rgb[0],cv2.COLOR_RGB2BGR))
        write(root/'render.json',dict(renderer='Blender EEVEE',physics_backend='pybullet',video_sha256=digest(ref/'rgb.mp4'),frames=61,width=width,height=height,stage='reference_rendered',camera_audit_sha256=digest(ref/'camera_audit.json'),camera_projection_max_abs_error=audit['pixel_projection_max_abs_error'],camera_audit=audit))
        print(task_id,'Blender reference complete',flush=True)


def prepare_frontend_files(reference):
    import torch,yaml
    depth=np.load(reference/'depth_frames.npy')
    torch.save(torch.from_numpy(depth),reference/'depth_frames.pt');torch.save(torch.from_numpy(depth[0]),reference/'initial_depth.pt')
    data=json.loads((reference/'scene_data.json').read_text())
    (reference/'scene_data.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
    write(reference/'depth_frames.json',dict(source='pybullet_metric_depth',depth_shape=list(depth.shape),conditioning_video=str(reference/'rgb.mp4'),conditioning_video_sha256=digest(reference/'rgb.mp4'),camera_calibration_sha256=digest(reference/'camera_calibration_info.json')))


def flow_tasks(workspace,output,selected):
    from .robotics_resource_lock import exclusive_gpu_job
    repo=workspace/'code/WorldOlympiad-main';d2f=workspace/'code/Dream2Flow-main-cae-min64-vis08-v1'
    env=os.environ.copy();env.update(PYTHONPATH=str(repo)+os.pathsep+str(d2f),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for task_id in selected:
        root=output/'tasks'/task_id;ref=root/'reference';flow=ref/'dream2flow/object_flow_result.pt'
        if (root/'cae_frontend.json').exists():
            existing=json.loads((root/'cae_frontend.json').read_text())
            if existing['source_video_sha256']==digest(ref/'rgb.mp4') and existing['flow_sha256']==digest(flow) and existing['flow_npz_sha256']==digest(flow.with_suffix('.npz')):continue
        if not (root/'render.json').exists(): raise ValueError(f'{task_id}: render stage has not completed')
        subprocess.run([str(workspace/'envs/worldolympiad/bin/python'),'-m','autoresearch.physical_reference','--output',str(output),'--stage','prepare-depth','--tasks',task_id],env=env,check=True)
        flow.parent.mkdir(exist_ok=True)
        command=[str(workspace/'envs/dream2flow/bin/python'),str(repo/'scripts/run_dream2flow_reference.py'),'--dream2flow-repo',str(d2f),'--scene-dir',str(ref),'--video-role','reference','--source-model','physical_reference_only','--depth-mode','playback','--flow-only','--force-flow','--flow-output',str(flow),'--mask-erosion-kernel-size','3','--particle-ratio','0.01','--minimum-particles','64','--maximum-particles','1024','--expected-cotracker-visibility-threshold','0.8','--grounding-dino-model-path',str(workspace/'models/grounding-dino-base'),'--sam2-model-path',str(workspace/'models/sam2-hiera-large')]
        with exclusive_gpu_job(workspace/'outputs'),(root/'cae_frontend.log').open('a') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run([str(workspace/'envs/dream2flow/bin/python'),str(repo/'scripts/export_dream2flow_object_flow_npz.py'),str(flow),str(flow.with_suffix('.npz'))],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with np.load(flow.with_suffix('.npz')) as data:
            shape=data['position'].shape
            if shape[0]!=61 or not data['valid_mask'][0].any():raise ValueError('flow frame/validity mismatch')
        write(root/'cae_frontend.json',dict(stage='reference_flow_ready',flow_sha256=digest(flow),flow_npz_sha256=digest(flow.with_suffix('.npz')),shape=list(shape),source_video_sha256=digest(ref/'rgb.mp4'),model_comparison_score=None,reason='reference only; no generated video or robot registration',depth_source='pybullet_metric_depth'))
        print(task_id,'CAE reference flow complete',flush=True)


def summarize(output):
    cases=[]
    discovered={path.name for path in (output/'tasks').glob('*') if path.is_dir()}
    for task_id in sorted(set(TASKS)|discovered):
        root=output/'tasks'/task_id
        item={'task_id':task_id,'simulation':None,'render':None,'cae_frontend':None}
        for name in ('simulation','render','cae_frontend'):
            path=root/(name+'.json')
            if path.is_file(): item[name]=json.loads(path.read_text())
        apparatus_path=root/'apparatus.json'
        if apparatus_path.exists():item['apparatus']=json.loads(apparatus_path.read_text())
        cases.append(item)
    generation=output/'generation_manifest.json'
    source=output/'scene_source.json'
    summary={'schema_version':'physical_reference_batch.v1','generation':json.loads(generation.read_text()) if generation.exists() else None,
        'scene_source':json.loads(source.read_text()) if source.exists() else None,'cases':cases,
        'counts':{stage:sum(case[stage] is not None for case in cases) for stage in ('simulation','render','cae_frontend')},
        'model_comparison_score':None,'scope':'reference-side PyBullet, Blender video and CAE object-flow frontend; no robot optimizer or model comparison claimed'}
    write(output/'summary.json',summary)
    print(json.dumps(summary['counts']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path(os.environ.get('WORLDOLYMPIAD_WORKSPACE', Path(__file__).resolve().parents[3])))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['generate','simulate','render','flow','prepare-depth','reference','all','status'],default='generate')
    parser.add_argument('--source-world',type=Path)
    parser.add_argument('--bindings',type=Path)
    parser.add_argument('--prompt-id',help='SceneCode generation only: select a candidate from physical_prompt_catalog_v1.json')
    parser.add_argument('--tasks',nargs='+',default=list(BASE_TASKS))
    args=parser.parse_args()
    workspace,output=args.workspace.resolve(),args.output.resolve()
    if args.stage in ('simulate','reference','all'):
        unknown=set(args.tasks)-set(TASKS)
        if unknown:parser.error('legacy simulation has no declared program for '+', '.join(sorted(unknown)))
    if args.prompt_id:
        if args.stage!='generate':parser.error('--prompt-id currently supports --stage generate; new task execution bindings are listed in the catalog')
        catalog=json.loads((Path(__file__).parent/'configs/physical_prompt_catalog_v1.json').read_text())
        records={item['id']:item for item in catalog['items']}
        if args.prompt_id not in records:parser.error('unknown prompt id: '+args.prompt_id)
        record=records[args.prompt_id]
        generate(workspace,output,scene_prompt=record['scene_prompt'],task_record=record)
        return
    if args.stage=='all':
        house=generate(workspace,output)
        world,bindings=bind_generated_scene(output,house)
        simulate_tasks(workspace,output,world,bindings,args.tasks)
        render_tasks(workspace,output,args.tasks)
        flow_tasks(workspace,output,args.tasks)
        summarize(output)
        return
    if args.stage=='generate':generate(workspace,output)
    if args.stage in ('simulate','reference'):
        if not args.source_world or not args.bindings:parser.error('simulation requires --source-world and --bindings')
        simulate_tasks(workspace,output,args.source_world.resolve(),json.loads(args.bindings.read_text()),args.tasks)
    if args.stage in ('render','reference'):render_tasks(workspace,output,args.tasks)
    if args.stage in ('flow','reference'):flow_tasks(workspace,output,args.tasks)
    if args.stage=='prepare-depth':
        for task in args.tasks:prepare_frontend_files(output/'tasks'/task/'reference')
    else: summarize(output)


if __name__=='__main__': main()
