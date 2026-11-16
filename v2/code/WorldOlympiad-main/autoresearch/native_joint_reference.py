"""Contact-only native reference control for a declared single passive joint."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import sha256_file
from .articulated_reference_replay_binding import validate_reference_binding
from .articulated_native_task import reference_actions_for_native_replay, JointTermination


def pose_matrix(position, xyzw):
    value=np.eye(4);value[:3,:3]=Rotation.from_quat(xyzw).as_matrix();value[:3,3]=position
    return value


def jacobian_at_link_origin(com_jacobian, link_position, com_position):
    """PhysX reports COM Jacobians; pose control uses the rigid link origin."""
    value=np.asarray(com_jacobian,dtype=float).copy()
    offset=np.asarray(link_position)-np.asarray(com_position)
    value[:3]+=np.cross(value[3:].T,offset).T
    return value


def contact_balance_step(forces, offset_world, *, gain, limit):
    forces=np.asarray(forces,dtype=float).reshape(2,3)
    strength=np.linalg.norm(forces,axis=1)
    axis=forces[0]-forces[1]
    norm=float(np.linalg.norm(axis))
    result=np.asarray(offset_world,dtype=float).copy()
    if norm>1e-6 and strength.sum()>1e-6:
        result+=gain*(strength[0]-strength[1])/strength.sum()*axis/norm
    norm=float(np.linalg.norm(result))
    return result if norm<=limit else result*limit/norm


def bounded_ik_step(q, jacobian, current, desired, *, damping, max_delta, limits):
    error=np.r_[desired[:3,3]-current[:3,3],
        Rotation.from_matrix(desired[:3,:3]@current[:3,:3].T).as_rotvec()]
    jacobian=np.asarray(jacobian,dtype=float)
    delta=jacobian.T@np.linalg.solve(jacobian@jacobian.T+damping*damping*np.eye(6),error)
    peak=float(np.max(np.abs(delta)))
    if peak>max_delta:delta*=max_delta/peak
    return np.clip(np.asarray(q)+delta,limits[:,0],limits[:,1])


def run(program_path, *, config, task, robot, target, arm_ids, gripper_ids, ee_ids,
        advance, measured, output, observe=None):
    import torch
    from .robotics_droid_control import executable_robolab_gripper_target
    from .cosmos3_droid_window import write_json_atomic
    program_path=Path(program_path);program=json.loads(program_path.read_text())
    if program.get('schema_version')!='native_joint_reference_program.v1':
        raise ValueError('unsupported native reference program')
    source=Path(program['approach_trajectory'])
    if sha256_file(source)!=program['approach_trajectory_sha256']:
        raise ValueError('reference approach controls changed')
    reference=json.loads(source.read_text());binding=validate_reference_binding(reference,config)
    actions=reference_actions_for_native_replay(reference,config['joint_name'])
    end=int(program['approach_last_frame'])
    if not 1<=end<len(actions):raise ValueError('reference approach endpoint is outside its controls')
    limits=robot.data.joint_pos_limits[0,arm_ids].detach().cpu().numpy()
    frames=[measured(0,[False,False])];controls=[];checker=JointTermination(task)
    terminal={'task_success':False,'done':False,'reason':None};failure=None;regrasps=0
    def execute(arm,grip,phase):
        nonlocal terminal
        if len(controls)>=program['max_steps']:raise RuntimeError('native reference control budget exhausted')
        target[0,arm_ids]=torch.as_tensor(arm,dtype=target.dtype,device=target.device)
        target[0,gripper_ids]=executable_robolab_gripper_target(grip)
        robot.set_joint_position_target(target)
        contacts=advance();step=len(controls)+1;frame=measured(step,contacts)
        if not frame['physical_valid']:raise RuntimeError('native reference violated robot or joint constraints')
        terminal=checker.step(frame['articulated'],contacts=any(contacts),gripper=frame['robot']['gripper'],step=step)
        frame['terminal']=terminal;frame['phase']=phase;frames.append(frame);controls.append([*list(arm),grip])
        if observe is not None and step in program.get('diagnostic_observation_steps',[]):
            observe(step,output/f'view_{step:04d}')
        if step%30==0:print(f"NATIVE_REFERENCE step={step} phase={phase} joint={frame['articulated']['position']:.6f} contact={contacts}",flush=True)
        return frame
    def ee_pose(frame):
        quat=frame['robot']['ee_quaternion_wxyz']
        return pose_matrix(frame['robot']['ee_position'],[*quat[1:],quat[0]])
    def command_pose(desired,frame,grip,phase):
        index=ee_ids[0]-1 if robot.is_fixed_base else ee_ids[0]
        com_jacobian=robot.root_physx_view.get_jacobians()[0,index,:,arm_ids].detach().cpu().numpy()
        com_position=robot.data.body_com_pos_w[0,ee_ids[0]].detach().cpu().numpy()
        jacobian=jacobian_at_link_origin(com_jacobian,frame['robot']['ee_position'],com_position)
        velocity=robot.data.joint_vel[0,arm_ids].detach().cpu().numpy()
        measured_velocity=robot.data.body_link_lin_vel_w[0,ee_ids[0]].detach().cpu().numpy()
        jacobian_diagnostic={
            'com_to_link_origin_world_m':(np.asarray(frame['robot']['ee_position'])-com_position).tolist(),
            'com_jacobian_link_velocity_error_m_s':float(np.linalg.norm(com_jacobian[:3]@velocity-measured_velocity)),
            'link_jacobian_link_velocity_error_m_s':float(np.linalg.norm(jacobian[:3]@velocity-measured_velocity))}
        arm=bounded_ik_step(frame['robot']['arm_joints'],jacobian,ee_pose(frame),desired,
            damping=program['ik_damping'],max_delta=program['max_joint_delta_rad'],limits=limits)
        result=execute(arm,grip,phase)
        result['reference_control_error']={
            'jacobian_diagnostic':jacobian_diagnostic,
            'desired_ee_position_world':desired[:3,3].tolist(),
            'desired_ee_quaternion_xyzw':Rotation.from_matrix(desired[:3,:3]).as_quat().tolist(),
            'position_error_m':float(np.linalg.norm(desired[:3,3]-ee_pose(result)[:3,3])),
            'rotation_error_rad':float(Rotation.from_matrix(desired[:3,:3]@ee_pose(result)[:3,:3].T).magnitude()),
            'minimum_arm_limit_margin_rad':float(np.min(np.minimum(np.asarray(result['robot']['arm_joints'])-limits[:,0],limits[:,1]-np.asarray(result['robot']['arm_joints']))))}
        return result
    try:
        for action in actions[1:end+1]:frame=execute(action[:7],float(action[7]),'approach_and_grasp')
        if program.get('grasp_seating_m',0):
            seating=ee_pose(frame)
            normal=Rotation.from_quat(frame['child_link_xyzw']).apply(reference['binding']['contact_normal_local_xyz'])
            for index in range(1,program['seating_steps']+1):
                desired=seating.copy();desired[:3,3]-=normal*program['grasp_seating_m']*index/program['seating_steps']
                frame=command_pose(desired,frame,0.,'handle_seating')
            grasp_count=0
            for _ in range(program['grasp_close_steps']):
                frame=execute(frame['robot']['arm_joints'],1.,'native_grasp_close')
                grasp_count=grasp_count+1 if all(frame['finger_contacts']) else 0
            if grasp_count<3:raise RuntimeError('seated native grasp did not become stable')
        if not all(frame['finger_contacts']):
            raise RuntimeError('native approach did not establish two-finger contact')
        if program.get('grasp_balance_m',0):
            forces=np.asarray(frame['robot']['finger_total_contact_forces_w']).reshape(2,3)
            strengths=np.linalg.norm(forces,axis=1)
            if min(strengths)<=1e-3:raise RuntimeError('grasp balance requires measured forces on both fingers')
            # Translate toward the stronger-contact finger to reduce that
            # jaw's penetration and seat the opposite jaw, without changing force.
            direction=forces[int(np.argmax(strengths))]/max(strengths)
            start=ee_pose(frame)
            for index in range(1,program['grasp_balance_steps']+1):
                desired=start.copy();desired[:3,3]+=direction*program['grasp_balance_m']*index/program['grasp_balance_steps']
                frame=command_pose(desired,frame,1.,'grasp_balance')
            if not all(frame['finger_contacts']):raise RuntimeError('force-balanced grasp lost contact')
        grasp_relative=np.linalg.inv(pose_matrix(frame['child_link_position'],frame['child_link_xyzw']))@ee_pose(frame)
        balance_local=np.zeros(3)
        lost=0
        sign=np.sign(task.target_position-task.closed_position)
        while frame['articulated']['target_error']>task.tolerance*.5:
            current=float(frame['articulated']['position'])
            coordinate=np.clip(current+sign*program['joint_lead'],task.lower,task.upper)
            coordinate= min(coordinate,task.target_position) if sign>0 else max(coordinate,task.target_position)
            position,quaternion=task.pose_at(coordinate)
            desired=pose_matrix(position,quaternion)@grasp_relative
            if program.get('adaptive_grasp_balance_gain_m',0):
                door_rotation=Rotation.from_quat(frame['child_link_xyzw'])
                offset=contact_balance_step(frame['robot']['finger_total_contact_forces_w'],
                    door_rotation.apply(balance_local),gain=program['adaptive_grasp_balance_gain_m'],
                    limit=program['adaptive_grasp_balance_limit_m'])
                balance_local=door_rotation.inv().apply(offset)
                desired[:3,3]+=Rotation.from_quat(quaternion).apply(balance_local)
            frame=command_pose(desired,frame,1.,'measured_joint_pull')
            frame['grasp_balance_offset_child_m']=balance_local.tolist()
            lost=0 if all(frame['finger_contacts']) else lost+1
            if lost>program['max_lost_contact_steps']:
                if regrasps>=program.get('max_regrasps',0):
                    raise RuntimeError('native reference lost its measured two-finger grasp')
                regrasps+=1
                # Recover at the observed door angle, with the fingers open.
                # This commands only robot joints; the passive door remains free.
                for _ in range(program['release_steps']):
                    frame=execute(frame['robot']['arm_joints'],0.,'regrasp_open')
                for _ in range(program['regrasp_pose_steps']):
                    position,quaternion=task.pose_at(frame['articulated']['position'])
                    desired=pose_matrix(position,quaternion)@grasp_relative
                    frame=command_pose(desired,frame,0.,'regrasp_center')
                stable=0
                for _ in range(program['grasp_close_steps']):
                    # The released door can still move during closure. Follow
                    # its measured angle rather than closing at an old pose.
                    position,quaternion=task.pose_at(frame['articulated']['position'])
                    desired=pose_matrix(position,quaternion)@grasp_relative
                    frame=command_pose(desired,frame,1.,'regrasp_close')
                    stable=stable+1 if all(frame['finger_contacts']) else 0
                if stable<3:raise RuntimeError('native regrasp did not establish stable two-finger contact')
                # Anchor the recovered contact frame to the actual seated
                # grasp, so the next pull does not jump back to the old grasp.
                grasp_relative=np.linalg.inv(pose_matrix(frame['child_link_position'],frame['child_link_xyzw']))@ee_pose(frame)
                balance_local=np.zeros(3)
                lost=0
        for _ in range(program['settle_steps']):frame=execute(frame['robot']['arm_joints'],1.,'grasp_settle')
        release_pose=ee_pose(frame)
        for _ in range(program['release_steps']):frame=execute(frame['robot']['arm_joints'],0.,'release')
        normal=Rotation.from_quat(frame['child_link_xyzw']).apply(reference['binding']['contact_normal_local_xyz'])
        for index in range(1,program['retreat_steps']+1):
            desired=release_pose.copy();desired[:3,3]+=normal*program['retreat_distance_m']*index/program['retreat_steps']
            frame=command_pose(desired,frame,0.,'retreat')
        for _ in range(program['final_hold_steps']):frame=execute(frame['robot']['arm_joints'],0.,'final_hold')
    except RuntimeError as exc:
        failure=str(exc)
    finally:
        np.save(output/'reference_controls.npy',np.asarray(controls,dtype=np.float32),allow_pickle=False)
        write_json_atomic(output/'trajectory.json',{'schema_version':'native_joint_reference_measured.v1','sample_hz':15,'frames':frames})
        result={'schema_version':'native_joint_reference_result.v1','passed':failure is None and terminal['task_success'],
            'failure':failure,'terminal':terminal,'maximum_joint_progress':max(abs(f['articulated']['position']-task.closed_position) for f in frames),
            'action_count':len(controls),'program_sha256':sha256_file(program_path),'reference_binding':binding,
            'controller_source_sha256':sha256_file(Path(__file__)),
            'regrasp_count':regrasps,
            'control_source':'native measured-pose IK and joint feedback','model_inference_launched':False,
            'object_pose_resets_after_initialization':False,'physical_robot_controller_changed':False}
        write_json_atomic(output/'result.json',result)
        if observe is not None:observe(len(controls),output/'terminal_view')
        print(json.dumps(result),flush=True)
    return result
