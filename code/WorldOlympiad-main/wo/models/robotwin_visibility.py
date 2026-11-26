"""Keep native camera settings; fit the scene with a common rigid planar transform."""
import numpy as np
from scipy.optimize import linprog


def corners(bounds):
    lo, hi = bounds
    return np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])


def fit_shift(K, E, width, height, points, footprints, workspace, margin=3):
    # Four LP variables: dx, dy, |dx|, |dy|. No scale, camera or asset edits.
    A, b = [], []
    def constrain(coeff, rhs):
        A.append([*coeff,0.,0.]); b.append(float(rhs))
    P = np.asarray(K) @ np.asarray(E)[:3]
    for p in points:
        h = P @ np.r_[p,1.]
        depth = np.asarray(E)[2] @ np.r_[p,1.]
        constrain(-np.asarray(E)[2,:2],depth-1e-3)
        for axis, maximum in [(0,width-1-margin),(1,height-1-margin)]:
            constrain(margin*P[2,:2]-P[axis,:2],h[axis]-margin*h[2])
            constrain(P[axis,:2]-maximum*P[2,:2],maximum*h[2]-h[axis])
    for lo, hi in footprints:
        for axis in (0,1):
            v = [0.,0.]; v[axis]=1.
            constrain(v,workspace[axis][1]-hi[axis])
            constrain(-np.asarray(v),lo[axis]-workspace[axis][0])
    A += [[1,0,-1,0],[-1,0,-1,0],[0,1,0,-1],[0,-1,0,-1]]
    b += [0.,0.,0.,0.]
    result = linprog([0,0,1,1],A_ub=A,b_ub=b,bounds=[(None,None),(None,None),(0,None),(0,None)],method='highs')
    return {'status':'ready' if result.success else 'infeasible',
            'shift_xy_m':result.x[:2].tolist() if result.success else None,
            'method':'minimum rigid XY translation; measured native bounds and unchanged head camera',
            'margin_pixels':margin}


def fit_planar(K, E, width, height, points, footprints, workspace, margin=3, *,
               geometry=None, interaction_points=()):
    result = fit_shift(K, E, width, height, points, footprints, workspace, margin)
    result['yaw_radians'] = 0.0
    if result['status'] == 'ready' or geometry is None:
        return result
    from .lingbot_va import _rotz
    for degrees in range(5, 181, 5):
        for sign in (1, -1):
            yaw = float(np.deg2rad(sign * degrees))
            R = _rotz(yaw)
            rotated = [item['points'] @ R.T for item in geometry]
            bounds = [(p.min(0), p.max(0)) for p in rotated]
            visible = [corners(b) for b, item in zip(bounds, geometry) if item['visible']]
            if len(interaction_points):
                visible.append(np.asarray(interaction_points) @ R.T)
            visible = np.vstack(visible)
            candidate = fit_shift(K, E, width, height, visible, bounds, workspace, margin)
            if candidate['status'] == 'ready':
                candidate.update(yaw_radians=yaw, method=(
                    'minimum feasible 5-degree rigid yaw and XY translation; '
                    'measured native bounds and unchanged head camera'))
                return candidate
    return result


def align(task, plan):
    from .lingbot_va import WORKSPACE_RT, to_rt, _rotz, _q_xyzw_mat, _mat_wxyz
    from .lingbot_va_client import SapienWorld
    import sapien
    world = SapienWorld(task,task.wo_handles,plan['support_id'],plan.get('articulations',{}))
    backgrounds = set(plan.get('fixed_background_ids', []))
    if backgrounds & set(plan['required_visible']):
        raise ValueError('required-visible objects cannot be excluded as fixed background')
    if any(not o['fixed'] or o['role'] != 'distractor' for o in plan['objects']
           if o['object_id'] in backgrounds):
        raise ValueError('only fixed distractors can be excluded from camera workspace fit')
    movable = [o for o in plan['objects'] if not (o['fixed'] and o['role']=='support')
               and o['object_id'] not in backgrounds]
    points, footprints, geometry = [], [], []
    for obj in movable:
        vertices = world.collision_vertices(obj['object_id'])
        bounds = (vertices.min(0), vertices.max(0)); footprints.append(bounds)
        geometry.append({'points':vertices, 'visible':obj['object_id'] in plan['required_visible']})
        if obj['object_id'] in plan['required_visible']:
            points.extend(corners(bounds))
    # A large fixture need not fit in the image; its measured operable region does.
    interactions = [to_rt(plan['transform'],point,[0,0,0,1])['p']
                    for point in plan['transform'].get('reference_interaction_points',[])]
    points.extend(interactions)
    if not points:
        return {'status':'infeasible','reason':'no measured visible task geometry'}
    cfg = task.cameras.get_config()['head_camera']
    names = task.cameras.static_camera_name
    camera = task.cameras.static_camera_list[names.index('head_camera')]
    result = fit_planar(np.asarray(cfg['intrinsic_cv']),np.asarray(cfg['extrinsic_cv']),
                       camera.get_width(),camera.get_height(),points,footprints,WORKSPACE_RT,
                        geometry=geometry, interaction_points=interactions)
    if result['status']!='ready':
        return result
    delta = np.r_[result['shift_xy_m'],0.]
    yaw = result['yaw_radians']
    if np.linalg.norm(delta)<1e-9 and yaw == 0.0:
        return result
    R = _rotz(yaw)
    for obj in plan['objects']:
        handle = task.wo_handles[obj['object_id']]
        pose = obj['pose_rt']
        p = R @ np.asarray(pose['p']) + delta
        q = pose['q_wxyz'] if yaw == 0.0 else _mat_wxyz(R @ _q_xyzw_mat(np.roll(pose['q_wxyz'], -1)))
        shifted = sapien.Pose(p=p, q=q)
        if hasattr(handle,'set_root_pose'):handle.set_root_pose(shifted)
        else:handle.set_pose(shifted)
        pose.update(p=p.tolist(), q_wxyz=q)
    tf = plan['transform']
    tf['xy_offset'] = (R[:2,:2] @ np.asarray(tf['xy_offset'])+delta[:2]).tolist()
    tf['yaw'] += yaw
    if 'footprint_rt' in tf:
        lo, hi = tf['footprint_rt']
        xy = np.array([[x,y] for x in (lo[0],hi[0]) for y in (lo[1],hi[1])]) @ R[:2,:2].T + delta[:2]
        tf['footprint_rt'] = [xy.min(0).tolist(), xy.max(0).tolist()]
    return result


def pixel_visibility(task, world, ids, targets=()):
    names = task.cameras.static_camera_name
    camera = task.cameras.static_camera_list[names.index('head_camera')]
    task.scene.update_render(); camera.take_picture()
    segmentation = camera.get_picture('Segmentation')[...,1]
    cfg = task.cameras.get_config()['head_camera']
    K,E = np.asarray(cfg['intrinsic_cv']),np.asarray(cfg['extrinsic_cv'])[:3]
    result = {}
    for oid in ids:
        actor_ids = [e.get_per_scene_id() for e in world._entities(oid)]
        mask = np.isin(segmentation,actor_ids)
        ys,xs = np.nonzero(mask)
        result[oid] = {'visible_pixels':int(len(xs)), 'actor_ids':actor_ids,
                       'visible_bbox_xy':[[int(xs.min()),int(ys.min())],[int(xs.max()),int(ys.max())]] if len(xs) else None,
                        'visibility_source':'native head camera Segmentation actor IDs after settling'}
        regions = []
        for target in targets:
            if target['object_id']!=oid:continue
            pose = world._root_pose(target['host'])
            p = (pose.to_transformation_matrix() @ np.r_[target['local_position'],1.])[:3]
            x = E @ np.r_[p,1.]; uv = (K @ x)[:2]/x[2]
            inside = x[2]>0 and 0<=uv[0]<mask.shape[1] and 0<=uv[1]<mask.shape[0]
            near = int(np.sum((xs-uv[0])**2+(ys-uv[1])**2 <= 6**2)) if inside else 0
            regions.append(dict(target,uv=uv.tolist(),visible=bool(inside and near),nearby_actor_pixels=near,
                                radius_pixels=6))
        result[oid]['operable_regions'] = regions
    return result
