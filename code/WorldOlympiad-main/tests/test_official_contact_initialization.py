import json
from pathlib import Path
from tempfile import TemporaryDirectory

from continuous_action.official_plan import (
    PROTOCOL, TRAJECTORY_SOURCE, UPSTREAM_COMMIT, sha256, validate_plan,
)


def test_contact_initialization_is_explicit_and_cannot_claim_a_grasp():
    lock=json.loads((Path(__file__).parents[1]/'scripts/official_dream2flow_source_lock.json').read_text())
    with TemporaryDirectory() as directory:
        root=Path(directory);plan=root/'plan.npz';video=root/'video.mp4'
        plan.write_bytes(b'optimizer artifact');video.write_bytes(b'native video')
        meta=dict(evaluation_protocol=PROTOCOL,trajectory_source=TRAJECTORY_SOURCE,
                  upstream_commit=UPSTREAM_COMMIT,video_role='reference',
                  source_video_sha256=sha256(video),plan_sha256=sha256(plan),
                  source_sha256=lock['source_sha256'],optimizer_config=lock['optimizer_numerical_defaults'],
                  initialization_kind='frame_zero_kinematic_contact',
                  initial_pose_source='registered_contact_target',object_to_robot_ik_used=True,
                  action_head_used=False,coordinate_frame='registered_world')
        for name in ['flow','registration','registered_urdf']:
            path=root/name;path.write_bytes(name.encode())
            meta[name+'_path']=str(path);meta[name+'_sha256']=sha256(path)
        plan.with_suffix('.json').write_text(json.dumps(meta))
        assert validate_plan(plan,video,'reference')['initialization_kind']=='frame_zero_kinematic_contact'
        meta['initial_pose_source']='registered_grasp_target'
        plan.with_suffix('.json').write_text(json.dumps(meta))
        try:
            validate_plan(plan,video,'reference')
        except ValueError as exc:
            assert 'contact target' in str(exc)
        else:
            raise AssertionError('a contact plan must not be mislabeled as a grasp')


