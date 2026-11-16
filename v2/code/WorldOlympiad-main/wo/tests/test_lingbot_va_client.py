import sys
import types
import numpy as np


def test_setup_native_task_reapplies_renderer_profile_after_robotwin_setup(monkeypatch):
    from wo.models.lingbot_va_client import setup_native_task

    events = []
    def set_denoiser(mode):
        events.append(("denoiser", mode))
    render = types.SimpleNamespace(set_ray_tracing_denoiser=set_denoiser)

    def setup_demo(**cfg):
        events.append(("setup_begin", cfg))
        render.set_ray_tracing_denoiser("oidn")
        events.append(("setup_end", cfg))

    task = types.SimpleNamespace(setup_demo=setup_demo)
    monkeypatch.setitem(sys.modules, "sapien", types.SimpleNamespace(render=render))
    monkeypatch.setenv("ROBOTWIN_RT_DENOISER", "none")

    setup_native_task(task, {"seed": 0})

    assert events == [("setup_begin", {"seed": 0}), ("denoiser", "none"),
                      ("setup_end", {"seed": 0})]
    assert render.set_ray_tracing_denoiser is set_denoiser


def test_restore_initial_object_poses_reanchors_dynamic_and_articulated(monkeypatch):
    class Pose:
        def __init__(self, p, q):
            self.p, self.q = list(p), list(q)
    monkeypatch.setitem(sys.modules, "sapien", types.SimpleNamespace(Pose=Pose))
    from wo.models.lingbot_va_client import restore_initial_object_poses

    class Body:
        def __init__(self): self.calls = []
        def set_root_pose(self, pose): self.calls.append(("pose", pose.p, pose.q))
        def set_root_velocity(self, v): self.calls.append(("velocity", v.copy()))
    body = Body()
    task = types.SimpleNamespace(wo_handles={"obj": body})
    plan = {"objects": [{"object_id": "obj", "fixed": False,
                          "pose_rt": {"p": [1, 2, 3], "q_wxyz": [1, 0, 0, 0]}}]}
    restore_initial_object_poses(task, plan)
    assert body.calls[0][1:] == ([1, 2, 3], [1, 0, 0, 0])
    assert np.array_equal(body.calls[1][1], np.zeros(6))



def test_native_capture_records_head_and_both_actual_wrist_views():
    from wo.models.lingbot_va_client import CAPTURE_CAMERAS, add_camera_frames
    recorded={name:[] for name in CAPTURE_CAMERAS}
    videos={name:types.SimpleNamespace(add=frames.append) for name,frames in recorded.items()}
    images={camera:np.full((2,3,3),value,np.uint8) for value,camera in enumerate(CAPTURE_CAMERAS.values())}
    obs={'observation':{camera:{'rgb':rgb} for camera,rgb in images.items()}}
    add_camera_frames(videos,obs);add_camera_frames(videos,obs)
    for name,camera in CAPTURE_CAMERAS.items():
        assert len(recorded[name])==2
        np.testing.assert_array_equal(recorded[name][0],images[camera])


def test_primary_right_alias_preserves_left_and_head_recordings(tmp_path):
    from wo.models.lingbot_va_client import CAPTURE_CAMERAS, write_camera_videos
    (tmp_path/'videos').mkdir()
    videos={};written=[]
    for name in CAPTURE_CAMERAS:
        path=tmp_path/f'videos/{name}.mp4'
        def write(path=path,name=name):
            path.write_bytes(name.encode());written.append(name)
        videos[name]=types.SimpleNamespace(path=path,write=write)
    for arm in ('left','right'):
        (tmp_path/f'initial_{arm}_wrist.png').write_bytes(arm.encode())
    for primary in ('left','right'):
        mapping=write_camera_videos(videos,tmp_path,primary)
        assert (tmp_path/'videos/wrist.mp4').read_bytes()==(primary+'_wrist').encode()
        assert (tmp_path/'initial_wrist.png').read_bytes()==primary.encode()
        assert mapping['wrist_alias']['camera']==primary+'_camera'
        assert mapping['wrist_alias']['source_video']==f'videos/{primary}_wrist.mp4'
        for name in CAPTURE_CAMERAS:
            assert (tmp_path/f'videos/{name}.mp4').read_bytes()==name.encode()
    assert len(written)==6
