from pathlib import Path

import pytest

from wo.render_plan import _prepared_urdf


def test_current_binding_ignores_old_generations_and_scales(tmp_path):
    directory = tmp_path / 'physics_sdf' / 'cabinet_s1.000'
    directory.mkdir(parents=True)
    (directory / 'old.urdf').write_text('old')
    current = directory / 'new.urdf'
    current.write_text('new')
    (directory / '.path').write_text(str(current))
    (directory / '.ok').write_text('/assets/new.sdf')
    assert _prepared_urdf(tmp_path, 'cabinet', {'sdf_path': '/assets/new.sdf', 'scale': 1.0}) == current


def test_stale_binding_uses_exact_source_not_glob(tmp_path):
    directory = tmp_path / 'physics_sdf' / 'cabinet_s0.500'
    directory.mkdir(parents=True)
    old = directory / 'old.urdf'
    old.write_text('old')
    (directory / '.path').write_text(str(old))
    (directory / '.ok').write_text('/assets/old.sdf')
    obj = {'sdf_path': '/assets/new.sdf', 'scale': 0.5}
    with pytest.raises(FileNotFoundError):
        _prepared_urdf(tmp_path, 'cabinet', obj)
    current = directory / 'new.urdf'
    current.write_text('new')
    assert _prepared_urdf(tmp_path, 'cabinet', obj) == current


def test_recorded_mount_is_replayed_without_changing_task_poses(tmp_path):
    import json
    import numpy as np
    from wo.render_plan import build_plan

    sdf = tmp_path / 'support.sdf'
    sdf.write_text('<sdf version="1.7"><model name="table"><link name="base"/></model></sdf>')
    urdf = tmp_path / 'robot.urdf'
    urdf.write_text('<robot name="robot"><link name="base"/></robot>')
    support = {'object_id': 'table', 'sdf_path': str(sdf),
               'position': [0, 0, 0], 'orientation_xyzw': [0, 0, 0, 1]}
    scene = {'support': support, 'objects': {}, 'robot_urdf': str(urdf)}
    robot = {'base_position': [-.444, -.171, .674],
             'base_orientation_xyzw': [0, 0, 0, 1],
             'joint_names': [], 'joint_positions': []}
    trajectory = {'frames': [{'robot': robot}, {'robot': robot}]}
    (tmp_path / 'scene_inputs.json').write_text(json.dumps(scene))
    (tmp_path / 'certificate.json').write_text('{}')
    (tmp_path / 'trajectory.json').write_text(json.dumps(trajectory))
    baseline = build_plan(tmp_path, views=('exterior_left',))
    assert baseline['primitives'] == []
    trajectory['robot_mount'] = {'height_m': .672, 'size_xy_m': [.3, .3]}
    (tmp_path / 'trajectory.json').write_text(json.dumps(trajectory))
    replay = build_plan(tmp_path, views=('exterior_left',))
    assert replay['primitives'] == [{'node': 'robot_mount/base', 'kind': 'box',
                                    'size': [.3, .3, .672], 'rgba': [.28, .3, .32, 1]}]
    assert replay['meshes'] == baseline['meshes']
    assert replay['cameras'] == baseline['cameras']
    for frame, original in zip(replay['frames'], baseline['frames']):
        mount = np.array(frame.pop('robot_mount/base')).reshape(4, 4)
        np.testing.assert_allclose(mount[:3, 3], [-.444, -.171, .336])
        assert frame == original


@pytest.mark.parametrize('suffix', ['.glb', '.obj'])
def test_showcase_bounds_match_blender_import_axes(tmp_path, suffix):
    import numpy as np
    import trimesh
    from wo.render_plan import _showcase_points

    mesh = trimesh.creation.box(extents=[.1, .2, .3])
    mesh.apply_translation([.4, .5, .6])
    path = tmp_path / ('box' + suffix)
    mesh.export(path)
    entry = {'file': str(path), 'node': 'bottle/base', 'scale': [1, 1, 1],
             'link_T_visual': np.eye(4).tolist()}
    poses = [{'bottle/base': np.eye(4).reshape(-1).tolist()}]
    points = np.array(_showcase_points({'objects': {'bottle': {}}}, [entry], poses, []))
    expected = [[.35, -.75, .4], [.45, -.45, .6]] if suffix == '.glb' else \
        [[.35, .4, .45], [.45, .6, .75]]
    np.testing.assert_allclose([points.min(axis=0), points.max(axis=0)], expected, atol=1e-7)


def test_renderer_uses_bundled_ffmpeg_when_host_has_none(monkeypatch, tmp_path):
    import imageio_ffmpeg
    from wo import render as renderer

    executable = tmp_path / "ffmpeg"
    executable.write_bytes(b"ffmpeg")
    monkeypatch.delenv("WO_FFMPEG", raising=False)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(executable))
    monkeypatch.setattr(renderer, "configure_egl_vendor", lambda env: None)
    assert renderer._render_environment()["WO_FFMPEG"] == str(executable)
