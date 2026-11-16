import math
import numpy as np
from wo.render_plan import _fit_showcase_camera, _look_at_gl, VIEWS
from wo.agent.stages import STAGES


def test_fixed_showcase_covers_complete_task_motion_at_readable_scale():
    orientation = _look_at_gl([1, 1, 1], [0, 0, 0])[:3, :3]
    points = np.array([[x, y, z] for x in [-.25, .25] for y in [-.12, .12] for z in [.8, 1.2]])
    camera = _fit_showcase_camera(points, orientation)
    local = (points - camera[:3, 3]) @ camera[:3, :3]
    depth = -local[:, 2]
    assert np.all(depth > 0)
    tan_v = math.tan(math.radians(55) / 2)
    assert np.all(np.abs(local[:, 1]) / depth < tan_v)
    assert np.all(np.abs(local[:, 0]) / depth < tan_v * 1280 / 720)
    assert np.linalg.norm(camera[:3, 3] - points.mean(axis=0)) < 2
    assert tuple(VIEWS) == ("exterior_left", "exterior_right", "top", "wrist")


def test_showcase_stage_writes_separate_presentation_output():
    stage = STAGES["showcase"]
    args = {"run": "dev", "case": "apple"}
    stage.validate(args)
    command = [str(x) for x in stage.command(args, 1)]
    assert command[command.index("--out") + 1] == "showcase"
    assert command[command.index("--views") + 1] == "showcase"
    assert "--trajectory" not in command


def test_articulated_support_geometry_is_not_discarded_as_background(tmp_path):
    import trimesh
    from wo.render_plan import _showcase_points
    mesh = tmp_path / "door.ply"
    trimesh.creation.box(extents=[.4, .1, .7]).export(mesh)
    pose = np.eye(4)
    pose[:3, 3] = [1, 0, 1]
    meshes = [{"node": "cabinet/door", "file": str(mesh), "scale": [1, 1, 1], "link_T_visual": np.eye(4).tolist()}]
    frames = [{"robot": {"end_effector_position": [0, 0, 1]}}]
    points = np.asarray(_showcase_points({"objects": {}}, meshes, [{"cabinet/door": pose.reshape(-1).tolist()}], frames, articulated_hosts=["cabinet"]))
    assert points[:, 0].max() > 1.19 and points[:, 2].max() > 1.34
