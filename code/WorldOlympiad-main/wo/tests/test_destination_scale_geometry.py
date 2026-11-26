"""Destination metrology must exclude native broadphase collision margins."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import trimesh

from wo.expert.embodiment import destination_scale
from wo.expert.run import _require_authored_scale
from wo.scene import SceneInputs, SceneObject, _quat_mat
from wo.sim.world import World, prepare_bodies
from wo.spec import TaskSpec, TaskObject, Step, Predicate
from wo.tasks.geometry_contract import collision_bounds


def mesh_scene(tmp_path, height=.1956):
    mesh = trimesh.creation.box([.24, .08, height / 2])
    mesh.export(tmp_path / "body.obj")
    target = tmp_path / "target.sdf"
    target.write_text(f"""<sdf version="1.7"><model name="target"><link name="base">
      <inertial><pose>.03 -.02 .015 0 0 0</pose><mass>.3</mass>
        <inertia><ixx>.002</ixx><iyy>.002</iyy><izz>.002</izz></inertia></inertial>
      <visual name="v"><geometry><mesh><uri>body.obj</uri><scale>.5 2 2</scale></mesh></geometry></visual>
      <collision name="c"><pose>.03 .02 .01 0 0 .4</pose>
        <geometry><mesh><uri>body.obj</uri><scale>.5 2 2</scale></mesh></geometry></collision>
      </link></model></sdf>""")
    support = tmp_path / "table.sdf"
    support.write_text("""<sdf version="1.7"><model name="table"><link name="base">
      <inertial><mass>10</mass><inertia><ixx>1</ixx><iyy>1</iyy><izz>1</izz></inertia></inertial>
      <collision name="c"><geometry><box><size>1 .8 .04</size></box></geometry></collision>
      </link></model></sdf>""")
    angle = .6
    obj = SceneObject("target", "destination", target, (.2, -.1, .8),
                      (0., 0., np.sin(angle/2), np.cos(angle/2)), scale=.577)
    scene = SceneInputs("margin", SceneObject("table", "support", support,
                        (0, 0, -.05), (0, 0, 0, 1), fixed=True), {"target": obj},
                        Path("unused_robot.urdf"), (0, 0, 0), 0.)
    return scene, mesh


def test_native_margin_does_not_trigger_second_scaling(tmp_path):
    scene, mesh = mesh_scene(tmp_path)
    bounds = collision_bounds(scene.objects["target"].sdf_path,
                              rotation=_quat_mat(scene.objects["target"].orientation_xyzw))
    size = bounds[1] - bounds[0]
    world = World(scene, prepare_bodies(scene, tmp_path / "physics"),
                  mode="objects", include=["target"])
    try:
        lo, hi = world.aabb("target")
        assert (hi-lo)[2] > .20 > size[2]
        assert (hi-lo)[2]-size[2] > .007
        assert world.object_pose("target")[0] == pytest.approx(scene.objects["target"].position)
    finally:
        world.close()
    assert size[2] == pytest.approx(.1956)
    # The SDF already contains its scale; the .577 lineage is not applied again.
    assert destination_scale(scene, "target", tmp_path) == 1.0


def test_rotation_applies_to_vertices_after_sdf_scale_and_pose(tmp_path):
    scene, mesh = mesh_scene(tmp_path)
    local_rotation = trimesh.transformations.euler_matrix(0, 0, .4)[:3, :3]
    object_rotation = _quat_mat(scene.objects["target"].orientation_xyzw)
    expected = (mesh.vertices * [.5, 2, 2]) @ local_rotation.T + [.03, .02, .01]
    legacy = collision_bounds(scene.objects["target"].sdf_path)
    np.testing.assert_allclose(legacy, [expected.min(0), expected.max(0)], atol=1e-10)
    expected = expected @ object_rotation.T
    rotated = collision_bounds(scene.objects["target"].sdf_path, rotation=object_rotation)
    np.testing.assert_allclose(rotated, [expected.min(0), expected.max(0)], atol=1e-10)


def test_real_oversize_requires_an_explicit_compatible_source(tmp_path):
    scene, _ = mesh_scene(tmp_path, height=.204)
    factor = destination_scale(scene, "target", tmp_path)
    assert factor == pytest.approx(.20/.204)
    spec = TaskSpec("locked", "pick_place", "Place the object in the target.",
        (TaskObject("table", "table", "support"),
         TaskObject("subject", "object", "manipuland"),
         TaskObject("target", "box", "destination",
                    geometry_constraints={"overall_dimensions_m": [.12, .16, .204]})),
        (Step("transport", "subject", Predicate("inside", "subject", "target"), target="target"),))
    before = scene.objects["target"].sdf_path.read_bytes()
    with pytest.raises(RuntimeError, match="cannot resize"):
        _require_authored_scale(spec, ["subject", "target"], round(factor, 3))
    assert scene.objects["target"].sdf_path.read_bytes() == before
    _require_authored_scale(spec, ["target"], 1.0)
    # A locked support outside the resized set must not disable legacy fitting.
    legacy = replace(spec, objects=(replace(spec.objects[0], geometry_constraints={"overall_dimensions_m": [1, .8, .04]}),
                                    spec.objects[1], replace(spec.objects[2], geometry_constraints={})))
    _require_authored_scale(legacy, ["subject", "target"], round(factor, 3))
    locked_subject = replace(legacy, objects=(legacy.objects[0],
        replace(legacy.objects[1], geometry_constraints={"overall_dimensions_m": [.1, .1, .1]}), legacy.objects[2]))
    with pytest.raises(RuntimeError, match="subject"):
        _require_authored_scale(locked_subject, ["subject"], .85)


@pytest.mark.parametrize("rotation", [np.ones((2, 2)), np.diag([2., 1., 1.]), np.diag([-1., 1., 1.])])
def test_invalid_rotation_fails_closed(tmp_path, rotation):
    scene, _ = mesh_scene(tmp_path)
    with pytest.raises(ValueError, match="proper rotation"):
        collision_bounds(scene.objects["target"].sdf_path, rotation=rotation)
