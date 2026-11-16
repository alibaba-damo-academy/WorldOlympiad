import pytest
import numpy as np

from autoresearch.articulated_link_frame_rebase import rebase_measured_child_frame
from autoresearch.math3d import compose
from autoresearch.models import BodyIR, Geometry, Inertial, LinkIR, Pose


def test_measured_child_frame_rebase_preserves_visual_and_collision_world_pose():
    visual = Geometry(name="door", kind="box", size=(0.4, 0.03, 0.7),
                      pose=Pose(xyz=(0, 0, -0.375)))
    collision = Geometry(name="door_collision", kind="box", size=(0.4, 0.03, 0.7),
                         pose=Pose(xyz=(0, 0, -0.375)))
    child = LinkIR(name="door", pose=Pose(xyz=(0.2, 0, 0.375)),
                   visuals=[visual], collisions=[collision],
                   inertial=Inertial(mass=1.0, com=(0, 0, -0.375),
                                     inertia=(1, 1, 1, 0, 0, 0)))
    body = BodyIR(name="cabinet", semantic_category="cabinet",
                  pose=Pose(xyz=(1, 2, 0.4)),
                  links=[LinkIR(name="base"), child])
    before = [compose(body.pose, compose(child.pose, item.pose))
              for item in (visual, collision)]
    receipt = rebase_measured_child_frame(
        body, child_link="door", measured_world=Pose(xyz=(1.2, 2, 0.4)))
    after = [compose(body.pose, compose(child.pose, item.pose))
             for item in (visual, collision)]
    assert receipt["passed"] is True
    assert receipt["frame_displacement_m"] == pytest.approx(0.375)
    assert child.pose.xyz == pytest.approx((0.2, 0, 0))
    assert np.allclose([item.xyz for item in after], [item.xyz for item in before])
    assert child.inertial.com == pytest.approx((0, 0, 0))


def test_rebase_rejects_unmeasured_orientation_change():
    child = LinkIR(name="door", pose=Pose(xyz=(0, 0, 0.375)))
    body = BodyIR(name="cabinet", semantic_category="cabinet",
                  links=[LinkIR(name="base"), child])
    with pytest.raises(ValueError, match="unsafe rotation"):
        rebase_measured_child_frame(
            body, child_link="door",
            measured_world=Pose(quat_xyzw=(0, 0, 0.70710678, 0.70710678)))
