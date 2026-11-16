import pytest

from autoresearch.models import BodyIR, Geometry, Inertial, JointIR, LinkIR, Pose
from autoresearch.worldir_fixed_joint_collapse import collapse_fixed_joint_tree


def _link(name, xyz):
    return LinkIR(
        name=name, pose=Pose(xyz=xyz),
        visuals=[Geometry(name=f"{name}_mesh", kind="box", size=(0.2, 0.2, 0.2))],
        collisions=[Geometry(name=f"{name}_solid", kind="box", size=(0.2, 0.2, 0.2))],
        inertial=Inertial(mass=1.0, com=(0, 0, 0),
                          inertia=(0.01, 0.01, 0.01, 0, 0, 0)),
    )


def test_fixed_cabinet_panels_collapse_into_two_link_articulation():
    body = BodyIR(
        name="cabinet", semantic_category="cabinet", static=True,
        links=[_link("body", (0, 0, 0)), _link("panel", (0, 0.2, 0)),
               _link("door", (0.3, 0, 0)), _link("handle", (0.4, 0, 0))],
        joints=[
            JointIR(name="panel_mount", type="fixed", parent="body", child="panel"),
            JointIR(name="door_hinge", type="revolute", parent="body", child="door"),
            JointIR(name="handle_mount", type="fixed", parent="door", child="handle"),
        ],
    )
    receipt = collapse_fixed_joint_tree(body)
    assert receipt["passed"] is True
    assert receipt["source_link_count"] == 4
    assert receipt["collapsed_link_count"] == 2
    assert receipt["maximum_world_geometry_position_error_m"] < 1e-10
    assert receipt["total_mass_kg"] == pytest.approx(4.0)
    assert {link.name for link in body.links} == {"body", "door"}
    assert [joint.name for joint in body.joints] == ["door_hinge"]
    assert all(len(link.visuals) == len(link.collisions) == 2 for link in body.links)
    assert all(link.inertial.mass == pytest.approx(2.0) for link in body.links)
