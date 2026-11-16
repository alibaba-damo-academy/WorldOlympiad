import pytest

from wo.tasks.closed_box_export import box_inertial, visual_to_collision_axes


def test_box_inertial_uses_native_axes_and_mesh_com():
    got = box_inertial(0.5, (0.2, 0.15, 0.08))
    assert got["center"] == pytest.approx((0.1, 0.075, 0.04))
    assert got["ixx"] == pytest.approx(0.0012041666666666665)
    assert got["iyy"] == pytest.approx(0.0019333333333333333)
    assert got["izz"] == pytest.approx(0.002604166666666667)


def test_box_inertial_accepts_translated_mesh_without_changing_shape_inertia():
    got = box_inertial(0.5, (0.2, 0.15, 0.08), center=(0.0, 0.0, 0.04))
    assert got["center"] == (0.0, 0.0, 0.04)
    assert got["iyy"] == pytest.approx(0.0019333333333333333)


def test_collision_axis_conversion_preserves_signed_translation():
    assert visual_to_collision_axes([(0.1, 0.075, 0.04)]) == [(0.1, -0.04, 0.075)]


def test_prepare_bodies_preserves_omitted_pose_fresh_and_cached(tmp_path):
    """Exercise the production preparation boundary, not just XML parsing.

    The tetrahedron is deliberately asymmetric and translated, so an AABB-derived
    COM would be observable.  Both preparation branches must leave the authored
    omitted pose and source bytes untouched.
    """
    import xml.etree.ElementTree as ET
    from wo.scene import SceneInputs, SceneObject
    from wo.sim.world import prepare_bodies

    obj = tmp_path / "asymmetric.obj"
    obj.write_text(
        "v 0.10 0.02 0.03\n"
        "v 0.34 0.02 0.03\n"
        "v 0.11 0.21 0.03\n"
        "v 0.12 0.04 0.29\n"
        "f 1 3 2\n f 1 2 4\n f 2 3 4\n f 3 1 4\n"
    )
    sdf = tmp_path / "asymmetric.sdf"
    sdf.write_text(
        '<sdf version="1.6"><model name="x"><link name="body">'
        '<inertial><mass>0.5</mass><inertia><ixx>1</ixx><iyy>1</iyy><izz>1</izz>'
        '<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>'
        f'<collision><geometry><mesh><uri>{obj}</uri></mesh></geometry></collision>'
        '</link></model></sdf>'
    )
    source_before = sdf.read_bytes()
    assert ET.parse(sdf).find('.//inertial/pose') is None
    def scene(sid):
        item = SceneObject(sid, "object", sdf, (0, 0, 0), (0, 0, 0, 1))
        support = SceneObject("support", "support", sdf, (0, 0, 0), (0, 0, 0, 1), fixed=True)
        return SceneInputs(sid, support, {sid: item}, sdf, (0, 0, 0), 0.0)

    fresh = prepare_bodies(scene("tetra"), tmp_path / "prepared")
    cached = prepare_bodies(scene("tetra"), tmp_path / "prepared")
    assert fresh["tetra"].physics_sdf == cached["tetra"].physics_sdf
    # Assert the actual prepared artifacts too: the historical defect mutated the
    # copied SDF while leaving the authored source unchanged.
    for prepared in (fresh["tetra"].physics_sdf, cached["tetra"].physics_sdf):
        assert ET.parse(prepared).find('.//inertial/pose') is None
    assert sdf.read_bytes() == source_before
    assert ET.parse(sdf).find('.//inertial/pose') is None
