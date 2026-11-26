import hashlib
import json

import numpy as np
import pytest

from autoresearch.robotics_articulated_camera import fit_articulated_camera_rig


@pytest.mark.parametrize("handle_axis", [[1., 0., 0.], [0., 0., 1.], [0.003, -0.007, 0.99997]])
def test_articulated_camera_preserves_asset_and_fits_reference_motion(tmp_path, monkeypatch, handle_axis):
    from scripts import render_scenecode_asset_identity as identity

    monkeypatch.setattr(identity, "sdf_collision_bounds", lambda _: (
        np.asarray([-.25, -.20, 0.]), np.asarray([.25, .20, .8])))
    host = tmp_path / "cabinet.sdf"
    host.write_text("immutable host")
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps({
        "passed": True, "host_object_id": "cabinet",
        "host_sdf_sha256": hashlib.sha256(host.read_bytes()).hexdigest(),
        "child_link": "door", "contact_target_local_xyz": [.1, 0., 0.],
        "contact_normal_local_xyz": [0., -1., 0.],
        "handle_axis_local_xyz": handle_axis,
    }))
    robot = {"base_position": [-.18, .12, .003],
             "base_orientation_xyzw": [0., 0., .09, .9959],
             "end_effector_position": [.2, .6, .6]}
    frames = [{"robot": robot, "objects": {"cabinet": {
        "link_states": {
            "body": {"position": [0., .55, .4]},
            "door": {"position": [x, y, .4],
                     "orientation_xyzw": [0., 0., 0., 1.]},
        }}}} for x, y in [(0., .55), (-.5, .1), (-.6, -.2)]]
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(json.dumps({"frames": frames}))
    output = tmp_path / "rig.json"
    required = {name: {"size_wh": [320, 180], "minimum_linear_fraction": .02}
                for name in ("exterior_left", "exterior_right")}
    assert fit_articulated_camera_rig(trajectory=trajectory, host_sdf=host,
                                      binding=binding, required_views=required,
                                      output=output) == output
    rig = json.loads(output.read_text())
    assert rig["selection"] == "native_contact_side_and_full_geometry_top_v5"
    assert set(rig["cameras"]) == {*required, "top"}
    assert all(row["backward_translation_m"] > 0
               for row in rig["camera_measurements"].values())
    assert all(row["full_motion"]["fully_in_frame"]
               for row in rig["camera_measurements"].values())
    assert fit_articulated_camera_rig(trajectory=trajectory, host_sdf=host,
                                      binding=binding, required_views=required,
                                      output=output) == output
    host.write_text("changed")
    with pytest.raises(ValueError, match="unbound host"):
        fit_articulated_camera_rig(trajectory=trajectory, host_sdf=host,
                                   binding=binding, required_views=required,
                                   output=output)
