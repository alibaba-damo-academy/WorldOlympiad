import hashlib
import shutil
import numpy as np
import pytest

from autoresearch.articulated_view_initialization import target_corners


def test_fixed_mesh_handle_is_measured_in_moving_door_frame(tmp_path):
    mesh = tmp_path / "handle.obj"
    mesh.write_text("""v -0.05 -0.01 -0.01
v 0.05 -0.01 -0.01
v 0.05 0.01 -0.01
v -0.05 0.01 -0.01
v -0.05 -0.01 0.01
v 0.05 -0.01 0.01
v 0.05 0.01 0.01
v -0.05 0.01 0.01
f 1 2 3 4
f 5 6 7 8
""")
    sdf = tmp_path / "cabinet.sdf"
    sdf.write_text("""<sdf version="1.9"><model name="cabinet">
      <frame name="base_link" attached_to="body"><pose relative_to="__model__">0 0 0 0 0 0</pose></frame>
      <link name="body"><pose>0 0 0 0 0 0</pose></link>
      <link name="door"><pose relative_to="base_link">0 0 0.375 0 0 0</pose>
        <collision><geometry><box><size>0.8 0.03 0.7</size></box></geometry></collision>
      </link>
      <link name="handle"><pose relative_to="door">0.4 0 0 0 0 0</pose>
        <collision><geometry><mesh><uri>handle.obj</uri></mesh></geometry></collision>
      </link></model></sdf>""")
    binding = {
        "child_link": "door", "contact_target_local_xyz": [0.4, 0, 0],
        "contact_frame_evidence": {
            "handle_link": "handle", "input_file_sha256": {
                str(mesh): hashlib.sha256(mesh.read_bytes()).hexdigest()
            },
        },
    }
    corners = target_corners(sdf, binding)
    assert corners.shape == (8, 3)
    assert np.allclose(corners.mean(axis=0), [0.4, 0, 0])
    assert np.allclose(corners.max(axis=0) - corners.min(axis=0),
                       [0.1, 0.02, 0.02])
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    shutil.copy2(sdf, relocated / sdf.name)
    shutil.copy2(mesh, relocated / mesh.name)
    assert np.allclose(target_corners(relocated / sdf.name, binding), corners)
    binding["contact_frame_evidence"]["input_file_sha256"][str(mesh)] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        target_corners(sdf, binding)
