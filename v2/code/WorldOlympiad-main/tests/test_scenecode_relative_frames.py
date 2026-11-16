import pytest
import json

from autoresearch.scenecode_importer import import_sdf_body


def test_fixed_handle_link_resolves_relative_to_moving_door(tmp_path):
    sdf = tmp_path / "cabinet.sdf"
    sdf.write_text("""<sdf version="1.9"><model name="cabinet">
      <frame name="base_link" attached_to="body"><pose relative_to="__model__">0 0 0 0 0 0</pose></frame>
      <link name="body"><pose relative_to="base_link">0 0 0 0 0 0</pose></link>
      <link name="door"><pose relative_to="body">0.2 0 0.4 0 0 0</pose>
        <visual><geometry><box><size>0.4 0.03 0.7</size></box></geometry></visual>
        <collision><geometry><box><size>0.4 0.03 0.7</size></box></geometry></collision>
      </link>
      <link name="handle"><pose relative_to="door">0.1 0 0 0 0 0</pose>
        <visual><geometry><box><size>0.1 0.02 0.02</size></box></geometry></visual>
        <collision><geometry><box><size>0.1 0.02 0.02</size></box></geometry></collision>
      </link>
      <joint name="hinge" type="revolute"><parent>body</parent><child>door</child>
        <axis><xyz>0 0 1</xyz></axis></joint>
      <joint name="handle_mount" type="fixed"><parent>door</parent><child>handle</child></joint>
    </model></sdf>""")
    body = import_sdf_body(sdf, semantic_category="cabinet", static=True)
    poses = {link.name: link.pose.xyz for link in body.links}
    assert poses["body"] == pytest.approx((0, 0, 0))
    assert poses["door"] == pytest.approx((0.2, 0, 0.4))
    assert poses["handle"] == pytest.approx((0.3, 0, 0.4))
    joints = {joint.name: joint for joint in body.joints}
    assert joints["handle_mount"].pose.xyz == pytest.approx(poses["handle"])


def test_sdf_relative_frame_cycle_rejected(tmp_path):
    sdf = tmp_path / "cycle.sdf"
    sdf.write_text("""<sdf version="1.9"><model name="cycle">
      <link name="a"><pose relative_to="b">0 0 0 0 0 0</pose></link>
      <link name="b"><pose relative_to="a">0 0 0 0 0 0</pose></link>
    </model></sdf>""")
    with pytest.raises(ValueError, match="cyclic"):
        import_sdf_body(sdf)


def test_single_material_gltf_color_enters_worldir(tmp_path):
    gltf = tmp_path / "door.gltf"
    gltf.write_text(json.dumps({"materials": [{"pbrMetallicRoughness": {
        "baseColorFactor": [0.07, 0.16, 0.28, 1.0]
    }}]}))
    sdf = tmp_path / "door.sdf"
    sdf.write_text("""<sdf version="1.9"><model name="door"><link name="panel">
      <visual><geometry><mesh><uri>door.gltf</uri></mesh></geometry></visual>
      <collision><geometry><box><size>0.4 0.03 0.7</size></box></geometry></collision>
    </link></model></sdf>""")
    body = import_sdf_body(sdf)
    assert body.links[0].visuals[0].rgba == pytest.approx((0.07, 0.16, 0.28, 1.0))
