from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

from scenecode_pybullet_bridge import prepare_physics_sdf


SDF = """<?xml version="1.0"?>
<sdf xmlns:drake="drake.mit.edu" version="1.7">
  <model name="asset">
    <link name="base_link">
      <inertial>
        <mass>0.25</mass>
        <inertia><ixx>1</ixx><iyy>1</iyy><izz>1</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <visual name="visual"><geometry><mesh><uri>asset.gltf</uri></mesh></geometry></visual>
      <collision name="collision">
        <surface><friction><ode><mu>0.4</mu><mu2>0.3</mu2></ode></friction></surface>
        <geometry><mesh><uri>collision.obj</uri><drake:declare_convex/></mesh></geometry>
      </collision>
    </link>
  </model>
</sdf>
"""

VISUAL_ONLY_SDF = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="region">
    <link name="base_link">
      <visual name="visual"><geometry><mesh><uri>region.gltf</uri></mesh></geometry></visual>
    </link>
  </model>
</sdf>
"""

VISUAL_ONLY_GLTF = """{
  "asset": {"version": "2.0"},
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
  "accessors": [{
    "componentType": 5126,
    "count": 24,
    "min": [-0.16, 0.0, -0.11],
    "max": [0.16, 0.002, 0.11],
    "type": "VEC3"
  }]
}
"""

INCOMPLETE_INERTIAL_SDF = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="asset">
    <link name="base_link">
      <inertial><mass>12</mass><pose>0 0 0 0 0 0</pose></inertial>
      <visual name="visual"><geometry><mesh><uri>asset.gltf</uri></mesh></geometry></visual>
      <collision name="collision"><geometry><mesh><uri>box.obj</uri></mesh></geometry></collision>
    </link>
  </model>
</sdf>
"""

BOX_OBJ = """v -1 -2 -3
v -1 -2 3
v -1 2 -3
v -1 2 3
v 1 -2 -3
v 1 -2 3
v 1 2 -3
v 1 2 3
f 1 2 4
"""


class PhysicsSdfTest(unittest.TestCase):
    def test_preserves_physics_and_removes_only_incompatible_visual_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.sdf").write_text(SDF, encoding="utf-8")
            (root / "collision.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
            )
            destination = root / "out" / "physics.sdf"
            artifact = prepare_physics_sdf(root / "source.sdf", destination)

            tree = ET.parse(destination)
            nodes = list(tree.getroot().iter())
            local_names = [node.tag.rsplit("}", 1)[-1] for node in nodes]
            self.assertNotIn("visual", local_names)
            self.assertNotIn("declare_convex", local_names)
            self.assertEqual(tree.findtext(".//mass"), "0.25")
            self.assertEqual(tree.findtext(".//mu"), "0.4")
            self.assertEqual(tree.findtext(".//mu2"), "0.3")
            self.assertEqual(
                tree.findtext(".//collision/geometry/mesh/uri"),
                str((root / "collision.obj").resolve()),
            )
            self.assertEqual(artifact.removed_visual_count, 1)
            self.assertEqual(artifact.collision_mesh_count, 1)
            self.assertEqual(artifact.collision_geometry_count, 1)
            self.assertFalse(artifact.visual_bounds_fallback)
            self.assertNotEqual(artifact.source_sha256, artifact.physics_sha256)

    def test_visual_only_region_fails_closed_without_explicit_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sdf"
            source.write_text(VISUAL_ONLY_SDF, encoding="utf-8")
            (root / "region.gltf").write_text(VISUAL_ONLY_GLTF, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no collision geometry"):
                prepare_physics_sdf(source, root / "physics.sdf")

    def test_visual_only_region_gets_noncolliding_bounds_box_when_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sdf"
            source.write_text(VISUAL_ONLY_SDF, encoding="utf-8")
            gltf = root / "region.gltf"
            gltf.write_text(VISUAL_ONLY_GLTF, encoding="utf-8")
            destination = root / "physics.sdf"
            artifact = prepare_physics_sdf(
                source, destination, allow_visual_bounds_fallback=True
            )

            tree = ET.parse(destination)
            self.assertEqual(
                tree.findtext(".//collision/geometry/box/size"), "0.32 0.22 0.002"
            )
            self.assertEqual(
                tree.findtext(".//collision/pose"), "0 -0 0.001 0 0 0"
            )
            self.assertEqual(artifact.removed_visual_count, 1)
            self.assertEqual(artifact.collision_mesh_count, 0)
            self.assertEqual(artifact.collision_geometry_count, 1)
            self.assertTrue(artifact.visual_bounds_fallback)
            self.assertEqual(artifact.visual_bounds_source, str(gltf.resolve()))
            self.assertEqual(
                artifact.visual_bounds_box_size_xyz_m, (0.32, 0.22, 0.002)
            )

    def test_never_overwrites_official_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.sdf"
            source.write_text(SDF, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                prepare_physics_sdf(source, source)

    def test_repairs_missing_inertia_from_collision_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sdf"
            source.write_text(INCOMPLETE_INERTIAL_SDF, encoding="utf-8")
            (root / "box.obj").write_text(BOX_OBJ, encoding="utf-8")
            artifact = prepare_physics_sdf(source, root / "physics.sdf")

            tree = ET.parse(root / "physics.sdf")
            self.assertEqual(float(tree.findtext(".//inertia/ixx")), 52.0)
            self.assertEqual(float(tree.findtext(".//inertia/iyy")), 40.0)
            self.assertEqual(float(tree.findtext(".//inertia/izz")), 20.0)
            self.assertEqual(float(tree.findtext(".//inertia/ixy")), 0.0)
            self.assertEqual(artifact.inertial_repair_count, 1)
            self.assertEqual(
                artifact.inertial_repairs[0]["policy"],
                "solid_enclosing_collision_aabb",
            )

    def test_preserves_complete_positive_inertia(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sdf"
            source.write_text(SDF, encoding="utf-8")
            (root / "collision.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
            )
            artifact = prepare_physics_sdf(source, root / "physics.sdf")
            self.assertEqual(artifact.inertial_repair_count, 0)
            self.assertEqual(ET.parse(root / "physics.sdf").findtext(".//inertia/ixx"), "1")


ANCHOR_ROOT_SDF = """<?xml version="1.0"?>
<sdf xmlns:drake="drake.mit.edu" version="1.7">
  <model name="kettle">
    <link name="base_link">
      <inertial>
        <mass>0.0</mass>
        <pose>0 0 0 0 0 0</pose>
        <inertia><ixx>1e-10</ixx><iyy>1e-10</iyy><izz>1e-10</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
    </link>
    <link name="body">
      <inertial>
        <mass>0.5</mass>
        <pose>0 0 0.1 0 0 0</pose>
        <inertia><ixx>0.002</ixx><iyy>0.002</iyy><izz>0.002</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <collision name="body_collision">
        <pose>0 0 0.1 0 0 0</pose>
        <geometry><mesh><uri>body.obj</uri></mesh></geometry>
      </collision>
    </link>
    <link name="lid">
      <pose relative_to="base_link">0 0 0.2 0 0 0</pose>
      <inertial>
        <mass>0.1</mass>
        <pose>0 0 0 0 0 0</pose>
        <inertia><ixx>0.0001</ixx><iyy>0.0001</iyy><izz>0.0001</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <collision name="lid_collision">
        <geometry><mesh><uri>lid.obj</uri></mesh></geometry>
      </collision>
    </link>
    <joint name="body_fixed_joint" type="fixed">
      <parent>base_link</parent>
      <child>body</child>
    </joint>
    <joint name="lid_hinge_joint" type="revolute">
      <parent>base_link</parent>
      <child>lid</child>
      <axis><xyz>1 0 0</xyz><limit><lower>-1.0</lower><upper>0</upper></limit></axis>
    </joint>
  </model>
</sdf>
"""


class AnchorRootCollapseTest(unittest.TestCase):
    """The zero-mass SceneCode root anchor must become a dynamic rigid body."""

    def test_fixed_subtree_folds_into_root_and_keeps_articulated_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sdf"
            source.write_text(ANCHOR_ROOT_SDF, encoding="utf-8")
            for name in ("body.obj", "lid.obj"):
                (root / name).write_text(
                    "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nf 1 2 3\n", encoding="utf-8"
                )

            artifact = prepare_physics_sdf(source, root / "physics.sdf")
            self.assertEqual(artifact.merged_root_link, "base_link")
            self.assertEqual(artifact.merged_fixed_links, ("body",))

            tree = ET.parse(root / "physics.sdf")
            model = tree.getroot().find("model")
            link_names = [link.get("name") for link in model.findall("link")]
            self.assertEqual(sorted(link_names), ["base_link", "lid"])
            root_link = model.find("link[@name='base_link']")
            self.assertEqual(len(root_link.findall("collision")), 1)
            self.assertAlmostEqual(
                float(root_link.findtext("inertial/mass")), 0.5, places=9
            )
            self.assertAlmostEqual(
                float(root_link.findtext("inertial/pose").split()[2]), 0.1, places=9
            )
            # The articulated child survives as an independent link.
            joints = {joint.get("name"): joint for joint in model.findall("joint")}
            self.assertEqual(list(joints), ["lid_hinge_joint"])
            self.assertEqual(joints["lid_hinge_joint"].findtext("parent"), "base_link")

    def test_geometry_in_the_root_is_left_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sdf"
            source.write_text(SDF, encoding="utf-8")
            (root / "collision.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
            )
            artifact = prepare_physics_sdf(source, root / "physics.sdf")
            self.assertEqual(artifact.merged_fixed_links, ())
            self.assertIsNone(artifact.merged_root_link)


if __name__ == "__main__":
    unittest.main()
