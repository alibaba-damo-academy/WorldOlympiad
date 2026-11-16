"""Tests for SDF rescaling functionality.

Tests cover:
- rescale_sdf() function (adding/multiplying scale elements)
- Inertia tensor scaling (scales as s^2)
- Link and joint pose translation scaling
- Model pose NOT scaled (world position preserved)
- Geometry round-trip (anchoring preserved after rescale)
- Articulated object scaling (multi-link models)
"""

import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import trimesh

from scenecode.agent_utils.mesh_physics_analyzer import MeshPhysicsAnalysis
from scenecode.agent_utils.sdf_generator import generate_drake_sdf, rescale_sdf


class TestRescaleSdfBasic(unittest.TestCase):
    """Test basic rescale_sdf functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_test_sdf(self, name: str = "test_object") -> Path:
        """Create a simple test SDF file."""
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        visual_path = self.temp_path / f"{name}.gltf"
        mesh.export(visual_path)

        physics = MeshPhysicsAnalysis(
            up_axis="+Z",
            front_axis="+Y",
            material="wood",
            mass_kg=10.0,
            mass_range_kg=(8.0, 12.0),
        )

        output_path = self.temp_path / f"{name}.sdf"
        generate_drake_sdf(
            visual_mesh_path=visual_path,
            collision_pieces=[mesh],
            physics_analysis=physics,
            output_path=output_path,
            asset_name=name,
        )
        return output_path

    def test_rescale_adds_scale_to_visual_mesh(self):
        """Test that rescale_sdf adds scale element to visual mesh geometry."""
        sdf_path = self._create_test_sdf("scale_visual")

        # Apply 1.5x scale.
        rescale_sdf(sdf_path, scale_factor=1.5)

        # Parse SDF and check for scale element.
        tree = ET.parse(sdf_path)
        visual_mesh = tree.find(".//visual/geometry/mesh")
        scale_elem = visual_mesh.find("scale")

        self.assertIsNotNone(scale_elem)
        scale_values = [float(v) for v in scale_elem.text.split()]
        self.assertEqual(len(scale_values), 3)
        for val in scale_values:
            self.assertAlmostEqual(val, 1.5, places=5)

    def test_rescale_adds_scale_to_collision_mesh(self):
        """Test that rescale_sdf adds scale element to collision mesh geometry."""
        sdf_path = self._create_test_sdf("scale_collision")

        rescale_sdf(sdf_path, scale_factor=0.8)

        tree = ET.parse(sdf_path)
        collision_mesh = tree.find(".//collision/geometry/mesh")
        scale_elem = collision_mesh.find("scale")

        self.assertIsNotNone(scale_elem)
        scale_values = [float(v) for v in scale_elem.text.split()]
        for val in scale_values:
            self.assertAlmostEqual(val, 0.8, places=5)

    def test_rescale_multiplies_existing_scale(self):
        """Test that rescale_sdf multiplies existing scale elements."""
        sdf_path = self._create_test_sdf("mult_scale")

        # Apply first scale.
        rescale_sdf(sdf_path, scale_factor=0.5)

        # Apply second scale.
        rescale_sdf(sdf_path, scale_factor=2.0)

        # Final scale should be 0.5 * 2.0 = 1.0.
        tree = ET.parse(sdf_path)
        visual_mesh = tree.find(".//visual/geometry/mesh")
        scale_elem = visual_mesh.find("scale")
        scale_values = [float(v) for v in scale_elem.text.split()]

        for val in scale_values:
            self.assertAlmostEqual(val, 1.0, places=5)

    def test_rescale_scales_inertia_tensor_squared(self):
        """Test that inertia tensor scales as s^2."""
        sdf_path = self._create_test_sdf("scale_inertia")

        # Get original inertia values.
        tree = ET.parse(sdf_path)
        inertia = tree.find(".//inertia")
        original_ixx = float(inertia.find("ixx").text)
        original_iyy = float(inertia.find("iyy").text)
        original_izz = float(inertia.find("izz").text)

        # Apply 2x scale.
        scale_factor = 2.0
        rescale_sdf(sdf_path, scale_factor=scale_factor)

        # Check scaled inertia values (should be 4x = 2^2).
        tree = ET.parse(sdf_path)
        inertia = tree.find(".//inertia")
        new_ixx = float(inertia.find("ixx").text)
        new_iyy = float(inertia.find("iyy").text)
        new_izz = float(inertia.find("izz").text)

        expected_factor = scale_factor**2  # Inertia scales as s^2.
        self.assertAlmostEqual(new_ixx, original_ixx * expected_factor, places=5)
        self.assertAlmostEqual(new_iyy, original_iyy * expected_factor, places=5)
        self.assertAlmostEqual(new_izz, original_izz * expected_factor, places=5)

    def test_rescale_scales_inertial_pose_translation(self):
        """Test that inertial (CoM) pose translation is scaled."""
        # Create mesh with offset center of mass.
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        mesh.apply_translation([0.5, 0, 0])  # Offset CoM in X.
        visual_path = self.temp_path / "offset_com.gltf"
        mesh.export(visual_path)

        physics = MeshPhysicsAnalysis(
            up_axis="+Z",
            front_axis="+Y",
            material="wood",
            mass_kg=10.0,
            mass_range_kg=(8.0, 12.0),
        )

        sdf_path = self.temp_path / "offset_com.sdf"
        generate_drake_sdf(
            visual_mesh_path=visual_path,
            collision_pieces=[mesh],
            physics_analysis=physics,
            output_path=sdf_path,
        )

        # Get original CoM pose.
        tree = ET.parse(sdf_path)
        com_pose = tree.find(".//inertial/pose")
        original_values = [float(v) for v in com_pose.text.split()]
        original_x = original_values[0]

        # Apply 2x scale.
        rescale_sdf(sdf_path, scale_factor=2.0)

        # Check scaled CoM position.
        tree = ET.parse(sdf_path)
        com_pose = tree.find(".//inertial/pose")
        new_values = [float(v) for v in com_pose.text.split()]
        new_x = new_values[0]

        self.assertAlmostEqual(new_x, original_x * 2.0, places=5)

    def test_rescale_does_not_scale_model_pose(self):
        """Test that model pose (world position) is NOT scaled."""
        sdf_path = self._create_test_sdf("model_pose")

        # Add a model pose to the SDF.
        tree = ET.parse(sdf_path)
        model = tree.find(".//model")
        pose_elem = ET.SubElement(model, "pose")
        pose_elem.text = "1.0 2.0 3.0 0 0 0"  # xyz rpy.
        tree.write(sdf_path)

        # Apply 2x scale.
        rescale_sdf(sdf_path, scale_factor=2.0)

        # Model pose should NOT be scaled.
        tree = ET.parse(sdf_path)
        model_pose = tree.find(".//model/pose")
        pose_values = [float(v) for v in model_pose.text.split()]

        self.assertAlmostEqual(pose_values[0], 1.0, places=5)
        self.assertAlmostEqual(pose_values[1], 2.0, places=5)
        self.assertAlmostEqual(pose_values[2], 3.0, places=5)

    def test_rescale_mass_not_scaled(self):
        """Test that mass is NOT scaled (constant mass assumption)."""
        sdf_path = self._create_test_sdf("mass_unchanged")

        tree = ET.parse(sdf_path)
        original_mass = float(tree.find(".//mass").text)

        rescale_sdf(sdf_path, scale_factor=2.0)

        tree = ET.parse(sdf_path)
        new_mass = float(tree.find(".//mass").text)

        self.assertAlmostEqual(new_mass, original_mass, places=5)


class TestRescaleSdfGeometryRoundTrip(unittest.TestCase):
    """Test that rescaling preserves anchor relationships (geometry round-trip).

    Key insight: Objects are canonicalized with mesh origin at anchor point.
    When we scale, the object should stay anchored at the same point.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_floor_object_stays_on_floor(self):
        """Verify floor objects stay anchored to floor after scaling.

        Floor objects have mesh origin at bottom center (z=0).
        After scaling, the bottom should still be at z=0.
        """
        # Create a box mesh with bottom at z=0.
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        # Move so bottom is at z=0.
        mesh.apply_translation([0, 0, 0.5])

        visual_path = self.temp_path / "floor_object.gltf"
        mesh.export(visual_path)

        physics = MeshPhysicsAnalysis(
            up_axis="+Z",
            front_axis="+Y",
            material="wood",
            mass_kg=10.0,
            mass_range_kg=(8.0, 12.0),
        )

        sdf_path = self.temp_path / "floor_object.sdf"
        generate_drake_sdf(
            visual_mesh_path=visual_path,
            collision_pieces=[mesh],
            physics_analysis=physics,
            output_path=sdf_path,
        )

        # Apply 2x scale.
        rescale_sdf(sdf_path, scale_factor=2.0)

        # The model pose should NOT be scaled, so the floor relationship is preserved.
        tree = ET.parse(sdf_path)
        model_pose = tree.find(".//model/pose")
        if model_pose is not None:
            pose_values = [float(v) for v in model_pose.text.split()]
            # z position (world) should be unchanged (0 for floor object).
            self.assertAlmostEqual(pose_values[2], 0.0, places=5)

    def test_visual_collision_scale_consistent(self):
        """Test that visual and collision meshes get same scale."""
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        visual_path = self.temp_path / "consistent.gltf"
        mesh.export(visual_path)

        physics = MeshPhysicsAnalysis(
            up_axis="+Z",
            front_axis="+Y",
            material="wood",
            mass_kg=10.0,
            mass_range_kg=(8.0, 12.0),
        )

        sdf_path = self.temp_path / "consistent.sdf"
        generate_drake_sdf(
            visual_mesh_path=visual_path,
            collision_pieces=[mesh],
            physics_analysis=physics,
            output_path=sdf_path,
        )

        rescale_sdf(sdf_path, scale_factor=1.5)

        tree = ET.parse(sdf_path)
        visual_scale = tree.find(".//visual/geometry/mesh/scale")
        collision_scale = tree.find(".//collision/geometry/mesh/scale")

        self.assertIsNotNone(visual_scale)
        self.assertIsNotNone(collision_scale)
        self.assertEqual(visual_scale.text, collision_scale.text)


class TestRescaleSdfArticulated(unittest.TestCase):
    """Test rescale_sdf with articulated (multi-link) models."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_articulated_sdf(self, name: str = "cabinet") -> Path:
        """Create a simple articulated SDF with two links and a joint."""
        sdf_content = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <pose>0 0 0 0 0 0</pose>
    <link name="base">
      <pose>0 0 0.5 0 0 0</pose>
      <inertial>
        <pose>0 0 0 0 0 0</pose>
        <mass>10.0</mass>
        <inertia>
          <ixx>1.0</ixx>
          <iyy>1.0</iyy>
          <izz>1.0</izz>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyz>0.0</iyz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>base.gltf</uri>
            <scale>0.8 0.8 0.8</scale>
          </mesh>
        </geometry>
      </visual>
      <collision name="collision">
        <geometry>
          <mesh>
            <uri>base.obj</uri>
            <scale>0.8 0.8 0.8</scale>
          </mesh>
        </geometry>
      </collision>
    </link>
    <link name="door">
      <pose>0.5 0 0.3 0 0 0</pose>
      <inertial>
        <pose>0.1 0 0 0 0 0</pose>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <iyy>0.1</iyy>
          <izz>0.1</izz>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyz>0.0</iyz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>door.gltf</uri>
          </mesh>
        </geometry>
      </visual>
    </link>
    <joint name="door_hinge" type="revolute">
      <parent>base</parent>
      <child>door</child>
      <pose>0.25 0 0 0 0 0</pose>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>0</lower>
          <upper>1.57</upper>
        </limit>
      </axis>
    </joint>
  </model>
</sdf>
""".format(
            name=name
        )
        sdf_path = self.temp_path / f"{name}.sdf"
        sdf_path.write_text(sdf_content)
        return sdf_path

    def test_rescale_multiplies_existing_link_scale(self):
        """Test that existing mesh scale elements are multiplied."""
        sdf_path = self._create_articulated_sdf("mult_existing")

        # Original scale is 0.8.
        rescale_sdf(sdf_path, scale_factor=1.25)

        # New scale should be 0.8 * 1.25 = 1.0.
        tree = ET.parse(sdf_path)
        base_visual_scale = tree.find(
            ".//link[@name='base']/visual/geometry/mesh/scale"
        )
        scale_values = [float(v) for v in base_visual_scale.text.split()]

        for val in scale_values:
            self.assertAlmostEqual(val, 1.0, places=5)

    def test_rescale_adds_scale_to_link_without_scale(self):
        """Test that scale is added to links that don't have one."""
        sdf_path = self._create_articulated_sdf("add_scale")

        # Door link has no scale element originally.
        rescale_sdf(sdf_path, scale_factor=1.5)

        tree = ET.parse(sdf_path)
        door_visual_scale = tree.find(
            ".//link[@name='door']/visual/geometry/mesh/scale"
        )

        self.assertIsNotNone(door_visual_scale)
        scale_values = [float(v) for v in door_visual_scale.text.split()]
        for val in scale_values:
            self.assertAlmostEqual(val, 1.5, places=5)

    def test_rescale_scales_link_pose_translations(self):
        """Test that link pose translations are scaled."""
        sdf_path = self._create_articulated_sdf("link_pose")

        # Original: base at z=0.5, door at x=0.5, z=0.3.
        tree = ET.parse(sdf_path)
        base_pose = tree.find(".//link[@name='base']/pose")
        door_pose = tree.find(".//link[@name='door']/pose")
        original_base_z = float(base_pose.text.split()[2])
        original_door_x = float(door_pose.text.split()[0])
        original_door_z = float(door_pose.text.split()[2])

        rescale_sdf(sdf_path, scale_factor=2.0)

        tree = ET.parse(sdf_path)
        base_pose = tree.find(".//link[@name='base']/pose")
        door_pose = tree.find(".//link[@name='door']/pose")
        new_base_z = float(base_pose.text.split()[2])
        new_door_x = float(door_pose.text.split()[0])
        new_door_z = float(door_pose.text.split()[2])

        self.assertAlmostEqual(new_base_z, original_base_z * 2.0, places=5)
        self.assertAlmostEqual(new_door_x, original_door_x * 2.0, places=5)
        self.assertAlmostEqual(new_door_z, original_door_z * 2.0, places=5)

    def test_rescale_scales_joint_pose_translations(self):
        """Test that joint pose translations are scaled."""
        sdf_path = self._create_articulated_sdf("joint_pose")

        # Original: joint at x=0.25.
        tree = ET.parse(sdf_path)
        joint_pose = tree.find(".//joint[@name='door_hinge']/pose")
        original_x = float(joint_pose.text.split()[0])

        rescale_sdf(sdf_path, scale_factor=2.0)

        tree = ET.parse(sdf_path)
        joint_pose = tree.find(".//joint[@name='door_hinge']/pose")
        new_x = float(joint_pose.text.split()[0])

        self.assertAlmostEqual(new_x, original_x * 2.0, places=5)

    def test_rescale_does_not_scale_model_pose_articulated(self):
        """Test that model pose is NOT scaled for articulated models."""
        sdf_path = self._create_articulated_sdf("model_pose_art")

        # Modify model pose.
        tree = ET.parse(sdf_path)
        model_pose = tree.find(".//model/pose")
        model_pose.text = "1.0 2.0 3.0 0 0 0"
        tree.write(sdf_path)

        rescale_sdf(sdf_path, scale_factor=2.0)

        tree = ET.parse(sdf_path)
        model_pose = tree.find(".//model/pose")
        pose_values = [float(v) for v in model_pose.text.split()]

        # Model pose should NOT be scaled.
        self.assertAlmostEqual(pose_values[0], 1.0, places=5)
        self.assertAlmostEqual(pose_values[1], 2.0, places=5)
        self.assertAlmostEqual(pose_values[2], 3.0, places=5)

    def test_rescale_scales_both_link_inertias(self):
        """Test that inertia is scaled for all links."""
        sdf_path = self._create_articulated_sdf("multi_inertia")

        # Get original inertia values.
        tree = ET.parse(sdf_path)
        base_ixx = float(tree.find(".//link[@name='base']//ixx").text)
        door_ixx = float(tree.find(".//link[@name='door']//ixx").text)

        rescale_sdf(sdf_path, scale_factor=2.0)

        tree = ET.parse(sdf_path)
        new_base_ixx = float(tree.find(".//link[@name='base']//ixx").text)
        new_door_ixx = float(tree.find(".//link[@name='door']//ixx").text)

        expected_factor = 4.0  # 2^2 for inertia.
        self.assertAlmostEqual(new_base_ixx, base_ixx * expected_factor, places=5)
        self.assertAlmostEqual(new_door_ixx, door_ixx * expected_factor, places=5)


if __name__ == "__main__":
    unittest.main()
