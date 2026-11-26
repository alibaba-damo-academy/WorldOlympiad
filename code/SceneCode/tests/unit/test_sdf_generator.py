import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import trimesh

from scenecode.agent_utils.materials import DEFAULT_FRICTION, get_friction
from scenecode.agent_utils.mesh_physics_analyzer import MeshPhysicsAnalysis
from scenecode.agent_utils.sdf_generator import (
    add_self_collision_filter,
    generate_drake_sdf,
)


class TestGetFriction(unittest.TestCase):
    """Test material friction lookup from materials.yaml."""

    def test_friction_common_materials(self):
        """Test friction coefficients for common materials from materials.yaml."""
        # Values should match materials.yaml.
        self.assertEqual(get_friction("wood"), 0.4)
        self.assertEqual(get_friction("plastic"), 0.35)
        self.assertEqual(get_friction("glass"), 0.9)
        self.assertEqual(get_friction("fabric"), 0.4)
        self.assertEqual(get_friction("rubber"), 1.15)
        self.assertEqual(get_friction("steel"), 0.74)

    def test_friction_case_insensitive(self):
        """Test that material lookup is case-insensitive."""
        self.assertEqual(get_friction("WOOD"), get_friction("wood"))
        self.assertEqual(get_friction("Wood"), get_friction("wood"))

    def test_friction_fallback_unknown_material(self):
        """Test fallback to default friction for unknown material."""
        friction = get_friction("unknown_material_xyz")
        self.assertEqual(friction, DEFAULT_FRICTION)


class TestGenerateDrakeSDF(unittest.TestCase):
    """Test Drake SDF generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_generate_drake_sdf_basic_cube(self):
        """Test SDF generation for a basic cube."""
        # Create a 1x1x1 cube.
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        visual_path = self.temp_path / "cube.gltf"
        mesh.export(visual_path)

        # Create simple collision piece.
        collision_pieces = [mesh]

        # Physics analysis.
        physics = MeshPhysicsAnalysis(
            up_axis="+Z",
            front_axis="+Y",
            material="wood",
            mass_kg=10.0,
            mass_range_kg=(8.0, 12.0),
        )

        # Generate SDF.
        output_path = self.temp_path / "cube.sdf"
        result_path = generate_drake_sdf(
            visual_mesh_path=visual_path,
            collision_pieces=collision_pieces,
            physics_analysis=physics,
            output_path=output_path,
            asset_name="test_cube",
        )

        # Verify file created.
        self.assertTrue(result_path.exists())
        self.assertEqual(result_path, output_path)

        # Parse and validate SDF structure.
        tree = ET.parse(result_path)
        root = tree.getroot()

        self.assertEqual(root.tag, "sdf")
        self.assertEqual(root.get("version"), "1.7")

        # Find model element.
        model = root.find("model")
        self.assertIsNotNone(model)
        self.assertEqual(model.get("name"), "test_cube")

        # Find link element.
        link = model.find("link")
        self.assertIsNotNone(link)

        # Verify inertial properties exist.
        inertial = link.find("inertial")
        self.assertIsNotNone(inertial)

        mass_elem = inertial.find("mass")
        self.assertIsNotNone(mass_elem)
        self.assertAlmostEqual(float(mass_elem.text), 10.0, places=5)

        # Verify inertia tensor has all components.
        inertia_elem = inertial.find("inertia")
        self.assertIsNotNone(inertia_elem)
        self.assertIsNotNone(inertia_elem.find("ixx"))
        self.assertIsNotNone(inertia_elem.find("iyy"))
        self.assertIsNotNone(inertia_elem.find("izz"))
        self.assertIsNotNone(inertia_elem.find("ixy"))
        self.assertIsNotNone(inertia_elem.find("ixz"))
        self.assertIsNotNone(inertia_elem.find("iyz"))

        # Verify visual geometry.
        visual = link.find("visual")
        self.assertIsNotNone(visual)
        visual_geom = visual.find("geometry")
        self.assertIsNotNone(visual_geom)
        visual_mesh = visual_geom.find("mesh")
        self.assertIsNotNone(visual_mesh)
        visual_uri = visual_mesh.find("uri")
        self.assertIsNotNone(visual_uri)
        self.assertEqual(visual_uri.text, "cube.gltf")

        # Verify collision geometry.
        collisions = link.findall("collision")
        self.assertEqual(len(collisions), 1)

        collision = collisions[0]
        self.assertEqual(collision.get("name"), "collision_0")

        # Verify friction properties.
        surface = collision.find("surface")
        self.assertIsNotNone(surface)
        friction_elem = surface.find("friction")
        self.assertIsNotNone(friction_elem)
        ode = friction_elem.find("ode")
        self.assertIsNotNone(ode)
        mu = ode.find("mu")
        self.assertIsNotNone(mu)
        self.assertAlmostEqual(float(mu.text), 0.4, places=3)  # Wood friction.

        # Verify declare_convex element exists (Drake optimization).
        collision_mesh = collision.find("geometry/mesh")
        declare_convex = collision_mesh.find("{drake.mit.edu}declare_convex")
        self.assertIsNotNone(declare_convex)

    def test_generate_drake_sdf_multiple_collision_pieces(self):
        """Test SDF generation with multiple collision pieces."""
        # Create visual mesh.
        visual_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        visual_path = self.temp_path / "multi_collision.gltf"
        visual_mesh.export(visual_path)

        # Create multiple collision pieces.
        piece1 = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
        piece2 = trimesh.creation.box(extents=[0.3, 0.3, 0.3])
        collision_pieces = [piece1, piece2]

        physics = MeshPhysicsAnalysis(
            up_axis="+Z",
            front_axis="+Y",
            material="metal",
            mass_kg=5.0,
            mass_range_kg=(4.0, 6.0),
        )

        output_path = self.temp_path / "multi_collision.sdf"
        generate_drake_sdf(
            visual_mesh_path=visual_path,
            collision_pieces=collision_pieces,
            physics_analysis=physics,
            output_path=output_path,
        )

        # Parse SDF.
        tree = ET.parse(output_path)
        root = tree.getroot()
        link = root.find("model").find("link")

        # Verify two collision pieces.
        collisions = link.findall("collision")
        self.assertEqual(len(collisions), 2)
        self.assertEqual(collisions[0].get("name"), "collision_0")
        self.assertEqual(collisions[1].get("name"), "collision_1")

        # Verify collision mesh files created.
        collision_0_path = self.temp_path / "multi_collision_collision_0.obj"
        collision_1_path = self.temp_path / "multi_collision_collision_1.obj"
        self.assertTrue(collision_0_path.exists())
        self.assertTrue(collision_1_path.exists())

    def test_generate_drake_sdf_invalid_mass_error(self):
        """Test error raised for invalid (non-positive) mass."""
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        visual_path = self.temp_path / "invalid_mass.gltf"
        mesh.export(visual_path)

        collision_pieces = [mesh]

        # Physics with zero mass.
        physics = MeshPhysicsAnalysis(
            up_axis="+Z",
            front_axis="+Y",
            material="wood",
            mass_kg=0.0,  # Invalid!
            mass_range_kg=(0.0, 0.0),
        )

        output_path = self.temp_path / "invalid_mass.sdf"

        with self.assertRaises(ValueError) as context:
            generate_drake_sdf(
                visual_mesh_path=visual_path,
                collision_pieces=collision_pieces,
                physics_analysis=physics,
                output_path=output_path,
            )

        self.assertIn("Mass must be positive", str(context.exception))

    def test_generate_drake_sdf_missing_visual_mesh_error(self):
        """Test error raised when visual mesh file doesn't exist."""
        nonexistent_path = self.temp_path / "nonexistent.gltf"

        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        collision_pieces = [mesh]

        physics = MeshPhysicsAnalysis(
            up_axis="+Z",
            front_axis="+Y",
            material="wood",
            mass_kg=10.0,
            mass_range_kg=(8.0, 12.0),
        )

        output_path = self.temp_path / "output.sdf"

        with self.assertRaises(FileNotFoundError) as context:
            generate_drake_sdf(
                visual_mesh_path=nonexistent_path,
                collision_pieces=collision_pieces,
                physics_analysis=physics,
                output_path=output_path,
            )

        self.assertIn("Visual mesh not found", str(context.exception))

    def test_generate_drake_sdf_empty_collision_pieces_error(self):
        """Test error raised for empty collision pieces list."""
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        visual_path = self.temp_path / "no_collision.gltf"
        mesh.export(visual_path)

        collision_pieces = []  # Empty!

        physics = MeshPhysicsAnalysis(
            up_axis="+Z",
            front_axis="+Y",
            material="wood",
            mass_kg=10.0,
            mass_range_kg=(8.0, 12.0),
        )

        output_path = self.temp_path / "no_collision.sdf"

        with self.assertRaises(ValueError) as context:
            generate_drake_sdf(
                visual_mesh_path=visual_path,
                collision_pieces=collision_pieces,
                physics_analysis=physics,
                output_path=output_path,
            )

        self.assertIn("collision_pieces cannot be empty", str(context.exception))

    def test_generate_drake_sdf_inertia_tensor_unit_cube(self):
        """Test inertia tensor calculation for unit cube with known values."""
        # For a 1x1x1 cube with mass m and side length a, centered at origin:
        # I_xx = I_yy = I_zz = (m * a²) / 6.
        # Off-diagonal elements should be close to 0 for centered cube.

        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        visual_path = self.temp_path / "unit_cube.gltf"
        mesh.export(visual_path)

        collision_pieces = [mesh]

        mass = 12.0  # Use 12kg for easy calculation.
        physics = MeshPhysicsAnalysis(
            up_axis="+Z",
            front_axis="+Y",
            material="wood",
            mass_kg=mass,
            mass_range_kg=(10.0, 14.0),
        )

        output_path = self.temp_path / "unit_cube.sdf"
        generate_drake_sdf(
            visual_mesh_path=visual_path,
            collision_pieces=collision_pieces,
            physics_analysis=physics,
            output_path=output_path,
        )

        # Parse SDF and check inertia values.
        tree = ET.parse(output_path)
        inertia = tree.find(".//inertia")

        ixx = float(inertia.find("ixx").text)
        iyy = float(inertia.find("iyy").text)
        izz = float(inertia.find("izz").text)
        ixy = float(inertia.find("ixy").text)
        ixz = float(inertia.find("ixz").text)
        iyz = float(inertia.find("iyz").text)

        # Expected: I_xx ≈ I_yy ≈ I_zz ≈ (mass * side²) / 6 = (12 * 1²) / 6 = 2.0.
        expected_diagonal = (mass * 1.0**2) / 6.0
        self.assertAlmostEqual(ixx, expected_diagonal, delta=0.1)
        self.assertAlmostEqual(iyy, expected_diagonal, delta=0.1)
        self.assertAlmostEqual(izz, expected_diagonal, delta=0.1)

        # Off-diagonal should be near zero for centered cube.
        self.assertAlmostEqual(ixy, 0.0, delta=0.1)
        self.assertAlmostEqual(ixz, 0.0, delta=0.1)
        self.assertAlmostEqual(iyz, 0.0, delta=0.1)

    def test_generate_drake_sdf_material_friction_mapping(self):
        """Test friction coefficient mapping for different materials."""
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        visual_path = self.temp_path / "friction_test.gltf"
        mesh.export(visual_path)
        collision_pieces = [mesh]

        # Values from materials.yaml.
        materials_and_frictions = [
            ("wood", 0.4),
            ("steel", 0.74),
            ("plastic", 0.35),
            ("glass", 0.9),
            ("rubber", 1.15),
            ("unknown_material", DEFAULT_FRICTION),  # Unknown uses default.
        ]

        for material, expected_friction in materials_and_frictions:
            with self.subTest(material=material):
                physics = MeshPhysicsAnalysis(
                    up_axis="+Z",
                    front_axis="+Y",
                    material=material,
                    mass_kg=10.0,
                    mass_range_kg=(8.0, 12.0),
                )

                output_path = self.temp_path / f"{material}.sdf"
                generate_drake_sdf(
                    visual_mesh_path=visual_path,
                    collision_pieces=collision_pieces,
                    physics_analysis=physics,
                    output_path=output_path,
                )

                # Parse and check friction.
                tree = ET.parse(output_path)
                mu = tree.find(".//surface/friction/ode/mu")
                self.assertIsNotNone(mu)
                self.assertAlmostEqual(
                    float(mu.text),
                    expected_friction,
                    places=3,
                    msg=f"Material: {material}",
                )

    def test_generate_drake_sdf_com_coordinate_transform(self):
        """Test center of mass is correctly transformed from Y-up to Z-up.

        Regression test for coordinate system bug where COM computed in Y-up
        GLTF coordinates was written to SDF without transformation to Z-up.
        Drake expects Z-up coordinates since it auto-converts Y-up GLTF visuals.
        """
        # Create box offset in Y direction (Y-up space).
        # In Y-up: COM at (0, 2, 0).
        # After Y-up to Z-up transform: COM should be at (0, 0, 2).
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        mesh.apply_translation([0, 2, 0])

        visual_path = self.temp_path / "offset_cube.gltf"
        mesh.export(visual_path)

        collision_pieces = [trimesh.creation.box(extents=[1.0, 1.0, 1.0])]

        physics = MeshPhysicsAnalysis(
            up_axis="+Z",
            front_axis="+Y",
            material="wood",
            mass_kg=10.0,
            mass_range_kg=(8.0, 12.0),
        )

        output_path = self.temp_path / "offset_cube.sdf"
        generate_drake_sdf(
            visual_mesh_path=visual_path,
            collision_pieces=collision_pieces,
            physics_analysis=physics,
            output_path=output_path,
        )

        # Parse SDF and check COM pose.
        tree = ET.parse(output_path)
        com_pose = tree.find(".//inertial/pose")
        self.assertIsNotNone(com_pose)

        # Parse pose: "x y z roll pitch yaw".
        pose_values = [float(v) for v in com_pose.text.split()]
        com_x, com_y, com_z = pose_values[0], pose_values[1], pose_values[2]

        # After Y-up to Z-up transform, (0, 2, 0) becomes (0, 0, 2).
        self.assertAlmostEqual(com_x, 0.0, delta=0.01)
        self.assertAlmostEqual(com_y, 0.0, delta=0.01)
        self.assertAlmostEqual(com_z, 2.0, delta=0.01)


class TestAddSelfCollisionFilter(unittest.TestCase):
    """Test self-collision filter injection for articulated SDF files."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _write_sdf(self, content: str) -> Path:
        """Write SDF content to a temp file and return the path."""
        sdf_path = self.temp_path / "test_model.sdf"
        sdf_path.write_text(content, encoding="utf-8")
        return sdf_path

    def test_multi_link_adds_filter(self):
        """Multi-link SDF gets collision filter with all links as members."""
        sdf_content = """\
<?xml version='1.0' encoding='utf-8'?>
<sdf xmlns:drake="drake.mit.edu" version="1.11">
  <model name="test_articulated">
    <link name="body"/>
    <link name="door"/>
    <joint name="door_joint" type="revolute">
      <parent>body</parent>
      <child>door</child>
      <axis><xyz>0 0 1</xyz></axis>
    </joint>
  </model>
</sdf>"""
        sdf_path = self._write_sdf(sdf_content)
        add_self_collision_filter(sdf_path)

        tree = ET.parse(sdf_path)
        ns = "drake.mit.edu"
        model = tree.find("model")
        group = model.find(f"{{{ns}}}collision_filter_group")

        self.assertIsNotNone(group)
        self.assertEqual(group.get("name"), "no_self_collision")

        members = [m.text for m in group.findall(f"{{{ns}}}member")]
        self.assertEqual(sorted(members), ["body", "door"])

        ignored = group.find(f"{{{ns}}}ignored_collision_filter_group")
        self.assertIsNotNone(ignored)
        self.assertEqual(ignored.text, "no_self_collision")

    def test_idempotent(self):
        """Running twice does not duplicate the collision filter group."""
        sdf_content = """\
<?xml version='1.0' encoding='utf-8'?>
<sdf xmlns:drake="drake.mit.edu" version="1.11">
  <model name="test_articulated">
    <link name="body"/>
    <link name="door"/>
  </model>
</sdf>"""
        sdf_path = self._write_sdf(sdf_content)
        add_self_collision_filter(sdf_path)
        add_self_collision_filter(sdf_path)

        tree = ET.parse(sdf_path)
        ns = "drake.mit.edu"
        model = tree.find("model")
        groups = model.findall(f"{{{ns}}}collision_filter_group")
        self.assertEqual(len(groups), 1)

    def test_single_link_skipped(self):
        """Single-link SDF is left unmodified."""
        sdf_content = """\
<?xml version='1.0' encoding='utf-8'?>
<sdf xmlns:drake="drake.mit.edu" version="1.11">
  <model name="simple_object">
    <link name="base_link"/>
  </model>
</sdf>"""
        sdf_path = self._write_sdf(sdf_content)
        add_self_collision_filter(sdf_path)

        tree = ET.parse(sdf_path)
        ns = "drake.mit.edu"
        model = tree.find("model")
        group = model.find(f"{{{ns}}}collision_filter_group")
        self.assertIsNone(group)

    def test_world_model_structure(self):
        """Handles <sdf><world><model> nesting correctly."""
        sdf_content = """\
<?xml version='1.0' encoding='utf-8'?>
<sdf xmlns:drake="drake.mit.edu" version="1.11">
  <world name="root_world">
    <model name="test_articulated">
      <link name="body"/>
      <link name="door"/>
      <link name="drawer"/>
    </model>
  </world>
</sdf>"""
        sdf_path = self._write_sdf(sdf_content)
        add_self_collision_filter(sdf_path)

        tree = ET.parse(sdf_path)
        ns = "drake.mit.edu"
        model = tree.find("world/model")
        group = model.find(f"{{{ns}}}collision_filter_group")

        self.assertIsNotNone(group)
        members = [m.text for m in group.findall(f"{{{ns}}}member")]
        self.assertEqual(sorted(members), ["body", "door", "drawer"])


if __name__ == "__main__":
    unittest.main()
