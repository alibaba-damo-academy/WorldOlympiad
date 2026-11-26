import tempfile
import unittest

from pathlib import Path

import numpy as np
import trimesh

from pydrake.all import RigidTransform, RollPitchYaw

from scenecode.agent_utils.room import (
    ObjectType,
    SceneObject,
    clone_scene_object,
    clone_scene_object,
    SupportSurface,
    UniqueID,
    serialize_composite_member_asset,
    serialize_composite_member_asset,
    extract_base_link_name_from_sdf,
)


class TestExtractBaseLinkNameFromSdf(unittest.TestCase):
    """Test cases for extract_base_link_name_from_sdf function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parent.parent / "test_data"

    def test_extract_body_link(self):
        """Test extracting body link from box SDF file."""
        sdf_path = self.test_data_dir / "simple_box.sdf"
        link_name = extract_base_link_name_from_sdf(sdf_path)
        self.assertEqual(link_name, "body")

    def test_extract_sphere_link(self):
        """Test extracting sphere_link from sphere SDF file."""
        sdf_path = self.test_data_dir / "simple_sphere.sdf"
        link_name = extract_base_link_name_from_sdf(sdf_path)
        self.assertEqual(link_name, "sphere_link")

    def test_extract_from_room_geometry(self):
        """Test extracting link name from room geometry SDF."""
        sdf_path = self.test_data_dir / "simple_room_geometry.sdf"
        link_name = extract_base_link_name_from_sdf(sdf_path)
        self.assertEqual(link_name, "room_geometry_body_link")

    def test_file_not_found_raises_error(self):
        """Test that missing file raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            extract_base_link_name_from_sdf(Path("/nonexistent/file.sdf"))
        self.assertIn("SDF file not found", str(cm.exception))

    def test_invalid_xml_raises_error(self):
        """Test that invalid XML raises ValueError."""
        # Create temporary invalid SDF file.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False) as f:
            f.write("invalid xml content <unclosed")
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as cm:
                extract_base_link_name_from_sdf(temp_path)
            self.assertIn("Failed to parse SDF file", str(cm.exception))
        finally:
            temp_path.unlink(missing_ok=True)


class TestUniqueID(unittest.TestCase):
    """Test cases for UniqueID class."""

    def test_generate(self):
        """Test that generate creates unique IDs."""
        id1 = UniqueID.generate()
        id2 = UniqueID.generate()

        self.assertIsInstance(id1, UniqueID)
        self.assertIsInstance(id2, UniqueID)
        self.assertNotEqual(id1, id2)
        self.assertEqual(len(str(id1)), 36)  # Standard UUID4 length

    def test_generate_unique(self):
        """Test that generate_unique creates unique IDs with base-36 suffixes."""
        existing: set[UniqueID] = set()

        # Generate first ID - should be name_0.
        id1 = UniqueID.generate_unique("Dining Table", existing)
        self.assertEqual(str(id1), "dining_table_0")
        existing.add(id1)

        # Generate second ID - should be name_1.
        id2 = UniqueID.generate_unique("Dining Table", existing)
        self.assertEqual(str(id2), "dining_table_1")
        existing.add(id2)

        # Generate third ID - should be name_2.
        id3 = UniqueID.generate_unique("Dining Table", existing)
        self.assertEqual(str(id3), "dining_table_2")

    def test_generate_unique_with_dict(self):
        """Test that generate_unique works with dict keys."""
        existing: dict[str, int] = {"dining_table_0": 1, "dining_table_1": 2}

        # Should skip 0 and 1, return 2.
        id1 = UniqueID.generate_unique("Dining Table", existing)
        self.assertEqual(str(id1), "dining_table_2")


class TestObjectType(unittest.TestCase):
    """Test cases for ObjectType enum."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        self.assertEqual(ObjectType.FURNITURE.value, "furniture")
        self.assertEqual(ObjectType.MANIPULAND.value, "manipuland")


class TestSupportSurface(unittest.TestCase):
    """Test cases for SupportSurface dataclass."""

    def test_creation(self):
        """Test that SupportSurface can be created with required fields."""
        surface_id = UniqueID.generate()
        bbox_min = np.array([0.0, 0.0, 0.0])
        bbox_max = np.array([1.0, 1.0, 0.1])
        transform = RigidTransform()

        surface = SupportSurface(
            surface_id=surface_id,
            bounding_box_min=bbox_min,
            bounding_box_max=bbox_max,
            transform=transform,
        )

        self.assertEqual(surface.surface_id, surface_id)
        np.testing.assert_array_equal(surface.bounding_box_min, bbox_min)
        np.testing.assert_array_equal(surface.bounding_box_max, bbox_max)
        self.assertEqual(surface.transform, transform)

    def test_to_world_pose_identity_transform(self):
        """Test SE(2) to SE(3) conversion with identity surface transform."""
        surface = SupportSurface(
            surface_id=UniqueID.generate(),
            bounding_box_min=np.array([-0.5, -0.1, -0.5]),
            bounding_box_max=np.array([0.5, 0.1, 0.5]),
            transform=RigidTransform(),
        )

        position_2d = np.array([0.1, 0.2])
        rotation_2d = np.pi / 4

        world_pose = surface.to_world_pose(
            position_2d=position_2d, rotation_2d=rotation_2d
        )

        expected_position = np.array([0.1, 0.2, 0.0])
        np.testing.assert_array_almost_equal(
            world_pose.translation(), expected_position, decimal=6
        )

        expected_rpy = RollPitchYaw([0.0, 0.0, np.pi / 4])
        np.testing.assert_array_almost_equal(
            world_pose.rotation().ToRollPitchYaw().vector(),
            expected_rpy.vector(),
            decimal=6,
        )

    def test_to_world_pose_translated_surface(self):
        """Test SE(2) to SE(3) conversion with translated surface."""
        surface_transform = RigidTransform(p=[1.0, 2.0, 3.0])
        surface = SupportSurface(
            surface_id=UniqueID.generate(),
            bounding_box_min=np.array([-0.5, -0.1, -0.5]),
            bounding_box_max=np.array([0.5, 0.1, 0.5]),
            transform=surface_transform,
        )

        position_2d = np.array([0.1, 0.2])
        rotation_2d = 0.0

        world_pose = surface.to_world_pose(
            position_2d=position_2d, rotation_2d=rotation_2d
        )

        expected_position = np.array([1.1, 2.2, 3.0])
        np.testing.assert_array_almost_equal(
            world_pose.translation(), expected_position, decimal=6
        )

    def test_to_world_pose_rotated_surface(self):
        """Test SE(2) to SE(3) conversion with rotated surface."""
        surface_transform = RigidTransform(
            rpy=RollPitchYaw([0.0, 0.0, np.pi / 2]), p=[0.0, 0.0, 0.0]
        )
        surface = SupportSurface(
            surface_id=UniqueID.generate(),
            bounding_box_min=np.array([-0.5, -0.1, -0.5]),
            bounding_box_max=np.array([0.5, 0.1, 0.5]),
            transform=surface_transform,
        )

        position_2d = np.array([0.1, 0.0])
        rotation_2d = 0.0

        world_pose = surface.to_world_pose(
            position_2d=position_2d, rotation_2d=rotation_2d
        )

        expected_position = np.array([-0.0, 0.1, 0.0])
        np.testing.assert_array_almost_equal(
            world_pose.translation(), expected_position, decimal=6
        )

    def test_contains_point_2d_inside(self):
        """Test that contains_point_2d returns True for points inside bounds."""
        # Create a simple rectangular mesh matching the bounding box.
        vertices = np.array(
            [
                [-0.5, -0.3, 0.0],
                [0.5, -0.3, 0.0],
                [0.5, 0.3, 0.0],
                [-0.5, 0.3, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        surface = SupportSurface(
            surface_id=UniqueID.generate(),
            bounding_box_min=np.array([-0.5, -0.3, 0.0]),
            bounding_box_max=np.array([0.5, 0.3, 0.1]),
            transform=RigidTransform(),
            mesh=mesh,
        )

        self.assertTrue(surface.contains_point_2d(np.array([0.0, 0.0])))
        self.assertTrue(surface.contains_point_2d(np.array([0.4, 0.2])))
        self.assertTrue(surface.contains_point_2d(np.array([-0.4, -0.2])))

    def test_contains_point_2d_on_boundary(self):
        """Test that contains_point_2d returns True for points on boundary."""
        # Create a simple rectangular mesh matching the bounding box.
        vertices = np.array(
            [
                [-0.5, -0.3, 0.0],
                [0.5, -0.3, 0.0],
                [0.5, 0.3, 0.0],
                [-0.5, 0.3, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        surface = SupportSurface(
            surface_id=UniqueID.generate(),
            bounding_box_min=np.array([-0.5, -0.3, 0.0]),
            bounding_box_max=np.array([0.5, 0.3, 0.1]),
            transform=RigidTransform(),
            mesh=mesh,
        )

        self.assertTrue(surface.contains_point_2d(np.array([-0.5, -0.3])))
        self.assertTrue(surface.contains_point_2d(np.array([0.5, 0.3])))
        self.assertTrue(surface.contains_point_2d(np.array([0.0, -0.3])))
        self.assertTrue(surface.contains_point_2d(np.array([0.0, 0.3])))

    def test_contains_point_2d_outside(self):
        """Test that contains_point_2d returns False for points outside bounds."""
        # Create a simple rectangular mesh matching the bounding box.
        vertices = np.array(
            [
                [-0.5, -0.3, 0.0],
                [0.5, -0.3, 0.0],
                [0.5, 0.3, 0.0],
                [-0.5, 0.3, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        surface = SupportSurface(
            surface_id=UniqueID.generate(),
            bounding_box_min=np.array([-0.5, -0.3, 0.0]),
            bounding_box_max=np.array([0.5, 0.3, 0.1]),
            transform=RigidTransform(),
            mesh=mesh,
        )

        self.assertFalse(surface.contains_point_2d(np.array([0.6, 0.0])))
        self.assertFalse(surface.contains_point_2d(np.array([-0.6, 0.0])))
        self.assertFalse(surface.contains_point_2d(np.array([0.0, 0.4])))
        self.assertFalse(surface.contains_point_2d(np.array([0.0, -0.4])))
        self.assertFalse(surface.contains_point_2d(np.array([1.0, 1.0])))


# Note: Tests for support surface extraction moved to test_support_surface_extraction.py
# which tests the new HSM-based multi-surface extraction algorithm.


class TestSceneObject(unittest.TestCase):
    """Test cases for SceneObject dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.object_id = UniqueID.generate()
        self.transform = RigidTransform()

    def test_minimal_creation(self):
        """Test creating SceneObject with minimal required fields."""
        obj = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=self.transform,
        )

        self.assertEqual(obj.object_id, self.object_id)
        self.assertEqual(obj.object_type, ObjectType.FURNITURE)
        self.assertEqual(obj.name, "Test Object")
        self.assertEqual(obj.description, "A test object")
        self.assertEqual(obj.transform, self.transform)
        self.assertIsNone(obj.geometry_path)
        self.assertIsNone(obj.sdf_path)
        self.assertIsNone(obj.image_path)
        self.assertEqual(obj.support_surfaces, [])
        self.assertEqual(obj.metadata, {})

    def test_full_creation(self):
        """Test creating SceneObject with all fields."""
        support_surface = SupportSurface(
            surface_id=UniqueID.generate(),
            bounding_box_min=np.array([0.0, 0.0, 0.0]),
            bounding_box_max=np.array([1.0, 1.0, 0.1]),
            transform=RigidTransform(),
        )

        obj = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.MANIPULAND,
            name="Test Object",
            description="A test object",
            transform=self.transform,
            geometry_path=Path("/path/to/geometry.glb"),
            sdf_path=Path("/path/to/model.sdf"),
            image_path=Path("/path/to/image.png"),
            support_surfaces=[support_surface],
            metadata={"material": "wood", "weight": 5.0},
        )

        self.assertEqual(obj.geometry_path, Path("/path/to/geometry.glb"))
        self.assertEqual(obj.sdf_path, Path("/path/to/model.sdf"))
        self.assertEqual(obj.image_path, Path("/path/to/image.png"))
        self.assertEqual(len(obj.support_surfaces), 1)
        self.assertEqual(obj.metadata["material"], "wood")
        self.assertEqual(obj.metadata["weight"], 5.0)

    def test_bounding_box_fields(self):
        """Test SceneObject with bounding box fields."""
        bbox_min = np.array([0.0, 0.0, 0.0])
        bbox_max = np.array([1.0, 2.0, 0.5])

        obj = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=self.transform,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )

        np.testing.assert_array_equal(obj.bbox_min, bbox_min)
        np.testing.assert_array_equal(obj.bbox_max, bbox_max)

    def test_compute_world_bounds_no_bounds(self):
        """Test compute_world_bounds returns None when no bounds available."""
        obj = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=self.transform,
        )

        result = obj.compute_world_bounds()
        self.assertIsNone(result)

    def test_compute_world_bounds_with_identity_transform(self):
        """Test compute_world_bounds with identity transform."""
        bbox_min = np.array([0.0, 0.0, 0.0])
        bbox_max = np.array([1.0, 2.0, 0.5])

        obj = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=RigidTransform(),  # Identity transform
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )

        world_min, world_max = obj.compute_world_bounds()

        # With identity transform, world bounds should equal object bounds.
        np.testing.assert_array_almost_equal(world_min, bbox_min)
        np.testing.assert_array_almost_equal(world_max, bbox_max)

    def test_compute_world_bounds_with_translation(self):
        """Test compute_world_bounds with translation."""
        bbox_min = np.array([0.0, 0.0, 0.0])
        bbox_max = np.array([1.0, 2.0, 0.5])
        translation = np.array([5.0, 3.0, 1.0])

        obj = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=RigidTransform(p=translation),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )

        world_min, world_max = obj.compute_world_bounds()

        # World bounds should be shifted by translation.
        expected_min = bbox_min + translation
        expected_max = bbox_max + translation
        np.testing.assert_array_almost_equal(world_min, expected_min)
        np.testing.assert_array_almost_equal(world_max, expected_max)

    def test_compute_world_bounds_with_rotation(self):
        """Test compute_world_bounds with 90-degree Z-axis rotation."""
        bbox_min = np.array([0.0, 0.0, 0.0])
        bbox_max = np.array([1.0, 2.0, 0.5])

        # 90-degree rotation around Z-axis.
        rotation = RollPitchYaw(0, 0, np.pi / 2)

        obj = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=RigidTransform(rpy=rotation, p=np.array([0.0, 0.0, 0.0])),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )

        world_min, world_max = obj.compute_world_bounds()

        # After 90-degree CCW rotation around Z:
        # X' = -Y, Y' = X, Z' = Z
        # Original corners [0,0,0]-[1,2,0.5] become [-2,0,0]-[0,1,0.5].
        np.testing.assert_array_almost_equal(world_min, [-2.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(world_max, [0.0, 1.0, 0.5])

    def test_get_effective_transform_with_internal_model_pose(self):
        """Articulated objects compose scene and internal model poses."""
        obj = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Articulated Cabinet",
            description="A test articulated cabinet",
            transform=RigidTransform(rpy=RollPitchYaw(0.0, 0.0, np.pi / 2), p=[1.0, 2.0, 0.0]),
            internal_model_pose=RigidTransform(
                rpy=RollPitchYaw(0.0, 0.0, np.pi),
                p=[0.0, -0.5, 0.0],
            ),
        )

        effective = obj.get_effective_transform()
        np.testing.assert_array_almost_equal(
            effective.translation(),
            np.array([1.5, 2.0, 0.0]),
        )
        self.assertAlmostEqual(
            effective.rotation().ToRollPitchYaw().yaw_angle(),
            -np.pi / 2,
            places=6,
        )

    def test_compute_world_bounds_with_internal_model_pose(self):
        """World bounds should honor articulated internal model pose."""
        obj = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Wardrobe",
            description="A test wardrobe",
            transform=RigidTransform(p=[2.0, 3.0, 0.0]),
            internal_model_pose=RigidTransform(p=[-1.0, 0.5, 0.0]),
            bbox_min=np.array([0.0, 0.0, 0.0]),
            bbox_max=np.array([1.0, 2.0, 0.5]),
        )

        world_min, world_max = obj.compute_world_bounds()
        np.testing.assert_array_almost_equal(world_min, [1.0, 3.5, 0.0])
        np.testing.assert_array_almost_equal(world_max, [2.0, 5.5, 0.5])

    def test_clone_scene_object_preserves_internal_model_pose(self):
        """Cloning scene objects should preserve articulated internal poses."""
        original = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Wardrobe",
            description="A test wardrobe",
            transform=self.transform,
            internal_model_pose=RigidTransform(
                rpy=RollPitchYaw(0.0, 0.0, np.pi),
                p=np.array([0.0, -0.25, 0.0]),
            ),
            metadata={"nested": {"keep": True}},
            bbox_min=np.array([0.0, 0.0, 0.0]),
            bbox_max=np.array([1.0, 2.0, 3.0]),
            scale_factor=1.5,
        )

        cloned = clone_scene_object(
            original,
            object_id=UniqueID("wardrobe_1"),
            transform=RigidTransform(p=[1.0, 2.0, 3.0]),
            placement_info=None,
        )

        np.testing.assert_array_almost_equal(
            cloned.internal_model_pose.translation(),
            original.internal_model_pose.translation(),
        )
        np.testing.assert_array_almost_equal(
            cloned.internal_model_pose.rotation().matrix(),
            original.internal_model_pose.rotation().matrix(),
        )
        self.assertEqual(cloned.metadata, original.metadata)
        self.assertIsNot(cloned.metadata, original.metadata)
        np.testing.assert_array_almost_equal(cloned.bbox_min, original.bbox_min)
        np.testing.assert_array_almost_equal(cloned.bbox_max, original.bbox_max)

    def test_serialize_composite_member_asset_includes_internal_model_pose(self):
        """Composite member metadata should carry internal model pose."""
        asset = SceneObject(
            object_id=UniqueID("wardrobe_0"),
            object_type=ObjectType.FURNITURE,
            name="Wardrobe",
            description="A test wardrobe",
            transform=RigidTransform(),
            internal_model_pose=RigidTransform(
                rpy=RollPitchYaw(0.0, 0.0, np.pi),
                p=np.array([0.0, -0.25, 0.0]),
            ),
            sdf_path=Path("/tmp/wardrobe.sdf"),
        )

        member = serialize_composite_member_asset(
            asset, RigidTransform(p=[1.0, 2.0, 3.0])
        )

        self.assertIn("internal_model_pose", member)
        self.assertEqual(member["sdf_path"], "/tmp/wardrobe.sdf")
        np.testing.assert_array_almost_equal(
            member["internal_model_pose"]["translation"],
            [0.0, -0.25, 0.0],
        )

    def test_content_hash_includes_bounding_box(self):
        """Test that content hash includes bounding box data."""
        bbox_min = np.array([0.0, 0.0, 0.0])
        bbox_max = np.array([1.0, 2.0, 0.5])

        obj1 = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=self.transform,
        )

        obj2 = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=self.transform,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )

        # Content hashes should be different.
        self.assertNotEqual(obj1.content_hash(), obj2.content_hash())

    def test_to_dict_minimal(self):
        """Test serialization of SceneObject with minimal fields."""
        obj = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=self.transform,
        )

        obj_dict = obj.to_dict()

        # Check required fields.
        self.assertEqual(obj_dict["object_id"], str(self.object_id))
        self.assertEqual(obj_dict["object_type"], ObjectType.FURNITURE.value)
        self.assertEqual(obj_dict["name"], "Test Object")
        self.assertEqual(obj_dict["description"], "A test object")
        self.assertIn("transform", obj_dict)
        self.assertIn("translation", obj_dict["transform"])
        self.assertIn("rotation_wxyz", obj_dict["transform"])

        # Check optional fields are None.
        self.assertIsNone(obj_dict["geometry_path"])
        self.assertIsNone(obj_dict["sdf_path"])
        self.assertIsNone(obj_dict["image_path"])
        self.assertEqual(obj_dict["support_surfaces"], [])
        self.assertIsNone(obj_dict["placement_info"])
        self.assertEqual(obj_dict["metadata"], {})

    def test_to_dict_with_paths(self):
        """Test serialization with paths (absolute and relative)."""
        obj = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.MANIPULAND,
            name="Test Object",
            description="A test object",
            transform=self.transform,
            geometry_path=Path("/tmp/scene/assets/object.glb"),
            sdf_path=Path("/tmp/scene/models/object.sdf"),
            image_path=Path("/tmp/scene/images/object.png"),
        )

        # Test with scene_dir for relative paths.
        scene_dir = Path("/tmp/scene")
        obj_dict = obj.to_dict(scene_dir=scene_dir)

        self.assertEqual(obj_dict["geometry_path"], "assets/object.glb")
        self.assertEqual(obj_dict["sdf_path"], "models/object.sdf")
        self.assertEqual(obj_dict["image_path"], "images/object.png")

        # Test without scene_dir for absolute paths.
        obj_dict_abs = obj.to_dict()

        self.assertEqual(obj_dict_abs["geometry_path"], "/tmp/scene/assets/object.glb")
        self.assertEqual(obj_dict_abs["sdf_path"], "/tmp/scene/models/object.sdf")
        self.assertEqual(obj_dict_abs["image_path"], "/tmp/scene/images/object.png")

    def test_to_dict_with_support_surfaces(self):
        """Test serialization with support surfaces."""
        surface = SupportSurface(
            surface_id=UniqueID.generate(),
            bounding_box_min=np.array([0.0, 0.0, 0.0]),
            bounding_box_max=np.array([1.0, 2.0, 0.1]),
            transform=RigidTransform(p=np.array([0.5, 1.0, 0.8])),
        )

        obj = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=self.transform,
            support_surfaces=[surface],
        )

        obj_dict = obj.to_dict()

        self.assertEqual(len(obj_dict["support_surfaces"]), 1)
        surf_dict = obj_dict["support_surfaces"][0]
        self.assertEqual(surf_dict["surface_id"], str(surface.surface_id))
        self.assertEqual(surf_dict["bounding_box_min"], [0.0, 0.0, 0.0])
        self.assertEqual(surf_dict["bounding_box_max"], [1.0, 2.0, 0.1])
        self.assertIn("transform", surf_dict)

    def test_from_dict_minimal(self):
        """Test deserialization of SceneObject with minimal fields."""
        obj_dict = {
            "object_id": str(self.object_id),
            "object_type": ObjectType.FURNITURE.value,
            "name": "Test Object",
            "description": "A test object",
            "transform": {
                "translation": [0.0, 0.0, 0.0],
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "geometry_path": None,
            "sdf_path": None,
            "image_path": None,
            "support_surfaces": [],
            "placement_info": None,
            "metadata": {},
            "bbox_min": None,
            "bbox_max": None,
            "immutable": False,
        }

        obj = SceneObject.from_dict(obj_dict)

        self.assertEqual(str(obj.object_id), str(self.object_id))
        self.assertEqual(obj.object_type, ObjectType.FURNITURE)
        self.assertEqual(obj.name, "Test Object")
        self.assertEqual(obj.description, "A test object")
        self.assertIsNone(obj.geometry_path)
        self.assertIsNone(obj.sdf_path)
        self.assertIsNone(obj.image_path)
        self.assertEqual(obj.support_surfaces, [])
        self.assertIsNone(obj.placement_info)
        self.assertEqual(obj.metadata, {})
        self.assertFalse(obj.immutable)

    def test_from_dict_with_paths(self):
        """Test deserialization with paths (absolute and relative)."""
        obj_dict = {
            "object_id": str(self.object_id),
            "object_type": ObjectType.MANIPULAND.value,
            "name": "Test Object",
            "description": "A test object",
            "transform": {
                "translation": [1.0, 2.0, 3.0],
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "geometry_path": "assets/object.glb",
            "sdf_path": "models/object.sdf",
            "image_path": "images/object.png",
            "support_surfaces": [],
            "placement_info": None,
            "metadata": {},
            "bbox_min": None,
            "bbox_max": None,
            "immutable": False,
        }

        # Test with scene_dir for path resolution.
        scene_dir = Path("/tmp/scene")
        obj = SceneObject.from_dict(obj_dict, scene_dir=scene_dir)

        self.assertEqual(obj.geometry_path, scene_dir / "assets/object.glb")
        self.assertEqual(obj.sdf_path, scene_dir / "models/object.sdf")
        self.assertEqual(obj.image_path, scene_dir / "images/object.png")

        # Test without scene_dir (paths as-is).
        obj_no_dir = SceneObject.from_dict(obj_dict)

        self.assertEqual(obj_no_dir.geometry_path, Path("assets/object.glb"))
        self.assertEqual(obj_no_dir.sdf_path, Path("models/object.sdf"))
        self.assertEqual(obj_no_dir.image_path, Path("images/object.png"))

    def test_serialization_roundtrip_minimal(self):
        """Test serialization and deserialization roundtrip with minimal fields."""
        original = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=RigidTransform(p=np.array([1.0, 2.0, 3.0])),
        )

        # Serialize and deserialize.
        obj_dict = original.to_dict()
        restored = SceneObject.from_dict(obj_dict)

        # Check equality.
        self.assertEqual(str(restored.object_id), str(original.object_id))
        self.assertEqual(restored.object_type, original.object_type)
        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.description, original.description)
        np.testing.assert_array_almost_equal(
            restored.transform.translation(), original.transform.translation()
        )

    def test_serialization_roundtrip_full(self):
        """Test serialization roundtrip with all fields."""
        surface = SupportSurface(
            surface_id=UniqueID.generate(),
            bounding_box_min=np.array([0.0, 0.0, 0.0]),
            bounding_box_max=np.array([1.0, 2.0, 0.1]),
            transform=RigidTransform(p=np.array([0.5, 1.0, 0.8])),
            link_name="E_drawer_1",  # For articulated FK transforms.
        )

        original = SceneObject(
            object_id=self.object_id,
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=RigidTransform(p=np.array([1.0, 2.0, 3.0])),
            internal_model_pose=RigidTransform(
                rpy=RollPitchYaw(0.0, 0.0, np.pi),
                p=np.array([0.0, -0.25, 0.0]),
            ),
            geometry_path=Path("/tmp/scene/assets/object.glb"),
            sdf_path=Path("/tmp/scene/models/object.sdf"),
            image_path=Path("/tmp/scene/images/object.png"),
            support_surfaces=[surface],
            metadata={"material": "wood", "weight": 5.0},
            bbox_min=np.array([0.0, 0.0, 0.0]),
            bbox_max=np.array([1.0, 2.0, 3.0]),
            immutable=True,
        )

        # Serialize and deserialize with scene_dir.
        scene_dir = Path("/tmp/scene")
        obj_dict = original.to_dict(scene_dir=scene_dir)
        restored = SceneObject.from_dict(obj_dict, scene_dir=scene_dir)

        # Check all fields.
        self.assertEqual(str(restored.object_id), str(original.object_id))
        self.assertEqual(restored.object_type, original.object_type)
        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.description, original.description)
        self.assertEqual(restored.geometry_path, original.geometry_path)
        self.assertEqual(restored.sdf_path, original.sdf_path)
        self.assertEqual(restored.image_path, original.image_path)
        np.testing.assert_array_almost_equal(
            restored.internal_model_pose.translation(),
            original.internal_model_pose.translation(),
        )
        np.testing.assert_array_almost_equal(
            restored.internal_model_pose.rotation().matrix(),
            original.internal_model_pose.rotation().matrix(),
        )
        self.assertEqual(len(restored.support_surfaces), 1)
        self.assertEqual(
            str(restored.support_surfaces[0].surface_id), str(surface.surface_id)
        )
        np.testing.assert_array_almost_equal(
            restored.support_surfaces[0].bounding_box_min, surface.bounding_box_min
        )
        np.testing.assert_array_almost_equal(
            restored.support_surfaces[0].bounding_box_max, surface.bounding_box_max
        )
        # Verify link_name is preserved for articulated FK transforms.
        self.assertEqual(restored.support_surfaces[0].link_name, surface.link_name)
        self.assertEqual(restored.metadata, original.metadata)
        np.testing.assert_array_almost_equal(restored.bbox_min, original.bbox_min)
        np.testing.assert_array_almost_equal(restored.bbox_max, original.bbox_max)
        self.assertTrue(restored.immutable)
