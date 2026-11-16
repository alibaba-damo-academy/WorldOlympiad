import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from pydrake.all import MultibodyPlant, RigidTransform, RollPitchYaw, SceneGraph

from scenecode.agent_utils.drake_utils import (
    create_drake_plant_and_scene_graph_from_scene,
)
from scenecode.agent_utils.house import RoomGeometry
from scenecode.agent_utils.room import (
    ObjectType,
    RoomScene,
    SceneObject,
    SupportSurface,
    UniqueID,
    serialize_rigid_transform,
    serialize_rigid_transform,
)


class TestScene(unittest.TestCase):
    """Test cases for RoomScene class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parent.parent / "test_data"
        self.floor_plan_path = self.test_data_dir / "simple_room_geometry.sdf"

        # Create RoomGeometry from SDF file.
        room_geometry_tree = ET.parse(self.floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=self.floor_plan_path,
        )
        self.scene = RoomScene(
            room_geometry=room_geometry, scene_dir=self.test_data_dir
        )

        self.test_object = SceneObject(
            object_id=UniqueID("test_object"),
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=RigidTransform(),
            sdf_path=self.test_data_dir / "simple_box.sdf",
        )

    def test_creation(self):
        """Test RoomScene creation with minimal fields."""
        room_geometry_tree = ET.parse(self.floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=self.floor_plan_path,
        )
        scene = RoomScene(room_geometry=room_geometry, scene_dir=self.test_data_dir)

        self.assertEqual(scene.room_geometry.sdf_path, self.floor_plan_path)
        self.assertEqual(scene.objects, {})
        self.assertEqual(scene.text_description, "")

    def test_add_object(self):
        """Test adding an object to the scene."""
        self.scene.add_object(self.test_object)

        self.assertIn(self.test_object.object_id, self.scene.objects)
        self.assertEqual(
            self.scene.objects[self.test_object.object_id], self.test_object
        )

    def test_remove_object(self):
        """Test removing an object from the scene."""
        self.scene.add_object(self.test_object)

        result = self.scene.remove_object(self.test_object.object_id)

        self.assertTrue(result)
        self.assertNotIn(self.test_object.object_id, self.scene.objects)

    def test_remove_nonexistent_object(self):
        """Test removing a nonexistent object returns False."""
        result = self.scene.remove_object(UniqueID.generate())
        self.assertFalse(result)

    def test_get_object(self):
        """Test getting an object by ID."""
        self.scene.add_object(self.test_object)

        retrieved = self.scene.get_object(self.test_object.object_id)

        self.assertEqual(retrieved, self.test_object)

    def test_get_nonexistent_object(self):
        """Test getting a nonexistent object returns None."""
        result = self.scene.get_object(UniqueID.generate())
        self.assertIsNone(result)

    def test_move_object(self):
        """Test moving an object to a new transform."""
        self.scene.add_object(self.test_object)
        new_transform = RigidTransform(
            RollPitchYaw(0.1, 0.2, 0.3), np.array([1.0, 2.0, 3.0])
        )

        result = self.scene.move_object(self.test_object.object_id, new_transform)

        self.assertTrue(result)
        self.assertEqual(
            self.scene.objects[self.test_object.object_id].transform, new_transform
        )

    def test_move_nonexistent_object(self):
        """Test moving a nonexistent object returns False."""
        result = self.scene.move_object(UniqueID.generate(), RigidTransform())
        self.assertFalse(result)

    def test_get_objects_by_type(self):
        """Test getting all objects of a specific type."""
        furniture_obj = SceneObject(
            object_id=UniqueID("furniture"),
            object_type=ObjectType.FURNITURE,
            name="Furniture",
            description="A furniture object",
            transform=RigidTransform(),
        )

        manipuland_obj = SceneObject(
            object_id=UniqueID("manipuland"),
            object_type=ObjectType.MANIPULAND,
            name="Manipuland",
            description="A manipuland object",
            transform=RigidTransform(),
        )

        self.scene.add_object(furniture_obj)
        self.scene.add_object(manipuland_obj)

        furniture_objects = self.scene.get_objects_by_type(ObjectType.FURNITURE)
        manipuland_objects = self.scene.get_objects_by_type(ObjectType.MANIPULAND)

        self.assertEqual(len(furniture_objects), 1)
        self.assertEqual(len(manipuland_objects), 1)
        self.assertEqual(furniture_objects[0], furniture_obj)
        self.assertEqual(manipuland_objects[0], manipuland_obj)

    def test_to_drake_directive_empty_scene(self):
        """Test generating Drake directive for empty scene."""
        directive = self.scene.to_drake_directive()

        expected_lines = [
            "directives:",
            "- add_model:",
            "    name: room_geometry",
            f"    file: file://{self.floor_plan_path}",
            "- add_weld:",
            "    parent: world",
            "    child: room_geometry::room_geometry_body_link",
        ]

        self.assertEqual(directive, "\n".join(expected_lines))

    def test_to_drake_directive_with_furniture(self):
        """Test generating Drake directive with furniture object."""
        furniture_obj = SceneObject(
            object_id=UniqueID("dining_table_0"),
            object_type=ObjectType.FURNITURE,
            name="Dining Table",
            description="A wooden table",
            transform=RigidTransform(
                RollPitchYaw(0.1, 0.2, 0.3), np.array([1.0, 2.0, 3.0])
            ),
            sdf_path=Path("/path/to/table.sdf"),
        )

        self.scene.add_object(furniture_obj)
        directive = self.scene.to_drake_directive()

        self.assertIn("- add_model:", directive)
        # Model name uses object_id directly.
        self.assertIn("name: dining_table_0", directive)
        self.assertIn("file: file:///path/to/table.sdf", directive)
        self.assertIn("- add_weld:", directive)
        self.assertIn("parent: world", directive)
        # Child references model name with link.
        self.assertIn("child: dining_table_0::base_link", directive)
        self.assertIn("translation: [1.0, 2.0, 3.0]", directive)
        self.assertIn("rotation: !AngleAxis", directive)

    def test_to_drake_directive_with_articulated_internal_model_pose(self):
        """Articulated furniture should export the effective base-link pose."""
        furniture_obj = SceneObject(
            object_id=UniqueID("wardrobe_0"),
            object_type=ObjectType.FURNITURE,
            name="Wardrobe",
            description="An articulated wardrobe",
            transform=RigidTransform(p=np.array([1.0, 2.0, 3.0])),
            internal_model_pose=RigidTransform(
                RollPitchYaw(0.0, 0.0, np.pi),
                np.array([0.0, -0.25, 0.0]),
            ),
            sdf_path=Path("/path/to/wardrobe.sdf"),
        )

        self.scene.add_object(furniture_obj)
        directive = self.scene.to_drake_directive()

        self.assertIn("file: file:///path/to/wardrobe.sdf", directive)
        self.assertIn("translation: [1.0, 1.75, 3.0]", directive)
        self.assertIn("angle_deg: 180.0", directive)

    def test_to_drake_directive_free_body_uses_effective_pose(self):
        """Free-body exports should also include articulated internal model poses."""
        manipuland_obj = SceneObject(
            object_id=UniqueID("cabinet_part"),
            object_type=ObjectType.MANIPULAND,
            name="Cabinet Part",
            description="An articulated cabinet part",
            transform=RigidTransform(p=np.array([0.5, 0.5, 1.0])),
            internal_model_pose=RigidTransform(p=np.array([0.25, -0.1, 0.0])),
            sdf_path=Path("/path/to/cabinet_part.sdf"),
        )

        self.scene.add_object(manipuland_obj)
        directive = self.scene.to_drake_directive()

        self.assertIn("file: file:///path/to/cabinet_part.sdf", directive)
        self.assertIn("default_free_body_pose:", directive)
        self.assertIn("translation: [0.75, 0.4, 1.0]", directive)

    def test_to_drake_directive_stack_member_uses_effective_pose(self):
        """Composite stack members should export base-link poses with internal offsets."""
        stack_obj = SceneObject(
            object_id=UniqueID("stack_0"),
            object_type=ObjectType.MANIPULAND,
            name="stack_2",
            description="A stack",
            transform=RigidTransform(),
            metadata={
                "composite_type": "stack",
                "member_assets": [
                    {
                        "asset_id": "wardrobe_0",
                        "name": "Wardrobe",
                        "transform": serialize_rigid_transform(
                            RigidTransform(p=np.array([1.0, 2.0, 3.0]))
                        ),
                        "internal_model_pose": serialize_rigid_transform(
                            RigidTransform(
                                RollPitchYaw(0.0, 0.0, np.pi),
                                np.array([0.0, -0.25, 0.0]),
                            )
                        ),
                        "sdf_path": "/path/to/wardrobe.sdf",
                    }
                ],
            },
        )

        self.scene.add_object(stack_obj)
        directive = self.scene.to_drake_directive()

        self.assertIn("file: file:///path/to/wardrobe.sdf", directive)
        self.assertIn("translation: [1.0, 1.75, 3.0]", directive)
        self.assertIn("angle_deg: 180.0", directive)

    def test_to_drake_directive_filled_container_members_use_effective_pose(self):
        """Filled container members should export effective articulated poses."""
        filled_obj = SceneObject(
            object_id=UniqueID("filled_container_0"),
            object_type=ObjectType.MANIPULAND,
            name="filled_bin",
            description="A filled bin",
            transform=RigidTransform(),
            metadata={
                "composite_type": "filled_container",
                "container_asset": {
                    "asset_id": "wardrobe_0",
                    "name": "Wardrobe",
                    "transform": serialize_rigid_transform(
                        RigidTransform(p=np.array([1.0, 2.0, 0.0]))
                    ),
                    "internal_model_pose": serialize_rigid_transform(
                        RigidTransform(p=np.array([0.0, -0.25, 0.0]))
                    ),
                    "sdf_path": "/path/to/wardrobe.sdf",
                },
                "fill_assets": [
                    {
                        "asset_id": "cup_0",
                        "name": "Cup",
                        "transform": serialize_rigid_transform(
                            RigidTransform(p=np.array([1.2, 2.0, 0.4]))
                        ),
                        "internal_model_pose": serialize_rigid_transform(
                            RigidTransform(p=np.array([0.1, 0.0, 0.0]))
                        ),
                        "sdf_path": "/path/to/cup.sdf",
                    }
                ],
            },
        )

        self.scene.add_object(filled_obj)
        directive = self.scene.to_drake_directive()

        self.assertIn("file: file:///path/to/wardrobe.sdf", directive)
        self.assertIn("translation: [1.0, 1.75, 0.0]", directive)
        self.assertIn("translation: [0.30000000000000004, 0.25, 0.4]", directive)

    def test_to_drake_directive_pile_member_uses_effective_pose(self):
        """Composite pile members should export base-link poses with internal offsets."""
        pile_obj = SceneObject(
            object_id=UniqueID("pile_0"),
            object_type=ObjectType.MANIPULAND,
            name="pile_2",
            description="A pile",
            transform=RigidTransform(),
            metadata={
                "composite_type": "pile",
                "member_assets": [
                    {
                        "asset_id": "wardrobe_0",
                        "name": "Wardrobe",
                        "transform": serialize_rigid_transform(
                            RigidTransform(p=np.array([1.0, 2.0, 3.0]))
                        ),
                        "internal_model_pose": serialize_rigid_transform(
                            RigidTransform(p=np.array([0.0, -0.25, 0.0]))
                        ),
                        "sdf_path": "/path/to/wardrobe.sdf",
                    }
                ],
            },
        )

        self.scene.add_object(pile_obj)
        directive = self.scene.to_drake_directive()

        self.assertIn("file: file:///path/to/wardrobe.sdf", directive)
        self.assertIn("translation: [1.0, 1.75, 3.0]", directive)

    def test_to_drake_directive_with_manipuland(self):
        """Test generating Drake directive with manipuland object."""
        manipuland_obj = SceneObject(
            object_id=UniqueID("cup"),
            object_type=ObjectType.MANIPULAND,
            name="Coffee Cup",
            description="A ceramic cup",
            transform=RigidTransform(np.array([0.5, 0.5, 1.0])),
            sdf_path=Path("/path/to/cup.sdf"),
        )

        self.scene.add_object(manipuland_obj)
        directive = self.scene.to_drake_directive()

        self.assertIn("- add_model:", directive)
        self.assertIn("name: coffee_cup", directive)
        self.assertIn("file: file:///path/to/cup.sdf", directive)
        self.assertIn("default_free_body_pose:", directive)
        self.assertIn("base_link:", directive)
        self.assertIn("translation: [0.5, 0.5, 1.0]", directive)
        # Manipulands should not be welded (they are free bodies).
        self.assertNotIn("child: coffee_cup", directive)

    def test_to_drake_directive_skips_objects_without_sdf(self):
        """Test that objects without sdf_path are skipped in directive."""
        obj_without_sdf = SceneObject(
            object_id=UniqueID("no_sdf"),
            object_type=ObjectType.FURNITURE,
            name="No SDF Object",
            description="An object without SDF",
            transform=RigidTransform(),
            sdf_path=None,
        )

        self.scene.add_object(obj_without_sdf)
        directive = self.scene.to_drake_directive()

        self.assertNotIn("no_sdf", directive)
        self.assertNotIn("No SDF Object", directive)

    def test_to_drake_directive_loadable_by_drake(self):
        """Test that generated Drake directive can be loaded by Drake."""
        # Create scene with test data.
        test_data_dir = Path(__file__).parent.parent / "test_data"
        floor_plan_path = test_data_dir / "simple_room_geometry.sdf"
        box_sdf_path = test_data_dir / "simple_box.sdf"
        sphere_sdf_path = test_data_dir / "simple_sphere.sdf"

        room_geometry_tree = ET.parse(floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=floor_plan_path,
        )
        scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

        # Add furniture object.
        furniture_obj = SceneObject(
            object_id=UniqueID("test_table"),
            object_type=ObjectType.FURNITURE,
            name="Test Table",
            description="A test table",
            transform=RigidTransform(np.array([1.0, 0.0, 0.5])),
            sdf_path=box_sdf_path,
        )

        # Add manipuland object.
        manipuland_obj = SceneObject(
            object_id=UniqueID("test_sphere"),
            object_type=ObjectType.MANIPULAND,
            name="Test Sphere",
            description="A test sphere",
            transform=RigidTransform(np.array([0.0, 1.0, 1.0])),
            sdf_path=sphere_sdf_path,
        )

        scene.add_object(furniture_obj)
        scene.add_object(manipuland_obj)

        # This should not raise an exception.
        plant, scene_graph = create_drake_plant_and_scene_graph_from_scene(scene)

        # Verify the plant was created successfully.
        self.assertIsInstance(plant, MultibodyPlant)
        self.assertIsInstance(scene_graph, SceneGraph)
        self.assertTrue(plant.is_finalized())

        # Should have floor plan + 2 objects = at least 3 model instances.
        self.assertGreater(plant.num_model_instances(), 2)

    def test_to_drake_directive_with_base_dir_uses_package_uris(self):
        """Test that passing base_dir generates package://scene/ URIs."""
        # Create scene with test data.
        test_data_dir = Path(__file__).parent.parent / "test_data"
        floor_plan_path = test_data_dir / "simple_room_geometry.sdf"
        box_sdf_path = test_data_dir / "simple_box.sdf"

        room_geometry_tree = ET.parse(floor_plan_path)
        room_geometry = RoomGeometry(
            sdf_tree=room_geometry_tree,
            sdf_path=floor_plan_path,
        )
        scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

        # Add furniture object.
        furniture_obj = SceneObject(
            object_id=UniqueID("test_table"),
            object_type=ObjectType.FURNITURE,
            name="Test Table",
            description="A test table",
            transform=RigidTransform(np.array([1.0, 0.0, 0.5])),
            sdf_path=box_sdf_path,
        )
        scene.add_object(furniture_obj)

        # Generate directive with base_dir to get package:// URIs.
        directive = scene.to_drake_directive(base_dir=test_data_dir)

        # Should NOT contain file:// prefix.
        self.assertNotIn("file://", directive)

        # Should contain package://scene/ URIs.
        self.assertIn("file: package://scene/simple_room_geometry.sdf", directive)
        self.assertIn("file: package://scene/simple_box.sdf", directive)

    def test_to_drake_directive_with_parent_frame(self):
        """Test that welded furniture uses parent_frame instead of world."""
        furniture_obj = SceneObject(
            object_id=UniqueID("bed_abc"),
            object_type=ObjectType.FURNITURE,
            name="Bed",
            description="A bed",
            transform=RigidTransform(np.array([0.5, 0.5, 0.0])),
            sdf_path=Path("/path/to/bed.sdf"),
        )

        self.scene.add_object(furniture_obj)
        directive = self.scene.to_drake_directive(
            parent_frame="room_bedroom_frame",
        )

        # Welded object should use room frame as parent.
        self.assertIn("parent: room_bedroom_frame", directive)
        # Should NOT reference world for the weld.
        self.assertNotIn("parent: world", directive)
        # Room-local coordinates (no offset).
        self.assertIn("translation: [0.5, 0.5, 0.0]", directive)

    def test_to_drake_directive_parent_frame_free_bodies_use_base_frame(self):
        """Test that free bodies include base_frame when parent_frame is set."""
        manipuland_obj = SceneObject(
            object_id=UniqueID("cup_xyz"),
            object_type=ObjectType.MANIPULAND,
            name="Cup",
            description="A cup",
            transform=RigidTransform(np.array([0.3, 0.2, 0.8])),
            sdf_path=Path("/path/to/cup.sdf"),
        )

        self.scene.add_object(manipuland_obj)
        directive = self.scene.to_drake_directive(
            parent_frame="room_bedroom_frame",
        )

        # Free body should include base_frame in default_free_body_pose.
        self.assertIn("base_frame: room_bedroom_frame", directive)
        # Room-local coordinates.
        self.assertIn("translation: [0.3, 0.2, 0.8]", directive)

    def test_to_drake_directive_default_parent_frame_uses_world(self):
        """Test that default parent_frame='world' includes base_frame: world."""
        manipuland_obj = SceneObject(
            object_id=UniqueID("ball_456"),
            object_type=ObjectType.MANIPULAND,
            name="Ball",
            description="A ball",
            transform=RigidTransform(np.array([1.0, 2.0, 0.5])),
            sdf_path=Path("/path/to/ball.sdf"),
        )

        self.scene.add_object(manipuland_obj)
        directive = self.scene.to_drake_directive()

        # Default parent_frame="world" should appear in base_frame.
        self.assertIn("base_frame: world", directive)
        self.assertIn("translation: [1.0, 2.0, 0.5]", directive)

    def test_content_hash_identical_scenes_same_hash(self):
        """Test that identical scenes produce identical content hashes."""
        # Create two identical scenes.
        scene1 = RoomScene(
            room_geometry=self.scene.room_geometry, scene_dir=self.test_data_dir
        )
        scene2 = RoomScene(
            room_geometry=self.scene.room_geometry, scene_dir=self.test_data_dir
        )

        # Both should have same hash.
        hash1 = scene1.content_hash()
        hash2 = scene2.content_hash()
        self.assertEqual(hash1, hash2)

        # Add identical objects to both scenes.
        obj1 = SceneObject(
            object_id=UniqueID("test_obj"),
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=RigidTransform(),
        )
        obj2 = SceneObject(
            object_id=obj1.object_id,  # Same ID.
            object_type=ObjectType.FURNITURE,
            name="Test Object",
            description="A test object",
            transform=RigidTransform(),
        )

        scene1.add_object(obj1)
        scene2.add_object(obj2)

        hash1_with_obj = scene1.content_hash()
        hash2_with_obj = scene2.content_hash()
        self.assertEqual(hash1_with_obj, hash2_with_obj)

    def test_content_hash_different_objects_different_hash(self):
        """Test that scenes with different objects have different hashes."""
        scene1 = RoomScene(
            room_geometry=self.scene.room_geometry, scene_dir=self.test_data_dir
        )
        scene2 = RoomScene(
            room_geometry=self.scene.room_geometry, scene_dir=self.test_data_dir
        )

        obj1 = SceneObject(
            object_id=UniqueID("obj1"),
            object_type=ObjectType.FURNITURE,
            name="Object 1",
            description="First object",
            transform=RigidTransform(),
        )

        obj2 = SceneObject(
            object_id=UniqueID("obj2"),
            object_type=ObjectType.FURNITURE,
            name="Object 2",
            description="Second object",
            transform=RigidTransform(),
        )

        scene1.add_object(obj1)
        scene2.add_object(obj2)

        hash1 = scene1.content_hash()
        hash2 = scene2.content_hash()
        self.assertNotEqual(hash1, hash2)

    def test_content_hash_transform_change_different_hash(self):
        """Test that moving objects changes the content hash."""
        scene = RoomScene(
            room_geometry=self.scene.room_geometry, scene_dir=self.test_data_dir
        )
        obj = SceneObject(
            object_id=UniqueID("movable_obj"),
            object_type=ObjectType.FURNITURE,
            name="Movable Object",
            description="An object that moves",
            transform=RigidTransform(),
        )
        scene.add_object(obj)

        hash_before = scene.content_hash()

        # Move the object.
        new_transform = RigidTransform(np.array([1.0, 2.0, 3.0]))
        scene.move_object(obj.object_id, new_transform)

        hash_after = scene.content_hash()
        self.assertNotEqual(hash_before, hash_after)

    def test_content_hash_text_description_affects_hash(self):
        """Test that text description affects content hash."""
        scene1 = RoomScene(
            room_geometry=self.scene.room_geometry,
            text_description="Description 1",
            scene_dir=self.test_data_dir,
        )
        scene2 = RoomScene(
            room_geometry=self.scene.room_geometry,
            text_description="Description 2",
            scene_dir=self.test_data_dir,
        )

        hash1 = scene1.content_hash()
        hash2 = scene2.content_hash()
        self.assertNotEqual(hash1, hash2)

    def test_content_hash_deterministic_across_calls(self):
        """Test that content hash is deterministic across multiple calls."""
        scene = RoomScene(
            room_geometry=self.scene.room_geometry, scene_dir=self.test_data_dir
        )
        obj = SceneObject(
            object_id=UniqueID("consistent_obj"),
            object_type=ObjectType.FURNITURE,
            name="Consistent Object",
            description="An object for consistency testing",
            transform=RigidTransform(np.array([1.0, 2.0, 3.0])),
        )
        scene.add_object(obj)

        # Hash should be same across multiple calls.
        hash1 = scene.content_hash()
        hash2 = scene.content_hash()
        hash3 = scene.content_hash()

        self.assertEqual(hash1, hash2)
        self.assertEqual(hash2, hash3)

    def test_content_hash_object_order_independence(self):
        """Test that object addition order doesn't affect content hash."""
        # Create two scenes and add objects in different orders.
        scene1 = RoomScene(
            room_geometry=self.scene.room_geometry, scene_dir=self.test_data_dir
        )
        scene2 = RoomScene(
            room_geometry=self.scene.room_geometry, scene_dir=self.test_data_dir
        )

        obj_a = SceneObject(
            object_id=UniqueID("obj_a_id"),  # Fixed IDs for determinism.
            object_type=ObjectType.FURNITURE,
            name="Object A",
            description="First object",
            transform=RigidTransform(np.array([1.0, 0.0, 0.0])),
        )

        obj_b = SceneObject(
            object_id=UniqueID("obj_b_id"),  # Fixed IDs for determinism.
            object_type=ObjectType.FURNITURE,
            name="Object B",
            description="Second object",
            transform=RigidTransform(np.array([0.0, 1.0, 0.0])),
        )

        # Add in different orders.
        scene1.add_object(obj_a)
        scene1.add_object(obj_b)

        scene2.add_object(obj_b)
        scene2.add_object(obj_a)

        hash1 = scene1.content_hash()
        hash2 = scene2.content_hash()
        self.assertEqual(hash1, hash2)

    def test_content_hash_empty_scene_consistent(self):
        """Test that empty scenes have consistent content hash."""
        scene1 = RoomScene(
            room_geometry=self.scene.room_geometry, scene_dir=self.test_data_dir
        )
        scene2 = RoomScene(
            room_geometry=self.scene.room_geometry, scene_dir=self.test_data_dir
        )

        hash1 = scene1.content_hash()
        hash2 = scene2.content_hash()
        self.assertEqual(hash1, hash2)

    def test_content_hash_missing_files_handled_gracefully(self):
        """Test that missing SDF files don't crash content hashing."""
        scene = RoomScene(
            room_geometry=self.scene.room_geometry, scene_dir=self.test_data_dir
        )
        obj = SceneObject(
            object_id=UniqueID("obj_with_missing_file"),
            object_type=ObjectType.FURNITURE,
            name="Object With Missing File",
            description="An object with missing SDF",
            transform=RigidTransform(),
            sdf_path=Path("/nonexistent/path/to/file.sdf"),
            geometry_path=Path("/nonexistent/path/to/geometry.obj"),
        )
        scene.add_object(obj)

        # Should not raise exception.
        hash_value = scene.content_hash()
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 64)  # SHA-256 hex length.

    def test_content_hash_support_surfaces_affect_hash(self):
        """Test that support surfaces affect content hash."""
        scene1 = RoomScene(
            room_geometry=self.scene.room_geometry, scene_dir=self.test_data_dir
        )
        scene2 = RoomScene(
            room_geometry=self.scene.room_geometry, scene_dir=self.test_data_dir
        )

        support_surface = SupportSurface(
            surface_id=UniqueID("test_surface"),
            bounding_box_min=np.array([0.0, 0.0, 0.0]),
            bounding_box_max=np.array([1.0, 1.0, 0.1]),
            transform=RigidTransform(),
        )

        obj1 = SceneObject(
            object_id=UniqueID("obj_with_surface"),
            object_type=ObjectType.FURNITURE,
            name="Object With Surface",
            description="An object with support surface",
            transform=RigidTransform(),
            support_surfaces=[support_surface],
        )

        obj2 = SceneObject(
            object_id=obj1.object_id,  # Same object but no support surfaces.
            object_type=ObjectType.FURNITURE,
            name="Object With Surface",
            description="An object with support surface",
            transform=RigidTransform(),
            support_surfaces=[],  # No support surfaces.
        )

        scene1.add_object(obj1)
        scene2.add_object(obj2)

        hash1 = scene1.content_hash()
        hash2 = scene2.content_hash()
        self.assertNotEqual(hash1, hash2)

    def test_to_state_dict(self):
        """Test that to_state_dict returns correct scene state dictionary."""
        # Test empty scene.
        empty_scene = RoomScene(
            room_geometry=self.scene.room_geometry,
            text_description="Empty test scene",
            scene_dir=self.test_data_dir,
        )
        state_dict = empty_scene.to_state_dict()

        # Check structure.
        self.assertIn("objects", state_dict)
        self.assertIn("text_description", state_dict)

        # Check empty scene values.
        self.assertEqual(state_dict["objects"], {})
        self.assertEqual(state_dict["text_description"], "Empty test scene")

        # Test scene with objects.
        self.scene.add_object(self.test_object)

        state_dict = self.scene.to_state_dict()

        # Check objects dict.
        self.assertEqual(len(state_dict["objects"]), 1)
        obj_id = str(self.test_object.object_id)
        self.assertIn(obj_id, state_dict["objects"])

        # Check object data serialization.
        obj_data = state_dict["objects"][obj_id]
        self.assertEqual(obj_data["name"], "Test Object")
        self.assertEqual(obj_data["description"], "A test object")
        self.assertEqual(obj_data["object_id"], obj_id)
        self.assertEqual(obj_data["object_type"], ObjectType.FURNITURE.value)

        # Check transform serialization.
        self.assertIn("transform", obj_data)
        transform_data = obj_data["transform"]
        self.assertIsInstance(transform_data["translation"], list)
        self.assertIsInstance(transform_data["rotation_wxyz"], list)
        self.assertEqual(len(transform_data["translation"]), 3)
        self.assertEqual(len(transform_data["rotation_wxyz"]), 4)  # Quaternion wxyz

        # Check support surfaces serialization.
        self.assertIsInstance(obj_data["support_surfaces"], list)

        # Check metadata and paths.
        self.assertEqual(obj_data["metadata"], {})
        self.assertIsNone(obj_data["geometry_path"])
        self.assertIsNone(obj_data["image_path"])
        self.assertIsInstance(obj_data["sdf_path"], str)
        self.assertEqual(obj_data["immutable"], False)

        # Check text description.
        self.assertEqual(state_dict["text_description"], "")

    def test_to_state_dict_multiple_objects(self):
        """Test to_state_dict with multiple objects."""
        # Create second test object.
        second_object = SceneObject(
            object_id=UniqueID("second_object"),
            name="Second Object",
            description="Another test piece",
            object_type=ObjectType.FURNITURE,
            transform=RigidTransform(p=[5.0, 6.0, 7.0]),
            sdf_path=None,
        )

        # Add both objects.
        self.scene.add_object(self.test_object)
        self.scene.add_object(second_object)

        state_dict = self.scene.to_state_dict()

        # Check objects dict has both objects.
        self.assertEqual(len(state_dict["objects"]), 2)

        # Verify both objects are present with correct IDs.
        obj_ids = set(state_dict["objects"].keys())
        expected_ids = {str(self.test_object.object_id), str(second_object.object_id)}
        self.assertEqual(obj_ids, expected_ids)

        # Verify object names in serialized data.
        obj1_data = state_dict["objects"][str(self.test_object.object_id)]
        obj2_data = state_dict["objects"][str(second_object.object_id)]
        self.assertEqual(obj1_data["name"], "Test Object")
        self.assertEqual(obj2_data["name"], "Second Object")


class TestSceneUniqueIDGeneration(unittest.TestCase):
    """Test cases for RoomScene.generate_unique_id() sequential numbering."""

    def setUp(self):
        """Set up test scene."""
        self.test_data_dir = Path(__file__).parent.parent / "test_data"
        # Create minimal RoomGeometry for testing ID generation.
        room_geometry = RoomGeometry(
            sdf_tree=ET.Element("sdf"),
            sdf_path=None,
        )
        self.scene = RoomScene(
            room_geometry=room_geometry, scene_dir=self.test_data_dir
        )

    def test_first_object_gets_zero_suffix(self):
        """Test that first object of a type gets _0 suffix."""
        object_id = self.scene.generate_unique_id("chair")
        self.assertEqual(str(object_id), "chair_0")

    def test_sequential_numbering(self):
        """Test that subsequent objects get sequential suffixes."""
        # Add first chair (suffix _0).
        chair1_id = self.scene.generate_unique_id("chair")
        chair1 = SceneObject(
            object_id=chair1_id,
            object_type=ObjectType.FURNITURE,
            name="chair",
            description="First chair",
            transform=RigidTransform(),
            sdf_path=None,
        )
        self.scene.add_object(chair1)

        # Add second chair (suffix _1).
        chair2_id = self.scene.generate_unique_id("chair")
        self.assertEqual(str(chair2_id), "chair_1")

    def test_base36_encoding(self):
        """Test that base-36 encoding works (0-9, a-z)."""
        # Add 11 chairs (chair_0 through chair_a).
        for i in range(11):
            chair_id = self.scene.generate_unique_id("chair")
            chair = SceneObject(
                object_id=chair_id,
                object_type=ObjectType.FURNITURE,
                name="chair",
                description=f"Chair {i+1}",
                transform=RigidTransform(),
                sdf_path=None,
            )
            self.scene.add_object(chair)

        # 12th chair should use 'b' (base-36 for 11).
        chair12_id = self.scene.generate_unique_id("chair")
        self.assertEqual(str(chair12_id), "chair_b")

    def test_different_types_independent(self):
        """Test that different object types have independent numbering."""
        chair_id = self.scene.generate_unique_id("chair")
        table_id = self.scene.generate_unique_id("table")

        # Both should start with suffix _0 (first of their type).
        self.assertEqual(str(chair_id), "chair_0")
        self.assertEqual(str(table_id), "table_0")


class TestRoomGeometry(unittest.TestCase):
    """Test cases for RoomGeometry serialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parent.parent / "test_data"
        self.sdf_path = self.test_data_dir / "simple_room_geometry.sdf"

    def test_to_dict_minimal(self):
        """Test RoomGeometry serialization with minimal fields."""
        sdf_tree = ET.parse(self.sdf_path)
        room_geometry = RoomGeometry(
            sdf_tree=sdf_tree,
            sdf_path=self.sdf_path,
            walls=[],
            floor=None,
            wall_normals={},
            width=5.0,
            length=6.0,
        )

        room_geometry_dict = room_geometry.to_dict()

        # Check basic fields.
        self.assertEqual(room_geometry_dict["width"], 5.0)
        self.assertEqual(room_geometry_dict["length"], 6.0)
        self.assertIsNone(room_geometry_dict["floor"])
        self.assertEqual(room_geometry_dict["sdf_path"], str(self.sdf_path))

    def test_to_dict_with_scene_dir(self):
        """Test RoomGeometry serialization with path relativization."""
        sdf_tree = ET.parse(self.sdf_path)

        # Create paths within a scene directory.
        scene_dir = Path("/tmp/scene")
        sdf_path = scene_dir / "room_geometry.sdf"

        room_geometry = RoomGeometry(
            sdf_tree=sdf_tree,
            sdf_path=sdf_path,
            walls=[],
            floor=None,
            wall_normals={},
            width=5.0,
            length=6.0,
        )

        room_geometry_dict = room_geometry.to_dict(scene_dir=scene_dir)

        # Paths should be relative.
        self.assertEqual(room_geometry_dict["sdf_path"], "room_geometry.sdf")

    def test_to_dict_with_floor(self):
        """Test RoomGeometry serialization with floor object."""
        sdf_tree = ET.parse(self.sdf_path)

        floor_obj = SceneObject(
            object_id=UniqueID.generate(),
            object_type=ObjectType.FLOOR,
            name="Floor",
            description="Floor object",
            transform=RigidTransform(),
        )

        room_geometry = RoomGeometry(
            sdf_tree=sdf_tree,
            sdf_path=self.sdf_path,
            walls=[],
            floor=floor_obj,
            wall_normals={},
            width=5.0,
            length=6.0,
        )

        room_geometry_dict = room_geometry.to_dict()

        # Floor should be serialized.
        self.assertIsNotNone(room_geometry_dict["floor"])
        self.assertEqual(room_geometry_dict["floor"]["name"], "Floor")
        self.assertEqual(
            room_geometry_dict["floor"]["object_type"], ObjectType.FLOOR.value
        )

    def test_from_dict_minimal(self):
        """Test RoomGeometry deserialization with minimal fields."""
        room_geometry_dict = {
            "sdf_path": str(self.sdf_path),
            "width": 5.0,
            "length": 6.0,
            "floor": None,
        }

        room_geometry = RoomGeometry.from_dict(room_geometry_dict)

        # Check fields.
        self.assertEqual(room_geometry.width, 5.0)
        self.assertEqual(room_geometry.length, 6.0)
        self.assertEqual(room_geometry.sdf_path, self.sdf_path)
        self.assertIsNone(room_geometry.floor)
        self.assertEqual(room_geometry.walls, [])
        self.assertEqual(room_geometry.wall_normals, {})

        # sdf_tree should be re-parsed from file.
        self.assertIsNotNone(room_geometry.sdf_tree)

    def test_from_dict_with_scene_dir(self):
        """Test RoomGeometry deserialization with path resolution."""
        # Copy test file to a temp location for this test.

        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir)
            test_sdf = scene_dir / "room_geometry.sdf"

            # Copy test file.
            shutil.copy(self.sdf_path, test_sdf)

            room_geometry_dict = {
                "sdf_path": "room_geometry.sdf",
                "width": 5.0,
                "length": 6.0,
                "floor": None,
            }

            room_geometry = RoomGeometry.from_dict(
                room_geometry_dict, scene_dir=scene_dir
            )

            # Paths should be resolved relative to scene_dir.
            self.assertEqual(room_geometry.sdf_path, test_sdf)
            self.assertIsNotNone(room_geometry.sdf_tree)

    def test_serialization_roundtrip_minimal(self):
        """Test RoomGeometry serialization roundtrip with minimal fields."""
        sdf_tree = ET.parse(self.sdf_path)
        original = RoomGeometry(
            sdf_tree=sdf_tree,
            sdf_path=self.sdf_path,
            walls=[],
            floor=None,
            wall_normals={},
            width=5.0,
            length=6.0,
        )

        # Serialize and deserialize.
        room_geometry_dict = original.to_dict()
        restored = RoomGeometry.from_dict(room_geometry_dict)

        # Check equality.
        self.assertEqual(restored.width, original.width)
        self.assertEqual(restored.length, original.length)
        self.assertEqual(restored.sdf_path, original.sdf_path)
        self.assertIsNone(restored.floor)
        self.assertIsNotNone(restored.sdf_tree)

    def test_serialization_roundtrip_with_floor(self):
        """Test RoomGeometry serialization roundtrip with floor object."""

        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir)
            test_sdf = scene_dir / "room_geometry.sdf"

            # Copy test files.
            shutil.copy(self.sdf_path, test_sdf)

            sdf_tree = ET.parse(test_sdf)
            floor_obj = SceneObject(
                object_id=UniqueID.generate(),
                object_type=ObjectType.FLOOR,
                name="Floor",
                description="Floor object",
                transform=RigidTransform(p=np.array([1.0, 2.0, 0.0])),
            )

            original = RoomGeometry(
                sdf_tree=sdf_tree,
                sdf_path=test_sdf,
                walls=[],
                floor=floor_obj,
                wall_normals={},
                width=5.0,
                length=6.0,
            )

            # Serialize and deserialize with scene_dir.
            room_geometry_dict = original.to_dict(scene_dir=scene_dir)
            restored = RoomGeometry.from_dict(room_geometry_dict, scene_dir=scene_dir)

            # Check all fields.
            self.assertEqual(restored.width, original.width)
            self.assertEqual(restored.length, original.length)
            self.assertEqual(restored.sdf_path, original.sdf_path)
            self.assertIsNotNone(restored.floor)
            self.assertEqual(restored.floor.name, original.floor.name)
            self.assertEqual(restored.floor.object_type, original.floor.object_type)
            np.testing.assert_array_almost_equal(
                restored.floor.transform.translation(),
                original.floor.transform.translation(),
            )

    def test_wall_normals_serialization_roundtrip(self):
        """Test wall_normals survives serialization roundtrip."""

        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir)
            test_sdf = scene_dir / "room_geometry.sdf"

            # Copy test file.
            shutil.copy(self.sdf_path, test_sdf)

            sdf_tree = ET.parse(test_sdf)

            # Create floor plan with wall_normals.
            wall_normals = {
                "left_wall": np.array([1.0, 0.0]),
                "right_wall": np.array([-1.0, 0.0]),
                "back_wall": np.array([0.0, 1.0]),
                "front_wall": np.array([0.0, -1.0]),
            }

            original = RoomGeometry(
                sdf_tree=sdf_tree,
                sdf_path=test_sdf,
                walls=[],
                floor=None,
                wall_normals=wall_normals,
                width=5.0,
                length=6.0,
            )

            # Serialize and deserialize.
            room_geometry_dict = original.to_dict(scene_dir=scene_dir)
            restored = RoomGeometry.from_dict(room_geometry_dict, scene_dir=scene_dir)

            # Verify wall_normals were preserved.
            self.assertEqual(len(restored.wall_normals), 4)
            for wall_name, expected_normal in wall_normals.items():
                self.assertIn(wall_name, restored.wall_normals)
                np.testing.assert_array_almost_equal(
                    restored.wall_normals[wall_name], expected_normal
                )

    def test_from_dict_missing_sdf_raises_error(self):
        """Test RoomGeometry.from_dict() raises ValueError on missing SDF file."""
        floor_plan_dict = {
            "sdf_path": "nonexistent.sdf",
            "width": 5.0,
            "length": 6.0,
            "floor": None,
            "wall_normals": {},
        }

        # Should raise ValueError, not just warn.
        with self.assertRaises(ValueError) as context:
            RoomGeometry.from_dict(floor_plan_dict)

        self.assertIn("SDF file not found", str(context.exception))


class TestSceneRoomGeometryIntegration(unittest.TestCase):
    """Integration tests for RoomScene serialization with floor plan."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parent.parent / "test_data"
        self.sdf_path = self.test_data_dir / "simple_room_geometry.sdf"

    def test_scene_serialization_with_floor_plan_and_walls(self):
        """
        Integration test: RoomScene serialization includes floor plan, and walls
        are correctly populated from scene.objects after restoration.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir)
            test_sdf = scene_dir / "room_geometry.sdf"

            # Copy test files.
            shutil.copy(self.sdf_path, test_sdf)

            # Create floor plan with floor object and wall_normals.
            sdf_tree = ET.parse(test_sdf)
            floor_obj = SceneObject(
                object_id=UniqueID.generate(),
                object_type=ObjectType.FLOOR,
                name="Floor",
                description="Test floor",
                transform=RigidTransform(),
            )

            wall_normals = {
                "left_wall": np.array([1.0, 0.0]),
                "right_wall": np.array([-1.0, 0.0]),
            }

            room_geometry = RoomGeometry(
                sdf_tree=sdf_tree,
                sdf_path=test_sdf,
                walls=[],  # Will be populated after Scene restoration
                floor=floor_obj,
                wall_normals=wall_normals,
                width=5.0,
                length=6.0,
            )

            # Create scene with floor plan and wall objects.
            scene = RoomScene(
                room_geometry=room_geometry,
                scene_dir=scene_dir,
                text_description="Test scene with floor plan",
            )

            # Add wall objects to scene.
            wall1 = SceneObject(
                object_id=UniqueID.generate(),
                object_type=ObjectType.WALL,
                name="Wall1",
                description="Test wall",
                transform=RigidTransform(),
            )
            wall2 = SceneObject(
                object_id=UniqueID.generate(),
                object_type=ObjectType.WALL,
                name="Wall2",
                description="Another test wall",
                transform=RigidTransform(),
            )
            scene.add_object(wall1)
            scene.add_object(wall2)

            # Manually populate room_geometry.walls (normally done by furniture agent).
            room_geometry.walls = [wall1, wall2]

            # Serialize scene.
            state_dict = scene.to_state_dict()

            # Verify room_geometry is in state_dict.
            self.assertIn("room_geometry", state_dict)
            self.assertIsNotNone(state_dict["room_geometry"])
            self.assertEqual(state_dict["room_geometry"]["width"], 5.0)
            self.assertEqual(state_dict["room_geometry"]["length"], 6.0)
            self.assertIsNotNone(state_dict["room_geometry"]["floor"])
            self.assertEqual(len(state_dict["room_geometry"]["wall_normals"]), 2)

            # Create new scene and restore.
            restored_scene = RoomScene(
                room_geometry=None,
                scene_dir=scene_dir,
                text_description="",
            )
            restored_scene.restore_from_state_dict(state_dict)

            # Verify floor plan was restored.
            self.assertIsNotNone(restored_scene.room_geometry)
            self.assertEqual(restored_scene.room_geometry.width, 5.0)
            self.assertEqual(restored_scene.room_geometry.length, 6.0)

            # Verify floor object was restored.
            self.assertIsNotNone(restored_scene.room_geometry.floor)
            self.assertEqual(restored_scene.room_geometry.floor.name, "Floor")

            # Verify wall_normals were restored.
            self.assertEqual(len(restored_scene.room_geometry.wall_normals), 2)
            np.testing.assert_array_almost_equal(
                restored_scene.room_geometry.wall_normals["left_wall"],
                np.array([1.0, 0.0]),
            )
            np.testing.assert_array_almost_equal(
                restored_scene.room_geometry.wall_normals["right_wall"],
                np.array([-1.0, 0.0]),
            )

            # CRITICAL: Verify walls were populated from scene.objects.
            self.assertEqual(len(restored_scene.room_geometry.walls), 2)
            wall_names = {w.name for w in restored_scene.room_geometry.walls}
            self.assertEqual(wall_names, {"Wall1", "Wall2"})

            # Verify walls in room_geometry.walls are the same objects as in scene.objects.
            for wall in restored_scene.room_geometry.walls:
                self.assertIn(wall.object_id, restored_scene.objects)
                self.assertIs(wall, restored_scene.objects[wall.object_id])


if __name__ == "__main__":
    unittest.main()
