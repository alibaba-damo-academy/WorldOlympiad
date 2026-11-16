import asyncio
import json
import math
import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock

import numpy as np

from omegaconf import OmegaConf
from pydrake.all import RigidTransform, RollPitchYaw

from scenecode.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID
from scenecode.furniture_agents.tools.furniture_tools import FurnitureTools
from scenecode.furniture_agents.tools.response_dataclasses import (
    FurnitureErrorType,
    Position3D,
    Rotation3D,
    SceneObjectInfo,
    SceneStateResult,
)
from scenecode.furniture_agents.tools.scene_tools import SceneTools


class BaseAgentToolsTest(unittest.TestCase):
    """Base class for agent tools tests with common setup/teardown."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_scene(self, objects=None, description="Test scene"):
        """Create a standard mock scene for testing."""
        mock_scene = Mock(spec=RoomScene)
        mock_scene.objects = objects or {}
        mock_scene.text_description = description
        mock_scene.action_log_path = None
        return mock_scene

    def create_mock_scene_object(
        self,
        name: str,
        object_type: ObjectType,
        position: list[float] = None,
        rotation: list[float] = None,
        sdf_path: str = None,
        geometry_path: str = None,
    ):
        """Create a minimal mock scene object with transform data.

        Args:
            name: Object name (e.g., "Table")
            object_type: ObjectType enum value
            position: [x, y, z] coordinates (defaults to [0, 0, 0])
            rotation: [roll, pitch, yaw] angles in radians (defaults to [0, 0, 0])
            sdf_path: Optional SDF file path
            geometry_path: Optional geometry file path

        Returns:
            Mock object with real Drake transform for testing.
        """
        position = position or [0.0, 0.0, 0.0]
        rotation = rotation or [0.0, 0.0, 0.0]

        mock = Mock()
        mock.name = name
        mock.object_type = object_type
        mock.description = f"Test {name.lower()}"
        mock.sdf_path = Path(sdf_path) if sdf_path else None
        mock.geometry_path = Path(geometry_path) if geometry_path else None

        # Use real Drake RigidTransform for compatibility with SimplifiedFurnitureInfo.
        rpy = RollPitchYaw(rotation[0], rotation[1], rotation[2])
        mock.transform = RigidTransform(rpy, position)

        # Add bounding box fields.
        mock.bbox_min = None
        mock.bbox_max = None

        # Add immutable field (defaults to False for regular furniture).
        mock.immutable = False

        return mock


class TestSceneTools(BaseAgentToolsTest):
    """Test SceneTools class contracts."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.mock_scene = self.create_mock_scene()

        # Load base configuration from actual config file.
        config_path = (
            Path(__file__).parent.parent.parent
            / "configs/furniture_agent/base_furniture_agent.yaml"
        )
        self.cfg = OmegaConf.load(config_path)

        self.scene_tools = SceneTools(scene=self.mock_scene, cfg=self.cfg)

    def test_scene_tools_initialization(self):
        """Test SceneTools initializes properly."""
        self.assertIsNotNone(self.scene_tools)
        self.assertEqual(self.scene_tools.scene, self.mock_scene)

    def test_get_current_scene_state_returns_json(self):
        """Unit test: SceneTools extracts object data and returns JSON."""
        # Create mock scene objects using helper function.
        table_mock = self.create_mock_scene_object(
            name="Table",
            object_type=ObjectType.FURNITURE,
            position=[1.0, 2.0, 3.0],
            rotation=[0.0, 0.0, math.radians(90.0)],
            sdf_path="/path/to/table.sdf",
            geometry_path="/path/to/table.obj",
        )
        table_mock.description = "A wooden table"  # Override generated description

        chair_mock = self.create_mock_scene_object(
            name="Chair",
            object_type=ObjectType.FURNITURE,
            position=[4.0, 5.0, 6.0],
            rotation=[0.0, 0.0, 0.0],
        )
        chair_mock.description = "A comfortable chair"  # Override generated description

        objects = {
            "table_1": table_mock,
            "chair_1": chair_mock,
        }
        self.mock_scene.objects = objects

        # Test the tool function directly.
        result = self.scene_tools._get_current_scene_impl()

        # Verify JSON-like result is returned.
        self.assertIsInstance(result, str)
        # The result should contain furniture information.
        self.assertIn("furniture", result.lower())

    def test_scene_state_result_serialization(self):
        """SceneStateResult DTOs serialize to valid JSON."""
        objects = [
            SceneObjectInfo(
                object_id="table_1",
                description="A wooden dining table",
                position=Position3D(x=1.0, y=2.0, z=3.0),
                rotation=Rotation3D(roll=0.0, pitch=0.0, yaw=90.0),
                object_type="FURNITURE",
                dimensions=None,
                world_bounds=None,
                immutable=False,
            ),
            SceneObjectInfo(
                object_id="chair_1",
                description="A wooden dining chair",
                position=Position3D(x=4.0, y=5.0, z=6.0),
                rotation=Rotation3D(roll=0.0, pitch=0.0, yaw=0.0),
                object_type="FURNITURE",
                dimensions=None,
                world_bounds=None,
                immutable=False,
            ),
        ]

        result = SceneStateResult(success=True, furniture_count=2, objects=objects)

        # Test JSON serialization.
        json_str = result.to_json()
        self.assertIsInstance(json_str, str)

        # Verify JSON content structure and values.
        parsed = json.loads(json_str)
        self.assertEqual(parsed["success"], True)
        self.assertEqual(parsed["furniture_count"], 2)
        self.assertEqual(len(parsed["objects"]), 2)

        # Check first object details.
        table_obj = parsed["objects"][0]
        self.assertEqual(table_obj["object_id"], "table_1")
        self.assertEqual(table_obj["object_type"], "FURNITURE")
        self.assertEqual(table_obj["position"]["x"], 1.0)
        self.assertEqual(table_obj["rotation"]["yaw"], 90.0)
        self.assertEqual(table_obj["immutable"], False)

        # Check second object details.
        chair_obj = parsed["objects"][1]
        self.assertEqual(chair_obj["object_id"], "chair_1")
        self.assertEqual(chair_obj["object_type"], "FURNITURE")
        self.assertIsNone(chair_obj["dimensions"])

        # Verify the JSON contains expected keywords.
        self.assertIn("furniture", json_str.lower())

    def test_tools_dictionary_exposed(self):
        """Test that tools dictionary is properly exposed."""
        # Verify tools dictionary exists and contains expected tools.
        self.assertTrue(hasattr(self.scene_tools, "tools"))
        self.assertIsInstance(self.scene_tools.tools, dict)

        # Should have at least the get_current_scene_state.
        self.assertIn("get_current_scene_state", self.scene_tools.tools)

    def test_tools_are_callable(self):
        """Test that SceneTools follow standard tool interface."""
        for tool_name, tool_func in self.scene_tools.tools.items():
            # FunctionTool objects are not directly callable, but they have an invoke
            # method.
            self.assertTrue(
                hasattr(tool_func, "on_invoke_tool"),
                f"SceneTools tool {tool_name} should have on_invoke_tool method",
            )

    def test_tool_invocation(self):
        """Test that SceneTools can be invoked and return appropriate types."""
        get_scene_tool = self.scene_tools.tools["get_current_scene_state"]

        # Provide minimal context and input for FunctionTool.
        mock_ctx = Mock()
        result = asyncio.run(get_scene_tool.on_invoke_tool(mock_ctx, {}))

        self.assertIsInstance(result, str)
        json.loads(result)  # Will raise if invalid JSON.

    def test_error_handling(self):
        """Test that SceneTools handle errors gracefully."""
        get_scene_tool = self.scene_tools.tools["get_current_scene_state"]

        try:
            mock_ctx = Mock()
            result = asyncio.run(get_scene_tool.on_invoke_tool(mock_ctx, {}))
            # Should return some form of response.
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Tool raised unhandled exception: {e}")


class TestFurnitureTools(BaseAgentToolsTest):
    """Test FurnitureTools class contracts."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.mock_scene = self.create_mock_scene()
        self.mock_asset_manager = Mock()

        # Load base and specific configs from actual config files.
        base_config_path = (
            Path(__file__).parent.parent.parent
            / "configs/furniture_agent/base_furniture_agent.yaml"
        )
        specific_config_path = (
            Path(__file__).parent.parent.parent
            / "configs/furniture_agent/stateful_furniture_agent.yaml"
        )
        base_config = OmegaConf.load(base_config_path)
        specific_config = OmegaConf.load(specific_config_path)

        # First merge base with specific config.
        merged_config = OmegaConf.merge(base_config, specific_config)

        # Define test overrides for fast testing.
        test_overrides = {
            "openai": {
                "model": "gpt-4o-mini",  # Cheaper model for testing
                "reasoning_effort": {
                    "planner": "low",  # Faster for tests
                    "designer": "low",
                    "critic": "low",
                },
                "verbosity": {
                    "planner": "low",
                    "designer": "low",
                    "critic": "low",
                },
            },
            "loop_detection": {
                "enabled": False,  # Disable loop detection for unit tests
            },
        }
        # Merge configs (base config provides all other values).
        self.test_config = OmegaConf.merge(merged_config, test_overrides)

        self.furniture_tools = FurnitureTools(
            scene=self.mock_scene,
            asset_manager=self.mock_asset_manager,
            cfg=self.test_config,
        )

    def test_furniture_tools_initialization(self):
        """Test FurnitureTools initializes properly."""
        self.assertIsNotNone(self.furniture_tools)
        self.assertEqual(self.furniture_tools.scene, self.mock_scene)
        self.assertEqual(self.furniture_tools.asset_manager, self.mock_asset_manager)

    def test_tools_dictionary_exposed(self):
        """Test that tools dictionary is properly exposed."""
        self.assertTrue(hasattr(self.furniture_tools, "tools"))
        self.assertIsInstance(self.furniture_tools.tools, dict)

        # Should have furniture-related tools.
        tools_keys = list(self.furniture_tools.tools.keys())
        self.assertGreater(len(tools_keys), 0, "Should have at least one tool")

    def test_tools_are_callable(self):
        """Test that FurnitureTools follow standard tool interface."""
        for tool_name, tool_func in self.furniture_tools.tools.items():
            self.assertTrue(
                hasattr(tool_func, "on_invoke_tool"),
                f"FurnitureTools tool {tool_name} should have on_invoke_tool method",
            )

    def test_scene_modification(self):
        """Test that FurnitureTools provide scene modification capabilities."""
        self.assertTrue(
            len(self.furniture_tools.tools) > 0,
            "FurnitureTools should expose furniture manipulation tools",
        )

    def test_add_furniture_preserves_internal_model_pose(self):
        """Placed furniture should keep articulated internal model pose."""
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0
        self.mock_scene.room_geometry.width = 10.0

        asset_id = UniqueID("wardrobe_asset_0")
        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = asset_id
        mock_asset.name = "Wardrobe"
        mock_asset.description = "A wardrobe"
        mock_asset.object_type = ObjectType.FURNITURE
        mock_asset.geometry_path = Path("/tmp/wardrobe.glb")
        mock_asset.sdf_path = Path("/tmp/wardrobe.sdf")
        mock_asset.image_path = None
        mock_asset.metadata = {}
        mock_asset.bbox_min = np.array([0.0, 0.0, 0.0])
        mock_asset.bbox_max = np.array([1.0, 0.5, 2.0])
        mock_asset.scale_factor = 1.0
        mock_asset.support_surfaces = []
        mock_asset.immutable = False
        mock_asset.internal_model_pose = RigidTransform(
            RollPitchYaw(0.0, 0.0, np.pi),
            [0.0, -0.25, 0.0],
        )
        mock_asset.transform = RigidTransform()

        self.mock_asset_manager.get_asset_by_id = Mock(return_value=mock_asset)
        self.mock_asset_manager.list_available_assets.return_value = [mock_asset]
        self.mock_scene.generate_unique_id = Mock(return_value=UniqueID("wardrobe_0"))

        result_json = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(asset_id),
            x=1.0,
            y=2.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=90.0,
        )

        self.assertTrue(json.loads(result_json)["success"])
        placed_obj = self.mock_scene.add_object.call_args[0][0]
        np.testing.assert_array_almost_equal(
            placed_obj.internal_model_pose.translation(),
            mock_asset.internal_model_pose.translation(),
        )
        np.testing.assert_array_almost_equal(
            placed_obj.internal_model_pose.rotation().matrix(),
            mock_asset.internal_model_pose.rotation().matrix(),
        )

    def test_multiple_furniture_placements_unique_ids(self):
        """Test that multiple placements of the same asset create unique object IDs."""

        # Mock floor plan for bounds checking.
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0
        self.mock_scene.room_geometry.width = 10.0

        # Mock a scene object to return from the asset registry.
        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = UniqueID("test_chair_gen")
        mock_asset.name = "Chair"
        mock_asset.description = "Test chair"
        mock_asset.object_type = ObjectType.FURNITURE
        mock_asset.geometry_path = Path("/test/chair.obj")
        mock_asset.sdf_path = Path("/test/chair.sdf")
        mock_asset.image_path = None
        mock_asset.support_surfaces = []
        mock_asset.metadata = {}
        mock_asset.transform = RigidTransform()

        # Mock asset manager to return this asset.
        self.mock_asset_manager.get_asset_by_id.return_value = mock_asset
        self.mock_asset_manager.list_available_assets.return_value = [mock_asset]

        # Mock scene.add_object to capture what gets added.
        added_objects = []

        def capture_add_object(obj):
            added_objects.append(obj)

        self.mock_scene.add_object = capture_add_object

        # Mock generate_unique_id to return unique IDs for each call.
        self.mock_scene.generate_unique_id.side_effect = [
            UniqueID("chair"),
            UniqueID("chair_2"),
        ]

        # First placement.
        result1_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=0, y=0, z=0
        )

        # Second placement.
        result2_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=2, y=0, z=0
        )

        # Parse results.
        result1 = json.loads(result1_str)
        result2 = json.loads(result2_str)

        # Both should succeed.
        self.assertTrue(result1["success"], f"First placement failed: {result1}")
        self.assertTrue(result2["success"], f"Second placement failed: {result2}")

        # Object IDs should be different.
        obj_id1 = result1["object_id"]
        obj_id2 = result2["object_id"]
        self.assertNotEqual(
            obj_id1, obj_id2, "Multiple placements should create unique object IDs"
        )

        # Should have added two different objects to scene.
        self.assertEqual(len(added_objects), 2)
        self.assertNotEqual(
            str(added_objects[0].object_id), str(added_objects[1].object_id)
        )

    def test_asset_id_not_found_error_handling(self):
        """Test error handling when asset_id doesn't exist in registry."""
        # Mock asset manager to return None (asset not found).
        self.mock_asset_manager.get_asset_by_id.return_value = None
        self.mock_asset_manager.list_available_assets.return_value = []

        result_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id="nonexistent_asset_id", x=0, y=0, z=0
        )

        result = json.loads(result_str)
        self.assertFalse(result["success"])
        self.assertIn("not found in registry", result["message"])

    def test_immutable_objects_cannot_be_moved_or_removed(self):
        """Test that immutable objects (walls) reject move/remove operations."""
        # Create immutable object.
        immutable_obj = SceneObject(
            object_id=UniqueID("wall"),
            object_type=ObjectType.WALL,
            name="Test Wall",
            description="An immutable wall",
            transform=RigidTransform(),
            immutable=True,
        )

        self.mock_scene.objects = {immutable_obj.object_id: immutable_obj}
        self.mock_scene.get_object.return_value = immutable_obj

        # Test: Cannot move immutable objects.
        move_result = json.loads(
            self.furniture_tools._move_furniture_impl(
                object_id=str(immutable_obj.object_id),
                x=1,
                y=1,
                z=0,
                roll=0,
                pitch=0,
                yaw=0,
            )
        )
        self.assertFalse(move_result["success"])
        self.assertEqual(
            move_result["error_type"], FurnitureErrorType.IMMUTABLE_OBJECT.value
        )

        # Test: Cannot remove immutable objects.
        remove_result = json.loads(
            self.furniture_tools._remove_furniture_impl(
                object_id=str(immutable_obj.object_id)
            )
        )
        self.assertFalse(remove_result["success"])
        self.assertEqual(
            remove_result["error_type"], FurnitureErrorType.IMMUTABLE_OBJECT.value
        )

    def test_add_furniture_out_of_bounds_x_positive(self):
        """Test placement fails when X coordinate exceeds positive floor boundary."""
        # Mock floor plan with 10m x 8m dimensions.
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0  # X: [-5, 5]
        self.mock_scene.room_geometry.width = 8.0  # Y: [-4, 4]

        # Mock asset.
        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = UniqueID("test_chair")
        mock_asset.name = "Chair"
        self.mock_asset_manager.get_asset_by_id.return_value = mock_asset

        # Attempt placement at X=6.0 (exceeds max X=5.0).
        result_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=6.0, y=0.0, z=0.0
        )

        result = json.loads(result_str)
        self.assertFalse(result["success"])
        self.assertEqual(
            result["error_type"], FurnitureErrorType.POSITION_OUT_OF_BOUNDS.value
        )
        self.assertIn("out of floor plan bounds", result["message"])
        self.assertIn("X=[-5.000, 5.000]", result["message"])

    def test_add_furniture_out_of_bounds_x_negative(self):
        """Test placement fails when X coordinate exceeds negative floor boundary."""
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0  # X: [-5, 5]
        self.mock_scene.room_geometry.width = 8.0  # Y: [-4, 4]

        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = UniqueID("test_table")
        mock_asset.name = "Table"
        self.mock_asset_manager.get_asset_by_id.return_value = mock_asset

        # Attempt placement at X=-5.5 (exceeds min X=-5.0).
        result_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=-5.5, y=0.0, z=0.0
        )

        result = json.loads(result_str)
        self.assertFalse(result["success"])
        self.assertEqual(
            result["error_type"], FurnitureErrorType.POSITION_OUT_OF_BOUNDS.value
        )

    def test_add_furniture_out_of_bounds_y_positive(self):
        """Test placement fails when Y coordinate exceeds positive floor boundary."""
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0  # X: [-5, 5]
        self.mock_scene.room_geometry.width = 8.0  # Y: [-4, 4]

        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = UniqueID("test_sofa")
        mock_asset.name = "Sofa"
        self.mock_asset_manager.get_asset_by_id.return_value = mock_asset

        # Attempt placement at Y=4.5 (exceeds max Y=4.0).
        result_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=0.0, y=4.5, z=0.0
        )

        result = json.loads(result_str)
        self.assertFalse(result["success"])
        self.assertEqual(
            result["error_type"], FurnitureErrorType.POSITION_OUT_OF_BOUNDS.value
        )
        self.assertIn("Y=[-4.000, 4.000]", result["message"])

    def test_add_furniture_out_of_bounds_y_negative(self):
        """Test placement fails when Y coordinate exceeds negative floor boundary."""
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0  # X: [-5, 5]
        self.mock_scene.room_geometry.width = 8.0  # Y: [-4, 4]

        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = UniqueID("test_lamp")
        mock_asset.name = "Lamp"
        self.mock_asset_manager.get_asset_by_id.return_value = mock_asset

        # Attempt placement at Y=-4.1 (exceeds min Y=-4.0).
        result_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=0.0, y=-4.1, z=0.0
        )

        result = json.loads(result_str)
        self.assertFalse(result["success"])
        self.assertEqual(
            result["error_type"], FurnitureErrorType.POSITION_OUT_OF_BOUNDS.value
        )

    def test_add_furniture_at_boundary(self):
        """Test placement succeeds when exactly at floor boundary."""
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0  # X: [-5, 5]
        self.mock_scene.room_geometry.width = 8.0  # Y: [-4, 4]

        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = UniqueID("test_chair")
        mock_asset.name = "Chair"
        mock_asset.description = "Test chair"
        mock_asset.object_type = ObjectType.FURNITURE
        mock_asset.geometry_path = Path("/test/chair.obj")
        mock_asset.sdf_path = Path("/test/chair.sdf")
        mock_asset.image_path = None
        mock_asset.support_surfaces = []
        mock_asset.metadata = {}
        mock_asset.transform = RigidTransform()

        self.mock_asset_manager.get_asset_by_id.return_value = mock_asset
        self.mock_scene.generate_unique_id.return_value = UniqueID("chair")
        self.mock_scene.add_object = Mock()

        # Placement exactly at boundary (X=5.0, Y=4.0) should succeed.
        result_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=5.0, y=4.0, z=0.0
        )

        result = json.loads(result_str)
        self.assertTrue(result["success"], f"Boundary placement failed: {result}")

    def test_move_furniture_out_of_bounds(self):
        """Test moving furniture to out-of-bounds position fails."""
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0  # X: [-5, 5]
        self.mock_scene.room_geometry.width = 8.0  # Y: [-4, 4]

        # Create movable furniture object.
        furniture_obj = SceneObject(
            object_id=UniqueID("chair"),
            object_type=ObjectType.FURNITURE,
            name="Test Chair",
            description="A movable chair",
            transform=RigidTransform(),
            immutable=False,
        )

        self.mock_scene.get_object.return_value = furniture_obj

        # Attempt to move to out-of-bounds position (X=7.0).
        result_str = self.furniture_tools._move_furniture_impl(
            object_id=str(furniture_obj.object_id),
            x=7.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
        )

        result = json.loads(result_str)
        self.assertFalse(result["success"])
        self.assertEqual(
            result["error_type"], FurnitureErrorType.POSITION_OUT_OF_BOUNDS.value
        )
        self.assertIn("out of floor plan bounds", result["message"])

    def test_bounds_check_before_noise_application(self):
        """Test that bounds are checked before placement noise is applied."""
        self.mock_scene.room_geometry = Mock()
        self.mock_scene.room_geometry.length = 10.0  # X: [-5, 5]
        self.mock_scene.room_geometry.width = 8.0  # Y: [-4, 4]

        mock_asset = Mock(spec=SceneObject)
        mock_asset.object_id = UniqueID("test_table")
        mock_asset.name = "Table"
        self.mock_asset_manager.get_asset_by_id.return_value = mock_asset

        # Attempt placement with requested position just outside bounds.
        # If bounds check happens before noise, should fail immediately.
        # If after noise, might succeed depending on noise direction.
        result_str = self.furniture_tools._add_furniture_to_scene_impl(
            asset_id=str(mock_asset.object_id), x=5.01, y=0.0, z=0.0
        )

        result = json.loads(result_str)
        # Should fail because bounds check happens before noise.
        self.assertFalse(result["success"])
        self.assertEqual(
            result["error_type"], FurnitureErrorType.POSITION_OUT_OF_BOUNDS.value
        )
        # Error message should show the exact requested coordinates, not noisy ones.
        self.assertIn("(5.010,", result["message"])


class TestFacingCheck(BaseAgentToolsTest):
    """Test facing check tool for spatial relationships between objects."""

    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.mock_scene = self.create_mock_scene()

        # Mock get_object to return objects from the dict.
        def mock_get_object(obj_id):
            return self.mock_scene.objects.get(obj_id)

        self.mock_scene.get_object = mock_get_object

        # Load base configuration from actual config file.
        config_path = (
            Path(__file__).parent.parent.parent
            / "configs/furniture_agent/base_furniture_agent.yaml"
        )
        self.cfg = OmegaConf.load(config_path)

        self.scene_tools = SceneTools(scene=self.mock_scene, cfg=self.cfg)

    def create_scene_object_with_bbox(
        self,
        name: str,
        position: list[float],
        yaw_degrees: float = 0.0,
        bbox_min: list[float] = None,
        bbox_max: list[float] = None,
    ) -> SceneObject:
        """Create a SceneObject with real transform and bounding box.

        Args:
            name: Object name.
            position: [x, y, z] position in world frame.
            yaw_degrees: Yaw rotation in degrees (around z-axis).
            bbox_min: Object-frame bbox minimum [x, y, z].
            bbox_max: Object-frame bbox maximum [x, y, z].

        Returns:
            SceneObject with real RigidTransform and bounding box.
        """
        # Default bounding box: 1m cube centered at origin.
        if bbox_min is None:
            bbox_min = [-0.5, -0.5, -0.5]
        if bbox_max is None:
            bbox_max = [0.5, 0.5, 0.5]

        # Create real RigidTransform.
        yaw_rad = math.radians(yaw_degrees)
        transform = RigidTransform(
            rpy=RollPitchYaw(roll=0.0, pitch=0.0, yaw=yaw_rad),
            p=position,
        )

        return SceneObject(
            object_id=UniqueID(name),
            object_type=ObjectType.FURNITURE,
            name=name,
            description=f"Test {name}",
            transform=transform,
            bbox_min=np.array(bbox_min),
            bbox_max=np.array(bbox_max),
        )

    def test_facing_at_zero_degrees(self):
        """Test object A facing object B at 0° yaw (aligned with +y)."""
        # Object A at origin facing +y direction.
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=0.0,
        )
        # Object B directly in front of A (along +y).
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"], f"Operation should succeed: {result}")
        self.assertTrue(
            result["is_facing"],
            "Chair should be facing table when aligned at 0°",
        )
        # Optimal rotation should be close to 0° (already facing).
        self.assertLess(
            abs(result["optimal_rotation_degrees"]),
            5.0,
            "Optimal rotation should be near 0° when already facing",
        )

    def test_not_facing_at_90_degrees(self):
        """Test object A not facing object B when rotated 90° away."""
        # Object A at origin rotated 90° (facing +x instead of +y).
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=90.0,
        )
        # Object B still at +y direction.
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertFalse(
            result["is_facing"],
            "Chair rotated 90° should not be facing table at +y",
        )
        # Optimal rotation should be 0° (absolute rotation to face +y direction).
        self.assertAlmostEqual(
            result["optimal_rotation_degrees"],
            0.0,
            delta=5.0,
            msg="Should need 0° absolute rotation to face table at +y",
        )

    def test_not_facing_at_180_degrees(self):
        """Test object A not facing object B when rotated 180° (facing away)."""
        # Object A facing -y direction (180° away from +y).
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=180.0,
        )
        # Object B at +y direction.
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertFalse(
            result["is_facing"],
            "Chair rotated 180° should not be facing table",
        )
        # Optimal rotation should be 0° (absolute rotation to face +y direction).
        self.assertAlmostEqual(
            result["optimal_rotation_degrees"],
            0.0,
            delta=5.0,
            msg="Should need 0° absolute rotation to face table at +y",
        )

    def test_facing_at_45_degrees(self):
        """Test object A partially facing object B at 45° angle."""
        # Object A rotated 45° from +y axis.
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=45.0,
        )
        # Object B at +y direction.
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"])
        # At 45° the ray might still intersect depending on bbox size.
        # The important part is the optimal rotation.
        self.assertAlmostEqual(
            result["optimal_rotation_degrees"],
            0.0,
            delta=5.0,
            msg="Should need 0° absolute rotation to optimally face table at +y",
        )

    def test_facing_with_arbitrary_positions(self):
        """Test facing check with objects at arbitrary positions."""
        # Object A at (1, 1, 0) facing +y.
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[1.0, 1.0, 0.0],
            yaw_degrees=0.0,
        )
        # Object B at (1, 3, 0) - still along A's +y axis.
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[1.0, 3.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertTrue(
            result["is_facing"],
            "Chair should be facing table when aligned along +y",
        )

    def test_facing_with_offset_target(self):
        """Test facing check when target is offset to the side."""
        # Object A at origin facing +y.
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=0.0,
        )
        # Object B offset to the side (+x direction).
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[2.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"])
        # Should indicate absolute rotation needed to face the offset target.
        # Target at (2, 2) from origin requires -45° (45° clockwise) rotation.
        # At yaw=-45°, local +y points to world (sin(45°), cos(45°)) = northeast.
        self.assertAlmostEqual(
            result["optimal_rotation_degrees"],
            -45.0,
            delta=5.0,
            msg="Should need -45° absolute rotation to face offset target",
        )

    def test_object_without_bbox(self):
        """Test error handling when object lacks bounding box."""
        # Object A with bbox.
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
        )
        # Object B without bbox.
        obj_b = SceneObject(
            object_id=UniqueID("table"),
            object_type=ObjectType.FURNITURE,
            name="table",
            description="Test table",
            transform=RigidTransform(),
            bbox_min=None,
            bbox_max=None,
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertFalse(result["success"], "Should fail when object lacks bbox")
        self.assertIn("bounding box", result["message"].lower())

    def test_invalid_object_id(self):
        """Test error handling when object ID doesn't exist."""
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id="nonexistent_id",
        )

        result = json.loads(result_str)

        self.assertFalse(result["success"], "Should fail with invalid object ID")
        self.assertIn("not found", result["message"].lower())

    def test_optimal_rotation_achieves_facing(self):
        """Test that applying optimal rotation results in facing relationship.

        This is a round-trip test that verifies the mathematical consistency
        of the facing check implementation. It ensures that:
        1. The optimal rotation is an absolute value (not a delta)
        2. Applying the optimal rotation achieves a facing relationship
        3. The optimal rotation remains consistent (doesn't change to ~0)
        """
        # Start with chair facing wrong direction (perpendicular).
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=90.0,  # Facing +x instead of +y
        )
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[0.0, 2.0, 0.0],  # Directly in front (+y direction)
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        # First check - should NOT be facing.
        result1_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result1 = json.loads(result1_str)

        self.assertTrue(result1["success"])
        self.assertFalse(
            result1["is_facing"], "Chair at 90° should not be facing table at +y"
        )

        optimal_rotation = result1["optimal_rotation_degrees"]

        # Apply the optimal rotation (absolute value).
        obj_a_rotated = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=optimal_rotation,  # Use absolute value directly
        )

        # Update scene with rotated object.
        self.mock_scene.objects = {
            obj_a_rotated.object_id: obj_a_rotated,
            obj_b.object_id: obj_b,
        }

        # Second check - should NOW be facing.
        result2_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a_rotated.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result2 = json.loads(result2_str)

        self.assertTrue(result2["success"])
        self.assertTrue(
            result2["is_facing"],
            f"Chair at optimal rotation ({optimal_rotation:.1f}°) should be facing table",
        )

        # Since optimal_rotation is absolute, it should remain approximately
        # the same value (not become ~0).
        self.assertAlmostEqual(
            result2["optimal_rotation_degrees"],
            optimal_rotation,
            delta=5.0,
            msg="Optimal rotation should be consistent (absolute value, not delta)",
        )

    def test_facing_away_from_wall(self):
        """Test object facing away from wall with direction='away'."""
        # Furniture at origin rotated 180° (facing -y, back toward +y).
        obj_a = self.create_scene_object_with_bbox(
            name="desk",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=180.0,
        )
        # Wall at +y direction.
        obj_b = self.create_scene_object_with_bbox(
            name="wall",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
            direction="away",
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"], f"Operation should succeed: {result}")
        self.assertTrue(
            result["is_facing"],
            "Desk rotated 180° should be facing away from wall at +y",
        )
        # Optimal rotation should be close to 180° (already facing away).
        self.assertAlmostEqual(
            result["optimal_rotation_degrees"],
            180.0,
            delta=5.0,
            msg="Optimal rotation should be near 180° when already facing away",
        )

    def test_not_facing_away(self):
        """Test object facing toward wall with direction='away' returns False."""
        # Furniture at origin rotated 0° (facing +y, front toward wall).
        obj_a = self.create_scene_object_with_bbox(
            name="desk",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=0.0,
        )
        # Wall at +y direction.
        obj_b = self.create_scene_object_with_bbox(
            name="wall",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
            direction="away",
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertFalse(
            result["is_facing"],
            "Desk rotated 0° should NOT be facing away from wall at +y",
        )
        # Optimal rotation should be 180° (absolute rotation to face away).
        self.assertAlmostEqual(
            result["optimal_rotation_degrees"],
            180.0,
            delta=5.0,
            msg="Should need 180° absolute rotation to face away from wall at +y",
        )

    def test_facing_toward_with_explicit_direction(self):
        """Test explicit direction='toward' works same as default."""
        # Object A at origin facing +y.
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=0.0,
        )
        # Object B at +y direction.
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
            direction="toward",
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"])
        self.assertTrue(
            result["is_facing"],
            "Chair should be facing toward table when aligned at 0°",
        )

    def test_invalid_direction_parameter(self):
        """Test error handling for invalid direction parameter."""
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.0],
        )
        obj_b = self.create_scene_object_with_bbox(
            name="table",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
            direction="invalid_direction",
        )

        result = json.loads(result_str)

        self.assertFalse(result["success"], "Should fail with invalid direction")
        self.assertIn("Invalid direction parameter", result["message"])
        self.assertIn("toward", result["message"].lower())
        self.assertIn("away", result["message"].lower())

    def test_optimal_rotation_for_facing_away(self):
        """Test that optimal rotation for facing away is 180° offset."""
        # Object A at origin rotated 90° (facing +x).
        obj_a = self.create_scene_object_with_bbox(
            name="desk",
            position=[0.0, 0.0, 0.0],
            yaw_degrees=90.0,
        )
        # Wall at +y direction.
        obj_b = self.create_scene_object_with_bbox(
            name="wall",
            position=[0.0, 2.0, 0.0],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        # Check facing toward (should suggest 0°).
        result_toward_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
            direction="toward",
        )

        result_toward = json.loads(result_toward_str)

        # Check facing away (should suggest 180°, which is 180° from toward).
        result_away_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
            direction="away",
        )

        result_away = json.loads(result_away_str)

        self.assertTrue(result_toward["success"])
        self.assertTrue(result_away["success"])

        # The difference should be approximately 180°.
        rotation_diff = abs(
            result_away["optimal_rotation_degrees"]
            - result_toward["optimal_rotation_degrees"]
        )
        # Account for wrapping (e.g., 170° to -170° is 340°, but should be 20°).
        if rotation_diff > 180.0:
            rotation_diff = 360.0 - rotation_diff

        self.assertAlmostEqual(
            rotation_diff,
            180.0,
            delta=5.0,
            msg="Optimal rotation for 'away' should be 180° from 'toward'",
        )

    def test_facing_with_z_height_difference(self):
        """Test facing check works with chair at different Z-height than table.

        Regression test for bug where 3D ray-AABB intersection failed when
        chair's center was at different Z-height than table's thin AABB.
        The fix uses 2D ray-rectangle intersection in XY plane.
        """
        # Chair at seat height (Z=0.5m).
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.0, 0.5],
            yaw_degrees=0.0,
            bbox_min=[-0.3, -0.3, -0.3],
            bbox_max=[0.3, 0.3, 0.3],
        )
        # Thin round table near ground (Z=0.05m, height=0.1m).
        obj_b = self.create_scene_object_with_bbox(
            name="table_round",
            position=[0.0, 1.5, 0.05],
            bbox_min=[-0.5, -0.5, -0.05],
            bbox_max=[0.5, 0.5, 0.05],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"], f"Operation should succeed: {result}")
        self.assertTrue(
            result["is_facing"],
            "Chair at different Z-height should still be facing table in XY plane",
        )
        # Optimal rotation should be close to 0° (already aligned in XY plane).
        self.assertLess(
            abs(result["optimal_rotation_degrees"]),
            5.0,
            "Optimal rotation should be near 0° when aligned in XY plane",
        )

    def test_facing_with_small_misalignment_round_table(self):
        """Test chair close to round table with small misalignment and Z-height difference.

        Regression test for Z-height mismatch bug. The 2D ray-rectangle
        intersection ensures Z-height differences don't cause false negatives.
        """
        # Chair at seat height (Z=0.5m), positioned in front of table.
        obj_a = self.create_scene_object_with_bbox(
            name="chair",
            position=[0.0, 0.8, 0.5],  # 0.8m in front (+Y), elevated
            yaw_degrees=2.0,  # Slightly misaligned (2° off from 0°)
            bbox_min=[-0.3, -0.3, -0.3],
            bbox_max=[0.3, 0.3, 0.3],
        )
        # Round table at ground level (different Z than chair).
        obj_b = self.create_scene_object_with_bbox(
            name="table_round",
            position=[0.0, 2.0, 0.0],  # 2m away from origin
            bbox_min=[-0.7, -0.7, -0.05],  # Large enough to catch the ray
            bbox_max=[0.7, 0.7, 0.05],
        )

        self.mock_scene.objects = {obj_a.object_id: obj_a, obj_b.object_id: obj_b}

        result_str = self.scene_tools._check_facing_impl(
            object_a_id=str(obj_a.object_id),
            object_b_id=str(obj_b.object_id),
        )

        result = json.loads(result_str)

        self.assertTrue(result["success"], f"Operation should succeed: {result}")
        self.assertTrue(
            result["is_facing"],
            "Chair with 2° misalignment at different Z-height should be "
            "facing table in XY plane (2D ray-rectangle intersection)",
        )
        # Optimal rotation should be close to 0° (facing +Y direction).
        self.assertLess(
            abs(result["optimal_rotation_degrees"]),
            5.0,
            "Optimal rotation should be near 0° for chair facing +Y toward table",
        )


if __name__ == "__main__":
    unittest.main()
