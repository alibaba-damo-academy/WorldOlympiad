import json
import math
import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import trimesh

from omegaconf import OmegaConf
from pydrake.all import RigidTransform, RollPitchYaw

from scenecode.agent_utils.asset_manager import AssetManager
from scenecode.agent_utils.placement_noise import PlacementNoiseMode
from scenecode.agent_utils.room import RoomScene, SupportSurface, UniqueID
from scenecode.manipuland_agents.tools.manipuland_tools import ManipulandTools


class TestManipulandTools(unittest.TestCase):
    """Test ManipulandTools class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Load configuration from actual config file.
        config_path = (
            Path(__file__).parent.parent.parent
            / "configs/manipuland_agent/base_manipuland_agent.yaml"
        )
        self.cfg = OmegaConf.load(config_path)

        # Create mock scene.
        self.mock_scene = Mock(spec=RoomScene)
        self.mock_scene.objects = {}
        self.mock_scene.action_log_path = None

        # Create mock asset manager.
        self.mock_asset_manager = Mock(spec=AssetManager)

        # Create mock support surface.
        self.mock_surface = Mock(spec=SupportSurface)
        self.mock_surface.surface_id = UniqueID("test_surface_001")
        self.mock_surface.bounding_box_min = np.array([-0.5, -0.5, 0.0])
        self.mock_surface.bounding_box_max = np.array(
            [0.5, 0.5, 0.5]
        )  # 0.5m clearance.
        self.mock_surface.contains_point_2d = Mock(return_value=True)
        self.mock_surface.to_world_pose = Mock(
            return_value=RigidTransform(p=[0.0, 0.0, 1.0])
        )

        # Create support surfaces dict (multi-surface API).
        self.support_surfaces = {str(self.mock_surface.surface_id): self.mock_surface}

        # Create manipuland tools instance.
        self.manipuland_tools = ManipulandTools(
            scene=self.mock_scene,
            asset_manager=self.mock_asset_manager,
            cfg=self.cfg,
            current_furniture_id=UniqueID("furniture_001"),
            support_surfaces=self.support_surfaces,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_set_noise_profile_natural(self):
        """Test set_noise_profile with natural mode."""
        self.manipuland_tools.set_noise_profile(PlacementNoiseMode.NATURAL)
        self.assertEqual(
            self.manipuland_tools.active_noise_profile,
            self.cfg.placement_noise.natural_profile,
        )

    def test_set_noise_profile_perfect(self):
        """Test set_noise_profile with perfect mode."""
        self.manipuland_tools.set_noise_profile(PlacementNoiseMode.PERFECT)
        self.assertEqual(
            self.manipuland_tools.active_noise_profile,
            self.cfg.placement_noise.perfect_profile,
        )

    def test_move_manipuland_out_of_bounds_fails(self):
        """Test that move_manipuland fails when position is out of bounds."""
        # Create a mock object.
        object_id = UniqueID("manipuland_001")
        mock_object = Mock()
        mock_object.name = "Test Object"
        mock_object.placement_info = Mock()
        mock_object.placement_info.position_2d = np.array([0.0, 0.0])
        mock_object.placement_info.rotation_2d = 0.0
        mock_object.placement_info.parent_surface_id = self.mock_surface.surface_id

        self.mock_scene.get_object = Mock(return_value=mock_object)

        # Set surface to reject the position.
        self.mock_surface.contains_point_2d = Mock(return_value=False)

        # Try to move to out-of-bounds position.
        result_json = self.manipuland_tools._move_manipuland_impl(
            object_id=str(object_id),
            surface_id=str(self.mock_surface.surface_id),
            position_x=10.0,
            position_z=10.0,
        )

        # Parse JSON to check error.
        result_dict = json.loads(result_json)
        self.assertFalse(result_dict["success"])
        self.assertEqual(result_dict["error_type"], "position_out_of_bounds")

    def test_move_manipuland_nonexistent_object_fails(self):
        """Test that move_manipuland fails when object doesn't exist."""
        self.mock_scene.get_object = Mock(return_value=None)

        object_id = "nonexistent_object"
        result_json = self.manipuland_tools._move_manipuland_impl(
            object_id=object_id,
            surface_id=str(self.mock_surface.surface_id),
            position_x=0.1,
            position_z=0.1,
        )

        # Parse JSON to check error.
        result_dict = json.loads(result_json)
        self.assertFalse(result_dict["success"])
        self.assertEqual(result_dict["error_type"], "object_not_found")

    @patch.object(
        ManipulandTools, "_validate_convex_hull_footprint", return_value=(True, None)
    )
    @patch.object(ManipulandTools, "_is_top_surface", return_value=True)
    def test_move_manipuland_no_movement_fails(self, mock_is_top, mock_validate):
        """Test that move_manipuland fails when trying to move to same position."""
        # Create a mock object at position (0.1, 0.1).
        object_id = UniqueID("manipuland_001")
        current_position = np.array([0.1, 0.1])
        current_rotation = math.radians(45.0)

        mock_object = Mock()
        mock_object.name = "Test Object"
        mock_object.bbox_min = np.array([0.0, 0.0, 0.0])
        mock_object.bbox_max = np.array(
            [0.1, 0.1, 0.1]
        )  # 0.1m height, fits in 0.5m clearance.
        mock_object.placement_info = Mock()
        mock_object.placement_info.position_2d = current_position
        mock_object.placement_info.rotation_2d = current_rotation
        mock_object.placement_info.parent_surface_id = self.mock_surface.surface_id

        self.mock_scene.get_object = Mock(return_value=mock_object)

        # Try to move to same position.
        result_json = self.manipuland_tools._move_manipuland_impl(
            object_id=str(object_id),
            surface_id=str(self.mock_surface.surface_id),
            position_x=0.1,
            position_z=0.1,
            rotation_degrees=45.0,
        )

        # Parse JSON to check error.
        result_dict = json.loads(result_json)
        self.assertFalse(result_dict["success"])
        self.assertEqual(result_dict["error_type"], "no_movement")

    @patch.object(
        ManipulandTools, "_validate_convex_hull_footprint", return_value=(True, None)
    )
    @patch.object(ManipulandTools, "_is_top_surface", return_value=True)
    def test_move_manipuland_no_placement_info_fails(self, mock_is_top, mock_validate):
        """Test that move_manipuland fails when object has no placement_info."""
        # Create a mock object without placement_info.
        object_id = UniqueID("manipuland_001")
        mock_object = Mock()
        mock_object.name = "Test Object"
        mock_object.bbox_min = np.array([0.0, 0.0, 0.0])
        mock_object.bbox_max = np.array(
            [0.1, 0.1, 0.1]
        )  # 0.1m height, fits in 0.5m clearance.
        mock_object.placement_info = None

        self.mock_scene.get_object = Mock(return_value=mock_object)

        # Try to move object.
        result_json = self.manipuland_tools._move_manipuland_impl(
            object_id=str(object_id),
            surface_id=str(self.mock_surface.surface_id),
            position_x=0.1,
            position_z=0.1,
        )

        # Parse JSON to check error.
        result_dict = json.loads(result_json)
        self.assertFalse(result_dict["success"])
        self.assertEqual(result_dict["error_type"], "invalid_operation")
        self.assertIn("no placement info", result_dict["message"].lower())

    def test_move_manipuland_invalid_surface_fails(self):
        """Test that move_manipuland fails when surface_id doesn't exist."""
        # Create a mock object on a known surface.
        object_id = UniqueID("manipuland_001")
        mock_object = Mock()
        mock_object.name = "Test Object"
        mock_object.placement_info = Mock()
        mock_object.placement_info.position_2d = np.array([0.0, 0.0])
        mock_object.placement_info.rotation_2d = 0.0
        mock_object.placement_info.parent_surface_id = self.mock_surface.surface_id

        self.mock_scene.get_object = Mock(return_value=mock_object)

        # Try to move object to non-existent surface.
        result_json = self.manipuland_tools._move_manipuland_impl(
            object_id=str(object_id),
            surface_id="nonexistent_surface_id",
            position_x=0.1,
            position_z=0.1,
        )

        # Parse JSON to check error.
        result_dict = json.loads(result_json)
        self.assertFalse(result_dict["success"])
        self.assertEqual(result_dict["error_type"], "invalid_surface")

    @patch.object(
        ManipulandTools, "_validate_convex_hull_footprint", return_value=(True, None)
    )
    @patch.object(ManipulandTools, "_is_top_surface", return_value=True)
    @patch("scenecode.manipuland_agents.tools.manipuland_tools.apply_placement_noise")
    def test_move_manipuland_applies_noise(
        self, mock_apply_noise, mock_is_top, mock_validate
    ):
        """Test that move_manipuland applies placement noise."""
        # Create a mock object.
        object_id = UniqueID("manipuland_001")
        mock_object = Mock()
        mock_object.name = "Test Object"
        mock_object.bbox_min = np.array([0.0, 0.0, 0.0])
        mock_object.bbox_max = np.array(
            [0.1, 0.1, 0.1]
        )  # 0.1m height, fits in 0.5m clearance.
        mock_object.placement_info = Mock()
        mock_object.placement_info.position_2d = np.array([0.0, 0.0])
        mock_object.placement_info.rotation_2d = 0.0
        mock_object.placement_info.parent_surface_id = self.mock_surface.surface_id

        self.mock_scene.get_object = Mock(return_value=mock_object)

        # Set up noise mock to return a slightly modified transform.
        def noise_side_effect(transform, **kwargs):
            return RigidTransform(p=[0.1, 0.1, 1.0])

        mock_apply_noise.side_effect = noise_side_effect

        # Move to new position.
        result_json = self.manipuland_tools._move_manipuland_impl(
            object_id=str(object_id),
            surface_id=str(self.mock_surface.surface_id),
            position_x=0.2,
            position_z=0.2,
        )

        # Verify noise was applied.
        mock_apply_noise.assert_called_once()
        call_kwargs = mock_apply_noise.call_args[1]
        self.assertEqual(
            call_kwargs["position_xy_std_meters"],
            self.cfg.placement_noise.natural_profile.position_xy_std_meters,
        )
        self.assertEqual(
            call_kwargs["rotation_yaw_std_degrees"],
            self.cfg.placement_noise.natural_profile.rotation_yaw_std_degrees,
        )

    @patch.object(
        ManipulandTools, "_validate_convex_hull_footprint", return_value=(True, None)
    )
    @patch.object(ManipulandTools, "_is_top_surface", return_value=True)
    @patch("scenecode.manipuland_agents.tools.manipuland_tools.apply_placement_noise")
    def test_place_manipuland_preserves_internal_model_pose(
        self, mock_apply_noise, mock_is_top, mock_validate
    ):
        """Placed manipulands should keep articulated internal model pose."""
        asset_id = UniqueID("asset_002")
        mock_asset = Mock()
        mock_asset.object_id = asset_id
        mock_asset.name = "Drawer Insert"
        mock_asset.description = "An articulated insert"
        mock_asset.geometry_path = Path("/tmp/insert.obj")
        mock_asset.sdf_path = Path("/tmp/insert.sdf")
        mock_asset.image_path = None
        mock_asset.metadata = {}
        mock_asset.bbox_min = np.array([0.0, 0.0, 0.0])
        mock_asset.bbox_max = np.array([0.1, 0.1, 0.1])
        mock_asset.scale_factor = 1.0
        mock_asset.internal_model_pose = RigidTransform(
            RollPitchYaw(0.0, 0.0, np.pi),
            [0.0, -0.1, 0.0],
        )
        mock_asset.support_surfaces = []
        mock_asset.immutable = False

        self.mock_asset_manager.get_asset_by_id = Mock(return_value=mock_asset)
        self.mock_scene.generate_unique_id = Mock(
            return_value=UniqueID("manipuland_001")
        )
        mock_apply_noise.side_effect = lambda transform, **kwargs: transform

        result_json = self.manipuland_tools._place_manipuland_on_surface_impl(
            asset_id=str(asset_id),
            surface_id=str(self.mock_surface.surface_id),
            position_x=0.1,
            position_z=0.1,
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

    @patch.object(
        ManipulandTools, "_validate_convex_hull_footprint", return_value=(True, None)
    )
    @patch.object(ManipulandTools, "_is_top_surface", return_value=True)
    @patch("scenecode.manipuland_agents.tools.manipuland_tools.apply_placement_noise")
    def test_placement_applies_noise(
        self, mock_apply_noise, mock_is_top, mock_validate
    ):
        """Test that place_manipuland_on_surface applies placement noise."""
        # Create a mock asset.
        asset_id = UniqueID("asset_001")
        mock_asset = Mock()
        mock_asset.object_id = asset_id
        mock_asset.name = "Test Manipuland"
        mock_asset.description = "A test object"
        mock_asset.geometry_path = Path("/tmp/test.obj")
        mock_asset.sdf_path = Path("/tmp/test.sdf")
        mock_asset.image_path = None
        mock_asset.metadata = {}
        mock_asset.bbox_min = np.array([0.0, 0.0, 0.0])
        mock_asset.bbox_max = np.array([0.1, 0.1, 0.1])

        self.mock_asset_manager.get_asset_by_id = Mock(return_value=mock_asset)
        self.mock_scene.generate_unique_id = Mock(
            return_value=UniqueID("manipuland_001")
        )

        # Set up noise mock to return a slightly modified transform.
        def noise_side_effect(transform, **kwargs):
            return RigidTransform(p=[0.1, 0.1, 1.0])

        mock_apply_noise.side_effect = noise_side_effect

        # Place manipuland.
        result_json = self.manipuland_tools._place_manipuland_on_surface_impl(
            asset_id=str(asset_id),
            surface_id=str(self.mock_surface.surface_id),
            position_x=0.1,
            position_z=0.1,
        )

        # Verify noise was applied.
        mock_apply_noise.assert_called_once()
        call_kwargs = mock_apply_noise.call_args[1]
        self.assertEqual(
            call_kwargs["position_xy_std_meters"],
            self.cfg.placement_noise.natural_profile.position_xy_std_meters,
        )
        self.assertEqual(
            call_kwargs["rotation_yaw_std_degrees"],
            self.cfg.placement_noise.natural_profile.rotation_yaw_std_degrees,
        )

    def test_validate_convex_hull_strict_containment(self):
        """Test convex hull validation with strict containment (0% overlap)."""
        # Create a simple square mesh (1m x 1m).
        vertices = np.array(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Save mesh to temp file.
        mesh_path = self.temp_dir / "test_square.obj"
        mesh.export(mesh_path)

        # Create a circular surface (radius 1.0m).
        circle_surface = Mock(spec=SupportSurface)
        circle_surface.surface_id = UniqueID("circle_surface_001")

        def contains_point(point_2d):
            # Circle of radius 1.0 centered at origin.
            return np.linalg.norm(point_2d) <= 1.0

        circle_surface.contains_point_2d = Mock(side_effect=contains_point)

        # Test: Centered square corners at distance 0.707m should fit in 1.0m radius.
        is_valid, error_msg = self.manipuland_tools._validate_convex_hull_footprint(
            target_surface=circle_surface,
            geometry_path=mesh_path,
            position_2d=np.array([0.0, 0.0]),
            rotation_degrees=0.0,
            allow_overlap_ratio=0.0,
        )
        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)

        # Test: Offset square should fail.
        is_valid, error_msg = self.manipuland_tools._validate_convex_hull_footprint(
            target_surface=circle_surface,
            geometry_path=mesh_path,
            position_2d=np.array([0.5, 0.5]),
            rotation_degrees=0.0,
            allow_overlap_ratio=0.0,
        )
        self.assertFalse(is_valid)
        self.assertIsNotNone(error_msg)

    def test_validate_convex_hull_with_overlap_tolerance(self):
        """Test convex hull validation with 15% overlap tolerance."""
        # Create a simple square mesh (1m x 1m).
        vertices = np.array(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Save mesh to temp file.
        mesh_path = self.temp_dir / "test_square.obj"
        mesh.export(mesh_path)

        # Create a circular surface (radius 0.65m).
        circle_surface = Mock(spec=SupportSurface)
        circle_surface.surface_id = UniqueID("circle_surface_001")

        def contains_point(point_2d):
            return np.linalg.norm(point_2d) <= 0.65

        circle_surface.contains_point_2d = Mock(side_effect=contains_point)

        # With 15% shrink, corners at 0.707 * 0.85 = 0.601m should fit in 0.65m radius.
        is_valid, error_msg = self.manipuland_tools._validate_convex_hull_footprint(
            target_surface=circle_surface,
            geometry_path=mesh_path,
            position_2d=np.array([0.0, 0.0]),
            rotation_degrees=0.0,
            allow_overlap_ratio=0.15,
        )
        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)

    def test_top_surface_uses_overlap_tolerance(self):
        """Test that top surfaces use configured overlap tolerance."""
        # Load base configuration.
        base_config_path = (
            Path(__file__).parent.parent.parent
            / "configs/manipuland_agent/base_manipuland_agent.yaml"
        )
        base_cfg = OmegaConf.load(base_config_path)

        # Load stateful configuration.
        stateful_config_path = (
            Path(__file__).parent.parent.parent
            / "configs/manipuland_agent/stateful_manipuland_agent.yaml"
        )
        stateful_cfg = OmegaConf.load(stateful_config_path)

        # Merge configs (simulating Hydra defaults resolution).
        cfg_with_validation = OmegaConf.merge(base_cfg, stateful_cfg)

        # Create new manipuland tools with updated config.
        tools = ManipulandTools(
            scene=self.mock_scene,
            asset_manager=self.mock_asset_manager,
            cfg=cfg_with_validation,
            current_furniture_id=UniqueID("furniture_001"),
            support_surfaces=self.support_surfaces,
        )

        # Verify tolerance is loaded from config.
        expected_tolerance = self.cfg.placement_validation.top_surface_overlap_tolerance
        self.assertEqual(tools.top_surface_overlap_tolerance, expected_tolerance)

    def test_non_top_surface_uses_strict_containment(self):
        """Test that non-top surfaces use strict containment."""
        # Create two surfaces at different heights.
        top_surface = Mock(spec=SupportSurface)
        top_surface.surface_id = UniqueID("top_surface_001")
        top_surface.bounding_box_min = np.array([-0.5, -0.5, 1.0])
        top_surface.bounding_box_max = np.array([0.5, 0.5, 1.5])
        top_surface.transform = RigidTransform(p=[0.0, 0.0, 1.5])  # Z=1.5m (top)

        shelf_surface = Mock(spec=SupportSurface)
        shelf_surface.surface_id = UniqueID("shelf_surface_001")
        shelf_surface.bounding_box_min = np.array([-0.5, -0.5, 0.5])
        shelf_surface.bounding_box_max = np.array([0.5, 0.5, 1.0])
        shelf_surface.transform = RigidTransform(p=[0.0, 0.0, 0.75])  # Z=0.75m (shelf)

        # Create a mock furniture object with these surfaces.
        mock_furniture = Mock()
        mock_furniture.support_surfaces = [top_surface, shelf_surface]

        # Add furniture to scene.
        furniture_id = UniqueID("furniture_001")
        self.mock_scene.objects = {furniture_id: mock_furniture}

        # Update support surfaces.
        multi_surfaces = {
            str(top_surface.surface_id): top_surface,
            str(shelf_surface.surface_id): shelf_surface,
        }
        tools = ManipulandTools(
            scene=self.mock_scene,
            asset_manager=self.mock_asset_manager,
            cfg=self.cfg,
            current_furniture_id=furniture_id,
            support_surfaces=multi_surfaces,
        )

        # Top surface should be identified correctly.
        self.assertTrue(tools._is_top_surface(str(top_surface.surface_id)))
        self.assertFalse(tools._is_top_surface(str(shelf_surface.surface_id)))


if __name__ == "__main__":
    unittest.main()
