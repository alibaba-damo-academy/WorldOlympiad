import json
import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from omegaconf import OmegaConf
from pydrake.all import RigidTransform, RollPitchYaw

from scenecode.agent_utils.asset_manager import AssetManager
from scenecode.agent_utils.room import RoomScene, UniqueID
from scenecode.wall_agents.tools.wall_tools import WallTools


class TestWallToolsInternalModelPose(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        config_path = (
            Path(__file__).parent.parent.parent
            / "configs/wall_agent/base_wall_agent.yaml"
        )
        self.cfg = OmegaConf.load(config_path)
        self.mock_scene = Mock(spec=RoomScene)
        self.mock_scene.objects = {}
        self.mock_scene.action_log_path = None
        self.mock_scene.generate_unique_id = Mock(return_value=UniqueID("wall_0"))

        self.mock_asset_manager = Mock(spec=AssetManager)
        self.mock_surface = Mock()
        self.mock_surface.surface_id = UniqueID("wall_surface_0")
        self.mock_surface.check_object_bounds = Mock(return_value=(True, None))
        self.mock_surface.to_world_pose = Mock(return_value=RigidTransform(p=[1.0, 0.0, 1.5]))

        self.wall_tools = WallTools(
            scene=self.mock_scene,
            wall_surfaces=[self.mock_surface],
            asset_manager=self.mock_asset_manager,
            cfg=self.cfg,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("scenecode.wall_agents.tools.wall_tools.apply_wall_placement_noise")
    def test_place_wall_object_preserves_internal_model_pose(self, mock_noise):
        asset_id = UniqueID("art_0")
        asset = Mock()
        asset.object_id = asset_id
        asset.name = "Wall Art"
        asset.description = "A wall object"
        asset.geometry_path = Path("/tmp/wall_art.obj")
        asset.sdf_path = Path("/tmp/wall_art.sdf")
        asset.image_path = None
        asset.metadata = {}
        asset.bbox_min = np.array([0.0, 0.0, 0.0])
        asset.bbox_max = np.array([1.0, 0.1, 1.0])
        asset.scale_factor = 1.0
        asset.support_surfaces = []
        asset.immutable = False
        asset.internal_model_pose = RigidTransform(RollPitchYaw(0.0, 0.0, np.pi), [0.0, -0.2, 0.0])

        self.mock_asset_manager.get_asset_by_id = Mock(return_value=asset)
        mock_noise.return_value = (0.5, 1.2, 15.0)

        result_json = self.wall_tools._place_wall_object_impl(
            asset_id=str(asset_id),
            wall_surface_id=str(self.mock_surface.surface_id),
            position_x=0.5,
            position_z=1.2,
        )

        self.assertTrue(json.loads(result_json)["success"])
        placed_obj = self.mock_scene.add_object.call_args[0][0]
        np.testing.assert_array_almost_equal(
            placed_obj.internal_model_pose.translation(),
            asset.internal_model_pose.translation(),
        )
        np.testing.assert_array_almost_equal(
            placed_obj.internal_model_pose.rotation().matrix(),
            asset.internal_model_pose.rotation().matrix(),
        )
