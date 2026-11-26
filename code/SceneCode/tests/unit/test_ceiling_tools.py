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
from scenecode.ceiling_agents.tools.ceiling_tools import CeilingTools


class TestCeilingToolsInternalModelPose(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        config_path = (
            Path(__file__).parent.parent.parent
            / "configs/ceiling_agent/base_ceiling_agent.yaml"
        )
        self.cfg = OmegaConf.load(config_path)
        self.mock_scene = Mock(spec=RoomScene)
        self.mock_scene.objects = {}
        self.mock_scene.action_log_path = None
        self.mock_scene.generate_unique_id = Mock(return_value=UniqueID("ceiling_0"))
        self.mock_asset_manager = Mock(spec=AssetManager)

        self.ceiling_tools = CeilingTools(
            scene=self.mock_scene,
            room_bounds=(-2.0, -2.0, 2.0, 2.0),
            ceiling_height=2.5,
            asset_manager=self.mock_asset_manager,
            cfg=self.cfg,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("scenecode.ceiling_agents.tools.ceiling_tools.apply_ceiling_placement_noise")
    def test_place_ceiling_object_preserves_internal_model_pose(self, mock_noise):
        asset_id = UniqueID("light_0")
        asset = Mock()
        asset.object_id = asset_id
        asset.name = "Pendant Light"
        asset.description = "A ceiling object"
        asset.geometry_path = Path("/tmp/light.obj")
        asset.sdf_path = Path("/tmp/light.sdf")
        asset.image_path = None
        asset.metadata = {}
        asset.bbox_min = np.array([0.0, 0.0, 0.0])
        asset.bbox_max = np.array([0.5, 0.5, 0.5])
        asset.scale_factor = 1.0
        asset.support_surfaces = []
        asset.immutable = False
        asset.internal_model_pose = RigidTransform(RollPitchYaw(0.0, 0.0, np.pi), [0.0, -0.1, 0.0])

        self.mock_asset_manager.get_asset_by_id = Mock(return_value=asset)
        mock_noise.return_value = (0.25, -0.25, 10.0)

        result_json = self.ceiling_tools._place_ceiling_object_impl(
            asset_id=str(asset_id),
            position_x=0.25,
            position_y=-0.25,
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
