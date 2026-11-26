"""
Integration test for manipuland placement on furniture surfaces.
Places coffee mugs on a work desk.
"""

import json
import logging
import shutil
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from omegaconf import OmegaConf
from pydrake.all import RigidTransform, RollPitchYaw

from scenecode.agent_utils.asset_manager import AssetManager
from scenecode.agent_utils.blender.server_manager import BlenderServer
from scenecode.agent_utils.house import RoomGeometry
from scenecode.agent_utils.physics_validation import compute_scene_collisions
from scenecode.agent_utils.rendering_manager import RenderingManager
from scenecode.agent_utils.room import (
    AgentType,
    ObjectType,
    RoomScene,
    SceneObject,
    UniqueID,
    extract_and_propagate_support_surfaces,
)
from scenecode.agent_utils.vlm_service import VLMService
from scenecode.manipuland_agents.tools.manipuland_tools import ManipulandTools
from scenecode.manipuland_agents.tools.vision_tools import ManipulandVisionTools
from scenecode.utils.logging import ConsoleLogger
from tests.integration.common import has_openai_key

console_logger = logging.getLogger(__name__)


def _normalize_image_to_rgb(img: np.ndarray) -> np.ndarray:
    """Normalize image array to RGB format.

    Converts RGBA to RGB by compositing onto white background.
    Handles grayscale by converting to RGB.

    Args:
        img: Input image array (any format).

    Returns:
        RGB image array with shape (H, W, 3).
    """
    if len(img.shape) == 2:
        # Grayscale to RGB.
        return np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        # RGBA to RGB via alpha compositing onto white background.
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        white_bg = np.ones_like(rgb) * 255.0
        composited = rgb * alpha + white_bg * (1 - alpha)
        return composited.astype(np.uint8)
    elif img.shape[2] == 3:
        return img
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")


def compute_l2_norm_difference(img1_path: Path, img2_path: Path) -> float:
    """Compute L2 norm (Euclidean distance) between two images in pixel space.

    Args:
        img1_path: Path to first image.
        img2_path: Path to second image.

    Returns:
        L2 norm of the pixel-wise difference, normalized by number of pixels.

    Raises:
        AssertionError: If images have different dimensions.
    """
    from PIL import Image as PILImage

    img1 = _normalize_image_to_rgb(np.array(PILImage.open(img1_path)))
    img2 = _normalize_image_to_rgb(np.array(PILImage.open(img2_path)))

    # Verify same shape.
    assert (
        img1.shape == img2.shape
    ), f"Image shape mismatch: {img1.shape} vs {img2.shape}"

    # Convert to float for precision in difference calculation.
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)

    # Compute pixel-wise difference.
    diff = img1_float - img2_float

    # Compute L2 norm and normalize by number of pixels.
    l2_norm = np.sqrt(np.sum(diff**2))
    normalized_l2 = l2_norm / (img1.shape[0] * img1.shape[1])

    return normalized_l2


@unittest.skipIf(not has_openai_key(), "Requires OPENAI_API_KEY")
class TestManipulandPlacement(unittest.TestCase):

    def _check_collisions(self, scene: RoomScene, context: str) -> list:
        """Check collisions and verify expected count.

        Args:
            scene: RoomScene to check for collisions.
            context: Description of when check is happening (for logging).

        Returns:
            List of collision objects found.
        """
        console_logger.info(f"Checking collisions {context}...")
        collisions = compute_scene_collisions(scene=scene)
        console_logger.info(f"Found {len(collisions)} collision(s) {context}")

        for collision in collisions:
            console_logger.warning(f"  - {collision.to_description()}")

        self.assertEqual(
            len(collisions),
            0,
            msg=(
                f"Expected 0 collisions {context}, found {len(collisions)}: "
                + ", ".join(c.to_description() for c in collisions)
            ),
        )
        return collisions

    def test_place_manipulands_on_desk(self):
        """Test placing 3 coffee mugs on work desk at different positions."""
        # Set up test data paths.
        test_data_dir = Path(__file__).parent.parent / "test_data" / "realistic_scene"
        floor_plan_path = test_data_dir / "room_geometry.sdf"

        # Verify test data exists.
        if not floor_plan_path.exists():
            self.fail(f"Test data file not found: {floor_plan_path}")

        # Create 5x5m room centered at origin.
        floor_plan_tree = ET.parse(floor_plan_path)
        wall_normals = {
            "left_wall": np.array([1.0, 0.0]),
            "right_wall": np.array([-1.0, 0.0]),
            "back_wall": np.array([0.0, 1.0]),
            "front_wall": np.array([0.0, -1.0]),
        }

        room_geometry = RoomGeometry(
            sdf_tree=floor_plan_tree,
            sdf_path=floor_plan_path,
            wall_normals=wall_normals,
        )
        scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

        # Load work desk assets.
        desk_sdf = (
            test_data_dir / "generated_assets/sdf/work_desk_1761578426/work_desk.sdf"
        )
        desk_gltf = (
            test_data_dir / "generated_assets/sdf/work_desk_1761578426/work_desk.gltf"
        )
        if not desk_sdf.exists():
            self.fail(f"Asset not found: {desk_sdf}")
        if not desk_gltf.exists():
            self.fail(f"Asset not found: {desk_gltf}")

        # Place desk at non-origin position with rotation to test coordinate transforms.
        # Position: offset from origin but centered in room, Rotation: 35 degrees around Z.
        desk_transform = RigidTransform(
            RollPitchYaw(0, 0, np.deg2rad(35.0)), np.array([0.5, 0.3, 0.0])
        )
        desk_obj = SceneObject(
            object_id=scene.generate_unique_id("work_desk"),
            object_type=ObjectType.FURNITURE,
            name="work_desk",
            description="Test desk for manipuland placement",
            transform=desk_transform,
            sdf_path=desk_sdf,
            geometry_path=desk_gltf,
            bbox_min=np.array([-0.70, -0.365, 0.0]),
            bbox_max=np.array([0.70, 0.365, 0.761]),
        )
        scene.add_object(desk_obj)

        # Extract desk's support surfaces.
        support_surfaces = extract_and_propagate_support_surfaces(
            scene=scene,
            furniture_object=desk_obj,
            config=None,  # Use default HSM config.
        )
        console_logger.info(
            f"Extracted {len(support_surfaces)} support surface(s) from desk"
        )
        for i, surface in enumerate(support_surfaces):
            console_logger.info(
                f"  Surface {i}: ID={surface.surface_id}, "
                f"area={surface.area:.3f} m², "
                f"bounds_min={surface.bounding_box_min}, "
                f"bounds_max={surface.bounding_box_max}"
            )
        # Assign support surfaces to desk so vision tools can find manipulands on it.
        desk_obj.support_surfaces = support_surfaces

        # Use the first (largest) surface for placement.
        primary_surface = support_surfaces[0]

        # Load base configuration.
        config_path = (
            Path(__file__).parent.parent.parent
            / "configs/manipuland_agent/base_manipuland_agent.yaml"
        )
        base_cfg = OmegaConf.load(config_path)

        # Override config to enable support surface debug visualization.
        test_overrides = {
            "rendering": {
                "annotations": {
                    "enable_support_surface_debug": False,
                }
            }
        }
        test_cfg = OmegaConf.merge(base_cfg, test_overrides)

        # Create test logger and VLM service.
        # Use temp directory for AssetManager to avoid polluting test outputs.
        temp_dir = tempfile.mkdtemp()
        temp_logger = ConsoleLogger(output_dir=Path(temp_dir))
        vlm_service = VLMService()

        # Create AssetManager for testing.
        # collision_client=None because test doesn't generate collision geometry.
        asset_manager = AssetManager(
            logger=temp_logger,
            vlm_service=vlm_service,
            blender_server=None,
            collision_client=None,
            cfg=test_cfg,
            agent_type=AgentType.MANIPULAND,
        )

        # Create support surfaces dict for multi-surface API.
        support_surfaces_dict = {str(s.surface_id): s for s in support_surfaces}

        # Create ManipulandTools instance.
        manipuland_tools = ManipulandTools(
            scene=scene,
            asset_manager=asset_manager,
            cfg=test_cfg,
            current_furniture_id=desk_obj.object_id,
            support_surfaces=support_surfaces_dict,
        )

        # Load coffee mug asset from test data and register it.
        coffee_mug_dir = (
            test_data_dir / "generated_assets/manipulands/sdf/coffee_mug_1762992163"
        )
        coffee_mug_sdf = coffee_mug_dir / "coffee_mug.sdf"
        coffee_mug_gltf = coffee_mug_dir / "coffee_mug.gltf"

        if not coffee_mug_sdf.exists():
            self.fail(f"Coffee mug SDF not found: {coffee_mug_sdf}")
        if not coffee_mug_gltf.exists():
            self.fail(f"Coffee mug GLTF not found: {coffee_mug_gltf}")

        # Register the coffee mug asset with the asset manager.
        coffee_mug_id = UniqueID("coffee_mug_1762992163")
        coffee_mug_asset = SceneObject(
            object_id=coffee_mug_id,
            object_type=ObjectType.MANIPULAND,
            name="coffee_mug",
            description="white ceramic coffee mug",
            transform=RigidTransform(),
            sdf_path=coffee_mug_sdf,
            geometry_path=coffee_mug_gltf,
            bbox_min=np.array([-0.03758648, -0.05243881, 0.0]),
            bbox_max=np.array([0.03758648, 0.05243881, 0.08]),
        )
        asset_manager.registry.register(coffee_mug_asset)

        # Place 3 coffee mugs at different positions and rotations.
        placements = [
            {"x": -0.3, "y": -0.2, "rotation": 0.0, "name": "mug_1"},
            {"x": 0.0, "y": 0.1, "rotation": 45.0, "name": "mug_2"},
            {"x": 0.4, "y": -0.15, "rotation": 270.0, "name": "mug_3"},
        ]

        placed_mugs = []
        for i, placement in enumerate(placements, 1):
            console_logger.info(
                f"Placing mug {i} at ({placement['x']}, {placement['y']}) "
                f"with rotation {placement['rotation']}°"
            )

            result_json = manipuland_tools._place_manipuland_on_surface_impl(
                asset_id=str(coffee_mug_id),
                surface_id=str(primary_surface.surface_id),
                position_x=placement["x"],
                position_z=placement["y"],
                rotation_degrees=placement["rotation"],
            )

            result = json.loads(result_json)
            console_logger.info(
                f"Placement result {i}: success={result.get('success')}, "
                f"message={result.get('message')}"
            )

            self.assertTrue(
                result["success"],
                msg=f"Mug {i} placement should succeed: {result.get('message')}",
            )

            # Store placed mug info for validation.
            placed_mugs.append(
                {
                    "object_id": result["object_id"],
                    "position": result["world_position"],
                    "placement_spec": placement,
                }
            )

        # Validate all mugs are on the desk (Z > 0.5), not on floor (Z ≈ 0).
        console_logger.info("Validating mug Z-coordinates...")
        for i, mug_info in enumerate(placed_mugs, 1):
            mug_z = mug_info["position"]["z"]
            console_logger.info(f"Mug {i} Z-coordinate: {mug_z:.3f}m")

            self.assertGreater(
                mug_z,
                0.5,
                msg=(
                    f"Mug {i} should be on desk surface (Z > 0.5m), "
                    f"not on floor (Z ≈ 0). Found Z={mug_z:.3f}m"
                ),
            )

        # Render scene using main pipeline code (ManipulandVisionTools).
        console_logger.info("Rendering scene using ManipulandVisionTools...")

        # Create RenderingManager using test_cfg (with debug visualization enabled).
        rendering_manager = RenderingManager(
            cfg=test_cfg.rendering, logger=temp_logger, subdirectory="manipulands"
        )

        # Create and start Blender server for rendering.
        blender_server = BlenderServer(
            host="127.0.0.1",
            port_range=(8010, 8020),
        )
        blender_server.start()

        # Create ManipulandVisionTools - this is the actual component the agent uses!
        vision_tools = ManipulandVisionTools(
            scene=scene,
            rendering_manager=rendering_manager,
            cfg=test_cfg,
            current_furniture_id=desk_obj.object_id,
            blender_server=blender_server,
        )

        # Call the actual vision tool implementation (same code path as agent).
        result_message = vision_tools._observe_scene_impl()
        console_logger.info(f"Vision tool returned {len(result_message)} outputs")

        # Verify renders were created by RenderingManager.
        renders_base_dir = temp_logger.output_dir / "scene_renders" / "manipulands"
        self.assertTrue(
            renders_base_dir.exists(),
            msg=f"Renders should be created at {renders_base_dir}",
        )

        # Find the renders directory (will be renders_001).
        renders_dir = next(renders_base_dir.glob("renders_*"))

        # Copy renders to test output directory for manual inspection.
        test_output_dir = Path(__file__).parent / "test_outputs"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        manipuland_test_dir = test_output_dir / f"test_manipuland_placement_{timestamp}"
        manipuland_test_dir.mkdir(parents=True, exist_ok=True)

        # Copy all rendered images.
        image_paths = []
        for img_path in sorted(renders_dir.glob("*.png")):
            dest = manipuland_test_dir / f"mugs_on_desk_{img_path.name}"
            shutil.copy(img_path, dest)
            image_paths.append(img_path)

        console_logger.info(f"Renders copied to {manipuland_test_dir}")

        # Compare rendered images against reference images using L2 norm.
        reference_dir = Path(__file__).parent / "reference_renders"

        # Find top view (0_top.png).
        top_view = next(p for p in image_paths if "0_top" in p.name)
        top_reference = reference_dir / "manipuland_overlay_top.png"

        # Find side view 2 (2_side.png).
        side_view_2 = next(p for p in image_paths if "2_side" in p.name)
        side_reference = reference_dir / "manipuland_overlay_side_2.png"

        # Verify reference images exist.
        self.assertTrue(
            top_reference.exists(),
            msg=f"Reference image not found: {top_reference}",
        )
        self.assertTrue(
            side_reference.exists(),
            msg=f"Reference image not found: {side_reference}",
        )

        # Compare top view.
        top_l2 = compute_l2_norm_difference(img1_path=top_view, img2_path=top_reference)
        console_logger.info(f"Top view L2 norm: {top_l2:.4f}")

        # Compare side view 2.
        side_l2 = compute_l2_norm_difference(
            img1_path=side_view_2, img2_path=side_reference
        )
        console_logger.info(f"Side view 2 L2 norm: {side_l2:.4f}")

        # Assert L2 norms are below threshold.
        THRESHOLD = 0.05  # Tight tolerance for deterministic rendering.

        self.assertLessEqual(
            top_l2,
            THRESHOLD,
            msg=f"Top view L2 norm {top_l2:.2f} exceeds threshold {THRESHOLD}",
        )
        self.assertLessEqual(
            side_l2,
            THRESHOLD,
            msg=f"Side view 2 L2 norm {side_l2:.2f} exceeds threshold {THRESHOLD}",
        )

        # Clean up Blender server and temp directory.
        blender_server.stop()
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Check for collisions.
        self._check_collisions(scene=scene, context="after placing mugs")

        console_logger.info(
            f"Test completed successfully! Renders saved to {manipuland_test_dir}"
        )
        console_logger.info(f"Top view L2 norm: {top_l2:.4f} (threshold: {THRESHOLD})")
        console_logger.info(
            f"Side view 2 L2 norm: {side_l2:.4f} (threshold: {THRESHOLD})"
        )

    def test_multi_surface_placement(self):
        """Test placing mugs on artistic multi-surface shelf with z-rotation."""
        # Set up test data paths.
        # Use test_data root as scene_dir to allow assets from multiple subdirectories.
        test_data_root = Path(__file__).parent.parent / "test_data"
        realistic_scene_dir = test_data_root / "realistic_scene"
        floor_plan_path = realistic_scene_dir / "room_geometry.sdf"

        # Verify test data exists.
        if not floor_plan_path.exists():
            self.fail(f"Test data file not found: {floor_plan_path}")

        # Create 5x5m room centered at origin.
        floor_plan_tree = ET.parse(floor_plan_path)
        wall_normals = {
            "left_wall": np.array([1.0, 0.0]),
            "right_wall": np.array([-1.0, 0.0]),
            "back_wall": np.array([0.0, 1.0]),
            "front_wall": np.array([0.0, -1.0]),
        }

        room_geometry = RoomGeometry(
            sdf_tree=floor_plan_tree,
            sdf_path=floor_plan_path,
            wall_normals=wall_normals,
        )
        # Use test_data_root as scene_dir to allow shelf from support_surface_algorithm/.
        scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_root)

        # Load shelf asset (artistic multi-surface shelf).
        shelf_gltf = (
            Path(__file__).parent.parent
            / "test_data/support_surface_algorithm/artistic_shelf_canonical.gltf"
        )
        if not shelf_gltf.exists():
            self.fail(f"Shelf asset not found: {shelf_gltf}")

        # Create SDF for shelf (minimal SDF pointing to visual geometry).
        # Store in test_data_root to satisfy scene_dir path validation.
        shelf_sdf_content = f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="artistic_shelf">
    <link name="base_link">
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>{shelf_gltf.absolute()}</uri>
          </mesh>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>"""
        # Create temp directory for SDF file.
        test_temp_dir = Path(tempfile.mkdtemp())
        shelf_sdf = test_temp_dir / "artistic_shelf.sdf"
        shelf_sdf.write_text(shelf_sdf_content)

        # Create separate temp dir for logging (outside test_data).
        temp_dir = tempfile.mkdtemp()

        # Place shelf at origin with 25-degree Z-rotation.
        shelf_transform = RigidTransform(
            RollPitchYaw(0, 0, np.deg2rad(25.0)), np.array([0.0, 0.0, 0.0])
        )
        shelf_obj = SceneObject(
            object_id=scene.generate_unique_id("artistic_shelf"),
            object_type=ObjectType.FURNITURE,
            name="artistic_shelf",
            description="Artistic multi-surface shelf",
            transform=shelf_transform,
            sdf_path=shelf_sdf,
            geometry_path=shelf_gltf,
            # Approximate bounds from GLTF mesh data (actual bounds computed from mesh).
            bbox_min=np.array([-0.8, 0.0, -0.25]),
            bbox_max=np.array([0.8, 2.2, 0.25]),
        )
        scene.add_object(shelf_obj)

        # Extract shelf support surfaces.
        console_logger.info("Extracting support surfaces from artistic shelf...")
        support_surfaces = extract_and_propagate_support_surfaces(
            scene=scene,
            furniture_object=shelf_obj,
            config=None,
        )
        console_logger.info(
            f"Extracted {len(support_surfaces)} support surface(s) from shelf"
        )
        for i, surface in enumerate(support_surfaces):
            console_logger.info(
                f"  Surface {i}: ID={surface.surface_id}, "
                f"area={surface.area:.3f} m², "
                f"height={surface.transform.translation()[2]:.3f}m"
            )

        # Verify we found multiple surfaces (at least 2 for multi-surface rendering).
        self.assertGreaterEqual(
            len(support_surfaces),
            2,
            msg=f"Expected at least 2 surfaces for multi-surface test, found "
            f"{len(support_surfaces)}",
        )

        # Assign support surfaces to shelf for vision tools.
        shelf_obj.support_surfaces = support_surfaces

        # Load base configuration.
        config_path = (
            Path(__file__).parent.parent.parent
            / "configs/manipuland_agent/base_manipuland_agent.yaml"
        )
        base_cfg = OmegaConf.load(config_path)

        # Override config to disable debug visualization and reduce retries.
        test_overrides = {
            "rendering": {
                "annotations": {
                    "enable_support_surface_debug": False,
                    "enable_convex_hull_debug": False,  # Enable for visual debugging.
                },
                "retry_count": 3,  # Reduce retries for faster test failure.
            }
        }
        test_cfg = OmegaConf.merge(base_cfg, test_overrides)

        # Create test logger and VLM service.
        temp_logger = ConsoleLogger(output_dir=Path(temp_dir))
        vlm_service = VLMService()

        # Create AssetManager for testing.
        # collision_client=None because test doesn't generate collision geometry.
        asset_manager = AssetManager(
            logger=temp_logger,
            vlm_service=vlm_service,
            blender_server=None,
            collision_client=None,
            cfg=test_cfg,
            agent_type=AgentType.MANIPULAND,
        )

        # Create support surfaces dict for multi-surface API.
        support_surfaces_dict = {str(s.surface_id): s for s in support_surfaces}

        # Create ManipulandTools instance.
        manipuland_tools = ManipulandTools(
            scene=scene,
            asset_manager=asset_manager,
            cfg=test_cfg,
            current_furniture_id=shelf_obj.object_id,
            support_surfaces=support_surfaces_dict,
        )

        # Load coffee mug asset from test data and register it.
        coffee_mug_dir = (
            realistic_scene_dir
            / "generated_assets/manipulands/sdf/coffee_mug_1762992163"
        )
        coffee_mug_sdf = coffee_mug_dir / "coffee_mug.sdf"
        coffee_mug_gltf = coffee_mug_dir / "coffee_mug.gltf"

        if not coffee_mug_sdf.exists():
            self.fail(f"Coffee mug SDF not found: {coffee_mug_sdf}")
        if not coffee_mug_gltf.exists():
            self.fail(f"Coffee mug GLTF not found: {coffee_mug_gltf}")

        # Register the coffee mug asset with the asset manager.
        coffee_mug_id = UniqueID("coffee_mug_1762992163")
        coffee_mug_asset = SceneObject(
            object_id=coffee_mug_id,
            object_type=ObjectType.MANIPULAND,
            name="coffee_mug",
            description="white ceramic coffee mug",
            transform=RigidTransform(),
            sdf_path=coffee_mug_sdf,
            geometry_path=coffee_mug_gltf,
            bbox_min=np.array([-0.03758648, -0.05243881, 0.0]),
            bbox_max=np.array([0.03758648, 0.05243881, 0.08]),
        )
        asset_manager.registry.register(coffee_mug_asset)

        # Place mugs on different shelves to test multi-surface functionality.
        # Use specific surface indices to test color-coding across varied surfaces.
        placements = [
            {
                "surface_idx": 0,  # S_0
                "x": 0.2,
                "y": 0.15,
                "rotation": 0.0,
                "name": "mug_shelf_0",
            },
            {
                "surface_idx": 14,  # S_e (15th surface)
                "x": 0.0,
                "y": 0.0,
                "rotation": 45.0,
                "name": "mug_shelf_1",
            },
            {
                "surface_idx": 2,  # S_2
                "x": 0.0,
                "y": 0.0,
                "rotation": 90.0,
                "name": "mug_shelf_2",
            },
        ]

        placed_mugs = []
        for i, placement in enumerate(placements, 1):
            surface_idx = placement["surface_idx"]
            target_surface = support_surfaces[surface_idx]
            surface_id = target_surface.surface_id
            console_logger.info(
                f"Placing mug {i} on surface {surface_id} (index {surface_idx}) "
                f"at ({placement['x']}, {placement['y']}) with rotation "
                f"{placement['rotation']}°"
            )

            result_json = manipuland_tools._place_manipuland_on_surface_impl(
                asset_id=str(coffee_mug_id),
                surface_id=str(surface_id),
                position_x=placement["x"],
                position_z=placement["y"],
                rotation_degrees=placement["rotation"],
            )

            result = json.loads(result_json)
            console_logger.info(
                f"Placement result {i}: success={result.get('success')}, "
                f"message={result.get('message')}"
            )

            self.assertTrue(
                result["success"],
                msg=f"Mug {i} placement should succeed: {result.get('message')}",
            )

            # Store placed mug info for validation.
            placed_mugs.append(
                {
                    "object_id": result["object_id"],
                    "position": result["world_position"],
                    "surface_id": surface_id,
                    "surface_idx": surface_idx,
                    "placement_spec": placement,
                }
            )

        # Validate mugs are at different heights (on different shelves).
        console_logger.info("Validating mug positions on different shelves...")
        for i, mug_info in enumerate(placed_mugs, 1):
            mug_z = mug_info["position"]["z"]
            console_logger.info(
                f"Mug {i} Z-coordinate: {mug_z:.3f}m "
                f"(surface: {mug_info['surface_id']}, index: {mug_info['surface_idx']})"
            )

            # Mugs should be above floor (Z > 0.1m at minimum).
            self.assertGreater(
                mug_z,
                0.1,
                msg=f"Mug {i} should be on shelf (Z > 0.1m), not on floor. "
                f"Found Z={mug_z:.3f}m",
            )

        # Render scene using ManipulandVisionTools.
        console_logger.info("Rendering multi-surface scene...")

        # Create RenderingManager.
        rendering_manager = RenderingManager(
            cfg=test_cfg.rendering, logger=temp_logger, subdirectory="manipulands"
        )

        # Create and start Blender server for rendering.
        blender_server = BlenderServer(
            host="127.0.0.1",
            port_range=(8010, 8020),
        )
        blender_server.start()

        # Create ManipulandVisionTools.
        vision_tools = ManipulandVisionTools(
            scene=scene,
            rendering_manager=rendering_manager,
            cfg=test_cfg,
            current_furniture_id=shelf_obj.object_id,
            blender_server=blender_server,
        )

        # Call vision tool to render scene.
        result_message = vision_tools._observe_scene_impl()
        console_logger.info(f"Vision tool returned {len(result_message)} outputs")

        # Verify renders were created by RenderingManager.
        renders_base_dir = temp_logger.output_dir / "scene_renders" / "manipulands"
        self.assertTrue(
            renders_base_dir.exists(),
            msg=f"Renders should be created at {renders_base_dir}",
        )

        # Find the renders directory (will be renders_001).
        renders_dir = next(renders_base_dir.glob("renders_*"))

        # Copy renders to test output directory for manual inspection.
        test_output_dir = Path(__file__).parent / "test_outputs"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        multi_surface_test_dir = (
            test_output_dir / f"test_multi_surface_placement_{timestamp}"
        )
        multi_surface_test_dir.mkdir(parents=True, exist_ok=True)

        # Copy all rendered images.
        image_paths = []
        for img_path in sorted(renders_dir.glob("*.png")):
            dest = multi_surface_test_dir / f"shelf_{img_path.name}"
            shutil.copy(img_path, dest)
            image_paths.append(img_path)

        console_logger.info(f"Renders copied to {multi_surface_test_dir}")

        # Verify multi-surface rendering produced multiple top views.
        top_views = [p for p in image_paths if "_top_" in p.name]
        console_logger.info(
            f"Found {len(top_views)} top view(s): {[p.name for p in top_views]}"
        )

        # With >1 surface, should have multiple top views (one per surface).
        if len(support_surfaces) > 1:
            self.assertGreater(
                len(top_views),
                1,
                msg=f"Expected multiple top views for {len(support_surfaces)} surfaces, "
                f"found {len(top_views)}",
            )

            # Verify surface IDs are in filenames.
            for top_view in top_views:
                # File should be named like "0_top_S_0.png", "1_top_S_1.png", etc.
                self.assertIn(
                    "S_",
                    top_view.name,
                    msg=f"Top view filename should include surface ID: {top_view.name}",
                )

        # Compare rendered images against reference images using L2 norm.
        reference_dir = Path(__file__).parent / "reference_renders"

        # Find top view for S_2 (should be named like "2_top_S_2.png").
        top_view_s2 = next((p for p in image_paths if "top_S_2" in p.name), None)
        top_reference = reference_dir / "multi_surface_overlay_top_S_2.png"

        # Find side view 1 (named "1_side.png").
        side_view_1 = next((p for p in image_paths if "1_side" in p.name), None)
        side_reference = reference_dir / "multi_surface_overlay_side_1.png"

        # Find side view 2 (back view - should only show S_0 due to occlusion).
        side_view_2 = next((p for p in image_paths if "2_side" in p.name), None)
        side_2_reference = reference_dir / "multi_surface_overlay_side_2_back.png"

        # Verify we found the expected views.
        self.assertIsNotNone(
            top_view_s2,
            msg="Could not find top view for S_2 in rendered images",
        )
        self.assertIsNotNone(
            side_view_1,
            msg="Could not find side view 1 in rendered images",
        )
        self.assertIsNotNone(
            side_view_2,
            msg="Could not find side view 2 (back view) in rendered images",
        )

        # Verify reference images exist.
        self.assertTrue(
            top_reference.exists(),
            msg=f"Reference image not found: {top_reference}",
        )
        self.assertTrue(
            side_reference.exists(),
            msg=f"Reference image not found: {side_reference}",
        )
        self.assertTrue(
            side_2_reference.exists(),
            msg=f"Reference image not found: {side_2_reference}",
        )

        # Compare top view S_2.
        top_s2_l2 = compute_l2_norm_difference(
            img1_path=top_view_s2, img2_path=top_reference
        )
        console_logger.info(f"Top view S_2 L2 norm: {top_s2_l2:.4f}")

        # Compare side view 1.
        side_1_l2 = compute_l2_norm_difference(
            img1_path=side_view_1, img2_path=side_reference
        )
        console_logger.info(f"Side view 1 L2 norm: {side_1_l2:.4f}")

        # Compare side view 2 (back view - critical for occlusion regression).
        side_2_l2 = compute_l2_norm_difference(
            img1_path=side_view_2, img2_path=side_2_reference
        )
        console_logger.info(f"Side view 2 (back) L2 norm: {side_2_l2:.4f}")

        # Assert L2 norms are below threshold.
        THRESHOLD = 0.05  # Tight tolerance for deterministic rendering.

        self.assertLessEqual(
            top_s2_l2,
            THRESHOLD,
            msg=f"Top view S_2 L2 norm {top_s2_l2:.4f} exceeds threshold {THRESHOLD}",
        )
        self.assertLessEqual(
            side_1_l2,
            THRESHOLD,
            msg=f"Side view 1 L2 norm {side_1_l2:.4f} exceeds threshold {THRESHOLD}",
        )
        self.assertLessEqual(
            side_2_l2,
            THRESHOLD,
            msg=f"Side view 2 (back) L2 norm {side_2_l2:.4f} exceeds threshold "
            f"{THRESHOLD}. This may indicate a regression in surface occlusion.",
        )

        # Clean up Blender server and temp directories.
        blender_server.stop()
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(test_temp_dir, ignore_errors=True)

        console_logger.info(
            f"Multi-surface test completed! Renders saved to {multi_surface_test_dir}"
        )
        console_logger.info(
            f"Placed {len(placed_mugs)} mugs on {len(support_surfaces)} shelves"
        )
        console_logger.info(
            f"Top view S_2 L2 norm: {top_s2_l2:.4f} (threshold: {THRESHOLD})"
        )
        console_logger.info(
            f"Side view 1 L2 norm: {side_1_l2:.4f} (threshold: {THRESHOLD})"
        )
        console_logger.info(
            f"Side view 2 (back) L2 norm: {side_2_l2:.4f} (threshold: {THRESHOLD})"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
