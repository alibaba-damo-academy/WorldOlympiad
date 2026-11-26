import logging
import shutil
import time
import unittest
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np

from omegaconf import OmegaConf
from PIL import Image as PILImage
from pydrake.all import RigidTransform, RollPitchYaw

from scenecode.agent_utils.blender.server_manager import BlenderServer
from scenecode.agent_utils.house import RoomGeometry
from scenecode.agent_utils.rendering import render_scene_for_agent_observation
from scenecode.agent_utils.room import ObjectType, RoomScene, SceneObject, UniqueID

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


class TestRenderingAnnotations(unittest.TestCase):
    """Integration test for rendering with overlay annotations.

    Tests the complete annotation pipeline including:
    - Set-of-mark labels (cyan rectangles with object names)
    - Bounding box wireframes (cyan)
    - Direction arrows (yellow)
    - Dense metric markers (coordinate dots)
    - Partial wall hiding (top view hides all, side views hide facing walls)

    Uses a realistic scene.
    """

    def test_realistic_scene_with_annotations(self):
        """Test rendering with realistic scene from test data.

        Uses a realistic scene to verify annotations work with complex, realistic
        geometry.
        """
        # Set up test data paths.
        test_data_dir = Path(__file__).parent.parent / "test_data" / "realistic_scene"
        floor_plan_path = test_data_dir / "room_geometry.sdf"

        # Verify test data exists.
        if not floor_plan_path.exists():
            self.fail(f"Test data file not found: {floor_plan_path}")

        # Create test scene with realistic office layout.
        floor_plan_tree = ET.parse(floor_plan_path)

        # Compute wall normals for the 5x5m room centered at origin.
        # Room geometry: left_wall at x=-2.5, right_wall at x=2.5,
        # back_wall at y=-2.5, front_wall at y=2.5.
        # Room-facing normals point from wall center toward room center (0, 0).
        wall_normals = {
            "left_wall": np.array([1.0, 0.0]),  # Points right (toward room center).
            "right_wall": np.array([-1.0, 0.0]),  # Points left.
            "back_wall": np.array([0.0, 1.0]),  # Points forward.
            "front_wall": np.array([0.0, -1.0]),  # Points backward.
        }

        # Create wall SceneObjects for the 5x5m room (walls at ±2.5m).
        wall_height = 2.5
        wall_thickness = 0.05
        walls = []

        # Wall specs: (name, center_x, center_y, bbox_width, bbox_depth).
        wall_specs = [
            ("left_wall", -2.5, 0.0, wall_thickness, 5.0),
            ("right_wall", 2.5, 0.0, wall_thickness, 5.0),
            ("back_wall", 0.0, -2.5, 5.0, wall_thickness),
            ("front_wall", 0.0, 2.5, 5.0, wall_thickness),
        ]

        for name, cx, cy, w, d in wall_specs:
            wall_obj = SceneObject(
                object_id=UniqueID(name),
                object_type=ObjectType.WALL,
                name=name,
                description=f"Room {name}",
                transform=RigidTransform(p=[cx, cy, wall_height / 2.0]),
                bbox_min=np.array([-w / 2, -d / 2, -wall_height / 2]),
                bbox_max=np.array([w / 2, d / 2, wall_height / 2]),
                immutable=True,
            )
            walls.append(wall_obj)

        room_geometry = RoomGeometry(
            sdf_tree=floor_plan_tree,
            sdf_path=floor_plan_path,
            walls=walls,
            wall_normals=wall_normals,
        )
        scene = RoomScene(room_geometry=room_geometry, scene_dir=test_data_dir)

        # Add walls to scene.
        for wall in room_geometry.walls:
            scene.add_object(wall)

        # Add work desk 1 (left desk).
        desk1_sdf = (
            test_data_dir / "generated_assets/sdf/work_desk_1761578426/work_desk.sdf"
        )
        if not desk1_sdf.exists():
            self.fail(f"Asset not found: {desk1_sdf}")

        desk1_transform = RigidTransform(
            RollPitchYaw(0, 0, np.deg2rad(177.88211711796515)),
            np.array([-0.9007158897038726, 2.0158714047391593, 0.0]),
        )
        desk1_obj = SceneObject(
            object_id=scene.generate_unique_id("work_desk"),
            object_type=ObjectType.FURNITURE,
            name="work_desk",
            description="Left work desk",
            transform=desk1_transform,
            sdf_path=desk1_sdf,
            bbox_min=np.array([-0.75, -0.40, 0.0]),
            bbox_max=np.array([0.75, 0.40, 0.75]),
        )
        scene.add_object(desk1_obj)

        # Add work desk 2 (right desk).
        desk2_transform = RigidTransform(
            RollPitchYaw(0, 0, np.deg2rad(179.46903325513824)),
            np.array([0.9306640131256646, 2.056572144807821, 0.0]),
        )
        desk2_obj = SceneObject(
            object_id=scene.generate_unique_id("work_desk"),
            object_type=ObjectType.FURNITURE,
            name="work_desk",
            description="Right work desk",
            transform=desk2_transform,
            sdf_path=desk1_sdf,
            bbox_min=np.array([-0.75, -0.40, 0.0]),
            bbox_max=np.array([0.75, 0.40, 0.75]),
        )
        scene.add_object(desk2_obj)

        # Add office chair 1 (left chair).
        chair_sdf = (
            test_data_dir
            / "generated_assets/sdf/office_chair_1761578426/office_chair.sdf"
        )
        if not chair_sdf.exists():
            self.fail(f"Asset not found: {chair_sdf}")

        chair1_transform = RigidTransform(
            RollPitchYaw(0, 0, np.deg2rad(0.3827468506781701)),
            np.array([-0.9390650553576015, 1.0558966670872212, 0.0]),
        )
        chair1_obj = SceneObject(
            object_id=scene.generate_unique_id("office_chair"),
            object_type=ObjectType.FURNITURE,
            name="office_chair",
            description="Left office chair",
            transform=chair1_transform,
            sdf_path=chair_sdf,
            bbox_min=np.array([-0.35, -0.35, 0.0]),
            bbox_max=np.array([0.35, 0.35, 1.0]),
        )
        scene.add_object(chair1_obj)

        # Add office chair 2 (right chair).
        chair2_transform = RigidTransform(
            RollPitchYaw(0, 0, np.deg2rad(-0.32474451276576466)),
            np.array([0.8867592644814793, 1.1223604081323186, 0.0]),
        )
        chair2_obj = SceneObject(
            object_id=scene.generate_unique_id("office_chair"),
            object_type=ObjectType.FURNITURE,
            name="office_chair",
            description="Right office chair",
            transform=chair2_transform,
            sdf_path=chair_sdf,
            bbox_min=np.array([-0.35, -0.35, 0.0]),
            bbox_max=np.array([0.35, 0.35, 1.0]),
        )
        scene.add_object(chair2_obj)

        # Add printer.
        printer_sdf = (
            test_data_dir / "generated_assets/sdf/printer_1761578426/printer.sdf"
        )
        if not printer_sdf.exists():
            self.fail(f"Asset not found: {printer_sdf}")

        printer_transform = RigidTransform(
            RollPitchYaw(0, 0, np.deg2rad(179.07295554448734)),
            np.array([-1.9335813090688696, 1.8095589189833219, 0.0]),
        )
        printer_obj = SceneObject(
            object_id=scene.generate_unique_id("printer"),
            object_type=ObjectType.FURNITURE,
            name="printer",
            description="Office printer",
            transform=printer_transform,
            sdf_path=printer_sdf,
            bbox_min=np.array([-0.35, -0.40, 0.0]),
            bbox_max=np.array([0.35, 0.40, 0.906]),
        )
        scene.add_object(printer_obj)

        # King bed.
        king_bed_sdf = (
            test_data_dir / "generated_assets/sdf/king_bed_1761578419/king_bed.sdf"
        )
        king_bed_transform = RigidTransform(
            RollPitchYaw(0, 0, 0),
            np.array([0.0, -0.718, 0.0]),
        )
        king_bed_obj = SceneObject(
            object_id=scene.generate_unique_id("king_bed"),
            object_type=ObjectType.FURNITURE,
            name="king_bed",
            description="Modern king-sized bed with upholstered headboard",
            transform=king_bed_transform,
            sdf_path=king_bed_sdf,
            bbox_min=np.array([-1.0, -1.2320592403411865, 0.0]),
            bbox_max=np.array([1.0, 1.2320592403411865, 1.2834309339523315]),
        )
        scene.add_object(king_bed_obj)

        # Create rendering config with ALL annotations enabled.
        rendering_cfg = OmegaConf.create(
            {
                "layout": "top_plus_sides",
                "top_view_width": 1024,
                "top_view_height": 1024,
                "side_view_count": 4,
                "side_view_width": 512,
                "side_view_height": 512,
                "background_color": [1.0, 1.0, 1.0],
                "server_startup_delay": 0.1,
                "port_cleanup_delay": 0.1,
                "annotations": {
                    "enable_set_of_mark_labels": True,
                    "enable_bounding_boxes": True,
                    "enable_direction_arrows": True,
                    "enable_partial_walls": True,
                    "enable_support_surface_debug": False,
                    "enable_convex_hull_debug": False,
                },
            }
        )

        # Create output directory for visual inspection.
        test_output_dir = Path(__file__).parent / "test_outputs"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        realistic_test_dir = test_output_dir / f"realistic_scene_test_{timestamp}"
        realistic_test_dir.mkdir(parents=True, exist_ok=True)

        # Create and start Blender server for rendering.
        blender_server = BlenderServer(
            host="127.0.0.1",
            port_range=(8010, 8020),
        )
        blender_server.start()

        # Render scene with annotations.
        try:
            image_paths = render_scene_for_agent_observation(
                scene=scene,
                cfg=rendering_cfg,
                blender_server=blender_server,
            )

            # Verify rendering produces correct number of views (1 top + 4 side).
            self.assertEqual(len(image_paths), 5)

            # Copy rendered images to test output directory for visual inspection.
            for img_path in image_paths:
                self.assertTrue(img_path.exists(), f"Image not found: {img_path}")
                self.assertGreater(
                    img_path.stat().st_size, 0, f"Empty image file: {img_path}"
                )

                # Copy to test output directory.
                dest_path = realistic_test_dir / img_path.name
                shutil.copy(img_path, dest_path)

                # Load and verify image has expected dimensions.
                img = PILImage.open(img_path)
                img_array = np.array(img)

                # Check if it's top or side view based on filename.
                if "top" in img_path.name:
                    expected_height, expected_width = 1024, 1024
                else:
                    expected_height, expected_width = 512, 512

                # Verify spatial dimensions (accept both RGB and RGBA).
                self.assertEqual(
                    img_array.shape[:2],
                    (expected_height, expected_width),
                    f"Unexpected image dimensions for {img_path.name}",
                )
                # Verify color channels (3 for RGB or 4 for RGBA).
                self.assertIn(
                    img_array.shape[2],
                    [3, 4],
                    f"Unexpected channel count for {img_path.name}",
                )

                # Verify image contains non-uniform content.
                channels = img_array.shape[2]
                unique_pixels = len(np.unique(img_array.reshape(-1, channels), axis=0))
                self.assertGreater(
                    unique_pixels,
                    10,
                    f"Image {img_path.name} should contain varied pixel content",
                )

            # Compare rendered images against reference images using L2 norm.
            reference_dir = Path(__file__).parent / "reference_renders"

            # Find top view (0_top.png).
            top_view = next(p for p in image_paths if "0_top" in p.name)
            top_reference = reference_dir / "furniture_overlay_top.png"

            # Find side view 2 (2_side.png).
            side_view_2 = next(p for p in image_paths if "2_side" in p.name)
            side_reference = reference_dir / "furniture_overlay_side_2.png"

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
            top_l2 = compute_l2_norm_difference(
                img1_path=top_view, img2_path=top_reference
            )
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

            # Log path for visual inspection.
            console_logger.info(
                f"Realistic scene renders saved to: {realistic_test_dir.absolute()}"
            )
            console_logger.info(
                "Inspect these images to verify annotations work with realistic geometry."
            )
            console_logger.info(
                f"Top view L2 norm: {top_l2:.4f} (threshold: {THRESHOLD})"
            )
            console_logger.info(
                f"Side view 2 L2 norm: {side_l2:.4f} (threshold: {THRESHOLD})"
            )

        except Exception as e:
            self.fail(f"Realistic scene rendering failed: {e}")
        finally:
            blender_server.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
