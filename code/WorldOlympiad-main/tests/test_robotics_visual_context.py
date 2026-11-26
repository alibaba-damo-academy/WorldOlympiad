"""Targeted verification for the rendered in-context visual review gate.

The gate is the persistent runtime consumer of the household appearance /
proportion guidance: ``scripts/package_robotics_reference.py`` calls
``certify_reference_visual_context`` for every published reference and
``autoresearch/robotics_contracts.py`` refuses to publish a v6 manifest without
a passing certificate.  These tests exercise the deterministic part of the gate
on synthetic registered labels, so the pixel-measurement, fail-closed and
actionable-feedback behaviour is verified without a simulator or a network call.
"""

from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch.robotics_visual_context import (  # noqa: E402
    VisionReviewUnavailable,
    build_in_context_montage,
    certify_reference_visual_context,
    cie_lstar,
    measure_in_context_appearance,
    rgb_distance,
    vision_context_review,
    _vision_verdict_passed,
    articulated_host_visibility,
)


def test_same_asset_alias_does_not_need_mutual_colour_separation() -> None:
    verdict = {
        "objects_recognizable": {name: True for name in ("bottle", "table", "table_surface")},
        "object_material_verdicts": {
            name: {"material_reads_authentically": True}
            for name in ("bottle", "table", "table_surface")
        },
        "materials_contextual": True,
        "materials_self_consistent": True,
        "proportions_plausible": True,
        "cross_object_separation": {
            "bottle|table": "strong",
            "bottle|table_surface": "strong",
            "table|table_surface": "none",
        },
    }
    alias = [{"object_ids": ["table", "table_surface"]}]
    passed, problems, notes = _vision_verdict_passed(
        verdict, object_ids=["bottle", "table", "table_surface"],
        same_asset_aliases=alias,
    )
    assert passed and not problems
    assert any("inventory-certified" in note for note in notes)
    verdict["cross_object_separation"]["bottle|table"] = "none"
    passed, problems, _ = _vision_verdict_passed(
        verdict, object_ids=["bottle", "table", "table_surface"],
        same_asset_aliases=alias,
    )
    assert not passed
    assert any("bottle|table separation is none" in problem for problem in problems)


def test_unconfigured_vision_route_is_explicitly_deferred(tmp_path: Path) -> None:
    montage = tmp_path / "montage.png"
    _write_image(montage, (80, 120, 160))
    with mock.patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
        with unittest.TestCase().assertRaisesRegex(
            VisionReviewUnavailable, "unconfigured"
        ):
            vision_context_review(montage=montage, prompt="inspect")


def test_gateway_timeout_is_recorded_as_vision_unavailable(tmp_path: Path) -> None:
    montage = tmp_path / "montage.png"
    _write_image(montage, (80, 120, 160))
    timeout = urllib.error.HTTPError(
        "https://api.openai.com/v1/responses", 524, "Gateway timeout", {},
        io.BytesIO(b"upstream timed out"),
    )
    with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
            mock.patch("urllib.request.urlopen", side_effect=timeout):
        with unittest.TestCase().assertRaisesRegex(
            VisionReviewUnavailable, "HTTP 524"
        ):
            vision_context_review(montage=montage, prompt="inspect")


def test_articulated_component_is_measured_against_its_host(tmp_path: Path) -> None:
    render = _render_fixture(
        tmp_path, subject_rgb=(30, 60, 120),
        support_rgb=(120, 120, 120), destination_rgb=(230, 230, 225),
    )
    for view in render["views"].values():
        view["conditioning_semantic"]["object_values"] = {
            "cabinet_door": 1, "cabinet_host": 3,
        }
    task = SimpleNamespace(
        instruction="Pull the cabinet door open.",
        objects=[SimpleNamespace(
            object_id="cabinet_host", role="destination",
            visual_description="A light cabinet body.",
        )],
        articulation_components=[SimpleNamespace(
            component_id="cabinet_door", host_object_id="cabinet_host",
            visual_description="A dark cabinet door.",
        )],
    )
    path = certify_reference_visual_context(
        task=task, trajectory=_trajectory(), render=render,
        output=tmp_path / "certificate", vision="disabled",
    )
    report = json.loads(path.read_text())
    assert report["measurement_policy"] == "visible_shape_or_colour_and_articulated_host_exposure_v3"
    assert report["rendered_appearance"]["checks"] == {
        "separated:cabinet_host": True,
        "separated:cabinet_door": True,
    }
    assert report["checks"]["articulated_host_visible:cabinet_door"]


def test_articulated_host_cannot_disappear_behind_door() -> None:
    component = SimpleNamespace(
        component_id="cabinet_door", host_object_id="cabinet_host"
    )
    appearance = {"objects": {
        "top": {"cabinet_host": {"pixel_count": 1200},
                "cabinet_door": {"pixel_count": 1000}},
        "exterior_left": {"cabinet_host": {"pixel_count": 135},
                          "cabinet_door": {"pixel_count": 1323}},
        "exterior_right": {"cabinet_host": {"pixel_count": 478},
                           "cabinet_door": {"pixel_count": 1729}},
    }}
    report = articulated_host_visibility(appearance, [component])
    assert not report["cabinet_door"]["passed"]
    appearance["objects"]["exterior_right"]["cabinet_host"]["pixel_count"] = 600
    assert articulated_host_visibility(appearance, [component])["cabinet_door"]["passed"]


def test_rendered_signature_reports_rgb_channel_order(tmp_path: Path) -> None:
    render = _render_fixture(
        tmp_path, subject_rgb=(200, 40, 10),
        support_rgb=(100, 100, 100), destination_rgb=(10, 40, 200),
    )
    report = measure_in_context_appearance(
        render, _trajectory(), views=VIEWS,
        object_roles={"subject": "manipuland", "destination": "destination"},
    )
    # Fixture colours are assigned in OpenCV's BGR channel order.
    assert report["objects"]["top"]["subject"]["mean_rgb_8bit"] == [10.0, 40.0, 200.0]


PHASES = ("initial", "grasp", "carry", "final")
VIEWS = ("top", "exterior_left", "exterior_right")


def _write_image(path: Path, colour: tuple[int, int, int], size: int = 32) -> None:
    image = np.full((size, size, 3), (24, 24, 24), dtype=np.uint8)
    image[8:24, 8:24] = colour
    assert cv2.imwrite(str(path), image)


def _render_fixture(root: Path, *, subject_rgb, support_rgb, destination_rgb) -> dict:
    """One synthetic registered render manifest with three labelled objects."""

    frames = []
    for index, phase in enumerate(PHASES):
        image_path = root / f"frame_{index:03d}.png"
        _write_image(image_path, (24, 24, 24))
        frames.append(
            {
                "frame": index,
                "time_s": index / 15.0,
                "phase": phase,
                "image": str(image_path),
            }
        )
    # Registered label mask covering the three task objects plus the gripper.
    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[4:12, 4:12] = 1
    mask[4:12, 18:26] = 2
    mask[18:28, 4:14] = 3
    mask[18:28, 18:28] = 4
    mask_path = root / "labels.png"
    assert cv2.imwrite(str(mask_path), mask)
    colour = {
        1: subject_rgb,
        2: support_rgb,
        3: destination_rgb,
        4: (200, 200, 200),
    }
    labelled = np.full((32, 32, 3), (24, 24, 24), dtype=np.uint8)
    for value, rgb in colour.items():
        labelled[mask == value] = rgb
    frame_zero = Path(frames[0]["image"])
    assert cv2.imwrite(str(frame_zero), labelled)
    views = {
        view_id: {
            "size_wh": [32, 32],
            "frames": list(frames),
            "conditioning_semantic": {
                "frame": 0,
                "image": str(mask_path),
                "object_values": {
                    "subject": 1,
                    "support": 2,
                    "destination": 3,
                },
                "end_effector_value": 4,
            },
        }
        for view_id in VIEWS
    }
    return {"schema_version": "scenecode_robotics_multiview_render.v4", "views": views}


def _trajectory() -> dict:
    return {
        "frames": [
            {"frame": index, "phase": phase} for index, phase in enumerate(PHASES)
        ]
    }


def _task() -> SimpleNamespace:
    return SimpleNamespace(
        instruction="Pick up the kettle and place it upright on the coaster.",
        objects=[
            SimpleNamespace(
                object_id="subject",
                role="manipuland",
                visual_description="A realistic household kettle.",
            ),
            SimpleNamespace(
                object_id="support",
                role="support",
                visual_description="An ordinary household coffee table.",
            ),
            SimpleNamespace(
                object_id="destination",
                role="destination",
                visual_description="A plain, thin, flat household coaster.",
            ),
        ],
    )


class VisualContextGateTests(unittest.TestCase):
    def test_helpers_measure_lightness_and_distance(self) -> None:
        self.assertGreater(cie_lstar((255, 255, 255)), cie_lstar((0, 0, 0)))
        self.assertGreater(rgb_distance((0, 0, 255), (255, 224, 189)), 0.5)

    def test_near_identical_palette_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The delivered baseline palette: all three objects warm greys.
            render = _render_fixture(
                root,
                subject_rgb=(120, 120, 112),
                support_rgb=(122, 122, 115),
                destination_rgb=(124, 122, 118),
            )
            report = measure_in_context_appearance(
                render,
                _trajectory(),
                views=VIEWS,
                object_roles={"subject": "manipuland", "support": "support", "destination": "destination"},
            )
            self.assertFalse(report["passed"])
            self.assertTrue(
                any(not value for value in report["checks"].values())
            )
            with self.assertRaises(ValueError):
                certify_reference_visual_context(
                    task=_task(),
                    trajectory=_trajectory(),
                    render=render,
                    output=root / "cert",
                    vision="disabled",
                )

    def test_montage_uses_real_rendered_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render = _render_fixture(
                root,
                subject_rgb=(90, 40, 10),
                support_rgb=(60, 120, 160),
                destination_rgb=(240, 235, 215),
            )
            montage = build_in_context_montage(
                render, _trajectory(), root, views=VIEWS
            )
            self.assertTrue(Path(montage["path"]).is_file())
            self.assertEqual(montage["frame_indices"]["initial"], 0)
            self.assertEqual(montage["frame_indices"]["final"], len(PHASES) - 1)
            image = cv2.imread(str(montage["path"]))
            self.assertIsNotNone(image)
            # three view rows and four phase columns of 32x32 tiles
            self.assertEqual(image.shape[0], 3 * 32)
            self.assertEqual(image.shape[1], 4 * 32)

    def test_montage_pads_phase_rows_when_closeups_are_wider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            render = _render_fixture(root, subject_rgb=(90, 40, 10),
                support_rgb=(60, 120, 160), destination_rgb=(240, 235, 215))
            source={"view_id":"top","frame":0,"bbox_xyxy":[0,0,10,10],"pixels":100}
            crop={"image":np.zeros((32,70,3),dtype=np.uint8),"scale":1.0,"label":"x"}
            with mock.patch("autoresearch.robotics_visual_context.closeup_source",return_value=source), \
                 mock.patch("autoresearch.robotics_visual_context.closeup_crop",return_value=crop):
                montage=build_in_context_montage(render,_trajectory(),root,views=VIEWS,
                    object_roles={"subject":"manipuland","destination":"destination"})
            image=cv2.imread(str(montage["path"]))
            self.assertEqual(image.shape[1],140)
            self.assertEqual(image.shape[0],4*32)


if __name__ == "__main__":
    unittest.main()
