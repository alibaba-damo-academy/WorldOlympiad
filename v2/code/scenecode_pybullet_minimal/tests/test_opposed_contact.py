"""Regression tests for the opposed-contact carry criterion.

The carry certificate must reject a carry that is not a real opposed pinch.  The
original test projected the two finger contacts onto the jaw axis relative to the
subject's root link origin.  That projection is exact only while the grasp site
is close to the root link: for a tall household object carried by a handle the
reference distance multiplies any residual swing of the subject inside the grip.

These tests pin the corrected pair of criteria:
  * the projection margin is still computed and recorded, and
  * the reference-frame-independent contact-normal opposition is what a carry
    must satisfy, including the same-side "carry" that the original test was
    written to catch.
"""

from __future__ import annotations

import pytest

import numpy as np

from scenecode_pybullet_bridge.policy_scene import (
    opposed_normal_margin,
    opposed_projection_margin,
)


FINGER_NAMES = ["left_inner_finger", "right_inner_finger"]
FINGER_LINKS = [11, 14]


def _frame(
    *,
    link_position,
    left_position,
    right_position,
    left_normal,
    right_normal,
    left_finger,
    right_finger,
):
    return {
        "robot": {
            "finger_link_positions": {
                "left_inner_finger": list(left_finger),
                "right_inner_finger": list(right_finger),
            }
        },
        "cup": {"link_position": list(link_position)},
        "robot_cup_contact_evidence": [
            {
                "robot_link_index": 11,
                "position_on_cup_world": list(left_position),
                "normal_on_cup_world": list(left_normal),
                "normal_force_n": 100.0,
                "contact_distance_m": 0.0,
            },
            {
                "robot_link_index": 14,
                "position_on_cup_world": list(right_position),
                "normal_on_cup_world": list(right_normal),
                "normal_force_n": 100.0,
                "contact_distance_m": 0.0,
            },
        ],
    }


def test_opposed_pinch_passes_both_criteria() -> None:
    # Jaws close along +Y; each pad presses a bar face from the opposite side.
    frame = _frame(
        link_position=(0.0, 0.0, 0.0),
        left_position=(0.0, -0.015, 0.26),
        right_position=(0.0, 0.015, 0.26),
        left_normal=(0.0, -1.0, 0.0),
        right_normal=(0.0, 1.0, 0.0),
        left_finger=(0.0, -0.036, 1.10),
        right_finger=(0.0, 0.036, 1.10),
    )
    assert opposed_normal_margin(
        frame, finger_names=FINGER_NAMES, finger_links=FINGER_LINKS
    ) == 1.0
    assert opposed_projection_margin(
        frame, finger_names=FINGER_NAMES, finger_links=FINGER_LINKS
    ) > 0.004


def test_same_side_pinch_is_rejected_by_the_normal_test() -> None:
    # One finger inside a hollow rim, one outside on the same rim: both contact
    # normals point the same way, so this is not an opposed pinch even though
    # the two contacts exist and straddle the root link origin.
    frame = _frame(
        link_position=(0.0, 0.0, 0.0),
        left_position=(0.0, -0.015, 0.26),
        right_position=(0.0, 0.015, 0.26),
        left_normal=(0.0, 1.0, 0.0),
        right_normal=(0.0, 1.0, 0.0),
        left_finger=(0.0, -0.036, 1.10),
        right_finger=(0.0, 0.036, 1.10),
    )
    assert opposed_normal_margin(
        frame, finger_names=FINGER_NAMES, finger_links=FINGER_LINKS
    ) < 0.0
    # The geometric proxy cannot tell the difference, which is exactly why the
    # normal test is required as well.
    assert opposed_projection_margin(
        frame, finger_names=FINGER_NAMES, finger_links=FINGER_LINKS
    ) > 0.004


def test_tall_subject_swing_keeps_normal_opposition_but_breaks_projection() -> None:
    # The subject is pinched 0.26 m above its root link and has swung 10 degrees
    # inside the grip: the contacts and normals are fixed on the subject, so the
    # normals stay opposed, while the distant root link moves out of the contact
    # span and the projection proxy fails.
    angle = np.radians(10.0)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    bar_center_from_root = np.array([0.0, 0.0, 0.26])
    swipe = rotation @ bar_center_from_root
    left_local = np.array([0.0, -0.015, 0.0])
    right_local = np.array([0.0, 0.015, 0.0])
    left_normal_local = np.array([0.0, -1.0, 0.0])
    right_normal_local = np.array([0.0, 1.0, 0.0])
    frame = _frame(
        # Root link stays at the origin; the swung subject moves relative to it.
        link_position=(0.0, 0.0, 0.0),
        left_position=list(swipe + rotation @ left_local),
        right_position=list(swipe + rotation @ right_local),
        left_normal=list(rotation @ left_normal_local),
        right_normal=list(rotation @ right_normal_local),
        left_finger=(0.0, -0.036, 1.10),
        right_finger=(0.0, 0.036, 1.10),
    )
    normal_margin = opposed_normal_margin(
        frame, finger_names=FINGER_NAMES, finger_links=FINGER_LINKS
    )
    projection_margin = opposed_projection_margin(
        frame, finger_names=FINGER_NAMES, finger_links=FINGER_LINKS
    )
    assert normal_margin > 0.5
    assert normal_margin == pytest.approx(np.cos(angle))
    assert projection_margin < 0.0


def test_margins_are_invariant_to_a_global_frame_rotation() -> None:
    # Rotating the whole gripper + subject rigidly (expressing the same episode
    # in another world frame) must not change either margin: a certificate may
    # not depend on the coordinate frame the episode happens to live in.
    def rot_x(a):
        return np.array(
            [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
        )

    def rot_y(a):
        return np.array(
            [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        )

    def rot_z(a):
        return np.array(
            [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
        )

    angles = np.radians((17.0, -23.0, 41.0))
    rotation = rot_z(angles[2]) @ rot_y(angles[1]) @ rot_x(angles[0])

    base = _frame(
        link_position=(0.0, 0.0, 0.0),
        left_position=(0.0, -0.015, 0.26),
        right_position=(0.0, 0.015, 0.26),
        left_normal=(0.0, -1.0, 0.0),
        right_normal=(0.0, 1.0, 0.0),
        left_finger=(0.0, -0.036, 1.10),
        right_finger=(0.0, 0.036, 1.10),
    )
    rotated = {
        "robot": {
            "finger_link_positions": {
                name: list(rotation @ np.asarray(value))
                for name, value in base["robot"]["finger_link_positions"].items()
            }
        },
        "cup": {
            "link_position": list(
                rotation @ np.asarray(base["cup"]["link_position"])
            )
        },
        "robot_cup_contact_evidence": [
            {
                **item,
                "position_on_cup_world": list(
                    rotation @ np.asarray(item["position_on_cup_world"])
                ),
                "normal_on_cup_world": list(
                    rotation @ np.asarray(item["normal_on_cup_world"])
                ),
            }
            for item in base["robot_cup_contact_evidence"]
        ],
    }
    for frame in (base, rotated):
        assert opposed_normal_margin(
            frame, finger_names=FINGER_NAMES, finger_links=FINGER_LINKS
        ) == pytest.approx(1.0)
    assert opposed_projection_margin(
        rotated, finger_names=FINGER_NAMES, finger_links=FINGER_LINKS
    ) == pytest.approx(
        opposed_projection_margin(
            base, finger_names=FINGER_NAMES, finger_links=FINGER_LINKS
        )
    )

