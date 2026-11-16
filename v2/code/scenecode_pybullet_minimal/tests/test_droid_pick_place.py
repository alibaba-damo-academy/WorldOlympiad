from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scenecode_pybullet_bridge.droid_pick_place import (
    CARRY_CLEARANCE_OVER_DESTINATION_M,
    MAXIMUM_BASE_OFFSET_Z_M,
    MINIMUM_BASE_OFFSET_Z_M,
    OFFICIAL_DROID_FORCES,
    OFFICIAL_DROID_RESET,
    PHASE_ACTION_COUNTS,
    SUPPORTED_SEMANTIC_OPERATORS,
    DroidPickPlaceInputs,
    axis_alignment_quaternion_xyzw,
    carry_transfer_target_z,
    compensated_place_end_effector_target,
    placement_descent_is_supported,
    resolve_grasp_waypoints,
    run_droid_pick_place,
    stack_subject_link_target_from_aabbs,
)
from scenecode_pybullet_bridge.policy_scene import (
    PyBulletDroidPolicyScene,
    inserted_geometry_passed,
)


class DroidPickPlaceContractTest(unittest.TestCase):
    def test_insert_accepts_partial_or_complete_target_volume_occupancy(self) -> None:
        common = {
            "center_inside_target_xy": True,
            "target_contact": True,
            "target_bottom_z_m": 0.90,
            "target_top_z_m": 1.00,
            "alignment_cosine": 0.99,
        }
        self.assertTrue(
            inserted_geometry_passed(
                **common, subject_bottom_z_m=0.96, subject_top_z_m=1.06
            )
        )
        self.assertTrue(
            inserted_geometry_passed(
                **common, subject_bottom_z_m=0.91, subject_top_z_m=0.98
            )
        )
        self.assertFalse(
            inserted_geometry_passed(
                **common, subject_bottom_z_m=0.80, subject_top_z_m=0.89
            )
        )
        self.assertFalse(
            inserted_geometry_passed(
                **common, subject_bottom_z_m=0.99, subject_top_z_m=1.09
            )
        )

    def test_axis_alignment_rotates_arbitrary_primary_axis_to_world_up(self) -> None:
        quaternion = axis_alignment_quaternion_xyzw((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        x, y, z, w = quaternion
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        np.testing.assert_allclose(rotation @ (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), atol=1e-9)

    def test_terminal_alignment_requires_both_axes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = DroidPickPlaceInputs(
                table_sdf=root / "missing_table.sdf",
                cup_sdf=root / "missing_cup.sdf",
                tray_sdf=root / "missing_tray.sdf",
                panda_urdf=root / "missing_robot.urdf",
                output_dir=root / "output",
                terminal_alignment_axis_local=(1.0, 0.0, 0.0),
            )
            with self.assertRaisesRegex(ValueError, "both local and world axes"):
                run_droid_pick_place(inputs)

    def test_place_target_compensates_measured_subject_from_gripper_offset(self) -> None:
        target, measured_offset = compensated_place_end_effector_target(
            current_end_effector_position=(0.20, -0.10, 0.60),
            current_subject_link_position=(0.215, -0.122, 0.455),
            target_link_position=(-0.04, 0.03, 0.90),
            target_vertical_offset_m=0.025,
        )
        np.testing.assert_allclose(measured_offset, (0.015, -0.022, -0.145))
        np.testing.assert_allclose(target, (-0.055, 0.052, 1.07))
        np.testing.assert_allclose(target + measured_offset, (-0.04, 0.03, 0.925))

    def test_carry_transfer_clears_a_tall_destination_top_face(self) -> None:
        # A bottle carried on a 0.345 m pinched offset would still hang below a
        # 1.737 m basket rim at the compiled 1.992 m lift, dragging it through
        # the weave.  The transfer must rise to the measured top plus clearance.
        transfer_z, evidence = carry_transfer_target_z(
            compiled_lift_z_m=1.992419029486713,
            measured_end_effector_z_m=1.9924218654632568,
            measured_subject_bottom_z_m=1.6452488962824958,
            measured_target_top_z_m=1.7373863063306798,
        )
        required = (
            1.7373863063306798
            + CARRY_CLEARANCE_OVER_DESTINATION_M
            + (1.9924218654632568 - 1.6452488962824958)
        )
        self.assertAlmostEqual(evidence["required_transfer_z_m"], required)
        self.assertAlmostEqual(transfer_z, required)
        self.assertGreater(transfer_z, evidence["compiled_lift_z_m"])

    def test_carry_transfer_keeps_the_compiled_lift_when_it_already_clears(self) -> None:
        transfer_z, evidence = carry_transfer_target_z(
            compiled_lift_z_m=1.99,
            measured_end_effector_z_m=1.99,
            measured_subject_bottom_z_m=1.645,
            measured_target_top_z_m=1.20,
        )
        self.assertAlmostEqual(transfer_z, 1.99)
        self.assertLessEqual(evidence["required_transfer_z_m"], transfer_z)

    def test_carry_transfer_rejects_degenerate_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            carry_transfer_target_z(
                compiled_lift_z_m=float("nan"),
                measured_end_effector_z_m=1.0,
                measured_subject_bottom_z_m=0.6,
                measured_target_top_z_m=0.5,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            carry_transfer_target_z(
                compiled_lift_z_m=1.0,
                measured_end_effector_z_m=1.0,
                measured_subject_bottom_z_m=0.6,
                measured_target_top_z_m=0.5,
                clearance_m=-0.01,
            )

    def test_placement_descent_holds_only_after_contact_stops_the_subject(self) -> None:
        # Contact plus a stalled subject means the destination carries it.
        self.assertTrue(
            placement_descent_is_supported(
                previous_subject_bottom_z_m=1.5754,
                subject_bottom_z_m=1.5753,
                in_contact=True,
            )
        )
        # A wall graze that still lets the subject fall is not support.
        self.assertFalse(
            placement_descent_is_supported(
                previous_subject_bottom_z_m=1.80,
                subject_bottom_z_m=1.78,
                in_contact=True,
            )
        )
        # A stalled subject with no target contact is a hanging carry.
        self.assertFalse(
            placement_descent_is_supported(
                previous_subject_bottom_z_m=1.80,
                subject_bottom_z_m=1.80,
                in_contact=False,
            )
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            placement_descent_is_supported(
                previous_subject_bottom_z_m=float("inf"),
                subject_bottom_z_m=1.0,
                in_contact=True,
            )

    def test_base_offset_envelope_bounds_are_ordered(self) -> None:
        self.assertGreater(MINIMUM_BASE_OFFSET_Z_M, 0.0)
        self.assertGreater(MAXIMUM_BASE_OFFSET_Z_M, MINIMUM_BASE_OFFSET_Z_M)
        self.assertGreater(CARRY_CLEARANCE_OVER_DESTINATION_M, 0.0)

    def test_grasp_offset_at_the_adapter_ceiling_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = DroidPickPlaceInputs(
                table_sdf=root / "missing_table.sdf",
                cup_sdf=root / "missing_cup.sdf",
                tray_sdf=root / "missing_tray.sdf",
                panda_urdf=root / "missing_robot.urdf",
                output_dir=root / "output",
                grasp_base_offset_z_m=MAXIMUM_BASE_OFFSET_Z_M,
            )
            with self.assertRaisesRegex(ValueError, "base-height offsets"):
                run_droid_pick_place(inputs)

    def test_stack_target_uses_runtime_top_and_subject_bottom_offset(self) -> None:
        target, evidence = stack_subject_link_target_from_aabbs(
            subject_link_position=(0.10, 0.20, 1.10),
            subject_aabb=((0.09, 0.19, 1.08), (0.11, 0.21, 1.12)),
            target_aabb=((-0.06, -0.04, 0.90), (-0.02, 0.02, 0.94)),
            release_clearance_m=0.003,
        )
        np.testing.assert_allclose(target, (-0.04, -0.01, 0.963))
        self.assertAlmostEqual(evidence["subject_bottom_from_link_m"], -0.02)
        self.assertAlmostEqual(evidence["target_top_z_m"], 0.94)
        self.assertAlmostEqual(evidence["release_clearance_m"], 0.003)

    def test_timeline_matches_one_conditioning_plus_222_actions(self) -> None:
        self.assertEqual(sum(PHASE_ACTION_COUNTS.values()), 222)
        self.assertEqual(len(OFFICIAL_DROID_RESET), 7)
        self.assertEqual(len(OFFICIAL_DROID_FORCES), 7)
        self.assertTrue(np.isfinite(OFFICIAL_DROID_RESET).all())
        self.assertEqual(
            PHASE_ACTION_COUNTS,
            {
                "settle": 15,
                "approach": 30,
                "descend": 24,
                "grasp": 18,
                "lift": 24,
                "transfer": 36,
                "lower": 24,
                "release": 18,
                "retreat_vertical": 18,
                "retreat_back": 6,
                "final_settle": 9,
            },
        )

    def test_invalid_height_contract_fails_before_loading_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = DroidPickPlaceInputs(
                table_sdf=root / "missing_table.sdf",
                cup_sdf=root / "missing_cup.sdf",
                tray_sdf=root / "missing_tray.sdf",
                panda_urdf=root / "missing_robot.urdf",
                output_dir=root / "output",
                grasp_base_offset_z_m=0.40,
                hover_base_offset_z_m=0.30,
                lift_base_offset_z_m=0.43,
            )
            with self.assertRaisesRegex(ValueError, "base-height offsets"):
                run_droid_pick_place(inputs)

    def test_invalid_grasp_yaw_fails_before_loading_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = DroidPickPlaceInputs(
                table_sdf=root / "missing_table.sdf",
                cup_sdf=root / "missing_cup.sdf",
                tray_sdf=root / "missing_tray.sdf",
                panda_urdf=root / "missing_robot.urdf",
                output_dir=root / "output",
                grasp_yaw_deg=181.0,
            )
            with self.assertRaisesRegex(ValueError, "grasp yaw"):
                run_droid_pick_place(inputs)

    def test_invalid_place_target_offset_fails_before_loading_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = DroidPickPlaceInputs(
                table_sdf=root / "missing_table.sdf",
                cup_sdf=root / "missing_cup.sdf",
                tray_sdf=root / "missing_tray.sdf",
                panda_urdf=root / "missing_robot.urdf",
                output_dir=root / "output",
                place_target_offset_z_m=0.51,
            )
            with self.assertRaisesRegex(ValueError, "place target Z offset"):
                run_droid_pick_place(inputs)

    def test_invalid_place_target_xy_offset_fails_before_loading_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = DroidPickPlaceInputs(
                table_sdf=root / "missing_table.sdf",
                cup_sdf=root / "missing_cup.sdf",
                tray_sdf=root / "missing_tray.sdf",
                panda_urdf=root / "missing_robot.urdf",
                output_dir=root / "output",
                place_target_offset_xy_m=(float("nan"), 0.0),
            )
            with self.assertRaisesRegex(ValueError, "target XY offset"):
                run_droid_pick_place(inputs)

    def test_invalid_target_frame_count_fails_before_loading_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = DroidPickPlaceInputs(
                table_sdf=root / "missing_table.sdf",
                cup_sdf=root / "missing_cup.sdf",
                tray_sdf=root / "missing_tray.sdf",
                panda_urdf=root / "missing_robot.urdf",
                output_dir=root / "output",
                target_frame_count=222,
            )
            with self.assertRaisesRegex(ValueError, "target frame count"):
                run_droid_pick_place(inputs)

    def test_invalid_success_predicate_fails_before_loading_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = DroidPickPlaceInputs(
                table_sdf=root / "missing_table.sdf",
                cup_sdf=root / "missing_cup.sdf",
                tray_sdf=root / "missing_tray.sdf",
                panda_urdf=root / "missing_robot.urdf",
                output_dir=root / "output",
                success_predicate="looks_good",
            )
            with self.assertRaisesRegex(ValueError, "success predicate"):
                run_droid_pick_place(inputs)

    def test_driver_maps_semantic_predicates_to_supported_physical_checks(self) -> None:
        driver = Path(__file__).parents[1] / "scripts" / "run_droid_pick_place.py"
        spec = importlib.util.spec_from_file_location("_droid_pick_place_driver", driver)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        aliases = module._SUCCESS_PREDICATE_ALIASES
        # The adapter advertises the semantic "inside" predicate, so the driver
        # must resolve it to the concrete physical container check.
        self.assertEqual(aliases["inside"], "in_container")
        self.assertEqual(aliases["in_container"], "in_container")
        supported = PyBulletDroidPolicyScene.supported_success_predicates
        self.assertTrue(set(aliases.values()) <= supported)

    def test_every_compiled_terminal_operator_reaches_the_expert(self) -> None:
        # The compiler emits place/stack/insert/cover as terminal operators for
        # ordinary pick-plus-terminal graphs; the adapter must accept them, not
        # only the semantic subset with a dedicated branch.
        self.assertTrue(
            {"place", "stack", "insert", "cover", "flip", "strike", "pour"}
            <= SUPPORTED_SEMANTIC_OPERATORS
        )
        driver = Path(__file__).parents[1] / "scripts" / "run_droid_pick_place.py"
        spec = importlib.util.spec_from_file_location("_droid_pick_place_driver", driver)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parser = module.build_parser()
        for operator in sorted(SUPPORTED_SEMANTIC_OPERATORS):
            args = parser.parse_args(
                [
                    "--table-sdf",
                    "table.sdf",
                    "--cup-sdf",
                    "cup.sdf",
                    "--tray-sdf",
                    "tray.sdf",
                    "--panda-urdf",
                    "robot.urdf",
                    "--output-dir",
                    "out",
                    "--semantic-operator",
                    operator,
                ]
            )
            self.assertEqual(args.semantic_operator, operator)

    def test_place_family_semantic_operators_pass_input_validation(self) -> None:
        # With missing assets the expert must fail while loading them, not while
        # rejecting the terminal operator the compiler produced.
        for operator in ("place", "stack", "insert", "cover"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs = DroidPickPlaceInputs(
                    table_sdf=root / "missing_table.sdf",
                    cup_sdf=root / "missing_cup.sdf",
                    tray_sdf=root / "missing_tray.sdf",
                    panda_urdf=root / "missing_robot.urdf",
                    output_dir=root / "output",
                    semantic_operator=operator,
                )
                with self.assertRaises(FileNotFoundError):
                    run_droid_pick_place(inputs)

    def test_upright_certification_requires_terminal_alignment_axes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = DroidPickPlaceInputs(
                table_sdf=root / "missing_table.sdf",
                cup_sdf=root / "missing_cup.sdf",
                tray_sdf=root / "missing_tray.sdf",
                panda_urdf=root / "missing_robot.urdf",
                output_dir=root / "output",
                terminal_upright_min_cosine=0.99,
            )
            with self.assertRaisesRegex(ValueError, "requires terminal alignment axes"):
                run_droid_pick_place(inputs)

    def test_upright_certification_min_cosine_must_be_a_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = DroidPickPlaceInputs(
                table_sdf=root / "missing_table.sdf",
                cup_sdf=root / "missing_cup.sdf",
                tray_sdf=root / "missing_tray.sdf",
                panda_urdf=root / "missing_robot.urdf",
                output_dir=root / "output",
                terminal_alignment_axis_local=(0.0, 0.0, 1.0),
                terminal_alignment_axis_world=(0.0, 0.0, 1.0),
                terminal_upright_min_cosine=0.0,
            )
            with self.assertRaisesRegex(ValueError, "must lie in"):
                run_droid_pick_place(inputs)

    def test_vertical_waypoints_anchor_on_measured_aabb_bottom(self) -> None:
        # Generated SDFs put the root link anywhere in the body: an AABB whose
        # bottom sits above the root origin (root at the body centre) or below
        # it (inertial pose outside the geometry).  The compiled offsets are
        # measured from the lowest collision point, so the pinch follows the
        # AABB bottom instead of the origin.
        waypoints = resolve_grasp_waypoints(
            subject_base_xy=(0.10, -0.20),
            subject_aabb=((0.05, -0.25, 0.97), (0.15, -0.15, 1.03)),
            grasp_offset_xy_m=(0.0, 0.0),
            grasp_base_offset_z_m=0.12,
            hover_base_offset_z_m=0.38,
            lift_base_offset_z_m=0.43,
            semantic_operator="place",
        )
        self.assertAlmostEqual(waypoints["subject_bottom_z_m"], 0.97)
        self.assertAlmostEqual(waypoints["grasp_z_m"], 1.09)
        self.assertAlmostEqual(waypoints["hover_z_m"], 1.35)
        self.assertAlmostEqual(waypoints["lift_z_m"], 1.40)
        # The lateral site stays in the measured base frame and is clamped into
        # the collision footprint rather than trusting a stale pre-settle pose.
        shifted = resolve_grasp_waypoints(
            subject_base_xy=(0.10, -0.20),
            subject_aabb=((0.05, -0.25, 0.97), (0.15, -0.15, 1.03)),
            grasp_offset_xy_m=(0.30, 0.0),
            grasp_base_offset_z_m=0.12,
            hover_base_offset_z_m=0.38,
            lift_base_offset_z_m=0.43,
            semantic_operator="pour",
        )
        self.assertLessEqual(shifted["grasp_xy"][0], 0.15 - 0.002 + 1e-12)
        self.assertAlmostEqual(shifted["grasp_xy"][1], -0.20, places=9)

    def test_flip_and_strike_grasp_the_measured_body_centre(self) -> None:
        waypoints = resolve_grasp_waypoints(
            subject_base_xy=(0.10, -0.20),
            subject_aabb=((0.05, -0.25, 0.97), (0.15, -0.15, 1.03)),
            grasp_offset_xy_m=(0.30, 0.25),
            grasp_base_offset_z_m=0.12,
            hover_base_offset_z_m=0.38,
            lift_base_offset_z_m=0.43,
            semantic_operator="strike",
        )
        np.testing.assert_allclose(waypoints["grasp_xy"], (0.10, -0.20))
        self.assertAlmostEqual(waypoints["grasp_z_m"], 1.0)
        self.assertAlmostEqual(waypoints["hover_z_m"], 1.21)
        self.assertAlmostEqual(waypoints["lift_z_m"], 1.27)


if __name__ == "__main__":
    unittest.main()
