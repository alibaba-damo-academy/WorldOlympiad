"""Experimental task-bound DROID pour action proposal.

This is an isolated, human-authored candidate. It emits actions only; the
WorldOlympiad trusted replay and independent pour verifier own every verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoresearch.robotics_pour_replay import (
    _seat_contents, build_scene_inputs, resolve_pour_case,
)
from autoresearch.robotics_reference_materializer import pick_place_execution_geometry
from scenecode_pybullet_bridge import DroidPickPlaceInputs, run_droid_pick_place
from scenecode_pybullet_bridge.droid_pick_place import PHASE_ACTION_COUNTS
from scenecode_pybullet_bridge.policy_scene import PyBulletDroidPolicyScene
from scripts.render_scenecode_asset_identity import sdf_collision_bounds


def propose(*, workspace: Path, task: Path, generation: Path,
            asset_revision: Path, output: Path) -> Path:
    workspace, output = workspace.resolve(), output.resolve()
    case = resolve_pour_case(workspace=workspace, task_path=task,
                             generation_path=generation,
                             asset_revision=asset_revision)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("candidate output already exists")
    output.mkdir(parents=True)
    proposal = json.loads((asset_revision / "candidate_manifest.json").read_text())
    grasp_offset = proposal["grasp_site_local_xy_m"]
    grasp_yaw = float(proposal["grasp_yaw_deg"])
    source_low, source_high = sdf_collision_bounds(case["source_sdf"])
    target_low, target_high = sdf_collision_bounds(case["target_sdf"])
    geometry = pick_place_execution_geometry(
        subject_height_m=float(source_high[2] - source_low[2]),
        destination_height_m=float(target_high[2] - target_low[2]),
        terminal_operator="place")
    pickup = DroidPickPlaceInputs(
        table_sdf=case["table_sdf"], cup_sdf=case["source_sdf"],
        tray_sdf=case["target_sdf"], panda_urdf=case["robot_urdf"],
        output_dir=output / "registered_pickup",
        cup_center_xy=case["source_xy"], tray_center_xy=case["target_xy"],
        cup_orientation_xyzw=case["source_orientation"],
        tray_orientation_xyzw=case["target_orientation"],
        grasp_yaw_deg=grasp_yaw,
        grasp_offset_xy_m=(float(grasp_offset[0]), float(grasp_offset[1])),
        grasp_base_offset_z_m=geometry["grasp_base_offset_z_m"],
        hover_base_offset_z_m=geometry["hover_base_offset_z_m"],
        lift_base_offset_z_m=geometry["lift_base_offset_z_m"],
        place_target_offset_z_m=geometry["place_target_offset_z_m"],
        semantic_operator="place", success_predicate="in_container")
    pickup_result = run_droid_pick_place(pickup)
    base_actions = np.load(Path(pickup_result.actions_path))
    prefix_count = sum(PHASE_ACTION_COUNTS[name] for name in
                       ("settle", "approach", "descend", "grasp", "lift"))
    actions = [row.tolist() for row in base_actions[:prefix_count]]
    planning = output / "planning"
    with PyBulletDroidPolicyScene(build_scene_inputs(case, planning)) as scene:
        _seat_contents(scene, case["content_instances"])
        for index, action in enumerate(actions):
            scene.execute_action(action, source_window=-1,
                                 source_action_index=index, phase_override="pickup")
        p = scene.p
        if len(scene.frames[-1]["robot_cup_contact_links"]) < 2:
            raise RuntimeError("registered grasp lost dual contact before pour planning")

        def ee_pose():
            return p.getLinkState(scene.panda, scene.end_effector,
                                  computeForwardKinematics=True)[4:6]

        def append_pose(target_position, target_orientation, count, phase):
            start_position, start_orientation = ee_pose()
            start = np.asarray(start_position, dtype=float)
            target = np.asarray(target_position, dtype=float)
            for index in range(count):
                fraction = (index + 1) / count
                smooth = fraction * fraction * (3 - 2 * fraction)
                point = start + (target - start) * smooth
                orientation = p.getQuaternionSlerp(start_orientation,
                                                   target_orientation, smooth)
                joint_target = p.calculateInverseKinematics(
                    scene.panda, scene.end_effector, point.tolist(), orientation,
                    maxNumIterations=300, residualThreshold=1e-6)
                if len(joint_target) < 7 or not np.isfinite(joint_target[:7]).all():
                    raise RuntimeError("candidate IK failed")
                action = [*map(float, joint_target[:7]), 1.0]
                scene.execute_action(action, source_window=-1,
                                     source_action_index=len(actions),
                                     phase_override=phase)
                actions.append(action)

        def hold(count, grip, phase):
            for _ in range(count):
                action = [*map(float, scene.joint_position()[:7]), float(grip)]
                scene.execute_action(action, source_window=-1,
                                     source_action_index=len(actions),
                                     phase_override=phase)
                actions.append(action)

        def rotate_over_target(target_orientation, count, phase):
            """Move the gripper pivot while holding the container over the bin."""
            ee_position, start_orientation = ee_pose()
            source_position, source_orientation = p.getBasePositionAndOrientation(scene.cup)
            inverse_ee = p.invertTransform(ee_position, start_orientation)
            source_from_ee = p.multiplyTransforms(
                inverse_ee[0], inverse_ee[1], source_position, source_orientation)[0]
            held_source_center = np.asarray(source_position, dtype=float)
            for index in range(count):
                fraction = (index + 1) / count
                smooth = fraction * fraction * (3 - 2 * fraction)
                orientation = p.getQuaternionSlerp(
                    start_orientation, target_orientation, smooth)
                rotated_offset = p.multiplyTransforms(
                    (0, 0, 0), orientation, source_from_ee, (0, 0, 0, 1))[0]
                point = held_source_center - np.asarray(rotated_offset)
                joint_target = p.calculateInverseKinematics(
                    scene.panda, scene.end_effector, point.tolist(), orientation,
                    maxNumIterations=300, residualThreshold=1e-6)
                if len(joint_target) < 7 or not np.isfinite(joint_target[:7]).all():
                    raise RuntimeError("candidate anchored tilt IK failed")
                action = [*map(float, joint_target[:7]), 1.0]
                scene.execute_action(action, source_window=-1,
                                     source_action_index=len(actions),
                                     phase_override=phase)
                actions.append(action)

        first_source = np.asarray(p.getBasePositionAndOrientation(scene.cup)[0])
        first_ee, _ = ee_pose()
        bin_initial = np.asarray(p.getBasePositionAndOrientation(scene.tray)[0])
        source_bottom = float(p.getAABB(scene.cup)[0][2])
        target_aabb = np.asarray(p.getAABB(scene.tray), dtype=float)
        clearance = max(0.12, float(target_aabb[1, 2] + 0.12 - source_bottom))
        append_pose([*first_ee[:2], float(first_ee[2] + clearance)],
                    ee_pose()[1], 30, "clearance_lift")

        current_ee, upright_orientation = ee_pose()
        current_source = np.asarray(p.getBasePositionAndOrientation(scene.cup)[0])
        source_from_ee = current_source - np.asarray(current_ee)
        target_center = 0.5 * (target_aabb[0] + target_aabb[1])
        desired_source_xy = np.asarray([
            target_aabb[0, 0] + 0.30 * (target_aabb[1, 0] - target_aabb[0, 0]),
            target_center[1],
        ])
        desired_ee_xy = desired_source_xy - source_from_ee[:2]
        append_pose([*desired_ee_xy, current_ee[2]], upright_orientation,
                    40, "over_target")
        bin_after_transfer = np.asarray(p.getBasePositionAndOrientation(scene.tray)[0])
        if np.linalg.norm(bin_after_transfer - bin_initial) > 0.05:
            raise RuntimeError("target moved before the pour: path clearance failed")

        position, orientation = ee_pose()
        # The opening must point substantially downward. At ~90 degrees,
        # gravity only pins contents against the side wall of an open bin.
        tip = p.getQuaternionFromEuler((0.0, np.deg2rad(155.0), 0.0))
        tilted_orientation = p.multiplyTransforms(
            (0, 0, 0), tip, (0, 0, 0), orientation)[1]
        rotate_over_target(tilted_orientation, 40, "pour_tilt")
        hold(45, 1.0, "pour_hold")
        # Leave the bin before turning the wide source upright. Untipping it
        # above the bin swept its rim into the target and displaced the bin.
        current_ee, current_orientation = ee_pose()
        append_pose([current_ee[0], current_ee[1], current_ee[2] + 0.15],
                    current_orientation, 25, "clearance_after_pour")
        current_ee, current_orientation = ee_pose()
        current_source = np.asarray(p.getBasePositionAndOrientation(scene.cup)[0])
        source_from_ee = current_source - np.asarray(current_ee)
        desired_ee_xy = np.asarray(case["source_xy"]) - source_from_ee[:2]
        append_pose([*desired_ee_xy, current_ee[2]], current_orientation,
                    40, "return_source")
        rotate_over_target(upright_orientation, 35, "recover_upright")
        current_ee, _ = ee_pose()
        source_bottom = float(p.getAABB(scene.cup)[0][2])
        table_top = float(p.getAABB(scene.table)[1][2])
        append_pose([current_ee[0], current_ee[1],
                     current_ee[2] + table_top + 0.01 - source_bottom],
                    current_orientation, 30, "lower_source")
        hold(18, 0.0, "release_source")
        current_ee, _ = ee_pose()
        append_pose([current_ee[0], current_ee[1], current_ee[2] + 0.12],
                    current_orientation, 18, "retreat")
        hold(18, 0.0, "final_settle")
        scene.write_outputs()
    path = output / "actions.json"
    path.write_text(json.dumps({"schema_version": "droid_joint_actions.v1",
                                "actions": actions}, sort_keys=True) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--asset-revision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(propose(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
