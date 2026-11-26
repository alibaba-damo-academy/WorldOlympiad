"""Trusted PyBullet replay of task-bound DROID actions for a pour capability.

Candidates provide joint actions only. This module owns object loading, three-
dimensional dynamics and raw trajectory measurement outside candidate code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .robotics_contracts import TaskInstructionSpec
from .robotics_task_assets import resolve_task_assets
from .robotics_task_compiler import compile_task


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_actions(path: Path) -> list[list[float]]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "droid_joint_actions.v1":
        raise ValueError("pour candidate needs registered DROID joint actions")
    rows = payload.get("actions")
    if not isinstance(rows, list) or not 30 <= len(rows) <= 750:
        raise ValueError("pour action count must be in [30, 750]")
    actions = []
    for row in rows:
        if (not isinstance(row, list) or len(row) != 8
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in row)
                or not 0.0 <= row[7] <= 1.0):
            raise ValueError("pour action must be eight finite DROID coordinates")
        actions.append([float(value) for value in row])
    return actions


def resolve_pour_case(*, workspace: Path, task_path: Path,
                      generation_path: Path,
                      asset_revision: Path | None = None) -> dict[str, Any]:
    from .robotics_reference_materializer import (
        _object_orientations, _object_positions, _primary_support_id,
        normalized_object_layout,
    )
    workspace, task_path, generation_path = (Path(value).resolve() for value in
                                             (workspace, task_path, generation_path))
    output_root = workspace / "outputs"
    task_path.relative_to(output_root / "robotics_task_batches")
    generation_path.relative_to(output_root / "robotics_scenecode_generations")
    task = TaskInstructionSpec.model_validate_json(task_path.read_text())
    generation = json.loads(generation_path.read_text())
    if (generation.get("passed") is not True
            or generation.get("task_sha256") != _sha(task_path)
            or generation.get("case_id") != task.case_id):
        raise ValueError("pour replay task differs from certified SceneCode generation")
    plan = compile_task(task)
    controller = plan["controller_program"]
    if len(controller) != 2 or [row["operator"] for row in controller] != ["pick", "pour"]:
        raise ValueError("selected task does not declare one pick-then-pour program")
    pour = controller[1]
    source_id = pour["subjects"][0]
    target_id = pour["target"]
    contents_id = pour["constraints"].get("contents_object")
    if (not isinstance(contents_id, str) or not isinstance(target_id, str)
            or not any(row["subject"] == contents_id and row["object"] == target_id
                       and row["predicate"] in {"inside", "in_container"}
                       for row in plan["success_program"])):
        raise ValueError("pour has no independent contents-to-target predicate")
    inventory_path = generation_path.parent / "task_inventory_certificate.json"
    recovery_path = generation_path.parent / "task_object_recovery.json"
    if (generation.get("task_inventory_certificate_sha256") != _sha(inventory_path)
            or generation.get("task_object_recovery_sha256") != _sha(recovery_path)):
        raise ValueError("pour inventory or composite recovery changed")
    inventory = json.loads(inventory_path.read_text())
    recovery = json.loads(recovery_path.read_text())
    if inventory.get("passed") is not True:
        raise ValueError("pour task asset inventory is not admitted")
    matches = {row["task_object_id"]: row for row in inventory["matches"]}
    if set(matches) != {item.object_id for item in task.objects}:
        raise ValueError("pour inventory does not cover exact task objects")
    house_path = Path(inventory["house_state"]).resolve()
    house_path.relative_to(generation_path.parent)
    if _sha(house_path) != inventory["house_state_sha256"]:
        raise ValueError("pour house state changed")
    house = json.loads(house_path.read_text())
    rooms = house["rooms"]
    objects = {object_id: record for room in rooms.values()
               for object_id, record in room["objects"].items()}
    base_contents = matches[contents_id]["scene_object_id"]
    source_scene = matches[source_id]["scene_object_id"]
    groups = [row["assets"] for row in recovery.get("separated", [])
              if base_contents in row.get("assets", []) and source_scene in row.get("assets", [])]
    if len(groups) != 1:
        raise ValueError("pour contents do not have one certified composite expansion")
    geometry = objects[base_contents]["geometry_path"]
    source_content_instances = sorted(name for name in groups[0]
                                      if name != source_scene and name in objects
                                      and objects[name]["geometry_path"] == geometry)
    if len(source_content_instances) < 2 or base_contents not in source_content_instances:
        raise ValueError("pour contents lack multiple independent dynamic instances")
    # The renderer and model-native admission address the declared task object
    # id. Keep the other recovered pieces distinct, but bind the first physical
    # instance to that task id so the conditioning mask has an exact object key.
    content_instances = [contents_id, *(name for name in source_content_instances
                                        if name != base_contents)]
    support_id = _primary_support_id(task=task, matches=matches,
                                     house_state=house_path)
    assets = resolve_task_assets(workspace=workspace, generation=generation_path)
    positions = _object_positions(house_path)
    orientations = _object_orientations(house_path)
    target_scene = matches[target_id]["scene_object_id"]
    layout, layout_evidence = normalized_object_layout({
        source_id: positions[source_scene], target_id: positions[target_scene]})
    source_sdf = Path(assets.sdf_by_object_id[source_id])
    content_sdf = Path(assets.sdf_by_object_id[contents_id])
    revision_sha = None
    if asset_revision is not None:
        asset_revision = Path(asset_revision).resolve()
        asset_revision.relative_to(output_root.resolve())
        derivative_path = asset_revision / "candidate_manifest.json"
        if derivative_path.is_file():
            derivative = json.loads(derivative_path.read_text())
            variant = Path(derivative.get("candidate_sdf", "")).resolve()
            variant.relative_to(asset_revision)
            if (derivative.get("schema_version") != "worldolympiad_asset_derivative.v1"
                    or Path(derivative["source_sdf"]).resolve() != source_sdf
                    or derivative.get("source_sha256") != _sha(source_sdf)
                    or derivative.get("candidate_sha256") != _sha(variant)):
                raise ValueError("pour source asset derivative changed")
            for original in source_sdf.parent.iterdir():
                if original.is_file() and original != source_sdf:
                    copied = asset_revision / original.name
                    if not copied.is_file() or _sha(copied) != _sha(original):
                        raise ValueError("pour source derivative changed a shared mesh")
            source_sdf = variant
            revision_sha = _sha(derivative_path)
        else:
            manifests = []
            variants = {}
            for name, original in (("source", source_sdf), ("content", content_sdf)):
                directory = asset_revision / name
                manifest_path = directory / "physical_variant_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                variant = directory / original.name
                if (manifest.get("schema_version") != "scenecode_physical_scale_variant.v1"
                        or Path(manifest["source_sdf"]).resolve() != original
                        or manifest.get("changed_files") != [original.name]
                        or manifest.get("visual_collision_scale_coherent") is not True
                        or manifest["source_file_sha256"].get(original.name) != _sha(original)
                        or manifest["output_file_sha256"].get(original.name) != _sha(variant)):
                    raise ValueError("pour asset candidate is not a bound physical derivative")
                manifests.append((manifest_path, manifest))
                variants[name] = variant
            # The source body and its loose contents may require different sizes:
            # the gripper constrains the source, while native camera pixels constrain
            # each content instance. Physical fit is measured by _seat_contents.
            revision_sha = hashlib.sha256("".join(_sha(path) for path, _ in manifests).encode()).hexdigest()
            source_sdf, content_sdf = variants["source"], variants["content"]
    return {"task": task, "task_path": task_path, "generation_path": generation_path,
            "inventory_path": inventory_path, "recovery_path": recovery_path,
            "source_id": source_id, "target_id": target_id,
            "contents_id": contents_id, "content_instances": content_instances,
            "source_content_instances": source_content_instances,
            "source_xy": layout[source_id], "target_xy": layout[target_id],
            "source_orientation": orientations[source_scene],
            "target_orientation": orientations[target_scene],
            "content_sdf": content_sdf,
            "source_sdf": source_sdf,
            "asset_revision": asset_revision,
            "asset_revision_sha256": revision_sha,
            "target_sdf": Path(assets.sdf_by_object_id[target_id]),
            "table_sdf": Path(assets.sdf_by_object_id[support_id]),
            "robot_urdf": Path(assets.robot_urdf),
            "layout": layout_evidence}


def _seat_contents(scene, instance_ids: list[str]) -> dict[str, Any]:
    """Place distinct dynamic bodies inside the measured source cavity."""
    p = scene.p
    source_aabb = p.getAABB(scene.cup)
    lower, upper = np.asarray(source_aabb[0]), np.asarray(source_aabb[1])
    source_span = upper - lower
    axis = 0 if source_span[0] >= source_span[1] else 1
    cube_spans = [np.asarray(p.getAABB(scene.objects[name])[1])
                  - np.asarray(p.getAABB(scene.objects[name])[0])
                  for name in instance_ids]
    footprint = max(float(span[axis]) for span in cube_spans)
    spacing = footprint * 1.2
    packed_span = spacing * (len(instance_ids) - 1) + footprint
    if packed_span > 0.72 * source_span[axis]:
        raise ValueError("task contents cannot fit the certified source container: "
                         f"source_span={source_span.tolist()} packed_span={packed_span:.4f} "
                         f"count={len(instance_ids)}")
    center = (lower + upper) / 2.0
    placed = {}
    for index, name in enumerate(instance_ids):
        body = scene.objects[name]
        cube_aabb = p.getAABB(body)
        cube_center = (np.asarray(cube_aabb[0]) + np.asarray(cube_aabb[1])) / 2.0
        target_center = center.copy()
        target_center[axis] += (index - (len(instance_ids) - 1) / 2.0) * spacing
        target_center[2] = lower[2] + max(0.02, 0.12 * source_span[2]) + cube_spans[index][2] / 2.0
        base_position, orientation = p.getBasePositionAndOrientation(body)
        destination = np.asarray(base_position) + target_center - cube_center
        p.resetBasePositionAndOrientation(body, destination.tolist(), orientation)
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
        placed[name] = target_center.tolist()
    for _ in range(scene.physics_hz):
        scene._step_physics_once()
    scene.frames.clear()
    scene.physics_step = 0
    scene._reset_active_tracking()
    scene._record(phase="conditioning", source_window=None,
                  source_action_index=None, commanded_action=None,
                  clamped_joint_indices=[])
    settled = scene.frames[0]["objects"]
    for name in instance_ids:
        position = settled[name]["position"]
        if not (lower[0] < position[0] < upper[0]
                and lower[1] < position[1] < upper[1]
                and lower[2] < position[2] < upper[2]):
            raise ValueError("task contents did not settle inside source container")
    return {"source_aabb_initial": [lower.tolist(), upper.tolist()],
            "placement_centers": placed}


def build_scene_inputs(case: dict[str, Any], output_dir: Path):
    """One frozen DROID scene binding for candidate planning and trusted replay."""
    from scenecode_pybullet_bridge.droid_pick_place import (
        OFFICIAL_DROID_FORCES, OFFICIAL_DROID_RESET,
    )
    from scenecode_pybullet_bridge.policy_scene import PolicySceneInputs, PolicySceneObject

    # The bridge settles its initial scene before this runner can seat the
    # contents. Keep the still-unseated bodies clear of the container during
    # that internal settle; _seat_contents then resets and records frame zero.
    additional = tuple(PolicySceneObject(
        object_id=name, sdf_path=case["content_sdf"],
        center_xy=(2.0 + index * 0.15, 2.0), role="manipuland")
        for index, name in enumerate(case["content_instances"]))
    return PolicySceneInputs(
        table_sdf=case["table_sdf"], cup_sdf=case["source_sdf"],
        tray_sdf=case["target_sdf"], panda_urdf=case["robot_urdf"],
        output_dir=output_dir, robot_embodiment="robolab_droid",
        initial_arm_joint_positions=OFFICIAL_DROID_RESET,
        disable_robot_gravity=True, arm_position_gain=0.5,
        arm_joint_forces=OFFICIAL_DROID_FORCES,
        action_authority="candidate DROID actions replayed by trusted PyBullet",
        cup_center_xy=case["source_xy"], tray_center_xy=case["target_xy"],
        cup_orientation_xyzw=case["source_orientation"],
        tray_orientation_xyzw=case["target_orientation"],
        success_predicate="contact", subject_id=case["source_id"],
        target_id=case["target_id"], additional_objects=additional)


def replay(*, workspace: Path, task_path: Path, generation_path: Path,
           actions_path: Path, output_dir: Path,
           asset_revision: Path | None = None) -> Path:
    case = resolve_pour_case(workspace=workspace, task_path=task_path,
                             generation_path=generation_path,
                             asset_revision=asset_revision)
    actions_path, output_dir = actions_path.resolve(), output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("trusted pour replay output already exists")
    actions = load_actions(actions_path)
    from scenecode_pybullet_bridge.policy_scene import PyBulletDroidPolicyScene
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = build_scene_inputs(case, output_dir)
    with PyBulletDroidPolicyScene(inputs) as scene:
        placement = _seat_contents(scene, case["content_instances"])
        for index, action in enumerate(actions):
            scene.execute_action(action, source_window=-1,
                                 source_action_index=index,
                                 phase_override="candidate_action")
        simulator_result = scene.write_outputs()
        contacts = {name: len(scene.p.getContactPoints(
            bodyA=scene.objects[name], bodyB=scene.tray))
            for name in case["content_instances"]}
        final_aabbs = {name: [list(bound) for bound in scene.p.getAABB(scene.objects[name])]
                       for name in case["content_instances"]}
        target_aabb = [list(bound) for bound in scene.p.getAABB(scene.tray)]
    trajectory = output_dir / "trajectory.json"
    manifest = output_dir / "trusted_replay.json"
    manifest.write_text(json.dumps({
        "schema_version": "worldolympiad_trusted_pour_replay.v1",
        "task_sha256": _sha(case["task_path"]),
        "generation_sha256": _sha(case["generation_path"]),
        "inventory_sha256": _sha(case["inventory_path"]),
        "recovery_sha256": _sha(case["recovery_path"]),
        "asset_revision": str(case["asset_revision"]) if case["asset_revision"] else None,
        "asset_revision_sha256": case["asset_revision_sha256"],
        "actions_sha256": _sha(actions_path),
        "trajectory_sha256": _sha(trajectory),
        "source_id": case["source_id"], "target_id": case["target_id"],
        "contents_id": case["contents_id"],
        "content_instances": case["content_instances"],
        "layout": case["layout"], "initial_placement": placement,
        "final_content_aabbs": final_aabbs,
        "final_target_aabb": target_aabb,
        "final_content_target_contacts": contacts,
        "object_attachment_constraint_count": simulator_result.object_attachment_constraint_count,
        "gripper_coupling_constraint_count": simulator_result.gripper_coupling_constraint_count,
        "action_count": len(actions),
        "candidate_verdict_ignored": True,
    }, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-revision", type=Path)
    args = parser.parse_args()
    print(replay(workspace=args.workspace, task_path=args.task,
                 generation_path=args.generation,
                 actions_path=args.actions, output_dir=args.output,
                 asset_revision=args.asset_revision))


if __name__ == "__main__":
    main()
