"""Independent task-predicate admission of a trusted DROID pour replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .robotics_pour_replay import resolve_pour_case


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _position(frame: dict, object_id: str) -> np.ndarray:
    value = np.asarray(frame["objects"][object_id]["position"], dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("trusted replay contains an invalid object position")
    return value


def _vertical_axis(frame: dict, object_id: str) -> np.ndarray:
    x, y, z, w = frame["objects"][object_id]["orientation_xyzw"]
    values = np.asarray([x, y, z, w], dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or not np.isclose(
            np.linalg.norm(values), 1.0, atol=1e-3):
        raise ValueError("trusted replay contains an invalid orientation")
    # Third column of the quaternion rotation matrix.
    return np.asarray([2 * (x * z + w * y),
                       2 * (y * z - w * x),
                       1 - 2 * (x * x + y * y)])


def verify(*, workspace: Path, task_path: Path, generation_path: Path,
           replay_path: Path, output_path: Path) -> dict:
    replay_path, output_path = replay_path.resolve(), output_path.resolve()
    replay = json.loads(replay_path.read_text())
    case = resolve_pour_case(workspace=workspace, task_path=task_path,
                             generation_path=generation_path,
                             asset_revision=Path(replay["asset_revision"])
                                 if replay.get("asset_revision") else None)
    trajectory_path = replay_path.parent / "trajectory.json"
    expected = {"task_sha256": _sha(case["task_path"]),
                "generation_sha256": _sha(case["generation_path"]),
                "inventory_sha256": _sha(case["inventory_path"]),
                "recovery_sha256": _sha(case["recovery_path"]),
                "trajectory_sha256": _sha(trajectory_path)}
    if (replay.get("schema_version") != "worldolympiad_trusted_pour_replay.v1"
            or any(replay.get(key) != value for key, value in expected.items())
            or replay.get("content_instances") != case["content_instances"]
            or replay.get("asset_revision_sha256") != case["asset_revision_sha256"]
            or replay.get("source_id") != case["source_id"]
            or replay.get("target_id") != case["target_id"]):
        raise ValueError("trusted pour replay is not bound to the frozen task and assets")
    trajectory = json.loads(trajectory_path.read_text())
    frames = trajectory.get("frames")
    if not isinstance(frames, list) or len(frames) != replay["action_count"] + 1:
        raise ValueError("trusted pour replay has an incomplete action timeline")
    source_id = case["source_id"]
    contents = case["content_instances"]
    initial, terminal = frames[0], frames[-1]
    source_positions = np.asarray([_position(frame, source_id) for frame in frames])
    target_positions = np.asarray([_position(frame, case["target_id"]) for frame in frames])
    content_positions = {name: np.asarray([_position(frame, name) for frame in frames])
                         for name in contents}
    first_source = source_positions[0]
    target_aabb = np.asarray(replay["final_target_aabb"], dtype=float)
    initial_source_aabb = np.asarray(replay["initial_placement"]["source_aabb_initial"],
                                     dtype=float)
    if (target_aabb.shape != (2, 3) or initial_source_aabb.shape != (2, 3)
            or not np.isfinite(target_aabb).all()
            or not np.isfinite(initial_source_aabb).all()):
        raise ValueError("trusted pour geometry is incomplete")
    initial_contents_inside = all(
        np.all(content_positions[name][0, :2] > initial_source_aabb[0, :2])
        and np.all(content_positions[name][0, :2] < initial_source_aabb[1, :2])
        and initial_source_aabb[0, 2] < content_positions[name][0, 2] < initial_source_aabb[1, 2]
        for name in contents)
    max_lift = float(np.max(source_positions[:, 2] - first_source[2]))
    initial_axis = _vertical_axis(initial, source_id)
    tilt_degrees = [math.degrees(math.acos(float(np.clip(
        np.dot(initial_axis, _vertical_axis(frame, source_id)), -1.0, 1.0))))
        for frame in frames]
    max_tilt = max(tilt_degrees)
    target_radius = 0.5 * float(np.linalg.norm(target_aabb[1, :2] - target_aabb[0, :2]))
    source_near_target = any(
        np.linalg.norm(source[:2] - target[:2]) < target_radius + 0.04
        and source[2] > first_source[2] + 0.06
        and tilt >= 45.0
        and len(frame.get("robot_cup_contact_links", [])) >= 2
        for frame, source, target, tilt in zip(
            frames, source_positions, target_positions, tilt_degrees))
    target_stable = bool(np.max(np.linalg.norm(
        target_positions - target_positions[0], axis=1)) < 0.05)
    robot_contact = any(
        len(frame.get("robot_cup_contact_links", [])) >= 2
        and any(float(row.get("normal_force_n", 0)) > 0
                for row in frame.get("robot_cup_contact_evidence", []))
        and tilt >= 25.0
        for frame, tilt in zip(frames, tilt_degrees))
    release = all(not frame.get("robot_cup_contact_links")
                  for frame in frames[-5:])
    final_inside = {}
    final_stable = {}
    free_of_source = {}
    plausible_dynamics = {}
    for name in contents:
        final_aabb = np.asarray(replay["final_content_aabbs"][name], dtype=float)
        if final_aabb.shape != (2, 3) or not np.isfinite(final_aabb).all():
            raise ValueError("trusted content AABB is invalid")
        center = content_positions[name][-1]
        final_inside[name] = bool(
            np.all(final_aabb[0, :2] >= target_aabb[0, :2] + 0.002)
            and np.all(final_aabb[1, :2] <= target_aabb[1, :2] - 0.002)
            and final_aabb[0, 2] >= target_aabb[0, 2] - 0.005
            and final_aabb[1, 2] <= target_aabb[1, 2] + 0.02
            and replay["final_content_target_contacts"].get(name, 0) > 0)
        final_stable[name] = bool(np.max(np.linalg.norm(
            np.diff(content_positions[name][-8:], axis=0), axis=1)) < 0.01)
        free_of_source[name] = bool(np.linalg.norm(
            center[:2] - source_positions[-1, :2]) > 0.06)
        plausible_dynamics[name] = bool(np.max(np.linalg.norm(
            np.diff(content_positions[name], axis=0), axis=1)) < 0.15)
    distinct_contents = all(np.linalg.norm(
        content_positions[left][-1] - content_positions[right][-1]) > 0.01
        for index, left in enumerate(contents) for right in contents[index + 1:])
    checks = {"initial_contents_in_source": initial_contents_inside,
              "source_lifted": max_lift >= 0.10,
              "source_tilted": max_tilt >= 45.0,
              "source_over_target": source_near_target,
              "target_remained_stable": target_stable,
              "robot_contact_during_tilt": robot_contact,
              "robot_released_source": release,
              "all_contents_inside_target": all(final_inside.values()),
              "all_contents_stable": all(final_stable.values()),
              "all_contents_clear_of_source": all(free_of_source.values()),
              "all_content_motion_plausible": all(plausible_dynamics.values()),
              "contents_remain_distinct": distinct_contents,
              "no_object_attachment": replay["object_attachment_constraint_count"] == 0}
    passed = all(checks.values())
    result = {"schema_version": "worldolympiad_independent_pour_verdict.v1",
              "passed": passed, "quality_score": sum(checks.values()) / len(checks),
              "task_sha256": expected["task_sha256"],
              "replay_sha256": _sha(replay_path),
              "content_instances": contents, "checks": checks,
              "measurements": {"max_source_lift_m": max_lift,
                               "max_source_tilt_degrees": max_tilt,
                               "final_inside": final_inside,
                               "final_stable": final_stable,
                               "free_of_source": free_of_source}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(workspace=args.workspace, task_path=args.task,
                            generation_path=args.generation,
                            replay_path=args.replay, output_path=args.output),
                     sort_keys=True))


if __name__ == "__main__":
    main()
