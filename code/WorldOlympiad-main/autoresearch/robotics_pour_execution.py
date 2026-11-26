"""Registered DROID pour execution with an external physical success verdict.

The controller emits actions. Trusted replay and the protected verifier decide
whether the loose contents actually transfer; the bridge's pick/place result
is retained as raw evidence and cannot award pour success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .robotics_pour_controller import propose
from .robotics_pour_replay import replay, resolve_pour_case
from .robotics_pour_verifier import verify


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(*, workspace: Path, task: Path, generation: Path,
            asset_revision: Path, output: Path) -> Path:
    workspace, task, generation, asset_revision, output = (
        Path(value).resolve() for value in
        (workspace, task, generation, asset_revision, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("pour execution output already exists")
    case = resolve_pour_case(workspace=workspace, task_path=task,
                             generation_path=generation,
                             asset_revision=asset_revision)
    output.mkdir(parents=True, exist_ok=True)
    actions = propose(workspace=workspace, task=task, generation=generation,
                      asset_revision=asset_revision, output=output / "controller")
    replay_path = replay(workspace=workspace, task_path=task,
                         generation_path=generation, actions_path=actions,
                         asset_revision=asset_revision,
                         output_dir=output / "trusted")
    verdict_path = output / "independent_pour_verdict.json"
    verdict = verify(workspace=workspace, task_path=task,
                     generation_path=generation, replay_path=replay_path,
                     output_path=verdict_path)
    if verdict.get("passed") is not True:
        raise ValueError("independent pour verifier rejected DROID replay")
    trusted = json.loads(replay_path.read_text())
    shutil.copyfile(output / "trusted" / "trajectory.json", output / "trajectory.json")
    table = {
        "schema_version": "robotics_scene_object_table.v1",
        "objects": [
            {"object_id": case["source_id"], "sdf_path": str(case["source_sdf"]),
             "role": "support"},
            {"object_id": case["target_id"], "sdf_path": str(case["target_sdf"]),
             "role": "destination"},
            *({"object_id": name, "sdf_path": str(case["content_sdf"]),
               "role": "manipuland"} for name in case["content_instances"]),
        ],
    }
    (output / "scene_objects_manifest.json").write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n")
    result = {
        "schema_version": "robotics_droid_pour_execution.v1",
        "success": True,
        "operator": "pour",
        "robot_embodiment": "robolab_droid",
        "success_predicate": "all_independent_contents_inside_target_stable",
        "object_attachment_constraint_count": trusted["object_attachment_constraint_count"],
        "expected_gripper_coupling_constraint_count": 0,
        "gripper_coupling_constraint_count": trusted["gripper_coupling_constraint_count"],
        "task_sha256": _sha(task),
        "generation_sha256": _sha(generation),
        "asset_revision_sha256": case["asset_revision_sha256"],
        "controller_actions_sha256": _sha(actions),
        "trusted_replay_sha256": _sha(replay_path),
        "independent_verdict_sha256": _sha(verdict_path),
        "independent_verdict_path": str(verdict_path),
        "independent_verdict": verdict,
        "trajectory_sha256": _sha(output / "trajectory.json"),
        "scene_objects_manifest_sha256": _sha(output / "scene_objects_manifest.json"),
        "content_instances": case["content_instances"],
        "candidate_reported_success_ignored": True,
    }
    path = output / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--asset-revision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(execute(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
