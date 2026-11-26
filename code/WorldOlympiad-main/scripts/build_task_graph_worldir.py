#!/usr/bin/env python3
"""Compile a recipe-free task graph and all of its reference assets to WorldIR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoresearch.cosmos_worldir_scene import resolve_worldir_scene
from autoresearch.isaac_compiler import prepare_isaac_bundle
from autoresearch.io import sha256_file, write_json
from autoresearch.models import Pose, WorldIR
from autoresearch.robotics_runtime_sdf_rebind import bind_runtime_physics_sdfs
from autoresearch.robotics_runtime_urdf_rebind import bind_runtime_physics_urdfs
from autoresearch.worldir_fixed_joint_collapse import collapse_fixed_joint_tree
from autoresearch.models import ContactMaterial
from autoresearch.materials import MATERIALS


def bind_task_physics(reference_manifest, prepared_assets, output_dir, digests, *, allow_subject_joint_lock):
    """Partition authoritative formats; never substitute source SDF for URDF physics."""
    sdf_assets, urdf_assets = {}, {}
    for key, record in prepared_assets.items():
        if record.get('physics_urdf') and record.get('physics_sdf'):
            raise ValueError(f'{key}: ambiguous authoritative physics formats')
        (urdf_assets if record.get('physics_urdf') else sdf_assets)[key] = record
    if not prepared_assets:
        raise ValueError('DROID reference needs prepared physics assets')
    runtime = {}
    if sdf_assets:
        runtime.update(bind_runtime_physics_sdfs(reference_manifest, sdf_assets,
            output_dir / 'sdf', digests, allow_subject_joint_lock=allow_subject_joint_lock))
    if urdf_assets:
        runtime.update(bind_runtime_physics_urdfs(reference_manifest, urdf_assets,
            output_dir / 'urdf', digests))
    return runtime


def initial_joint_metadata(record):
    """Propagate measured joint evidence; missing legacy snapshots stay missing."""
    import math
    if 'initial_joint_positions' not in record:
        return {}
    values = record['initial_joint_positions']
    source = record.get('initial_joint_state_source')
    if not isinstance(values, dict) or not isinstance(source, str) or not source.strip():
        raise ValueError('initial joint measurements require source provenance')
    if any(not isinstance(name, str) or not name or not math.isfinite(float(value))
           for name, value in values.items()):
        raise ValueError('invalid initial joint measurement')
    return {'initial_joint_positions': dict(values), 'initial_joint_state_source': source}


def import_task_urdf(path, **kwargs):
    """Import measured expert physics, merging only lossless fixed subtrees."""
    metadata = kwargs.pop('metadata')
    target, support = kwargs.pop('target'), kwargs.pop('support')
    body = import_urdf_body(path, **kwargs)
    body.target, body.support = target, support
    body.material = MATERIALS.get(body.semantic_category, ContactMaterial())
    body.metadata.update(metadata)
    if any(j.type == 'fixed' for j in body.joints):
        body.metadata['fixed_joint_collapse'] = collapse_fixed_joint_tree(body)
    return body
from autoresearch.scenecode_importer import import_sdf_body
from autoresearch.urdf_importer import import_urdf_body
from scripts.build_cup_to_tray_worldir import (
    _compiled_mesh_source_keys,
    _copy_content_bound,
    _portable_body,
)


def build_worldir(
    *,
    reference_manifest: Path,
    task_graph_path: Path,
    trajectory_path: Path,
    robot_urdf: Path,
    output_dir: Path,
    scene_id: str,
    prompt: str,
) -> dict[str, str]:
    reference_manifest = Path(reference_manifest).resolve()
    reference_root = reference_manifest.parent
    task_graph_path = Path(task_graph_path).resolve()
    trajectory_path = Path(trajectory_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    graph_snapshot = output / "provenance" / "task_graph.json"
    trajectory_snapshot = output / "provenance" / "reference_trajectory.json"
    graph_digest = _copy_content_bound(task_graph_path, graph_snapshot)
    trajectory_digest = _copy_content_bound(trajectory_path, trajectory_snapshot)
    graph = json.loads(task_graph_path.read_text(encoding="utf-8"))
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    bindings = resolve_worldir_scene(task_graph=graph, trajectory=trajectory)
    runtime_sdfs = bind_task_physics(
        reference_manifest,
        trajectory["prepared_assets"],
        output / "runtime_assets",
        json.loads(reference_manifest.read_text(encoding="utf-8")).get(
            "bundle_artifacts_sha256"
        ) or {},
        allow_subject_joint_lock=not bool(graph.get("articulation_components")),
    )

    bodies = []
    mesh_sources: dict[str, Path] = {}
    support_names = [binding.body_name for binding in bindings if binding.role == "support"]
    default_support = support_names[0] if support_names else None
    for binding in bindings:
        sdf = runtime_sdfs[binding.asset_key]
        if binding.pose is None:
            pose = Pose()
        else:
            pose = Pose(
                xyz=tuple(float(value) for value in binding.pose["position"]),
                quat_xyzw=tuple(float(value) for value in binding.pose["orientation_xyzw"]),
            )
        importer = (import_task_urdf if trajectory['prepared_assets'][binding.asset_key].get('physics_urdf')
                    else import_sdf_body)
        body = importer(
            sdf,
            body_name=binding.body_name,
            semantic_category=binding.category,
            pose=pose,
            static=not binding.dynamic,
            target=binding.role in {"manipuland", "tool", "destination"},
            support=(default_support if binding.dynamic and binding.body_name != default_support else None),
            metadata={
                "task_object_id": binding.task_object_id,
                "trajectory_object_name": binding.trajectory_object_name,
                "task_role": binding.role,
                "prepared_asset_key": binding.asset_key,
                "source": "SceneCode reference bundle",
                **initial_joint_metadata(trajectory['prepared_assets'][binding.asset_key]),
            },
        )
        bodies.append(_portable_body(body, output, mesh_sources))

    frame0 = trajectory["frames"][0]
    robot_source = import_urdf_body(robot_urdf, body_name="robot", pose=Pose())
    robot_pose = Pose(
        xyz=tuple(float(value) for value in frame0["robot"]["base_position"]),
        quat_xyzw=tuple(float(value) for value in frame0["robot"]["base_orientation_xyzw"]),
    )
    robot = _portable_body(
        robot_source.model_copy(update={"pose": robot_pose}), output, mesh_sources
    )
    world = WorldIR(
        scene_id=scene_id,
        prompt=prompt,
        bodies=[*bodies, robot],
        provenance={
            "task_graph": {
                "path": "provenance/task_graph.json",
                "sha256": graph_digest,
            },
            "action_trajectory": {
                "path": "provenance/reference_trajectory.json",
                "sha256": trajectory_digest,
                "frame_count": len(trajectory["frames"]),
            },
            "scene_object_bindings": [binding.__dict__ for binding in bindings],
            "native_isaac_embodiment": "RoboLab DroidCfg",
        },
    )
    world_path = world.write_json(output / "world_ir.json")
    compile_plan = prepare_isaac_bundle(
        world,
        output / "isaac",
        native_body_names=["robot"],
        preserve_visual_materials=True,
        mesh_source_overrides={
            key: mesh_sources[key] for key in _compiled_mesh_source_keys(world, set())
        },
        authority_bindings={
            "task_graph": {
                "path": "../provenance/task_graph.json",
                "sha256": graph_digest,
                "role": "typed task graph authority",
            },
            "action_trajectory": {
                "path": "../provenance/reference_trajectory.json",
                "sha256": trajectory_digest,
                "role": "SceneCode reference action trajectory authority",
            },
        },
    )
    receipt = {
        "schema_version": "cosmos_task_graph_worldir_build.v1",
        "world_ir": str(world_path),
        "world_ir_sha256": sha256_file(world_path),
        "isaac_compile_plan": str(compile_plan),
        "isaac_compile_plan_sha256": sha256_file(compile_plan),
        "body_names": [body.name for body in world.bodies],
        "task_graph_sha256": sha256_file(task_graph_path),
        "trajectory_sha256": sha256_file(trajectory_path),
    }
    write_json(output / "build_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--task-graph", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    receipt = build_worldir(
        reference_manifest=args.reference_manifest,
        task_graph_path=args.task_graph,
        trajectory_path=args.trajectory,
        robot_urdf=args.robot_urdf,
        output_dir=args.output_dir,
        scene_id=args.scene_id,
        prompt=args.prompt,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
