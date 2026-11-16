"""Task-data-driven passive PyBullet reference execution.

SceneCode owns the assets. This executor binds declared objects, applies a
small set of physical interventions, and delegates success to an independent
predicate verifier. It never routes by task name or task family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .agentic_contracts import AgentTask
from .agentic_physics_verifier import verify_passive_reference
from .physical_reference import camera_at, digest, write


def bind_agent_scene(task: AgentTask, house_state: Path, output: Path) -> tuple[Path, Path]:
    """Reuse SceneCode's one-to-one SDF inventory for a passive task."""
    from .robotics_scenecode_case_generator import certify_task_inventory
    from .scenecode_importer import import_house_state

    # The existing inventory checker only needs objects, articulation and a
    # case identifier. Keep its identity and geometry admission in one place.
    class InventoryTask:
        objects = task.objects
        articulation_components = task.articulation_components
        case_id = task.task_id
        success_predicates = task.success_predicates

    inventory = certify_task_inventory(task=InventoryTask, house_state=house_state)
    if inventory["passed"] is not True:
        raise ValueError("SceneCode task inventory did not certify every object")
    world = import_house_state(house_state, scene_id=task.task_id)
    bindings = {}
    for match in inventory["matches"]:
        source = Path(match["sdf_path"]).resolve()
        identity = match["scene_object_id"]
        candidates = [body for body in world.bodies
                      if body.source_path and Path(body.source_path).resolve() == source
                      and (body.name == identity or body.name.endswith("_" + identity))]
        if len(candidates) != 1:
            raise ValueError("certified SceneCode object does not bind uniquely to WorldIR")
        bindings[match["task_object_id"]] = {
            "body": candidates[0].name, "grounding": next(obj.visual_description for obj in task.objects
                                               if obj.object_id == match["task_object_id"]),
            "sdf_sha256": match["sdf_sha256"],
        }
    output.mkdir(parents=True, exist_ok=True)
    world_path = output / "generated_world_ir.json"
    if world_path.exists():
        from .models import WorldIR
        existing = WorldIR.model_validate_json(world_path.read_text())
        if existing != world:
            raise ValueError("published SceneCode WorldIR differs from selected scene")
    else:
        world.write_json(world_path)
    binding_payload = {
        "schema_version": "worldolympiad_task_object_bindings.v1",
        "task_sha256": task.content_hash(),
        "house_state_sha256": digest(house_state),
        "world_ir_sha256": digest(world_path), "bindings": bindings,
    }
    binding_path = output / "generated_bindings.json"
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding_payload:
            raise ValueError("published task binding differs from selected scene")
    else:
        write(binding_path, binding_payload)
    inventory_path = output / "task_inventory_certificate.json"
    if inventory_path.exists():
        if json.loads(inventory_path.read_text()) != inventory:
            raise ValueError("published task inventory differs from selected scene")
    else:
        write(inventory_path, inventory)
    return world_path, binding_path


def _vector(value, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite 3-vector")
    return result


def simulate_agent_task(*, task: AgentTask, workspace: Path, output: Path,
                        source_world: Path, binding_manifest: Path) -> Path:
    """Execute one declared passive task with measured contact/state evidence."""
    if task.embodiment is not None:
        raise ValueError("passive executor cannot run a robot embodiment")
    if task.reference_fps != 15 or task.reference_duration_s != 4:
        raise ValueError("current native physics renderer needs a 4 s / 15 fps reference")
    import pybullet as p
    from .compiler import compile_world
    from .models import WorldIR
    from .runner import _base_link_pose, _render

    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    source_world, binding_manifest = source_world.resolve(), binding_manifest.resolve()
    binding = json.loads(binding_manifest.read_text())
    if (binding.get("task_sha256") != task.content_hash()
            or binding.get("world_ir_sha256") != digest(source_world)):
        raise ValueError("SceneCode binding differs from the selected task or world")
    completed = output / "tasks" / task.task_id / "simulation.json"
    if completed.is_file():
        data = json.loads(completed.read_text())
        certificate_path = Path(data.get("physical_certificate", ""))
        if (data.get("task_sha256") == task.content_hash()
                and data.get("stage") == "reference_simulated"
                and certificate_path.is_file()
                and digest(certificate_path) == data.get("physical_certificate_sha256")
                and json.loads(certificate_path.read_text()).get("passed") is True):
            return completed
        raise ValueError("existing physical reference has stale or failed evidence")
    selected = binding["bindings"]
    if set(selected) != {obj.object_id for obj in task.objects}:
        raise ValueError("SceneCode binding does not cover the task exactly")
    subject = next((obj for obj in task.objects if obj.role == "manipuland"), None)
    support = next((obj for obj in task.objects if obj.role == "support"), None)
    if subject is None or support is None:
        raise ValueError("passive reference needs a manipuland and support")
    aliases = {subject.object_id: "subject", support.object_id: "table"}
    aliases.update({obj.object_id: obj.object_id for obj in task.objects
                    if obj.object_id not in aliases})
    original = WorldIR.model_validate_json(source_world.read_text())
    by_name = {body.name: body for body in original.bodies}
    bodies = []
    for obj in task.objects:
        source_name = selected[obj.object_id]["body"]
        if source_name not in by_name:
            raise ValueError("selected task body is absent from WorldIR")
        body = by_name[source_name].model_copy(deep=True)
        body.name = aliases[obj.object_id]
        body.static = bool(obj.geometry_constraints.get("static", obj.role == "support"))
        body.target = obj.object_id == subject.object_id
        body.support = None if obj.role == "support" else "table"
        body.metadata.pop("precompiled_urdf_source", None)
        body.metadata["visual_mesh_coordinate_frame"] = "worldir_z_up"
        bodies.append(body)
    selected_names = {record["body"] for record in selected.values()}
    for body in original.bodies:
        if body.name in selected_names or not body.static:
            continue
        background = body.model_copy(deep=True)
        background.name = "background_" + background.name
        background.target = False
        background.support = None
        background.metadata["visual_mesh_coordinate_frame"] = "worldir_z_up"
        bodies.append(background)
    root = output / "tasks" / task.task_id
    ref = root / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    scene_source = write(output / "scene_source.json", {
        "source_world": str(source_world), "source_sha256": digest(source_world),
        "bindings": str(binding_manifest), "bindings_sha256": digest(binding_manifest),
        "task_sha256": task.content_hash(),
    })
    world = WorldIR(scene_id=task.task_id, prompt=task.instruction, bodies=bodies,
                    provenance={"source_world": str(source_world),
                                "source_sha256": digest(source_world)})
    world_path = root / "world_ir.json"
    world.write_json(world_path)
    bundle_path = compile_world(world, root / "pybullet_bundle")
    bundle = json.loads(bundle_path.read_text())
    p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.setPhysicsEngineParameter(numSolverIterations=100,
                                    deterministicOverlappingPairs=1)
        ids = {}
        for name, record in bundle["bodies"].items():
            ids[name] = p.loadURDF(record["urdf_path"],
                                   basePosition=record["pose"]["xyz"],
                                   baseOrientation=record["pose"]["quat_xyzw"],
                                   useFixedBase=record["static"],
                                   flags=p.URDF_USE_INERTIA_FROM_FILE)
        low, high = np.asarray(p.getAABB(ids["table"]))
        center = (low + high) / 2
        center[2] = high[2]
        elevated = False
        for obj in task.objects:
            name = aliases[obj.object_id]
            restitution = float(obj.geometry_constraints.get("restitution", .05))
            if not 0 <= restitution <= 1:
                raise ValueError("restitution must be in [0, 1]")
            for link in range(-1, p.getNumJoints(ids[name])):
                p.changeDynamics(ids[name], link, lateralFriction=.3,
                                 linearDamping=0, angularDamping=0,
                                 restitution=restitution)
            if obj.role == "support":
                continue
            offset = _vector(obj.geometry_constraints.get("initial_offset_m"),
                             f"{obj.object_id}.initial_offset_m")
            if offset[2] < 0:
                raise ValueError("initial object is below the support")
            elevated = elevated or offset[2] > .15
            lower, upper = np.asarray(p.getAABB(ids[name]))
            position, orientation = p.getBasePositionAndOrientation(ids[name])
            target = center + offset
            target[2] += (upper[2] - lower[2]) / 2 + .001
            p.resetBasePositionAndOrientation(ids[name],
                np.asarray(position) + target - (lower + upper) / 2, orientation)
        for step in task.operation_program:
            if step.operation == "release":
                continue
            if len(step.subjects) != 1:
                raise ValueError("passive operation needs one subject")
            body = ids[aliases[step.subjects[0]]]
            if step.operation == "set_velocity":
                velocity = _vector(step.constraints.get("linear_velocity_m_s"),
                                   "linear_velocity_m_s")
                p.resetBaseVelocity(body, linearVelocity=velocity.tolist())
            elif step.operation != "apply_force":
                raise ValueError(f"missing passive operation capability: {step.operation}")
        camera = camera_at(center, elevated=elevated)
        write(ref / "calibration.json", camera)
        write(ref / "camera_calibration_info.json", {"head_camera": {
            "intrinsics": camera["intrinsics"],
            "extrinsics": camera["camera_to_world_opencv"]}})
        names = list(ids)
        timestamps, positions, rotations, speeds, depths, masks, segmentations, contacts = [], [], [], [], [], [], [], []
        link_records = []
        for name, body in ids.items():
            link_records.append((name + "::" + p.getBodyInfo(body)[0].decode(), body, -1))
            for joint in range(p.getNumJoints(body)):
                link_records.append((name + "::" + p.getJointInfo(body, joint)[12].decode(),
                                     body, joint))
        link_positions, link_orientations = [], []
        for tick in range(961):
            if tick % 16 == 0:
                timestamps.append(tick / 240)
                poses = [_base_link_pose(p, ids[name]) for name in names]
                positions.append([pose[0] for pose in poses])
                rotations.append([pose[1] for pose in poses])
                speeds.append([p.getBaseVelocity(ids[name])[0] for name in names])
                links = [_base_link_pose(p, body) if joint == -1 else
                         p.getLinkState(body, joint, computeForwardKinematics=True)[4:6]
                         for _, body, joint in link_records]
                link_positions.append([pose[0] for pose in links])
                link_orientations.append([pose[1] for pose in links])
                _, depth, segmentation = _render(p, camera, p.ER_TINY_RENDERER)
                depths.append(depth)
                masks.append(segmentation == ids["subject"])
                segmentations.append(segmentation)
            if tick == 960:
                break
            for step in task.operation_program:
                if step.operation != "apply_force":
                    continue
                constraints = step.constraints
                start = float(constraints.get("start_s", 0))
                end = float(constraints.get("end_s", task.reference_duration_s))
                if not 0 <= start < end <= task.reference_duration_s:
                    raise ValueError("force interval is outside reference duration")
                if not start <= tick / 240 < end:
                    continue
                body = ids[aliases[step.subjects[0]]]
                force = _vector(constraints.get("force_n"), "force_n")
                lower, upper = np.asarray(p.getAABB(body))
                at = (lower + upper) / 2
                height_fraction = float(constraints.get("point_height_fraction", .5))
                if not 0 <= height_fraction <= 1:
                    raise ValueError("force point_height_fraction is outside [0,1]")
                at[2] = lower[2] + height_fraction * (upper[2] - lower[2])
                p.applyExternalForce(body, -1, force.tolist(), at.tolist(), p.WORLD_FRAME)
            p.stepSimulation()
            task_ids = set(ids[aliases[obj.object_id]] for obj in task.objects)
            for contact in p.getContactPoints():
                if contact[1] in task_ids and contact[2] in task_ids:
                    contacts.append({"time_s": (tick + 1) / 240,
                                     "body_a": contact[1], "body_b": contact[2],
                                     "distance_m": contact[8],
                                     "normal_force_n": contact[9]})
        trajectory = ref / "state_trajectory.npz"
        np.savez_compressed(trajectory, timestamps=np.asarray(timestamps),
                            body_names=np.asarray(names), positions=np.asarray(positions),
                            orientations_xyzw=np.asarray(rotations),
                            linear_velocities=np.asarray(speeds),
                            link_names=np.asarray([item[0] for item in link_records]),
                            link_positions=np.asarray(link_positions),
                            link_orientations_xyzw=np.asarray(link_orientations))
        np.save(ref / "depth_frames.npy", np.asarray(depths, dtype=np.float32))
        segmentation_path = ref / "segmentation_frames.npy"
        np.save(segmentation_path, np.asarray(segmentations, dtype=np.int32))
        np.save(ref / "initial_subject_mask.npy", masks[0])
        contacts_path = write(ref / "contacts.json", contacts)
        y, x = np.nonzero(masks[0])
        if not len(x):
            raise ValueError("primary task object is invisible at frame zero")
        write(ref / "scene_data.json", {
            "instruction": task.instruction,
            "object_name": subject.visual_description,
            "canonical_object_name": "subject",
            "initial_component_region": {
                "schema_version": "initial_component_box.v1", "frame": 0,
                "image_size_wh": [camera["width"], camera["height"]],
                "component_id": "subject",
                "bbox_xyxy": [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)],
                "source_kind": "pybullet_frame_zero_projection",
            },
        })
        body_ids = {obj.object_id: ids[aliases[obj.object_id]] for obj in task.objects}
        certificate = verify_passive_reference(task=task, trajectory=trajectory,
            contacts_path=contacts_path, body_ids=body_ids, aliases=aliases)
        certificate_path = write(root / "physical_certificate.json", certificate)
        simulation = write(root / "simulation.json", {
            "schema_version": "worldolympiad_passive_reference.v1",
            "task_id": task.task_id, "task_sha256": task.content_hash(),
            "physics_backend": "pybullet", "physics_dt_s": 1 / 240,
            "fps": task.reference_fps, "frames": len(timestamps),
            "duration_s": task.reference_duration_s,
            "source_manifest_sha256": digest(scene_source),
            "bundle_path": str(bundle_path), "target_body_id": ids["subject"],
            "trajectory_sha256": digest(trajectory), "instruction": task.instruction,
            "contacts_sha256": digest(contacts_path),
            "segmentation_sha256": digest(segmentation_path),
            "body_instance_map": body_ids,
            "physical_certificate": str(certificate_path),
            "physical_certificate_sha256": digest(certificate_path),
            "stage": "reference_simulated" if certificate["passed"] else "reference_failed",
        })
        if not certificate["passed"]:
            raise ValueError("independent passive task predicate failed")
        return simulation
    finally:
        p.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("scene", "simulate", "render"))
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--house-state", type=Path)
    parser.add_argument("--source-world", type=Path)
    parser.add_argument("--bindings", type=Path)
    parser.add_argument("--flow", action="store_true")
    parser.add_argument("--model-profile")
    parser.add_argument("--prompt-profile", choices=("frozen_task", "motion_events_v2",
                                                    "event_minimal_v3"),
                        default="frozen_task")
    parser.add_argument("--prompt-profile-file", type=Path)
    args = parser.parse_args()
    task = AgentTask.model_validate_json(args.task.read_text())
    if args.stage == "scene":
        from .physical_reference import generate
        house = args.house_state or generate(args.workspace, args.output,
            scene_prompt=task.scene_brief,
            task_record={"id": task.task_id, "instruction": task.instruction})
        world, binding = bind_agent_scene(task, house, args.output)
        print(json.dumps({"world_ir": str(world), "bindings": str(binding)}))
    elif args.stage == "simulate":
        if args.source_world is None or args.bindings is None:
            parser.error("simulate requires --source-world and --bindings")
        result = simulate_agent_task(task=task, workspace=args.workspace,
            output=args.output, source_world=args.source_world,
            binding_manifest=args.bindings)
        print(json.dumps({"simulation": str(result)}))
    else:
        from .physical_reference import flow_tasks, render_tasks
        render_tasks(args.workspace, args.output, [task.task_id])
        if args.flow:
            flow_tasks(args.workspace, args.output, [task.task_id])
        if args.model_profile == "lingbot_world2_14b_causal_fast":
            from .lingbot_world2_adapter import prepare_world2_case, validate_input_contract
            from .agentic_physics_prompt import select_prompt
            prompt, receipt = select_prompt(task, args.model_profile, args.prompt_profile)
            if args.prompt_profile_file is not None:
                selected_profile = json.loads(args.prompt_profile_file.read_text())
                if selected_profile != {
                    "schema_version": "worldolympiad_physics_prompt_profile.v1",
                    "profile": args.prompt_profile,
                    "task_file_sha256": digest(args.task),
                    "model_profile": args.model_profile,
                }:
                    raise ValueError("Physics→WM prompt profile differs from selected task")
            prepared = args.output / "model_inputs" / args.model_profile / task.task_id
            contract = prepared / "input_contract.json"
            receipt_path = prepared / "prompt_receipt.json"
            if contract.is_file():
                existing = validate_input_contract(contract)
                if (not receipt_path.is_file()
                        or json.loads(receipt_path.read_text()) != receipt
                        or existing["conditioning"]["prompt"] != prompt):
                    raise ValueError("existing World2 input bundle has another prompt profile")
            else:
                prepare_world2_case(args.output / "tasks" / task.task_id,
                                    prepared, prompt)
                receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"render": str(args.output / "tasks" / task.task_id /
                                         "render.json")}))


if __name__ == "__main__":
    main()
