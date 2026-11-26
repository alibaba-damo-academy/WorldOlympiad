"""Resolve task components against SceneCode's official articulated SDF output."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path

from .io import sha256_file, write_json
from .models import Contract
from .robotics_contracts import RoboticsTaskBatchManifest, TaskInstructionSpec
from .scenecode_importer import import_sdf_body
from .runtime_paths import scenecode_python, scenecode_runtime


class ArticulationBindingCertificate(Contract):
    schema_version: str = "robotics_articulation_binding.v1"
    case_id: str
    component_id: str
    semantic_kind: str
    host_object_id: str
    host_sdf: str
    host_sdf_sha256: str
    joint_name: str
    joint_type: str
    parent_link: str
    child_link: str
    axis_xyz: tuple[float, float, float]
    lower_limit: float
    upper_limit: float
    closed_position: float
    open_position: float
    travel: float
    open_delta_xyz: tuple[float, float, float]
    contact_target_local_xyz: tuple[float, float, float] | None = None
    contact_normal_local_xyz: tuple[float, float, float] | None = None
    handle_axis_local_xyz: tuple[float, float, float] | None = None
    contact_frame_evidence: dict | None = None
    child_visual_count: int
    child_collision_count: int
    passed: bool


def _safe_text(value: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _normalized_axis(value: tuple[float, float, float]) -> tuple[float, float, float]:
    if not all(math.isfinite(item) for item in value):
        raise ValueError("articulation joint axis must be a finite three-vector")
    norm = math.sqrt(sum(item * item for item in value))
    if norm <= 1.0e-9:
        raise ValueError("articulation joint axis must be non-zero")
    return tuple(item / norm for item in value)  # type: ignore[return-value]


def _scenecode_joint_endpoints(lower: float, upper: float) -> tuple[float, float]:
    """Run SceneCode's official Drake helper in its own dependency environment."""

    workspace = Path(
        os.environ.get("WORLDOLYMPIAD_WORKSPACE", str(Path(__file__).resolve().parents[3]))
    )
    python = scenecode_python(workspace)
    runtime = scenecode_runtime(workspace)
    code = (
        "import json,sys; "
        "from scenecode.agent_utils.drake_utils import get_closed_position,get_open_position; "
        "lo,hi=map(float,sys.argv[1:3]); "
        "print(json.dumps([get_closed_position(lo,hi),get_open_position(lo,hi)]))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{runtime}:{env.get('PYTHONPATH', '')}".rstrip(":")
    result = subprocess.run(
        [str(python), "-c", code, str(lower), str(upper)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    closed, opened = json.loads(result.stdout)
    return float(closed), float(opened)


def _handle_frame_from_geometry(source: Path, moving_link: str) -> tuple[dict, dict | None]:
    """Measure a unique fixed handle descendant in the moving link's frame."""
    import numpy as np
    from scripts.render_scenecode_asset_identity import _sdf_frame_transforms, _pose_matrix, _geometry_bounds, _corners
    from .scenecode_mesh_frame_conversion import _corners as mesh_corners, Y_UP_TO_Z_UP

    model, frames = _sdf_frame_transforms(source)
    descendants = {moving_link}
    while True:
        expanded = descendants | {joint.findtext("child") for joint in model.findall("joint")
            if joint.get("type") == "fixed" and joint.findtext("parent") in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    handles = [link for link in model.findall("link") if link.get("name") in descendants
               and "handle" in _safe_text(link.get("name", "")).split()]
    if not handles:
        return {}, None
    if len(handles) != 1:
        raise ValueError("moving component has multiple handle candidates; explicit binding is required")
    inverse = np.linalg.inv(frames[moving_link])
    dependencies = {str(source): sha256_file(source)}

    def points(link):
        values = []
        for node in link.findall("collision"):
            geometry = node.find("geometry")
            uri = geometry.findtext("mesh/uri") if geometry is not None else None
            if uri:
                mesh = (source.parent / uri).resolve()
                local = mesh_corners(mesh)
                if mesh.suffix.lower() in {".gltf", ".glb"}:
                    local = local @ Y_UP_TO_Z_UP.T
                scale = np.fromstring(geometry.findtext("mesh/scale", "1 1 1"), sep=" ")
                local = local * scale
                dependencies[str(mesh)] = sha256_file(mesh)
            else:
                bounds = _geometry_bounds(geometry, source) if geometry is not None else None
                if bounds is None:
                    raise ValueError("handle binding has unsupported collision geometry")
                local = _corners(bounds)
            pose = node.find("pose")
            parent = (pose.get("relative_to") if pose is not None else None) or link.get("name")
            transform = inverse @ frames[parent] @ _pose_matrix(pose.text if pose is not None else None)
            values.append(local @ transform[:3, :3].T + transform[:3, 3])
        if not values:
            raise ValueError("contact frame requires actual handle and moving-link collision geometry")
        return np.concatenate(values)

    handle = points(handles[0])
    door = points(model.find(f'link[@name="{moving_link}"]'))
    eigenvalues, axes = np.linalg.eigh(np.cov(handle.T))
    if eigenvalues[-1] <= 1.2 * eigenvalues[-2]:
        raise ValueError("handle geometry has no unique long axis")
    axis = axes[:, -1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    center = (handle.min(axis=0) + handle.max(axis=0)) / 2
    door_eigenvalues, door_axes = np.linalg.eigh(np.cov(door.T))
    if door_eigenvalues[1] <= 1.2 * door_eigenvalues[0]:
        raise ValueError("moving component has no unique surface normal")
    normal = door_axes[:, 0]
    displacement = center - (door.min(axis=0) + door.max(axis=0)) / 2
    if abs(float(np.dot(normal, displacement))) <= 1e-6:
        raise ValueError("handle geometry does not determine an outward contact side")
    if np.dot(normal, displacement) < 0:
        normal = -normal
    normal -= np.dot(normal, axis) * axis
    if np.linalg.norm(normal) <= 1e-6:
        raise ValueError("handle axis is parallel to the contact normal")
    normal /= np.linalg.norm(normal)
    frame = {"contact_target_local_xyz": tuple(center), "contact_normal_local_xyz": tuple(normal),
             "handle_axis_local_xyz": tuple(axis)}
    evidence = {"method": "fixed_handle_collision_geometry_v1", "moving_link": moving_link,
                "handle_link": handles[0].get("name"), "input_file_sha256": dependencies,
                "coordinate_frame": moving_link}
    return frame, evidence


def resolve_articulation_binding(
    *,
    task: TaskInstructionSpec,
    component_id: str,
    host_sdf: str | Path,
    output: str | Path | None = None,
) -> ArticulationBindingCertificate:
    components = {
        item.component_id: item for item in task.articulation_components
    }
    if component_id not in components:
        raise ValueError(f"task has no articulation component {component_id!r}")
    component = components[component_id]
    source = Path(host_sdf).resolve()
    imported = import_sdf_body(
        source,
        body_name=component.host_object_id,
        semantic_category=component.semantic_kind,
        static=True,
    )
    links = {item.name: item for item in imported.links}
    candidates = []
    # Use the typed component description as a semantic tie-breaker when an
    # asset contains several joints of the same type (for example, multiple
    # drawers but only one explicitly described as handle-bearing).  This
    # remains asset- and case-name independent.
    semantic_tokens = set(
        _safe_text(f"{component.semantic_kind} {component.visual_description}").split()
    )
    for joint in imported.joints:
        if joint.type != component.joint_type:
            continue
        name_text = _safe_text(f"{joint.name} {joint.child}")
        score = len(semantic_tokens & set(name_text.split()))
        candidates.append((score, joint))
    if not candidates:
        raise ValueError(
            f"host asset has no {component.joint_type} joint for {component_id}"
        )
    best_score = max(score for score, _ in candidates)
    best = [joint for score, joint in candidates if score == best_score]
    if len(best) != 1:
        raise ValueError(
            "articulation component binding is ambiguous; generated asset needs "
            "one uniquely named matching joint"
        )
    joint = best[0]
    child = links[joint.child]
    visual_count = len(child.visuals)
    collision_count = len(child.collisions)
    if visual_count < 1 or collision_count < 1:
        raise ValueError(
            "articulation moving link needs both visual and collision geometry"
        )
    axis = _normalized_axis(joint.axis)
    lower, upper = joint.lower, joint.upper
    if (
        lower is None
        or upper is None
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower >= upper
    ):
        raise ValueError("articulation joint limits must be finite and ordered")
    closed, opened = _scenecode_joint_endpoints(lower, upper)
    travel = abs(opened - closed)
    if travel <= 1.0e-6:
        raise ValueError("articulation joint has no usable travel")

    frame = {}
    frame_evidence = None
    frame_keys = ("contact_target_local_xyz", "contact_normal_local_xyz", "handle_axis_local_xyz")
    if any(key in component.geometry_constraints for key in frame_keys):
        for key in frame_keys:
            values = component.geometry_constraints.get(key)
            if not isinstance(values, (tuple, list)) or len(values) != 3:
                raise ValueError("handle contact frame requires three complete 3-vectors")
            values = tuple(float(value) for value in values)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("handle contact frame must be finite")
            frame[key] = values if key == "contact_target_local_xyz" else _normalized_axis(values)
        if abs(sum(a*b for a,b in zip(frame["contact_normal_local_xyz"], frame["handle_axis_local_xyz"]))) > .99:
            raise ValueError("handle axis and contact normal must not be parallel")
    elif "handle" in semantic_tokens:
        frame, frame_evidence = _handle_frame_from_geometry(source, joint.child)
    delta = tuple(float(item * (opened - closed)) for item in axis)
    if joint.type == "revolute" and frame:
        # Legacy field describes contact-point displacement in the closed child
        # frame, never radians multiplied by the hinge axis.
        point = frame["contact_target_local_xyz"]
        angle = opened - closed
        cross = (axis[1]*point[2]-axis[2]*point[1], axis[2]*point[0]-axis[0]*point[2], axis[0]*point[1]-axis[1]*point[0])
        dot = sum(a*b for a,b in zip(axis,point))
        delta = tuple(point[i]*(math.cos(angle)-1) + cross[i]*math.sin(angle) + axis[i]*dot*(1-math.cos(angle)) for i in range(3))

    certificate = ArticulationBindingCertificate(
        case_id=task.case_id,
        component_id=component.component_id,
        semantic_kind=component.semantic_kind,
        host_object_id=component.host_object_id,
        host_sdf=str(source),
        host_sdf_sha256=sha256_file(source),
        joint_name=joint.name,
        joint_type=joint.type,
        parent_link=joint.parent,
        child_link=joint.child,
        axis_xyz=axis,
        lower_limit=lower,
        upper_limit=upper,
        closed_position=closed,
        open_position=opened,
        travel=travel,
        open_delta_xyz=delta,
        **frame,
        contact_frame_evidence=frame_evidence,
        child_visual_count=visual_count,
        child_collision_count=collision_count,
        passed=True,
    )
    if output is not None:
        write_json(
            output,
            certificate.model_dump(mode="json", exclude_none=True),
        )
    return certificate


def _relocated_handle_geometry_matches(
    *,
    task: TaskInstructionSpec,
    component_id: str,
    host_sdf: str | Path,
    existing: ArticulationBindingCertificate,
) -> bool:
    """Accept a relocated but byte-identical handle geometry source.

    A published certificate pins the digest of every mesh the contact frame was
    derived from.  A visual repair install can move an unchanged asset into a
    new directory, which leaves those recorded paths dangling while the admitted
    host SDF digest is untouched.  Re-derive the frame from the currently
    admitted host SDF and require identical input digests *and* an identical
    frame, so only a byte-identical relocation is accepted and any real geometry
    change still fails closed.
    """
    recorded = (existing.contact_frame_evidence or {}).get("input_file_sha256") or {}
    if not recorded:
        return False
    try:
        reproduced = resolve_articulation_binding(
            task=task, component_id=component_id, host_sdf=host_sdf
        )
    except (ValueError, KeyError, OSError):
        return False
    fresh = (reproduced.contact_frame_evidence or {}).get("input_file_sha256") or {}
    if not fresh or sorted(str(value) for value in fresh.values()) != sorted(
        str(value) for value in recorded.values()
    ):
        return False
    for key in (
        "contact_target_local_xyz",
        "contact_normal_local_xyz",
        "handle_axis_local_xyz",
    ):
        previous = getattr(existing, key)
        current = getattr(reproduced, key)
        if previous is None or current is None:
            return False
        if len(previous) != len(current) or any(
            abs(float(a) - float(b)) > 1.0e-9
            for a, b in zip(previous, current)
        ):
            return False
    return True


def certify_case_articulation(
    *,
    output_root: str | Path,
    batch_id: str,
    case_id: str,
    component_id: str,
) -> Path:
    """Bind one typed component to a certified generated SceneCode joint."""

    root = Path(output_root).resolve()
    batch_path = root / "robotics_task_batches" / batch_id / "batch_manifest.json"
    batch = RoboticsTaskBatchManifest.model_validate_json(
        batch_path.read_text(encoding="utf-8")
    )
    records = [item for item in batch.cases if item.case_id == case_id]
    if len(records) != 1:
        raise ValueError("selected batch must contain the case exactly once")
    record = records[0]
    task_path = root / record.task_path
    if sha256_file(task_path) != record.task_sha256:
        raise ValueError("selected articulation task digest changed")
    task = TaskInstructionSpec.model_validate_json(
        task_path.read_text(encoding="utf-8")
    )
    components = {
        item.component_id: item for item in task.articulation_components
    }
    if component_id not in components:
        raise ValueError("selected task has no requested articulation component")
    component = components[component_id]

    generation_root = root / "robotics_scenecode_generations" / batch_id / case_id
    generation = json.loads(
        (generation_root / "generation_manifest.json").read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (generation_root / "task_inventory_certificate.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        generation.get("task_sha256") != record.task_sha256
        or generation.get("passed") is not True
        or generation.get("semantic_identity_passed") is not True
        or inventory.get("passed") is not True
    ):
        raise ValueError("SceneCode candidate is not admitted for articulation")
    matches = {
        item["task_object_id"]: item for item in inventory.get("matches", [])
    }
    host = matches.get(component.host_object_id)
    if host is None:
        raise ValueError("SceneCode inventory has no articulated host asset")
    host_sdf = Path(host["sdf_path"])
    if not host_sdf.is_file() or sha256_file(host_sdf) != host.get("sdf_sha256"):
        raise ValueError("SceneCode articulated host SDF digest changed")

    output = (
        root
        / "robotics_articulation_bindings"
        / batch_id
        / case_id
        / f"{component_id}.json"
    )
    if output.exists():
        existing = ArticulationBindingCertificate.model_validate_json(
            output.read_text(encoding="utf-8")
        )
        if existing.case_id != case_id or existing.component_id != component_id:
            raise ValueError("published articulation binding belongs to another case")
        if (
            existing.host_sdf_sha256 != host["sdf_sha256"]
            or existing.passed is not True
        ):
            # The admitted host asset moved to another revision.  Re-certify the
            # binding against the asset the inventory binds now instead of
            # stopping, and keep the superseded certificate: the measured
            # geometry is re-derived from the current SDF, never carried over.
            if (root / "robotics_reference_bundles" / record.reference_run_id).exists():
                raise ValueError(
                    "cannot re-certify an articulation binding consumed by a reference"
                )
            archive = output.parent / "superseded"
            archive.mkdir(exist_ok=True)
            (archive / f"{output.stem}.{sha256_file(output)}.json").write_bytes(
                output.read_bytes()
            )
            refreshed = resolve_articulation_binding(
                task=task, component_id=component_id, host_sdf=host_sdf
            )
            if refreshed.contact_target_local_xyz is None:
                raise ValueError(
                    "the current articulated host has no resolvable handle contact geometry"
                )
            temporary = output.with_suffix(".tmp")
            write_json(temporary, refreshed.model_dump(mode="json", exclude_none=True))
            temporary.replace(output)
            return output
        if existing.contact_target_local_xyz is not None or "handle" not in _safe_text(component.visual_description).split():
            evidence = existing.contact_frame_evidence or {}
            dependencies = evidence.get("input_file_sha256") or {}
            for name, expected in dependencies.items():
                path = Path(name)
                if path.is_file() and sha256_file(path) != expected:
                    raise ValueError("certified handle contact geometry changed")
            missing = sorted(name for name in dependencies if not Path(name).is_file())
            if missing and not _relocated_handle_geometry_matches(
                task=task,
                component_id=component_id,
                host_sdf=host_sdf,
                existing=existing,
            ):
                raise ValueError(
                    "certified handle contact geometry source is unavailable and the "
                    "admitted host SDF does not reproduce it byte-for-byte: "
                    + ", ".join(missing)
                )
            return output
        if (root / "robotics_reference_bundles" / record.reference_run_id).exists():
            raise ValueError("cannot upgrade a contact binding consumed by a reference")
        upgraded = resolve_articulation_binding(task=task, component_id=component_id, host_sdf=host_sdf)
        if upgraded.contact_target_local_xyz is None:
            raise ValueError("generated asset has no resolvable handle contact geometry")
        archive = output.parent / "superseded"
        archive.mkdir(exist_ok=True)
        (archive / f"{output.stem}.{sha256_file(output)}.json").write_bytes(output.read_bytes())
        temporary = output.with_suffix(".tmp")
        write_json(temporary, upgraded.model_dump(mode="json", exclude_none=True))
        temporary.replace(output)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    resolve_articulation_binding(
        task=task,
        component_id=component_id,
        host_sdf=host_sdf,
        output=output,
    )
    return output


def refresh_case_articulation_inventory(*, output_root, batch_id, case_id) -> dict:
    """Rebind jointless lookalikes to an existing qualified scene host.

    Only declared articulation hosts may change. The old image certificate is
    archived and fresh identity review is mandatory before physical admission.
    """
    import hashlib
    from .robotics_resource_lock import exclusive_case_asset_edit
    from .robotics_scenecode_case_generator import _load_case, certify_task_inventory, write_final_inventory_binding

    root = Path(output_root).resolve()
    with exclusive_case_asset_edit(root, batch_id, case_id):
        task, task_path, _ = _load_case(output_root=root, batch_id=batch_id, case_id=case_id)
        case = root / "robotics_scenecode_generations" / batch_id / case_id
        batch = RoboticsTaskBatchManifest.model_validate_json(
            (root / "robotics_task_batches" / batch_id / "batch_manifest.json").read_text())
        record = next(row for row in batch.cases if row.case_id == case_id)
        if (root / "robotics_reference_bundles" / record.reference_run_id).exists():
            raise ValueError("cannot rebind an asset consumed by a reference")
        generation_path, inventory_path = case / "generation_manifest.json", case / "task_inventory_certificate.json"
        before = {p: p.read_bytes() for p in (generation_path, inventory_path)}
        generation, inventory = [json.loads(before[p]) for p in (generation_path, inventory_path)]
        house = Path(generation["house_state"])
        if (generation["task_sha256"] != sha256_file(task_path)
                or generation["task_inventory_certificate_sha256"] != sha256_file(inventory_path)
                or generation["house_state_sha256"] != sha256_file(house)
                or inventory["house_state_sha256"] != sha256_file(house)):
            raise ValueError("current articulation source binding changed")
        updated = certify_task_inventory(task=task, house_state=house)
        if not updated["passed"]:
            raise ValueError("scene has no complete task inventory")
        hosts = {component.host_object_id for component in task.articulation_components}
        old = {item["task_object_id"]: item for item in inventory["matches"]}
        changed = []
        for item in updated["matches"]:
            previous = old[item["task_object_id"]]
            if item["task_object_id"] in hosts and not item.get("articulation_joint_qualified"):
                raise ValueError("scene has no qualified joint for the declared host")
            if any(item.get(key) != previous.get(key) for key in ("scene_object_id", "sdf_path", "sdf_sha256")):
                if item["task_object_id"] not in hosts:
                    raise ValueError("articulation repair would change an unrelated task asset")
                changed.append({"object_id": item["task_object_id"], "old": previous, "new": item})
        if not changed:
            return {"changed": False}
        identity = case / "asset_identity"
        repair = case / "inventory_rebindings" / hashlib.sha256(json.dumps(changed, sort_keys=True).encode()).hexdigest()
        repair.mkdir(parents=True)
        try:
            if identity.exists():
                identity.rename(repair / "previous_identity")
            write_final_inventory_binding(inventory=updated, inventory_path=inventory_path, house_state=house)
            generation["task_inventory_certificate_sha256"] = sha256_file(inventory_path)
            generation["semantic_identity_passed"] = False
            generation.pop("asset_identity", None)
            generation_path.write_text(json.dumps(generation, indent=2) + "\n")
            receipt = {"changed": True, "asset_admitted": False, "changes": changed,
                       "required_next_step": "fresh_identity_render_and_model_review"}
            (repair / "rebind.json").write_text(json.dumps(receipt, indent=2) + "\n")
            return receipt
        except BaseException:
            for path, content in before.items():
                path.write_bytes(content)
            if (repair / "previous_identity").exists():
                (repair / "previous_identity").rename(identity)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--component-id", required=True)
    parser.add_argument("--host-sdf")
    parser.add_argument("--output")
    parser.add_argument("--output-root")
    parser.add_argument("--batch-id")
    parser.add_argument("--case-id")
    args = parser.parse_args()
    if args.output_root or args.batch_id or args.case_id:
        if not all((args.output_root, args.batch_id, args.case_id)):
            parser.error("case certification needs output-root, batch-id, and case-id")
        print(
            certify_case_articulation(
                output_root=args.output_root,
                batch_id=args.batch_id,
                case_id=args.case_id,
                component_id=args.component_id,
            )
        )
        return
    if not all((args.task, args.host_sdf, args.output)):
        parser.error("direct certification needs task, host-sdf, and output")
    task = TaskInstructionSpec.model_validate_json(
        Path(args.task).read_text(encoding="utf-8")
    )
    certificate = resolve_articulation_binding(
        task=task,
        component_id=args.component_id,
        host_sdf=args.host_sdf,
        output=args.output,
    )
    print(json.dumps(certificate.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
