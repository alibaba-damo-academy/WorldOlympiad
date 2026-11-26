"""Independent task-predicate checks for passive PyBullet references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .agentic_contracts import AgentTask


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _up(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([2 * (x*z + y*w), 2 * (y*z - x*w), 1 - 2 * (x*x + y*y)])


def verify_passive_reference(*, task: AgentTask, trajectory: Path,
                             contacts_path: Path, body_ids: dict[str, int],
                             aliases: dict[str, str]) -> dict:
    with np.load(trajectory, allow_pickle=False) as data:
        names = [str(value) for value in data["body_names"].tolist()]
        positions = np.asarray(data["positions"], dtype=float)
        orientations = np.asarray(data["orientations_xyzw"], dtype=float)
        velocities = np.asarray(data["linear_velocities"], dtype=float)
        times = np.asarray(data["timestamps"], dtype=float)
    if (positions.ndim != 3 or positions.shape[-1] != 3
            or orientations.shape != (*positions.shape[:2], 4)
            or velocities.shape != positions.shape
            or times.shape != (len(positions),)
            or not np.isfinite(positions).all()
            or not np.isfinite(orientations).all()
            or not np.isfinite(velocities).all()
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("invalid measured passive-physics trajectory")
    contacts = json.loads(contacts_path.read_text())
    indices = {name: index for index, name in enumerate(names)}

    def index(object_id: str) -> int:
        alias = aliases.get(object_id, object_id)
        if alias not in indices or object_id not in body_ids:
            raise ValueError("task object is missing from measured simulation")
        return indices[alias]

    def contact_rows(subject: str, target: str | None):
        first = body_ids[subject]
        second = body_ids[target] if target else None
        return [row for row in contacts
                if (first in (row["body_a"], row["body_b"])
                    and (second is None or second in (row["body_a"], row["body_b"]))) ]

    checks = []
    for predicate in task.success_predicates:
        subject = predicate.subject
        target = predicate.object
        subject_index = index(subject)
        if target:
            index(target)
        minimum = predicate.threshold
        if predicate.kind == "contact":
            observed = bool(contact_rows(subject, target))
            measurement = {"contact_count": len(contact_rows(subject, target))}
        elif predicate.kind == "moved":
            distance = float(np.linalg.norm(positions[-1, subject_index] -
                                            positions[0, subject_index]))
            observed = distance >= (minimum if minimum is not None else 0.05)
            measurement = {"displacement_m": distance}
        elif predicate.kind == "rebounded":
            rows = contact_rows(subject, target)
            first_contact = min((float(row["time_s"]) for row in rows), default=math.inf)
            before = velocities[times < first_contact, subject_index, 2]
            after = velocities[times > first_contact, subject_index, 2]
            observed = bool(len(before) and len(after)
                            and np.min(before) < -0.1 and np.max(after) > 0.1)
            measurement = {"first_contact_s": first_contact if rows else None,
                           "post_contact_max_upward_m_s": float(np.max(after)) if len(after) else None}
        elif predicate.kind == "tilted":
            start = _up(orientations[0, subject_index])
            end = _up(orientations[-1, subject_index])
            angle = math.acos(float(np.clip(np.dot(start, end), -1, 1)))
            observed = angle >= (minimum if minimum is not None else 0.35)
            measurement = {"tilt_radians": angle}
        else:
            raise ValueError(f"no independent passive predicate evaluator: {predicate.kind}")
        checks.append({"predicate": predicate.kind, "subject": subject,
                       "object": target, "passed": bool(observed), **measurement})
    return {"schema_version": "worldolympiad_passive_physics_certificate.v1",
            "task_id": task.task_id, "task_sha256": task.content_hash(),
            "trajectory_sha256": _sha(trajectory),
            "contacts_sha256": _sha(contacts_path),
            "evaluator_sha256": _sha(Path(__file__)),
            "checks": checks, "passed": all(row["passed"] for row in checks)}


def verify_recorded_simulation(*, task_path: Path, simulation_path: Path,
                               output: Path) -> dict:
    """Re-evaluate a candidate's raw trajectory in a separate trusted process."""
    task = AgentTask.model_validate_json(task_path.read_text())
    simulation = json.loads(simulation_path.read_text())
    if simulation.get("task_sha256") != task.content_hash():
        raise ValueError("simulation belongs to another task")
    reference = simulation_path.parent / "reference"
    trajectory = reference / "state_trajectory.npz"
    contacts = reference / "contacts.json"
    if _sha(trajectory) != simulation.get("trajectory_sha256"):
        raise ValueError("candidate trajectory differs from its recorded digest")
    if _sha(contacts) != simulation.get("contacts_sha256"):
        raise ValueError("candidate contacts differ from their recorded digest")
    primary = next(obj.object_id for obj in task.objects if obj.role == "manipuland")
    support = next(obj.object_id for obj in task.objects if obj.role == "support")
    aliases = {primary: "subject", support: "table"}
    aliases.update({obj.object_id: obj.object_id for obj in task.objects
                    if obj.object_id not in aliases})
    result = verify_passive_reference(
        task=task, trajectory=trajectory, contacts_path=contacts,
        body_ids=simulation["body_instance_map"], aliases=aliases,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_recorded_simulation(task_path=args.task,
                                        simulation_path=args.simulation,
                                        output=args.output)
    print(json.dumps({"passed": result["passed"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
