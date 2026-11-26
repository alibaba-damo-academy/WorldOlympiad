"""Resolve a recipe-free robotics reference into native WorldIR scene inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SceneObjectBinding:
    body_name: str
    trajectory_object_name: str
    task_object_id: str
    asset_key: str
    role: str
    category: str
    pose: dict[str, list[float]] | None
    dynamic: bool


def _role(record: dict[str, Any]) -> str:
    role = record.get("role")
    if not isinstance(role, str) or not role:
        raise ValueError("task scene object has no role")
    return role


def resolve_worldir_scene(
    *,
    task_graph: dict[str, Any],
    trajectory: dict[str, Any],
) -> list[SceneObjectBinding]:
    """Bind task-graph entities to initial states and bundle physics assets.

    Trajectory entity names are authoritative for simulator bodies. Task-graph
    ids provide semantic identity. Legacy ``cup``/``tray`` trajectory aliases
    are resolved by their declared roles rather than by task class or case id.
    """

    prepared_assets = trajectory.get("prepared_assets")
    frames = trajectory.get("frames")
    if not isinstance(prepared_assets, dict) or not isinstance(frames, list) or not frames:
        raise ValueError("reference trajectory lacks prepared assets or frames")
    frame_objects = frames[0].get("objects")
    scene = task_graph.get("scene_generation", {}).get("objects")
    if not isinstance(frame_objects, dict) or not isinstance(scene, list):
        raise ValueError("reference lacks frame-zero object states or typed scene objects")

    graph_by_id: dict[str, dict[str, Any]] = {}
    for record in scene:
        if not isinstance(record, dict):
            raise ValueError("task scene object must be an object")
        object_id = record.get("object_id")
        if not isinstance(object_id, str) or not object_id or object_id in graph_by_id:
            raise ValueError("task scene object ids must be unique non-empty strings")
        graph_by_id[object_id] = record

    state_to_graph: dict[str, str] = {}
    for state_name in frame_objects:
        if state_name in graph_by_id:
            state_to_graph[state_name] = state_name

    unresolved_graph = set(graph_by_id) - set(state_to_graph.values())
    unresolved_states = set(frame_objects) - set(state_to_graph)
    controller = task_graph.get("controller_program") or []
    grouped_ids = {
        str(value)
        for operation in controller
        if isinstance(operation, dict)
        for key, value in (operation.get("constraints") or {}).items()
        if key.endswith("_object") and isinstance(value, str) and value in graph_by_id
    }
    for state_name in sorted(unresolved_states):
        state = frame_objects[state_name]
        if not isinstance(state, dict):
            raise ValueError(f"frame-zero state for {state_name!r} is invalid")
        state_role = _role(state)
        candidates = [
            object_id for object_id in unresolved_graph
            if _role(graph_by_id[object_id]) == state_role
        ]
        candidates.extend(
            object_id for object_id in grouped_ids
            if object_id not in candidates
            and _role(graph_by_id[object_id]) == state_role
        )
        if len(candidates) == 1:
            state_to_graph[state_name] = candidates[0]
            unresolved_graph.discard(candidates[0])

    result: list[SceneObjectBinding] = []
    primary_id = None
    destination_id = None
    if controller:
        first_subjects = controller[0].get("subjects") or []
        if first_subjects:
            primary_id = str(first_subjects[0])
        for record in reversed(controller):
            if record.get("target"):
                destination_id = str(record["target"])
                break
    alias_by_task_id = {}
    if primary_id:
        alias_by_task_id[primary_id] = "cup"
    if destination_id:
        alias_by_task_id[destination_id] = "tray"
    used_body_names: set[str] = set()
    for state_name, state in frame_objects.items():
        if state_name not in state_to_graph:
            raise ValueError(f"trajectory object {state_name!r} has no task-graph identity")
        graph_record = graph_by_id[state_to_graph[state_name]]
        role = _role(state)
        asset_key = state_name if state_name in prepared_assets else None
        if asset_key is None:
            asset_key = {"manipuland": "cup", "destination": "tray"}.get(role)
        if asset_key is None or asset_key not in prepared_assets:
            raise ValueError(f"no bundle physics asset is bound to trajectory object {state_name!r}")
        pose = {
            "position": list(state["link_position"]),
            "orientation_xyzw": list(state["link_orientation_xyzw"]),
        }
        task_object_id = state_to_graph[state_name]
        body_name = state_name
        if state_name not in {"cup", "tray"}:
            body_name = alias_by_task_id.get(task_object_id, state_name)
        if body_name in used_body_names:
            body_name = state_name
        used_body_names.add(body_name)
        result.append(SceneObjectBinding(
            body_name=body_name,
            trajectory_object_name=state_name,
            task_object_id=task_object_id,
            asset_key=asset_key,
            role=role,
            category=str(graph_record.get("category") or "object"),
            pose=pose,
            dynamic=role in {"manipuland", "tool"},
        ))

    for object_id in sorted(unresolved_graph):
        record = graph_by_id[object_id]
        role = _role(record)
        if role not in {"support", "fixture"}:
            raise ValueError(f"task object {object_id!r} has no trajectory state")
        asset_key = object_id if object_id in prepared_assets else None
        if asset_key is None:
            asset_key = "table" if "table" in prepared_assets else None
        if asset_key is None:
            raise ValueError(f"no static bundle physics asset is bound to {object_id!r}")
        result.append(SceneObjectBinding(
            body_name=object_id,
            trajectory_object_name=object_id,
            task_object_id=object_id,
            asset_key=asset_key,
            role=role,
            category=str(record.get("category") or "object"),
            pose=None,
            dynamic=False,
        ))

    if len({item.body_name for item in result}) != len(result):
        raise ValueError("resolved WorldIR body names are not unique")
    return result
