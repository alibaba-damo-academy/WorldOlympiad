"""Typed task specification shared by scene generation, the expert, model adapters
and evaluation.

A task is a set of objects with roles plus an ordered list of steps.  Every step
names one motion primitive and the goal predicate that must hold when the step
ends.  Nothing downstream re-reads the free-text instruction to infer intent:
orientation requirements, release conditions, etc. are explicit constraints.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROLES = {"manipuland", "destination", "support", "tool", "articulated_host", "distractor"}

# primitive -> predicate kinds it can be asked to achieve
PRIMITIVES: dict[str, set[str]] = {
    "transport": {"inside", "on", "stacked", "on_region", "inserted", "hung", "covering"},
    "reorient": {"upright", "inverted", "pose_delta", "on", "on_region"},
    "push": {"on_region", "pose_delta", "contact"},
    "articulate": {"joint_open", "joint_closed", "pressed"},
    "tool_contact": {"contact", "pose_delta", "on_region"},
}

PREDICATES = set().union(*PRIMITIVES.values())


@dataclass(frozen=True)
class Predicate:
    kind: str
    subject: str
    target: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in PREDICATES:
            raise ValueError(f"unknown predicate kind {self.kind!r}")


@dataclass(frozen=True)
class Step:
    primitive: str
    subject: str
    goal: Predicate
    target: str | None = None
    tool: str | None = None
    # explicit constraints only: keep_upright, release ("stable"), approach hints
    constraints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"unknown primitive {self.primitive!r}")
        if self.goal.kind not in PRIMITIVES[self.primitive]:
            raise ValueError(
                f"primitive {self.primitive!r} cannot achieve predicate {self.goal.kind!r}"
            )


@dataclass(frozen=True)
class TaskObject:
    object_id: str
    category: str
    role: str
    description: str = ""
    initial_container: str | None = None
    inventory_object_id: str | None = None
    scene_member_id: str | None = None
    # Physical authoring facts, passed to SceneCode independently of prose.
    geometry_constraints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if bool(self.scene_member_id) != bool(self.inventory_object_id):
            raise ValueError("scene_member_id and inventory_object_id must be declared together")
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r} for {self.object_id}")
        if 'fixed' in self.geometry_constraints and type(self.geometry_constraints['fixed']) is not bool:
            raise ValueError(f"geometry_constraints.fixed must be bool for {self.object_id}")


@dataclass(frozen=True)
class Articulation:
    component_id: str
    host_object_id: str
    joint_type: str  # revolute | prismatic
    description: str = ""


@dataclass(frozen=True)
class TaskSpec:
    case_id: str
    family: str
    instruction: str
    objects: tuple[TaskObject, ...]
    steps: tuple[Step, ...]
    articulations: tuple[Articulation, ...] = ()
    robot: str = "franka_robotiq_2f85"
    fps: float = 15.0
    max_duration_s: float = 40.0
    initial_state: str = ""
    seed: int = 0
    # Optional source-bound visual-only derivatives. Entries are explicit so a
    # material change cannot silently replace an unrelated scene asset.
    visual_variants: tuple[dict[str, Any], ...] = ()

    schema_version = "wo_task_spec.v1"

    def object(self, object_id: str) -> TaskObject:
        for item in self.objects:
            if item.object_id == object_id:
                return item
        raise KeyError(object_id)

    @property
    def support(self) -> TaskObject | None:
        return next((o for o in self.objects if o.role == "support"), None)

    def validate(self) -> None:
        ids = [o.object_id for o in self.objects] + [a.component_id for a in self.articulations]
        if len(ids) != len(set(ids)):
            raise ValueError("object/component ids must be unique")
        known = set(ids)
        containers = {o.object_id: o.initial_container for o in self.objects
                      if o.initial_container is not None}
        object_ids = {o.object_id for o in self.objects}
        for oid, container in containers.items():
            if container not in object_ids or container == oid:
                raise ValueError(f"invalid initial container for {oid!r}")
            seen = {oid}
            while container in containers:
                if container in seen:
                    raise ValueError("initial container relations form a cycle")
                seen.add(container)
                container = containers[container]
        for step in self.steps:
            for ref in (step.subject, step.target, step.tool, step.goal.subject, step.goal.target):
                if ref is not None and ref not in known:
                    raise ValueError(f"step references unknown object {ref!r}")
        if not self.steps:
            raise ValueError("task has no steps")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = self.schema_version
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskSpec":
        if payload.get("schema_version") != cls.schema_version:
            raise ValueError(f"not a {cls.schema_version} payload")
        spec = cls(
            case_id=payload["case_id"],
            family=payload["family"],
            instruction=payload["instruction"],
            objects=tuple(TaskObject(**o) for o in payload["objects"]),
            steps=tuple(
                Step(**{**s, "goal": Predicate(**s["goal"])}) for s in payload["steps"]
            ),
            articulations=tuple(Articulation(**a) for a in payload.get("articulations", ())),
            robot=payload.get("robot", "franka_robotiq_2f85"),
            fps=float(payload.get("fps", 15.0)),
            max_duration_s=float(payload.get("max_duration_s", 40.0)),
            initial_state=payload.get("initial_state", ""),
            seed=int(payload.get("seed", 0)),
            visual_variants=tuple(dict(v) for v in payload.get("visual_variants", ())),
        )
        spec.validate()
        return spec

    @classmethod
    def load(cls, path: str | Path) -> "TaskSpec":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") == cls.schema_version:
            return cls.from_dict(payload)
        return from_legacy_v3(payload)


# --------------------------------------------------------------------------
# legacy robotics_task_instruction.v3 -> TaskSpec
# --------------------------------------------------------------------------

_LEGACY_PREDICATE = {
    "inside": "inside",
    "in_container": "inside",
    "on": "on",
    "on_region": "on_region",
    "stacked": "stacked",
    "inserted": "inserted",
    "hung": "hung",
    "covering": "covering",
    "contact": "contact",
    "pressed": "pressed",
    "drawer_open": "joint_open",
    "door_open": "joint_open",
    "closed": "joint_closed",
    "inverted": "inverted",
    "pose_delta": "pose_delta",
}

_LEGACY_PRIMITIVE = {
    "place": "transport",
    "stack": "transport",
    "insert": "transport",
    "hang": "transport",
    "cover": "transport",
    "push": "push",
    "pull": "articulate",
    "close": "articulate",
    "press": "articulate",
    "flip": "reorient",
    "strike": "tool_contact",
}

_LEGACY_ROLE = {"manipuland": "manipuland", "destination": "destination", "support": "support",
                "tool": "tool", "distractor": "distractor"}


def from_legacy_v3(payload: dict[str, Any]) -> TaskSpec:
    if payload.get("schema_version") not in {
        "robotics_task_instruction.v2",
        "robotics_task_instruction.v3",
    }:
        raise ValueError(f"unsupported task schema {payload.get('schema_version')!r}")
    objects = tuple(
        TaskObject(
            object_id=o["object_id"],
            category=o.get("category", ""),
            role=_LEGACY_ROLE.get(o.get("role", ""), "distractor"),
            description=o.get("visual_description", ""),
        )
        for o in payload["objects"]
    )
    articulations = tuple(
        Articulation(
            component_id=a["component_id"],
            host_object_id=a["host_object_id"],
            joint_type=a.get("joint_type", "revolute"),
            description=a.get("visual_description", ""),
        )
        for a in payload.get("articulation_components", ())
    )
    predicates = [
        Predicate(
            kind=_LEGACY_PREDICATE[p["kind"]],
            subject=p["subject"],
            target=p.get("object"),
            params={
                **dict(p.get("params") or {}),
                **({"threshold": p["threshold"]} if p.get("threshold") is not None else {}),
            },
        )
        for p in payload.get("success_predicates", ())
    ]
    steps: list[Step] = []
    for op in payload.get("action_program", ()):
        operation = op["operation"]
        if operation == "pick":
            continue  # a pick is the first half of the following transport
        primitive = _LEGACY_PRIMITIVE.get(operation)
        if primitive is None:
            raise ValueError(f"legacy operation {operation!r} has no primitive")
        subject = op["subjects"][0]
        goal = next((p for p in predicates if p.subject == subject), None)
        if goal is None:
            raise ValueError(f"no success predicate for step subject {subject!r}")
        constraints = dict(op.get("constraints") or {})
        if constraints.pop("upright", False):
            constraints["keep_upright"] = True
        steps.append(
            Step(
                primitive=primitive,
                subject=subject,
                target=op.get("target"),
                goal=goal,
                constraints=constraints,
            )
        )
    family = steps[0].primitive if steps else "unknown"
    spec = TaskSpec(
        case_id=payload["case_id"],
        family=family,
        instruction=payload["instruction"],
        objects=objects,
        steps=tuple(steps),
        articulations=articulations,
        robot=payload.get("robot", "franka_robotiq_2f85"),
        fps=float(payload.get("fps", 15.0)),
        initial_state=payload.get("initial_state", ""),
        seed=int(payload.get("case_seed", 0)) % (2**31),
    )
    spec.validate()
    return spec
