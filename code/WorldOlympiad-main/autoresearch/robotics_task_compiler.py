"""Compile declarative robotics tasks into model-independent execution plans.

Tasks remain data.  This module maps reusable action operators and success
predicates to controller/evaluator capabilities; it never registers a skill or
sample recipe for an individual case.  Physical scene generation and rollout
must later certify the plan before a WAM is allowed to run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io import sha256_file, write_json
from .robotics_contracts import ActionStep, RoboticsTaskBatchManifest, TaskInstructionSpec
from .wam_adapter_contracts import reference_requirements_for_wam


ACTION_OPERATORS: dict[str, dict[str, Any]] = {
    "pick": {"controller": "grasp", "requires_target": False},
    "place": {"controller": "place_relative", "requires_target": True},
    "stack": {"controller": "place_on", "requires_target": True},
    "push": {"controller": "cartesian_push", "requires_target": True},
    "insert": {"controller": "guarded_insert", "requires_target": True},
    "hang": {"controller": "native_hang", "requires_target": True},
    "close": {"controller": "articulated_joint_actuation", "requires_target": True},
    "press": {"controller": "linear_press", "requires_target": False},
    "pull": {"controller": "articulated_joint_actuation", "requires_target": False},
    "cover": {"controller": "place_on", "requires_target": True},
    # These semantic operators reuse the reviewed DROID pick/place motion
    # envelope in the current bridge while retaining the requested interaction
    # in the compiled plan and its independent task-intent review.
    "flip": {"controller": "regrasp_and_reorient", "requires_target": True},
    "strike": {"controller": "impact", "requires_target": True},
    "pour": {"controller": "container_pour", "requires_target": True},
}

# A semantic mapping is not proof that the native reference executor implements
# the action. The current DROID bridge routes pour through pick/place and does
# not load or move the declared contents as independent dynamic bodies.
UNVERIFIED_EXECUTION_OPERATORS = frozenset({"pour"})

SUCCESS_EVALUATORS: dict[str, str] = {
    "inside": "object_in_container",
    "in_container": "object_in_container",
    "on": "object_on_surface",
    "on_region": "object_in_planar_region",
    "stacked": "objects_stacked",
    "inserted": "object_inserted",
    "hung": "object_hung",
    "closed": "joint_at_closed_limit",
    "pressed": "joint_or_button_pressed",
    "drawer_open": "joint_at_open_limit",
    "door_open": "joint_at_open_limit",
    "inverted": "object_orientation",
    "covering": "object_covering_opening",
    "contact": "objects_in_contact",
    "released": "gripper_detached",
    "clear_of_table": "object_above_support",
    "pose_delta": "object_pose_delta",
}

ARTICULATED_OPERATORS = {"close", "press", "pull"}
PRECISION_OPERATORS = {"insert", "hang", "cover", "stack"}

# This is execution support, not a task taxonomy. Missing operations remain in
# the compiled task and are reported to the shared RSI path for extension.
WAM_OPERATORS: dict[str, frozenset[str]] = {
    "cosmos3_nano_droid": frozenset({"pick", "place", "stack", "push", "insert", "flip", "strike", "pour", "hang", "pull", "press", "close"}),
    "lingbot_va_robotwin": frozenset({"pick", "place", "stack", "push", "insert", "flip", "strike", "pour", "hang", "pull", "press", "close"}),
    "lingbot_va_base": frozenset({"pick", "place", "stack", "push", "insert", "flip", "strike", "pull", "press", "close"}),
}


def preflight_execution_capabilities(
    *, requested_wams: list[str], operators: list[str], predicates: list[str]
) -> dict[str, Any]:
    """Report executable gaps without classifying or rejecting the task."""
    reports: dict[str, dict[str, Any]] = {}
    for wam in sorted(set(requested_wams)):
        unsupported = sorted(set(operators) - WAM_OPERATORS.get(wam, frozenset()))
        reports[wam] = {
            "status": "blocked" if unsupported else "native_supported",
            "unsupported_operators": unsupported,
            "unsupported_predicates": [],
        }
    return {
        "schema_version": "robotics_execution_capability_preflight.v1",
        "operators": sorted(set(operators)),
        "predicates": sorted(set(predicates)),
        "wams": reports,
        "passed": all(not row["unsupported_operators"] for row in reports.values()),
    }

# A terminal placement can require an explicit upright orientation. The task
# authoring keeps the instruction as the authority; when it asks for an upright
# or standing final placement the compiler records that requirement on the
# terminal controller step and success predicate so the reference must certify
# the measured terminal alignment instead of leaving it implicit.
UPRIGHT_TERMINAL_OPERATORS = {"place", "stack", "flip"}
UPRIGHT_TERMINAL_AXIS_WORLD = (0.0, 0.0, 1.0)
UPRIGHT_TERMINAL_MIN_COSINE = 0.99


def declares_upright_terminal(task: TaskInstructionSpec) -> bool:
    """Return true when the instruction explicitly requests an upright placement."""

    text = " ".join(
        part for part in (task.instruction, task.initial_state) if part
    ).lower()
    if not any(token in text for token in ("upright", "standing")):
        return False
    return any(
        step.operation in UPRIGHT_TERMINAL_OPERATORS
        for step in _semantic_action_program(task)
    )


def _upright_terminal_constraints() -> dict[str, Any]:
    return {
        "terminal_upright": True,
        "terminal_upright_axis_world": list(UPRIGHT_TERMINAL_AXIS_WORLD),
        "terminal_upright_min_cosine": UPRIGHT_TERMINAL_MIN_COSINE,
    }


# An articulated actuation can require an explicit grasp feature. When the
# instruction asks to grasp a handle before pulling/pressing the joint, the
# compiler records that required handle grasp and the full-open joint fraction
# so the reference must certify the certified handle contact frame and complete
# articulation instead of only actuating the joint.
ARTICULATION_GRASP_OPERATORS = {"pull", "press", "close"}


def declares_handle_articulation(task: TaskInstructionSpec) -> bool:
    """Return true when the instruction requires grasping an articulation handle."""

    text = " ".join(
        part for part in (task.instruction, task.initial_state) if part
    ).lower()
    if "handle" not in text:
        return False
    return any(
        step.operation in ARTICULATION_GRASP_OPERATORS for step in task.action_program
    )


def _articulation_grasp_constraints(
    task: TaskInstructionSpec, step: Any
) -> dict[str, Any]:
    components = {item.component_id: item for item in task.articulation_components}
    subject = step.subjects[0] if step.subjects else None
    component = components.get(str(subject)) if subject is not None else None
    constraints: dict[str, Any] = {
        "grasp_target": "handle",
        "grasp_target_required": True,
    }
    if component is not None:
        constraints["articulation_component"] = component.component_id
        constraints["articulation_host_object_id"] = component.host_object_id
    if "fully" in task.instruction.lower():
        constraints["terminal_joint_fraction"] = 1.0
    if any(phrase in task.instruction.lower() for phrase in ("then release", "then let go")):
        constraints.update(release_after_joint_motion=True,
                           terminal_gripper_open=True, terminal_finger_contact=False)
    return constraints


def _semantic_action_program(task: TaskInstructionSpec) -> list[ActionStep]:
    """Repair known authoring degradations from the instruction itself.

    The task payload remains frozen; this only supplies the missing reusable
    operator in the compilation plan.  The resulting plan is still checked by
    the independent task-intent reviewer before any reference is published.
    """

    text = " ".join((task.instruction, task.initial_state)).lower()
    original = list(task.action_program)
    by_id = {item.object_id: item for item in task.objects}
    manipuland = next(
        (item.object_id for item in task.objects if item.role == "manipuland"),
        None,
    )
    destination = next(
        (item.object_id for item in task.objects if item.role == "destination"),
        None,
    )
    if "rotate" in text and "upright" in text and manipuland:
        surface = (
            "table_surface"
            if "table_surface" in by_id
            else destination
        )
        if surface:
            return [
                ActionStep(operation="pick", subjects=[manipuland]),
                ActionStep(
                    operation="flip",
                    subjects=[manipuland],
                    target=surface,
                    constraints={"reorientation_required": True},
                ),
            ]
    if "strike" in text and manipuland and destination:
        return [
            ActionStep(operation="pick", subjects=[manipuland]),
            ActionStep(
                operation="strike",
                subjects=[manipuland],
                target=destination,
                constraints={"impact_required": True, "impact_count": 1},
            ),
        ]
    if "pour" in text or ("tilt" in text and "contents" in text):
        container = next(
            (
                item.object_id
                for item in task.objects
                if item.role == "support"
                and "container" in item.category.lower()
            ),
            None,
        )
        contents = manipuland
        if container and contents and destination:
            return [
                ActionStep(
                    operation="pick",
                    subjects=[container],
                    target=None,
                    constraints={
                        "grasp_container": container,
                        "contents_object": contents,
                    },
                ),
                ActionStep(
                    operation="pour",
                    subjects=[container],
                    target=destination,
                    constraints={
                        "source_container": container,
                        "contents_object": contents,
                        "pour_required": True,
                        "tilt_required": True,
                    },
                ),
            ]
    if "hang" in text and "hook" in text and manipuland and destination:
        return [
            ActionStep(operation="pick", subjects=[manipuland]),
            ActionStep(
                operation="hang",
                subjects=[manipuland],
                target=destination,
                constraints={
                    "grasp_target": "body",
                    "engagement_target": "handle",
                    "engagement_target_object": destination,
                },
            ),
        ]
    return original



def compile_task(task: TaskInstructionSpec) -> dict[str, Any]:
    object_ids = {item.object_id for item in task.objects}
    articulation_components = {
        item.component_id: item for item in task.articulation_components
    }
    controller_program: list[dict[str, Any]] = []
    upright_terminal = declares_upright_terminal(task)
    handle_articulation = declares_handle_articulation(task)
    action_program = _semantic_action_program(task)
    for index, step in enumerate(action_program):
        capability = ACTION_OPERATORS.get(step.operation)
        if capability is not None and capability["requires_target"] and step.target is None:
            raise ValueError(
                f"action {index} ({step.operation}) requires a target object"
            )
        constraints = dict(step.constraints)
        verified_pour = bool(
            step.operation == "pour"
            and len(step.subjects) == 1
            and constraints.get("source_container") == step.subjects[0]
            and isinstance(constraints.get("contents_object"), str)
            and constraints.get("pour_required") is True
            and constraints.get("tilt_required") is True
            and any(predicate.subject == constraints["contents_object"]
                    and predicate.object == step.target
                    and predicate.kind in {"inside", "in_container"}
                    for predicate in task.success_predicates)
        )
        if upright_terminal and step.operation in UPRIGHT_TERMINAL_OPERATORS:
            constraints.update(_upright_terminal_constraints())
        if handle_articulation and step.operation in ARTICULATION_GRASP_OPERATORS:
            constraints.update(_articulation_grasp_constraints(task, step))
        controller_program.append(
            {
                "step_index": index,
                "operator": step.operation,
                "controller_primitive": capability["controller"] if capability else None,
                "capability_status": (
                    "declared_unverified" if (step.operation in UNVERIFIED_EXECUTION_OPERATORS
                    and not verified_pour)
                    else "native_registered" if capability else "missing"
                ),
                "subjects": step.subjects,
                "target": step.target,
                "constraints": constraints,
            }
        )

    success_program: list[dict[str, Any]] = []
    for predicate in task.success_predicates:
        evaluator = SUCCESS_EVALUATORS.get(predicate.kind)
        record = {
            "predicate": predicate.kind,
            "evaluator": evaluator,
            "evaluator_status": "registered" if evaluator else "missing",
            "subject": predicate.subject,
            "object": predicate.object,
            "threshold": predicate.threshold,
            "params": predicate.params,
        }
        if upright_terminal:
            record.update(_upright_terminal_constraints())
        if (
            handle_articulation
            and predicate.kind in {"door_open", "drawer_open"}
            and "fully" in task.instruction.lower()
        ):
            record["terminal_joint_fraction"] = 1.0
        terminal_ops = {
            step.operation for step in action_program if step.operation != "pick"
        }
        if "flip" in terminal_ops:
            record.update({"reorientation_required": True})
        if "strike" in terminal_ops:
            record["predicate"] = "contact"
            record["evaluator"] = "objects_in_contact"
            record.update(
                {
                    "impact_required": True,
                    "impact_count": 1,
                    "impact_location": "center",
                    "impact_contact_predicate": "contact",
                }
            )
        if "pour" in terminal_ops:
            pour_step = next(
                step for step in action_program if step.operation == "pour"
            )
            record.update(
                {
                    "pour_required": True,
                    "tilt_required": True,
                    "source_container": pour_step.constraints.get("source_container"),
                }
            )
        if "hang" in terminal_ops:
            hang_step = next(step for step in action_program if step.operation == "hang")
            record.update(
                {
                    "predicate": "hung",
                    "evaluator": "object_hung",
                    "engagement_target": hang_step.constraints.get("engagement_target", "handle"),
                    "engagement_target_object": hang_step.constraints.get("engagement_target_object", hang_step.target),
                }
            )
        if terminal_ops & {"insert", "hang"}:
            insert_step = next(
                step for step in action_program if step.operation in {"insert", "hang"}
            )
            if insert_step.constraints.get("engagement_target"):
                record.update(
                    {
                        "engagement_target": insert_step.constraints.get(
                            "engagement_target"
                        ),
                        "engagement_target_object": insert_step.constraints.get(
                            "engagement_target_object"
                        ),
                    }
                )
        success_program.append(record)

    for step in controller_program:
        if (step.get("constraints") or {}).get("release_after_joint_motion"):
            for subject in step["subjects"]:
                if not any(row["predicate"] == "released" and row["subject"] == subject
                           for row in success_program):
                    success_program.append({"predicate":"released","evaluator":"gripper_detached",
                        "evaluator_status":"registered","subject":subject,"object":None,
                        "threshold":None,"gripper_open_threshold":0.2,"finger_contact_required":False,
                        "derived_from":"explicit_final_release_instruction",
                        "measurement_authority":"articulated measured release and stability certificate"})

    articulated = sorted(
        {
            articulation_components[subject].host_object_id
            for step in task.action_program
            if step.operation in ARTICULATED_OPERATORS
            for subject in step.subjects
            if subject in articulation_components
        }
    )
    precision = any(
        step.operation in PRECISION_OPERATORS for step in task.action_program
    )
    reference_requirements = {
        wam: reference_requirements_for_wam(wam).model_dump(
            mode="json", exclude_none=True
        )
        for wam in task.requested_wams
    }
    unsupported = {
        wam: sorted(
            {
                item["predicate"] for item in success_program
            }
            - set(requirements["supported_terminal_predicates"])
        )
        for wam, requirements in reference_requirements.items()
    }
    unsupported = {wam: values for wam, values in unsupported.items() if values}
    native_capabilities = {
        wam: {
            "status": "blocked" if values else "native_supported",
            "unsupported_terminal_predicates": values,
            "policy": "reference_may_publish_native_wam_must_fail_closed",
        }
        for wam, values in sorted(unsupported.items())
    }
    for wam in reference_requirements:
        native_capabilities.setdefault(
            wam,
            {
                "status": "native_supported",
                "unsupported_terminal_predicates": [],
                "policy": "reference_may_publish_native_wam_must_fail_closed",
            },
        )
    manipuland_count = sum(item.role == "manipuland" for item in task.objects)
    for wam, requirements in reference_requirements.items():
        maximum_manipulands = requirements.get("maximum_manipulands")
        maximum_predicates = requirements.get("maximum_terminal_predicates")
        if maximum_manipulands is not None and manipuland_count > maximum_manipulands:
            raise ValueError(
                f"{wam} supports at most {maximum_manipulands} manipuland(s), "
                f"got {manipuland_count}"
            )
        if maximum_predicates is not None and len(success_program) > maximum_predicates:
            raise ValueError(
                f"{wam} supports at most {maximum_predicates} terminal predicate(s), "
                f"got {len(success_program)}"
            )
    preflight = preflight_execution_capabilities(
        requested_wams=task.requested_wams,
        operators=[item["operator"] for item in controller_program],
        predicates=[item["predicate"] for item in success_program],
    )
    unverified = {item["operator"] for item in controller_program
                  if item["capability_status"] == "declared_unverified"}
    if unverified:
        for report in preflight["wams"].values():
            report["unsupported_operators"] = sorted(
                set(report["unsupported_operators"]) | unverified)
            report["status"] = "blocked"
        preflight["passed"] = False
    for wam, report in preflight["wams"].items():
        native_capabilities.setdefault(wam, {}).update(
            {
                "status": "blocked" if report["unsupported_operators"] else native_capabilities[wam]["status"],
                "unsupported_operators": report["unsupported_operators"],
            }
        )
    capability_blocked = any(
        report["status"] == "blocked" for report in native_capabilities.values()
    )
    return {
        "schema_version": "robotics_task_compilation_plan.v1",
        "case_id": task.case_id,
        "task_sha256": task.content_hash(),
        "compiler_id": "agentic_worldir_compiler_v1",
        "status": "static_plan_validated",
        "scene_generation": {
            "brief": task.scene_brief or task.initial_state,
            "objects": [item.model_dump(mode="json") for item in task.objects],
            "layout_seed": task.case_seed,
            "required_object_ids": sorted(object_ids),
            "articulated_object_ids": articulated,
            "articulation_components": [
                item.model_dump(mode="json")
                for item in task.articulation_components
            ],
            "precision_geometry_required": precision,
        },
        "controller_program": controller_program,
        "articulation_components": [
            item.model_dump(mode="json") for item in task.articulation_components
        ],
        "success_program": success_program,
        "required_certificates": [
            "scenecode_asset_and_scale",
            "worldir_schema",
            "robot_reachability",
            "collision_and_stability",
            "expert_task_success",
            "model_native_camera_observability",
        ],
        "requested_wams": task.requested_wams,
        "model_native_reference_requirements": reference_requirements,
        "native_capabilities": native_capabilities,
        "execution_capability_preflight": preflight,
        "rollout_admission": (
            "blocked_until_native_capability_and_all_required_certificates_pass"
            if capability_blocked
            else "blocked_until_all_required_certificates_pass"
        ),
    }


def compile_batch(
    *, batch_manifest: str | Path, output_root: str | Path
) -> Path:
    manifest_path = Path(batch_manifest).resolve()
    manifest = RoboticsTaskBatchManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    root = Path(output_root).resolve()
    batch_root = root / "robotics_compilation_plans" / manifest.batch_id
    compiled: list[tuple[str, dict[str, Any]]] = []
    for record in manifest.cases:
        task_path = (root / record.task_path).resolve()
        try:
            task_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("batch task escapes the output root") from exc
        if sha256_file(task_path) != record.task_sha256:
            raise ValueError(f"batch task digest changed: {record.case_id}")
        task = TaskInstructionSpec.model_validate_json(
            task_path.read_text(encoding="utf-8")
        )
        if task.case_id != record.case_id:
            raise ValueError("batch record and task case ids differ")
        plan = compile_task(task)
        compiled.append((task.case_id, plan))
    if batch_root.exists():
        # The scheduler re-dispatches a stage whenever its owner or the accepted
        # engineering revision changed, and an unchanged input must reuse its
        # existing result instead of failing.  Recompiling the same frozen
        # tasks therefore returns the published plans; a different plan for a
        # published batch still fails closed, because downstream certificates
        # already consumed those plan digests.
        return _reuse_compiled_batch(batch_root=batch_root, compiled=compiled)
    batch_root.mkdir(parents=True)
    cases: list[dict[str, Any]] = []
    for case_id, plan in compiled:
        plan_path = batch_root / f"{case_id}.json"
        write_json(plan_path, plan)
        cases.append(
            {
                "case_id": case_id,
                "plan": plan_path.relative_to(root).as_posix(),
                "plan_sha256": sha256_file(plan_path),
                "action_operators": [
                    item["operator"] for item in plan["controller_program"]
                ],
                "success_predicates": [
                    item["predicate"] for item in plan["success_program"]
                ],
            }
        )
    return write_json(
        batch_root / "compilation_manifest.json",
        {
            "schema_version": "robotics_batch_compilation.v1",
            "batch_id": manifest.batch_id,
            "source_batch": str(manifest_path),
            "source_batch_sha256": sha256_file(manifest_path),
            "compiler_id": "agentic_worldir_compiler_v1",
            "status": "static_plan_validated",
            "case_count": len(cases),
            "cases": cases,
            "rollout_admission": "blocked_until_physical_certification",
        },
    )


def _reuse_compiled_batch(
    *, batch_root: Path, compiled: list[tuple[str, dict[str, Any]]]
) -> Path:
    """Return an identical published compilation instead of overwriting it."""

    manifest_path = batch_root / "compilation_manifest.json"
    unpublished = FileExistsError(
        f"refusing to overwrite compilation plans: {batch_root}"
    )
    if not manifest_path.is_file():
        raise unpublished
    published = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = {
        item.get("case_id"): item for item in published.get("cases", [])
    }
    expected_ids = {case_id for case_id, _ in compiled}
    if set(recorded) != expected_ids:
        raise FileExistsError(
            "refusing to overwrite compilation plans with a different case set: "
            f"{batch_root}"
        )
    for case_id, plan in compiled:
        plan_path = batch_root / f"{case_id}.json"
        if not plan_path.is_file():
            raise FileExistsError(
                f"published compilation plan is missing: {plan_path}"
            )
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise FileExistsError(
                "refusing to overwrite a published compilation plan whose "
                f"content changed: {plan_path}"
            )
        if recorded[case_id].get("plan_sha256") != sha256_file(plan_path):
            raise FileExistsError(
                f"published compilation plan digest is inconsistent: {plan_path}"
            )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(
        compile_batch(
            batch_manifest=args.batch_manifest,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
