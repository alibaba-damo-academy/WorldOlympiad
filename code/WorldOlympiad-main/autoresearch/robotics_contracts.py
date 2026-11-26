"""Fail-closed contracts for the robotics imagination-video harness.

The contracts deliberately separate one simulated trajectory from its rendered
camera views.  WAM adapters consume named physical views; they may resize or
tile them, but may never invent a missing view by copying another camera.
"""

from __future__ import annotations

import math
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .models import Contract


CameraRole = Literal[
    "top",
    "wrist",
    "exterior_left",
    "exterior_right",
    "native_high",
    "native_left_wrist",
    "native_right_wrist",
]
WAMName = Literal[
    "lingbot_va_base",
    "lingbot_va_robotwin",
    "cosmos3_nano_droid",
]
CompatibilityStatus = Literal["native_compatible", "ood_stress", "blocked"]


class TaskObject(Contract):
    object_id: str
    category: str
    role: str
    visual_description: str
    geometry_constraints: dict[str, Any] = Field(default_factory=dict)

    @field_validator("object_id", "category", "role")
    @classmethod
    def safe_task_symbol(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", value):
            raise ValueError("task object ids, categories, and roles must be safe symbols")
        return value


class ArticulationComponentSpec(Contract):
    """A task-addressable moving component inside one SceneCode asset."""

    component_id: str
    host_object_id: str
    semantic_kind: str
    joint_type: Literal["prismatic", "revolute"]
    visual_description: str
    geometry_constraints: dict[str, Any] = Field(default_factory=dict)

    @field_validator("component_id", "host_object_id", "semantic_kind")
    @classmethod
    def safe_component_symbol(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", value):
            raise ValueError("articulation component symbols must be safe identifiers")
        return value


class SuccessPredicate(Contract):
    kind: str
    subject: str
    object: str | None = None
    threshold: float | None = None
    # Preserve declarative predicate semantics (for example a measured asset
    # up-axis) through the typed scene/compiler boundary.
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind", "subject", "object")
    @classmethod
    def safe_predicate_symbol(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_.-]{0,63}", value
        ):
            raise ValueError("success predicate symbols must be safe identifiers")
        return value


class ActionStep(Contract):
    """One declarative action in an agent-authored task program.

    The operation name is deliberately open-ended.  Simulator compilers report
    whether they can realize it; the task contract does not turn every action
    family into a skill or a per-case recipe.
    """

    operation: str
    subjects: list[str]
    target: str | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)

    @field_validator("operation", "target")
    @classmethod
    def safe_action_symbol(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_.-]{0,63}", value
        ):
            raise ValueError("action operation and target must be safe symbols")
        return value

    @field_validator("subjects")
    @classmethod
    def nonempty_safe_subjects(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("an action step needs at least one subject")
        if any(
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", value)
            for value in values
        ):
            raise ValueError("action subjects must be safe symbols")
        return values


class ObjectInContainerCaseParameters(Contract):
    """Frozen, simulator-independent layout for one generated case."""

    manipuland_center_xy_m: tuple[float, float]
    destination_center_xy_m: tuple[float, float]

    @model_validator(mode="after")
    def finite_distinct_centers(self) -> "ObjectInContainerCaseParameters":
        values = (*self.manipuland_center_xy_m, *self.destination_center_xy_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("generated case centers must be finite")
        distance = math.dist(
            self.manipuland_center_xy_m, self.destination_center_xy_m
        )
        if distance <= 0.05:
            raise ValueError("generated manipuland and destination must be distinct")
        return self


class TaskInstructionSpec(Contract):
    schema_version: Literal[
        "robotics_task_instruction.v2", "robotics_task_instruction.v3"
    ] = "robotics_task_instruction.v3"
    case_id: str
    case_seed: int | None = None
    case_parameters: ObjectInContainerCaseParameters | dict[str, Any] | None = None
    instruction: str
    scene_brief: str | None = None
    action_program: list[ActionStep] = Field(default_factory=list)
    robot: Literal["franka_panda", "franka_robotiq_2f85"] = "franka_panda"
    simulator: Literal["pybullet"] = "pybullet"
    objects: list[TaskObject]
    articulation_components: list[ArticulationComponentSpec] = Field(
        default_factory=list
    )
    initial_state: str
    success_predicates: list[SuccessPredicate]
    duration_sec: float = 12.0
    fps: float = 15.0
    # Deprecated task-level hint retained only for schema compatibility.
    # Registered model adapters own every formal reference render size.
    output_size_wh: tuple[int, int] | None = None
    requested_wams: list[WAMName] = Field(
        default_factory=lambda: ["lingbot_va_base", "cosmos3_nano_droid"]
    )

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_family_label(cls, value: Any) -> Any:
        # Old frozen task files may still carry this non-semantic label. It is
        # ignored at the input boundary and is never part of the task model.
        if isinstance(value, dict) and "task_family" in value:
            value = dict(value)
            value.pop("task_family", None)
        return value

    @model_validator(mode="after")
    def executable_task(self) -> "TaskInstructionSpec":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", self.case_id):
            raise ValueError("case_id must be a safe 1-96 character identifier")
        object_ids = [item.object_id for item in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("task object ids must be unique")
        component_ids = [item.component_id for item in self.articulation_components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("articulation component ids must be unique")
        overlap = set(object_ids) & set(component_ids)
        if overlap:
            raise ValueError(
                f"object and articulation component ids overlap: {sorted(overlap)}"
            )
        unknown_hosts = {
            item.host_object_id for item in self.articulation_components
        } - set(object_ids)
        if unknown_hosts:
            raise ValueError(
                f"articulation components reference unknown hosts: {sorted(unknown_hosts)}"
            )
        roles = {item.role for item in self.objects}
        if "support" not in roles or (
            "manipuland" not in roles and not self.articulation_components
        ):
            raise ValueError(
                "task needs a support and either a rigid manipuland or an "
                "articulation component"
            )
        if not self.success_predicates:
            raise ValueError("task needs deterministic success predicates")
        known = set(object_ids) | set(component_ids)
        referenced: set[str] = set()
        for predicate in self.success_predicates:
            referenced.add(predicate.subject)
            if predicate.object is not None:
                referenced.add(predicate.object)
        for step in self.action_program:
            referenced.update(step.subjects)
            if step.target is not None:
                referenced.add(step.target)
        unknown = referenced - known
        if unknown:
            raise ValueError(f"task program references unknown objects: {sorted(unknown)}")
        if self.duration_sec <= 0:
            raise ValueError("reference duration must be positive")
        if self.output_size_wh is not None and any(
            value <= 0 for value in self.output_size_wh
        ):
            raise ValueError("legacy output_size_wh values must be positive")
        if self.fps < 12:
            raise ValueError("reference rendering must be at least 12 fps")
        if len(self.requested_wams) != len(set(self.requested_wams)):
            raise ValueError("requested WAMs must be unique")
        generated = (self.case_seed, self.case_parameters)
        if any(value is not None for value in generated):
            if any(value is None for value in generated):
                raise ValueError(
                    "generated tasks must bind case_seed and case_parameters together"
                )
            if self.case_seed is None or self.case_seed < 0:
                raise ValueError("generated task case_seed must be non-negative")
        return self


class RoboticsTaskBatchCase(Contract):
    case_id: str
    case_seed: int
    task_path: str
    task_sha256: str
    recipe_id: str | None = None
    compiler_id: str = "agentic_worldir_compiler_v1"
    semantic_signature: str | None = None
    reference_run_id: str
    wam_run_id: str


class RoboticsTaskBatchManifest(Contract):
    schema_version: Literal[
        "robotics_task_batch.v1", "robotics_task_batch.v2"
    ] = "robotics_task_batch.v2"
    batch_id: str
    requested_wam: WAMName
    identity_seed: int
    generator_id: str
    proposal_sha256: str | None = None
    generation_mode: Literal[
        "legacy_layout_sampler",
        "agent_authored_declarative",
        "model_authored_declarative",
    ] = (
        "model_authored_declarative"
    )
    cases: list[RoboticsTaskBatchCase]

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_family_label(cls, value: Any) -> Any:
        if isinstance(value, dict) and "task_family" in value:
            value = dict(value)
            value.pop("task_family", None)
        return value

    @model_validator(mode="after")
    def frozen_unique_cases(self) -> "RoboticsTaskBatchManifest":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", self.batch_id):
            raise ValueError("batch_id must be a safe 1-96 character identifier")
        if not self.cases:
            raise ValueError("robotics task batch must contain cases")
        case_ids = [case.case_id for case in self.cases]
        seeds = [case.case_seed for case in self.cases]
        task_paths = [case.task_path for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("robotics task batch case ids must be unique")
        if len(seeds) != len(set(seeds)):
            raise ValueError("robotics task batch seeds must be unique")
        if len(task_paths) != len(set(task_paths)):
            raise ValueError("robotics task batch task paths must be unique")
        signatures = [case.semantic_signature for case in self.cases]
        if self.schema_version == "robotics_task_batch.v2":
            if self.proposal_sha256 is None or len(self.proposal_sha256) != 64:
                raise ValueError("declarative batch needs its proposal SHA-256")
            if any(not value for value in signatures):
                raise ValueError("declarative batch cases need semantic signatures")
            if len(signatures) != len(set(signatures)):
                raise ValueError(
                    "declarative batch contains exact semantic duplicates"
                )
        return self


class CameraViewArtifact(Contract):
    view_id: CameraRole
    camera_type: Literal["fixed_world", "robot_mounted"]
    camera_signature: str
    video: str
    conditioning_image: str
    cae_base_scene: str
    width: int
    height: int
    fps: float
    frame_count: int
    sha256: str

    @field_validator("camera_signature", "sha256")
    @classmethod
    def digest_like(cls, value: str) -> str:
        if len(value) < 16:
            raise ValueError("camera signatures and artifact digests must be explicit")
        return value


class RoboticsReferenceManifest(Contract):
    schema_version: Literal[
        "robotics_reference_bundle.v4",
        "robotics_reference_bundle.v5",
        "robotics_reference_bundle.v6",
    ] = (
        "robotics_reference_bundle.v5"
    )
    case_id: str
    recipe_id: str | None = None
    harness_id: str | None = None
    task_graph: str | None = None
    task_graph_sha256: str | None = None
    scene_generation_manifest: str | None = None
    scene_generation_manifest_sha256: str | None = None
    semantic_identity_certificate: str | None = None
    semantic_identity_certificate_sha256: str | None = None
    task_instruction: str
    scene_prompt_bundle: str
    source_scenecode_release: str
    robot_embodiment: Literal["franka_panda", "franka_robotiq_2f85"] = (
        "franka_panda"
    )
    visual_variant_manifest: str | None = None
    visual_variant_manifest_sha256: str | None = None
    visual_physics_alignment_certificate: str | None = None
    visual_physics_alignment_certificate_sha256: str | None = None
    physical_variant_manifest: str | None = None
    physical_variant_manifest_sha256: str | None = None
    camera_rig: str | None = None
    camera_rig_sha256: str | None = None
    simulator: Literal["pybullet"] = "pybullet"
    renderer: Literal["scenecode_official_blender"] = (
        "scenecode_official_blender"
    )
    trajectory: str
    trajectory_sha256: str
    trajectory_json: str
    trajectory_json_sha256: str
    simulation_certificate: str
    simulation_passed: bool
    physical_render_certificate: str
    physical_render_passed: bool
    views: list[CameraViewArtifact]
    quality_certificate: str
    quality_passed: bool
    visual_context_certificate: str | None = None
    visual_context_certificate_sha256: str | None = None
    visual_context_passed: bool | None = None
    bundle_artifacts_sha256: dict[str, str] | None = None

    @model_validator(mode="after")
    def certified_multiview(self) -> "RoboticsReferenceManifest":
        if self.schema_version in {
            "robotics_reference_bundle.v4",
            "robotics_reference_bundle.v5",
        } and not self.recipe_id:
            raise ValueError("legacy reference bundle must bind one reviewed recipe")
        if self.schema_version == "robotics_reference_bundle.v6":
            if self.recipe_id is not None:
                raise ValueError("agentic reference v6 must not bind a sample recipe")
            if not self.harness_id:
                raise ValueError("agentic reference v6 must bind one reusable harness")
            provenance = {
                "task graph": (self.task_graph, self.task_graph_sha256),
                "scene generation": (
                    self.scene_generation_manifest,
                    self.scene_generation_manifest_sha256,
                ),
                "semantic identity": (
                    self.semantic_identity_certificate,
                    self.semantic_identity_certificate_sha256,
                ),
            }
            for label, (path, digest) in provenance.items():
                if not path or not digest or len(digest) != 64:
                    raise ValueError(
                        f"agentic reference v6 must bind its {label} artifact and digest"
                    )
        if len(self.trajectory_sha256) != 64 or len(self.trajectory_json_sha256) != 64:
            raise ValueError("reference trajectories need explicit SHA-256 digests")
        if (self.visual_variant_manifest is None) != (
            self.visual_variant_manifest_sha256 is None
        ):
            raise ValueError("visual variant manifest path and digest must be paired")
        if (self.visual_physics_alignment_certificate is None) != (
            self.visual_physics_alignment_certificate_sha256 is None
        ):
            raise ValueError("visual/physics alignment path and digest must be paired")
        if (self.visual_physics_alignment_certificate_sha256 is not None
                and len(self.visual_physics_alignment_certificate_sha256) != 64):
            raise ValueError("visual/physics alignment needs a SHA-256 digest")
        if (
            self.visual_variant_manifest_sha256 is not None
            and len(self.visual_variant_manifest_sha256) != 64
        ):
            raise ValueError("visual variant manifest needs an explicit SHA-256 digest")
        if (self.physical_variant_manifest is None) != (
            self.physical_variant_manifest_sha256 is None
        ):
            raise ValueError("physical variant manifest path and digest must be paired")
        if (
            self.physical_variant_manifest_sha256 is not None
            and len(self.physical_variant_manifest_sha256) != 64
        ):
            raise ValueError("physical variant manifest needs an explicit SHA-256 digest")
        if (self.camera_rig is None) != (self.camera_rig_sha256 is None):
            raise ValueError("camera rig path and digest must be paired")
        if self.camera_rig_sha256 is not None and len(self.camera_rig_sha256) != 64:
            raise ValueError("camera rig needs an explicit SHA-256 digest")
        if (self.visual_context_certificate is None) != (
            self.visual_context_certificate_sha256 is None
        ):
            raise ValueError(
                "in-context visual context certificate path and digest must be paired"
            )
        if self.schema_version == "robotics_reference_bundle.v6":
            if self.visual_context_certificate is None:
                raise ValueError(
                    "agentic reference v6 must bind a rendered in-context visual "
                    "context certificate"
                )
            if self.visual_context_passed is not True:
                raise ValueError(
                    "agentic reference v6 cannot be published with a failed "
                    "in-context visual context review"
                )
        if (
            not self.simulation_passed
            or not self.physical_render_passed
            or not self.quality_passed
        ):
            raise ValueError("uncertified robotics references cannot enter WAM inference")
        by_id = {view.view_id: view for view in self.views}
        if len(by_id) != len(self.views):
            raise ValueError("reference camera view_ids must be unique")
        required = {"top", "wrist", "exterior_left", "exterior_right"}
        missing = required - set(by_id)
        if missing:
            raise ValueError(f"reference bundle is missing physical views: {sorted(missing)}")
        signatures = [view.camera_signature for view in self.views]
        if len(signatures) != len(set(signatures)):
            raise ValueError("physical camera roles may not share one camera signature")
        if by_id["wrist"].camera_type != "robot_mounted":
            raise ValueError("wrist must be rendered from a robot-mounted camera")
        if any(by_id[name].camera_type != "fixed_world" for name in required - {"wrist"}):
            raise ValueError("top and exterior cameras must be fixed-world cameras")
        if self.schema_version in {
            "robotics_reference_bundle.v5",
            "robotics_reference_bundle.v6",
        }:
            if not self.bundle_artifacts_sha256:
                raise ValueError(
                    "robotics reference v5 must bind every bundle artifact"
                )
            for label, paths in {
                "videos": [view.video for view in self.views],
                "conditioning images": [
                    view.conditioning_image for view in self.views
                ],
                "CAE base scenes": [view.cae_base_scene for view in self.views],
            }.items():
                if len(paths) != len(set(paths)):
                    raise ValueError(
                        f"robotics reference v5 physical views must use distinct {label}"
                    )
            timelines = {(view.frame_count, view.fps) for view in self.views}
            if (
                len(timelines) != 1
                or any(
                    view.frame_count < 2
                    or view.fps <= 0
                    or view.width <= 0
                    or view.height <= 0
                    for view in self.views
                )
            ):
                raise ValueError(
                    "robotics reference v5 views must share one positive timeline"
                )
            for path, digest in self.bundle_artifacts_sha256.items():
                relative = PurePosixPath(path)
                if (
                    not path
                    or path in {".", "robotics_reference_manifest.json"}
                    or relative.is_absolute()
                    or PureWindowsPath(path).is_absolute()
                    or path != relative.as_posix()
                    or ".." in relative.parts
                    or "\\" in path
                ):
                    raise ValueError(
                        f"unsafe robotics reference artifact path: {path!r}"
                    )
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError(
                        f"invalid robotics reference artifact digest: {path!r}"
                    )

            file_paths = {
                "task_instruction.json",
                self.trajectory,
                self.trajectory_json,
                self.simulation_certificate,
                self.physical_render_certificate,
                self.quality_certificate,
                *(view.video for view in self.views),
                *(view.conditioning_image for view in self.views),
            }
            if self.visual_variant_manifest is not None:
                file_paths.add(self.visual_variant_manifest)
            if self.physical_variant_manifest is not None:
                file_paths.add(self.physical_variant_manifest)
            if self.camera_rig is not None:
                file_paths.add(self.camera_rig)
            for path in file_paths:
                if path not in self.bundle_artifacts_sha256:
                    raise ValueError(
                        "robotics reference v5 artifact map does not bind declared "
                        f"file: {path!r}"
                    )
            declared_digests = {
                self.trajectory: self.trajectory_sha256,
                self.trajectory_json: self.trajectory_json_sha256,
                **{view.video: view.sha256 for view in self.views},
            }
            if self.visual_variant_manifest is not None:
                assert self.visual_variant_manifest_sha256 is not None
                declared_digests[self.visual_variant_manifest] = (
                    self.visual_variant_manifest_sha256
                )
            if self.physical_variant_manifest is not None:
                assert self.physical_variant_manifest_sha256 is not None
                declared_digests[self.physical_variant_manifest] = (
                    self.physical_variant_manifest_sha256
                )
            if self.camera_rig is not None:
                assert self.camera_rig_sha256 is not None
                declared_digests[self.camera_rig] = self.camera_rig_sha256
            for path, digest in declared_digests.items():
                if (
                    not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or self.bundle_artifacts_sha256[path] != digest
                ):
                    raise ValueError(
                        "robotics reference v5 declared digest differs from its "
                        f"artifact map: {path!r}"
                    )
            directory_paths = {
                self.scene_prompt_bundle,
                *(view.cae_base_scene for view in self.views),
            }
            for path in directory_paths:
                relative = PurePosixPath(path)
                if (
                    not path
                    or path == "."
                    or relative.is_absolute()
                    or PureWindowsPath(path).is_absolute()
                    or path != relative.as_posix()
                    or ".." in relative.parts
                    or "\\" in path
                ):
                    raise ValueError(
                        f"unsafe robotics reference directory path: {path!r}"
                    )
                prefix = f"{path}/"
                if not any(
                    artifact.startswith(prefix)
                    for artifact in self.bundle_artifacts_sha256
                ):
                    raise ValueError(
                        "robotics reference v5 artifact map does not bind declared "
                        f"directory: {path!r}"
                    )
        elif self.bundle_artifacts_sha256 is not None:
            raise ValueError(
                "legacy robotics reference v4 cannot claim a v5 artifact map"
            )
        return self

    def view(self, view_id: CameraRole) -> CameraViewArtifact:
        for view in self.views:
            if view.view_id == view_id:
                return view
        raise KeyError(view_id)


class WAMInputManifest(Contract):
    schema_version: Literal["wam_input_bundle.v4", "wam_input_bundle.v5"] = (
        "wam_input_bundle.v5"
    )
    case_id: str
    wam: WAMName
    source_model: str
    task_instruction: str
    model_conditioning_prompt: str | None = None
    prompt_manifest: str | None = None
    prompt_manifest_sha256: str | None = None
    reference_manifest: str
    reference_manifest_sha256: str | None = None
    reference_snapshot: str | None = None
    reference_task_snapshot: str | None = None
    reference_task_sha256: str | None = None
    adapter_contract: str
    adapter_contract_sha256: str
    input_images: dict[str, str]
    source_views: dict[str, CameraRole]
    formal_reference_view: CameraRole
    reference_start_frame_index: Literal[0] = 0
    reference_frame_count: int
    runtime_controller: Literal[
        "official_robolab_task", "task_graph_runtime", "upstream_model"
    ]
    official_runtime_task: str | None = None
    formal_reference_transform: Literal[
        "identity_exact",
        "center_crop_cover_zoom_1p0",
        "center_crop_cover_zoom_1p2",
        "center_crop_cover_zoom_1p35",
        "aspect_preserving_letterbox",
    ]
    model_input_size_wh: tuple[int, int]
    simulator_state: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def common_input_invariants(self) -> "WAMInputManifest":
        if len(self.adapter_contract_sha256) != 64:
            raise ValueError("adapter contract must have an explicit SHA-256 digest")
        if self.reference_frame_count < 2:
            raise ValueError("formal reference segment needs at least two frames")
        if self.runtime_controller == "official_robolab_task":
            if not self.official_runtime_task:
                raise ValueError("official RoboLab control needs a task identifier")
        elif self.runtime_controller == "task_graph_runtime":
            if self.official_runtime_task not in {"worldolympiad/task_graph_fixed_horizon", "worldolympiad/task_graph_native_episode"}:
                raise ValueError("task-graph control needs a bound native runtime identity")
        elif self.official_runtime_task is not None:
            raise ValueError("only RoboLab-controlled rollouts may bind a RoboLab task")
        if set(self.input_images) != set(self.source_views):
            raise ValueError("each WAM image needs one explicit source view")
        if len(set(self.source_views.values())) != len(self.source_views):
            raise ValueError("a physical view may not be duplicated across WAM camera inputs")
        if min(self.model_input_size_wh) <= 0:
            raise ValueError("model input dimensions must be positive")
        if self.model_conditioning_prompt is not None and not self.model_conditioning_prompt.strip():
            raise ValueError("model conditioning prompt may not be blank")
        prompt_binding = (
            self.prompt_manifest,
            self.prompt_manifest_sha256,
        )
        if self.schema_version == "wam_input_bundle.v5":
            if self.wam not in {"lingbot_va_base", "cosmos3_nano_droid"}:
                raise ValueError(
                    "WAM input v5 requires a registered typed prompt model"
                )
            if self.model_conditioning_prompt is None:
                raise ValueError("WAM input v5 requires a rendered model prompt")
            if any(value is None for value in prompt_binding):
                raise ValueError(
                    "WAM input v5 requires a prompt manifest path and digest"
                )
            assert self.prompt_manifest is not None
            assert self.prompt_manifest_sha256 is not None
            prompt_path = Path(self.prompt_manifest)
            if (
                prompt_path.is_absolute()
                or not prompt_path.parts
                or ".." in prompt_path.parts
                or self.prompt_manifest in {"", "."}
            ):
                raise ValueError(
                    "WAM input prompt manifest must be a bundle-relative path"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", self.prompt_manifest_sha256):
                raise ValueError(
                    "WAM input prompt manifest needs an explicit SHA-256 digest"
                )
            if not isinstance(self.reference_manifest_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", self.reference_manifest_sha256
            ):
                raise ValueError(
                    "WAM input v5 must bind the exact reference manifest digest"
                )
            snapshot_binding = (
                self.reference_snapshot,
                self.reference_task_snapshot,
                self.reference_task_sha256,
            )
            if any(value is None for value in snapshot_binding):
                raise ValueError(
                    "WAM input v5 requires a self-contained reference/task snapshot"
                )
            for label, value in (
                ("reference snapshot", self.reference_snapshot),
                ("reference task snapshot", self.reference_task_snapshot),
            ):
                assert isinstance(value, str)
                relative = PurePosixPath(value)
                if (
                    not value
                    or value == "."
                    or relative.is_absolute()
                    or PureWindowsPath(value).is_absolute()
                    or value != relative.as_posix()
                    or ".." in relative.parts
                    or "\\" in value
                ):
                    raise ValueError(f"WAM input {label} must be bundle-relative")
            if not isinstance(self.reference_task_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", self.reference_task_sha256
            ):
                raise ValueError("WAM input reference task needs an exact digest")
        else:
            if any(value is not None for value in prompt_binding):
                raise ValueError(
                    "legacy WAM input v4 may not claim a typed prompt manifest"
                )
            if self.reference_manifest_sha256 is not None:
                raise ValueError(
                    "legacy WAM input v4 may not claim a v5 reference digest"
                )
            if any(
                value is not None
                for value in (
                    self.reference_snapshot,
                    self.reference_task_snapshot,
                    self.reference_task_sha256,
                )
            ):
                raise ValueError("legacy WAM input v4 may not claim v5 snapshots")
        if self.wam in ("lingbot_va_base", "lingbot_va_robotwin"):
            prompt = self.model_conditioning_prompt or self.task_instruction
            # Upstream LingBot includes the prompt in one filesystem component
            # and appends a timestamp. Keep a fail-closed safety margin below
            # Linux's 255-byte component limit before allocating GPU memory.
            if len(prompt.encode("utf-8")) > 220:
                raise ValueError(
                    "LingBot conditioning prompt exceeds the 220-byte upstream "
                    "runtime path budget"
                )
        return self


class WAMProfileCompatibilityReport(Contract):
    """Audited upstream-interface decision made before media preparation."""

    schema_version: Literal["wam_profile_compatibility.v1"] = (
        "wam_profile_compatibility.v1"
    )
    case_id: str
    wam: WAMName
    source_model: str
    adapter_profile: str
    reference_manifest: str
    reference_manifest_sha256: str
    reference_robot: str
    reference_end_effector: str
    published_embodiment: str
    required_camera_mapping: dict[str, str]
    state_conventions: dict[str, str]
    compatibility_status: CompatibilityStatus
    inference_allowed: bool
    benchmark_interpretation: str
    limitations: list[str]
    evidence: list[str]

    @model_validator(mode="after")
    def honest_compatibility(self) -> "WAMProfileCompatibilityReport":
        if self.compatibility_status == "blocked" and self.inference_allowed:
            raise ValueError("a blocked WAM profile cannot allow inference")
        if self.compatibility_status == "native_compatible" and self.limitations:
            raise ValueError("native compatibility cannot carry embodiment limitations")
        if self.compatibility_status == "ood_stress":
            if not self.limitations:
                raise ValueError("OOD stress profiles must state their limitations")
            if "OOD" not in self.benchmark_interpretation:
                raise ValueError("OOD stress interpretation must be explicit")
        if len(self.required_camera_mapping) != len(
            set(self.required_camera_mapping.values())
        ):
            raise ValueError("WAM profile camera slots must map to distinct views")
        return self


class ConditioningDetection(Contract):
    role: Literal["manipuland", "destination", "end_effector"]
    object_id: str | None = None
    prompt: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    linear_fraction: float
    fully_in_frame: bool = True
    inside_size_contract: bool


class ConditioningViewObservation(Contract):
    camera_key: str
    source_view: CameraRole
    image: str
    image_sha256: str
    size_wh: tuple[int, int]
    black_fraction: float
    luminance_stddev: float
    detections: list[ConditioningDetection]
    role_size_contracts: dict[str, tuple[float, float]]
    required_roles: list[Literal["manipuland", "destination", "end_effector"]]
    visible_roles: list[Literal["manipuland", "destination", "end_effector"]]
    required_object_ids: list[str] = Field(default_factory=list)
    visible_object_ids: list[str] = Field(default_factory=list)
    passed: bool


class ConditioningObservabilityCertificate(Contract):
    schema_version: Literal[
        "conditioning_observability.v1",
        "conditioning_observability.v2",
    ] = (
        "conditioning_observability.v2"
    )
    case_id: str
    wam: WAMName
    input_manifest: str
    input_manifest_sha256: str
    profile_report: str
    profile_report_sha256: str
    detector: str
    detector_model_revision: str
    visibility_authority: Literal[
        "semantic_detector_legacy",
        "deterministic_renderer_labels_primary_semantic_detector_secondary",
    ] = "semantic_detector_legacy"
    score_threshold: float
    role_size_contracts: dict[str, tuple[float, float]]
    formal_camera_key: str
    views: list[ConditioningViewObservation]
    checks: dict[str, bool]
    blockers: list[str]
    passed: bool

    @model_validator(mode="after")
    def fail_closed(self) -> "ConditioningObservabilityCertificate":
        keys = [item.camera_key for item in self.views]
        if len(keys) != len(set(keys)):
            raise ValueError("conditioning observations must have unique camera keys")
        if self.formal_camera_key not in keys:
            raise ValueError("formal camera observation is missing")
        expected = all(self.checks.values()) and all(item.passed for item in self.views)
        if self.passed != expected:
            raise ValueError("observability pass state disagrees with its evidence")
        if self.passed and self.blockers:
            raise ValueError("a passing observability certificate cannot have blockers")
        if not self.passed and not self.blockers:
            raise ValueError("a failed observability certificate needs exact blockers")
        return self


class WAMConditioningCandidate(Contract):
    frame_index: int
    input_manifest: str
    profile_report: str
    observability_certificate: str
    passed: bool
    blockers: list[str]


class WAMConditioningSelection(Contract):
    schema_version: Literal["wam_conditioning_selection.v1"] = (
        "wam_conditioning_selection.v1"
    )
    case_id: str
    wam: WAMName
    reference_manifest: str
    reference_manifest_sha256: str
    candidate_frame_indices: list[int]
    candidates: list[WAMConditioningCandidate]
    selected_frame_index: int | None = None
    selected_input_manifest: str | None = None
    selected_profile_report: str | None = None
    selected_observability_certificate: str | None = None
    selection_rule: Literal[
        "first_passing_simulator_candidate_without_wam_output_v1"
    ] = "first_passing_simulator_candidate_without_wam_output_v1"
    blockers: list[str] = Field(default_factory=list)
    passed: bool

    @model_validator(mode="after")
    def first_passing_candidate(self) -> "WAMConditioningSelection":
        frames = [item.frame_index for item in self.candidates]
        if frames != self.candidate_frame_indices or len(frames) != len(set(frames)):
            raise ValueError("conditioning candidates must preserve the declared order")
        selected = next((item for item in self.candidates if item.passed), None)
        if selected is None:
            if self.passed:
                raise ValueError("conditioning selection passed without a passing candidate")
            if any(
                value is not None
                for value in (
                    self.selected_frame_index,
                    self.selected_input_manifest,
                    self.selected_profile_report,
                    self.selected_observability_certificate,
                )
            ):
                raise ValueError("failed conditioning selection cannot select artifacts")
            if not self.blockers:
                raise ValueError("failed conditioning selection needs exact blockers")
            return self
        if not self.passed:
            raise ValueError("conditioning selection found a pass but reports failure")
        if self.blockers:
            raise ValueError("passing conditioning selection cannot have blockers")
        if self.selected_frame_index != selected.frame_index:
            raise ValueError("conditioning selection did not choose the first pass")
        if self.selected_input_manifest != selected.input_manifest:
            raise ValueError("selected input manifest disagrees with the first pass")
        if self.selected_profile_report != selected.profile_report:
            raise ValueError("selected profile report disagrees with the first pass")
        if self.selected_observability_certificate != selected.observability_certificate:
            raise ValueError("selected observability certificate disagrees with the first pass")
        return self


class WAMOutputManifest(Contract):
    schema_version: Literal["wam_output_bundle.v3", "wam_output_bundle.v4"] = (
        "wam_output_bundle.v4"
    )
    case_id: str
    wam: WAMName
    source_model: str
    checkpoint_revision: str
    input_manifest: str
    adapter_contract_sha256: str
    model_metadata: str
    raw_video: str
    raw_video_certificate: str | None = None
    selected_view: CameraRole
    crop_xywh: tuple[int, int, int, int]
    aligned_video: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    termination_reason: Literal[
        "task_completed", "official_task_timeout", "native_task_timeout", "reference_horizon_complete",
        "upstream_stopped", "evaluation_window_complete"
    ]
    task_success: bool
    conditioning_frame_mae_8bit: float
    quality_certificate: str
    quality_passed: Literal[True] = True
    profile_report: str | None = None
    observability_certificate: str | None = None
    detection_report: str | None = None
    bundle_artifacts_sha256: dict[str, str] | None = None

    @model_validator(mode="after")
    def exact_output_identity(self) -> "WAMOutputManifest":
        if len(self.adapter_contract_sha256) != 64:
            raise ValueError("WAM output must bind the adapter contract digest")
        if self.fps <= 0 or self.frame_count < 2 or self.duration_sec <= 0:
            raise ValueError("WAM output timing must be positive and non-empty")
        if not all(
            math.isfinite(value)
            for value in (
                self.fps,
                self.duration_sec,
                self.conditioning_frame_mae_8bit,
            )
        ):
            raise ValueError("WAM output numeric evidence must be finite")
        if abs(self.duration_sec - self.frame_count / self.fps) > 1e-6:
            raise ValueError("WAM output duration must describe its native frame timeline")
        if (self.termination_reason != "evaluation_window_complete"
                and self.task_success != (self.termination_reason == "task_completed")):
            raise ValueError("WAM task success must agree with its termination reason")
        if self.schema_version == "wam_output_bundle.v4":
            if self.raw_video_certificate is None:
                raise ValueError("WAM output v4 must bind raw video provenance")
            certificate = Path(self.raw_video_certificate)
            if (
                certificate.is_absolute()
                or self.raw_video_certificate != certificate.as_posix()
                or ".." in certificate.parts
            ):
                raise ValueError("WAM raw video certificate must be bundle-relative")
            if not self.bundle_artifacts_sha256:
                raise ValueError("WAM output v4 must bind every bundle artifact")
            preflight = (
                self.profile_report,
                self.observability_certificate,
                self.detection_report,
            )
            if any(value is not None for value in preflight) and not all(
                value is not None for value in preflight
            ):
                raise ValueError("WAM output must bind all preflight artifacts together")
            for value in (item for item in preflight if item is not None):
                relative = PurePosixPath(value)
                if (
                    relative.is_absolute()
                    or PureWindowsPath(value).is_absolute()
                    or value != relative.as_posix()
                    or ".." in relative.parts
                    or "\\" in value
                ):
                    raise ValueError("WAM preflight artifact must be bundle-relative")
            for path, digest in self.bundle_artifacts_sha256.items():
                relative = Path(path)
                if (
                    relative.is_absolute()
                    or path != relative.as_posix()
                    or ".." in relative.parts
                    or path == "output_manifest.json"
                ):
                    raise ValueError(f"unsafe WAM bundle artifact path: {path!r}")
                if len(digest) != 64:
                    raise ValueError(f"invalid WAM bundle artifact digest: {path!r}")
        else:
            if self.bundle_artifacts_sha256 is not None:
                raise ValueError("legacy WAM output v3 cannot claim a v4 artifact map")
            if self.raw_video_certificate is not None:
                raise ValueError("legacy WAM output v3 cannot claim v4 raw provenance")
            if any(
                value is not None
                for value in (
                    self.profile_report,
                    self.observability_certificate,
                    self.detection_report,
                )
            ):
                raise ValueError("legacy WAM output v3 cannot claim v4 preflight evidence")
        return self


class SemanticRoleTrack(Contract):
    role: Literal["manipuland", "destination", "end_effector"]
    visible_sample_count: int
    sample_count: int
    visibility_fraction: float
    first_center_xy: tuple[float, float] | None = None
    last_center_xy: tuple[float, float] | None = None
    max_center_displacement_fraction: float | None = None
    first_linear_fraction: float | None = None
    last_linear_fraction: float | None = None
    max_scale_ratio: float | None = None


class WAMSemanticMotionReport(Contract):
    schema_version: Literal["wam_semantic_motion.v1"] = "wam_semantic_motion.v1"
    case_id: str
    wam: WAMName
    output_manifest: str
    output_manifest_sha256: str
    detection_report: str
    detection_report_sha256: str
    sampled_frame_indices: list[int]
    role_tracks: list[SemanticRoleTrack]
    metrics: dict[str, float | bool | None]
    findings: list[str]
    task_progress_observed: bool
    preserved_for_evaluation: Literal[True] = True

    @model_validator(mode="after")
    def evidence_not_filter(self) -> "WAMSemanticMotionReport":
        roles = [item.role for item in self.role_tracks]
        if roles != ["manipuland", "destination", "end_effector"]:
            raise ValueError("semantic audit must report all roles in fixed order")
        if not self.sampled_frame_indices:
            raise ValueError("semantic audit needs sampled frames")
        if not self.task_progress_observed and not self.findings:
            raise ValueError("absent task progress needs a diagnostic finding")
        return self


class EvaluationPairManifest(Contract):
    schema_version: Literal["robotics_evaluation_pair.v2"] = (
        "robotics_evaluation_pair.v2"
    )
    case_id: str
    wam: WAMName
    instruction: str
    reference_view: CameraRole
    reference_video: str
    generated_view: CameraRole
    generated_video: str
    cae_base_scene: str
    size_wh: tuple[int, int]
    fps: float
    alignment_certificate: str
    reference_sha256: str
    generated_sha256: str

    @model_validator(mode="after")
    def matched_view(self) -> "EvaluationPairManifest":
        if self.reference_view != self.generated_view:
            raise ValueError("CAE inputs must represent the same physical camera role")
        return self


SIMULATOR_PRIMITIVE_SDF_AUTHORITY = "simulator_builtin"

# Simulator-owned built-ins are loaded by the physics engine itself and own no
# SceneCode asset.  They are listed explicitly so a trajectory can never use
# this declaration to hide a real asset from source archiving.
SIMULATOR_PRIMITIVE_SOURCES = frozenset({"PyBullet plane.urdf"})


def simulator_primitive_record(record: Any) -> dict[str, Any] | None:
    """Return the normalized declaration of one simulator-owned prepared asset.

    A ``trajectory.prepared_assets`` entry is normally a SceneCode SDF asset
    that must be archived into the reference bundle.  A simulator built-in is
    not: it is declared by its built-in source, body id and fixed flag, and it
    must never carry SDF paths, otherwise the packager could skip archiving a
    real asset.  ``None`` means the record is not such a declaration.
    """

    if not isinstance(record, dict):
        return None
    if record.get("source_sdf") or record.get("physics_sdf"):
        return None
    source = record.get("source")
    if not isinstance(source, str) or source not in SIMULATOR_PRIMITIVE_SOURCES:
        return None
    body_id = record.get("body_id")
    if not isinstance(body_id, int) or isinstance(body_id, bool):
        return None
    if record.get("fixed") is not True:
        return None
    return {
        "sdf_authority": SIMULATOR_PRIMITIVE_SDF_AUTHORITY,
        "source": source,
        "body_id": body_id,
        "fixed": True,
    }


class GenerationSkillSelection(Contract):
    skill_id: str
    reason: str
    inputs: list[str]
    outputs: list[str]


class GenerationPlan(Contract):
    schema_version: Literal["robotics_generation_plan.v2"] = (
        "robotics_generation_plan.v2"
    )
    case_id: str
    task_digest: str
    routing_inputs: dict[str, Any]
    selected_skills: list[GenerationSkillSelection]
    skipped_skills: list[dict[str, str]]
    selection_modified: Literal[False] = False
    validation_status: Literal["ok"] = "ok"

    @model_validator(mode="after")
    def model_independent(self) -> "GenerationPlan":
        forbidden = {"model_id", "checkpoint", "model_score"} & set(self.routing_inputs)
        if forbidden:
            raise ValueError(f"planning cannot route on evaluated model identity: {forbidden}")
        ids = [item.skill_id for item in self.selected_skills]
        if len(ids) != len(set(ids)):
            raise ValueError("generation skills must be unique")
        return self
