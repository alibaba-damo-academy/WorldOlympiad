"""Versioned, content-addressed contracts for the SimReady pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .io import write_json


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def write_json(self, path: str | Path) -> Path:
        return write_json(
            path,
            self.model_dump(mode="json", exclude_none=True),
        )


class Pose(Contract):
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    @field_validator("quat_xyzw")
    @classmethod
    def normalized_quaternion(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        norm = sum(component * component for component in value) ** 0.5
        if norm < 1e-12:
            raise ValueError("quaternion norm must be non-zero")
        return tuple(float(component / norm) for component in value)  # type: ignore[return-value]


class Geometry(Contract):
    name: str
    kind: Literal["mesh", "box", "sphere", "cylinder"]
    pose: Pose = Field(default_factory=Pose)
    path: str | None = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    size: tuple[float, float, float] | None = None
    radius: float | None = None
    length: float | None = None
    rgba: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def validate_kind_fields(self) -> "Geometry":
        required = {
            "mesh": self.path,
            "box": self.size,
            "sphere": self.radius,
            "cylinder": self.radius is not None and self.length is not None,
        }
        if not required[self.kind]:
            raise ValueError(f"geometry kind {self.kind!r} is missing required parameters")
        return self


class Inertial(Contract):
    mass: float
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia: tuple[float, float, float, float, float, float]
    # ixx, iyy, izz, ixy, ixz, iyz

    @field_validator("mass")
    @classmethod
    def positive_mass(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("dynamic mass must be positive")
        return float(value)


class ContactMaterial(Contract):
    name: str = "generic"
    lateral_friction: float = 0.5
    rolling_friction: float = 0.0
    restitution: float = 0.05
    linear_damping: float = 0.04
    angular_damping: float = 0.04
    density_kg_m3: float | None = None


class LinkIR(Contract):
    name: str
    pose: Pose = Field(default_factory=Pose)
    inertial: Inertial | None = None
    visuals: list[Geometry] = Field(default_factory=list)
    collisions: list[Geometry] = Field(default_factory=list)


class JointIR(Contract):
    name: str
    type: Literal["fixed", "revolute", "continuous", "prismatic"]
    parent: str
    child: str
    pose: Pose = Field(default_factory=Pose)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    lower: float | None = None
    upper: float | None = None
    effort: float = 100.0
    velocity: float = 10.0
    damping: float = 0.1


class BodyIR(Contract):
    name: str
    semantic_category: str
    pose: Pose = Field(default_factory=Pose)
    internal_model_pose: Pose = Field(default_factory=Pose)
    links: list[LinkIR]
    joints: list[JointIR] = Field(default_factory=list)
    material: ContactMaterial = Field(default_factory=ContactMaterial)
    static: bool = False
    target: bool = False
    support: str | None = None
    affordances: list[str] = Field(default_factory=list)
    source_path: str | None = None
    source_sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_topology(self) -> "BodyIR":
        names = [link.name for link in self.links]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"body {self.name!r} must have unique links")
        known = set(names)
        children: set[str] = set()
        for joint in self.joints:
            if joint.parent not in known or joint.child not in known:
                raise ValueError(f"joint {joint.name!r} refers to an unknown link")
            if joint.child in children:
                raise ValueError(f"link {joint.child!r} has multiple parent joints")
            children.add(joint.child)
        return self


class RoomIR(Contract):
    name: str
    pose: Pose = Field(default_factory=Pose)
    body_names: list[str] = Field(default_factory=list)


class WorldIR(Contract):
    schema_version: Literal["worldir.v1"] = "worldir.v1"
    scene_id: str
    prompt: str
    units: Literal["m"] = "m"
    up_axis: Literal["Z"] = "Z"
    provenance: dict[str, Any] = Field(default_factory=dict)
    rooms: list[RoomIR] = Field(default_factory=list)
    bodies: list[BodyIR]

    @model_validator(mode="after")
    def validate_names(self) -> "WorldIR":
        names = [body.name for body in self.bodies]
        if len(names) != len(set(names)):
            raise ValueError("body names must be unique")
        known = set(names)
        for room in self.rooms:
            missing = set(room.body_names) - known
            if missing:
                raise ValueError(f"room {room.name!r} refers to unknown bodies: {sorted(missing)}")
        return self


class CameraProfile(Contract):
    width: int = 960
    height: int = 540
    fov_y_deg: float = 50.0
    near_m: float = 0.02
    far_m: float = 20.0
    target_fraction_min: float = 0.03
    target_fraction_max: float = 0.20


class Trigger(Contract):
    operation: Literal["set_velocity", "apply_force", "apply_torque", "remove_support"]
    vector: tuple[float, float, float] = (0.0, 0.0, 0.0)
    duration_s: float = 0.0
    position_local: tuple[float, float, float] | None = None


class OutcomePredicate(Contract):
    kind: Literal["target_zone", "tip", "stack_collapse"]
    min_drop_m: float = 0.0
    min_displacement_m: float = 0.0
    min_tilt_deg: float = 0.0
    max_final_distance_m: float | None = None
    require_support_loss: bool = False
    require_support_retained: bool = False
    require_contact_change: bool = False


class RobotProgram(Contract):
    schema_version: Literal["robot_program.v1"] = "robot_program.v1"
    robot: str = "robot"
    contact_object: str | None = None
    destination_object: str | None = None
    operation: Literal["cartesian_push", "cartesian_pick_place"] = "cartesian_push"
    control_mode: Literal[
        "kinematic_joint_position",
        "continuous_kinematic_joint_position",
    ] = "kinematic_joint_position"
    end_effector_link_name: str = "worldolympiad_pusher"
    arm_joint_names: list[str] = Field(
        default_factory=lambda: [f"panda_joint{index}" for index in range(1, 8)]
    )
    finger_joint_names: list[str] = Field(
        default_factory=lambda: ["panda_finger_joint1", "panda_finger_joint2"]
    )
    motion_start_s: float = 0.25
    motion_end_s: float = 1.50
    # Includes the 6 cm half-extent of the certified pusher tool plus a 4 cm
    # pre-contact gap, so the hidden settle cannot trigger the event early.
    approach_clearance_m: float = 0.10
    travel_distance_m: float = 0.25
    base_distance_from_contact_m: float = 0.35
    contact_height_fraction: float = 0.55
    approach_height_m: float = 0.18
    lift_height_m: float = 0.22
    grasp_phase: float = 0.28
    release_phase: float = 0.86
    max_joint_force_n: float = 1000.0
    position_gain: float = 0.60
    max_joint_speed_rad_s: float = 2.0
    contact_only: bool = True

    @model_validator(mode="after")
    def valid_robot_program(self) -> "RobotProgram":
        if not 0 <= self.motion_start_s < self.motion_end_s:
            raise ValueError("robot motion interval must be ordered and non-negative")
        if (
            self.approach_clearance_m < 0
            or self.travel_distance_m <= 0
            or self.base_distance_from_contact_m <= 0
        ):
            raise ValueError("robot approach and travel distances must be valid")
        if not 0 <= self.contact_height_fraction <= 1:
            raise ValueError("robot contact height fraction must lie in [0, 1]")
        if self.approach_height_m <= 0 or self.lift_height_m <= 0:
            raise ValueError("pick/place approach and lift heights must be positive")
        if not 0.20 < self.grasp_phase < 0.46:
            raise ValueError("pick/place grasp_phase must lie between descent and lift")
        if not 0.82 < self.release_phase < 1:
            raise ValueError("pick/place release_phase must lie between placement and retreat")
        if self.operation == "cartesian_pick_place" and not self.destination_object:
            raise ValueError("cartesian_pick_place requires destination_object")
        if self.max_joint_speed_rad_s <= 0:
            raise ValueError("max_joint_speed_rad_s must be positive")
        return self


class EpisodeSpec(Contract):
    schema_version: Literal["episode.v1"] = "episode.v1"
    episode_id: str
    scene_path: str
    event_type: Literal["block_sort", "cup_tip", "block_push", "stack_collapse"]
    target: str
    support: str
    actors: list[str]
    seed: int = 0
    physics_dt_s: float = 1.0 / 240.0
    hidden_settle_s: float = 1.0
    duration_s: float = 4.0
    fps: int = 15
    trigger_time_s: float = 0.25
    trigger: Trigger
    trigger_grid: list[float] = Field(default_factory=list)
    expected: OutcomePredicate
    camera: CameraProfile = Field(default_factory=CameraProfile)
    robot_program: RobotProgram | None = None
    variant: Literal["reference", "identity", "opposite_direction", "changed_speed", "no_trigger"] = "reference"

    @model_validator(mode="after")
    def validate_timeline(self) -> "EpisodeSpec":
        if self.physics_dt_s <= 0 or self.duration_s <= 0 or self.fps <= 0:
            raise ValueError("physics_dt_s, duration_s, and fps must be positive")
        if not 0 <= self.trigger_time_s < self.duration_s:
            raise ValueError("trigger_time_s must lie inside the recorded rollout")
        return self


class SceneAssetRef(Contract):
    asset_id: str
    semantic_class: str
    asset_path: str | None = None
    link_names: list[str] = Field(default_factory=list)


class EpisodeScene(Contract):
    world_ir_path: str
    assets: list[SceneAssetRef]

    @model_validator(mode="after")
    def unique_assets(self) -> "EpisodeScene":
        names = [asset.asset_id for asset in self.assets]
        if not names or len(names) != len(set(names)):
            raise ValueError("episode.v2 scene assets must be non-empty and unique")
        return self


class InitialBodyState(Contract):
    pose: Pose = Field(default_factory=Pose)
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_positions: dict[str, float] = Field(default_factory=dict)
    joint_velocities: dict[str, float] = Field(default_factory=dict)


class BodyPhysicsOverride(Contract):
    mass: float | None = None
    lateral_friction: float | None = None
    rolling_friction: float | None = None
    restitution: float | None = None
    linear_damping: float | None = None
    angular_damping: float | None = None

    @model_validator(mode="after")
    def valid_physics(self) -> "BodyPhysicsOverride":
        if self.mass is not None and self.mass <= 0:
            raise ValueError("physics mass override must be positive")
        for name in ("lateral_friction", "rolling_friction", "linear_damping", "angular_damping"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.restitution is not None and not 0 <= self.restitution <= 1:
            raise ValueError("restitution must lie in [0, 1]")
        return self


class EpisodePhysics(Contract):
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    timestep_s: float = 1.0 / 240.0
    bodies: dict[str, BodyPhysicsOverride] = Field(default_factory=dict)

    @field_validator("timestep_s")
    @classmethod
    def positive_timestep(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("physics timestep must be positive")
        return float(value)


class InterventionSpec(Contract):
    type: Literal[
        "set_velocity",
        "apply_force",
        "apply_torque",
        "remove_support",
        "joint_motor",
        "constraint_release",
        "object_collision",
    ]
    target: str
    value: tuple[float, float, float] | float | None = None
    duration_s: float = 0.0
    start_time_s: float = 0.25
    position_local: tuple[float, float, float] | None = None
    secondary_object: str | None = None
    joint_name: str | None = None

    @model_validator(mode="after")
    def validate_intervention(self) -> "InterventionSpec":
        if self.start_time_s < 0 or self.duration_s < 0:
            raise ValueError("intervention times must be non-negative")
        if self.type == "object_collision" and not self.secondary_object:
            raise ValueError("object_collision requires secondary_object")
        if self.type == "joint_motor" and not self.joint_name:
            raise ValueError("joint_motor requires joint_name")
        return self


class ObservationSpec(Contract):
    target_object: str
    camera_id: str = "camera_0"
    duration_s: float = 4.0
    fps: int = 15
    include_endpoints: bool = True
    camera: CameraProfile = Field(default_factory=CameraProfile)

    @model_validator(mode="after")
    def validate_observation(self) -> "ObservationSpec":
        if self.duration_s <= 0 or self.fps <= 0:
            raise ValueError("observation duration and fps must be positive")
        return self

    @property
    def expected_frame_count(self) -> int:
        base = round(self.duration_s * self.fps)
        return base + 1 if self.include_endpoints else base


class EpisodeFactors(Contract):
    interaction: Literal["push", "collision", "drop", "articulate", "roll", "tip"]
    dynamics: Literal["quasi_static", "acceleration", "ballistic", "bouncing"]
    speed: Literal["slow", "medium", "fast"]
    occlusion: Literal["none", "partial", "severe"]
    contact_complexity: Literal["zero", "single_contact", "multi_contact"]
    camera_motion: Literal["static", "moving"]
    extra: dict[str, str | int | float | bool] = Field(default_factory=dict)


class CounterfactualDeclaration(Contract):
    base_episode_id: str
    changed_factors: list[str]

    @field_validator("changed_factors")
    @classmethod
    def declared_changes_are_unique(cls, value: list[str]) -> list[str]:
        if not value or len(value) != len(set(value)):
            raise ValueError("counterfactual changed_factors must be non-empty and unique")
        return value


class EpisodeSpecV2(Contract):
    """Factor-explicit experiment contract used by Module 2 and Module 3."""

    schema_version: Literal["episode.v2"] = "episode.v2"
    episode_id: str
    scene: EpisodeScene
    initial_state: dict[str, InitialBodyState]
    physics: EpisodePhysics = Field(default_factory=EpisodePhysics)
    intervention: InterventionSpec
    observation: ObservationSpec
    factors: EpisodeFactors
    expected: OutcomePredicate | None = None
    seed: int = 0
    threshold_profile: str
    counterfactual: CounterfactualDeclaration | None = None

    @model_validator(mode="after")
    def validate_references(self) -> "EpisodeSpecV2":
        assets = {asset.asset_id for asset in self.scene.assets}
        unknown_states = set(self.initial_state) - assets
        unknown_physics = set(self.physics.bodies) - assets
        if unknown_states:
            raise ValueError(f"initial_state refers to unknown assets: {sorted(unknown_states)}")
        if unknown_physics:
            raise ValueError(f"physics refers to unknown assets: {sorted(unknown_physics)}")
        required = {self.intervention.target, self.observation.target_object}
        if self.intervention.secondary_object:
            required.add(self.intervention.secondary_object)
        unknown = required - assets
        if unknown:
            raise ValueError(f"episode.v2 refers to unknown assets: {sorted(unknown)}")
        if self.intervention.start_time_s >= self.observation.duration_s:
            raise ValueError("intervention must start inside the observation interval")
        return self


class ProbeResult(Contract):
    name: str
    passed: bool
    measured: float | int | str | bool | dict[str, Any] | list[Any] | None = None
    threshold: float | int | str | bool | dict[str, Any] | list[Any] | None = None
    detail: str = ""


class RepairRecord(Contract):
    attempt: int
    operation: str
    target: str
    detail: str


class PhysicsReport(Contract):
    schema_version: Literal["physics_report.v1"] = "physics_report.v1"
    episode_id: str
    status: Literal["pass", "repaired_pass", "quarantined"]
    world_hash: str
    episode_hash: str
    pybullet_version: str
    renderer: str
    probes: list[ProbeResult]
    repairs: list[RepairRecord] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class EpisodeManifest(Contract):
    schema_version: Literal["episode_manifest.v1"] = "episode_manifest.v1"
    episode_id: str
    benchmark_status: Literal["pipeline_validation"] = "pipeline_validation"
    prompt: str
    world_ir_path: str
    episode_spec_path: str
    physics_report_path: str
    physics_certificate_path: str | None = None
    observation_certificate_path: str | None = None
    rgb_video_path: str
    rgb_frames_dir: str
    depth_dir: str
    segmentation_dir: str
    segmentation_encoding_path: str
    calibration_path: str
    state_trajectory_path: str
    simulator_object_flow_path: str | None = None
    contact_trajectory_path: str
    target_instance_id: int
    body_instance_map: dict[str, int]
    presentation_renderer: str = "pybullet_debug"
    presentation_render_report_path: str | None = None
    cae_manifest_path: str | None = None
    artifact_sha256: dict[str, str] = Field(default_factory=dict)


class ThresholdSource(Contract):
    repository: str
    revision: str
    files: list[str]


class SceneCodePhysicsThresholds(Contract):
    object_penetration_m: float = 0.001
    manipuland_support_tolerance_m: float = 0.02
    furniture_floor_tolerance_m: float = 0.05
    minimum_mesh_dimension_m: float = 0.001
    mesh_relative_dimension: float = 0.02
    top_surface_overlap_tolerance: float = 0.05
    stb_simulation_s: float = 5.0
    stb_timestep_s: float = 0.001
    stb_displacement_m: float = 0.01
    stb_rotation_rad: float = 0.1
    code_object_max_attempts: int = 3
    vhacd_max_convex_hulls: int = 64


class Dream2FlowFrontendThresholds(Contract):
    particle_sampler: Literal["torch_geometric.nn.fps_random_start_false"] = (
        "torch_geometric.nn.fps_random_start_false"
    )
    minimum_particles: int = 64
    maximum_particles: int = 1024
    rgb_input_range: tuple[float, float] = (0.0, 255.0)
    visibility_threshold: float = 0.8


class ObservationThresholds(Contract):
    status: Literal["bootstrap", "pilot_calibrated", "calibrated"] = "bootstrap"
    visible_ratio_min: float
    first_frame_pixels_min: int
    target_fraction_min: float
    target_fraction_max: float
    depth_valid_ratio_min: float


class FlowThresholds(Contract):
    status: Literal["bootstrap", "pilot_calibrated", "calibrated"] = "bootstrap"
    rmse_m_max: float
    direction_cosine_min: float
    missing_frame_fraction_max: float
    longest_missing_run_frames_max: int
    calibration_manifest_sha256: str | None = None


class ThresholdProfile(Contract):
    schema_version: Literal["threshold_profile.v1"] = "threshold_profile.v1"
    profile_id: str
    source: ThresholdSource
    scenecode: SceneCodePhysicsThresholds
    dream2flow: Dream2FlowFrontendThresholds
    observation: ObservationThresholds
    flow: FlowThresholds
    calibration: dict[str, Any] | None = None


CertificateStatus = Literal["pass", "repaired_pass", "quarantined"]


class CertificateBase(Contract):
    episode_id: str
    status: CertificateStatus
    episode_hash: str
    threshold_profile_id: str
    threshold_profile_hash: str
    probes: list[ProbeResult]
    failure_codes: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def status_matches_probes(self) -> "CertificateBase":
        passed = all(probe.passed for probe in self.probes) and not self.failure_codes
        if passed != (self.status in {"pass", "repaired_pass"}):
            raise ValueError("certificate status is inconsistent with probes/failure_codes")
        return self


class PhysicsCertificate(CertificateBase):
    schema_version: Literal["physics_certificate.v2"] = "physics_certificate.v2"
    world_hash: str
    simulator: str
    repairs: list[RepairRecord] = Field(default_factory=list)


class ObservationCertificate(CertificateBase):
    schema_version: Literal["observation_certificate.v1"] = "observation_certificate.v1"
    renderer: str
    frame_count: int
    target_instance_id: int


class FlowCertificate(CertificateBase):
    schema_version: Literal["flow_certificate.v1"] = "flow_certificate.v1"
    simulator_flow_sha256: str
    dream2flow_reference_sha256: str
    factor_stratum: dict[str, str | int | float | bool] = Field(default_factory=dict)


class FreshGenerationCertificate(CertificateBase):
    schema_version: Literal["fresh_generation_certificate.v1"] = (
        "fresh_generation_certificate.v1"
    )
    generation_root: str
    house_state_sha256: str
    assets: dict[str, dict[str, Any]] = Field(default_factory=dict)


class BenchmarkCertificate(Contract):
    schema_version: Literal["benchmark_certificate.v1"] = "benchmark_certificate.v1"
    episode_id: str
    status: CertificateStatus
    threshold_profile_id: str
    threshold_profile_hash: str
    physics_certificate_path: str
    observation_certificate_path: str
    flow_certificate_path: str
    fresh_generation_certificate_path: str
    certificate_sha256: dict[str, str]
    failure_codes: list[str] = Field(default_factory=list)


class CertifiedEpisodeManifest(Contract):
    schema_version: Literal["certified_episode_manifest.v1"] = (
        "certified_episode_manifest.v1"
    )
    episode_id: str
    benchmark_status: Literal["pipeline_validation"] = "pipeline_validation"
    certification_status: Literal["pass", "repaired_pass"]
    threshold_profile_id: str
    episode_manifest_path: str
    benchmark_certificate_path: str
    simulator_object_flow_path: str
    rgb_video_path: str
    depth_dir: str
    segmentation_dir: str
    state_trajectory_path: str
    contact_trajectory_path: str


class ReferenceCorpusEntry(Contract):
    """One evaluator-valid reference used to calibrate benchmark thresholds."""

    episode_id: str
    event_type: str
    factors: dict[str, str | int | float | bool] = Field(default_factory=dict)
    benchmark_status: Literal["pass", "repaired_pass"]
    flow_rmse_m: float
    direction_cosine: float
    missing_frame_fraction: float
    longest_missing_run_frames: int
    visible_ratio: float
    first_frame_pixels: int
    first_frame_bbox_fraction: float
    depth_valid_ratio: float
    source_root: str


class ReferenceCorpusManifest(Contract):
    schema_version: Literal["reference_corpus.v1"] = "reference_corpus.v1"
    corpus_id: str
    entries: list[ReferenceCorpusEntry]
    strata: dict[str, list[str | int | float | bool]] = Field(default_factory=dict)
    source_runs: list[str]
    calibration_policy: dict[str, Any]

    @model_validator(mode="after")
    def unique_episode_sources(self) -> "ReferenceCorpusManifest":
        keys = [(item.source_root, item.episode_id) for item in self.entries]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("reference corpus entries must be non-empty and unique")
        return self


class EvaluationRecord(Contract):
    """Evaluator-valid observation consumed by the Module 3 research loop."""

    episode_id: str
    model_id: str
    score_path: float
    score_velocity: float
    score_contact: float
    factors: dict[str, str | int | float | bool]
    benchmark_certificate_status: Literal["pass", "repaired_pass", "quarantined"]
    artifact_path: str | None = None


class FailureCluster(Contract):
    schema_version: Literal["failure_cluster.v2"] = "failure_cluster.v2"
    cluster_id: str
    model_id: str
    metric: Literal["score_path", "score_velocity", "score_contact"]
    factor_signature: dict[str, str | int | float | bool]
    episode_ids: list[str]
    count: int
    mean_score: float
    baseline_mean_score: float
    effect: float
    severity: float


class ResearchHypothesis(Contract):
    schema_version: Literal["research_hypothesis.v1"] = "research_hypothesis.v1"
    hypothesis_id: str
    cluster_id: str
    factor: str
    claim: str
    test_values: list[str | int | float | bool]
    metric: Literal["score_path", "score_velocity", "score_contact"]
    status: Literal["proposed", "supported", "rejected", "inconclusive"] = "proposed"


class EffectEstimate(Contract):
    factor: str
    low_value: str | int | float | bool
    high_value: str | int | float | bool
    low_mean: float
    high_mean: float
    effect: float
    sample_count: int
    conclusion: Literal["supported", "rejected", "inconclusive"]


class FactorialAnalysis(Contract):
    schema_version: Literal["factorial_analysis.v1"] = "factorial_analysis.v1"
    metric: Literal["score_path", "score_velocity", "score_contact"]
    factor_a: str
    factor_b: str
    cell_means: dict[str, float]
    main_effect_a: float
    main_effect_b: float
    interaction_effect: float
    sample_count: int


class FailureBoundary(Contract):
    schema_version: Literal["failure_boundary.v1"] = "failure_boundary.v1"
    factor: str
    metric: Literal["score_path", "score_velocity", "score_contact"]
    threshold: float
    direction: Literal["increasing", "decreasing"]
    passing_value: float | None = None
    failing_value: float | None = None
    boundary_estimate: float | None = None
    tested_values: list[float]
    status: Literal["bracketed", "all_pass", "all_fail", "insufficient"]


class FailureModeReport(Contract):
    schema_version: Literal["failure_mode_report.v2"] = "failure_mode_report.v2"
    failure_mode_id: str
    model_id: str
    phenomenon: str
    metric: Literal["score_path", "score_velocity", "score_contact"]
    primary_factor: str
    hypothesis_status: Literal["supported", "rejected", "inconclusive"]
    effect: EffectEstimate
    boundary: FailureBoundary | None = None
    evidence_episode_ids: list[str]
    evaluator_valid_only: Literal[True] = True
    confidence: Literal["low", "medium", "high"]


class ResearchProbe(Contract):
    """One controlled factor value requested by the outer research loop."""

    schema_version: Literal["research_probe.v1"] = "research_probe.v1"
    probe_id: str
    model_id: str
    hypothesis_id: str
    base_episode_id: str
    factor: str
    value: str | int | float | bool
    controlled_factors: dict[str, str | int | float | bool]
    priority: float
    rationale: str


class ResearchIterationPlan(Contract):
    """Budget-bounded next action emitted by the deterministic research loop."""

    schema_version: Literal["research_iteration_plan.v1"] = "research_iteration_plan.v1"
    metric: Literal["score_path", "score_velocity", "score_contact"]
    failure_threshold: float
    episode_budget: int
    evaluator_valid_records: int
    clusters: list[FailureCluster]
    hypotheses: list[ResearchHypothesis]
    probes: list[ResearchProbe]
    stop_reason: Literal[
        "continue", "no_evaluator_valid_records", "no_failures",
        "budget_exhausted", "factor_space_exhausted",
    ]

    @model_validator(mode="after")
    def valid_budget(self) -> "ResearchIterationPlan":
        if self.episode_budget < 0 or len(self.probes) > self.episode_budget:
            raise ValueError("research probes must fit the non-negative episode budget")
        return self


class BatchJob(Contract):
    job_id: str
    prompt: str
    priority: int = 0

    @model_validator(mode="after")
    def safe_job(self) -> "BatchJob":
        import re

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", self.job_id):
            raise ValueError("batch job_id must be a safe 1-96 character identifier")
        if not self.prompt.strip():
            raise ValueError("batch prompt must be non-empty")
        return self


class BatchManifest(Contract):
    schema_version: Literal["autoresearch_batch.v1"] = "autoresearch_batch.v1"
    batch_id: str
    jobs: list[BatchJob]

    @model_validator(mode="after")
    def unique_jobs(self) -> "BatchManifest":
        import re

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", self.batch_id):
            raise ValueError("batch_id must be a safe 1-96 character identifier")
        ids = [item.job_id for item in self.jobs]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("batch jobs must be non-empty and unique")
        return self
