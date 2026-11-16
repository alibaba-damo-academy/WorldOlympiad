"""Frozen, model-specific contracts for robotics world-action-model adapters.

The shared WAM manifest records artifacts and case identity.  This module owns
the upstream interface facts that must never be inferred by a generic runner:
camera keys, spatial transforms, native timing, state conventions, and output
extraction.  Every prepared input bundle writes one digest-bound contract.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .models import Contract


LINGBOT_BASE_WAM = "lingbot_va_base"
LINGBOT_ROBOTWIN_WAM = "lingbot_va_robotwin"
LINGBOT_WAM = LINGBOT_BASE_WAM
LINGBOT_SOURCE_MODEL = "Robbyant/lingbot-va-base"
LINGBOT_PROFILE_ID = "lingbot_va_base.demo_i2av.v1"
LINGBOT_MODEL_CONFIG = "demo_i2av"

LINGBOT_ROBOTWIN_SOURCE_MODEL = "robbyant/lingbot-va-posttrain-robotwin"
LINGBOT_ROBOTWIN_PROFILE_ID = "lingbot_va_posttrain_robotwin.robotwin_i2av.v1"
LINGBOT_ROBOTWIN_MODEL_CONFIG = "robotwin_i2av"

COSMOS_WAM = "cosmos3_nano_droid"
COSMOS_SOURCE_MODEL = "nvidia/Cosmos3-Nano-Policy-DROID"
COSMOS_PROFILE_ID = "cosmos3_nano_policy_droid.fixed_exterior_left_cae.v3"

RolloutProtocol = Literal[
    "open_loop_autoregressive_video_action",
    "simulator_closed_loop",
]

TaskVisualRole = Literal["manipuland", "destination", "end_effector"]
# Every entry has a real downstream realization in the shared action/operator
# compiler and the reference materializer: rigid place/stack/insert/cover/push,
# the button control event ("pressed"), and measured articulated joint motion
# ("door_open"/"drawer_open").  A name is never added here as a bare claim.
TerminalPredicate = Literal[
    "inside",
    "in_container",
    "on_region",
    "stacked",
    "inserted",
    "hung",
    "contact",
    "covering",
    "pressed",
    "door_open",
    "drawer_open",
]


class ReferenceRoleRequirement(Contract):
    """One model-facing visibility floor; large objects are never rejected."""

    role: TaskVisualRole
    minimum_linear_fraction: float
    # A large host may legitimately extend past the frame edge: what the video
    # has to show is the manipulated object and the gripper.  The field stays
    # available for roles that really must be whole.
    require_complete_frame: bool = True

    @model_validator(mode="after")
    def usable_floor(self) -> "ReferenceRoleRequirement":
        if not 0.0 < self.minimum_linear_fraction < 1.0:
            raise ValueError("reference visibility floor must lie in (0,1)")
        return self


class ReferenceCameraRequirement(Contract):
    source_view: Literal[
        "top",
        "wrist",
        "exterior_left",
        "exterior_right",
        "native_high",
        "native_left_wrist",
        "native_right_wrist",
    ]
    size_wh: tuple[int, int]
    required_roles: list[ReferenceRoleRequirement]
    require_all_task_instances: Literal[True] = True


class WAMReferenceRequirements(Contract):
    """Facts the reference generator must satisfy before publishing a bundle.

    This contract is available before a case or reference exists.  It keeps
    model-native cameras and observable task semantics in the adapter while
    leaving task content, object identity, and rollout length unconstrained.
    """

    schema_version: Literal["wam_reference_requirements.v1"] = (
        "wam_reference_requirements.v1"
    )
    wam: Literal[
        "lingbot_va_base",
        "lingbot_va_robotwin",
        "cosmos3_nano_droid",
    ]
    profile_id: str
    cameras: list[ReferenceCameraRequirement]
    supported_terminal_predicates: list[TerminalPredicate]
    maximum_manipulands: int | None = None
    maximum_terminal_predicates: int | None = None
    native_render_policy: Literal["direct_native_size_no_rescale"] = (
        "direct_native_size_no_rescale"
    )
    visibility_authority: Literal[
        "deterministic_renderer_labels_primary_semantic_detector_secondary"
    ] = "deterministic_renderer_labels_primary_semantic_detector_secondary"

    @model_validator(mode="after")
    def complete_requirements(self) -> "WAMReferenceRequirements":
        views = [camera.source_view for camera in self.cameras]
        if len(views) != len(set(views)):
            raise ValueError("reference requirement views must be unique")
        if len(self.supported_terminal_predicates) != len(
            set(self.supported_terminal_predicates)
        ):
            raise ValueError("supported terminal predicates must be unique")
        for value in (self.maximum_manipulands, self.maximum_terminal_predicates):
            if value is not None and value < 1:
                raise ValueError("adapter cardinality limits must be positive")
        return self


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenTemporalContract(Contract):
    """Native cadence plus reference provenance; rollout length stays upstream-owned."""

    schema_version: Literal["wam_temporal_contract.v4"] = "wam_temporal_contract.v4"
    reference_start_frame_index: Literal[0] = 0
    reference_fps: float
    reference_frame_count: int
    native_output_fps: float
    duration_policy: Literal["upstream_runtime_owned_v1"] = "upstream_runtime_owned_v1"
    continuation_policy: Literal[
        "monolithic_causal_vae_prefix",
        "generated_visual_and_predicted_state_autoregression",
        "simulator_rgb_and_measured_state_refresh_32_steps",
        "robotwin_rgb_and_measured_state_refresh_per_chunk",
    ]

    @model_validator(mode="after")
    def exact_timeline(self) -> "FrozenTemporalContract":
        if self.reference_fps <= 0 or self.native_output_fps <= 0:
            raise ValueError("reference and native output FPS must be positive")
        if self.reference_frame_count < 2:
            raise ValueError("reference provenance needs at least two frames")
        return self


class CameraInputContract(Contract):
    input_key: str
    source_view: Literal[
        "top",
        "wrist",
        "exterior_left",
        "exterior_right",
        "native_high",
        "native_left_wrist",
        "native_right_wrist",
    ]
    size_wh: tuple[int, int]
    spatial_transform: Literal[
        "identity_exact",
        "center_crop_cover_zoom_1p0",
        "center_crop_cover_zoom_1p2",
        "center_crop_cover_zoom_1p35",
        "cover_zoom_1p0_anchor_x_0p36",
        "aspect_preserving_letterbox",
        "openpi_pil_bilinear_pad_640x360",
        "openpi_pil_bilinear_pad_640x360_then_torch_bilinear_half",
        "lingbot_torch_bilinear_align_corners_false",
    ]
    required_roles: list[Literal["manipuland", "destination", "end_effector"]]
    layout_region_xywh: tuple[int, int, int, int] | None = None

    @model_validator(mode="after")
    def valid_camera(self) -> "CameraInputContract":
        if min(self.size_wh) <= 0:
            raise ValueError("camera input dimensions must be positive")
        if len(self.required_roles) != len(set(self.required_roles)):
            raise ValueError("required camera roles must be unique")
        return self


class ModelStateContract(Contract):
    conditioning: Literal[
        "images_and_text",
        "images_text_and_initial_robot_state",
        "three_images_text_and_dual_arm_state",
    ]
    action_space: Literal[
        "internal_30d_provenance",
        "absolute_joint_position7_gripper",
        "relative_dual_end_effector16",
    ]
    initial_state_fields: list[str] = Field(default_factory=list)
    gripper_convention: Literal[
        "model_internal",
        "droid_raw_0_open_1_closed",
        "robotwin_normalized_dual_gripper",
    ]
    actions_scored: Literal[False] = False


class OutputExtractionContract(Contract):
    raw_layout: Literal[
        "top_then_wrist_horizontal",
        "left_right_wrist_above_high",
        "wrist_above_exterior_left_and_exterior_right",
    ]
    formal_view: Literal[
        "top",
        "wrist",
        "exterior_left",
        "exterior_right",
        "native_high",
    ]
    formal_size_wh: tuple[int, int]
    extraction_policy: Literal[
        "first_equal_width_panel",
        "bottom_full_width_panel",
        "top_two_thirds_then_resize_640x360",
        "wrist360_then_bottom_left_restore_bottom_crop_320x180",
        "wrist360_then_bottom_right_restore_bottom_crop_320x180",
        "bottom_320x256_high_panel",
    ]


class WAMAdapterContract(Contract):
    schema_version: Literal["wam_adapter_contract.v1", "wam_adapter_contract.v2"] = (
        "wam_adapter_contract.v2"
    )
    case_id: str
    wam: Literal[
        "lingbot_va_base",
        "lingbot_va_robotwin",
        "cosmos3_nano_droid",
    ]
    profile_id: str
    source_model: str
    upstream_config: str
    native_input_size_wh: tuple[int, int]
    camera_inputs: list[CameraInputContract]
    temporal: FrozenTemporalContract
    state: ModelStateContract
    output: OutputExtractionContract
    rollout_protocol: RolloutProtocol | None = None
    benchmark_interpretation: Literal["imagination_video_only"] = (
        "imagination_video_only"
    )

    @model_validator(mode="after")
    def exact_upstream_profile(self) -> "WAMAdapterContract":
        keys = [item.input_key for item in self.camera_inputs]
        views = [item.source_view for item in self.camera_inputs]
        if len(keys) != len(set(keys)) or len(views) != len(set(views)):
            raise ValueError("adapter camera keys and physical source views must be unique")

        if self.wam == LINGBOT_BASE_WAM:
            expected_cameras = [
                (
                    "observation.images.top",
                    "top",
                    (256, 256),
                    "identity_exact",
                    ["manipuland", "destination", "end_effector"],
                    None,
                ),
                (
                    "observation.images.wrist",
                    "wrist",
                    (256, 256),
                    "identity_exact",
                    [],
                    None,
                ),
            ]
            expected = {
                "profile_id": LINGBOT_PROFILE_ID,
                "source_model": LINGBOT_SOURCE_MODEL,
                "upstream_config": LINGBOT_MODEL_CONFIG,
                "native_input_size_wh": (256, 256),
                "native_output_fps": 10.0,
                "continuation": "monolithic_causal_vae_prefix",
                "conditioning": "images_and_text",
                "action_space": "internal_30d_provenance",
                "initial_state_fields": [],
                "gripper": "model_internal",
                "raw_layout": "top_then_wrist_horizontal",
                "formal_view": "top",
                "formal_size_wh": (256, 256),
                "extraction": "first_equal_width_panel",
                "rollout_protocol": "open_loop_autoregressive_video_action",
            }
        elif self.wam == LINGBOT_ROBOTWIN_WAM:
            expected_cameras = [
                (
                    "observation.images.cam_high",
                    "native_high",
                    (320, 256),
                    "identity_exact",
                    ["manipuland", "destination", "end_effector"],
                    (0, 128, 320, 256),
                ),
                (
                    "observation.images.cam_left_wrist",
                    "native_left_wrist",
                    (160, 128),
                    "identity_exact",
                    [],
                    (0, 0, 160, 128),
                ),
                (
                    "observation.images.cam_right_wrist",
                    "native_right_wrist",
                    (160, 128),
                    "identity_exact",
                    [],
                    (160, 0, 160, 128),
                ),
            ]
            expected = {
                "profile_id": LINGBOT_ROBOTWIN_PROFILE_ID,
                "source_model": LINGBOT_ROBOTWIN_SOURCE_MODEL,
                "upstream_config": LINGBOT_ROBOTWIN_MODEL_CONFIG,
                "native_input_size_wh": (320, 384),
                "native_output_fps": 10.0,
                "continuation": "robotwin_rgb_and_measured_state_refresh_per_chunk",
                "conditioning": "three_images_text_and_dual_arm_state",
                "action_space": "relative_dual_end_effector16",
                "initial_state_fields": [
                    "left_end_effector_pose7",
                    "left_gripper",
                    "right_end_effector_pose7",
                    "right_gripper",
                ],
                "gripper": "robotwin_normalized_dual_gripper",
                "raw_layout": "left_right_wrist_above_high",
                "formal_view": "native_high",
                "formal_size_wh": (320, 256),
                "extraction": "bottom_320x256_high_panel",
                "rollout_protocol": "simulator_closed_loop",
            }
        else:
            if self.output.formal_view != "exterior_left":
                raise ValueError(
                    "Cosmos formal CAE must use the registered fixed exterior-left plane"
                )
            expected_cameras = [
                (
                    "wrist",
                    "wrist",
                    (640, 360),
                    "identity_exact",
                    [],
                    (0, 0, 640, 360),
                ),
                (
                    "exterior_left",
                    "exterior_left",
                    (320, 180),
                    "identity_exact",
                    ["manipuland", "destination", "end_effector"],
                    (0, 360, 320, 180),
                ),
                (
                    "exterior_right",
                    "exterior_right",
                    (320, 180),
                    "identity_exact",
                    ["manipuland", "destination"],
                    (320, 360, 320, 180),
                ),
            ]
            expected = {
                "profile_id": COSMOS_PROFILE_ID,
                "source_model": COSMOS_SOURCE_MODEL,
                "upstream_config": "DROID_joint_pos",
                "native_input_size_wh": (640, 540),
                "native_output_fps": 15.0,
                "continuation": "simulator_rgb_and_measured_state_refresh_32_steps",
                "conditioning": "images_text_and_initial_robot_state",
                "action_space": "absolute_joint_position7_gripper",
                "initial_state_fields": [
                    "panda_joint1",
                    "panda_joint2",
                    "panda_joint3",
                    "panda_joint4",
                    "panda_joint5",
                    "panda_joint6",
                    "panda_joint7",
                    "gripper_position",
                ],
                "gripper": "droid_raw_0_open_1_closed",
                "raw_layout": "wrist_above_exterior_left_and_exterior_right",
                "formal_view": "exterior_left",
                "formal_size_wh": (320, 180),
                "extraction": "wrist360_then_bottom_left_restore_bottom_crop_320x180",
                "rollout_protocol": "simulator_closed_loop",
            }

        if self.schema_version == "wam_adapter_contract.v2" and self.rollout_protocol is None:
            raise ValueError("WAM adapter contract v2 requires an explicit rollout protocol")
        effective_rollout_protocol = self.rollout_protocol
        if effective_rollout_protocol is None:
            effective_rollout_protocol = (
                "open_loop_autoregressive_video_action"
                if self.wam == LINGBOT_BASE_WAM
                else "simulator_closed_loop"
            )

        actual_cameras = [
            (
                item.input_key,
                item.source_view,
                item.size_wh,
                item.spatial_transform,
                item.required_roles,
                item.layout_region_xywh,
            )
            for item in self.camera_inputs
        ]
        if actual_cameras != expected_cameras:
            raise ValueError(f"{self.wam} camera contract does not match its profile")
        actual: dict[str, Any] = {
            "profile_id": self.profile_id,
            "source_model": self.source_model,
            "upstream_config": self.upstream_config,
            "native_input_size_wh": self.native_input_size_wh,
            "native_output_fps": self.temporal.native_output_fps,
            "continuation": self.temporal.continuation_policy,
            "conditioning": self.state.conditioning,
            "action_space": self.state.action_space,
            "initial_state_fields": self.state.initial_state_fields,
            "gripper": self.state.gripper_convention,
            "raw_layout": self.output.raw_layout,
            "formal_view": self.output.formal_view,
            "formal_size_wh": self.output.formal_size_wh,
            "extraction": self.output.extraction_policy,
            "rollout_protocol": effective_rollout_protocol,
        }
        mismatched = {
            key: {"expected": value, "actual": actual[key]}
            for key, value in expected.items()
            if actual[key] != value
        }
        if mismatched:
            raise ValueError(f"{self.wam} adapter profile mismatch: {mismatched}")
        return self

    @property
    def source_views(self) -> dict[str, str]:
        return {item.input_key: item.source_view for item in self.camera_inputs}

    @property
    def input_sizes(self) -> dict[str, tuple[int, int]]:
        return {item.input_key: item.size_wh for item in self.camera_inputs}

    @property
    def required_roles(self) -> dict[str, list[str]]:
        return {item.input_key: list(item.required_roles) for item in self.camera_inputs}

    @property
    def effective_rollout_protocol(self) -> RolloutProtocol:
        if self.rollout_protocol is not None:
            return self.rollout_protocol
        return (
            "open_loop_autoregressive_video_action"
            if self.wam == LINGBOT_BASE_WAM
            else "simulator_closed_loop"
        )

    def assert_input_binding(
        self,
        *,
        wam: str,
        source_model: str,
        source_views: dict[str, str],
        formal_reference_view: str,
        reference_start_frame_index: int,
        reference_frame_count: int,
        model_input_size_wh: tuple[int, int],
    ) -> None:
        checks = {
            "wam": (wam, self.wam),
            "source_model": (source_model, self.source_model),
            "source_views": (source_views, self.source_views),
            "formal_reference_view": (
                formal_reference_view,
                self.output.formal_view,
            ),
            "reference_start_frame_index": (
                reference_start_frame_index,
                self.temporal.reference_start_frame_index,
            ),
            "reference_frame_count": (
                reference_frame_count,
                self.temporal.reference_frame_count,
            ),
            "model_input_size_wh": (model_input_size_wh, self.native_input_size_wh),
        }
        mismatched = {
            key: {"actual": actual, "expected": expected}
            for key, (actual, expected) in checks.items()
            if actual != expected
        }
        if mismatched:
            raise ValueError(f"WAM input manifest violates adapter contract: {mismatched}")


def _temporal(
    *,
    reference_fps: float,
    reference_frame_count: int,
    native_output_fps: float,
    continuation_policy: str,
) -> FrozenTemporalContract:
    return FrozenTemporalContract(
        reference_fps=reference_fps,
        reference_frame_count=reference_frame_count,
        native_output_fps=native_output_fps,
        continuation_policy=continuation_policy,
    )


def build_lingbot_adapter_contract(
    *,
    case_id: str,
    reference_fps: float,
    reference_frame_count: int,
) -> WAMAdapterContract:
    return WAMAdapterContract(
        case_id=case_id,
        wam=LINGBOT_WAM,
        profile_id=LINGBOT_PROFILE_ID,
        source_model=LINGBOT_SOURCE_MODEL,
        upstream_config=LINGBOT_MODEL_CONFIG,
        native_input_size_wh=(256, 256),
        camera_inputs=[
            CameraInputContract(
                input_key="observation.images.top",
                source_view="top",
                size_wh=(256, 256),
                spatial_transform="identity_exact",
                required_roles=["manipuland", "destination", "end_effector"],
            ),
            CameraInputContract(
                input_key="observation.images.wrist",
                source_view="wrist",
                size_wh=(256, 256),
                spatial_transform="identity_exact",
                required_roles=[],
            ),
        ],
        temporal=_temporal(
            reference_fps=reference_fps,
            reference_frame_count=reference_frame_count,
            native_output_fps=10.0,
            continuation_policy="monolithic_causal_vae_prefix",
        ),
        state=ModelStateContract(
            conditioning="images_and_text",
            action_space="internal_30d_provenance",
            initial_state_fields=[],
            gripper_convention="model_internal",
        ),
        output=OutputExtractionContract(
            raw_layout="top_then_wrist_horizontal",
            formal_view="top",
            formal_size_wh=(256, 256),
            extraction_policy="first_equal_width_panel",
        ),
        rollout_protocol="open_loop_autoregressive_video_action",
    )


def build_cosmos_adapter_contract(
    *,
    case_id: str,
    reference_fps: float,
    reference_frame_count: int,
    formal_view: Literal["exterior_left"] = "exterior_left",
) -> WAMAdapterContract:
    return WAMAdapterContract(
        case_id=case_id,
        wam=COSMOS_WAM,
        profile_id=COSMOS_PROFILE_ID,
        source_model=COSMOS_SOURCE_MODEL,
        upstream_config="DROID_joint_pos",
        native_input_size_wh=(640, 540),
        camera_inputs=[
            CameraInputContract(
                input_key="wrist",
                source_view="wrist",
                size_wh=(640, 360),
                spatial_transform="identity_exact",
                # This is a native conditioning plane, not the formal scored
                # view. A gripper-mounted camera cannot be required to observe
                # the gripper that carries it, and normal closed-loop motion
                # can temporarily move task objects outside this close view.
                # Decode, size, camera binding, and image-quality checks still
                # apply; semantic completeness is enforced on exterior_left.
                required_roles=[],
                layout_region_xywh=(0, 0, 640, 360),
            ),
            CameraInputContract(
                input_key="exterior_left",
                source_view="exterior_left",
                size_wh=(320, 180),
                spatial_transform="identity_exact",
                required_roles=["manipuland", "destination", "end_effector"],
                layout_region_xywh=(0, 360, 320, 180),
            ),
            CameraInputContract(
                input_key="exterior_right",
                source_view="exterior_right",
                size_wh=(320, 180),
                spatial_transform="identity_exact",
                required_roles=["manipuland", "destination"],
                layout_region_xywh=(320, 360, 320, 180),
            ),
        ],
        temporal=_temporal(
            reference_fps=reference_fps,
            reference_frame_count=reference_frame_count,
            native_output_fps=15.0,
            continuation_policy="simulator_rgb_and_measured_state_refresh_32_steps",
        ),
        state=ModelStateContract(
            conditioning="images_text_and_initial_robot_state",
            action_space="absolute_joint_position7_gripper",
            initial_state_fields=[
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
                "gripper_position",
            ],
            gripper_convention="droid_raw_0_open_1_closed",
        ),
        output=OutputExtractionContract(
            raw_layout="wrist_above_exterior_left_and_exterior_right",
            formal_view=formal_view,
            formal_size_wh=(320, 180),
            extraction_policy="wrist360_then_bottom_left_restore_bottom_crop_320x180",
        ),
        rollout_protocol="simulator_closed_loop",
    )


def build_lingbot_robotwin_adapter_contract(
    *,
    case_id: str,
    reference_fps: float,
    reference_frame_count: int,
) -> WAMAdapterContract:
    return WAMAdapterContract(
        case_id=case_id,
        wam=LINGBOT_ROBOTWIN_WAM,
        profile_id=LINGBOT_ROBOTWIN_PROFILE_ID,
        source_model=LINGBOT_ROBOTWIN_SOURCE_MODEL,
        upstream_config=LINGBOT_ROBOTWIN_MODEL_CONFIG,
        native_input_size_wh=(320, 384),
        camera_inputs=[
            CameraInputContract(
                input_key="observation.images.cam_high",
                source_view="native_high",
                size_wh=(320, 256),
                spatial_transform="identity_exact",
                required_roles=["manipuland", "destination", "end_effector"],
                layout_region_xywh=(0, 128, 320, 256),
            ),
            CameraInputContract(
                input_key="observation.images.cam_left_wrist",
                source_view="native_left_wrist",
                size_wh=(160, 128),
                spatial_transform="identity_exact",
                required_roles=[],
                layout_region_xywh=(0, 0, 160, 128),
            ),
            CameraInputContract(
                input_key="observation.images.cam_right_wrist",
                source_view="native_right_wrist",
                size_wh=(160, 128),
                spatial_transform="identity_exact",
                required_roles=[],
                layout_region_xywh=(160, 0, 160, 128),
            ),
        ],
        temporal=_temporal(
            reference_fps=reference_fps,
            reference_frame_count=reference_frame_count,
            native_output_fps=10.0,
            continuation_policy="robotwin_rgb_and_measured_state_refresh_per_chunk",
        ),
        state=ModelStateContract(
            conditioning="three_images_text_and_dual_arm_state",
            action_space="relative_dual_end_effector16",
            initial_state_fields=[
                "left_end_effector_pose7",
                "left_gripper",
                "right_end_effector_pose7",
                "right_gripper",
            ],
            gripper_convention="robotwin_normalized_dual_gripper",
        ),
        output=OutputExtractionContract(
            raw_layout="left_right_wrist_above_high",
            formal_view="native_high",
            formal_size_wh=(320, 256),
            extraction_policy="bottom_320x256_high_panel",
        ),
        rollout_protocol="simulator_closed_loop",
    )


def reference_requirements_for_wam(wam: str) -> WAMReferenceRequirements:
    """Return case-independent requirements used before reference generation."""

    profiles: dict[str, tuple[str, list[tuple[str, tuple[int, int], list[str]]]]] = {
        LINGBOT_BASE_WAM: (
            LINGBOT_PROFILE_ID,
            [
                ("top", (256, 256), ["manipuland", "destination", "end_effector"]),
                ("wrist", (256, 256), []),
            ],
        ),
        LINGBOT_ROBOTWIN_WAM: (
            LINGBOT_ROBOTWIN_PROFILE_ID,
            [
                (
                    "native_high",
                    (320, 256),
                    ["manipuland", "destination", "end_effector"],
                ),
                ("native_left_wrist", (160, 128), []),
                ("native_right_wrist", (160, 128), []),
            ],
        ),
        COSMOS_WAM: (
            COSMOS_PROFILE_ID,
            [
                ("wrist", (640, 360), []),
                (
                    "exterior_left",
                    (320, 180),
                    ["manipuland", "destination", "end_effector"],
                ),
                ("exterior_right", (320, 180), ["manipuland", "destination"]),
            ],
        ),
    }
    if wam not in profiles:
        raise ValueError(f"unsupported WAM reference requirements: {wam!r}")
    profile_id, camera_rows = profiles[wam]
    # Visibility floors only have to make the object legible in the frame the
    # model conditions on; the previous values rejected usable references.
    role_floors = {
        "manipuland": 0.02,
        "destination": 0.02,
        "end_effector": 0.03 if wam == LINGBOT_BASE_WAM else 0.05,
    }
    cameras = [
        ReferenceCameraRequirement(
            source_view=view,  # type: ignore[arg-type]
            size_wh=size,
            required_roles=[
                ReferenceRoleRequirement(
                    role=role,  # type: ignore[arg-type]
                    minimum_linear_fraction=role_floors[role],
                    require_complete_frame=role != "destination",
                )
                for role in roles
            ],
        )
        for view, size, roles in camera_rows
    ]
    return WAMReferenceRequirements(
        wam=wam,  # type: ignore[arg-type]
        profile_id=profile_id,
        cameras=cameras,
        supported_terminal_predicates=[
            "inside",
            "in_container",
            "on_region",
            "stacked",
            "inserted",
            "covering",
            "pressed",
            "door_open",
            "drawer_open",
            "contact",
        ] + (["hung"] if wam in {LINGBOT_ROBOTWIN_WAM, COSMOS_WAM} else []),
        maximum_manipulands=None,
        maximum_terminal_predicates=None,
    )


def load_adapter_contract(
    input_manifest_path: str | Path,
    artifact: str,
    expected_sha256: str,
) -> tuple[Path, WAMAdapterContract]:
    manifest_path = Path(input_manifest_path).resolve()
    raw = Path(artifact)
    path = raw.resolve() if raw.is_absolute() else (manifest_path.parent / raw).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"WAM adapter contract is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"WAM adapter contract digest changed: expected {expected_sha256}, got {actual}"
        )
    return path, WAMAdapterContract.model_validate_json(path.read_text(encoding="utf-8"))


def profile_summary(wam: str) -> dict[str, Any]:
    if wam == LINGBOT_BASE_WAM:
        return {
            "source_model": LINGBOT_SOURCE_MODEL,
            "adapter_profile": LINGBOT_PROFILE_ID,
            "published_embodiment": "LingBot demo single-arm embodiment",
            "camera_mapping": {
                "observation.images.top": "top",
                "observation.images.wrist": "wrist",
            },
            "state_conventions": {
                "conditioning": "images and task text",
                "actions": "internal 30D predictions retained as provenance only",
                "image_layout": "top then wrist latent width; first panel is formal top",
                "native_video_fps": "10",
                "rollout_protocol": "open_loop_autoregressive_video_action",
            },
            "evidence": [
                "lingbot-va/wan_va/configs/va_demo_cfg.py",
                "lingbot-va/wan_va/wan_va_server.py",
            ],
            "limitations": [
                "demo_i2av is an open-loop autoregressive video/action generator; it never receives simulator feedback.",
                "The released demo_i2av example is not documented as native Panda-hand data.",
                "This adapter measures generated-domain generalization, not in-distribution robot control.",
            ],
        }
    if wam == LINGBOT_ROBOTWIN_WAM:
        return {
            "source_model": LINGBOT_ROBOTWIN_SOURCE_MODEL,
            "adapter_profile": LINGBOT_ROBOTWIN_PROFILE_ID,
            "upstream_config": LINGBOT_ROBOTWIN_MODEL_CONFIG,
            "published_embodiment": "RoboTwin dual-arm manipulation profile",
            "camera_mapping": {
                "observation.images.cam_high": "native_high",
                "observation.images.cam_left_wrist": "native_left_wrist",
                "observation.images.cam_right_wrist": "native_right_wrist",
            },
            "state_conventions": {
                "conditioning": "three physical images, task text, and measured dual-arm state",
                "actions": "released relative dual-end-effector action head executed by RoboTwin",
                "image_layout": "two 160x128 auxiliaries above one 320x256 high view",
                "native_video_fps": "10",
                "rollout_protocol": "simulator_closed_loop",
                "termination": "RoboTwin eval_success or native task step_lim",
            },
            "evidence": [
                "lingbot-va/wan_va/configs/va_robotwin_cfg.py",
                "lingbot-va/wan_va/wan_va_server.py::_encode_obs",
                "robbyant/lingbot-va-posttrain-robotwin",
            ],
            "limitations": [
                "This profile is valid only for genuine RoboTwin dual-arm episodes; it cannot consume a single-Panda SceneCode bundle.",
                "The formal SceneCode/RoboTwin adapter renders directly at 320x256 high and 160x128 wrist model planes; resizing is rejected.",
                "The formal WorldOlympiad video is the generated high panel; wrist panels and action heads remain provenance.",
            ],
        }
    if wam == COSMOS_WAM:
        return {
            "source_model": COSMOS_SOURCE_MODEL,
            "adapter_profile": COSMOS_PROFILE_ID,
            "published_embodiment": "Franka arm with Robotiq 2F-85 DROID end effector",
            "camera_mapping": {
                "wrist": "wrist",
                "exterior_left": "exterior_left",
                "exterior_right": "exterior_right",
            },
            "state_conventions": {
                "joint_position": "seven Franka arm radians",
                "gripper_position": "DROID raw 0=open, 1=closed",
                "actions": "absolute joint_pos predictions retained as provenance only",
                "image_layout": "640x360 wrist above two 320x180 exterior views",
                "native_video_fps": "15",
                "formal_video": (
                    "native fixed exterior-left plane, padded to 320x180 for calibrated "
                    "Dream2Flow CAE; wrist remains a diagnostic plane"
                ),
                "continuation": "fresh simulator RGB and measured state every 32 steps",
                "rollout_protocol": "simulator_closed_loop",
            },
            "evidence": [
                "cosmos-framework/cosmos_predict2/_src/predict2/action/droid/server.py",
                "RoboLab DROID Franka+Robotiq wrist-camera USD",
            ],
            "limitations": [
                "Closed loop refreshes at the released 32-step chunk horizon, not every video frame.",
                "The moving wrist plane has no generated per-frame camera extrinsics and is not used for formal 3-D CAE.",
                "The frame-composed wrist camera can still self-occlude during poor policy motion.",
                "Within-window imagination may blur or drift before the next real observation refresh.",
            ],
        }
    raise ValueError(f"unsupported WAM adapter profile: {wam!r}")
