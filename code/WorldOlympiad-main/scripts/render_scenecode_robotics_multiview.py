#!/usr/bin/env python3
"""Replay one PyBullet trajectory through four official SceneCode cameras."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    ApplyCameraConfig,
    CameraConfig,
    DiagramBuilder,
    Parser,
    Quaternion,
    RenderEngineGltfClientParams,
    RenderEngineVtkParams,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Transform,
)

from scenecode.agent_utils.blender.server_manager import BlenderServer
from autoresearch.robotics_scene_objects import (
    articulated_component_links,
    carry_evidence_frame_ids,
    load_scene_object_sources,
)
from autoresearch.robotics_native_cameras import ROBOLAB_DROID_FIXED_CAMERAS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


REFERENCE_FIXED_CAMERAS = {
    # ``top`` must be a real high-angle workspace view for demo_i2av, not a
    # renamed front exterior camera.  The slight -Y offset avoids a singular
    # look-at basis while preserving a near-overhead view.
    "top": ((0.72, -0.72, 1.62), (0.0, 0.01, 0.86)),
    # The Panda fingers close along world Y in this registered recipe.  Side
    # cameras therefore look mostly along +/-X so both contact faces remain
    # visible instead of hiding the far finger behind the cup.
    "exterior_left": ((-1.08, -0.10, 1.42), (0.0, 0.01, 0.88)),
    # Screenshot-matched oblique front-right view: cabinet front stays in
    # frame while the DROID base at +X remains visible on its right.
    "exterior_right": ((1.45, -1.65, 1.18), (0.0, 0.0, 0.62)),
}

# RoboLab commit f72d39445ab699c9cfd9bf551e8d21b6075293b9 publishes the DROID
# over-shoulder camera offsets in the robot-root frame.  ``_robolab_fixed_camera_pose``
# converts them to Drake's +Z-forward/+Y-down camera axes and composes the
# registered Panda base pose; the materializer's framing derivative reads the
# same definition so both sides of the reference pipeline agree.

# Dream2Flow's released trajectory optimizer uses the standard Panda
# description (seven arm joints plus one actuated finger mimic joint). This
# fixed planning seed is copied into every CAE camera scene so reference and
# imagination videos are converted to EE action trajectories symmetrically.
DREAM2FLOW_PANDA_START_JOINTS = [
    0.0,
    -math.pi / 5.0,
    0.0,
    -4.0 * math.pi / 5.0,
    0.0,
    3.0 * math.pi / 5.0,
    0.0,
    0.0,
]


def _parse_aspect_ratio(value: str) -> float:
    """Parse WIDTH:HEIGHT and return a positive width / height ratio."""

    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*",
        value,
    )
    if match is None:
        raise argparse.ArgumentTypeError(
            "aspect ratio must use WIDTH:HEIGHT, for example 16:9"
        )
    width, height = (float(part) for part in match.groups())
    if width <= 0.0 or height <= 0.0:
        raise argparse.ArgumentTypeError("aspect ratio dimensions must be positive")
    return width / height


def _preserve_horizontal_fov_y(
    source_fov_y_deg: float,
    source_aspect: float,
    output_aspect: float,
) -> float:
    """Return output vertical FOV while retaining the source horizontal FOV."""

    horizontal_tangent = source_aspect * math.tan(
        math.radians(source_fov_y_deg) / 2.0
    )
    return math.degrees(2.0 * math.atan(horizontal_tangent / output_aspect))


def _carry_evidence_frame_ids(frames: list[dict], max_samples: int = 12) -> set[int]:
    """Choose deterministic grasp-carry frames for the visual contact gate."""

    return carry_evidence_frame_ids(frames, max_samples=max_samples)


def _dual_contact_evidence_frame_ids(
    frames: list[dict], max_samples: int = 12
) -> set[int]:
    """Choose policy-rollout frames with simultaneous two-finger contact.

    Closed-loop policy rollouts do not have the hand-authored phase labels used
    by the expert reference trajectory.  Contact evidence is nevertheless
    available directly from PyBullet, so diagnostics can request semantic
    masks without pretending that a failed rollout entered a carry phase.
    """

    contact = [
        int(frame["frame"])
        for frame in frames
        if len(set(frame.get("robot_cup_contact_links") or [])) >= 2
    ]
    if not contact:
        return set()
    count = min(max_samples, len(contact))
    indices = np.linspace(0, len(contact) - 1, num=count, dtype=np.int64)
    return {contact[int(index)] for index in indices}


def _push_contact_evidence_frame_ids(
    frames: list[dict], max_samples: int = 12
) -> set[int]:
    """Choose deterministic push frames with registered robot-object contact."""

    contact = [
        int(frame["frame"])
        for frame in frames
        if frame.get("phase") == "push"
        and bool(frame.get("robot_cup_contact_links"))
    ]
    if not contact:
        raise ValueError("trajectory has no push frames with robot-object contact")
    count = min(max_samples, len(contact))
    indices = np.linspace(0, len(contact) - 1, num=count, dtype=np.int64)
    return {contact[int(index)] for index in indices}


def _pose(position: list[float], quaternion_xyzw: list[float]) -> RigidTransform:
    x, y, z, w = quaternion_xyzw
    return RigidTransform(Quaternion(w, x, y, z), position)


def _look_at(eye: tuple[float, float, float], target: tuple[float, float, float]) -> RigidTransform:
    eye_v = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye_v
    forward /= np.linalg.norm(forward)
    up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = RotationMatrix(np.column_stack((right, down, forward)))
    return RigidTransform(rotation, eye_v)


def _robolab_fixed_camera_pose(
    base_position: list[float],
    position: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float],
    base_orientation_xyzw: list[float] | None = None,
) -> RigidTransform:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    quaternion_norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion_norm) or quaternion_norm < 1e-9:
        raise ValueError("RoboLab fixed-camera quaternion is invalid")
    quaternion /= quaternion_norm
    world_from_robot = (
        _pose(base_position, base_orientation_xyzw)
        if base_orientation_xyzw is not None
        else RigidTransform(RotationMatrix.MakeZRotation(math.pi / 2.0), base_position)
    )
    robot_from_opengl_camera = RigidTransform(
        Quaternion(*quaternion), position
    )
    opengl_from_drake_camera = RigidTransform(
        RotationMatrix(np.diag([1.0, -1.0, -1.0]))
    )
    return world_from_robot.multiply(robot_from_opengl_camera).multiply(
        opengl_from_drake_camera
    )


# ``REFERENCE_FIXED_CAMERAS`` is authored in the reference-contact world, where the
# workspace sits at a fixed world height.  The registered DROID recipe instead welds
# the workspace to the Panda base, which sits on the raised DROID platform
# (base z ~= 1.40 m); a world-fixed ``top`` therefore looked ~0.7 m below the
# manipuland and lost the whole robot, producing a static "top" reference video.
# These offsets keep the authored high-angle geometry but express it in the same
# robot-root frame the exterior cameras use, so the view follows the workcell.
# They are the authored pose re-expressed in that frame: with the base at the
# authored (0, -0.43, 0.002) the two definitions coincide.
ROBOLAB_DROID_TOP_CAMERA_M = {
    "eye": (1.16, -0.72, 1.12),
    "target": (0.44, 0.0, 0.17),
}


def _robolab_robot_frame_look_at(
    base_position: list[float],
    base_orientation_xyzw: list[float] | None,
    eye_local: tuple[float, float, float],
    target_local: tuple[float, float, float],
) -> RigidTransform:
    """Aim a camera authored in the robot root frame at a point in the same frame."""

    world_from_robot = (
        _pose(base_position, base_orientation_xyzw)
        if base_orientation_xyzw is not None
        else RigidTransform(RotationMatrix.MakeZRotation(math.pi / 2.0), base_position)
    )
    rotation = world_from_robot.rotation()
    origin = np.asarray(world_from_robot.translation(), dtype=np.float64)

    def world(local):
        return origin + np.asarray(rotation.multiply(np.asarray(local, dtype=np.float64)))

    return _look_at(tuple(world(eye_local)), tuple(world(target_local)))


def _fixed_camera_poses(
    preset: str, base_position: list[float], base_orientation_xyzw: list[float] | None = None
) -> dict[str, RigidTransform]:
    poses = {
        "top": _robolab_robot_frame_look_at(
            base_position,
            base_orientation_xyzw,
            ROBOLAB_DROID_TOP_CAMERA_M["eye"],
            ROBOLAB_DROID_TOP_CAMERA_M["target"],
        )
        if preset == "robolab_droid"
        else _look_at(*REFERENCE_FIXED_CAMERAS["top"]),
    }
    if preset == "reference_contact":
        poses.update(
            {
                name: _look_at(*REFERENCE_FIXED_CAMERAS[name])
                for name in ("exterior_left", "exterior_right")
            }
        )
    elif preset == "robolab_droid":
        poses.update(
            {
                name: _robolab_fixed_camera_pose(
                    base_position, position, quaternion, base_orientation_xyzw
                )
                for name, (position, quaternion) in ROBOLAB_DROID_FIXED_CAMERAS.items()
            }
        )
    else:
        raise ValueError(f"unknown fixed camera preset: {preset!r}")
    return poses


def _camera_transform(pose: RigidTransform, base_frame: str | None = None) -> Transform:
    transform = Transform(pose)
    transform.base_frame = base_frame
    return transform


def _robolab_droid_wrist_extrinsic() -> RigidTransform:
    """RoboLab DROID wrist-camera mount expressed in Drake/OpenCV axes.

    RoboLab publishes the mount on the Robotiq base link as position
    ``(0.011, -0.031, -0.074)`` and OpenGL quaternion (wxyz)
    ``(-0.420, 0.570, 0.576, -0.409)``.  Drake's RGB-D camera uses +Z forward
    and +Y down.  The values below additionally apply RoboLab's published
    ``panda_link8 -> Robotiq base_link`` fixed joint and invert our URDF's
    ``panda_link8 -> panda_hand`` joint.  They are therefore the audited
    RoboLab mount expressed in our ``panda_hand`` frame, not a hand-tuned pose.
    """

    rotation = RotationMatrix(
        np.asarray(
            [
                [0.7194223862, 0.6601828980, 0.2158471020],
                [-0.6945692212, 0.6827953722, 0.2266364413],
                [0.0022421003, -0.3129680829, 0.9497609973],
            ]
        )
    )
    return RigidTransform(rotation, (-0.0742462104, -0.0304056063, 0.0291740107))


def _robolab_droid_base_wrist_extrinsic() -> RigidTransform:
    """RoboLab wrist camera expressed in PickNik's Robotiq base frame.

    The pinned RoboLab USD ``wrist_camera_flipped`` prim is authored in the
    RoboLab base frame at translation ``(0.011, -0.031, -0.074)`` and OpenGL
    quaternion (wxyz) ``(-0.4006971915, 0.5812758123, 0.5733257568,
    -0.4157616714)``.  RoboLab's base axes differ from the pinned PickNik URDF
    base axes even though their physical link origins coincide.  The values
    below compose both the audited RoboLab-to-PickNik base-axis registration
    and the OpenGL-to-Drake/OpenCV camera-axis rotation.  This is not a tuned
    look-at pose.
    """

    rotation = RotationMatrix(
        np.asarray(
            [
                [0.9997098581, 0.0214786747, 0.0109025789],
                [-0.0238843654, 0.9425649207, 0.3331679867],
                [-0.0031203816, -0.3333317219, 0.9428044476],
            ]
        )
    )
    return RigidTransform(rotation, (-0.031, -0.074, 0.011))


def _wrist_extrinsic(preset: str) -> RigidTransform:
    if preset == "robolab_droid":
        return _robolab_droid_wrist_extrinsic()
    raise ValueError(f"unknown wrist camera preset: {preset!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--camera-rig", type=Path)
    parser.add_argument("--table-sdf", type=Path)
    parser.add_argument("--cup-sdf", type=Path)
    parser.add_argument("--tray-sdf", type=Path)
    parser.add_argument("--scene-objects-manifest", type=Path)
    parser.add_argument("--articulation-binding", type=Path)
    parser.add_argument("--panda-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--blender-settings-file",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "deployment"
            / "ziplab5090"
            / "scenecode_robotics_render_settings.py"
        ),
        help="Audited SceneCode Blender settings applied to every trajectory frame.",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fov-y-deg", type=float, default=52.0)
    parser.add_argument(
        "--preserve-horizontal-fov-from-aspect",
        type=_parse_aspect_ratio,
        default=None,
        metavar="WIDTH:HEIGHT",
        help=(
            "Interpret --fov-y-deg and --wrist-fov-y-deg in this source aspect, "
            "then adjust the output vertical FOV to retain horizontal coverage. "
            "Use 16:9 when natively rendering a square model view from a 16:9 reference."
        ),
    )
    parser.add_argument(
        "--wrist-fov-y-deg",
        type=float,
        default=None,
        help="Optional registered wrist-only vertical FOV; fixed cameras retain --fov-y-deg.",
    )
    parser.add_argument(
        "--wrist-camera-preset",
        choices=("robolab_droid",),
        default="robolab_droid",
    )
    parser.add_argument(
        "--robot-embodiment",
        choices=("panda_stock", "robolab_droid"),
        default="panda_stock",
        help="Select robot link names and the audited wrist-camera parent frame.",
    )
    parser.add_argument(
        "--fixed-camera-preset",
        choices=("reference_contact", "robolab_droid"),
        default="reference_contact",
    )
    parser.add_argument(
        "--wrist-optical-yaw-deg",
        type=float,
        default=0.0,
        help="Audited diagnostic rotation about the wrist camera's image-down axis.",
    )
    parser.add_argument(
        "--wrist-optical-pitch-deg",
        type=float,
        default=0.0,
        help="Audited diagnostic rotation about the wrist camera's image-right axis.",
    )
    parser.add_argument(
        "--wrist-optical-forward-offset-m",
        type=float,
        default=0.0,
        help="Audited diagnostic translation along the wrist camera optical axis.",
    )
    parser.add_argument("--port", type=int, default=8166)
    parser.add_argument("--frames", default="all", help="all or comma-separated frame ids")
    parser.add_argument(
        "--semantic-evidence-mode",
        choices=("carry_phases", "dual_contact", "push_contact", "articulation_contact", "none"),
        default="carry_phases",
        help=(
            "Select semantic-mask evidence from expert carry phases, from "
            "PyBullet dual-contact or push-contact frames, or disable it for "
            "conditioning-only renders."
        ),
    )
    parser.add_argument(
        "--views",
        default="all",
        help="all or a comma-separated subset of top,wrist,exterior_left,exterior_right",
    )
    parser.add_argument("--resume", action="store_true", help="reuse complete frame PNGs from an interrupted attempt")
    parser.add_argument(
        "--instruction",
        default="Pick up the blue cup, place it inside the shallow blue tray, and release it.",
    )
    parser.add_argument("--object-name", default="blue ceramic cup")
    parser.add_argument("--canonical-object-name", default="blue_cup")
    parser.add_argument("--semantic-subject-object-id")
    parser.add_argument("--semantic-destination-object-id")
    parser.add_argument("--semantic-support-object-id")
    parser.add_argument("--manifest-name", default="render_manifest.json")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("render width and height must be positive")
    if not re.fullmatch(r"render_manifest(?:\.[A-Za-z0-9_-]+)?\.json", args.manifest_name):
        raise ValueError("unsafe render manifest filename")
    if args.scene_objects_manifest is None:
        if args.cup_sdf is None or args.tray_sdf is None or args.table_sdf is None:
            raise ValueError(
                "legacy render requires --table-sdf, --cup-sdf, and --tray-sdf"
            )
        scene_object_sources = None
    else:
        if args.cup_sdf is not None or args.tray_sdf is not None:
            raise ValueError(
                "--scene-objects-manifest is mutually exclusive with legacy cup/tray inputs"
            )
        scene_object_sources = load_scene_object_sources(
            args.scene_objects_manifest.resolve(),
            minimum_objects=1 if args.articulation_binding is not None else 2,
        )
    if args.articulation_binding is not None and scene_object_sources is None:
        raise ValueError("articulation binding requires --scene-objects-manifest")

    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    frames = trajectory.get("frames") or []
    if not frames:
        raise ValueError("trajectory is empty")
    articulation_binding: dict[str, object] | None = None
    if args.articulation_binding is not None:
        articulation_binding = json.loads(
            args.articulation_binding.resolve().read_text(encoding="utf-8")
        )
        if articulation_binding.get("passed") is not True:
            raise ValueError("articulation binding is not passing")
    if args.frames != "all":
        requested = {int(value) for value in args.frames.split(",")}
        frames = [frame for frame in frames if int(frame["frame"]) in requested]
        if len(frames) != len(requested):
            raise ValueError("one or more requested frame ids are absent from the trajectory")
    if args.semantic_evidence_mode == "carry_phases":
        evidence_frame_ids = _carry_evidence_frame_ids(frames)
    elif args.semantic_evidence_mode == "dual_contact":
        evidence_frame_ids = _dual_contact_evidence_frame_ids(frames)
    elif args.semantic_evidence_mode == "push_contact":
        evidence_frame_ids = _push_contact_evidence_frame_ids(frames)
    elif args.semantic_evidence_mode == "articulation_contact":
        evidence_frame_ids = {
            int(frame["frame"])
            for frame in frames
            if (frame.get("articulated") or {}).get("contact_evidence")
            or frame.get("robot_articulation_contact_evidence")
        }
    else:
        evidence_frame_ids = set()
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite multiview render: {output}")
    if args.resume and (output / "render_manifest.json").exists():
        raise FileExistsError("render is already complete; refuse to resume a published manifest")
    output.mkdir(parents=True, exist_ok=args.resume)
    output_aspect = args.width / args.height

    def view_fov_y_deg(name: str) -> float:
        source_fov = (
            args.wrist_fov_y_deg
            if name == "wrist" and args.wrist_fov_y_deg is not None
            else args.fov_y_deg
        )
        if args.preserve_horizontal_fov_from_aspect is None:
            return source_fov
        return _preserve_horizontal_fov_y(
            source_fov,
            args.preserve_horizontal_fov_from_aspect,
            output_aspect,
        )
    log_suffix = Path(args.manifest_name).stem.replace("render_manifest", "").strip(".")
    log_name = f"blender_server_{log_suffix}.log" if log_suffix else "blender_server.log"
    settings_file = args.blender_settings_file.resolve()
    if not settings_file.is_file():
        raise FileNotFoundError(settings_file)
    server = BlenderServer(
        port=args.port,
        log_file=output / log_name,
        gpu_id=0,
        bpy_settings_file=settings_file,
    )

    try:
        server.start()
        server.wait_until_ready(timeout=90.0)
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        asset_parser = Parser(plant)
        asset_parser.SetAutoRenaming(True)
        panda_mesh_package = args.panda_urdf.resolve().parent / "meshes"
        if panda_mesh_package.is_dir():
            asset_parser.package_map().Add("meshes", str(panda_mesh_package))
        table_model = None
        if args.table_sdf is not None:
            table_model = asset_parser.AddModels(str(args.table_sdf.resolve()))[0]
        if scene_object_sources is None:
            cup_model = asset_parser.AddModels(str(args.cup_sdf.resolve()))[0]
            tray_model = asset_parser.AddModels(str(args.tray_sdf.resolve()))[0]
            scene_object_models = {"cup": cup_model, "tray": tray_model}
            conditioning_object_sources = {
                args.semantic_subject_object_id or "cup": "cup",
                args.semantic_destination_object_id or "tray": "tray",
            }
            # 2026-09-11 household repair: the in-context visual review needs the
            # support surface the manipuland sits on.  Registering it here makes
            # the frame-zero conditioning mask carry the tabletop, so rendered
            # RGB separation can be measured against it, not just against the
            # destination.
            if args.semantic_support_object_id and table_model is not None:
                scene_object_models["support"] = table_model
                conditioning_object_sources[
                    args.semantic_support_object_id
                ] = "support"
            if len(conditioning_object_sources) not in {2, 3}:
                raise ValueError(
                    "conditioning subject, destination and optional support ids must differ"
                )
        else:
            scene_object_models = {
                object_id: asset_parser.AddModels(str(source))[0]
                for object_id, source in scene_object_sources.items()
            }
            conditioning_object_sources = {
                object_id: object_id for object_id in scene_object_models
            }
        if articulation_binding is not None:
            host_object_id = str(articulation_binding["host_object_id"])
            if host_object_id not in scene_object_models:
                raise ValueError("articulation binding host is absent from scene objects")
            host_sdf = Path(scene_object_sources[host_object_id])
            if str(articulation_binding.get("host_sdf_sha256")) != _sha256(host_sdf):
                raise ValueError("articulation binding host digest differs from renderer input")
        panda_model = asset_parser.AddModels(str(args.panda_urdf.resolve()))[0]
        table_body = (
            plant.get_body(plant.GetBodyIndices(table_model)[0])
            if table_model is not None
            else None
        )
        scene_object_bodies = {
            object_id: plant.get_body(plant.GetBodyIndices(model)[0])
            for object_id, model in scene_object_models.items()
            # The support surface is welded to the world and has no per-frame
            # trajectory state; it is registered for labelling only.
            if object_id != "support"
        }
        # A SceneCode model can be multi-link (an empty root link plus rigid
        # children, optionally an articulated child).  The label image carries
        # one render label per rendered body, so every body of the model must map
        # to its task object; keying only on the root body silently drops the
        # manipuland from the deterministic semantic mask.
        scene_object_model_bodies = {
            object_id: [
                plant.get_body(index) for index in plant.GetBodyIndices(model)
            ]
            for object_id, model in scene_object_models.items()
        }
        panda_base_body = plant.GetBodyByName("panda_link0", panda_model)
        panda_base = frames[0]["robot"]["base_position"]
        panda_orientation = frames[0]["robot"].get("base_orientation_xyzw", [0., 0., math.sin(math.pi/4), math.cos(math.pi/4)])
        if table_body is not None:
            plant.WeldFrames(
                plant.world_frame(), table_body.body_frame(), RigidTransform.Identity()
            )
        plant.WeldFrames(
            plant.world_frame(),
            panda_base_body.body_frame(),
            _pose(panda_base, panda_orientation),
        )
        if trajectory.get("prepared_assets", {}).get("ground_plane"):
            plant.RegisterVisualGeometry(plant.world_body(), RigidTransform([0,0,-.025]),
                Box(8.,8.,.05), "reference_ground", np.array([.65,.65,.65,1.]))
        mount = trajectory.get("robot_mount")
        if mount and float(mount.get("height_m",0))>0:
            height=float(mount["height_m"])
            plant.RegisterVisualGeometry(plant.world_body(),
                RigidTransform([panda_base[0],panda_base[1],height/2]),
                Box(*mount["size_xy_m"],height), "reference_robot_mount", np.array([.28,.3,.32,1.]))
        plant.Finalize()
        scene_object_render_labels = {
            object_id: np.asarray(
                [int(body.index()) for body in bodies], dtype=np.int32
            )
            for object_id, bodies in scene_object_model_bodies.items()
        }
        articulation_joint = None
        articulation_child_body = None
        articulation_component_id = None
        if articulation_binding is not None:
            host_object_id = str(articulation_binding["host_object_id"])
            articulation_component_id = str(articulation_binding["component_id"])
            host_model = scene_object_models[host_object_id]
            articulation_joint = plant.GetJointByName(
                str(articulation_binding["joint_name"]), host_model
            )
            articulation_child_body = plant.GetBodyByName(
                str(articulation_binding["child_link"]), host_model
            )
            expected_joint_type = str(articulation_binding["joint_type"])
            if (
                expected_joint_type == "prismatic"
                and articulation_joint.num_positions() != 1
            ) or (
                expected_joint_type == "revolute"
                and articulation_joint.num_positions() != 1
            ):
                raise ValueError("articulation binding joint is not single-DOF in Drake")
            component_links = articulated_component_links(
                host_sdf, articulation_child_body.name())
            component_labels = np.asarray([
                int(plant.GetBodyByName(name, host_model).index())
                for name in component_links
            ], dtype=np.int32)
            host_labels = scene_object_render_labels[host_object_id]
            scene_object_render_labels[host_object_id] = host_labels[
                ~np.isin(host_labels, component_labels)]
            if not len(scene_object_render_labels[host_object_id]):
                raise ValueError("articulated host has no static visual labels")
            scene_object_render_labels[articulation_component_id] = component_labels
            conditioning_object_sources = {
                articulation_component_id: articulation_component_id,
                host_object_id: host_object_id,
            }
        if args.robot_embodiment == "robolab_droid":
            wrist_body = plant.GetBodyByName("base_link", panda_model)
            left_finger_body = plant.GetBodyByName("left_inner_finger", panda_model)
            right_finger_body = plant.GetBodyByName("right_inner_finger", panda_model)
            left_finger_semantic_bodies = [
                plant.GetBodyByName(name, panda_model)
                for name in (
                    "left_outer_knuckle",
                    "left_outer_finger",
                    "left_inner_knuckle",
                    "left_inner_finger",
                )
            ]
            right_finger_semantic_bodies = [
                plant.GetBodyByName(name, panda_model)
                for name in (
                    "right_outer_knuckle",
                    "right_outer_finger",
                    "right_inner_knuckle",
                    "right_inner_finger",
                )
            ]
            wrist_extrinsic = _robolab_droid_base_wrist_extrinsic()
        else:
            wrist_body = plant.GetBodyByName("panda_hand", panda_model)
            left_finger_body = plant.GetBodyByName("panda_leftfinger", panda_model)
            right_finger_body = plant.GetBodyByName("panda_rightfinger", panda_model)
            left_finger_semantic_bodies = [left_finger_body]
            right_finger_semantic_bodies = [right_finger_body]
            wrist_extrinsic = _wrist_extrinsic(args.wrist_camera_preset)
        left_finger_render_labels = np.asarray(
            [int(body.index()) for body in left_finger_semantic_bodies],
            dtype=np.int32,
        )
        right_finger_render_labels = np.asarray(
            [int(body.index()) for body in right_finger_semantic_bodies],
            dtype=np.int32,
        )

        rigid_camera_poses = _fixed_camera_poses(
            args.fixed_camera_preset, panda_base, panda_orientation
        )
        if args.camera_rig is not None:
            camera_rig = json.loads(args.camera_rig.read_text())
            if camera_rig["schema_version"] != "worldolympiad_camera_rig.v1":
                raise ValueError("unknown camera rig schema")
            for name, view in camera_rig["cameras"].items():
                if name not in ("top", "exterior_left", "exterior_right"):
                    raise ValueError("camera override must name a fixed view")
                if (args.views == "all" or name in args.views.split(",")) and (view["width"], view["height"]) != (args.width, args.height):
                    raise ValueError("camera rig/native render size mismatch")
                if (args.views == "all" or name in args.views.split(",")) and abs(view["fov_y_deg"] - args.fov_y_deg)>1e-8:
                    raise ValueError("camera rig FOV mismatch")
                rigid_camera_poses[name] = RigidTransform(
                    Quaternion(*view["quaternion_opengl_wxyz"]), view["position_world"]
                ).multiply(RigidTransform(RotationMatrix(np.diag([1.,-1.,-1.]))))
        else:
            camera_rig = None
        camera_poses = {
            name: _camera_transform(pose) for name, pose in rigid_camera_poses.items()
        }
        wrist_extrinsic = wrist_extrinsic.multiply(
            RigidTransform(
                RotationMatrix.MakeYRotation(
                    math.radians(args.wrist_optical_yaw_deg)
                )
            )
        ).multiply(
            RigidTransform(
                RotationMatrix.MakeXRotation(
                    math.radians(args.wrist_optical_pitch_deg)
                )
            )
        ).multiply(
            RigidTransform((0.0, 0.0, args.wrist_optical_forward_offset_m))
        )
        wrist_mount = (camera_rig or {}).get("wrist_mount") or {}
        mount_translation = wrist_mount.get("translation_local_xyz")
        if mount_translation is not None:
            values = np.asarray(mount_translation, dtype=float)
            if (values.shape != (3,) or not np.isfinite(values).all()
                    or np.linalg.norm(values) > 0.20):
                raise ValueError("wrist camera translation exceeds its bounded mount")
            wrist_extrinsic = RigidTransform(
                wrist_extrinsic.rotation(), wrist_extrinsic.translation() + values)
        optical_delta = wrist_mount.get("optical_delta_opengl_wxyz")
        if optical_delta is not None:
            values = np.asarray(optical_delta, dtype=float)
            if (values.shape != (4,) or not np.isfinite(values).all()
                    or abs(np.linalg.norm(values) - 1.0) > 1e-5):
                raise ValueError("wrist optical delta must be a unit WXYZ quaternion")
            basis = np.diag([1.0, -1.0, -1.0])
            delta_opengl = RotationMatrix(Quaternion(*values)).matrix()
            wrist_extrinsic = wrist_extrinsic.multiply(RigidTransform(
                RotationMatrix(basis @ delta_opengl @ basis)))
        wrist_camera_parent_frame = (
            f"{plant.GetModelInstanceName(panda_model)}::{wrist_body.name()}"
        )
        camera_poses["wrist"] = _camera_transform(
            wrist_extrinsic, wrist_camera_parent_frame
        )
        if args.views != "all":
            requested_views = [value.strip() for value in args.views.split(",") if value.strip()]
            unknown = set(requested_views) - set(camera_poses)
            if unknown or len(requested_views) != len(set(requested_views)):
                raise ValueError(f"invalid or duplicate camera views: {sorted(unknown)}")
            camera_poses = {name: camera_poses[name] for name in requested_views}
        for name, transform in camera_poses.items():
            effective_fov_y_deg = view_fov_y_deg(name)
            camera = CameraConfig(
                name=name,
                X_PB=transform,
                width=args.width,
                height=args.height,
                focal=CameraConfig.FovDegrees(y=effective_fov_y_deg),
                background=Rgba(0.055, 0.055, 0.055, 1.0),
                renderer_name=f"scenecode_blender_{name}",
                renderer_class=RenderEngineGltfClientParams(base_url=server.get_url()),
            )
            ApplyCameraConfig(
                config=camera,
                builder=builder,
                plant=plant,
                scene_graph=scene_graph,
            )
            builder.ExportOutput(
                builder.GetSubsystemByName(f"rgbd_sensor_{name}").color_image_output_port(),
                f"rgba_{name}",
            )
            depth_camera = CameraConfig(
                name=f"{name}_cae_depth",
                X_PB=transform,
                width=args.width,
                height=args.height,
                focal=CameraConfig.FovDegrees(y=effective_fov_y_deg),
                background=Rgba(0.055, 0.055, 0.055, 1.0),
                renderer_name=f"drake_vtk_depth_{name}",
                renderer_class=RenderEngineVtkParams(),
            )
            ApplyCameraConfig(
                config=depth_camera,
                builder=builder,
                plant=plant,
                scene_graph=scene_graph,
            )
            builder.ExportOutput(
                builder.GetSubsystemByName(
                    f"rgbd_sensor_{name}_cae_depth"
                ).depth_image_32F_output_port(),
                f"depth_{name}",
            )
            builder.ExportOutput(
                builder.GetSubsystemByName(
                    f"rgbd_sensor_{name}_cae_depth"
                ).label_image_output_port(),
                f"label_{name}",
            )
            (output / name / "frames").mkdir(parents=True, exist_ok=args.resume)
            (output / name / "semantic_masks").mkdir(
                parents=True, exist_ok=args.resume
            )
            (output / name / "conditioning_semantic").mkdir(
                parents=True, exist_ok=args.resume
            )

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyMutableContextFromRoot(context)
        drake_names = [
            name.removesuffix("_q").removesuffix("_x")
            for name in plant.GetPositionNames(panda_model, add_model_instance_prefix=False)
        ]
        records = {name: [] for name in camera_poses}
        semantic_records = {name: [] for name in camera_poses}
        conditioning_semantic_records: dict[str, dict] = {}
        conditioning_object_values = {
            object_id: index + 1
            for index, object_id in enumerate(sorted(conditioning_object_sources))
        }
        conditioning_end_effector_value = len(conditioning_object_values) + 1
        robot_registration_records: list[dict] = []
        for frame in frames:
            robot = frame["robot"]
            robot_names = list(robot["joint_names"])
            robot_positions = list(robot["joint_positions"])
            if len(robot_names) != len(robot_positions) or len(set(robot_names)) != len(
                robot_names
            ):
                raise RuntimeError("robot joint state has invalid names or dimensions")
            if set(drake_names) != set(robot_names):
                raise RuntimeError(
                    "robot joint names differ between PyBullet and Drake: "
                    f"{sorted(set(robot_names) ^ set(drake_names))}"
                )
            position_by_name = dict(zip(robot_names, robot_positions, strict=True))
            plant.SetPositions(
                plant_context,
                panda_model,
                [position_by_name[name] for name in drake_names],
            )
            if articulation_joint is not None:
                articulated_state = frame.get("articulated") or {}
                if articulated_state.get("joint_name") != articulation_joint.name():
                    raise ValueError("trajectory articulation joint differs from renderer binding")
                joint_position = float(articulated_state.get("position"))
                if not math.isfinite(joint_position):
                    raise ValueError("trajectory articulation position is not finite")
                if str(articulation_binding["joint_type"]) == "prismatic":
                    articulation_joint.set_translation(plant_context, joint_position)
                else:
                    articulation_joint.set_angle(plant_context, joint_position)
            if scene_object_sources is None:
                frame_objects = {"cup": frame["cup"], "tray": frame["tray"]}
                active_subject_id = "cup"
            else:
                frame_objects = frame.get("objects") or {}
                if set(frame_objects) != set(scene_object_bodies):
                    raise ValueError(
                        "trajectory scene-object state differs from its object table"
                    )
                if articulation_binding is not None:
                    active_subject_id = str(articulation_binding["component_id"])
                else:
                    active_subject_id = frame.get("active_subject_id")
                    if active_subject_id not in scene_object_bodies:
                        raise ValueError("trajectory active subject is absent from the object table")
            for object_id, body in scene_object_bodies.items():
                state = frame_objects[object_id]
                plant.SetFreeBodyPose(
                    plant_context,
                    body,
                    _pose(state["link_position"], state["link_orientation_xyzw"]),
                )
            expected_fingers = frame["robot"].get("finger_link_positions") or {}
            finger_residuals = {}
            for body in (left_finger_body, right_finger_body):
                expected = np.asarray(expected_fingers.get(body.name()), dtype=np.float64)
                if expected.shape != (3,):
                    raise ValueError(
                        f"trajectory is missing PyBullet link pose for {body.name()}"
                    )
                rendered = plant.EvalBodyPoseInWorld(
                    plant_context, body
                ).translation()
                finger_residuals[body.name()] = {
                    "pybullet_world_position": expected.tolist(),
                    "drake_world_position": rendered.tolist(),
                    "translation_residual_m": float(np.linalg.norm(rendered - expected)),
                }
            robot_registration_records.append(
                {
                    "frame": int(frame["frame"]),
                    "fingers": finger_residuals,
                    **(
                        {
                            "articulation": {
                                "joint_name": articulation_joint.name(),
                                "child_link": articulation_child_body.name(),
                                "pybullet_world_position": list(
                                    ((frame.get("objects") or {}).get(
                                        str(articulation_binding["host_object_id"]), {}
                                    ).get("link_states") or {}).get(
                                        str(articulation_binding["child_link"]), {}
                                    ).get("position", [])
                                ),
                                "drake_world_position": plant.EvalBodyPoseInWorld(
                                    plant_context, articulation_child_body
                                ).translation().tolist(),
                            }
                        }
                        if articulation_joint is not None and articulation_child_body is not None
                        else {}
                    ),
                }
            )
            for name in camera_poses:
                destination = output / name / "frames" / f"{int(frame['frame']):05d}.png"
                if destination.is_file():
                    existing = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
                    if existing is None or existing.shape[:2] != (args.height, args.width):
                        raise RuntimeError(f"cached rendered frame is invalid: {destination}")
                else:
                    rgba = np.asarray(diagram.GetOutputPort(f"rgba_{name}").Eval(context).data)
                    if not cv2.imwrite(str(destination), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)):
                        raise RuntimeError(f"unable to write rendered frame: {destination}")
                if int(frame["frame"]) in evidence_frame_ids:
                    semantic_destination = (
                        output
                        / name
                        / "semantic_masks"
                        / f"{int(frame['frame']):05d}.png"
                    )
                    if semantic_destination.is_file():
                        semantic = cv2.imread(
                            str(semantic_destination), cv2.IMREAD_UNCHANGED
                        )
                        if semantic is None or semantic.shape != (args.height, args.width):
                            raise RuntimeError(
                                f"cached semantic mask is invalid: {semantic_destination}"
                            )
                    else:
                        label = np.asarray(
                            diagram.GetOutputPort(f"label_{name}").Eval(context).data,
                            dtype=np.int16,
                        )
                        if label.ndim == 3 and label.shape[-1] == 1:
                            label = label[..., 0]
                        if label.shape != (args.height, args.width):
                            raise RuntimeError(
                                f"{name} label image has unexpected shape {label.shape}"
                            )
                        semantic = np.zeros(label.shape, dtype=np.uint8)
                        semantic[
                            np.isin(
                                label,
                                scene_object_render_labels[active_subject_id],
                            )
                        ] = 1
                        semantic[np.isin(label, left_finger_render_labels)] = 2
                        semantic[np.isin(label, right_finger_render_labels)] = 3
                        if not cv2.imwrite(str(semantic_destination), semantic):
                            raise RuntimeError(
                                f"unable to write semantic mask: {semantic_destination}"
                            )
                    semantic_records[name].append(
                        {
                            "frame": int(frame["frame"]),
                            "time_s": float(frame["time_s"]),
                            "phase": str(frame["phase"]),
                            "active_subject_id": active_subject_id,
                            "image": str(semantic_destination),
                        }
                    )
                if frame is frames[0]:
                    label = np.asarray(
                        diagram.GetOutputPort(f"label_{name}").Eval(context).data,
                        dtype=np.int32,
                    )
                    if label.ndim == 3 and label.shape[-1] == 1:
                        label = label[..., 0]
                    if label.shape != (args.height, args.width):
                        raise RuntimeError(
                            f"{name} label image has unexpected shape {label.shape}"
                        )
                    conditioning_semantic = np.zeros(label.shape, dtype=np.uint16)
                    for object_id, semantic_value in conditioning_object_values.items():
                        conditioning_semantic[
                            np.isin(
                                label,
                                scene_object_render_labels[
                                    conditioning_object_sources[object_id]
                                ],
                            )
                        ] = semantic_value
                    conditioning_semantic[
                        np.isin(
                            label,
                            np.concatenate(
                                [left_finger_render_labels, right_finger_render_labels]
                            ),
                        )
                    ] = conditioning_end_effector_value
                    conditioning_semantic_path = (
                        output
                        / name
                        / "conditioning_semantic"
                        / f"{int(frame['frame']):05d}.png"
                    )
                    if not cv2.imwrite(
                        str(conditioning_semantic_path), conditioning_semantic
                    ):
                        raise RuntimeError(
                            "unable to write deterministic conditioning mask: "
                            f"{conditioning_semantic_path}"
                        )
                    conditioning_semantic_records[name] = {
                        "frame": int(frame["frame"]),
                        "image": str(conditioning_semantic_path),
                        "object_values": conditioning_object_values,
                        "end_effector_value": conditioning_end_effector_value,
                    }
                    depth = np.asarray(
                        diagram.GetOutputPort(f"depth_{name}").Eval(context).data,
                        dtype=np.float32,
                    ).copy()
                    if depth.ndim == 3 and depth.shape[-1] == 1:
                        depth = depth[..., 0]
                    if depth.shape != (args.height, args.width):
                        raise RuntimeError(
                            f"{name} initial depth has unexpected shape {depth.shape}"
                        )
                    finite = depth[np.isfinite(depth) & (depth > 0)]
                    if finite.size == 0:
                        raise RuntimeError(f"{name} initial depth has no finite pixels")
                    depth[~np.isfinite(depth) | (depth <= 0)] = float(finite.max())
                    base = output / name / "cae_base_scene"
                    base.mkdir(exist_ok=True)
                    torch.save(torch.from_numpy(depth), base / "initial_depth.pt")
                    camera_to_world = (
                        plant.EvalBodyPoseInWorld(plant_context, wrist_body)
                        .multiply(wrist_extrinsic)
                        if name == "wrist"
                        else rigid_camera_poses[name]
                    )
                    effective_fov_y_deg = view_fov_y_deg(name)
                    focal = args.height / (
                        2.0 * math.tan(math.radians(effective_fov_y_deg) / 2.0)
                    )
                    (base / "camera_calibration_info.json").write_text(
                        json.dumps(
                            {
                                "head_camera": {
                                    "intrinsics": [
                                        [focal, 0.0, args.width / 2.0],
                                        [0.0, focal, args.height / 2.0],
                                        [0.0, 0.0, 1.0],
                                    ],
                                    "extrinsics": camera_to_world.GetAsMatrix4().tolist(),
                                }
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    (base / "scene_data.yaml").write_text(
                        json.dumps(
                            {
                                "instruction": args.instruction,
                                "object_name": args.object_name,
                                "canonical_object_name": args.canonical_object_name,
                                "robot_start_joints": DREAM2FLOW_PANDA_START_JOINTS,
                                "physical_view": name,
                                "depth_backend": "Drake RenderEngineVtk with SceneCode SDF and PyBullet state",
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    first_rgb = cv2.imread(str(destination), cv2.IMREAD_COLOR)
                    if first_rgb is None or not cv2.imwrite(str(base / "camera_rgb.png"), first_rgb):
                        raise RuntimeError(f"unable to publish {name} CAE camera RGB")
                records[name].append(
                    {
                        "frame": int(frame["frame"]),
                        "time_s": float(frame["time_s"]),
                        "phase": str(frame["phase"]),
                        "image": str(destination),
                    }
                )

        manifest = {
            "schema_version": (
                "scenecode_robotics_multiview_render.v4"
                if scene_object_sources is not None
                else "scenecode_robotics_multiview_render.v3"
            ),
            "renderer": "SceneCode Drake glTF client -> official Blender /render",
            "trajectory": str(args.trajectory.resolve()),
            "width": args.width,
            "height": args.height,
            "fov_y_deg": view_fov_y_deg("top"),
            "source_fov_y_deg": args.fov_y_deg,
            "fov_policy": (
                "preserve_horizontal_from_source_aspect"
                if args.preserve_horizontal_fov_from_aspect is not None
                else "direct_vertical_fov"
            ),
            "source_aspect_ratio": args.preserve_horizontal_fov_from_aspect,
            "fixed_camera_preset": args.fixed_camera_preset,
            "camera_rig": str(args.camera_rig.resolve()) if args.camera_rig else None,
            "camera_rig_sha256": _sha256(args.camera_rig) if args.camera_rig else None,
            "robot_embodiment": args.robot_embodiment,
            "wrist_camera_parent_body": wrist_body.name(),
            "wrist_camera_parent_frame": wrist_camera_parent_frame,
            "semantic_evidence_mode": args.semantic_evidence_mode,
            "scene_objects_manifest": (
                str(args.scene_objects_manifest.resolve())
                if args.scene_objects_manifest is not None
                else None
            ),
            "articulation_binding": (
                str(args.articulation_binding.resolve())
                if args.articulation_binding is not None
                else None
            ),
            "scene_object_ids": sorted(scene_object_bodies),
            "blender_settings_file": str(settings_file),
            "robot_registration": {
                "authority": "PyBullet link-frame world poses compared with Drake replay",
                "frames": robot_registration_records,
                "maximum_finger_translation_residual_m": max(
                    item["translation_residual_m"]
                    for record in robot_registration_records
                    for item in record["fingers"].values()
                ),
            },
            "views": {
                name: {
                    "camera_type": "robot_mounted" if name == "wrist" else "fixed_world",
                    "size_wh": [args.width, args.height],
                    "camera_signature": json.dumps(camera_poses[name].__repr__()),
                    "camera_preset": (
                        args.wrist_camera_preset
                        if name == "wrist"
                        else (
                            "fixed_world_top"
                            if name == "top"
                            else args.fixed_camera_preset
                        )
                    ),
                    "fov_y_deg": view_fov_y_deg(name),
                    "source_fov_y_deg": (
                        args.wrist_fov_y_deg
                        if name == "wrist" and args.wrist_fov_y_deg is not None
                        else args.fov_y_deg
                    ),
                    "wrist_optical_yaw_deg": (
                        args.wrist_optical_yaw_deg if name == "wrist" else None
                    ),
                    "wrist_optical_pitch_deg": (
                        args.wrist_optical_pitch_deg if name == "wrist" else None
                    ),
                    "wrist_optical_forward_offset_m": (
                        args.wrist_optical_forward_offset_m if name == "wrist" else None
                    ),
                    "cae_base_scene": str(output / name / "cae_base_scene"),
                    "semantic_mask_values": {
                        "background": 0,
                        "cup": 1,
                        "left_finger": 2,
                        "right_finger": 3,
                        "subject": 1,
                        "destination": 2,
                        **(
                            {
                                str(articulation_binding["component_id"]): 1,
                                str(articulation_binding["host_object_id"]): 2,
                            }
                            if articulation_binding is not None
                            else {}
                        ),
                    },
                    "semantic_finger_bodies": {
                        "left": [body.name() for body in left_finger_semantic_bodies],
                        "right": [body.name() for body in right_finger_semantic_bodies],
                    },
                    "semantic_masks": semantic_records[name],
                    "conditioning_semantic": conditioning_semantic_records[name],
                    "frames": records[name],
                }
                for name in camera_poses
            },
        }
        manifest_path = output / args.manifest_name
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(manifest_path)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
