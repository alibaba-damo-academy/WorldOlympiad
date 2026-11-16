#!/usr/bin/env python3
"""Replay a PyBullet trajectory through SceneCode's official Blender renderer."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import requests
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ApplyCameraConfig,
    CameraConfig,
    DiagramBuilder,
    Parser,
    Quaternion,
    RenderEngineGltfClientParams,
    Rgba,
    RigidTransform,
    RotationMatrix,
    SpatialInertia,
    Sphere,
    Transform,
    UnitInertia,
)

from scenecode.agent_utils.blender.request_lock import acquire_blender_request_lock
from scenecode.agent_utils.blender.server_manager import BlenderServer


def _rigid_transform(position: list[float], quaternion_xyzw: list[float]) -> RigidTransform:
    x, y, z, w = quaternion_xyzw
    return RigidTransform(Quaternion(w, x, y, z), position)


def _select_frames(frames: list[dict], requested: str) -> list[dict]:
    if requested == "all":
        return frames
    indices = [int(value) for value in requested.split(",")]
    by_index = {int(frame["frame"]): frame for frame in frames}
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise ValueError(f"trajectory does not contain frame(s): {missing}")
    return [by_index[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--table-sdf", type=Path, required=True)
    parser.add_argument("--cup-sdf", type=Path, required=True)
    parser.add_argument("--tray-sdf", type=Path, required=True)
    parser.add_argument("--panda-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", default="all", help="all or comma-separated indices")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--port", type=int, default=8166)
    args = parser.parse_args()

    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    frames = _select_frames(trajectory["frames"], args.frames)
    if not frames:
        raise ValueError("trajectory is empty")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "blender_server.log"
    server = BlenderServer(port=args.port, log_file=log_file, gpu_id=0)

    try:
        server.start()
        server.wait_until_ready(timeout=90.0)

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        asset_parser = Parser(plant)
        asset_parser.SetAutoRenaming(True)
        asset_parser.package_map().Add(
            "meshes", str(args.panda_urdf.resolve().parent / "meshes")
        )
        table_model = asset_parser.AddModels(str(args.table_sdf.resolve()))[0]
        cup_model = asset_parser.AddModels(str(args.cup_sdf.resolve()))[0]
        tray_model = asset_parser.AddModels(str(args.tray_sdf.resolve()))[0]
        panda_model = asset_parser.AddModels(str(args.panda_urdf.resolve()))[0]

        # SceneCode's official renderer frames every request from its current
        # geometry bounds.  Eight effectively invisible, welded points establish
        # one fixed trajectory-wide box, with enough margin to keep the complete
        # arm in frame while avoiding per-frame camera zoom.
        camera_anchor_model = plant.AddModelInstance("camera_framing_anchor")
        camera_anchor = plant.AddRigidBody(
            "camera_framing_anchor",
            camera_anchor_model,
            SpatialInertia(
                mass=1.0,
                p_PScm_E=(0.0, 0.0, 0.0),
                G_SP_E=UnitInertia.SolidSphere(0.001),
            ),
        )
        for index, (x, y, z) in enumerate(
            (x, y, z)
            for x in (-0.95, 0.95)
            for y in (-0.80, 0.65)
            for z in (-0.10, 1.80)
        ):
            plant.RegisterVisualGeometry(
                camera_anchor,
                RigidTransform((x, y, z)),
                Sphere(0.00001),
                f"camera_framing_anchor_visual_{index}",
                np.array([0.0, 0.0, 0.0, 0.0]),
            )

        table_body = plant.get_body(plant.GetBodyIndices(table_model)[0])
        cup_body = plant.get_body(plant.GetBodyIndices(cup_model)[0])
        tray_body = plant.get_body(plant.GetBodyIndices(tray_model)[0])
        panda_base_body = plant.GetBodyByName("panda_link0", panda_model)
        panda_base = frames[0]["robot"]["base_position"]
        plant.WeldFrames(plant.world_frame(), table_body.body_frame(), RigidTransform.Identity())
        plant.WeldFrames(
            plant.world_frame(),
            panda_base_body.body_frame(),
            RigidTransform(RotationMatrix.MakeZRotation(math.pi / 2.0), panda_base),
        )
        plant.WeldFrames(
            plant.world_frame(),
            camera_anchor.body_frame(),
            RigidTransform.Identity(),
        )
        plant.Finalize()

        camera = CameraConfig(
            X_PB=Transform(RigidTransform.Identity()),
            width=4,
            height=4,
            background=Rgba(0.0, 0.0, 0.0, 1.0),
            renderer_class=RenderEngineGltfClientParams(
                base_url=server.get_url(), render_endpoint="render_overlay"
            ),
        )
        ApplyCameraConfig(
            config=camera,
            builder=builder,
            plant=plant,
            scene_graph=scene_graph,
        )
        builder.ExportOutput(
            builder.GetSubsystemByName(
                f"rgbd_sensor_{camera.name}"
            ).color_image_output_port(),
            "rgba_image",
        )
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyMutableContextFromRoot(context)

        drake_joint_names = plant.GetPositionNames(
            panda_model, add_model_instance_prefix=False
        )
        rendered: list[dict] = []
        for frame in frames:
            robot = frame["robot"]
            normalized_drake_names = [
                name.removesuffix("_q").removesuffix("_x")
                for name in drake_joint_names
            ]
            if normalized_drake_names != robot["joint_names"]:
                raise RuntimeError(
                    "Panda joint order differs between PyBullet and Drake: "
                    f"{robot['joint_names']} != {normalized_drake_names}"
                )
            name = f"frame_{int(frame['frame']):05d}"
            expected = output_dir / f"{name}.png"
            if expected.is_file():
                rendered.append(
                    {
                        "frame": frame["frame"],
                        "time_s": frame["time_s"],
                        "phase": frame["phase"],
                        "image": str(expected),
                    }
                )
                continue
            plant.SetPositions(plant_context, panda_model, robot["joint_positions"])
            plant.SetFreeBodyPose(
                plant_context,
                cup_body,
                _rigid_transform(
                    frame["cup"]["link_position"],
                    frame["cup"]["link_orientation_xyzw"],
                ),
            )
            plant.SetFreeBodyPose(
                plant_context,
                tray_body,
                _rigid_transform(
                    frame["tray"]["link_position"],
                    frame["tray"]["link_orientation_xyzw"],
                ),
            )

            frame_output_dir = output_dir / "raw_views" / name
            frame_output_dir.mkdir(parents=True, exist_ok=True)
            config = {
                "output_dir": str(frame_output_dir),
                "layout": "top_plus_sides",
                "top_view_width": args.width,
                "top_view_height": args.height,
                "side_view_count": 1,
                "side_view_width": args.width,
                "side_view_height": args.height,
                "wall_normals": {},
                "annotations": {
                    "enable_set_of_mark_labels": False,
                    "enable_bounding_boxes": False,
                    "enable_direction_arrows": False,
                    "enable_coordinate_grid": False,
                    "enable_partial_walls": False,
                    "enable_support_surface_debug": False,
                    "enable_convex_hull_debug": False,
                    "show_coordinate_frame": False,
                    "rendering_mode": "furniture",
                },
                "scene_objects": [],
                "taa_samples": 16,
                "side_view_elevation_degrees": 30.0,
                "side_view_start_azimuth_degrees": -45.0,
                "include_vertical_views": False,
            }
            with acquire_blender_request_lock("render_overlay.pick_place"):
                response = requests.post(
                    f"{server.get_url()}/set_overlay_config", json=config, timeout=10
                )
                response.raise_for_status()
                _ = diagram.GetOutputPort("rgba_image").Eval(context)

            side_candidates = sorted(frame_output_dir.glob("*_side.png"))
            if len(side_candidates) != 1:
                raise RuntimeError(
                    f"renderer did not produce one side image for {name}: "
                    f"{side_candidates}"
                )
            shutil.copy2(side_candidates[0], expected)
            rendered.append(
                {
                    "frame": frame["frame"],
                    "time_s": frame["time_s"],
                    "phase": frame["phase"],
                    "image": str(expected),
                }
            )

        manifest = {
            "schema_version": "scenecode_official_render_replay_v1",
            "renderer": "SceneCode Drake glTF client -> official Blender server",
            "camera_direction": [0.6123724357, -0.6123724357, 0.5],
            "width": args.width,
            "height": args.height,
            "frames": rendered,
        }
        (output_dir / "render_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))
    finally:
        server.stop()


if __name__ == "__main__":
    main()
