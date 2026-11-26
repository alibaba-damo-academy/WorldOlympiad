#!/usr/bin/env python3
"""Render independent SceneCode robotics views in parallel and merge manifests."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


VIEWS = ("top", "wrist", "exterior_left", "exterior_right")


def _view_size(value: str) -> tuple[str, tuple[int, int]]:
    match = re.fullmatch(r"(top|wrist|exterior_left|exterior_right)=([1-9][0-9]*)x([1-9][0-9]*)", value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "view size must be VIEW=WIDTHxHEIGHT, for example top=256x256"
        )
    return match.group(1), (int(match.group(2)), int(match.group(3)))


def _view_frames(value: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"(top|wrist|exterior_left|exterior_right)=(all|[0-9]+(?:,[0-9]+)*)", value
    )
    if match is None:
        raise argparse.ArgumentTypeError(
            "view frames must be VIEW=all or VIEW=0,3,7"
        )
    return match.group(1), match.group(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-script", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--camera-rig", type=Path)
    parser.add_argument("--views", default="all")
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
    )
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument(
        "--view-size",
        action="append",
        type=_view_size,
        default=[],
        help="Override one physical view with a direct native render size.",
    )
    parser.add_argument(
        "--frames",
        default="all",
        help="all or a comma-separated reviewed key-frame set forwarded to every view",
    )
    parser.add_argument(
        "--view-frames", action="append", type=_view_frames, default=[],
        help="Override the rendered frame set for one physical view.",
    )
    parser.add_argument("--fov-y-deg", type=float, default=52.0)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--canonical-object-name", required=True)
    parser.add_argument("--semantic-subject-object-id")
    parser.add_argument("--semantic-destination-object-id")
    parser.add_argument("--semantic-support-object-id")
    parser.add_argument("--wrist-camera-preset", default="robolab_droid")
    parser.add_argument(
        "--robot-embodiment",
        choices=("panda_stock", "robolab_droid"),
        default="panda_stock",
    )
    parser.add_argument("--fixed-camera-preset", default="reference_contact")
    parser.add_argument(
        "--semantic-evidence-mode",
        choices=("carry_phases", "dual_contact", "push_contact", "articulation_contact", "none"),
        default="carry_phases",
    )
    parser.add_argument("--wrist-fov-y-deg", type=float)
    parser.add_argument(
        "--preserve-horizontal-fov-from-aspect",
        help="Forward WIDTH:HEIGHT source aspect to each native-aspect render worker.",
    )
    args = parser.parse_args()
    selected_views = VIEWS if args.views == "all" else tuple(args.views.split(","))
    if not selected_views or len(set(selected_views)) != len(selected_views) or not set(selected_views) <= set(VIEWS):
        raise ValueError("invalid selected camera views")
    if args.scene_objects_manifest is None:
        if args.cup_sdf is None or args.tray_sdf is None or args.table_sdf is None:
            raise ValueError(
                "legacy parallel render requires --table-sdf, --cup-sdf, and --tray-sdf"
            )
    elif args.cup_sdf is not None or args.tray_sdf is not None:
        raise ValueError(
            "scene-object manifest is mutually exclusive with legacy cup/tray SDFs"
        )
    if args.articulation_binding is not None and args.scene_objects_manifest is None:
        raise ValueError("articulation binding requires a scene-object manifest")

    output = args.output_dir.resolve()
    final_manifest = output / "render_manifest.json"
    if final_manifest.exists():
        raise FileExistsError(f"refusing to overwrite completed render: {final_manifest}")
    if args.port < 8166 or args.port + len(selected_views) - 1 > 8199:
        raise ValueError("parallel SceneCode Blender ports must stay in [8166, 8199]")
    output.mkdir(parents=True, exist_ok=True)
    view_sizes = {view: (args.width, args.height) for view in selected_views}
    overridden: set[str] = set()
    for view, size in args.view_size:
        if view in overridden:
            raise ValueError(f"duplicate native render size for {view}")
        overridden.add(view)
        view_sizes[view] = size

    frames_by_view = {view: args.frames for view in selected_views}
    overridden_frames: set[str] = set()
    for view, frames in args.view_frames:
        if view not in frames_by_view:
            raise ValueError(f"frame override names unselected view {view}")
        if view in overridden_frames:
            raise ValueError(f"duplicate frame override for {view}")
        overridden_frames.add(view)
        frames_by_view[view] = frames

    processes: list[tuple[str, subprocess.Popen, object]] = []
    lock_root = output / ".worker_locks"
    lock_root.mkdir(exist_ok=True)
    try:
        for index, view in enumerate(selected_views):
            width, height = view_sizes[view]
            manifest_name = f"render_manifest.{view}.json"
            command = [
                sys.executable,
                str(args.render_script.resolve()),
                "--trajectory", str(args.trajectory.resolve()),
                "--panda-urdf", str(args.panda_urdf.resolve()),
                "--output-dir", str(output),
                "--blender-settings-file", str(args.blender_settings_file.resolve()),
                "--width", str(width),
                "--height", str(height),
                "--frames", frames_by_view[view],
                "--fov-y-deg", str(args.fov_y_deg),
                "--port", str(args.port + index),
                "--views", view,
                "--resume",
                "--instruction", args.instruction,
                "--object-name", args.object_name,
                "--canonical-object-name", args.canonical_object_name,
                "--manifest-name", manifest_name,
                "--wrist-camera-preset", args.wrist_camera_preset,
                "--robot-embodiment", args.robot_embodiment,
                "--fixed-camera-preset", args.fixed_camera_preset,
                "--semantic-evidence-mode", args.semantic_evidence_mode,
            ]
            if args.camera_rig is not None:
                command.extend(["--camera-rig", str(args.camera_rig.resolve())])
            if args.table_sdf is not None:
                command.extend(["--table-sdf", str(args.table_sdf.resolve())])
            if args.articulation_binding is not None:
                command.extend(["--articulation-binding", str(args.articulation_binding.resolve())])
            if args.scene_objects_manifest is not None:
                command.extend(
                    [
                        "--scene-objects-manifest",
                        str(args.scene_objects_manifest.resolve()),
                    ]
                )
            else:
                command.extend(
                    [
                        "--cup-sdf",
                        str(args.cup_sdf.resolve()),
                        "--tray-sdf",
                        str(args.tray_sdf.resolve()),
                    ]
                )
                if args.semantic_subject_object_id is not None:
                    command.extend(
                        [
                            "--semantic-subject-object-id",
                            args.semantic_subject_object_id,
                        ]
                    )
                if args.semantic_destination_object_id is not None:
                    command.extend(
                        [
                            "--semantic-destination-object-id",
                            args.semantic_destination_object_id,
                        ]
                    )
                if args.semantic_support_object_id is not None:
                    command.extend(
                        [
                            "--semantic-support-object-id",
                            args.semantic_support_object_id,
                        ]
                    )
            if args.wrist_fov_y_deg is not None:
                command.extend(["--wrist-fov-y-deg", str(args.wrist_fov_y_deg)])
            if args.preserve_horizontal_fov_from_aspect is not None:
                command.extend(
                    [
                        "--preserve-horizontal-fov-from-aspect",
                        args.preserve_horizontal_fov_from_aspect,
                    ]
                )
            log = (output / f"parallel_{view}.log").open("w", encoding="utf-8")
            worker_env = dict(os.environ)
            # Every worker owns an independent Blender server, port, and temp
            # directory.  A node-global lock would serialize all four servers
            # and multiply reference latency without protecting shared state.
            # Keep per-server serialization while allowing the four physical
            # cameras to progress concurrently.
            worker_env["SCENECODE_BLENDER_GLOBAL_LOCK"] = str(
                lock_root / f"{view}.lock"
            )
            processes.append(
                (
                    view,
                    subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=worker_env,
                    ),
                    log,
                )
            )
        failures = []
        for view, process, log in processes:
            status = process.wait()
            log.close()
            if status != 0:
                failures.append((view, status))
        if failures:
            raise RuntimeError(f"parallel SceneCode render workers failed: {failures}")
    except BaseException:
        for _, process, log in processes:
            if process.poll() is None:
                process.terminate()
            if not log.closed:
                log.close()
        raise

    manifests = []
    for view in selected_views:
        path = output / f"render_manifest.{view}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if set(payload.get("views", {})) != {view}:
            raise ValueError(f"worker manifest does not contain only {view}: {path}")
        manifests.append(payload)
    shared_keys = (
        "schema_version",
        "renderer",
        "trajectory",
        "source_fov_y_deg",
        "fov_policy",
        "source_aspect_ratio",
        "fixed_camera_preset",
        "robot_embodiment",
        "wrist_camera_parent_body",
        "wrist_camera_parent_frame",
        "semantic_evidence_mode",
        "scene_objects_manifest",
        "articulation_binding",
        "scene_object_ids",
        "blender_settings_file",
        "camera_rig",
        "camera_rig_sha256",
        "robot_registration",
    )
    baseline = manifests[0]
    for payload in manifests[1:]:
        if any(payload.get(key) != baseline.get(key) for key in shared_keys):
            raise ValueError("parallel render manifests disagree on shared provenance")
    merged = {key: baseline.get(key) for key in shared_keys}
    if overridden:
        merged["schema_version"] = "scenecode_robotics_multiview_render.v4"
        merged["render_size_wh_by_view"] = {
            view: list(view_sizes[view]) for view in selected_views
        }
    else:
        merged["width"] = baseline["width"]
        merged["height"] = baseline["height"]
        merged["fov_y_deg"] = baseline["fov_y_deg"]
    merged["parallel_view_workers"] = len(selected_views)
    merged["parallel_request_lock_scope"] = (
        "one_lock_per_dedicated_blender_server_and_physical_view"
    )
    merged["views"] = {
        view: payload["views"][view]
        for view, payload in zip(selected_views, manifests, strict=True)
    }
    staging = final_manifest.with_suffix(".json.partial")
    staging.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staging.replace(final_manifest)
    print(final_manifest)


if __name__ == "__main__":
    main()
