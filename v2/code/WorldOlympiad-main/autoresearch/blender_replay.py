"""Render authoritative PyBullet link trajectories with Blender/PBR visuals."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import write_json
from .media import write_h264_rgb


@dataclass(frozen=True)
class BlenderReplayResult:
    rgb_frames: list[np.ndarray]
    report_path: Path


def _resolve_blender(executable: str | Path | None) -> Path:
    candidate = (
        str(executable)
        if executable is not None
        else os.environ.get("SCENECODE_BLENDER") or shutil.which("blender")
    )
    if not candidate:
        raise RuntimeError(
            "Blender 4.4 is required; pass blender_executable or set SCENECODE_BLENDER"
        )
    path = Path(candidate).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_rgb_frames(paths: list[Path]) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for Blender replay QA") from exc
    frames: list[np.ndarray] = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Blender emitted an unreadable frame: {path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def _appearance_metrics(frames: list[np.ndarray]) -> dict[str, float]:
    stack = np.asarray(frames, dtype=np.float32)
    luma = 0.2126 * stack[..., 0] + 0.7152 * stack[..., 1] + 0.0722 * stack[..., 2]
    chroma = stack.max(axis=-1) - stack.min(axis=-1)
    first = stack[0]
    motion = [float(np.mean(np.abs(frame - first))) for frame in stack[1:]]
    return {
        "mean_luma": float(luma.mean()),
        "dark_clip_fraction": float(np.mean(luma <= 3.0)),
        "bright_clip_fraction": float(np.mean(luma >= 252.0)),
        "mean_rgb_chroma": float(chroma.mean()),
        "maximum_mean_absolute_change_from_first": max(motion, default=0.0),
    }


def _mask_iou(rendered: np.ndarray, expected: np.ndarray) -> float:
    intersection = int(np.count_nonzero(rendered & expected))
    union = int(np.count_nonzero(rendered | expected))
    return 1.0 if union == 0 else intersection / union


def render_blender_replay(
    *,
    bundle_path: str | Path,
    trajectory_path: str | Path,
    calibration_path: str | Path,
    output_dir: str | Path,
    fps: int,
    target_name: str,
    blender_executable: str | Path | None = None,
    width: int = 1280,
    height: int = 720,
    segmentation_frames: list[np.ndarray] | None = None,
    target_instance_id: int | None = None,
    robot_instance_id: int | None = None,
) -> BlenderReplayResult:
    """Run the bundled Blender replay script and fail closed on visual defects."""

    if fps <= 0 or width < 960 or height < 720:
        raise ValueError("formal Blender replay requires positive fps and at least 960x720")
    bundle = Path(bundle_path).resolve()
    trajectory = Path(trajectory_path).resolve()
    calibration = Path(calibration_path).resolve()
    for path in (bundle, trajectory, calibration):
        if not path.is_file():
            raise FileNotFoundError(path)
    camera_payload = json.loads(calibration.read_text(encoding="utf-8"))
    destination = Path(output_dir).resolve()
    frames_dir = destination / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)
    masks_dir: Path | None = None
    if segmentation_frames is not None:
        if target_instance_id is None:
            raise ValueError("segmentation QA requires target_instance_id")
        masks_dir = destination / "instance_masks"
        masks_dir.mkdir()
    with np.load(trajectory, allow_pickle=False) as payload:
        timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
        required = {"link_names", "link_positions", "link_orientations_xyzw"}
        missing = required - set(payload.files)
        if missing:
            raise RuntimeError(f"trajectory is missing Blender replay fields: {sorted(missing)}")
    duration_s = float(timestamps[-1] - timestamps[0])
    expected_frames = round(duration_s * fps) + 1
    if duration_s < 4.0 - 1e-6:
        raise RuntimeError(f"formal imagination video is too short: {duration_s:.6f}s")

    config_path = write_json(
        destination / "replay_config.json",
        {
            "schema_version": "scenecode_pybullet_blender_replay.v1",
            "bundle_path": str(bundle),
            "trajectory_path": str(trajectory),
            "calibration_path": str(calibration),
            "frames_dir": str(frames_dir),
            "fps": fps,
            "width": width,
            "height": height,
            "target_name": target_name,
            "robot_name": "robot" if robot_instance_id is not None else None,
            "expected_frames": expected_frames,
            "render_engine": "BLENDER_EEVEE_NEXT",
            "masks_dir": str(masks_dir) if masks_dir is not None else None,
        },
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "blender_pybullet_replay.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    blender = _resolve_blender(blender_executable)
    completed = subprocess.run(
        [
            str(blender),
            "--background",
            "--factory-startup",
            "--python",
            str(script),
            "--",
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    (destination / "blender_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (destination / "blender_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "Blender replay failed; inspect "
            f"{destination / 'blender_stderr.log'} (exit={completed.returncode})"
        )
    script_report_path = destination / "blender_script_report.json"
    if not script_report_path.is_file():
        raise RuntimeError("Blender replay completed without an audit report")
    script_report = json.loads(script_report_path.read_text(encoding="utf-8"))
    paths = sorted(frames_dir.glob("*.png"))
    if len(paths) != expected_frames:
        raise RuntimeError(
            f"Blender replay emitted {len(paths)} frames, expected {expected_frames}"
        )
    frames = _load_rgb_frames(paths)
    shapes = {tuple(frame.shape) for frame in frames}
    expected_shape = (height, width, 3)
    if shapes != {expected_shape}:
        raise RuntimeError(f"Blender replay resolution mismatch: {sorted(shapes)}")
    metrics = _appearance_metrics(frames)
    mask_metrics: dict[str, float | list[float]] = {}
    if segmentation_frames is not None:
        if len(segmentation_frames) != expected_frames:
            raise RuntimeError(
                "segmentation QA frame count does not match Blender replay: "
                f"{len(segmentation_frames)} != {expected_frames}"
            )
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv is required for instance-mask QA") from exc
        target_ious: list[float] = []
        robot_ious: list[float] = []
        target_paths = sorted(masks_dir.glob("target_*.png")) if masks_dir else []
        robot_paths = sorted(masks_dir.glob("robot_*.png")) if masks_dir else []
        if len(target_paths) != expected_frames or len(robot_paths) != expected_frames:
            raise RuntimeError(
                "Blender compositor did not emit complete target/robot masks: "
                f"target={len(target_paths)}, robot={len(robot_paths)}"
            )
        for index, simulator_mask in enumerate(segmentation_frames):
            target_rendered = cv2.imread(str(target_paths[index]), cv2.IMREAD_GRAYSCALE) > 127
            target_expected = np.asarray(simulator_mask == target_instance_id, dtype=np.uint8)
            target_expected = cv2.resize(
                target_expected, (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            target_ious.append(_mask_iou(target_rendered, target_expected))
            if robot_instance_id is not None:
                robot_rendered = cv2.imread(str(robot_paths[index]), cv2.IMREAD_GRAYSCALE) > 127
                robot_expected = np.asarray(simulator_mask == robot_instance_id, dtype=np.uint8)
                robot_expected = cv2.resize(
                    robot_expected, (width, height), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                robot_ious.append(_mask_iou(robot_rendered, robot_expected))
        mask_metrics = {
            "target_iou_per_frame": target_ious,
            "target_iou_mean": float(np.mean(target_ious)),
            "target_iou_minimum": float(np.min(target_ious)),
        }
        if robot_ious:
            mask_metrics.update(
                {
                    "robot_iou_per_frame": robot_ious,
                    "robot_iou_mean": float(np.mean(robot_ious)),
                    "robot_iou_minimum": float(np.min(robot_ious)),
                }
            )
    probes = {
        "all_trajectory_links_mapped": not script_report.get("missing_trajectory_links"),
        "all_visual_geometry_loaded": not script_report.get("missing_visuals"),
        "target_visual_present": int(script_report.get("target_visual_count", 0)) > 0,
        "resolution_at_least_960x720": width >= 960 and height >= 720,
        "duration_at_least_4s": duration_s >= 4.0 - 1e-6,
        "brightness_in_range": 20.0 <= metrics["mean_luma"] <= 235.0,
        "clipping_bounded": (
            metrics["dark_clip_fraction"] <= 0.35
            and metrics["bright_clip_fraction"] <= 0.35
        ),
        "non_monochrome_scene": metrics["mean_rgb_chroma"] >= 3.0,
        "visible_motion": metrics["maximum_mean_absolute_change_from_first"] >= 0.75,
        "no_pybullet_rgb_fallback": True,
    }
    robot_side_alignment = camera_payload.get("selection_robot_side_alignment")
    if robot_side_alignment is not None:
        probes["camera_on_robot_work_side"] = float(robot_side_alignment) >= 0.10
    if mask_metrics:
        probes["target_visual_matches_pybullet"] = (
            float(mask_metrics["target_iou_mean"]) >= 0.60
            and float(mask_metrics["target_iou_minimum"]) >= 0.50
        )
        if "robot_iou_mean" in mask_metrics:
            probes["robot_visual_matches_pybullet"] = (
                float(mask_metrics["robot_iou_mean"]) >= 0.70
                and float(mask_metrics["robot_iou_minimum"]) >= 0.50
            )
    report = {
        "schema_version": "scenecode_pybullet_blender_render_report.v1",
        "status": "pass" if all(probes.values()) else "quarantined",
        "renderer": "blender_replay",
        "render_engine": script_report.get("render_engine"),
        "frame_count": len(frames),
        "fps": fps,
        "duration_s": duration_s,
        "width": width,
        "height": height,
        "target_name": target_name,
        "camera_robot_side_alignment": robot_side_alignment,
        "probes": probes,
        "appearance_metrics": metrics,
        "instance_mask_metrics": mask_metrics,
        "instance_mask_comparison": (
            "blender_amodal_selected_body_silhouette_vs_pybullet_visible_instance_mask"
            if mask_metrics
            else None
        ),
        "blender_script_report": script_report,
    }
    report_path = write_json(destination / "render_report.json", report)
    if report["status"] != "pass":
        failed = sorted(name for name, passed in probes.items() if not passed)
        raise RuntimeError(f"Blender replay was quarantined: {failed}; report={report_path}")
    write_h264_rgb(destination / "rgb.mp4", frames, fps, crf=16)
    return BlenderReplayResult(rgb_frames=frames, report_path=report_path)
