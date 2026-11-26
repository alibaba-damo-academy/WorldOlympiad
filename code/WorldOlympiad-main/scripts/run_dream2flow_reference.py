#!/usr/bin/env python3
"""Run Dream2Flow on one RGB video and export its inferred action trajectory."""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np


def _deterministic_farthest_point_indices(points, ratio: float):
    """Dependency-free diagnostic fallback for geometric FPS.

    Formal CAE runs require PyG's official pyg-lib operator. This implementation
    exists only for explicitly requested diagnostics on machines where the
    binary extension cannot be loaded. It selects the point farthest from the
    centroid first and performs the standard greedy update entirely in Torch.
    """

    import math
    import torch

    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError(f"FPS expects a non-empty NxD tensor, got {tuple(points.shape)}")
    if not 0 < ratio <= 1:
        raise ValueError(f"FPS ratio must lie in (0,1], got {ratio}")
    count = min(len(points), max(1, int(math.ceil(len(points) * ratio))))
    coordinates = points.detach().float()
    selected = torch.empty(count, dtype=torch.long, device=coordinates.device)
    centroid = coordinates.mean(dim=0, keepdim=True)
    current = int(torch.argmax(torch.sum((coordinates - centroid) ** 2, dim=1)).item())
    minimum_squared_distance = torch.full(
        (len(coordinates),), float("inf"), device=coordinates.device
    )
    for index in range(count):
        selected[index] = current
        squared_distance = torch.sum(
            (coordinates - coordinates[current : current + 1]) ** 2, dim=1
        )
        minimum_squared_distance = torch.minimum(
            minimum_squared_distance, squared_distance
        )
        current = int(torch.argmax(minimum_squared_distance).item())
    return selected


def _extract_chunked_depth(
    depth_extractor,
    video_path: Path,
    *,
    chunk_frames: int,
    overlap_frames: int,
):
    """Run stride-preserving depth inference in overlapping temporal tiles.

    SpatialTracker's geometry attention grows with the number of frames and a
    complete long rollout does not fit on a 32 GB device.  Every source frame
    is still inferred (no temporal subsampling); adjacent tiles are placed on
    a common depth scale with a robust affine fit and blended in their overlap.
    """

    import cv2
    import torch

    if chunk_frames < 2:
        raise ValueError("--depth-chunk-frames must be at least 2")
    if overlap_frames < 1 or overlap_frames >= chunk_frames:
        raise ValueError("depth chunk overlap must lie in [1, chunk_frames)")

    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"could not decode video frames: {video_path}")
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)

    assembled = None
    records: list[dict[str, object]] = []
    step = chunk_frames - overlap_frames
    for start in range(0, len(video), step):
        end = min(start + chunk_frames, len(video))
        if assembled is not None and end <= len(assembled):
            continue
        tile = depth_extractor.extract_depth_from_video(video[start:end]).detach().float().cpu()
        scale, shift = 1.0, 0.0
        overlap = 0 if assembled is None else min(len(assembled) - start, len(tile))
        if overlap > 0:
            previous = assembled[start : start + overlap]
            current = tile[:overlap]
            valid = (previous > 0) & (current > 0)
            if int(valid.sum()) < 1024:
                raise RuntimeError(
                    f"insufficient valid overlap depth in tile [{start},{end}): "
                    f"{int(valid.sum())} pixels"
                )
            prev_values = previous[valid]
            cur_values = current[valid]
            prev_q = torch.quantile(prev_values, torch.tensor([0.1, 0.5, 0.9]))
            cur_q = torch.quantile(cur_values, torch.tensor([0.1, 0.5, 0.9]))
            denominator = float(cur_q[2] - cur_q[0])
            if abs(denominator) > 1e-6:
                scale = float((prev_q[2] - prev_q[0]) / denominator)
            scale = float(np.clip(scale, 0.5, 2.0))
            shift = float(prev_q[1] - scale * cur_q[1])
            tile = scale * tile + shift
            tile[tile <= 0] = 0

            alpha = torch.linspace(0.0, 1.0, overlap).view(-1, 1, 1)
            current = tile[:overlap]
            previous_valid = previous > 0
            current_valid = current > 0
            blended = torch.where(
                previous_valid & current_valid,
                torch.lerp(previous, current, alpha),
                torch.where(previous_valid, previous, current),
            )
            assembled[start : start + overlap] = blended
            if overlap < len(tile):
                assembled = torch.cat([assembled, tile[overlap:]], dim=0)
        else:
            assembled = tile
        records.append(
            {
                "start_frame": start,
                "end_frame_exclusive": end,
                "overlap_frames": overlap,
                "depth_scale_to_previous": scale,
                "depth_shift_to_previous": shift,
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if end == len(video):
            break
    assert assembled is not None
    if len(assembled) != len(video):
        raise RuntimeError(f"chunked depth produced {len(assembled)} of {len(video)} frames")
    return assembled, records


def _append_terminal_hold(
    joint_trajectory: np.ndarray,
    ee_trajectory: np.ndarray,
    source_frame_indices: np.ndarray,
    *,
    source_fps: float,
    source_frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Represent the final frame's display interval without retiming motion.

    Video frame ``i`` is sampled at ``i / fps`` and the final frame remains on
    screen until ``frame_count / fps``.  CAE therefore appends a duplicate
    terminal pose at that boundary.  Moving the final sampled pose from
    ``(N-1)/fps`` to ``N/fps`` would silently slow the last motion segment.
    """

    indices = np.asarray(source_frame_indices, dtype=np.int64)
    if source_fps <= 0 or source_frame_count <= 0:
        raise ValueError("source FPS and frame count must be positive")
    if indices.ndim != 1 or not len(indices):
        raise ValueError("source frame indices must be a non-empty vector")
    if indices[0] != 0 or indices[-1] != source_frame_count - 1:
        raise ValueError("source frame indices must include both video endpoints")
    if len(joint_trajectory) != len(indices) or len(ee_trajectory) != len(indices):
        raise ValueError("trajectory and source-index lengths must match")

    timestamps = indices.astype(np.float64) / source_fps
    duration = source_frame_count / source_fps
    return (
        np.concatenate([joint_trajectory, joint_trajectory[-1:]], axis=0),
        np.concatenate([ee_trajectory, ee_trajectory[-1:]], axis=0),
        np.concatenate([indices, indices[-1:]], axis=0),
        np.concatenate([timestamps, np.asarray([duration], dtype=np.float64)]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dream2flow-repo", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--camera-name", default=None)
    parser.add_argument(
        "--object-name",
        default=None,
        help="Optional grounding phrase override; canonical object provenance remains in scene_data.yaml.",
    )
    parser.add_argument(
        "--video-role",
        choices=("reference", "imagination"),
        default="reference",
        help="Provenance role of scene-dir/rgb.mp4 in symmetric video evaluation.",
    )
    parser.add_argument("--source-model", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-mode", choices=("auto", "generate", "playback"), default="auto")
    parser.add_argument("--flow-output", default=None)
    parser.add_argument("--plan-output", default=None)
    parser.add_argument("--force-flow", action="store_true")
    parser.add_argument(
        "--flow-only",
        action="store_true",
        help=(
            "Export RGB-derived object flow and exit before the official "
            "Dream2Flow trajectory optimizer. Object flow is an intermediate "
            "artifact, not the formal CAE action trajectory."
        ),
    )
    parser.add_argument(
        "--depth-only",
        action="store_true",
        help=(
            "Generate and cache depth_frames.pt, then exit before loading "
            "segmentation and particle-tracking models."
        ),
    )
    parser.add_argument(
        "--region-only",
        action="store_true",
        help=(
            "Run first-frame GroundingDINO/SAM target selection and exit before "
            "expensive full-timeline depth or CoTracker inference."
        ),
    )
    parser.add_argument(
        "--promote-first-depth-to-initial",
        action="store_true",
        help=(
            "After generated depth is cached, replace initial_depth.pt with "
            "its first frame. Use only when simulator/sensor metric depth is "
            "unavailable, and share that same frame across both video roles."
        ),
    )
    parser.add_argument("--urdf-path", default="panda_description")
    parser.add_argument("--target-link-name", default="panda_hand_tcp")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--max-plan-timesteps", type=int, default=50)
    parser.add_argument(
        "--depth-temporal-stride",
        type=int,
        default=1,
        help=(
            "Run SpatialTracker on every Nth frame and interpolate depth back "
            "to the complete video timeline (useful on memory-limited GPUs)."
        ),
    )
    parser.add_argument(
        "--depth-chunk-frames",
        type=int,
        default=None,
        help=(
            "Optional overlapping temporal tile length for stride-preserving "
            "SpatialTracker inference on long videos."
        ),
    )
    parser.add_argument(
        "--depth-chunk-overlap",
        type=int,
        default=8,
        help="Overlap used with --depth-chunk-frames for depth-scale alignment.",
    )
    parser.add_argument(
        "--visualize-tracking",
        action="store_true",
        help="Save Dream2Flow's CoTracker overlay as track_vis.mp4.",
    )
    parser.add_argument(
        "--grounding-box-threshold",
        type=float,
        default=0.4,
        help="GroundingDINO box threshold (recorded in the exported metadata).",
    )
    parser.add_argument(
        "--grounding-dino-model-path",
        default="IDEA-Research/grounding-dino-base",
        help="Local directory or Hugging Face identifier for GroundingDINO.",
    )
    parser.add_argument(
        "--sam2-model-path",
        default="facebook/sam2-hiera-large",
        help="Local directory or Hugging Face identifier for SAM2.",
    )
    parser.add_argument(
        "--mask-erosion-kernel-size",
        type=int,
        default=11,
        help="Object-mask erosion kernel; use a smaller odd value for low-resolution videos.",
    )
    parser.add_argument("--target-box-fraction-min", type=float, default=0.03)
    parser.add_argument("--target-box-fraction-max", type=float, default=0.20)
    parser.add_argument(
        "--target-bbox-prior",
        type=float,
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        default=None,
        help=(
            "Optional first-frame target box shared by reference and imagination "
            "frontends. Certified simulator references derive it from frame-zero "
            "instance segmentation when omitted."
        ),
    )
    parser.add_argument(
        "--particle-ratio",
        type=float,
        default=0.01,
        help="Fraction of the eroded object mask sampled as CoTracker particles.",
    )
    parser.add_argument("--minimum-particles", type=int, default=64)
    parser.add_argument("--maximum-particles", type=int, default=1024)
    parser.add_argument(
        "--expected-cotracker-visibility-threshold",
        type=float,
        default=0.8,
        help="Fail closed unless the loaded CoTracker predictor uses this threshold.",
    )
    parser.add_argument(
        "--allow-pure-torch-fps-fallback",
        action="store_true",
        help=(
            "Allow the diagnostic pure-Torch FPS implementation when the official "
            "pyg-lib operator is unavailable. Formal benchmark runs must not use this."
        ),
    )
    args = parser.parse_args()

    dream_root = Path(args.dream2flow_repo).resolve()
    scene_dir = Path(args.scene_dir).resolve()
    if not (dream_root / "dream2flow").is_dir():
        raise FileNotFoundError(f"invalid Dream2Flow repository: {dream_root}")
    sys.path.insert(0, str(dream_root))

    import torch
    import yaml
    import cv2
    from PIL import Image
    from dream2flow.camera import CameraCalibration
    from dream2flow.flow.depth import (
        FileDepthExtractor,
        FileDepthExtractorConfig,
        SpaTrackerDepthExtractor,
        SpaTrackerDepthExtractorConfig,
    )
    from dream2flow.flow.generators.video_flow_generator import (
        VideoFlowGenerator,
        VideoFlowGeneratorConfig,
    )
    from dream2flow.flow.object_flow_result import ObjectFlowResult
    from dream2flow.flow.particle_trackers import OfflineParticleTrackerConfig
    from dream2flow.flow.region_selectors import (
        GroundedRegionSelector,
        GroundedRegionSelectorConfig,
    )
    from dream2flow.video.playback import PlaybackVideoSource

    generator_source = inspect.getsource(VideoFlowGenerator.generate)
    normalized_generator_source = re.sub(r"\s+", "", generator_source)
    if "video_frames.float()*255.0" not in normalized_generator_source:
        raise RuntimeError(
            "Dream2Flow must convert normalized [0,1] playback RGB to CoTracker's "
            "[0,255] input range; apply dream2flow_cotracker_input_range.patch"
        )
    if "min_num_particles" not in VideoFlowGeneratorConfig.__dataclass_fields__:
        raise RuntimeError(
            "Dream2Flow must support the audited minimum-particle contract; "
            "apply dream2flow_min64_particles.patch"
        )
    if args.minimum_particles < 1 or args.maximum_particles < args.minimum_particles:
        raise ValueError("particle bounds must satisfy 1 <= minimum <= maximum")

    try:
        import pyg_lib
        import torch_geometric
        import torch_geometric.typing
        from torch_geometric.nn import fps as pyg_fps

        if not torch_geometric.typing.WITH_FPS:
            raise ImportError("torch-geometric did not register the pyg-lib FPS operator")
        particle_sampler = "torch_geometric.nn.fps_random_start_false"
        particle_sampler_versions = {
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "pyg_lib": pyg_lib.__version__,
        }
    except (ImportError, OSError) as exc:
        if not args.allow_pure_torch_fps_fallback:
            raise RuntimeError(
                "The official torch_geometric.nn.fps backend is required. Install the "
                "matching pyg-lib wheel or pass --allow-pure-torch-fps-fallback only "
                "for a non-benchmark diagnostic run."
            ) from exc
        pyg_fps = None
        particle_sampler = "diagnostic_pure_torch_greedy_fps_centroid_seed_v1"
        particle_sampler_versions = {"torch": torch.__version__}

    class DeterministicVideoFlowGenerator(VideoFlowGenerator):
        def _fps(self, object_points, ratio):
            if pyg_fps is None:
                return _deterministic_farthest_point_indices(object_points, ratio)
            return pyg_fps(
                object_points.contiguous(),
                ratio=ratio,
                random_start=False,
            )

    class SizeConstrainedGroundedRegionSelector(GroundedRegionSelector):
        """Select the best DINO box inside the camera target-size contract."""

        def __init__(
            self,
            *selector_args,
            minimum_fraction,
            maximum_fraction,
            target_bbox_prior=None,
            target_bbox_prior_source=None,
            initial_component_region=None,
            **selector_kwargs,
        ):
            super().__init__(*selector_args, **selector_kwargs)
            self.minimum_fraction = minimum_fraction
            self.maximum_fraction = maximum_fraction
            self.target_bbox_prior = target_bbox_prior
            self.target_bbox_prior_source = target_bbox_prior_source
            self.selection_metadata = None
            self.initial_component_region = initial_component_region

        def _get_object_bbox(self, source_image, tracked_object_name):
            annotation = self.initial_component_region
            if annotation is not None:
                if annotation.get("schema_version") != "initial_component_box.v1" or annotation.get("frame") != 0:
                    raise ValueError("invalid initial component annotation")
                width, height = source_image.size
                if annotation.get("image_size_wh") != [width,height]:
                    raise ValueError("initial component annotation/camera size mismatch")
                bbox = [float(v) for v in annotation["bbox_xyxy"]]
                x0,y0,x1,y1=bbox
                if not np.isfinite(bbox).all() or not (0<=x0<x1<=width and 0<=y0<y1<=height):
                    raise ValueError("initial component box leaves the camera")
                if (x1-x0)*(y1-y0)<64:
                    raise ValueError("initial component box has insufficient pixels")
                self.selection_metadata = {
                    "selector":"renderer_initial_component_box_sam2_v1",
                    "object_name":tracked_object_name,
                    "target_bbox_prior_xyxy":bbox,
                    "target_bbox_prior_source":"shared_initial_scene_component_annotation",
                    "initial_component_annotation":annotation,
                    "camera_contract":"annotated_component_in_frame_min64_v1",
                    "selected":{"index":0,"bbox_xyxy":bbox,
                        "linear_fraction":max((x1-x0)/width,(y1-y0)/height),
                        "inside_component_frame_contract":True},
                    "candidates":[],
                    "future_reference_data_used":False,
                }
                return torch.tensor(bbox,dtype=torch.float32,device=self.device)
            prompt = self._format_object_name_for_grounding_dino(tracked_object_name)
            inputs = self._grounding_dino_processor(
                text=prompt,
                images=source_image,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self._grounding_dino_model(**inputs)
            result = self._grounding_dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.config.grounding_dino_box_threshold,
                text_threshold=self.config.grounding_dino_text_threshold,
                target_sizes=[source_image.size[::-1]],
            )[0]
            boxes = result["boxes"]
            scores = result["scores"]
            width, height = source_image.size
            candidates = []
            prior = self.target_bbox_prior
            for index, (box, score) in enumerate(zip(boxes, scores)):
                x_min, y_min, x_max, y_max = [float(value) for value in box]
                fraction = max((x_max - x_min) / width, (y_max - y_min) / height)
                prior_iou = None
                prior_center_distance = None
                if prior is not None:
                    px_min, py_min, px_max, py_max = prior
                    intersection_width = max(0.0, min(x_max, px_max) - max(x_min, px_min))
                    intersection_height = max(0.0, min(y_max, py_max) - max(y_min, py_min))
                    intersection = intersection_width * intersection_height
                    area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
                    prior_area = max(0.0, px_max - px_min) * max(0.0, py_max - py_min)
                    union = area + prior_area - intersection
                    prior_iou = intersection / union if union > 0 else 0.0
                    center_x = 0.5 * (x_min + x_max)
                    center_y = 0.5 * (y_min + y_max)
                    prior_center_x = 0.5 * (px_min + px_max)
                    prior_center_y = 0.5 * (py_min + py_max)
                    prior_center_distance = float(
                        np.hypot(center_x - prior_center_x, center_y - prior_center_y)
                        / np.hypot(width, height)
                    )
                candidates.append(
                    {
                        "index": index,
                        "score": float(score),
                        "bbox_xyxy": [x_min, y_min, x_max, y_max],
                        "linear_fraction": fraction,
                        "target_prior_iou": prior_iou,
                        "target_prior_center_distance": prior_center_distance,
                        "inside_camera_contract": (
                            self.minimum_fraction <= fraction <= self.maximum_fraction
                        ),
                    }
                )
            eligible = [item for item in candidates if item["inside_camera_contract"]]
            if not eligible:
                raise RuntimeError(
                    "GroundingDINO found no box inside the camera-size contract: "
                    + json.dumps(candidates, sort_keys=True)
                )
            if prior is None:
                selected = max(
                    eligible, key=lambda item: (item["score"], -item["index"])
                )
                selector_name = "grounding_dino_size_constrained_v1"
            else:
                selected = max(
                    eligible,
                    key=lambda item: (
                        item["target_prior_iou"],
                        -item["target_prior_center_distance"],
                        item["score"],
                        -item["index"],
                    ),
                )
                selector_name = "grounding_dino_size_spatial_prior_v1"
            self.selection_metadata = {
                "selector": selector_name,
                "object_name": tracked_object_name,
                "minimum_linear_fraction": self.minimum_fraction,
                "maximum_linear_fraction": self.maximum_fraction,
                "target_bbox_prior_xyxy": prior,
                "target_bbox_prior_source": self.target_bbox_prior_source,
                "selected": selected,
                "candidates": candidates,
            }
            return boxes[selected["index"]]

        def extract_region(self, output_dir, image, object_name):
            mask = super().extract_region(output_dir, image, object_name)
            if self.initial_component_region is not None:
                x0,y0,x1,y1=[int(v) for v in self.initial_component_region["bbox_xyxy"]]
                area=int(mask.sum().item())
                inside=int(mask[y0:y1,x0:x1].sum().item())
                fraction=inside/max(area,1)
                self.selection_metadata["sam2_mask_area"]=area
                self.selection_metadata["sam2_fraction_inside_annotation"]=fraction
                if area<64 or fraction<.75:
                    raise RuntimeError(f"SAM2 disagrees with the initial component annotation: area={area}, inside={fraction}")
            Path(output_dir, "region_candidates.json").write_text(
                json.dumps(self.selection_metadata, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            return mask

    files = {
        "video": scene_dir / "rgb.mp4",
        "image": scene_dir / "camera_rgb.png",
        "camera": scene_dir / "camera_calibration_info.json",
        "initial_depth": scene_dir / "initial_depth.pt",
        "scene": scene_dir / "scene_data.yaml",
        "depth_frames": scene_dir / "depth_frames.pt",
    }
    for key in ("video", "image", "camera", "initial_depth", "scene"):
        if not files[key].is_file():
            raise FileNotFoundError(f"scene is missing {key}: {files[key]}")
    flow_output = Path(args.flow_output).resolve() if args.flow_output else scene_dir / "object_flow_result.pt"
    plan_output = (
        Path(args.plan_output).resolve()
        if args.plan_output
        else scene_dir / "trajectory_optimization_plan.pt"
    )

    scene = yaml.safe_load(files["scene"].read_text(encoding="utf-8"))
    instruction = str(scene["instruction"])
    object_name = args.object_name or str(scene["object_name"])
    camera_payload = json.loads(files["camera"].read_text(encoding="utf-8"))
    camera_name = args.camera_name or (next(iter(camera_payload)) if len(camera_payload) == 1 else None)
    if camera_name is None:
        raise ValueError("--camera-name is required for multi-camera calibration")

    target_bbox_prior = (
        [float(value) for value in args.target_bbox_prior]
        if args.target_bbox_prior is not None
        else None
    )
    target_bbox_prior_source = "cli_shared_first_frame_bbox" if target_bbox_prior else None
    if target_bbox_prior is None and args.video_role == "reference":
        segmentation_path = scene_dir / "segmentation" / "0000.npy"
        encoding_path = scene_dir / "segmentation_encoding.json"
        if segmentation_path.is_file() and encoding_path.is_file():
            encoding = json.loads(encoding_path.read_text(encoding="utf-8"))
            target_id = int(encoding["target_instance_id"])
            target_mask = np.load(segmentation_path) == target_id
            rows, columns = np.where(target_mask)
            if len(rows):
                target_bbox_prior = [
                    float(columns.min()),
                    float(rows.min()),
                    float(columns.max() + 1),
                    float(rows.max() + 1),
                ]
                target_bbox_prior_source = (
                    "certified_reference_frame0_target_instance_bbox"
                )

    def make_region_selector():
        config = GroundedRegionSelectorConfig(
            grounding_dino_box_threshold=args.grounding_box_threshold,
            grounding_dino_model_path=args.grounding_dino_model_path,
            sam2_model_path=args.sam2_model_path,
        )
        return SizeConstrainedGroundedRegionSelector(
            config,
            device=args.device,
            minimum_fraction=args.target_box_fraction_min,
            maximum_fraction=args.target_box_fraction_max,
            target_bbox_prior=target_bbox_prior,
            target_bbox_prior_source=target_bbox_prior_source,
            initial_component_region=scene.get("initial_component_region"),
        )

    if args.region_only:
        if args.depth_only or args.flow_only:
            raise ValueError("--region-only cannot be combined with depth/flow-only modes")
        preflight_dir = scene_dir / "region_preflight"
        preflight_dir.mkdir(parents=True, exist_ok=True)
        start_image = torch.from_numpy(
            np.array(Image.open(files["image"]).convert("RGB"))
        ).float()
        start_image = (start_image / 255.0).to(args.device)
        selector = make_region_selector()
        selector.extract_region(str(preflight_dir), start_image, object_name)
        payload = {
            "schema_version": "dream2flow_region_preflight.v1",
            "passed": True,
            "video_role": args.video_role,
            "object_name": object_name,
            "selection": selector.selection_metadata,
            "depth_started": False,
            "flow_started": False,
        }
        (preflight_dir / "region_preflight.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if flow_output.is_file() and not args.force_flow:
        flow_result = ObjectFlowResult.load(str(flow_output), device=args.device)
        flow_source = "reused"
    else:
        camera = CameraCalibration.load(str(files["camera"]), device=args.device)
        start_image = torch.from_numpy(np.array(Image.open(files["image"]).convert("RGB"))).float()
        start_image = (start_image / 255.0).to(args.device)
        try:
            initial_depth = torch.load(files["initial_depth"], map_location=args.device, weights_only=True)
        except TypeError:
            initial_depth = torch.load(files["initial_depth"], map_location=args.device)
        video_frames = PlaybackVideoSource(str(files["video"]), device=args.device).generate_video(
            output_dir=str(scene_dir),
            text_prompt=instruction,
            start_image=start_image,
        )
        depth_mode = args.depth_mode
        if depth_mode == "auto":
            depth_mode = "playback" if files["depth_frames"].is_file() else "generate"
        if depth_mode == "playback":
            if args.depth_only:
                raise ValueError("--depth-only requires --depth-mode generate")
            if not files["depth_frames"].is_file():
                raise FileNotFoundError(files["depth_frames"])
            depth_extractor = FileDepthExtractor(
                FileDepthExtractorConfig(
                    depth_file_path=str(files["depth_frames"]),
                    resize_height=int(video_frames.shape[1]),
                    resize_width=int(video_frames.shape[2]),
                ),
                device=args.device,
            )
        else:
            if args.depth_temporal_stride != 1:
                raise ValueError(
                    "formal Dream2Flow object-flow extraction requires "
                    "--depth-temporal-stride 1"
                )
            depth_extractor = SpaTrackerDepthExtractor(
                SpaTrackerDepthExtractorConfig(
                    resize_height=int(video_frames.shape[1]),
                    resize_width=int(video_frames.shape[2]),
                ),
                device=args.device,
            )
            if args.depth_only:
                depth_chunk_records = None
                if args.depth_chunk_frames is None:
                    depth_frames = depth_extractor.extract_depth_from_file(
                        str(scene_dir), str(files["video"])
                    )
                else:
                    depth_frames, depth_chunk_records = _extract_chunked_depth(
                        depth_extractor,
                        files["video"],
                        chunk_frames=args.depth_chunk_frames,
                        overlap_frames=args.depth_chunk_overlap,
                    )
                files["depth_frames"].parent.mkdir(parents=True, exist_ok=True)
                torch.save(depth_frames.detach().cpu(), files["depth_frames"])
                if args.promote_first_depth_to_initial:
                    # Clone the frame view so torch.save serializes only the
                    # HxW tensor, not the full T-frame backing storage.
                    first_depth = depth_frames[0].detach().float().cpu().contiguous().clone()
                    if first_depth.ndim != 2:
                        raise ValueError(
                            "generated first-frame depth must be HxW, got "
                            f"{tuple(first_depth.shape)}"
                        )
                    target_height, target_width = start_image.shape[:2]
                    if first_depth.shape != (target_height, target_width):
                        first_depth = torch.nn.functional.interpolate(
                            first_depth[None, None],
                            size=(target_height, target_width),
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]
                    torch.save(first_depth, files["initial_depth"])
                metadata = {
                    "provider": "Dream2Flow/SpatialTrackerV2",
                    "conditioning_video": str(files["video"]),
                    "depth_output": str(files["depth_frames"]),
                    "depth_shape": list(depth_frames.shape),
                    "temporal_stride": args.depth_temporal_stride,
                    "temporal_chunk_frames": args.depth_chunk_frames,
                    "temporal_chunk_overlap": (
                        args.depth_chunk_overlap if args.depth_chunk_frames else None
                    ),
                    "temporal_chunk_records": depth_chunk_records,
                    "full_timeline_preserved": True,
                    "stage": "depth_only",
                    "first_depth_promoted_to_initial": bool(
                        args.promote_first_depth_to_initial
                    ),
                }
                files["depth_frames"].with_suffix(".json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(json.dumps(metadata, ensure_ascii=False, indent=2))
                return
        region_selector = make_region_selector()
        generator = DeterministicVideoFlowGenerator(
            depth_extractor=depth_extractor,
            device=args.device,
            config=VideoFlowGeneratorConfig(
                particle_ratio=args.particle_ratio,
                min_num_particles=args.minimum_particles,
                max_num_particles=args.maximum_particles,
                mask_erosion_kernel_size=args.mask_erosion_kernel_size,
                tracker=OfflineParticleTrackerConfig(
                    visualize_tracking=args.visualize_tracking
                ),
                region_selector=GroundedRegionSelectorConfig(
                    grounding_dino_box_threshold=args.grounding_box_threshold,
                    grounding_dino_model_path=args.grounding_dino_model_path,
                    sam2_model_path=args.sam2_model_path,
                ),
            ),
            region_selector=region_selector,
        )
        predictor_source = inspect.getsource(
            generator._particle_tracker._offline_cotracker._compute_sparse_tracks
        )
        threshold_match = re.search(r"\bthr\s*=\s*([0-9.]+)", predictor_source)
        loaded_visibility_threshold = (
            float(threshold_match.group(1)) if threshold_match else None
        )
        if loaded_visibility_threshold != args.expected_cotracker_visibility_threshold:
            raise RuntimeError(
                "loaded CoTracker visibility threshold does not match the audited "
                f"contract: loaded={loaded_visibility_threshold}, "
                f"expected={args.expected_cotracker_visibility_threshold}"
            )
        flow_result = generator.generate(
            output_dir=str(scene_dir),
            start_image=start_image,
            camera=camera,
            camera_name=camera_name,
            instruction=instruction,
            object_name=object_name,
            video_frames=video_frames,
            initial_depth=initial_depth,
            video_path=str(files["video"]),
        )
        flow_output.parent.mkdir(parents=True, exist_ok=True)
        flow_result.save(str(flow_output))
        if flow_result.pixel_trajectories is None:
            raise RuntimeError("Dream2Flow did not return auditable 2-D particle tracks")
        torch.save(
            flow_result.pixel_trajectories.detach().cpu(),
            scene_dir / "tracks_2d.pt",
        )
        flow_source = "generated"

    if args.flow_only:
        metadata = {
            "provider": "Dream2Flow",
            "evaluation_source": "rgb_video",
            "trajectory_source": "object_flow_particle_displacement",
            "video_role": args.video_role,
            "source_model": args.source_model,
            "action_head_used": False,
            "object_to_robot_ik_used": False,
            "conditioning_video": str(files["video"]),
            "instruction": instruction,
            "canonical_object_name": scene.get("canonical_object_name", object_name),
            "object_name": object_name,
            "camera_name": camera_name,
            "grounding_box_threshold": args.grounding_box_threshold,
            "mask_erosion_kernel_size": args.mask_erosion_kernel_size,
            "particle_ratio": args.particle_ratio,
            "minimum_particles": args.minimum_particles,
            "maximum_particles": args.maximum_particles,
            "depth_mode": args.depth_mode,
            "flow_source": flow_source,
            "flow_output": str(flow_output),
            "particle_sampler": particle_sampler,
            "particle_sampler_versions": particle_sampler_versions,
            "cotracker_rgb_input_range": [0.0, 255.0],
            "cotracker_visibility_threshold": args.expected_cotracker_visibility_threshold,
            "region_selector": (
                "grounding_dino_sam2_size_spatial_prior_v1"
                if target_bbox_prior is not None
                else "grounding_dino_sam2_size_constrained_v1"
            ),
            "target_bbox_prior_xyxy": target_bbox_prior,
            "target_bbox_prior_source": target_bbox_prior_source,
            "target_box_fraction_range": [
                args.target_box_fraction_min,
                args.target_box_fraction_max,
            ],
            "stage": "object_flow_only",
        }
        selection_path = scene_dir / "region_candidates.json"
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text())
            metadata["region_selector"] = selection["selector"]
            metadata["target_bbox_prior_xyxy"] = selection.get("target_bbox_prior_xyxy")
            metadata["target_bbox_prior_source"] = selection.get("target_bbox_prior_source")
            metadata["target_locator_contract"] = selection.get("camera_contract")
            if scene.get("initial_component_region") is not None:
                metadata["initial_component_region"] = scene["initial_component_region"]
                metadata["target_box_fraction_range"] = None
                metadata["future_reference_data_used"] = False
        flow_output.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return

    from dream2flow.planner.trajectory_optimizer import (
        TrajectoryOptimizer,
        TrajectoryOptimizerConfig,
    )

    planner = TrajectoryOptimizer(
        TrajectoryOptimizerConfig(
            urdf_path=args.urdf_path,
            target_link_name=args.target_link_name,
            max_iterations=args.max_iterations,
            visualize=False,
            max_num_timesteps_for_optimization=args.max_plan_timesteps,
        )
    )
    initial_joints = np.asarray(scene.get("robot_start_joints"), dtype=np.float64)
    if initial_joints.shape != (planner.num_actuated_joints,):
        raise ValueError(
            f"scene robot_start_joints has shape {initial_joints.shape}, "
            f"planner expects {(planner.num_actuated_joints,)}"
        )
    initial_pose = np.asarray(planner.get_end_effector_pose(initial_joints), dtype=np.float64)
    plan = planner.plan(
        flow_result=flow_result,
        initial_joints=initial_joints,
        initial_pose=initial_pose,
    )
    capture = cv2.VideoCapture(str(files["video"]))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if source_fps <= 0 or source_frame_count <= 0:
        raise RuntimeError(f"could not determine source video timing: {files['video']}")
    source_duration_sec = source_frame_count / source_fps
    flow_timestep_count = int(flow_result.flow.position.shape[0])
    flow_step_size = max(1, flow_timestep_count // args.max_plan_timesteps)
    raw_source_frame_indices = np.arange(
        0, flow_timestep_count, flow_step_size, dtype=np.int64
    )
    if raw_source_frame_indices.shape != (len(plan.ee_trajectory),):
        raise ValueError(
            "Dream2Flow plan/source timeline mismatch: "
            f"{raw_source_frame_indices.shape} vs {plan.ee_trajectory.shape}"
        )
    if flow_timestep_count != source_frame_count:
        raise ValueError(
            "Dream2Flow flow/video timeline mismatch: "
            f"flow={flow_timestep_count} video={source_frame_count}"
        )
    joint_trajectory = np.asarray(plan.joint_trajectory)
    ee_trajectory = np.asarray(plan.ee_trajectory)
    source_frame_indices = raw_source_frame_indices
    timestamps = source_frame_indices.astype(np.float64) / source_fps
    flow_valid_mask = np.asarray(
        flow_result.flow.valid_mask.detach().cpu(), dtype=bool
    )
    observed_source_frames = np.any(flow_valid_mask, axis=1)
    observed_frame_mask = observed_source_frames[source_frame_indices]
    plan_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "joint_trajectory": torch.from_numpy(joint_trajectory),
            "ee_trajectory": torch.from_numpy(ee_trajectory),
            "source_frame_indices": torch.from_numpy(source_frame_indices),
            "source_video_fps": source_fps,
            "source_video_frame_count": source_frame_count,
            "timestamps": torch.from_numpy(timestamps),
            "observed_frame_mask": torch.from_numpy(observed_frame_mask),
        },
        plan_output,
    )
    metadata = {
        "provider": "Dream2Flow",
        "evaluation_source": "rgb_video",
        "video_role": args.video_role,
        "source_model": args.source_model,
        "action_head_used": False,
        "conditioning_video": str(files["video"]),
        "instruction": instruction,
        "canonical_object_name": scene.get("canonical_object_name", object_name),
        "object_name": object_name,
        "camera_name": camera_name,
        "grounding_box_threshold": args.grounding_box_threshold,
        "mask_erosion_kernel_size": args.mask_erosion_kernel_size,
        "flow_source": flow_source,
        "flow_output": str(flow_output),
        "plan_output": str(plan_output),
        "joint_trajectory_shape": list(joint_trajectory.shape),
        "ee_trajectory_shape": list(ee_trajectory.shape),
        "raw_dream2flow_plan_timestep_count": int(len(plan.ee_trajectory)),
        "trajectory_source": "official_dream2flow_trajectory_optimizer_ee_trajectory",
        "object_flow_role": "intermediate_input_and_visibility_evidence_only",
        "terminal_hold_added": False,
        "flow_step_size": flow_step_size,
        "source_frame_indices": source_frame_indices.tolist(),
        "observed_frame_mask": observed_frame_mask.tolist(),
        "source_video_fps": source_fps,
        "source_video_frame_count": source_frame_count,
        "source_duration_sec": source_duration_sec,
        "timestamps_sec": timestamps.tolist(),
        "initial_pose_xyzw": initial_pose.tolist(),
    }
    plan_output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
