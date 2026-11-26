"""Helpers for preparing wall-art texture crops before Code_Object runs."""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import threading

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scenecode.agent_utils.geometry_generation_server.cuda_env_setup import (
    ensure_cuda_env,
)

console_logger = logging.getLogger(__name__)

DEFAULT_FRAME_PROMPTS = (
    "framed artwork",
    "picture frame",
    "framed painting",
    "framed poster",
)

_SAM3_PROCESSOR_CACHE: dict[str, Any] = {}
_SAM3_PROCESSOR_LOCK = threading.Lock()


@dataclass(frozen=True)
class WallArtPreprocessResult:
    """Artifacts produced for wall-art material generation."""

    picture_crop_path: Path
    frame_mask_path: Path | None
    summary_path: Path
    used_fallback: bool


def _ensure_sam3_import_path(repo_root: Path) -> None:
    sam3_path = repo_root / "external" / "SAM3"
    if str(sam3_path) not in sys.path:
        sys.path.insert(0, str(sam3_path))


def _load_sam3_processor(repo_root: Path, checkpoint_path: Path) -> Any:
    resolved_checkpoint = str(checkpoint_path.resolve())
    with _SAM3_PROCESSOR_LOCK:
        cached = _SAM3_PROCESSOR_CACHE.get(resolved_checkpoint)
        if cached is not None:
            return cached

        ensure_cuda_env()
        _ensure_sam3_import_path(repo_root)

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        console_logger.info("Loading SAM3 image model from %s", checkpoint_path)
        model = build_sam3_image_model(checkpoint_path=resolved_checkpoint)
        processor = Sam3Processor(model)
        _SAM3_PROCESSOR_CACHE[resolved_checkpoint] = processor
        return processor


def _tensor_to_numpy(value: Any) -> Any:
    import torch

    if not torch.is_tensor(value):
        return value

    value = value.detach().cpu()
    if value.dtype == torch.bfloat16:
        value = value.to(dtype=torch.float32)
    return value.numpy()


def _predict_prompt(
    image,
    sam3_processor: Any,
    prompt: str,
    threshold: float,
) -> dict[str, Any]:
    import numpy as np
    import torch

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        inference_state = sam3_processor.set_image(image)
        inference_state = sam3_processor.set_text_prompt(
            state=inference_state,
            prompt=prompt,
        )

    masks = _tensor_to_numpy(inference_state["masks"])
    scores = _tensor_to_numpy(inference_state["scores"])
    boxes = _tensor_to_numpy(inference_state.get("boxes"))

    if len(scores) == 0:
        raise ValueError(f"No instances detected for prompt '{prompt}'")

    best_idx = int(np.argmax(scores))
    raw_mask = masks[best_idx]
    while raw_mask.ndim > 2:
        raw_mask = raw_mask.squeeze(axis=0)

    mask = raw_mask
    if mask.dtype != np.uint8:
        mask = (mask > threshold).astype(np.uint8)

    box = None
    if boxes is not None and len(boxes) > best_idx:
        box = boxes[best_idx].tolist()

    return {
        "prompt": prompt,
        "score": float(scores[best_idx]),
        "mask": mask,
        "box": box,
    }


def _run_best_prompt(
    image,
    sam3_processor: Any,
    prompts: tuple[str, ...],
    threshold: float,
) -> dict[str, Any]:
    best_result: dict[str, Any] | None = None
    errors: list[str] = []

    for prompt in prompts:
        try:
            result = _predict_prompt(
                image=image,
                sam3_processor=sam3_processor,
                prompt=prompt,
                threshold=threshold,
            )
            if best_result is None or result["score"] > best_result["score"]:
                best_result = result
        except ValueError as exc:
            errors.append(str(exc))

    if best_result is None:
        raise ValueError("; ".join(errors) or "SAM could not find a valid mask")

    return best_result


def _mask_to_bbox(mask) -> tuple[int, int, int, int]:
    import numpy as np

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask is empty")

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _inset_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    inset_ratio: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    dx = max(1, int(round(width * inset_ratio)))
    dy = max(1, int(round(height * inset_ratio)))

    x0 = min(max(0, x0 + dx), image_size[0] - 1)
    y0 = min(max(0, y0 + dy), image_size[1] - 1)
    x1 = max(x0 + 1, min(image_size[0], x1 - dx))
    y1 = max(y0 + 1, min(image_size[1], y1 - dy))
    return x0, y0, x1, y1


def _write_summary(summary_path: Path, payload: dict[str, Any]) -> None:
    summary_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def prepare_wall_art_reference_assets(
    *,
    image_path: Path,
    output_dir: Path,
    sam3_checkpoint: Path,
    frame_prompts: tuple[str, ...] = DEFAULT_FRAME_PROMPTS,
    threshold: float = 0.5,
    frame_inset_ratio: float = 0.08,
) -> WallArtPreprocessResult:
    """Create a cropped picture texture for a wall-art candidate.

    The original image remains the structural reference for Code_Object. This helper
    only prepares the front-face texture expected by the WallArt constructor prompt.
    """
    from PIL import Image

    wall_art_dir = output_dir / 'wall_art'
    wall_art_dir.mkdir(parents=True, exist_ok=True)
    picture_crop_path = wall_art_dir / 'picture_crop.png'
    frame_mask_path = wall_art_dir / 'frame_mask.png'
    summary_path = wall_art_dir / 'frame_summary.json'

    def _fallback(exc: Exception) -> WallArtPreprocessResult:
        console_logger.warning(
            'Falling back to original image for wall-art texture crop %s: %s',
            image_path,
            exc,
        )
        image = Image.open(image_path).convert('RGB')
        image.save(picture_crop_path)
        payload = {
            'image_path': str(image_path),
            'sam3_checkpoint': str(sam3_checkpoint),
            'frame_prompts': list(frame_prompts),
            'threshold': threshold,
            'frame_inset_ratio': frame_inset_ratio,
            'used_fallback': True,
            'error': str(exc),
            'output_files': {
                'picture_crop': str(picture_crop_path),
                'frame_mask': None,
            },
        }
        _write_summary(summary_path, payload)
        return WallArtPreprocessResult(
            picture_crop_path=picture_crop_path,
            frame_mask_path=None,
            summary_path=summary_path,
            used_fallback=True,
        )

    try:
        if not image_path.exists():
            raise FileNotFoundError(f'Wall-art source image not found: {image_path}')
        if not sam3_checkpoint.exists():
            raise FileNotFoundError(f'SAM3 checkpoint not found: {sam3_checkpoint}')

        image = Image.open(image_path).convert('RGB')
        repo_root = Path(__file__).resolve().parents[2]
        sam3_processor = _load_sam3_processor(repo_root, sam3_checkpoint)
        frame_result = _run_best_prompt(
            image=image,
            sam3_processor=sam3_processor,
            prompts=frame_prompts,
            threshold=threshold,
        )
        frame_mask = frame_result['mask']
        frame_bbox = _mask_to_bbox(frame_mask)
        frame_crop_bbox = _inset_bbox(
            frame_bbox,
            image.size,
            inset_ratio=frame_inset_ratio,
        )
        frame_crop = image.crop(frame_crop_bbox)
        frame_crop.save(picture_crop_path)
        Image.fromarray((frame_mask * 255).astype('uint8'), mode='L').save(frame_mask_path)
        payload = {
            'image_path': str(image_path),
            'sam3_checkpoint': str(sam3_checkpoint),
            'frame_prompt': frame_result['prompt'],
            'frame_score': frame_result['score'],
            'frame_bbox_xyxy': list(frame_bbox),
            'frame_crop_bbox_xyxy': list(frame_crop_bbox),
            'used_fallback': False,
            'output_files': {
                'picture_crop': str(picture_crop_path),
                'frame_mask': str(frame_mask_path),
            },
        }
        _write_summary(summary_path, payload)
        return WallArtPreprocessResult(
            picture_crop_path=picture_crop_path,
            frame_mask_path=frame_mask_path,
            summary_path=summary_path,
            used_fallback=False,
        )
    except Exception as exc:
        return _fallback(exc)
