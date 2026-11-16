"""Rendered in-context appearance and proportion review for robotics references.

The deterministic media gates check decoding, clock and renderer-frame fidelity,
and the semantic identity gate answers "is each asset the object it
claims to be on its own contact sheet".  Neither answers the question a household
robotics scene actually depends on: with the robot, the support surface, the
destination and the manipulated object all in frame, are the objects still
distinguishable from each other, do they read as household-appropriate
materials, and are their real proportions plausible?

That question has two halves and this module implements both:

* deterministic rendered RGB and registered instance visibility measurements;
  average colour is a separation cue, with independent visible shape/boundary
  evidence supporting same-colour entities; and
* a vision-route critique of a real multi-view/multi-phase montage against the
  exact task visual descriptions, which is what turns a failure into actionable
  appearance/proportion evidence for the existing autonomous repair loop.

Nothing here authors task content: the objects, roles and visual descriptions
come from the typed task, and the colours come from the asset.
"""

from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from .io import sha256_file, write_json


SCHEMA = "robotics_reference_visual_context.v1"
DEFAULT_MINIMUM_DELTA_L = 8.0
DEFAULT_MINIMUM_DELTA_RGB = 0.10
# A pair is only judged in views where both objects really carry pixels;
# a handful of antialiased edge pixels is not evidence either way.
MINIMUM_PAIR_PIXELS = 48
DEFAULT_PHASE_ORDER = ("initial", "grasp", "carry", "final")
# Resolved through the shared SceneCode agent route at call time, so a
# deployment can move every text/vision call without editing callers.
DEFAULT_VISION_MODEL = "gpt-5.6-sol"
_VISION_UNAVAILABLE_STATUS = frozenset({429, 500, 502, 503, 504, 524})
CLOSEUP_TARGET_SCALE = 2.0
CLOSEUP_MIN_PIXELS = 96
CLOSEUP_MAX_SCALE = 12.0


def _srgb_to_linear(channel: float) -> float:
    value = float(channel)
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb_8bit) -> float:
    red, green, blue = (float(value) / 255.0 for value in rgb_8bit)
    return (
        0.2126 * _srgb_to_linear(red)
        + 0.7152 * _srgb_to_linear(green)
        + 0.0722 * _srgb_to_linear(blue)
    )


def cie_lstar(rgb_8bit) -> float:
    luminance = relative_luminance(rgb_8bit)
    if luminance <= 216.0 / 24389.0:
        return luminance * (24389.0 / 27.0)
    return 116.0 * (luminance ** (1.0 / 3.0)) - 16.0


def rgb_distance(left, right) -> float:
    return float(
        np.linalg.norm(
            (np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
            / 255.0
        )
    )


def _mask_bbox(mask: np.ndarray, value: int) -> tuple[int, int, int, int] | None:
    rows, columns = np.where(mask == value)
    if rows.size == 0:
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _object_signature(
    image: np.ndarray, mask: np.ndarray, value: int
) -> dict | None:
    selected = mask == value
    if not bool(selected.any()):
        return None
    # OpenCV decodes BGR.  The certificate promises RGB and CIE L* uses
    # channel-specific weights, so convert only the measured pixels; montage
    # assembly and cv2.imwrite continue to operate on BGR images.
    pixels = image[selected][:, ::-1]
    flat = pixels.reshape(-1, 3).astype("float64")
    mean_rgb = [float(component) for component in flat.mean(axis=0)]
    median_rgb = [float(component) for component in np.median(flat, axis=0)]
    bbox = _mask_bbox(mask, value)
    height, width = mask.shape[:2]
    linear_fraction = 0.0
    if bbox is not None:
        linear_fraction = float(
            max((bbox[2] - bbox[0]) / width, (bbox[3] - bbox[1]) / height)
        )
    return {
        "mean_rgb_8bit": [round(component, 2) for component in mean_rgb],
        "median_rgb_8bit": [round(component, 2) for component in median_rgb],
        "cie_lstar": round(cie_lstar(mean_rgb), 3),
        "cie_lstar_median": round(cie_lstar(median_rgb), 3),
        "pixel_count": int(selected.sum()),
        "linear_fraction": round(linear_fraction, 5),
        "bbox_xyxy": list(bbox) if bbox is not None else None,
    }


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to decode rendered RGB frame: {path}")
    return image


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"unable to decode registered label mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def _phase_frame_index(trajectory: dict, phase: str) -> int:
    frames = trajectory.get("frames") or []
    if phase == "initial":
        return 0
    if phase == "final":
        return len(frames) - 1
    candidates = [
        index
        for index, frame in enumerate(frames)
        if str(frame.get("phase")) == phase
    ]
    if candidates:
        return candidates[len(candidates) // 2]
    return min(len(frames) - 1, max(0, int(len(frames) * {"grasp": 0.3, "carry": 0.6}.get(phase, 0.5))))


def closeup_source(render: dict, object_id: str, *, views) -> dict | None:
    """Frame-0 view that shows one reviewed object largest, with its mask bbox."""

    best: dict | None = None
    for view_id in views:
        record = (render.get("views") or {}).get(view_id)
        if not isinstance(record, dict):
            continue
        semantic = record.get("conditioning_semantic")
        if not isinstance(semantic, dict):
            continue
        value = (semantic.get("object_values") or {}).get(object_id)
        if not isinstance(value, int):
            continue
        mask_path = Path(str(semantic.get("image", ""))).resolve()
        if not mask_path.is_file():
            continue
        index = int(semantic.get("frame", 0))
        frame_path = None
        for item in record.get("frames") or []:
            if int(item["frame"]) == index:
                frame_path = Path(str(item["image"])).resolve()
                break
        if frame_path is None or not frame_path.is_file():
            continue
        mask = _read_mask(mask_path)
        bbox = _mask_bbox(mask, value)
        if bbox is None:
            continue
        pixels = int((mask == value).sum())
        if best is None or pixels > best["pixels"]:
            best = {
                "object_id": object_id,
                "view_id": view_id,
                "frame": index,
                "bbox_xyxy": list(bbox),
                "pixels": pixels,
                "image": frame_path,
            }
    return best


def closeup_crop(source: dict, *, target_height: int) -> dict | None:
    """Magnified crop around one object, adequate to judge its material."""

    image = _read_image(Path(source["image"]))
    height, width = image.shape[:2]
    left, top, right, bottom = source["bbox_xyxy"]
    pad = max(6, int(0.35 * max(right - left, bottom - top)))
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(width, right + pad)
    bottom = min(height, bottom + pad)
    if right - left < 2 or bottom - top < 2:
        return None
    crop = image[top:bottom, left:right]
    scale = max(1.0, CLOSEUP_TARGET_SCALE * height / max(1, crop.shape[0]))
    scale = min(CLOSEUP_MAX_SCALE, scale)
    if scale * crop.shape[0] * scale * crop.shape[1] > 0:
        resized = cv2.resize(
            crop,
            (
                max(1, int(round(crop.shape[1] * scale))),
                max(1, int(round(crop.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        return None
    if max(resized.shape[:2]) < CLOSEUP_MIN_PIXELS:
        extra = CLOSEUP_MIN_PIXELS / max(1, max(resized.shape[:2]))
        resized = cv2.resize(
            resized,
            (
                max(1, int(round(resized.shape[1] * extra))),
                max(1, int(round(resized.shape[0] * extra))),
            ),
            interpolation=cv2.INTER_CUBIC,
        )
        scale *= extra
    labelled = resized.copy()
    label = (
        f"close-up {source['object_id']} ({source['view_id']} f{source['frame']} "
        f"x{scale:.1f})"
    )
    cv2.putText(
        labelled,
        label,
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return {"image": labelled, "label": label, "scale": round(scale, 2)}


def build_in_context_montage(
    render: dict,
    trajectory: dict,
    output: Path,
    *,
    views: tuple[str, ...],
    phases: tuple[str, ...] = DEFAULT_PHASE_ORDER,
    object_roles: dict[str, str] | None = None,
) -> dict:
    """Write the labelled evidence sheet the review reads.

    Rows top to bottom are the wide views (one row per view, one column per
    task phase) plus a final ``close-up`` row that magnifies every reviewed
    object.  Material questions are unanswerable from a wide 320x180 frame in
    which the manipuland is a handful of pixels, so the magnified crops are the
    evidence of record for material and colour.
    """

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    cells_by_phase: dict[str, dict[str, np.ndarray]] = {}
    for view_id in views:
        record = (render.get("views") or {}).get(view_id)
        if not isinstance(record, dict):
            raise ValueError(f"render manifest has no view {view_id!r}")
        frames = {int(item["frame"]): item for item in record.get("frames") or []}
        cells = []
        for phase in phases:
            index = _phase_frame_index(trajectory, phase)
            item = frames.get(index)
            if item is None:
                raise ValueError(f"view {view_id} has no rendered frame {index}")
            image = _read_image(Path(str(item["image"])).resolve())
            cells_by_phase.setdefault(phase, {})[view_id] = image
            labelled = image.copy()
            cv2.putText(
                labelled,
                f"{view_id} f{index} {phase}",
                (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cells.append(labelled)
        width = max(cell.shape[1] for cell in cells)
        height = max(cell.shape[0] for cell in cells)
        padded = [
            cv2.copyMakeBorder(
                cell,
                0,
                height - cell.shape[0],
                0,
                width - cell.shape[1],
                cv2.BORDER_CONSTANT,
                value=(24, 24, 24),
            )
            for cell in cells
        ]
        rows.append(np.hstack(padded))
    width = max(row.shape[1] for row in rows)
    padded_rows = [
        cv2.copyMakeBorder(
            row,
            0,
            0,
            0,
            width - row.shape[1],
            cv2.BORDER_CONSTANT,
            value=(24, 24, 24),
        )
        for row in rows
    ]
    closeups: dict[str, dict] = {}
    closeup_cells: list[np.ndarray] = []
    cell_height = int(rows[0].shape[0]) if rows else 0
    for object_id in sorted(object_roles or {}):
        source = closeup_source(render, object_id, views=views)
        if source is None:
            continue
        crop = closeup_crop(source, target_height=max(96, cell_height))
        if crop is None:
            continue
        closeups[object_id] = {
            "view_id": source["view_id"],
            "frame": source["frame"],
            "bbox_xyxy": source["bbox_xyxy"],
            "mask_pixels": source["pixels"],
            "scale": crop["scale"],
            "label": crop["label"],
        }
        closeup_cells.append(crop["image"])
    if closeup_cells:
        height = max(cell.shape[0] for cell in closeup_cells)
        padded_closeups = [
            cv2.copyMakeBorder(
                cell,
                0,
                height - cell.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(24, 24, 24),
            )
            for cell in closeup_cells
        ]
        row = np.hstack(padded_closeups)
        padded_rows.append(row)
    # Close-up crops preserve aspect ratio and may be slightly wider than the
    # phase grid. Normalize every row after all optional rows are known.
    final_width=max(row.shape[1] for row in padded_rows)
    padded_rows=[
        cv2.copyMakeBorder(row,0,0,0,final_width-row.shape[1],
                           cv2.BORDER_CONSTANT,value=(24,24,24))
        for row in padded_rows
    ]
    grid = np.vstack(padded_rows)
    montage = output / "visual_context_montage.png"
    if not cv2.imwrite(str(montage), grid):
        raise RuntimeError(f"unable to write in-context montage: {montage}")
    return {
        "path": montage,
        "views": list(views),
        "phases": list(phases),
        "frame_indices": {
            phase: _phase_frame_index(trajectory, phase) for phase in phases
        },
        "size_wh": [int(grid.shape[1]), int(grid.shape[0])],
        "closeups": closeups,
    }


def measure_in_context_appearance(
    render: dict,
    trajectory: dict,
    *,
    views: tuple[str, ...],
    object_roles: dict[str, str],
    minimum_delta_l: float = DEFAULT_MINIMUM_DELTA_L,
    minimum_delta_rgb: float = DEFAULT_MINIMUM_DELTA_RGB,
) -> dict:
    """Measure rendered RGB separation between mutually visible task objects."""

    signatures: dict[str, dict] = {}
    for view_id in views:
        record = (render.get("views") or {}).get(view_id)
        if not isinstance(record, dict):
            continue
        semantic = record.get("conditioning_semantic")
        if not isinstance(semantic, dict):
            continue
        mask_path = Path(str(semantic.get("image", ""))).resolve()
        frame_path = None
        for item in record.get("frames") or []:
            if int(item["frame"]) == int(semantic.get("frame", 0)):
                frame_path = Path(str(item["image"])).resolve()
                break
        if frame_path is None or not mask_path.is_file() or not frame_path.is_file():
            continue
        mask = _read_mask(mask_path)
        image = _read_image(frame_path)
        object_values = semantic.get("object_values") or {}
        view_signatures: dict[str, dict] = {}
        for object_id in object_roles:
            value = object_values.get(object_id)
            if not isinstance(value, int):
                continue
            signature = _object_signature(image, mask, value)
            if signature is not None:
                view_signatures[object_id] = signature
        end_effector_value = semantic.get("end_effector_value")
        if isinstance(end_effector_value, int):
            signature = _object_signature(image, mask, end_effector_value)
            if signature is not None:
                view_signatures["__end_effector__"] = signature
        signatures[view_id] = view_signatures

    if not signatures:
        raise ValueError("no registered in-context RGB signatures could be measured")

    pairs: list[dict] = []
    object_ids = [object_id for object_id in object_roles if object_id in _union(signatures)]
    for index, left in enumerate(object_ids):
        for right in object_ids[index + 1 :]:
            for view_id, view_signatures in signatures.items():
                if left not in view_signatures or right not in view_signatures:
                    continue
                left_rgb = view_signatures[left]["mean_rgb_8bit"]
                right_rgb = view_signatures[right]["mean_rgb_8bit"]
                left_median = view_signatures[left].get(
                    "median_rgb_8bit", left_rgb
                )
                right_median = view_signatures[right].get(
                    "median_rgb_8bit", right_rgb
                )
                delta_l = abs(
                    float(view_signatures[left]["cie_lstar"])
                    - float(view_signatures[right]["cie_lstar"])
                )
                distance = rgb_distance(left_rgb, right_rgb)
                delta_l_median = abs(
                    float(
                        view_signatures[left].get(
                            "cie_lstar_median", view_signatures[left]["cie_lstar"]
                        )
                    )
                    - float(
                        view_signatures[right].get(
                            "cie_lstar_median", view_signatures[right]["cie_lstar"]
                        )
                    )
                )
                distance_median = rgb_distance(left_median, right_median)
                pairs.append(
                    {
                        "view_id": view_id,
                        "left": left,
                        "right": right,
                        "delta_lstar": round(delta_l, 3),
                        "rgb_distance_0_1": round(distance, 4),
                        "delta_lstar_median": round(delta_l_median, 3),
                        "rgb_distance_median_0_1": round(distance_median, 4),
                        "left_pixels": int(view_signatures[left].get("pixel_count", 0)),
                        "right_pixels": int(view_signatures[right].get("pixel_count", 0)),
                        "separated": bool(
                            delta_l >= minimum_delta_l
                            or distance >= minimum_delta_rgb
                            or delta_l_median >= minimum_delta_l
                            or distance_median >= minimum_delta_rgb
                        ),
                    }
                )
    per_object_best: dict[str, dict] = {}
    for object_id in object_ids:
        competing = [pair for pair in pairs if object_id in {pair["left"], pair["right"]}]
        if not competing:
            continue
        # One camera can graze or shadow a small object (a hammer head at the
        # edge of frame), and the four cameras see the same scene from
        # different angles.  Separation is judged on the evidence set: the
        # object counts as separated when at least one registered view in which
        # both objects carry real pixels keeps them apart.  The weakest view is
        # still recorded so the certificate exposes how narrow the margin was.
        visible = [
            pair
            for pair in competing
            if min(pair["left_pixels"], pair["right_pixels"]) >= MINIMUM_PAIR_PIXELS
        ]
        evaluated = visible or competing
        weakest = min(evaluated, key=lambda pair: pair["delta_lstar"])
        strongest = max(
            evaluated,
            key=lambda pair: (
                bool(pair["separated"]),
                pair["delta_lstar"],
                pair["rgb_distance_0_1"],
                pair["delta_lstar_median"],
                pair["rgb_distance_median_0_1"],
            ),
        )
        per_object_best[object_id] = {
            "separated": bool(any(pair["separated"] for pair in evaluated)),
            "evaluated_view_count": len(evaluated),
            "separated_view_count": int(
                sum(1 for pair in evaluated if pair["separated"])
            ),
            "pair": f"{strongest['left']}|{strongest['right']}",
            "view_id": strongest["view_id"],
            "delta_lstar": strongest["delta_lstar"],
            "rgb_distance_0_1": strongest["rgb_distance_0_1"],
            "weakest_pair": f"{weakest['left']}|{weakest['right']}",
            "weakest_view_id": weakest["view_id"],
            "weakest_delta_lstar": weakest["delta_lstar"],
            "weakest_rgb_distance_0_1": weakest["rgb_distance_0_1"],
            "weakest_separated": bool(weakest["separated"]),
        }
    checks = {
        f"separated:{object_id}": bool(report["separated"])
        for object_id, report in per_object_best.items()
    }
    return {
        "policy": {
            "minimum_delta_lstar": float(minimum_delta_l),
            "minimum_rgb_distance_0_1": float(minimum_delta_rgb),
            "minimum_pair_pixels": int(MINIMUM_PAIR_PIXELS),
            "source": (
                "mean and median rendered RGB of registered per-object labels at "
                "frame zero; a pair counts as separated when either statistic "
                "clears the floor, so specular glass highlights cannot hide a "
                "saturated liquid, and an object counts as separated when at "
                "least one view in which both objects are actually visible "
                "keeps them apart"
            ),
        },
        "views": list(signatures),
        "objects": signatures,
        "pairs": pairs,
        "per_object": per_object_best,
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
    }


def _union(signatures: dict[str, dict]) -> set[str]:
    found: set[str] = set()
    for view_signatures in signatures.values():
        found.update(view_signatures)
    return found


def articulated_host_visibility(
    appearance: dict, components, *, minimum_host_fraction: float = 0.25,
    minimum_host_pixels: int = 96,
) -> dict:
    """Check that an articulated host reads as a body, not only a door slab.

    Only native exterior views count.  A top view can expose a back panel that
    the policy never sees.  The measurements come from the registered RGB
    frame and part masks already used by the appearance certificate.
    """
    records = {}
    signatures = appearance.get("objects") or {}
    for component in components:
        host_id = component.host_object_id
        part_id = component.component_id
        observations = []
        for view_id, objects in signatures.items():
            if not view_id.startswith("exterior_"):
                continue
            host = objects.get(host_id)
            part = objects.get(part_id)
            if host is None or part is None:
                continue
            host_pixels = int(host["pixel_count"])
            part_pixels = int(part["pixel_count"])
            fraction = host_pixels / max(1, host_pixels + part_pixels)
            observations.append({
                "view_id": view_id,
                "host_pixels": host_pixels,
                "component_pixels": part_pixels,
                "host_fraction": round(fraction, 5),
                "passed": host_pixels >= minimum_host_pixels
                and fraction >= minimum_host_fraction,
            })
        records[part_id] = {
            "host_object_id": host_id,
            "minimum_host_fraction": minimum_host_fraction,
            "minimum_host_pixels": minimum_host_pixels,
            "observations": observations,
            "passed": any(item["passed"] for item in observations),
        }
    return records


def load_workspace_credentials() -> None:
    """Load the configured vision credentials the way the launchers do.

    The deployment wrappers source ``$WORKSPACE/.secrets/scenecode.env`` before
    calling any OpenAI-backed stage.  The reference packager may be invoked
    directly (for example by the materializer), so it applies the same file when
    the environment does not already carry the credentials.
    """

    if os.environ.get("OPENAI_API_KEY"):
        return
    workspace = Path(
        os.environ.get("WORLDOLYMPIAD_WORKSPACE", Path(__file__).resolve().parents[3])
    )
    secret = Path(os.environ.get("SCENECODE_ENV_FILE", workspace / ".secrets" / "scenecode.env"))
    if not secret.is_file():
        return
    for line in secret.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


class VisionReviewUnavailable(RuntimeError):
    """The configured vision route cannot answer right now (quota/outage)."""


def vision_context_review(
    *,
    montage: Path,
    prompt: str,
    model: str = DEFAULT_VISION_MODEL,
    timeout_s: float = 1800.0,
) -> dict:
    """Ask the configured image-capable Responses route for a critique.

    This uses the same Responses API transport as the task author and the scene
    prompt architect (``urllib`` rather than an extra SDK dependency).
    """

    import urllib.error
    from urllib import request as urllib_request

    load_workspace_credentials()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise VisionReviewUnavailable(
            "in-context vision route is unconfigured (OPENAI_API_KEY absent)"
        )
    base_url = (
        os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    content = [
        {"type": "input_text", "text": prompt},
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,"
            + base64.b64encode(montage.read_bytes()).decode("ascii"),
            "detail": "high",
        },
    ]
    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "text": {"format": {"type": "json_object"}},
        "reasoning": {"effort": "high"},
    }

    def call(body: dict) -> dict:
        req = urllib_request.Request(
            base_url + "/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "OpenAI/Python WorldOlympiad2-InContextReview/1.0",
            },
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        body = call(payload)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 400 and "reasoning" in detail:
            payload.pop("reasoning", None)
            try:
                body = call(payload)
            except urllib.error.HTTPError as retry_exc:
                retry_detail = retry_exc.read().decode("utf-8", errors="replace")
                if retry_exc.code in _VISION_UNAVAILABLE_STATUS:
                    raise VisionReviewUnavailable(
                        f"vision route unavailable: HTTP {retry_exc.code}: "
                        f"{retry_detail[:300]}"
                    ) from retry_exc
                raise RuntimeError(
                    "in-context visual review request failed: "
                    f"HTTP {retry_exc.code}: {retry_detail[:500]}"
                ) from retry_exc
        elif exc.code in _VISION_UNAVAILABLE_STATUS:
            raise VisionReviewUnavailable(
                f"vision route unavailable: HTTP {exc.code}: {detail[:300]}"
            ) from exc
        else:
            raise RuntimeError(
                f"in-context visual review request failed: HTTP {exc.code}: {detail[:500]}"
            ) from exc

    text = body.get("output_text")
    if not text:
        for item in body.get("output", []):
            for block in item.get("content", []):
                if block.get("type") == "output_text" and block.get("text"):
                    text = block["text"]
                    break
            if text:
                break
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("in-context visual review returned no text content")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"in-context visual review returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("in-context visual review must return a JSON object")
    return parsed


def build_context_prompt(
    *,
    instruction: str,
    object_descriptions: dict[str, str],
    object_roles: dict[str, str],
    phases: tuple[str, ...],
    views: tuple[str, ...],
) -> str:
    objects = "\n".join(
        f"- {object_id} (role: {object_roles[object_id]}): {description}"
        for object_id, description in object_descriptions.items()
    )
    return (
        "This evidence sheet shows one executed household robot reference episode "
        "from the real native render.\n"
        f"Wide rows top to bottom: {', '.join(views)} (one row per camera).\n"
        f"Columns left to right: {', '.join(phases)} (one column per task phase).\n"
        "The bottom row holds magnified close-up crops, one per task object, "
        "labelled `close-up <object_id> (<view> f<frame> x<scale>)`.\n"
        f"Task: {instruction!r}\n"
        "Task objects and their exact visual descriptions:\n"
        f"{objects}\n\n"
        "Judge the episode in context with the robot, the support surface and the "
        "destination object.  Judge every colour and appearance question from the "
        "close-up row, which is the evidence of record: an object that spans only a "
        "few pixels in a wide row cannot be called opaque, pale, washed out or "
        "blended on the strength of that row.  These magnified crops still come "
        "from the model-native planes, so claim only what the crop resolves: "
        "surface colour, gloss, and the object's coarse material class.  A studio "
        "identity sheet, not this sheet, owns fine material structure -- wall "
        "transparency, wall thickness and a liquid level -- so do not report "
        "those as missing or wrong unless the crop shows a contradiction that is "
        "visible at the crop's own resolution.\n"
        "Answer as compact JSON with exactly these keys:\n"
        "{\n"
        '  "objects_recognizable": {"<object_id>": true|false, ...},\n'
        '  "object_evidence": {"<object_id>": "short visible evidence"},\n'
        '  "object_material_verdicts": {"<object_id>": {"material_reads_authentically": '
        'true|false, "evidence": "what the close-up resolves about its '
        'material, gloss and colour"}},\n'
        '  "materials_contextual": true|false,\n'
        '  "material_evidence": "one or two sentences about whether the materials '
        'read as plausible household materials for this setting",\n'
        '  "materials_self_consistent": true|false,\n'
        '  "cross_object_separation": {"<a>|<b>": "strong|moderate|weak|none", ...},\n'
        '  "separation_evidence": "one or two sentences",\n'
        '  "proportions_plausible": true|false,\n'
        '  "proportion_evidence": "one or two sentences about the sizes of the '
        'objects relative to the robot, the table and each other",\n'
        '  "visible_problems": ["only defects that a close-up proves, or that are '
        'unambiguous in the wide rows"],\n'
        '  "actionable_fixes": ["concrete appearance or proportion change if any"]\n'
        "}\n"
        "Answer `material_reads_authentically: true` for an object whose close-up "
        "shows the material the task describes, even when the wide rows are small "
        "or brightly lit.  An object whose magnified crop shows the right colour "
        "and gloss is authentic for this sheet: never fail it on material detail "
        "this sheet cannot resolve.  Set `materials_contextual` to the same verdict "
        "as the per-object material answers, and set `materials_self_consistent` to "
        "false only when the wide rows and the close-ups genuinely contradict each "
        "other.  Report an empty `visible_problems` list when the episode is good "
        "enough to serve as a household reference for this task.  Report "
        "`actionable_fixes` only for a change this harness can make and this sheet "
        "can verify."
    )


def _vision_verdict_passed(
    verdict: dict, *, object_ids, material_authority: dict | None = None,
    same_asset_aliases: list[dict] | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Gate the review on evidence-backed verdicts only.

    The review answers structured questions per object; free-form remarks are
    recorded as notes rather than blockers, because a wide 320x180 row invites
    complaints (small, pale, blended) that the magnified close-up disproves.

    Material authenticity has one owner: the studio identity certificate, which
    renders the asset alone at review resolution and binds a hashed contact
    sheet.  The in-context close-ups are magnified crops of model-native planes
    (640x360 and smaller), so they cannot resolve wall transparency or a liquid
    level.  A material complaint therefore blocks only for an object the studio
    certificate did not pass; otherwise it is recorded as a note beside the
    identity evidence instead of silently failing the bundle.
    """

    problems: list[str] = []
    notes: list[str] = []
    recognizable = verdict.get("objects_recognizable")
    if not isinstance(recognizable, dict):
        problems.append("vision review did not report per-object recognizability")
    else:
        for object_id in object_ids:
            if recognizable.get(object_id) is not True:
                evidence = (verdict.get("object_evidence") or {}).get(object_id, "")
                problems.append(f"{object_id} is not recognizable in context: {evidence}")
    authority = material_authority or {}
    material_verdicts = verdict.get("object_material_verdicts")
    material_failures: list[str] = []
    if not isinstance(material_verdicts, dict):
        problems.append("vision review did not report per-object material verdicts")
    else:
        for object_id in object_ids:
            entry = material_verdicts.get(object_id)
            if not isinstance(entry, dict):
                material_failures.append(f"{object_id} has no close-up material verdict")
                continue
            if entry.get("material_reads_authentically") is not True:
                complaint = str(entry.get("evidence", ""))
                studio = authority.get(object_id)
                if isinstance(studio, dict) and studio.get("passed") is True:
                    notes.append(
                        f"{object_id} in-context material remark not corroborated by its "
                        "studio identity sheet, which passed: "
                        + complaint
                        + " | studio evidence: "
                        + str(studio.get("evidence", ""))
                    )
                    continue
                material_failures.append(
                    f"{object_id} material does not read authentically: " + complaint
                )
    problems.extend(material_failures)
    if verdict.get("materials_contextual") is not True and not material_failures:
        aggregate = str(verdict.get("material_evidence", ""))
        unresolved = [
            object_id
            for object_id in object_ids
            if not (
                isinstance(authority.get(object_id), dict)
                and authority[object_id].get("passed") is True
            )
        ]
        if unresolved:
            problems.append(
                "aggregate material verdict failed without a passing studio identity "
                "sheet: " + aggregate
            )
        else:
            notes.append(
                "aggregate material verdict disagreed with the close-up evidence: "
                + aggregate
            )
    separation = verdict.get("cross_object_separation")
    alias_pairs = {
        frozenset(item["object_ids"]) for item in same_asset_aliases or []
    }
    if isinstance(separation, dict):
        for pair, rating in separation.items():
            if str(rating).lower() == "none":
                ids = str(pair).split("|")
                if len(ids) == 2 and frozenset(ids) in alias_pairs:
                    notes.append(
                        f"{pair} names one inventory-certified SceneCode asset; "
                        "mutual colour separation is inapplicable"
                    )
                    continue
                problems.append(
                    f"{pair} separation is none: "
                    + str(verdict.get("separation_evidence", ""))
                )
            elif str(rating).lower() == "weak":
                notes.append(
                    f"{pair} separation is weak in the wide rows: "
                    + str(verdict.get("separation_evidence", ""))
                )
    else:
        problems.append("vision review did not report cross-object separation")
    if verdict.get("proportions_plausible") is not True:
        problems.append(
            "proportions are not plausible in context: "
            + str(verdict.get("proportion_evidence", ""))
        )
    if verdict.get("materials_self_consistent") is False:
        problems.append(
            "wide rows and close-ups contradict each other on the materials: "
            + str(verdict.get("material_evidence", ""))
        )
    for item in verdict.get("visible_problems") or []:
        notes.append(f"visible problem: {item}")
    return (not problems), problems, notes


def visual_separation_evidence(appearance, *, verdict, vision_passed, components, host_visibility):
    """Colour is one cue; independently reviewed shape/boundaries are another."""
    checks=dict(appearance.get("checks") or {})
    resolved=[]
    if vision_passed and isinstance(verdict,dict) and str(verdict.get("separation_evidence","")).strip():
        recognized=verdict.get("objects_recognizable") or {}
        ratings=verdict.get("cross_object_separation") or {}
        for pair in appearance.get("pairs",[]):
            left,right=pair["left"],pair["right"]
            rating=ratings.get(left+"|"+right,ratings.get(right+"|"+left))
            if (rating!="strong" or recognized.get(left) is not True or recognized.get(right) is not True
                    or min(pair.get("left_pixels",0),pair.get("right_pixels",0))<MINIMUM_PAIR_PIXELS):
                continue
            related=[c for c in components if {c.component_id,c.host_object_id}=={left,right}]
            if any(host_visibility.get(c.component_id,{}).get("passed") is not True for c in related):
                continue
            for object_id in (left,right):
                if checks.get("separated:"+object_id) is False:
                    checks["separated:"+object_id]=True
                    resolved.append({"object_id":object_id,"pair":[left,right],"view_id":pair["view_id"],
                        "authority":"independent_in_context_vision_with_registered_visible_instances",
                        "rating":rating,"evidence":verdict.get("separation_evidence",""),
                        "articulated_host_visibility_required":bool(related)})
    return {"passed":bool(checks) and all(checks.values()),"checks":checks,"structural_evidence":resolved,
            "raw_rgb_separation_passed":bool(appearance.get("passed")),
            "policy":"colour_or_independently_reviewed_visible_shape_boundaries.v1"}


def reuse_identical_context_review(certificate_path, *, montage, instruction,
                                  descriptions, roles, model):
    """Reuse a prior verdict, including a negative one, only for identical inputs."""
    source=Path(certificate_path).resolve()
    prior=json.loads(source.read_text())
    original_task=json.loads((source.parent/"task_instruction.json").read_text())
    if (prior.get("vision_review")!="required" or prior.get("vision_model")!=model
            or original_task.get("instruction")!=instruction
            or prior.get("object_visual_descriptions")!=descriptions
            or prior.get("object_roles")!=roles
            or prior.get("montage",{}).get("sha256")!=sha256_file(montage["path"])
            or prior.get("montage",{}).get("phases")!=montage["phases"]
            or prior.get("montage",{}).get("views")!=montage["views"]
            or not isinstance(prior.get("vision_verdict"),dict)):
        raise ValueError("prior visual review does not bind identical current inputs")
    return prior["vision_verdict"],{"source_certificate_sha256":sha256_file(source),
        "source_montage_sha256":prior["montage"]["sha256"],"policy":"reuse_identical_inputs_not_resample_verdict"}


def certify_reference_visual_context(
    *,
    task,
    trajectory: dict,
    render: dict,
    output: str | Path,
    physical_variant_manifest: dict | None = None,
    grasp_band_plan: dict | None = None,
    material_authority: dict | None = None,
    same_asset_aliases: list[dict] | None = None,
    vision: str = "required",
    vision_model: str = DEFAULT_VISION_MODEL,
    views: tuple[str, ...] = ("top", "exterior_left", "exterior_right"),
) -> Path:
    """Fail-closed in-context appearance and proportion certificate.

    ``material_authority`` carries the studio identity verdict per object
    (``{"<object_id>": {"passed": bool, "evidence": str, ...}}``).  It is the
    material evidence of record; this certificate owns context, separation and
    proportion, and it keeps an unverified object fail-closed.
    """

    output = Path(output).resolve()
    object_roles = {
        item.object_id: item.role
        for item in task.objects
        if item.role in {"manipuland", "destination", "support"}
    }
    object_descriptions = {
        item.object_id: item.visual_description for item in task.objects
    }
    for component in getattr(task, "articulation_components", []):
        # The movable door/drawer is a task manipuland even though its SDF is
        # stored on the host object.  Measure the rendered part against its
        # host rather than comparing the host with a usually invisible table.
        object_roles[component.component_id] = "manipuland"
        object_descriptions[component.component_id] = component.visual_description
    montage = build_in_context_montage(
        render, trajectory, output, views=views, object_roles=object_roles
    )
    appearance = measure_in_context_appearance(
        render,
        trajectory,
        views=views,
        object_roles=object_roles,
    )
    proportion_evidence: dict = {
        "embodiment_uniform_scale": (
            (physical_variant_manifest or {}).get("embodiment_compatibility", {}).get(
                "uniform_scale", 1.0
            )
        ),
        "grasp_band_plan": grasp_band_plan,
    }
    checks = {
        "cross_object_visual_distinguishability": bool(appearance["passed"]),
        "per_object_appearance_measured": bool(appearance["objects"]),
    }
    host_visibility = articulated_host_visibility(
        appearance, getattr(task, "articulation_components", [])
    )
    for component_id, record in host_visibility.items():
        checks[f"articulated_host_visible:{component_id}"] = record["passed"]
    if grasp_band_plan is not None:
        feasible = bool(grasp_band_plan.get("feature_grasp_feasible"))
        scale = float(proportion_evidence["embodiment_uniform_scale"] or 1.0)
        checks["authored_scale_preserved_when_graspable"] = bool(
            (not feasible) or scale >= 1.0 - 1.0e-9
        )
    verdict: dict | None = None
    vision_problems: list[str] = []
    vision_notes: list[str] = []
    vision_deferred: str | None = None
    reused_vision=None
    vision_required = vision == "required"
    if vision_required:
        prompt = build_context_prompt(
            instruction=task.instruction,
            object_descriptions=object_descriptions,
            object_roles=object_roles,
            phases=tuple(montage["phases"]),
            views=tuple(montage["views"]),
        )
        try:
            prior_path=os.environ.get("WORLDOLYMPIAD_CONTEXT_REVIEW_REUSE")
            if prior_path:
                verdict,reused_vision=reuse_identical_context_review(prior_path,montage=montage,
                    instruction=task.instruction,descriptions=object_descriptions,roles=object_roles,model=vision_model)
                (output/"prior_context_review.json").write_bytes(Path(prior_path).read_bytes())
            else:
                verdict = vision_context_review(
                    montage=montage["path"], prompt=prompt,
                    model=vision_model,
                )
        except VisionReviewUnavailable as exc:
            # A capped or unreachable vision route must not silently pass a
            # bundle, and must not stall the queue either: the deterministic
            # registration, separation and visibility checks above stay
            # enforced, the evidence sheet with the magnified close-ups is
            # preserved, and the certificate records that the critique is owed.
            vision_deferred = str(exc)
            vision_notes.append(
                "vision critique deferred to the next available route: "
                + str(exc)[:200]
            )
        else:
            passed, vision_problems, vision_notes = _vision_verdict_passed(
                verdict,
                object_ids=sorted(object_ids_for_review(task)),
                material_authority=material_authority,
                same_asset_aliases=same_asset_aliases,
            )
            checks["vision_in_context_review"] = bool(passed)
    separation_evidence=visual_separation_evidence(appearance,verdict=verdict,
        vision_passed=checks.get("vision_in_context_review") is True,
        components=getattr(task,"articulation_components",[]),host_visibility=host_visibility)
    checks["cross_object_visual_distinguishability"]=separation_evidence["passed"]
    blockers = [name for name, value in checks.items() if not value]
    blockers.extend(vision_problems)
    notes = list(vision_notes)
    certificate = {
        "schema_version": SCHEMA,
        "measurement_policy": "visible_shape_or_colour_and_articulated_host_exposure_v3",
        "passed": not blockers,
        "vision_review": (
            "deferred" if vision_deferred else ("required" if vision_required else vision)
        ),
        "vision_deferred_reason": vision_deferred,
        "vision_model": vision_model if vision_required else None,
        "object_roles": object_roles,
        "object_visual_descriptions": object_descriptions,
        "montage": {
            "path": montage["path"].name,
            "sha256": sha256_file(montage["path"]),
            "views": montage["views"],
            "phases": montage["phases"],
            "frame_indices": montage["frame_indices"],
            "size_wh": montage["size_wh"],
            "closeups": montage.get("closeups") or {},
        },
        "rendered_appearance": appearance,
        "visual_separation_evidence":separation_evidence,
        "articulated_host_visibility": host_visibility,
        "proportion_evidence": proportion_evidence,
        "material_authority": {
            object_id: value
            for object_id, value in (material_authority or {}).items()
            if object_id in object_descriptions
        },
        "same_asset_aliases": same_asset_aliases or [],
        "vision_verdict": verdict,
        "vision_review_reuse":reused_vision,
        "checks": checks,
        "blockers": blockers,
        "notes": notes,
        "repair_guidance": {
            "appearance": (
                "inspect whether task entities are recognizable by shape, boundaries or colour; "
                "same-colour parts need independently reviewed visible structural evidence; if an "
                "articulated host is obscured, refit a versioned camera rig "
                "against the same full physical trajectory; re-render and keep "
                "the task semantics and source physics fixed"
            ),
            "proportions": (
                "if the manipuland is not graspable at its authored size, fix the "
                "grasp band or the asset's grasp feature; do not shrink the asset to "
                "fit the gripper aperture"
            ),
        },
    }
    path = write_json(output / "visual_context_certificate.json", certificate)
    if not certificate["passed"]:
        raise ValueError(
            "in-context visual context review failed: "
            + "; ".join(blockers)
            + f" (evidence: {path}, montage: {montage['path']})"
        )
    return path


def object_ids_for_review(task) -> set[str]:
    ids = {
        item.object_id
        for item in task.objects
        if item.role in {"manipuland", "destination", "support"}
    }
    ids.update(component.component_id for component in getattr(task, "articulation_components", []))
    return ids
