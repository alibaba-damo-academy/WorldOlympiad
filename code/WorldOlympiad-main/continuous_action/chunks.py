"""WorldOlympiad-compatible temporal chunk parsing and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChunkSpec:
    chunk_index: int
    start_sec: float
    end_sec: float
    caption: str = ""
    action: str | None = None

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


def parse_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return int(parts[0]) * 60.0 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
    raise ValueError(f"invalid timestamp: {value!r}")


def parse_interval(value: str) -> tuple[float, float]:
    cleaned = value.strip().strip("[]()")
    separator = "," if "," in cleaned else "-"
    if separator not in cleaned:
        raise ValueError(f"invalid interval: {value!r}")
    left, right = cleaned.split(separator, 1)
    return parse_timestamp(left), parse_timestamp(right)


def _items(payload: Any, keys: tuple[str, ...]) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                value = payload[key]
                if not isinstance(value, list):
                    raise ValueError(f"{key!r} must contain a list")
                return value
        return [payload]
    raise ValueError("chunk metadata must be a JSON list or object")


def load_worldolympiad_chunks(
    chunk_json: str | Path,
    prompt_json: str | Path | None = None,
    *,
    tolerance_sec: float = 1e-3,
) -> list[ChunkSpec]:
    """Load the benchmark's left-closed/right-open, contiguous chunk format."""

    chunk_payload = json.loads(Path(chunk_json).read_text(encoding="utf-8"))
    prompt_payload = (
        json.loads(Path(prompt_json).read_text(encoding="utf-8")) if prompt_json is not None else []
    )
    prompt_items = _items(prompt_payload, ("chunks", "prompts")) if prompt_json is not None else []
    prompts: dict[int, dict[str, Any]] = {}
    for fallback, raw in enumerate(prompt_items):
        item = raw if isinstance(raw, dict) else {"caption": str(raw)}
        index = int(item.get("chunk_index", item.get("index", fallback)))
        prompts[index] = item

    chunks: list[ChunkSpec] = []
    for fallback, raw in enumerate(_items(chunk_payload, ("chunks",))):
        if not isinstance(raw, dict):
            raise ValueError("each chunk entry must be an object")
        index = int(raw.get("chunk_index", raw.get("index", fallback)))
        prompt = prompts.get(index, {})
        start = raw.get("generated_start_sec", raw.get("start_sec", raw.get("start")))
        end = raw.get("generated_end_sec", raw.get("end_sec", raw.get("end")))
        if start is None or end is None:
            interval = raw.get("generated_interval") or raw.get("interval") or prompt.get("interval")
            if not isinstance(interval, str):
                raise ValueError(f"chunk {index} has no start/end or interval")
            start, end = parse_interval(interval)
        chunks.append(
            ChunkSpec(
                chunk_index=index,
                start_sec=parse_timestamp(start),
                end_sec=parse_timestamp(end),
                caption=str(
                    prompt.get("caption")
                    or prompt.get("prompt")
                    or prompt.get("text")
                    or raw.get("caption")
                    or ""
                ),
                action=prompt.get("action") or prompt.get("actions") or raw.get("action"),
            )
        )

    chunks.sort(key=lambda item: item.chunk_index)
    validate_worldolympiad_chunks(chunks, tolerance_sec=tolerance_sec)
    return chunks


def validate_worldolympiad_chunks(
    chunks: list[ChunkSpec],
    *,
    tolerance_sec: float = 1e-3,
) -> None:
    """Enforce the paper's at-most-six contiguous chunk contract."""

    if not chunks:
        raise ValueError("at least one chunk is required")
    if len(chunks) > 6:
        raise ValueError(f"WorldOlympiad permits at most six chunks, got {len(chunks)}")
    seen: set[int] = set()
    for position, chunk in enumerate(chunks):
        if chunk.chunk_index in seen:
            raise ValueError(f"duplicate chunk_index {chunk.chunk_index}")
        seen.add(chunk.chunk_index)
        if chunk.start_sec < 0 or chunk.end_sec <= chunk.start_sec:
            raise ValueError(f"invalid interval for chunk {chunk.chunk_index}")
        if position:
            gap = chunk.start_sec - chunks[position - 1].end_sec
            if abs(gap) > tolerance_sec:
                kind = "gap" if gap > 0 else "overlap"
                raise ValueError(
                    f"chunks {chunks[position - 1].chunk_index} and {chunk.chunk_index} "
                    f"have a {kind} of {abs(gap):.6f}s"
                )
