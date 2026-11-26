"""Typed filesystem boundary for reusable multi-object robotics scenes."""

from __future__ import annotations

import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET


SCHEMA_VERSION = "robotics_scene_object_table.v1"


def articulated_component_links(host_sdf: Path, child_link: str) -> list[str]:
    """The moving link and its fixed descendants share one semantic identity."""
    root = ET.parse(host_sdf).getroot()
    declared = {link.get("name") for link in root.findall(".//link")}
    if child_link not in declared:
        raise ValueError("articulation child link is absent from the rendered SDF")
    selected = {child_link}
    changed = True
    while changed:
        changed = False
        for joint in root.findall(".//joint"):
            parent, child = joint.findtext("parent"), joint.findtext("child")
            if (joint.get("type") == "fixed" and parent in selected
                    and child in declared and child not in selected):
                selected.add(child)
                changed = True
    return sorted(selected)


def carry_evidence_frame_ids(
    frames: list[dict], max_samples: int = 12
) -> set[int]:
    """Sample every compiled step/phase pair for registered visual evidence."""

    carry_phases = {"lift", "orient", "transfer", "lower"}
    carry_by_step_phase: dict[tuple[int, str], list[int]] = {}
    all_carry: list[int] = []
    for frame in frames:
        phase = str(frame.get("phase"))
        if phase in carry_phases:
            frame_id = int(frame["frame"])
            key = (int(frame.get("operator_step_index", 0)), phase)
            carry_by_step_phase.setdefault(key, []).append(frame_id)
            all_carry.append(frame_id)
    if not carry_by_step_phase:
        raise ValueError("trajectory has no pick/carry frames for grasp evidence")
    if max_samples < len(carry_by_step_phase):
        raise ValueError(
            "carry evidence budget cannot cover every operator step/phase pair"
        )
    selected = {
        values[len(values) // 2] for values in carry_by_step_phase.values()
    }
    candidates = sorted(set(all_carry) - selected)
    remaining = min(max_samples - len(selected), len(candidates))
    if remaining == 1:
        selected.add(candidates[len(candidates) // 2])
    elif remaining > 1:
        selected.update(
            candidates[int(index * (len(candidates) - 1) / (remaining - 1))]
            for index in range(remaining)
        )
    return selected


def load_scene_object_sources(
    path: str | Path, *, minimum_objects: int = 2
) -> dict[str, Path]:
    if minimum_objects < 1:
        raise ValueError("scene-object table minimum must be positive")
    manifest = Path(path).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported robotics scene-object table")
    result: dict[str, Path] = {}
    for item in payload.get("objects") or []:
        object_id = item.get("object_id")
        raw_path = item.get("sdf_path")
        if not isinstance(object_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", object_id
        ) is None:
            raise ValueError("scene-object table contains an unsafe object id")
        if object_id in result or not isinstance(raw_path, str):
            raise ValueError("scene-object table contains duplicate or missing data")
        source = Path(raw_path)
        if not source.is_absolute():
            source = (manifest.parent / source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        result[object_id] = source
    if len(result) < minimum_objects:
        raise ValueError(
            f"scene-object table requires at least {minimum_objects} task objects"
        )
    return result
