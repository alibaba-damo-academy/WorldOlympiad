#!/usr/bin/env python3
"""Build a deterministic, image-matched cabinet SDF for reference playback."""

from __future__ import annotations

import argparse
from pathlib import Path


def box(name: str, size: tuple[float, float, float], pose: tuple[float, float, float], color: str) -> str:
    sx, sy, sz = size
    x, y, z = pose
    return f'''<visual name="{name}_visual"><pose>{x} {y} {z} 0 0 0</pose><geometry><box><size>{sx} {sy} {sz}</size></box></geometry><material><ambient>{color} 1</ambient><diffuse>{color} 1</diffuse></material></visual>
<collision name="{name}_collision"><pose>{x} {y} {z} 0 0 0</pose><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>'''


def cylinder(name: str, radius: float, length: float, pose: tuple[float, float, float], color: str, axis: str = "y") -> str:
    x, y, z = pose
    rotation = {"x": "0 1.5707963 0", "y": "1.5707963 0 0", "z": "0 0 0"}[axis]
    return f'''<visual name="{name}_visual"><pose>{x} {y} {z} {rotation}</pose><geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry><material><ambient>{color} 1</ambient><diffuse>{color} 1</diffuse></material></visual>
<collision name="{name}_collision"><pose>{x} {y} {z} {rotation}</pose><geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry></collision>'''


def build() -> str:
    wood = "0.68 0.42 0.20"
    dark = "0.18 0.20 0.21"
    parts = [
        box("left_side", (0.08, 0.58, 1.82), (-0.56, 0.0, 0.95), wood),
        box("right_side", (0.08, 0.58, 1.82), (0.56, 0.0, 0.95), wood),
        box("top_cap", (1.20, 0.62, 0.08), (0.0, 0.0, 1.90), wood),
        box("plinth", (1.12, 0.54, 0.10), (0.0, 0.02, 0.08), wood),
        box("back", (1.04, 0.06, 1.72), (0.0, -0.25, 0.98), wood),
        box("shelf", (1.04, 0.50, 0.06), (0.0, 0.0, 1.02), wood),
        box("door_left", (0.51, 0.045, 0.82), (-0.26, 0.285, 1.45), wood),
        box("door_right", (0.51, 0.045, 0.82), (0.26, 0.285, 1.45), wood),
        cylinder("door_left_handle", 0.023, 0.34, (-0.04, 0.34, 1.45), dark, axis="z"),
        cylinder("door_right_handle", 0.023, 0.34, (0.04, 0.34, 1.45), dark, axis="z"),
    ]
    drawer_z = (0.82, 0.61, 0.40, 0.19)
    for index, z in enumerate(drawer_z, 1):
        if index == 1:
            continue
        parts.extend([
            box(f"fixed_drawer_{index}", (1.04, 0.045, 0.17), (0.0, 0.29, z), wood),
            cylinder(f"fixed_handle_{index}", 0.023, 0.42, (0.0, 0.38, z), dark, axis="x"),
        ])
    # Selected drawer is its own link; the bar is separated from the front by 9 cm.
    drawer_parts = [
        box("drawer_front", (1.04, 0.045, 0.17), (0.0, 0.29, 0.82), wood),
        box("drawer_bottom", (0.96, 0.48, 0.035), (0.0, 0.02, 0.725), wood),
        box("drawer_left_side", (0.04, 0.48, 0.16), (-0.50, 0.02, 0.81), wood),
        box("drawer_right_side", (0.04, 0.48, 0.16), (0.50, 0.02, 0.81), wood),
        box("drawer_back", (0.96, 0.04, 0.16), (0.0, -0.20, 0.81), wood),
    ]
    drawer_parts.extend([
        cylinder("handle_bar", 0.023, 0.48, (0.0, 0.38, 0.82), dark, axis="x"),
        cylinder("handle_post_left", 0.023, 0.09, (-0.21, 0.335, 0.82), dark, axis="y"),
        cylinder("handle_post_right", 0.023, 0.09, (0.21, 0.335, 0.82), dark, axis="y"),
    ])
    fixed = "\n".join(parts)
    moving = "\n".join(drawer_parts)
    return f'''<?xml version="1.0"?>
<sdf version="1.7"><model name="reference_cabinet"><static>false</static>
<link name="base_link">{fixed}</link>
<link name="drawer_01">{moving}</link>
<joint name="drawer_01_joint" type="prismatic"><parent>base_link</parent><child>drawer_01</child><axis><xyz>0 1 0</xyz><limit><lower>0</lower><upper>0.32</upper></limit><dynamics><damping>0.25</damping></dynamics></axis></joint>
</model></sdf>\n'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
