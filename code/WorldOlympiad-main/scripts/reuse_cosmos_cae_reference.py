#!/usr/bin/env python3
"""Reuse certified CAE reference evidence for a new output of the same scene."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

import cv2
from autoresearch.io import sha256_file, write_json


def prepare(source_case: Path, case_root: Path) -> Path:
    source_case, case_root = source_case.resolve(), case_root.resolve()
    old = json.loads((source_case / "worldolympiad_case.json").read_text())
    new = json.loads((case_root / "worldolympiad_case.json").read_text())
    for key in ("case_id", "model_id", "reference_view_id", "generated_view_id"):
        if old.get(key) != new.get(key):
            raise ValueError(f"CAE reference reuse changes {key}")
    if new["model_id"] != "cosmos3_nano_droid":
        raise ValueError("expected a Cosmos native pair")
    if sha256_file(source_case / old["reference_video"]) != sha256_file(case_root / new["reference_video"]):
        raise ValueError("CAE reference video differs")
    for name in ("camera_calibration_info.json", "initial_depth.pt"):
        if sha256_file(source_case / "cae_base_scene" / name) != sha256_file(case_root / "cae_base_scene" / name):
            raise ValueError(f"CAE frame-zero contract differs: {name}")
    registrations = [json.loads((root / "cae_base_scene/robot_registration.json").read_text())
                     for root in (source_case, case_root)]
    for registration in registrations:
        if sha256_file(Path(registration["source_urdf"])) != registration["source_urdf_sha256"]:
            raise ValueError("registered robot source differs from its bound digest")
    if ({k: v for k, v in registrations[0].items() if k != "source_urdf"}
            != {k: v for k, v in registrations[1].items() if k != "source_urdf"}):
        raise ValueError("CAE frame-zero robot registration differs")
    spec_path = source_case / "official_cae_pair_spec.json"
    if not spec_path.exists():
        spec_path = source_case / "cae_pair_spec.json"
    spec = json.loads(spec_path.read_text())
    pair = source_case / spec["output_dir"]
    registration = source_case / spec["registration"]
    if not (pair / "reference/dream2flow/trajectory_optimization_plan.npz").is_file():
        raise ValueError("selected reference has no official optimizer evidence")
    if sha256_file(pair / "reference/rgb.mp4") != sha256_file(case_root / new["reference_video"]):
        raise ValueError("cached optimizer reference video differs")
    output = case_root / "official_cae"
    receipt = case_root / "cae_reference_reuse.json"
    if output.exists():
        raise FileExistsError("CAE preparation already exists; inspect its reuse receipt")
    output.mkdir()
    reference = output / "reference"
    shutil.copytree(pair / "reference", reference)
    for p in pair.iterdir():
        if p.is_file():
            shutil.copy2(p, output / p.name)
    shutil.copy2(registration, output / "official_registration.json")
    imagination = output / "imagination"
    imagination.mkdir()
    for name in ("camera_calibration_info.json", "initial_depth.pt", "scene_data.yaml"):
        shutil.copy2(reference / name, imagination / name)
    video = case_root / new["generated_video"]
    shutil.copy2(video, imagination / "rgb.mp4")
    capture = cv2.VideoCapture(str(video))
    okay, frame = capture.read()
    capture.release()
    if not okay or frame.shape[:2] != (180, 320):
        raise ValueError("new Cosmos video lacks native geometry")
    if not cv2.imwrite(str(imagination / "camera_rgb.png"), frame):
        raise RuntimeError("cannot write native initial image")
    updated = {**spec, "output_dir": "official_cae",
               "registration": "official_cae/official_registration.json",
               "reference_video": new["reference_video"], "imagination_video": new["generated_video"]}
    write_json(case_root / "official_cae_pair_spec.json", updated)
    source_files = {str(p): sha256_file(p) for p in pair.rglob("*") if p.is_file() and "imagination" not in p.relative_to(pair).parts}
    source_files[str(registration)] = sha256_file(registration)
    write_json(receipt, {"schema_version": "cosmos_cae_reference_reuse.v1",
        "source_case": str(source_case), "source_case_sha256": sha256_file(source_case / "worldolympiad_case.json"),
        "new_case_sha256": sha256_file(case_root / "worldolympiad_case.json"),
        "reference_video_sha256": sha256_file(reference / "rgb.mp4"),
        "imagination_video_sha256": sha256_file(video), "sources_sha256": source_files,
        "reference_regenerated": False, "imagination_optimizer_reused": False})
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    print(prepare(**vars(parser.parse_args())))
