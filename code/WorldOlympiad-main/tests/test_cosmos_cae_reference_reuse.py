import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from autoresearch.io import sha256_file
from scripts.reuse_cosmos_cae_reference import prepare


def pair(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    case = {"case_id": "case", "model_id": "cosmos3_nano_droid", "reference_view_id": "exterior_left",
            "generated_view_id": "exterior_left", "reference_video": "ref.mp4", "generated_video": "gen.mp4"}
    for root in (old, new):
        (root / "cae_base_scene").mkdir(parents=True)
        (root / "worldolympiad_case.json").write_text(json.dumps(case))
        (root / "ref.mp4").write_bytes(b"immutable reference")
        for name in ("camera_calibration_info.json", "initial_depth.pt"):
            (root / "cae_base_scene" / name).write_bytes(b"same frame zero")
        urdf = root / "cae_base_scene/registered_droid.urdf"
        urdf.write_bytes(b"identical robot")
        registration = {"source_urdf": str(urdf), "source_urdf_sha256": sha256_file(urdf), "pose": [0, 0, 0]}
        (root / "cae_base_scene/robot_registration.json").write_text(json.dumps(registration))
    cached = old / "official_cae"
    (cached / "reference/dream2flow").mkdir(parents=True)
    (cached / "reference/dream2flow/trajectory_optimization_plan.npz").write_bytes(b"reference optimizer")
    (cached / "reference/rgb.mp4").write_bytes(b"immutable reference")
    for name in ("camera_calibration_info.json", "initial_depth.pt", "scene_data.yaml"):
        (cached / "reference" / name).write_bytes(b"reference sidecar")
    (cached / "official_registration.json").write_bytes(b"official registration")
    (cached / "imagination").mkdir()
    (cached / "imagination/old_optimizer.npz").write_bytes(b"must not reuse")
    (old / "official_cae_pair_spec.json").write_text(json.dumps({"output_dir": "official_cae",
        "registration": "official_cae/official_registration.json"}))
    writer = cv2.VideoWriter(str(new / "gen.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (320, 180))
    assert writer.isOpened()
    writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
    writer.release()
    return old, new


def test_reference_optimizer_reuse_keeps_sources_immutable_and_excludes_old_prediction(tmp_path):
    old, new = pair(tmp_path)
    before = {p: sha256_file(p) for p in old.rglob("*") if p.is_file()}
    receipt = json.loads(prepare(old, new).read_text())
    assert receipt["reference_regenerated"] is False
    assert receipt["imagination_optimizer_reused"] is False
    assert not (new / "official_cae/imagination/old_optimizer.npz").exists()
    assert (new / "official_cae/reference/dream2flow/trajectory_optimization_plan.npz").read_bytes() == b"reference optimizer"
    assert before == {p: sha256_file(p) for p in old.rglob("*") if p.is_file()}


def test_changed_camera_or_initial_robot_pose_cannot_reuse_reference_optimizer(tmp_path):
    old, new = pair(tmp_path)
    p = new / "cae_base_scene/robot_registration.json"
    registration = json.loads(p.read_text())
    registration["pose"] = [1, 0, 0]
    p.write_text(json.dumps(registration))
    with pytest.raises(ValueError, match="frame-zero robot registration differs"):
        prepare(old, new)
    assert not (new / "official_cae").exists()
