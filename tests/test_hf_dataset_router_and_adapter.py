"""Unit and contract tests for HuggingFace dataset routing and auto-adapter integration."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import hf_dataset_router as router  # noqa: E402
import benchmark as bench  # noqa: E402


SAMPLE_CASES = [
    {"case_id": "rd_0", "subset": "public", "source": "RoboDojo", "domain": "simulation", "embodiment": "ALOHA-AgileX", "task_family": "grasp_manipulate", "instruction": "Pick bottle.", "models": ["fastwam", "lingbot_va", "longwam_cod"]},
    {"case_id": "t30_dual", "subset": "public", "source": "Table30", "domain": "real", "embodiment": "ALOHA", "task_family": "stack", "instruction": "Stack bowls.", "models": ["fastwam", "lingbot_va", "longwam_cod"]},
    {"case_id": "rc_dual", "subset": "public", "source": "RoboCOIN", "domain": "real", "embodiment": "AgileX Cobot Magic", "task_family": "grasp_manipulate", "instruction": "Grasp toy.", "models": ["fastwam", "lingbot_va", "longwam_cod"]},
    {"case_id": "con_arx5", "subset": "constructed", "source": "Constructed", "domain": "simulation", "embodiment": "ARX5", "task_family": "cup_hanging", "instruction": "Hang cup.", "models": ["fastwam", "lingbot_va", "longwam_cod"], "reference_video": "benchmark/constructed/arx5/cup_hanging/c0/reference.mp4"},
    {"case_id": "ms_single", "subset": "public", "source": "ManiSkill3", "domain": "simulation", "embodiment": "Panda", "task_family": "stack", "instruction": "Stack cube.", "models": ["cosmos3", "gr00t_n1.6_droid"]},
    {"case_id": "rm_single", "subset": "public", "source": "RoboMIND", "domain": "real", "embodiment": "Franka", "task_family": "insert_align", "instruction": "Insert peg.", "models": ["cosmos3"]},
    {"case_id": "con_panda", "subset": "constructed", "source": "Constructed", "domain": "simulation", "embodiment": "Panda", "task_family": "cabinet_articulation", "instruction": "Open cabinet.", "models": ["cosmos3", "gr00t_n1.6_droid"], "reference_video": "benchmark/constructed/panda/cabinet_articulation/p0/reference.mp4"},
]


def test_dual_arm_hybrid_routing_selects_dual_and_prefers_sim_when_available() -> None:
    dual = router.filter_cases(SAMPLE_CASES, arm="dual", execution="sim_and_real")
    assert [c["case_id"] for c in dual] == ["rd_0", "t30_dual", "rc_dual", "con_arx5"]
    modes = {c["case_id"]: c["recommended_mode"] for c in dual}
    assert modes == {
        "rd_0": "sim_rollout",
        "t30_dual": "i2av",
        "rc_dual": "i2av",
        "con_arx5": "sim_rollout",
    }
    # Backwards-compatible alias video_all matches sim_and_real
    assert [c["case_id"] for c in router.filter_cases(SAMPLE_CASES, arm="dual", execution="video_all")] == [
        c["case_id"] for c in dual
    ]


def test_single_arm_sim_only_excludes_real_world_cases() -> None:
    sim_only = router.filter_cases(SAMPLE_CASES, arm="single", execution="sim_only")
    assert [c["case_id"] for c in sim_only] == ["ms_single", "con_panda"]
    assert all(c["recommended_mode"] == "sim_rollout" for c in sim_only)


def test_single_arm_hybrid_includes_real_and_sim() -> None:
    single_all = router.filter_cases(SAMPLE_CASES, arm="single", execution="sim_and_real")
    assert [c["case_id"] for c in single_all] == ["ms_single", "rm_single", "con_panda"]
    assert {c["case_id"]: c["recommended_mode"] for c in single_all} == {
        "ms_single": "sim_rollout",
        "rm_single": "i2av",
        "con_panda": "sim_rollout",
    }


def test_constructed_only_routing_isolates_in_repo_canonical_videos() -> None:
    con_dual = router.filter_cases(SAMPLE_CASES, arm="dual", execution="constructed_only")
    con_single = router.filter_cases(SAMPLE_CASES, arm="single", execution="constructed_only")
    assert [c["case_id"] for c in con_dual] == ["con_arx5"]
    assert [c["case_id"] for c in con_single] == ["con_panda"]


def test_routed_manifest_and_external_adapter_pass_benchmark_contract(tmp_path: Path) -> None:
    # Build a tiny valid 2-frame MP4 + initial PNG using cv2/PIL or ffmpeg
    import imageio_ffmpeg
    from PIL import Image

    case_dir = tmp_path / "cases" / "dual6_test_000"
    case_dir.mkdir(parents=True)
    init_png = case_dir / "initial_front.png"
    Image.new("RGB", (160, 120), (40, 120, 200)).save(init_png)
    ref_mp4 = case_dir / "reference.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y", "-loop", "1", "-i", str(init_png), "-t", "0.5",
            "-r", "10", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(ref_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    dataset_json = tmp_path / "dataset.json"
    dataset_json.write_text(json.dumps({
        "schema_version": "wov2_benchmark.v1",
        "dataset_id": "worldolympiad2_dual_constructed_only",
        "version": "2026.11.25",
        "source_repo": router.HF_REPO,
        "cases": [{
            "case_id": "dual6_test_000",
            "domain": "sim",
            "instruction": "Open the cabinet door.",
            "reference_view": "front",
            "reference_video": "cases/dual6_test_000/reference.mp4",
            "conditioning": {"front": "cases/dual6_test_000/initial_front.png"},
        }],
    }), encoding="utf-8")

    # 1. Validate dataset manifest via benchmark.py dataset() loader
    data, rows = bench.dataset(dataset_json)
    assert data["dataset_id"] == "worldolympiad2_dual_constructed_only"
    assert len(rows) == 1 and rows[0]["case_id"] == "dual6_test_000"

    # 2. Validate external wov2_adapter.v1 spec + routed manifest together via benchmark.py check CLI
    adapter_json = tmp_path / "adapter.json"
    adapter_json.write_text(json.dumps({
        "schema_version": "wov2_adapter.v1",
        "model": "example_embodied_wm",
        "modes": ["i2av"],
        "minimum_free_mib": 16000,
        "gpu_count": 1,
        "command": [sys.executable, "run_adapter.py", "--input", "{input}", "--output", "{output}"],
    }), encoding="utf-8")
    rc = bench.main(["check", "--manifest", str(dataset_json), "--adapter", str(adapter_json)])
    assert rc in (0, None)
