from __future__ import annotations

import importlib.util
import sys

from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_subset_sequential.py"
spec = importlib.util.spec_from_file_location("run_subset_sequential", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_load_prompt_jobs_reads_id_and_description(tmp_path):
    csv_path = tmp_path / "subset_sub.csv"
    csv_path.write_text(
        "ID,Description,ObjCount\n"
        "1,A bedroom with a bed,1\n"
        "8,A living room with a TV,4\n"
    )

    jobs = module.load_prompt_jobs(csv_path=csv_path, output_root=tmp_path / "outputs")

    assert [job.scene_id for job in jobs] == [1, 8]
    assert [job.prompt for job in jobs] == [
        "A bedroom with a bed",
        "A living room with a TV",
    ]
    assert jobs[0].output_dir == tmp_path / "outputs" / "scene_001"
    assert jobs[1].output_dir == tmp_path / "outputs" / "scene_008"


def test_build_child_command_uses_fixed_scene_output_dir(tmp_path):
    job = module.PromptJob(
        scene_id=12,
        prompt="A study with a desk",
        output_dir=tmp_path / "scene_012",
    )

    command = module.build_child_command(job=job, run_name="subset")

    assert command[:3] == [sys.executable, "main.py", "+name=subset"]
    assert f"hydra.run.dir={job.output_dir}" in command
    assert f"experiment.single_scene_id={job.scene_id}" in command
    assert f"experiment.fixed_scene_output_dir={job.output_dir}" in command
    prompt_override = next(
        part for part in command if part.startswith("experiment.single_prompt=")
    )
    assert "A study with a desk" in prompt_override


def test_build_child_command_forwards_materials_retrieval_server_port(tmp_path):
    job = module.PromptJob(
        scene_id=12,
        prompt="A study with a desk",
        output_dir=tmp_path / "scene_012",
    )

    command = module.build_child_command(
        job=job,
        run_name="subset",
        materials_retrieval_server_port=6992,
    )

    assert "experiment.materials_retrieval_server.port=6992" in command


def test_resolve_cleanup_ports_uses_command_overrides():
    command = [
        sys.executable,
        "main.py",
        "experiment.geometry_generation_server.port=8105",
        "experiment.materials_retrieval_server.port=8108",
    ]

    ports = module.resolve_cleanup_ports(
        command=command,
        default_ports={
            "geometry_generation_server": 7005,
            "hssd_retrieval_server": 7006,
            "articulated_retrieval_server": 7007,
            "materials_retrieval_server": 7008,
            "objaverse_retrieval_server": 7009,
        },
    )

    assert ports == (8105, 7006, 7007, 8108, 7009)


def test_run_job_forces_cleanup_after_completion_marker(tmp_path, monkeypatch):
    job = module.PromptJob(
        scene_id=3,
        prompt="A compact bedroom",
        output_dir=tmp_path / "scene_003",
    )

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    build_child_command_calls = []
    monkeypatch.setattr(
        module,
        "build_child_command",
        lambda *args, **kwargs: build_child_command_calls.append(kwargs) or [
            sys.executable,
            "-c",
            (
                "import time; "
                "print('ALL SCENES COMPLETED!', flush=True); "
                "time.sleep(60)"
            ),
        ],
    )
    monkeypatch.setattr(module, "resolve_cleanup_ports", lambda command: (8123, 8124))
    cleanup_calls = []
    monkeypatch.setattr(
        module,
        "cleanup_listening_ports",
        lambda ports: cleanup_calls.append(tuple(ports)) or [],
    )

    success = module.run_job(
        job=job,
        run_name="subset",
        cuda_visible_devices="0",
        grace_seconds=0.1,
        overwrite=False,
        materials_retrieval_server_port=6992,
    )

    assert success is True
    assert build_child_command_calls == [
        {
            "materials_retrieval_server_port": 6992,
            "start_stage": None,
            "stop_stage": None,
        }
    ]
    assert cleanup_calls == [(8123, 8124)]
    log_text = (job.output_dir / "sequential_runner.log").read_text()
    assert "[cleanup_ports] (8123, 8124)" in log_text
    assert "completion marker seen" in log_text
    assert "child did not exit after completion marker" in log_text


def test_run_job_forces_cleanup_after_scene_completion_marker(tmp_path, monkeypatch):
    job = module.PromptJob(
        scene_id=4,
        prompt="A bright studio apartment",
        output_dir=tmp_path / "scene_004",
    )

    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    build_child_command_calls = []
    monkeypatch.setattr(
        module,
        "build_child_command",
        lambda *args, **kwargs: build_child_command_calls.append(kwargs) or [
            sys.executable,
            "-c",
            (
                "import time; "
                "print('Scene generation completed successfully in 0:00:01', flush=True); "
                "time.sleep(60)"
            ),
        ],
    )
    monkeypatch.setattr(module, "resolve_cleanup_ports", lambda command: (8451, 8452))
    cleanup_calls = []
    monkeypatch.setattr(
        module,
        "cleanup_listening_ports",
        lambda ports: cleanup_calls.append(tuple(ports)) or [],
    )

    success = module.run_job(
        job=job,
        run_name="subset",
        cuda_visible_devices="0",
        grace_seconds=0.1,
        overwrite=False,
        materials_retrieval_server_port=6992,
    )

    assert success is True
    assert build_child_command_calls == [
        {
            "materials_retrieval_server_port": 6992,
            "start_stage": None,
            "stop_stage": None,
        }
    ]
    assert cleanup_calls == [(8451, 8452)]
    log_text = (job.output_dir / "sequential_runner.log").read_text()
    assert "[cleanup_ports] (8451, 8452)" in log_text
    assert "completion marker seen" in log_text
    assert "child did not exit after completion marker" in log_text
