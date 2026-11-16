import os
from pathlib import Path
import subprocess


def test_scene_launcher_limits_math_threads_before_dependency_preflight():
    script = Path(__file__).resolve().parents[2] / "deployment/ziplab5090/run_scenecode_case.sh"
    prefix = script.read_text().split("batch_id=", 1)[0]
    env = dict(os.environ, OMP_NUM_THREADS="64", OPENBLAS_NUM_THREADS="64", MKL_NUM_THREADS="64", NUMEXPR_NUM_THREADS="64")
    code = prefix + '\nprintf "%s %s %s %s" "$OMP_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$MKL_NUM_THREADS" "$NUMEXPR_NUM_THREADS"\n'
    result = subprocess.run(["bash", "-c", code], env=env, capture_output=True, text=True, check=True)
    assert result.stdout == "1 1 1 1"


def test_scene_launcher_uses_managed_agent_env_path_without_reading_it():
    script = Path(__file__).resolve().parents[2] / "deployment/ziplab5090/run_scenecode_case.sh"
    prefix = script.read_text().split("batch_id=", 1)[0]
    env = dict(os.environ)
    for key in ("SCENECODE_ENV_FILE", "WO_AGENT_ENV_FILE", "WORLDOLYMPIAD_WORKSPACE"):
        env.pop(key, None)
    env["WO_AGENT_ENV_FILE"] = "/private/managed/agent.env"
    result = subprocess.run(
        ["bash", "-c", prefix + 'printf "%s" "$secret_file"\n'],
        env=env, capture_output=True, text=True, check=True,
    )
    assert result.stdout == "/private/managed/agent.env"


def test_scene_launcher_explicit_scenecode_env_path_has_precedence():
    script = Path(__file__).resolve().parents[2] / "deployment/ziplab5090/run_scenecode_case.sh"
    prefix = script.read_text().split("batch_id=", 1)[0]
    env = dict(os.environ, SCENECODE_ENV_FILE="/explicit/scenecode.env", WO_AGENT_ENV_FILE="/managed/agent.env")
    result = subprocess.run(
        ["bash", "-c", prefix + 'printf "%s" "$secret_file"\n'],
        env=env, capture_output=True, text=True, check=True,
    )
    assert result.stdout == "/explicit/scenecode.env"
