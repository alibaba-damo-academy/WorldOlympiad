from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_run_eval_preserves_caller_private_env_file():
    lines = (ROOT / "scripts/run_eval.sh").read_text().splitlines()
    assignments = [line for line in lines if line.startswith("export WO_AGENT_ENV_FILE=")]
    assert assignments == [
        'export WO_AGENT_ENV_FILE="${WO_AGENT_ENV_FILE:-$ROOT/.secrets/scenecode.env}"'
    ]


def test_run_eval_exports_project_local_ffmpeg_without_overriding_user_choice():
    script = (ROOT / "scripts/run_eval.sh").read_text()
    assert 'FFMPEG_PY=$ROOT/envs/lingbot/bin/python' in script
    assert 'imageio_ffmpeg.get_ffmpeg_exe()' in script
    assert 'if [[ -z "${WO_FFMPEG:-}" ]]' in script
    assert 'export WO_FFMPEG' in script
