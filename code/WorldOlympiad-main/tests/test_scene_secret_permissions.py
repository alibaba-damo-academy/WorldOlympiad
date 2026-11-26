import os
from pathlib import Path
import subprocess


def test_scene_launcher_tightens_selected_api_env_permissions(tmp_path):
    launcher = Path(__file__).resolve().parents[1] / "deployment/ziplab5090/run_scenecode_case.sh"
    text = launcher.read_text()
    block = text.split("[[ -f $secret_file ]]", 1)[1].split("scenecode_preflight()", 1)[0]
    env_file = tmp_path / "managed-api.env"
    env_file.write_text("OPENAI_API_KEY=not-read-by-test\n")
    env_file.chmod(0o644)
    env = dict(os.environ, secret_file=str(env_file))

    result = subprocess.run(
        ["bash", "-c", "secret_file=$1; [[ -f $secret_file ]]" + block,
         "test", str(env_file)],
        env=env, capture_output=True, text=True, check=True,
    )

    assert result.stderr == ""
    assert env_file.stat().st_mode & 0o777 == 0o600
