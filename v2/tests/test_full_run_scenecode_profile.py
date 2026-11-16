from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_WORLD = ROOT / "code/WorldOlympiad-main"


def test_full_run_passes_agent_model_to_run_scoped_scenecode_profile(tmp_path: Path) -> None:
    workspace = tmp_path / "candidate"
    scripts = workspace / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/run_eval.sh", scripts / "run_eval.sh")
    shutil.copy2(ROOT / "scripts/prepare_scenecode_profile.py", scripts / "prepare_scenecode_profile.py")

    world = workspace / "code/WorldOlympiad-main"
    world.parent.mkdir(parents=True)
    world.symlink_to(SOURCE_WORLD, target_is_directory=True)

    profile_source = workspace / "runtime/scenecode/scenecode/code_object/agent_framework"
    profile_source.mkdir(parents=True)
    for name in ("config.yaml", "config_WallArt.yaml", "config_simplemanip.yaml", "config_structmanip.yaml"):
        (profile_source / name).write_text(
            "model:\n  llm:\n    provider: openai\n    model_name: gpt-5.6-sol\n",
            encoding="utf-8",
        )

    config = workspace / "config/two-interface.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({
        "agent_api": {"model": "gpt-6-sol"},
        "tested_model": {"scenecode_runtime": "runtime/scenecode"},
    }), encoding="utf-8")

    python = workspace / "envs/autoresearch/bin/python"
    python.parent.mkdir(parents=True)
    real_python = shlex.quote(sys.executable)
    python.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == "-m" && "${{2:-}}" == "wo.agent.launch" ]]; then
  exec {real_python} - "$@" <<'PY'
import json, os, sys
with open(os.environ["WO_TEST_CAPTURE"], "w", encoding="utf-8") as stream:
    json.dump({{
        "argv": sys.argv[1:],
        "profile_root": os.environ.get("WO_SCENECODE_SOURCE_CONFIG_ROOT"),
        "resource_phase": os.environ.get("WO_RESOURCE_PHASE"),
    }}, stream)
PY
fi
exec {real_python} "$@"
""")
    python.chmod(0o755)

    fake_bin = workspace / "test-bin"
    fake_bin.mkdir()
    fake_python3 = fake_bin / "python3"
    fake_python3.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
case "${{1:-}}" in
  */validate_run.py|*/prepare_rsi_base.py) exit 0 ;;
esac
exec {real_python} "$@"
""")
    fake_python3.chmod(0o755)
    preflight = scripts / "preflight.sh"
    preflight.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    preflight.chmod(0o755)

    capture = tmp_path / "launch.json"
    env = os.environ.copy()
    env.update({
        "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
        "WO_FFMPEG": "/bin/true",
        "WO_TEST_CAPTURE": str(capture),
    })
    result = subprocess.run(
        ["bash", str(scripts / "run_eval.sh"), "--phase", "full", "--run", "full-profile-test"],
        cwd=workspace, env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SceneCode source-profile model=gpt-6-sol; matches Agent API model" in result.stdout

    launched = json.loads(capture.read_text(encoding="utf-8"))
    profile = workspace / "outputs/wo_runs/full-profile-test/scenecode_profile"
    goal = workspace / "outputs/evaluator/goals/full-profile-test_evaluation.md"
    assert launched["profile_root"] == str(profile)
    assert launched["resource_phase"] == "evaluation"
    assert launched["argv"][0:2] == ["-m", "wo.agent.launch"]
    assert launched["argv"][launched["argv"].index("--goal") + 1] == str(goal)
    assert "--through-reference" not in launched["argv"]
    assert json.loads((profile / "profile_manifest.json").read_text())["agent_model"] == "gpt-6-sol"
    goal_text = goal.read_text(encoding="utf-8")
    assert "task to scored episode" in goal_text
    assert "source_config_root" in goal_text and str(profile) in goal_text
