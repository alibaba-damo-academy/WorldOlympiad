#!/usr/bin/env bash
set -euo pipefail

workspace=${WORLDOLYMPIAD_WORKSPACE:-/mnt/data/v-wangchengzhi/worldolympiad2}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
case_root=${1:?usage: run_worldolympiad_cae.sh CASE_ROOT [OUTPUT_JSON]}
case_root=$(realpath "$case_root")
output_json=${2:-$case_root/official_dream2flow_action_report.json}
python_bin=${WORLDOLYMPIAD_PYTHON:-$workspace/envs/worldolympiad/bin/python}

[[ -x $python_bin ]] || {
  echo "WorldOlympiad evaluator Python is missing: $python_bin" >&2
  exit 2
}
[[ -d $case_root ]] || {
  echo "Case root is missing: $case_root" >&2
  exit 2
}

PYTHONPATH="$world_repo${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" "$world_repo/autoresearch/official_dream2flow_action_report.py" \
  "$case_root" --output "$output_json"

"$python_bin" - "$output_json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
print(json.dumps({
    "output": str(path),
    "case_count": report["case_count"],
    "scored_count": report["scored_count"],
    "coverage_failure_count": report["coverage_failure_count"],
    "trajectory_source": "official_dream2flow_trajectory_optimizer_ee_trajectory",
}, sort_keys=True))
PY
