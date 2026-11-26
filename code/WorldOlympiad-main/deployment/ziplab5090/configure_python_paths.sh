#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
workspace=${WORLDOLYMPIAD_WORKSPACE:-$(cd -- "$script_dir/../../../.." && pwd -P)}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
scenecode_root=${SCENECODE_ROOT:-$workspace/code/scenecode_pybullet_minimal}

for env_name in autoresearch scenecode worldolympiad robotwin dream2flow dream2flow-planner lingbot robolab-isaac51; do
  python_bin="$workspace/envs/$env_name/bin/python"
  [[ -x $python_bin ]] || continue
  site_dir=$("$python_bin" -c 'import site; print(site.getsitepackages()[0])')
  install -d "$site_dir"
  printf '%s\n%s\n' "$world_repo" "$scenecode_root" > "$site_dir/worldolympiad2_paths.pth"
  printf 'configured %s\n' "$site_dir/worldolympiad2_paths.pth"
done

autoresearch_python="$workspace/envs/autoresearch/bin/python"
if [[ -x $autoresearch_python ]]; then
  "$autoresearch_python" -c 'import autoresearch; print("autoresearch:", autoresearch.__file__)'
fi
