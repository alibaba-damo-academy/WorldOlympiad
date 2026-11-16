#!/usr/bin/env bash
set -euo pipefail

workspace=${WORLDOLYMPIAD_WORKSPACE:-/mnt/data/v-wangchengzhi/worldolympiad2}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
environment_dir=${WORLDEVAL_ENV_DIR:-$workspace/envs/worldolympiad}
uv_bin=${UV_BIN:-$workspace/envs/bin/uv}
uv_cache=${UV_CACHE_DIR:-$workspace/runtime/uv-cache-worldeval}

[[ -x $uv_bin ]] || { echo "uv is missing: $uv_bin" >&2; exit 2; }
[[ -x $environment_dir/bin/python ]] || {
  echo "WorldOlympiad evaluation environment is missing: $environment_dir" >&2
  exit 2
}
[[ -f $world_repo/requirements-worldeval-torch-cu128.txt ]] || {
  echo "WorldOlympiad CUDA 12.8 requirements are missing" >&2
  exit 2
}

mkdir -p "$uv_cache"
export UV_CACHE_DIR=$uv_cache
"$uv_bin" pip uninstall --python "$environment_dir/bin/python" xformers || true
"$uv_bin" pip install --python "$environment_dir/bin/python" \
  -r "$world_repo/requirements-worldeval-torch-cu128.txt"
bash "$world_repo/deployment/ziplab5090/preflight_worldolympiad_evaluation.sh"
