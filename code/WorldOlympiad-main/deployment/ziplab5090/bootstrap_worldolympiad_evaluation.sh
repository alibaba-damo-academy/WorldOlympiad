#!/usr/bin/env bash
set -euo pipefail

workspace=${WORLDOLYMPIAD_WORKSPACE:-$(pwd)}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
environment_dir=${WORLDEVAL_ENV_DIR:-$workspace/envs/worldolympiad}
uv_bin=${UV_BIN:-$workspace/envs/bin/uv}
uv_cache=${UV_CACHE_DIR:-$workspace/runtime/uv-cache-worldeval}
python_version=${WORLDEVAL_PYTHON_VERSION:-3.11}
base_python=${WORLDEVAL_BASE_PYTHON:-$workspace/envs/scenecode/bin/python}
clip_source=${WORLDEVAL_CLIP_SOURCE:-git+https://github.com/ultralytics/CLIP.git}
partial="${environment_dir}.partial"

[[ -x $uv_bin ]] || { echo "uv is missing: $uv_bin" >&2; exit 2; }
[[ -f $world_repo/requirements-worldeval.txt ]] || {
  echo "WorldOlympiad evaluation requirements are missing" >&2
  exit 2
}
[[ -f $world_repo/requirements-worldeval-torch-cu128.txt ]] || {
  echo "WorldOlympiad CUDA 12.8 requirements are missing" >&2
  exit 2
}

mkdir -p "$uv_cache" "$(dirname "$environment_dir")"
export UV_CACHE_DIR=$uv_cache

if [[ ! -x $partial/bin/python ]]; then
  [[ ! -e $environment_dir ]] || {
    echo "Evaluation environment already exists but is incomplete: $environment_dir" >&2
    exit 2
  }
  if [[ -x $base_python ]] && "$base_python" -c \
    'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
    "$uv_bin" venv --python "$base_python" "$partial"
  else
    "$uv_bin" venv --python "$python_version" "$partial"
  fi
fi

"$uv_bin" pip install --python "$partial/bin/python" \
  -r "$world_repo/requirements-worldeval-torch-cu128.txt"
"$uv_bin" pip install --python "$partial/bin/python" \
  -r "$world_repo/requirements-worldeval.txt"
"$uv_bin" pip install --python "$partial/bin/python" \
  "$clip_source"

WORLDOLYMPIAD_WORKSPACE="$workspace" \
WORLD_REPO="$world_repo" \
WORLDEVAL_PYTHON="$partial/bin/python" \
UV_BIN="$uv_bin" \
  bash "$world_repo/deployment/ziplab5090/install_worldolympiad_gsplat.sh"

da3_source=$world_repo/Depth-Anything-3
if [[ ! -f $da3_source/pyproject.toml ]]; then
  [[ ! -e $da3_source ]] || {
    echo "Depth-Anything-3 path exists but is not a valid checkout: $da3_source" >&2
    exit 2
  }
  git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git "$da3_source"
fi
# DA3's upstream application extras permit an unconstrained xformers upgrade,
# which can replace the evaluator's pinned CUDA/PyTorch stack.  Runtime
# dependencies are curated above; install only the audited source package.
"$uv_bin" pip install --no-deps --python "$partial/bin/python" -e "$da3_source"

"$partial/bin/python" - <<'PY'
import clip
import cv2
import numpy
import open3d
import timm
import torch
import transformers
import ultralytics
from depth_anything_3.api import DepthAnything3
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

assert torch.cuda.is_available(), "CUDA is not available in the evaluation runtime"
assert callable(clip.simple_tokenizer.SimpleTokenizer()), (
    "SAM3 requires the callable tokenizer from ultralytics/CLIP"
)
major, minor = torch.cuda.get_device_capability()
device_arch = f"sm_{major}{minor}"
assert device_arch in torch.cuda.get_arch_list(), (
    f"PyTorch {torch.__version__} does not contain kernels for {device_arch}: "
    f"{torch.cuda.get_arch_list()}"
)
assert (torch.ones(1, device="cuda") + 1).item() == 2
print("WorldOlympiad evaluation imports passed")
PY

[[ ! -e $environment_dir ]] || {
  echo "Refusing to overwrite evaluation environment: $environment_dir" >&2
  exit 2
}
mv "$partial" "$environment_dir"
echo "$environment_dir/bin/python"
