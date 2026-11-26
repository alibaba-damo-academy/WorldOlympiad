#!/usr/bin/env bash
set -euo pipefail

workspace=${WORLDOLYMPIAD_WORKSPACE:-$(pwd)}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
evaluation_python=${WORLDEVAL_PYTHON:-$workspace/envs/worldolympiad/bin/python}
uv_bin=${UV_BIN:-$workspace/envs/bin/uv}
gsplat_commit=0b4dddf04cb687367602c01196913cde6a743d70
glm_commit=33b4a621a697a305bc3a7610d290677b96beb181
source_root=${WORLDEVAL_GSPLAT_SOURCE:-$workspace/code/deps/gsplat-$gsplat_commit}

[[ -x $evaluation_python ]] || { echo "Evaluator Python is missing: $evaluation_python" >&2; exit 2; }
[[ -x $uv_bin ]] || { echo "uv is missing: $uv_bin" >&2; exit 2; }
cuda_home=${CUDA_HOME:-}
nvcc_path=${NVCC:-}
if [[ -z $nvcc_path ]]; then
  if [[ -n $cuda_home && -x $cuda_home/bin/nvcc ]]; then
    nvcc_path=$cuda_home/bin/nvcc
  else
    nvcc_path=$(command -v nvcc || true)
  fi
fi
[[ -n $nvcc_path && -x $nvcc_path ]] || { echo "CUDA 12.8 nvcc is missing; set CUDA_HOME or PATH." >&2; exit 2; }
nvcc_path=$(readlink -f -- "$nvcc_path")
if [[ -z $cuda_home ]]; then cuda_home=${nvcc_path%/bin/nvcc}; fi
cuda_release=$($nvcc_path --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -n 1)
[[ $cuda_release == 12.8 ]] || { echo "CUDA 12.8 is required; found ${cuda_release:-unknown}." >&2; exit 2; }

if [[ ! -f $source_root/.worldolympiad_source_commit ]]; then
  [[ ! -e $source_root ]] || {
    echo "gsplat source exists without the registered commit marker: $source_root" >&2
    exit 2
  }
  scratch=$(mktemp -d /tmp/worldolympiad-gsplat.XXXXXX)
  cleanup() { rm -rf -- "$scratch"; }
  trap cleanup EXIT
  curl --http1.1 --fail --location --retry 8 --retry-all-errors \
    --output "$scratch/gsplat.tar.gz" \
    "https://codeload.github.com/nerfstudio-project/gsplat/tar.gz/$gsplat_commit"
  tar -xzf "$scratch/gsplat.tar.gz" -C "$scratch"
  extracted="$scratch/gsplat-$gsplat_commit"
  glm_dir="$extracted/gsplat/cuda/csrc/third_party/glm"
  curl --http1.1 --fail --location --retry 8 --retry-all-errors \
    --output "$scratch/glm.tar.gz" \
    "https://codeload.github.com/g-truc/glm/tar.gz/$glm_commit"
  tar -xzf "$scratch/glm.tar.gz" --strip-components=1 -C "$glm_dir"
  [[ -f $glm_dir/glm/gtc/type_ptr.hpp ]] || { echo "GLM source is incomplete" >&2; exit 3; }
  printf '%s\n' "$glm_commit" > "$glm_dir/.worldolympiad_source_commit"
  printf '%s\n' "$gsplat_commit" > "$extracted/.worldolympiad_source_commit"
  mkdir -p "$(dirname "$source_root")"
  mv "$extracted" "$source_root"
fi
[[ $(<"$source_root/.worldolympiad_source_commit") == "$gsplat_commit" ]] || {
  echo "Unexpected gsplat source commit marker" >&2
  exit 2
}
[[ $(<"$source_root/gsplat/cuda/csrc/third_party/glm/.worldolympiad_source_commit") == "$glm_commit" ]] || {
  echo "Unexpected GLM source commit marker" >&2
  exit 2
}

export PATH="$(dirname "$evaluation_python"):$PATH"
export CUDA_HOME=$cuda_home NVCC=$nvcc_path
export MAX_JOBS=${MAX_JOBS:-4}
"$uv_bin" pip install --python "$evaluation_python" --no-deps --no-build-isolation \
  --reinstall "$source_root"

"$evaluation_python" - <<'PY'
import torch
from gsplat import csrc, rasterization

means = torch.tensor([[0.0, 0.0, 2.0]], device="cuda")
quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
scales = torch.full((1, 3), 0.1, device="cuda")
opacities = torch.ones(1, device="cuda")
colors = torch.tensor([[1.0, 0.0, 0.0]], device="cuda")
viewmats = torch.eye(4, device="cuda").unsqueeze(0)
intrinsics = torch.tensor(
    [[[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]],
    device="cuda",
)
render, alpha, _ = rasterization(
    means, quats, scales, opacities, colors, viewmats, intrinsics, 64, 64
)
torch.cuda.synchronize()
assert render.shape == (1, 64, 64, 3) and alpha.max().item() > 0
print("WorldOlympiad gsplat CUDA extension passed")
PY
