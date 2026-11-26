#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
workspace=${WORLDOLYMPIAD_WORKSPACE:-$(cd -- "$script_dir/../../../.." && pwd -P)}
bootstrap_env=${BOOTSTRAP_ENV:-$workspace/envs/bootstrap}
hf_bin=${HF_BIN:-$bootstrap_env/bin/hf}
uvx_bin=${UVX_BIN:-$bootstrap_env/bin/uvx}
uv_python=${UV_PYTHON:-/root/miniconda3/bin/python}
uv_index=${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple}
hf_endpoint=${HF_ENDPOINT:-https://hf-mirror.com}
hf_workers=${HF_DOWNLOAD_WORKERS:-8}

[[ -x $hf_bin ]] || {
  echo "Hugging Face CLI is missing: $hf_bin" >&2
  exit 2
}

mkdir -p "$workspace/models" "$workspace/runtime/model-downloads"
export HF_ENDPOINT=$hf_endpoint
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-180}
export UV_DEFAULT_INDEX=$uv_index

download_hf() {
  local repo_id=$1
  local revision=$2
  local destination=$3
  shift 3
  local marker=$destination/.worldolympiad_download_complete
  if [[ -f $marker ]]; then
    echo "reusing model: $destination"
    return
  fi
  mkdir -p "$destination"
  "$hf_bin" download "$repo_id" \
    --revision "$revision" \
    --local-dir "$destination" \
    --max-workers "$hf_workers" \
    "$@"
  printf 'repo_id=%s\nrevision=%s\n' "$repo_id" "$revision" > "$marker"
}

download_hf \
  Robbyant/lingbot-va-base main \
  "$workspace/models/lingbot-va-base"
download_hf \
  nvidia/Cosmos3-Nano-Policy-DROID \
  6706d7680581c255ff61e0f3bb49d90eac55c79e \
  "$workspace/models/Cosmos3-Nano-Policy-DROID"
download_hf \
  Wan-AI/Wan2.2-TI2V-5B \
  921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  "$workspace/models/Wan2.2-TI2V-5B-vae" \
  --include Wan2.2_VAE.pth
download_hf \
  IDEA-Research/grounding-dino-base \
  12bdfa3120f3e7ec7b434d90674b3396eccf88eb \
  "$workspace/models/grounding-dino-base"
download_hf \
  facebook/sam2-hiera-large \
  e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251 \
  "$workspace/models/sam2-hiera-large"
download_hf \
  Yuxihenry/SpatialTrackerV2_Front \
  2e7164a93770cb934861d7de579bb81f58ad0b03 \
  "$workspace/models/SpatialTrackerV2_Front"
download_hf \
  Yuxihenry/SpatialTrackerV2-Offline \
  76e275b00f9c57dab71d46544df5255d4538106d \
  "$workspace/models/SpatialTrackerV2-Offline"

[[ -x $uvx_bin ]] || {
  echo "uvx is missing: $uvx_bin" >&2
  exit 2
}

download_modelscope() {
  local repo_id=$1
  local destination=$2
  local required_file=$3
  local partial="${destination}.partial"
  if [[ -f $destination/$required_file ]]; then
    echo "reusing evaluator model: $destination"
    return
  fi
  [[ ! -e $destination ]] || {
    echo "Incomplete evaluator destination must be quarantined first: $destination" >&2
    exit 2
  }
  mkdir -p "$partial"
  "$uvx_bin" --python "$uv_python" --from modelscope modelscope download \
    --model "$repo_id" --local_dir "$partial"
  [[ -f $partial/$required_file ]] || {
    echo "Downloaded evaluator model is incomplete: $repo_id" >&2
    exit 3
  }
  mv "$partial" "$destination"
}

evaluator_root=$workspace/models/worldolympiad-evaluators
mkdir -p "$evaluator_root"
download_modelscope Qwen/Qwen3-VL-8B-Instruct "$evaluator_root/QwenVL" config.json
download_modelscope depth-anything/DA3NESTED-GIANT-LARGE-1.1 "$evaluator_root/da3" config.json
download_modelscope facebook/sam3 "$evaluator_root/sam3" sam3.pt

echo "WorldOlympiad2 model set is ready"
