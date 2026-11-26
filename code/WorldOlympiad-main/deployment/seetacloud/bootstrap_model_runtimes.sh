#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
workspace=${WORLDOLYMPIAD_WORKSPACE:-$(cd -- "$script_dir/../../../.." && pwd -P)}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
uv_bin=${UV_BIN:-$workspace/envs/bootstrap/bin/uv}
target=${1:-all}

[[ -x $uv_bin ]] || { echo "uv is missing: $uv_bin" >&2; exit 2; }
[[ $target == all || $target == dream2flow || $target == lingbot || $target == cosmos ]] || {
  echo "usage: bootstrap_model_runtimes.sh [all|dream2flow|lingbot|cosmos]" >&2
  exit 2
}

bootstrap_dream2flow() {
  local source=$workspace/code/Dream2Flow-main-cae-min64-vis08-v1
  local cotracker_source=$workspace/code/cotracker3-vis08-v1
  local environment=$workspace/envs/dream2flow
  local wheel_cache=$workspace/cache/wheels
  local pyg_lib_wheel=$wheel_cache/pyg_lib-0.7.0+pt29cu128-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
  local torch_home=$workspace/runtime/torch-home-cotracker-vis08
  [[ -f $source/pyproject.toml && -f $cotracker_source/checkpoints/scaled_offline.pth ]] || {
    echo "Dream2Flow or CoTracker source is incomplete" >&2
    exit 2
  }
  [[ -f $pyg_lib_wheel ]] || {
    echo "The pinned official pyg-lib wheel is missing: $pyg_lib_wheel" >&2
    exit 2
  }
  if grep -q 'thr = 0.9' "$cotracker_source/cotracker/predictor.py"; then
    patch --directory="$cotracker_source" --forward --strip=1 \
      < "$world_repo/deployment/ziplab5090/patches/cotracker_visibility_08.patch"
  fi
  if [[ ! -x $environment/bin/python ]]; then
    /root/miniconda3/bin/conda create -y -p "$environment" --override-channels \
      -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
      -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
      python=3.10 pip
  fi
  "$uv_bin" pip install --python "$environment/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
  "$uv_bin" pip install --python "$environment/bin/python" hatchling
  SAM2_BUILD_CUDA=0 "$uv_bin" pip install --python "$environment/bin/python" \
    --no-build-isolation \
    torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 \
    numpy==2.2.6 scipy==1.15.3 opencv-python==5.0.0.93 \
    transformers==4.55.2 tensordict==0.13.0 einops==0.8.2 \
    hydra-core==1.3.5 torch-geometric==2.8.0.post1 \
    'sam-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4' \
    viser==1.0.30 google-genai==2.16.0 python-dotenv \
    Pillow==12.3.0 PyYAML==6.0.3 imageio==2.37.4 imageio-ffmpeg==0.6.0
  "$uv_bin" pip install --python "$environment/bin/python" --no-deps "$pyg_lib_wheel"

  mkdir -p "$torch_home/hub/checkpoints"
  ln -sfn "$cotracker_source" "$torch_home/hub/facebookresearch_co-tracker_main"
  ln -sfn "$cotracker_source/checkpoints/scaled_offline.pth" \
    "$torch_home/hub/checkpoints/scaled_offline.pth"
  : > "$torch_home/hub/trusted_list"

  PYTHONPATH="$source" TORCH_HOME="$torch_home" "$environment/bin/python" - <<'PY'
import torch
import torch_geometric.typing
from torch_geometric.nn.pool import fps

assert torch.cuda.is_available()
major, minor = torch.cuda.get_device_capability()
assert f"sm_{major}{minor}" in torch.cuda.get_arch_list()
assert torch_geometric.typing.WITH_FPS
points = torch.arange(0, 192, dtype=torch.float32, device="cuda").reshape(64, 3)
batch = torch.zeros(64, dtype=torch.long, device="cuda")
first = fps(points, batch=batch, ratio=0.5, random_start=False)
second = fps(points, batch=batch, ratio=0.5, random_start=False)
assert torch.equal(first, second) and first.numel() == 32
print("Dream2Flow runtime passed", torch.__version__, torch.cuda.get_device_name())
PY
}

bootstrap_lingbot() {
  local source=$workspace/code/lingbot-va
  local environment=$workspace/envs/lingbot
  [[ -f $source/requirements.txt && -f $source/wan_va/wan_va_server.py ]] || {
    echo "LingBot-VA source is incomplete: $source" >&2
    exit 2
  }
  if [[ ! -x $environment/bin/python ]]; then
    "$uv_bin" venv --python /root/miniconda3/bin/python3 "$environment"
  fi
  "$uv_bin" pip install --python "$environment/bin/python" \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
  "$uv_bin" pip install --python "$environment/bin/python" \
    diffusers==0.36.0 transformers==4.55.2 accelerate einops easydict \
    'numpy==1.26.4' tqdm 'imageio[ffmpeg]' websockets msgpack \
    opencv-python-headless matplotlib ftfy safetensors Pillow

  # The released base config selects PyTorch SDPA.  Make the upstream optional
  # flash-attn import match that runtime choice without changing model math.
  if ! grep -q 'flash_attn_func = None' "$source/wan_va/modules/model.py"; then
    patch --directory="$source" --forward --strip=1 \
      < "$world_repo/deployment/ziplab5090/lingbot_torch_attention.patch"
  fi
  PYTHONPATH="$source/wan_va" "$environment/bin/python" - <<'PY'
import torch
from configs import VA_CONFIGS
from modules.model import WanAttention

assert torch.cuda.is_available()
assert "demo_i2av" in VA_CONFIGS
assert WanAttention(dim=128, heads=2, attn_mode="torch").attn_op is not None
major, minor = torch.cuda.get_device_capability()
assert f"sm_{major}{minor}" in torch.cuda.get_arch_list()
print("LingBot-VA-base runtime passed", torch.__version__, torch.cuda.get_device_name())
PY
}

bootstrap_cosmos() {
  local source=$workspace/code/cosmos-framework
  local environment=$workspace/envs/cosmos
  local wheel_cache=$workspace/cache/wheels
  local flash_attn_wheel=$wheel_cache/flash_attn-2.7.4.post1+cu128.torch210-cp313-cp313-linux_x86_64.whl
  local flash_attn_3_wheel=$wheel_cache/flash_attn_3_nv-1.0.3+cu128.torch210-cp39-abi3-linux_x86_64.whl
  local megatron_archive=$workspace/cache/source/Megatron-LM-de56227b26ba56de885897361272513697de7dc6.tar.gz
  local -a sync_args=(
    --frozen
    --no-dev
    --all-extras
    --group cu128-train
  )
  [[ -f $source/pyproject.toml && -f $source/uv.lock ]] || {
    echo "Cosmos Framework source is incomplete: $source" >&2
    exit 2
  }
  if [[ ! -x $environment/bin/python ]]; then
    /root/miniconda3/bin/conda create -y -p "$environment" --override-channels \
      -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
      -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
      python=3.13 pip
  fi
  # GitHub release redirects are unreliable on some GPU hosts.  Sync the
  # complete official inference/training environment while omitting only the
  # two cached release wheels, then install those exact locked builds last so
  # uv cannot remove them as part of environment reconciliation.
  if [[ -f $flash_attn_wheel && -f $flash_attn_3_wheel ]]; then
    sync_args+=(
      --no-install-package flash-attn
      --no-install-package flash-attn-3-nv
    )
  fi
  # The lockfile pins Megatron-LM to this exact upstream commit.  When the
  # host cannot sustain a GitHub git fetch, an archived copy of that same
  # commit avoids a network-only failure without changing the dependency.
  if [[ -f $megatron_archive ]]; then
    sync_args+=(--no-install-package megatron-core)
  fi
  (
    cd "$source"
    export UV_PROJECT_ENVIRONMENT=$environment
    export UV_PYTHON=$environment/bin/python
    export UV_PYTHON_DOWNLOADS=never
    "$uv_bin" sync "${sync_args[@]}"
  )
  if [[ -f $flash_attn_wheel && -f $flash_attn_3_wheel ]]; then
    "$uv_bin" pip install --python "$environment/bin/python" --no-deps \
      "$flash_attn_wheel" "$flash_attn_3_wheel"
  fi
  if [[ -f $megatron_archive ]]; then
    "$uv_bin" pip install --python "$environment/bin/python" --no-deps \
      "$megatron_archive"
  fi
  "$environment/bin/python" - <<'PY'
import torch
from cosmos_framework.scripts.action_policy_server_robolab import RobolabServerArgs

assert torch.cuda.is_available()
major, minor = torch.cuda.get_device_capability()
assert f"sm_{major}{minor}" in torch.cuda.get_arch_list()
assert RobolabServerArgs is not None
print("Cosmos3-Nano-DROID runtime passed", torch.__version__, torch.cuda.get_device_name())
PY
}

if [[ $target == all || $target == dream2flow ]]; then
  bootstrap_dream2flow
fi
if [[ $target == all || $target == lingbot ]]; then
  bootstrap_lingbot
fi
if [[ $target == all || $target == cosmos ]]; then
  bootstrap_cosmos
fi
