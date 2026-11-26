#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
mode=${1:---install}
case "$mode" in
  --install|--online-install|--check-only) ;;
  *) echo "usage: setup_runtime.sh [--install|--online-install|--check-only]" >&2; exit 2 ;;
esac
WORLD=$ROOT/code/WorldOlympiad-main
AUTORESEARCH_BOOTSTRAP_DIR=${AUTORESEARCH_BOOTSTRAP_DIR:-$ROOT/cache/bootstrap/autoresearch}
uv_bin=${UV_BIN:-$ROOT/envs/bootstrap/bin/uv}
if [[ ! -x $uv_bin ]]; then uv_bin=$(command -v uv || true); fi
python310=${PYTHON310:-$(command -v python3.10 || true)}
python311=${PYTHON311:-$(command -v python3.11 || true)}
python3 "$ROOT/scripts/setup_sources.py" --verify-only
python3 "$ROOT/scripts/setup_robotwin_assets.py" --check-only
for script in \
  deployment/ziplab5090/bootstrap_autoresearch.sh \
  deployment/ziplab5090/bootstrap_scenecode.sh \
  deployment/ziplab5090/bootstrap_robotwin_runtime.sh \
  deployment/ziplab5090/bootstrap_worldolympiad_evaluation.sh \
  deployment/ziplab5090/stage_scenecode_runtime.sh \
  deployment/ziplab5090/stage_scenecode_pybullet_bridge.sh; do
  [[ -f $WORLD/$script ]] || { echo "pinned bootstrap script is missing: $script" >&2; exit 2; }
done
for patch in \
  deployment/patches/robotwin_lingbot_native_camera.patch \
  deployment/patches/robotwin_lingbot_upstream_intrinsics.patch \
  deployment/ziplab5090/lingbot_torch_attention.patch; do
  [[ -f $WORLD/$patch ]] || { echo "locked runtime patch is missing: $patch" >&2; exit 2; }
done
if [[ $mode == --check-only ]]; then
  python3 "$ROOT/scripts/stage_runtime_sources.py" --verify-only
  [[ -f $ROOT/runtime/scenecode/.worldolympiad_scenecode_runtime_v1 ]] || {
    echo "SceneCode runtime copy has not been staged" >&2; exit 2;
  }
  "$ROOT/scripts/preflight.sh" runtimes
  exit 0
fi
runtime_names=(autoresearch worldeval scenecode robotwin lingbot)
all_ready=1 any_final=0
for name in "${runtime_names[@]}"; do
  if [[ -d $ROOT/envs/$name ]]; then any_final=1; else all_ready=0; fi
done
for path in "$ROOT/envs/autoresearch.partial" "$ROOT/envs/scenecode.partial" \
            "$ROOT/envs/robotwin.partial" "$ROOT/envs/lingbot.partial" \
            "$ROOT/envs/worldeval.partial" "$ROOT/envs/robotwin.partial"; do
  [[ ! -e $path && ! -L $path ]] || { echo "partial runtime exists and was preserved: $path" >&2; exit 2; }
done
if [[ $all_ready == 1 ]]; then
  # Reuse exact existing environments, while building this repository's
  # source-bound runtime overlays and SceneCode runtime in its own runtime/.
  python3 "$ROOT/scripts/stage_runtime_sources.py"
  WORLDOLYMPIAD_WORKSPACE="$ROOT" WORLD_REPO="$WORLD" \
  SCENECODE_SOURCE_ROOT="$ROOT/code/SceneCode" \
  SCENECODE_RUNTIME_ROOT="$ROOT/runtime/scenecode" \
  SCENECODE_PYTHON="$ROOT/envs/scenecode/bin/python" \
    bash "$WORLD/deployment/ziplab5090/stage_scenecode_runtime.sh"
  if [[ -e $WORLD/Depth-Anything-3 || -L $WORLD/Depth-Anything-3 ]]; then
    [[ -L $WORLD/Depth-Anything-3 && $(readlink -f "$WORLD/Depth-Anything-3") == "$(readlink -f "$ROOT/code/Depth-Anything-3")" ]] || {
      echo "WorldOlympiad Depth-Anything-3 path is occupied by an unapproved item; preserved." >&2; exit 2;
    }
  else
    ln -s "$ROOT/code/Depth-Anything-3" "$WORLD/Depth-Anything-3"
  fi
  python3 "$ROOT/scripts/stage_runtime_sources.py" --verify-only
  "$ROOT/scripts/preflight.sh" runtimes
  echo "Existing Python environments reused; source-bound runtimes were staged and import/version preflight passed."
  exit 0
fi
if [[ $any_final == 1 ]]; then
  echo "Some final environments exist but the full set is incomplete. Preserving them; refusing a mixed install." >&2
  exit 2
fi
[[ -n $uv_bin && -x $uv_bin ]] || { echo "uv is required; no global install was attempted." >&2; exit 2; }
[[ -n $python310 && -x $python310 ]] || { echo "Python 3.10 is required for RoboTwin and LingBot-VA." >&2; exit 2; }
"$python310" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))' || {
  echo "PYTHON310 must point to Python 3.10" >&2; exit 2;
}
cuda_home=${CUDA_HOME:-}
nvcc_path=${NVCC:-}
if [[ -z $nvcc_path ]]; then
  if [[ -n $cuda_home && -x $cuda_home/bin/nvcc ]]; then
    nvcc_path=$cuda_home/bin/nvcc
  else
    nvcc_path=$(command -v nvcc || true)
  fi
fi
[[ -n $nvcc_path && -x $nvcc_path ]] || {
  echo "CUDA 12.8 nvcc is required; set CUDA_HOME or put nvcc on PATH." >&2; exit 2;
}
nvcc_path=$(readlink -f -- "$nvcc_path")
if [[ -z $cuda_home ]]; then cuda_home=${nvcc_path%/bin/nvcc}; fi
cuda_release=$("$nvcc_path" --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' | head -n 1)
[[ $cuda_release == 12.8 ]] || {
  echo "The locked cu128 runtime requires CUDA toolkit 12.8; found ${cuda_release:-unknown}." >&2; exit 2;
}
export CUDA_HOME=$cuda_home NVCC=$nvcc_path
if [[ $mode == --online-install ]]; then
  [[ -n $python311 && -x $python311 ]] || {
    echo "Python 3.11 is required for the online install; set PYTHON311 if it is not on PATH." >&2; exit 2;
  }
  "$python311" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))' || {
    echo "PYTHON311 must point to Python 3.11" >&2; exit 2;
  }
  echo "Online install enabled: package downloads will use the configured Python indexes."
else
  [[ -s $AUTORESEARCH_BOOTSTRAP_DIR/python311.tar.gz ]] || {
    echo "Pinned Python archive is missing: $AUTORESEARCH_BOOTSTRAP_DIR/python311.tar.gz" >&2
    echo "Use --online-install with Python 3.11, uv, and network access, or provide the verified archive." >&2
    exit 2
  }
  printf '%s  %s\n' b9fff092374acebc451bb881156afe6a58991213d25386358742fee296128909 "$AUTORESEARCH_BOOTSTRAP_DIR/python311.tar.gz" | sha256sum -c -
  [[ -d $AUTORESEARCH_BOOTSTRAP_DIR/wheels ]] || {
    echo "Pinned offline autoresearch wheel cache is missing; no runtime was changed." >&2
    echo "Use --online-install with Python 3.11, uv, and network access, or provide the wheel cache." >&2
    exit 2
  }
fi
# Refuse before staging partial environments when the later CUDA smoke has no eligible card.
python3 "$ROOT/scripts/select_setup_gpu.py" >/dev/null
export WORLDOLYMPIAD_WORKSPACE="$ROOT" WORLD_REPO="$WORLD" UV_BIN="$uv_bin"
export AUTORESEARCH_BOOTSTRAP_DIR AUTORESEARCH_ENV_DIR="$ROOT/envs/autoresearch"
export ROBOTWIN_REPO="$ROOT/runtime/RoboTwin" ROBOTWIN_ENV="$ROOT/envs/robotwin"
export ROBOTWIN_CUROBO_REPO="$ROOT/code/curobo-0.7.8"
export ROBOTWIN_BOOTSTRAP_PYTHON="$python310" ROBOTWIN_UV_PYTHON_DIR="$ROOT/runtime/python310"
export SCENECODE_SOURCE_ROOT="$ROOT/code/SceneCode" SCENECODE_RUNTIME_ROOT="$ROOT/runtime/scenecode"
export SCENECODE_ENV="$ROOT/envs/scenecode" BLENDER_PARENT="$ROOT/runtime"
export WORLDEVAL_ENV_DIR="$ROOT/envs/worldeval" WORLDEVAL_BASE_PYTHON="$ROOT/envs/scenecode/bin/python"
export UV_CACHE_DIR="$ROOT/cache/uv"
python3 "$ROOT/scripts/stage_runtime_sources.py"
bash "$WORLD/deployment/ziplab5090/stage_scenecode_runtime.sh"
if [[ -e $WORLD/Depth-Anything-3 || -L $WORLD/Depth-Anything-3 ]]; then
  [[ -L $WORLD/Depth-Anything-3 && $(readlink -f "$WORLD/Depth-Anything-3") == "$(readlink -f "$ROOT/code/Depth-Anything-3")" ]] || {
    echo "WorldOlympiad Depth-Anything-3 path is occupied by an unapproved item; preserved." >&2; exit 2;
  }
else
  ln -s "$ROOT/code/Depth-Anything-3" "$WORLD/Depth-Anything-3"
fi
setup_gpu() {
  local record index uuid free util
  record=$(python3 "$ROOT/scripts/select_setup_gpu.py")
  IFS=$'\t' read -r index uuid free util <<<"$record"
  export CUDA_VISIBLE_DEVICES=$index
  printf 'CUDA smoke test uses allowlisted GPU index %s (UUID %s, free %s MiB, util %s%%).\n' "$index" "$uuid" "$free" "$util"
}
if [[ $mode == --online-install ]]; then
  autoresearch_partial="$ROOT/envs/autoresearch.partial"
  "$uv_bin" venv --python "$python311" "$autoresearch_partial"
  "$uv_bin" pip install --python "$autoresearch_partial/bin/python" \
    -r "$WORLD/requirements-autoresearch.txt"
  "$uv_bin" pip install --python "$autoresearch_partial/bin/python" \
    -r "$ROOT/requirements-agent-runtime.txt"
  "$autoresearch_partial/bin/python" -c \
    'import cv2, numpy, pybullet, pydantic, scipy, trimesh; print("autoresearch dependencies OK")'
  mv "$autoresearch_partial" "$ROOT/envs/autoresearch"
else
  AUTORESEARCH_BOOTSTRAP_DIR="$AUTORESEARCH_BOOTSTRAP_DIR" \
    bash "$WORLD/deployment/ziplab5090/bootstrap_autoresearch.sh"
  "$ROOT/envs/autoresearch/bin/python" -m pip install \
    --no-index --find-links "$AUTORESEARCH_BOOTSTRAP_DIR/wheels" \
    -r "$ROOT/requirements-agent-runtime.txt"
fi
"$ROOT/envs/autoresearch/bin/python" -c \
  'import agents, openai; print("Agent API SDKs OK", openai.__version__)'
SCENECODE_PYBULLET_BRIDGE_ROOT="$ROOT/runtime/scenecode_pybullet_minimal" \
AUTORESEARCH_PYTHON="$ROOT/envs/autoresearch/bin/python" \
  bash "$WORLD/deployment/ziplab5090/stage_scenecode_pybullet_bridge.sh"
SCENECODE_ROOT="$SCENECODE_RUNTIME_ROOT" SCENECODE_ENV="$SCENECODE_ENV" \
BLENDER_PARENT="$BLENDER_PARENT" bash "$WORLD/deployment/ziplab5090/bootstrap_scenecode.sh"
setup_gpu
WORLD_REPO="$WORLD" WORLDEVAL_ENV_DIR="$WORLDEVAL_ENV_DIR" \
WORLDEVAL_BASE_PYTHON="$ROOT/envs/scenecode/bin/python" UV_BIN="$uv_bin" \
UV_CACHE_DIR="$UV_CACHE_DIR" WORLDEVAL_PYTHON_VERSION=3.11 \
WORLDEVAL_CLIP_SOURCE="git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40" \
  bash "$WORLD/deployment/ziplab5090/bootstrap_worldolympiad_evaluation.sh"
setup_gpu
ROBOTWIN_BOOTSTRAP_PYTHON="$python310" ROBOTWIN_REPO="$ROBOTWIN_REPO" \
ROBOTWIN_ENV="$ROBOTWIN_ENV" ROBOTWIN_CUROBO_REPO="$ROBOTWIN_CUROBO_REPO" \
ROBOTWIN_UV_PYTHON_DIR="$ROBOTWIN_UV_PYTHON_DIR" UV_BIN="$uv_bin" \
  bash "$WORLD/deployment/ziplab5090/bootstrap_robotwin_runtime.sh"
setup_gpu
"$uv_bin" venv --python "$python310" "$ROOT/envs/lingbot"
"$uv_bin" pip install --python "$ROOT/envs/lingbot/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
"$uv_bin" pip install --python "$ROOT/envs/lingbot/bin/python" \
  diffusers==0.36.0 transformers==4.55.2 accelerate einops easydict \
  numpy==1.26.4 tqdm 'imageio[ffmpeg]' websockets msgpack \
  opencv-python-headless matplotlib ftfy safetensors Pillow
PYTHONPATH="$ROOT/runtime/lingbot-va/wan_va" "$ROOT/envs/lingbot/bin/python" - <<'PY'
import torch
from configs import VA_CONFIGS
from modules.model import WanAttention
assert torch.cuda.is_available()
assert "demo_i2av" in VA_CONFIGS
assert WanAttention(dim=128, heads=2, attn_mode="torch").attn_op is not None
major, minor = torch.cuda.get_device_capability()
assert f"sm_{major}{minor}" in torch.cuda.get_arch_list()
print("LingBot-VA runtime smoke passed", torch.__version__)
PY
python3 "$ROOT/scripts/stage_runtime_sources.py" --verify-only
"$ROOT/scripts/preflight.sh" runtimes
