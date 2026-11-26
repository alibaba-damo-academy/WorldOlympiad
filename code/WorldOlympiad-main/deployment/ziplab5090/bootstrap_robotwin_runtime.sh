#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
workspace=${WORLDOLYMPIAD_WORKSPACE:-$(cd -- "$script_dir/../../../.." && pwd -P)}
robotwin_repo=${ROBOTWIN_REPO:-$workspace/code/RoboTwin}
uv=${UV_BIN:-$workspace/envs/bootstrap/bin/uv}
environment=${ROBOTWIN_ENV:-$workspace/envs/robotwin}
partial=${environment}.partial
python_install_dir=${ROBOTWIN_UV_PYTHON_DIR:-$workspace/runtimes/uv-python}
bootstrap_python=${ROBOTWIN_BOOTSTRAP_PYTHON:-$workspace/envs/dream2flow/bin/python}
curobo_repo=${ROBOTWIN_CUROBO_REPO:-$workspace/code/curobo-0.7.8}
torch_index=${ROBOTWIN_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}

[[ -x $uv ]] || { echo "uv is missing: $uv" >&2; exit 2; }
[[ -f $robotwin_repo/scripts/requirements.txt ]] || {
  echo "RoboTwin source is missing: $robotwin_repo" >&2
  exit 2
}
[[ -f $curobo_repo/setup.py ]] || {
  echo "cuRobo v0.7.8 source is missing: $curobo_repo" >&2
  exit 2
}
[[ ! -e $environment ]] || {
  echo "RoboTwin environment already exists: $environment" >&2
  exit 0
}
[[ ! -e $partial ]] || {
  echo "partial RoboTwin environment already exists: $partial" >&2
  exit 2
}

export UV_PYTHON_INSTALL_DIR=$python_install_dir
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
mkdir -p "$python_install_dir"

[[ -x $bootstrap_python ]] || {
  echo "Python 3.10 bootstrap interpreter is missing: $bootstrap_python" >&2
  exit 2
}
python_version=$($bootstrap_python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
[[ $python_version == 3.10 ]] || {
  echo "RoboTwin requires Python 3.10, got $python_version from $bootstrap_python" >&2
  exit 2
}

"$uv" venv --python "$bootstrap_python" "$partial"
"$uv" pip install --python "$partial/bin/python" \
  -r "$robotwin_repo/scripts/requirements.txt"
"$uv" pip install --python "$partial/bin/python" --index "$torch_index" \
  torch==2.9.0 torchvision==0.24.0
"$uv" pip install --python "$partial/bin/python" \
  setuptools==69.5.1 warp-lang==1.12.0 wheel ninja
cuda_home=${CUDA_HOME:-}
nvcc_path=${NVCC:-}
if [[ -z $nvcc_path ]]; then
  if [[ -n $cuda_home && -x $cuda_home/bin/nvcc ]]; then
    nvcc_path=$cuda_home/bin/nvcc
  else
    nvcc_path=$(command -v nvcc || true)
  fi
fi
[[ -n $nvcc_path && -x $nvcc_path ]] || { echo "CUDA nvcc is required; set CUDA_HOME or PATH." >&2; exit 2; }
nvcc_path=$(readlink -f -- "$nvcc_path")
if [[ -z $cuda_home ]]; then cuda_home=${nvcc_path%/bin/nvcc}; fi
export CUDA_HOME=$cuda_home NVCC=$nvcc_path
MAX_JOBS=${MAX_JOBS:-8} "$uv" pip install --python "$partial/bin/python" \
  --no-build-isolation --editable "$curobo_repo"

"$partial/bin/python" - "$partial" <<'PY'
from __future__ import annotations

import pathlib
import sys

environment = pathlib.Path(sys.argv[1])
site_packages = next((environment / "lib").glob("python*/site-packages"))

urdf_loader = site_packages / "sapien" / "wrapper" / "urdf_loader.py"
text = urdf_loader.read_text(encoding="utf-8")
text = text.replace('with open(urdf_file, "r") as f:', 'with open(urdf_file, "r", encoding="utf-8") as f:')
text = text.replace('srdf_file = urdf_file[:-4] + "srdf"', 'srdf_file = urdf_file[:-4] + ".srdf"')
text = text.replace('with open(srdf_file, "r") as f:', 'with open(srdf_file, "r", encoding="utf-8") as f:')
urdf_loader.write_text(text, encoding="utf-8")

planner = site_packages / "mplib" / "planner.py"
text = planner.read_text(encoding="utf-8")
old = "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:"
new = "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"
if old in text:
    planner.write_text(text.replace(old, new), encoding="utf-8")
PY

PYTHONPATH="$robotwin_repo" "$partial/bin/python" - <<'PY'
import torch

import cv2
import curobo
from curobo.curobolib import (
    geom_cu,
    kinematics_fused_cu,
    lbfgs_step_cu,
    line_search_cu,
    tensor_step_cu,
)
import gymnasium
import h5py
import mplib
import numpy
import sapien.core as sapien
import toppra
import trimesh
import warp

major, minor = torch.cuda.get_device_capability(0)
architecture = f"sm_{major}{minor}"
if architecture not in torch.cuda.get_arch_list():
    raise RuntimeError(
        f"PyTorch {torch.__version__} does not contain kernels for {architecture}: "
        f"{torch.cuda.get_arch_list()}"
    )
cuda_probe = torch.arange(4, device="cuda", dtype=torch.float32).square().sum().item()
if cuda_probe != 14.0:
    raise RuntimeError(f"CUDA execution probe failed: {cuda_probe}")

engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene(sapien.SceneConfig())
scene.set_timestep(1 / 250)
print(
    "RoboTwin runtime passed",
    {
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "cuda_probe": cuda_probe,
        "curobo_extensions": 5,
    },
)
PY

mv "$partial" "$environment"
echo "$environment"
