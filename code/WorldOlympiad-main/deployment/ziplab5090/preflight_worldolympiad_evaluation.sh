#!/usr/bin/env bash
set -euo pipefail

workspace=${WORLDOLYMPIAD_WORKSPACE:-$(pwd)}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
evaluation_python=${WORLDEVAL_PYTHON:-$workspace/envs/worldolympiad/bin/python}
export PATH="$(dirname "$evaluation_python"):$PATH"

required=(
  "$evaluation_python"
  "$world_repo/Depth-Anything-3/pyproject.toml"
  "$world_repo/model/qwenvl_server.py"
  "$world_repo/physical/serve_sam3.py"
  "$world_repo/3d_metrics/serve_reward_3d.py"
  "$world_repo/weights/QwenVL/config.json"
  "$world_repo/weights/QwenVL/model.safetensors.index.json"
  "$world_repo/weights/da3/config.json"
  "$world_repo/weights/da3/model.safetensors"
  "$world_repo/weights/sam3/sam3.pt"
  "$world_repo/weights/clip/ViT-B-32.pt"
)
missing=0
for path in "${required[@]}"; do
  if [[ ! -e $path ]]; then
    echo "missing: $path" >&2
    missing=1
  fi
done
[[ $missing == 0 ]] || exit 2

"$evaluation_python" - "$world_repo" <<'PY'
import json
import hashlib
import importlib.metadata
import shutil
import sys
from pathlib import Path

import clip
import cv2
from e3nn.o3 import matrix_to_angles, wigner_D
from gsplat import csrc, rasterization
import numpy
import open3d
from safetensors import safe_open
import timm
import torch
import transformers
import ultralytics
from depth_anything_3.api import DepthAnything3
from transformers import Qwen3VLForConditionalGeneration

world_repo = Path(sys.argv[1]).resolve()
weights = world_repo / "weights"

# Validate the complete local checkpoint inventory before any persistent
# service reserves GPU memory.  AutoConfig is intentionally not used for DA3:
# its repository config is consumed by DepthAnything3 rather than Transformers.
qwen_index = json.loads((weights / "QwenVL/model.safetensors.index.json").read_text())
qwen_shards = sorted(set(qwen_index.get("weight_map", {}).values()))
assert qwen_shards, "QwenVL checkpoint index contains no shards"
for shard in qwen_shards:
    path = weights / "QwenVL" / shard
    assert path.is_file() and path.stat().st_size > 0, f"missing QwenVL shard: {path}"
    with safe_open(path, framework="pt", device="cpu") as handle:
        assert handle.keys(), f"empty QwenVL shard: {path}"

for relative in ("sam3/sam3.pt", "da3/model.safetensors"):
    path = weights / relative
    assert path.is_file() and path.stat().st_size > 0, f"empty checkpoint: {path}"
with safe_open(weights / "da3/model.safetensors", framework="pt", device="cpu") as handle:
    assert handle.keys(), "DA3 checkpoint contains no tensors"

clip_path = weights / "clip/ViT-B-32.pt"
clip_expected_sha256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
with clip_path.open("rb") as stream:
    clip_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
assert clip_sha256 == clip_expected_sha256, (
    f"CLIP ViT-B/32 checksum mismatch: {clip_sha256}"
)

assert torch.cuda.is_available(), "CUDA is unavailable"
assert callable(clip.simple_tokenizer.SimpleTokenizer()), (
    "SAM3 requires the callable tokenizer from ultralytics/CLIP"
)
assert shutil.which("ninja"), "gsplat CUDA extension requires ninja on PATH"
gsplat_direct_url = importlib.metadata.distribution("gsplat").read_text("direct_url.json") or ""
assert "gsplat-0b4dddf04cb687367602c01196913cde6a743d70" in gsplat_direct_url, (
    "DA3 requires gsplat built from its registered upstream commit"
)
angles = matrix_to_angles(torch.eye(3).unsqueeze(0))
assert len(angles) == 3 and wigner_D(1, *angles).shape[-2:] == (3, 3), (
    "DA3 spherical-harmonics rotation dependencies are unusable"
)
major, minor = torch.cuda.get_device_capability()
device_arch = f"sm_{major}{minor}"
assert device_arch in torch.cuda.get_arch_list(), (
    f"PyTorch {torch.__version__} does not contain kernels for {device_arch}: "
    f"{torch.cuda.get_arch_list()}"
)
assert (torch.ones(1, device="cuda") + 1).item() == 2

# Exercise the compiled CUDA rasterizer, not merely its Python wrapper.
device = "cuda"
means = torch.tensor([[0.0, 0.0, 2.0]], device=device)
quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
scales = torch.full((1, 3), 0.1, device=device)
opacities = torch.ones(1, device=device)
colors = torch.tensor([[1.0, 0.0, 0.0]], device=device)
viewmats = torch.eye(4, device=device).unsqueeze(0)
intrinsics = torch.tensor(
    [[[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]],
    device=device,
)
render, alpha, _ = rasterization(
    means, quats, scales, opacities, colors, viewmats, intrinsics, 64, 64
)
torch.cuda.synchronize()
assert render.shape == (1, 64, 64, 3) and alpha.max().item() > 0, (
    "gsplat CUDA rasterization preflight failed"
)
print("WorldOlympiad evaluator preflight passed")
PY
