#!/bin/bash

# See https://github.com/Tencent-Hunyuan/Hunyuan3D-2?tab=readme-ov-file#install-requirements

set -euo pipefail

cd external/Hunyuan3D-2

# Install requirements.
uv pip install -e .
cd hy3dgen/texgen/custom_rasterizer
uv run python setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
uv run python setup.py install
