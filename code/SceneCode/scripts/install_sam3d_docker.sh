#!/bin/bash

# Non-interactive SAM3D installation for Docker builds.
# Differences from install_sam3d.sh:
# - No interactive prompts (auto-accept everything).
# - Skips CUDA detection/installation (already in base image).
# - Skips HuggingFace checkpoint download (mounted at runtime).
# - Keeps: repo cloning, dependency installation, CUDA package builds.

set -euo pipefail

echo "========================================="
echo "SAM3D Docker Installation"
echo "========================================="
echo ""

# CUDA is pre-installed in the Docker image.
echo "Using CUDA_HOME: ${CUDA_HOME}"
nvcc --version

echo ""
echo "Step 1: Cloning repositories..."

mkdir -p external
cd external

# Clone SAM 3D Objects repository.
if [ ! -d "sam-3d-objects" ]; then
    git clone https://github.com/facebookresearch/sam-3d-objects.git
    echo "Cloned sam-3d-objects"
else
    echo "sam-3d-objects already exists"
fi

# Clone SAM3 repository.
if [ ! -d "SAM3" ]; then
    git clone https://github.com/facebookresearch/sam3.git SAM3
    echo "Cloned SAM3"
else
    echo "SAM3 already exists"
fi

echo ""
echo "Step 2: Installing SAM3..."
cd SAM3
uv pip install -e ".[notebooks]"
cd ..
echo "SAM3 installed"

echo ""
echo "Step 3: Installing SAM 3D Objects dependencies..."

cd sam-3d-objects

# Install non-CUDA dependencies from requirements.txt.
echo "Installing sam-3d-objects core dependencies..."
grep -v -E "^(torch|torchvision|torchaudio|cuda-python|nvidia-|MoGe|flash_attn|bpy|wandb|jupyter|tensorboard|Flask|webdataset|sagemaker)" requirements.txt > /tmp/filtered_requirements.txt
uv pip install -r /tmp/filtered_requirements.txt

# Install CUDA-dependent packages with --no-build-isolation.
echo ""
echo "Installing gsplat..."
uv pip install --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

echo ""
echo "Installing nvdiffrast..."
uv pip install --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git"

echo ""
echo "Pre-compiling nvdiffrast CUDA extensions..."
python3 << 'PYEOF'
import sys
import os

try:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available - pre-compilation will happen on first use")
        sys.exit(0)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print("Compiling nvdiffrast CUDA kernels...")

    import nvdiffrast.torch as dr
    ctx = dr.RasterizeCudaContext()

    import torch.utils.cpp_extension as cpp_ext
    build_dir = cpp_ext._get_build_directory("nvdiffrast_plugin", False)
    so_path = os.path.join(build_dir, "nvdiffrast_plugin.so")

    if os.path.exists(so_path):
        size_mb = os.path.getsize(so_path) / (1024 * 1024)
        print(f"SUCCESS: {so_path} ({size_mb:.1f} MB)")
    else:
        print("WARNING: .so file not found, compilation may have failed")
        sys.exit(1)

except Exception as e:
    print(f"Pre-compilation failed: {e}")
    print("NOTE: nvdiffrast will compile on first SAM3D use")
    sys.exit(0)  # Non-fatal.
PYEOF

echo ""
echo "Installing kaolin 0.17.0..."
uv pip install --no-build-isolation \
    "git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0"

echo ""
echo "Installing pytorch3d from source..."
uv pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git"

echo ""
echo "Installing inference dependencies..."
uv pip install seaborn==0.13.2 gradio==5.49.0 imageio utils3d

echo ""
echo "Installing MoGe depth model..."
uv pip install "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"

cd ..

echo ""
echo "========================================="
echo "SAM3D Docker Installation Complete!"
echo "========================================="
echo ""
echo "Checkpoints must be mounted at runtime:"
echo "  -v ./external/checkpoints:/app/external/checkpoints"
echo ""
