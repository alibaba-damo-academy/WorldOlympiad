#!/usr/bin/env bash
#
# Download HSSD preprocessed data required for asset retrieval.
# This script downloads:
# 1. CLIP indices and embeddings for semantic search (~60MB)
# 2. Pre-validated support surfaces from HSM (~2GB)
#
# Based on HSM's setup.sh but simplified for our needs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

PREPROCESSED_DIR="$DATA_DIR/preprocessed"
HSSD_MODELS_DIR="$DATA_DIR/hssd-models"
SUPPORT_SURFACES_DIR="$HSSD_MODELS_DIR/support-surfaces"

echo "=========================================="
echo "HSSD Preprocessed Data Download"
echo "=========================================="
echo

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed. Please install it first."
        exit 1
    fi
}

check_command wget
check_command unzip

mkdir -p "$DATA_DIR"

echo "Downloading preprocessed data (~60MB)..."
echo

PREPROCESSED_URL="https://github.com/3dlg-hcvc/hsm/releases/latest/download/data.zip"
PREPROCESSED_ZIP="$DATA_DIR/preprocessed_data.zip"

if [ -f "$PREPROCESSED_ZIP" ]; then
    echo "Preprocessed data archive already exists, skipping download."
else
    wget --no-verbose --show-progress "$PREPROCESSED_URL" -O "$PREPROCESSED_ZIP"
fi

echo
echo "Extracting preprocessed data..."

if [ -d "$PREPROCESSED_DIR" ]; then
    echo "Warning: $PREPROCESSED_DIR already exists."
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping extraction."
    else
        rm -rf "$PREPROCESSED_DIR"
        # Zip contains data/ folder, so extract to project root.
        unzip -q "$PREPROCESSED_ZIP" -d "$PROJECT_ROOT"
    fi
else
    # Zip contains data/ folder, so extract to project root.
    unzip -q "$PREPROCESSED_ZIP" -d "$PROJECT_ROOT"
fi

echo
echo "Cleaning up archive..."
rm "$PREPROCESSED_ZIP"

echo
echo "Downloading pre-validated support surfaces (~2GB)..."
echo

SUPPORT_SURFACES_URL="https://github.com/3dlg-hcvc/hsm/releases/latest/download/support-surfaces.zip"
SUPPORT_SURFACES_ZIP="$DATA_DIR/support_surfaces.zip"

if [ -f "$SUPPORT_SURFACES_ZIP" ]; then
    echo "Support surfaces archive already exists, skipping download."
else
    wget --no-verbose --show-progress "$SUPPORT_SURFACES_URL" -O "$SUPPORT_SURFACES_ZIP"
fi

echo
echo "Extracting support surfaces..."

mkdir -p "$HSSD_MODELS_DIR"

if [ -d "$SUPPORT_SURFACES_DIR" ]; then
    echo "Warning: $SUPPORT_SURFACES_DIR already exists."
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping extraction."
    else
        rm -rf "$SUPPORT_SURFACES_DIR"
        unzip -q "$SUPPORT_SURFACES_ZIP" -d "$HSSD_MODELS_DIR"
    fi
else
    unzip -q "$SUPPORT_SURFACES_ZIP" -d "$HSSD_MODELS_DIR"
fi

echo
echo "Cleaning up archive..."
rm "$SUPPORT_SURFACES_ZIP"

echo
echo "=========================================="
echo "Data downloaded successfully!"
echo
echo "Preprocessed data: $PREPROCESSED_DIR"
echo "Support surfaces: $SUPPORT_SURFACES_DIR"
echo
echo "Next steps:"
echo "1. Download HSSD models (~72GB):"
echo "   cd $DATA_DIR"
echo "   git lfs install"
echo "   git clone git@hf.co:datasets/hssd/hssd-models"
echo
echo "2. Enable HSSD in your config:"
echo "   asset_manager:"
echo "     strategy: \"hssd\""
echo "=========================================="
