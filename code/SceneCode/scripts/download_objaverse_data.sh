#!/usr/bin/env bash
#
# Download ObjectThor (Objaverse) data required for asset retrieval.
# This script downloads:
# 1. ObjectThor assets (~50GB compressed, ~100GB extracted)
# 2. ObjectThor annotations (~60MB)
# 3. Pre-computed CLIP features (~200MB)
#
# Data is cached in ~/.objathor-assets (shared across repos).
# A symlink is created at data/objathor-assets for config compatibility.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

# ObjectThor version to download.
OBJATHOR_VERSION="2023_09_23"

# Default cache location (shared across repos).
CACHE_DIR="$HOME/.objathor-assets/$OBJATHOR_VERSION"
OBJAVERSE_DIR="$DATA_DIR/objathor-assets"

echo "=========================================="
echo "ObjectThor (Objaverse) Data Download"
echo "=========================================="
echo

# Check for required Python environment.
if ! python -c "import objathor" 2>/dev/null; then
    echo "Error: objathor package not found."
    echo "Run 'uv sync' to install dependencies."
    exit 1
fi

echo "ObjectThor version: $OBJATHOR_VERSION"
echo "Cache location: $CACHE_DIR"
echo

# Download using objathor's load_* functions (uses default cache path).
# These functions skip download if data already exists.
python -c "
from objathor.dataset import (
    DatasetSaveConfig,
    load_assets_path,
    load_annotations_path,
    load_features_dir,
)

dsc = DatasetSaveConfig(VERSION='$OBJATHOR_VERSION')

print('Downloading/verifying assets...')
assets_path = load_assets_path(dsc)
print(f'  Assets: {assets_path}')

print('Downloading/verifying annotations...')
annotations_path = load_annotations_path(dsc)
print(f'  Annotations: {annotations_path}')

print('Downloading/verifying CLIP features...')
features_dir = load_features_dir(dsc)
print(f'  Features: {features_dir}')

print()
print('All data ready!')
"

# Create symlink for config compatibility (data/objathor-assets -> ~/.objathor-assets/VERSION).
mkdir -p "$DATA_DIR"
if [ -L "$OBJAVERSE_DIR" ]; then
    rm "$OBJAVERSE_DIR"
elif [ -d "$OBJAVERSE_DIR" ]; then
    echo "Warning: $OBJAVERSE_DIR exists as directory, not creating symlink"
fi

if [ ! -e "$OBJAVERSE_DIR" ] && [ -d "$CACHE_DIR" ]; then
    ln -s "$CACHE_DIR" "$OBJAVERSE_DIR"
    echo "Created symlink: $OBJAVERSE_DIR -> $CACHE_DIR"
fi

echo
echo "=========================================="
echo "Setup complete!"
echo
echo "Data location: $OBJAVERSE_DIR"
echo
echo "Next steps:"
echo "1. Run preprocessing script:"
echo "   python scripts/prepare_objaverse.py"
echo
echo "2. Enable Objaverse in your config:"
echo "   asset_manager:"
echo "     general_asset_source: \"objaverse\""
echo "=========================================="
