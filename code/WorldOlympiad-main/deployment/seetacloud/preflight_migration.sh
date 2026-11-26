#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
workspace=${WORLDOLYMPIAD_WORKSPACE:-$(cd -- "$script_dir/../../../.." && pwd -P)}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
# Existing server runtimes are authoritative. This check never installs packages.
python3 "$world_repo/scripts/runtime.py" --workspace "$workspace" --check
required=(
 "$workspace/code/RoboTwin" "$workspace/code/RoboLab-isaac51"
 "$workspace/code/SceneCode-main" "$workspace/code/lingbot-va" "$workspace/code/cosmos-framework"
 "$workspace/code/Dream2Flow-main-cae-min64-vis08-v1" "$workspace/code/Dream2Flow-main-official-20260809"
 "$workspace/models/Cosmos3-Nano-Policy-DROID/config.json"
 "$workspace/models/lingbot-va-posttrain-robotwin/transformer/config.json"
 "$workspace/models/Wan2.2-TI2V-5B-vae/Wan2.2_VAE.pth"
 "$world_repo/weights/QwenVL/config.json" "$world_repo/weights/da3/config.json" "$world_repo/weights/sam3/sam3.pt"
)
for path in "${required[@]}"; do
 [[ -e "$path" ]] || { echo "Configured runtime asset is missing: $path" >&2; exit 2; }
done
[[ -e /dev/nvidiactl ]] || { echo "GPU device is not attached" >&2; exit 2; }
echo "Existing WorldOlympiad server runtime preflight passed"
