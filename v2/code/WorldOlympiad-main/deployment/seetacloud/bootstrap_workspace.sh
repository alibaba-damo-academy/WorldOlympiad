#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
workspace=${WORLDOLYMPIAD_WORKSPACE:-$(cd -- "$script_dir/../../../.." && pwd -P)}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
evaluator_models=${WORLDEVAL_MODEL_DIR:-$workspace/models/worldolympiad-evaluators}

required_dirs=(code models envs outputs runtime logs docs)
for directory in "${required_dirs[@]}"; do
  mkdir -p "$workspace/$directory"
done

if [[ ! -e /dev/nvidiactl ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU devices are not attached to this container." >&2
  echo "Attach the NVIDIA GPU, then rerun this bootstrap." >&2
  exit 2
fi

nvidia-smi >/dev/null

[[ -d $world_repo ]] || {
  echo "WorldOlympiad code is missing: $world_repo" >&2
  exit 2
}
[[ -d $evaluator_models ]] || {
  echo "Evaluator models are missing: $evaluator_models" >&2
  exit 2
}

weights_link=$world_repo/weights
if [[ -L $weights_link ]]; then
  [[ $(readlink -f "$weights_link") == $(readlink -f "$evaluator_models") ]] || {
    echo "Evaluator weight link points to the wrong directory: $weights_link" >&2
    exit 2
  }
elif [[ -d $weights_link ]]; then
  # The deployed workspace uses per-model links and also retains CLIP assets.
  # Validate the existing layout without replacing or copying its checkpoints.
  for model_name in QwenVL da3 sam3; do
    [[ -e "$weights_link/$model_name" ]] || {
      echo "Existing weights directory lacks $model_name: $weights_link" >&2
      exit 2
    }
  done
elif [[ -e $weights_link ]]; then
  echo "Weights path is not a directory: $weights_link" >&2
  exit 2
else
  ln -s "$evaluator_models" "$weights_link"
fi

cat <<EOF
WorldOlympiad2 workspace is ready:
  workspace: $workspace
  code:      $world_repo
  models:    $workspace/models
EOF
