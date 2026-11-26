#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
world_repo=${WORLD_REPO:-$script_dir}
workspace=${WORLDOLYMPIAD_WORKSPACE:-$(cd -- "$world_repo/../.." && pwd -P)}
output_root=${WORLDOLYMPIAD_OUTPUT_ROOT:-$workspace/outputs}
scenecode_source=${SCENECODE_SOURCE_ROOT:-$workspace/code/SceneCode}
scenecode_runtime=${SCENECODE_RUNTIME_ROOT:-$workspace/runtime/scenecode}
scenecode_python=${SCENECODE_PYTHON:-$workspace/envs/scenecode/bin/python}
secret_file=${SCENECODE_ENV_FILE:-$workspace/.secrets/scenecode.env}
# Export the selected workspace and repository: the staged runtime script and
# the bridge resolve their own defaults from these, and a direct run must
# behave exactly like the managed tool path.
export WORLDOLYMPIAD_WORKSPACE=$workspace
export WORLD_REPO=$world_repo
# Keep each SceneCode/Blender child from allocating host-wide BLAS thread pools.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

batch_id=${1:?batch id is required}
case_id=${2:?case id is required}
# Regeneration is an explicit revision move: it archives the previous
# generation revision before SceneCode writes a new one.
supersede_existing=${WORLDOLYMPIAD_SCENECODE_SUPERSEDE:-0}

[[ $batch_id =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$ ]] || {
  echo "Unsafe batch id: $batch_id" >&2; exit 2;
}
[[ $case_id =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$ ]] || {
  echo "Unsafe case id: $case_id" >&2; exit 2;
}

# Check the selected embodiment before expensive asset authoring. Restoration
# only uses the existing pinned local sources and retains every original hash.
PYTHONPATH=$world_repo${PYTHONPATH:+:$PYTHONPATH} \
  "$workspace/envs/autoresearch/bin/python" -m autoresearch.robotics_robot_asset \
  --workspace "$workspace" --output-root "$output_root" --restore \
  --batch-id "$batch_id" --case-id "$case_id"

# The coordinator client may disconnect while a managed job is still running. Keep
# every SceneCode/Blender descendant independent of that session's stdout pipe;
# otherwise a later Blender server can abort with EPIPE despite the parent job
# surviving.  Constrain the caller-selected log to the harness output tree.
durable_job_log=${WORLDOLYMPIAD_DURABLE_JOB_LOG:-}
if [[ -n $durable_job_log ]]; then
  [[ $durable_job_log == "$output_root"/robotics_scenecode_job_logs/* ]] || {
    echo "Unsafe SceneCode durable job log: $durable_job_log" >&2; exit 2;
  }
  mkdir -p "$(dirname "$durable_job_log")"
  : > "$durable_job_log"
  exec >>"$durable_job_log" 2>&1
fi

[[ -f $secret_file ]] || { echo "SceneCode API environment is missing: $secret_file" >&2; exit 2; }
[[ $(stat -c '%a' "$secret_file") == 600 ]] || {
  echo "SceneCode API environment must have mode 600: $secret_file" >&2; exit 2;
}

scenecode_preflight() {
  [[ -x $scenecode_python ]] && "$scenecode_python" -c \
    'import agents, bpy, coacd, drake, hydra, mathutils, torch' >/dev/null 2>&1
}
if ! scenecode_preflight; then
  SCENECODE_ENV=$(dirname "$(dirname "$scenecode_python")") \
    bash "$world_repo/deployment/ziplab5090/bootstrap_scenecode.sh"
fi
scenecode_preflight || { echo "SceneCode dependency preflight failed" >&2; exit 2; }

SCENECODE_RUNTIME_ROOT=$scenecode_runtime \
SCENECODE_SOURCE_ROOT=$scenecode_source \
SCENECODE_PYTHON=$scenecode_python \
  bash "$world_repo/deployment/ziplab5090/stage_scenecode_runtime.sh" >/dev/null

# A run-scoped author profile owns these public selectors; credentials still
# come from the selected API environment. Unprofiled launches retain its defaults.
source_profile_options=()
if [[ -n ${SCENECODE_CODE_OBJECT_CONFIG_ROOT:-} ]]; then
  for option in SCENECODE_OPENAI_MODEL SCENECODE_CODE_OBJECT_CONFIG_ROOT \
    SCENECODE_CODE_OBJECT_API SCENECODE_CODE_OBJECT_STREAM \
    SCENECODE_CODE_OBJECT_RESPONSES_STREAM SCENECODE_CODE_OBJECT_READ_TIMEOUT; do
    [[ ${!option+x} ]] && source_profile_options+=("$option=${!option}")
  done
fi
set -a
# shellcheck disable=SC1090
source "$secret_file"
set +a
if (( ${#source_profile_options[@]} )); then
  export "${source_profile_options[@]}"
  unset SCENECODE_CODE_OBJECT_REASONING_EFFORT
fi
[[ -n ${OPENAI_API_KEY:-} ]] || { echo "OPENAI_API_KEY is not configured" >&2; exit 2; }

export OPENAI_AGENTS_DISABLE_TRACING=1
export SCENECODE_CODE_OBJECT_TEXT_ONLY=0
export SCENECODE_REQUIRE_REFERENCE_IMAGE=1
export SCENECODE_CODE_OBJECT_REQUIRE_NATIVE_IMAGE=1
export SCENECODE_CODE_OBJECT_API=responses
# Task-focused robotics scenes omit unrequested architectural decoration.
# Explicit 0 preserves wall/ceiling fixtures when required by the task.
export SCENECODE_ROBOTICS_FAST=${SCENECODE_ROBOTICS_FAST:-1}
export SCENECODE_BARE_WALLS=${SCENECODE_BARE_WALLS:-1}
export SCENECODE_BARE_CEILING=${SCENECODE_BARE_CEILING:-1}
export SCENECODE_PROTECT_PILOT_MANIPULANDS=${SCENECODE_PROTECT_PILOT_MANIPULANDS:-1}
export PATH=$workspace/runtime/blender-4.4.3-linux-x64:$PATH
export PYTHONPATH=$world_repo${PYTHONPATH:+:$PYTHONPATH}

generator_args=(
  --output-root "$output_root"
  --batch-id "$batch_id"
  --case-id "$case_id"
  --scenecode-root "$scenecode_runtime"
  --scenecode-python "$scenecode_python"
)
if [[ $supersede_existing == 1 ]]; then
  generator_args+=(--supersede-existing)
fi

exec "$scenecode_python" -m autoresearch.robotics_scenecode_case_generator \
  "${generator_args[@]}"
