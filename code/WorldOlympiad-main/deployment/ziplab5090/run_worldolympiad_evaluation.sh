#!/usr/bin/env bash
set -euo pipefail

workspace=${WORLDOLYMPIAD_WORKSPACE:-$(pwd)}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
evaluation_python=${WORLDEVAL_PYTHON:-$workspace/envs/worldolympiad/bin/python}
case_root=${1:?usage: run_worldolympiad_evaluation.sh CASE_ROOT PIPELINE}
pipeline=${2:?usage: run_worldolympiad_evaluation.sh CASE_ROOT PIPELINE}

case_root=$(realpath "$case_root")
allowed_root=$(realpath "$workspace/outputs/evaluation_cases")
case "$case_root/" in
  "$allowed_root"/*) ;;
  *) echo "Evaluation cases must be under $allowed_root" >&2; exit 2 ;;
esac
case "$pipeline" in
  cosmos-predict|lingbot-world) ;;
  *) echo "Unsupported real World Model pipeline: $pipeline" >&2; exit 2 ;;
esac
if [[ ! -x "$evaluation_python" ]]; then
  echo "WorldOlympiad evaluation environment is missing: $evaluation_python" >&2
  exit 2
fi

services_pid=
services_log_root=$case_root/_evaluator_services
qwen_url=${QWENVL_SERVER_URLS:-http://127.0.0.1:${WORLDEVAL_QWEN_PORT:-8108}}
sam_url=${SAM3_SERVER_URLS:-http://127.0.0.1:${WORLDEVAL_SAM_PORT:-8110}}
da3_url=${REWARD_3D_SERVER_URLS:-http://127.0.0.1:${WORLDEVAL_DA3_PORT:-8112}}
cleanup_services() {
  if [[ -n ${services_pid:-} ]] && kill -0 "$services_pid" 2>/dev/null; then
    kill -TERM "$services_pid" 2>/dev/null || true
    wait "$services_pid" 2>/dev/null || true
  fi
}
trap cleanup_services EXIT INT TERM

# Continuous Action is applicable to these dynamic Pilot cases. Prepare and
# extract both video roles symmetrically before the unified evaluator builds
# its manifest. Any failed extraction fails the evaluation rather than being
# silently downgraded to N/A.
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
if [[ $gpu_count -eq 1 ]]; then
  default_d2f_gpu_ids=0
  default_worker_gpu_slots=0
else
  default_d2f_gpu_ids=1
  default_worker_gpu_slots=2
fi
export D2F_GPU_IDS=${D2F_GPU_IDS:-$default_d2f_gpu_ids}
case ${WORLDEVAL_CAE_PROFILE:-robot_ee} in
  robot_ee) bash "$world_repo/deployment/ziplab5090/run_worldolympiad_cae.sh" "$case_root" ;;
  physics_object) ;; # Independently scored from the two video object-flow tracks.
  *) echo "Unsupported CAE profile" >&2; exit 2 ;;
esac

if [[ ${WORLDEVAL_MANAGE_SERVICES:-1} == 1 ]]; then
  rm -f "$services_log_root/ready"
  bash "$world_repo/deployment/ziplab5090/run_worldolympiad_services.sh" \
    "$services_log_root" \
    >"$case_root/evaluator_services.log" 2>&1 &
  services_pid=$!
  for _ in $(seq 1 240); do
    [[ -f $services_log_root/ready ]] && break
    kill -0 "$services_pid" 2>/dev/null || {
      tail -n 120 "$case_root/evaluator_services.log" >&2 || true
      exit 3
    }
    sleep 5
  done
  [[ -f $services_log_root/ready ]] || {
    echo "WorldOlympiad evaluator services did not become ready" >&2
    exit 3
  }
else
  for spec in "$qwen_url:QwenVL" "$sam_url:SAM3" "$da3_url:DA3"; do
    url=${spec%:*}
    name=${spec##*:}
    curl -fsS --max-time 5 "$url/health" >/dev/null || {
      echo "$name evaluator service is not healthy at $url" >&2
      exit 3
    }
  done
fi

"$evaluation_python" "$world_repo/batch_test/evaluate_pipelines.py" \
  --worldeval-root "$world_repo" \
  --project-root "$workspace" \
  --root "$case_root" \
  --domain-name autoresearch \
  --pipelines "$pipeline" \
  --gpu-slots "${WORLDEVAL_GPU_SLOTS:-$default_worker_gpu_slots}" \
  --workers "${WORLDEVAL_WORKERS:-1}" \
  --qwen-server-urls "$qwen_url" \
  --sam3-server-urls "$sam_url" \
  --reward-3d-server-urls "$da3_url" \
  --physical-batch-mode dimension \
  --run-clip-interaction \
  --print-skipped

mapfile -t score_files < <(find "$case_root" -mindepth 2 -maxdepth 2 -type f -name '*_judge_*.json' | sort)
if [[ ${#score_files[@]} -eq 0 ]]; then
  echo "WorldOlympiad produced no per-case score files under $case_root" >&2
  exit 4
fi
PYTHONPATH="$world_repo${PYTHONPATH:+:$PYTHONPATH}" \
  "$evaluation_python" -m autoresearch.failure_detector \
  "${score_files[@]}" \
  --output "$case_root/failure_report.json" \
  --markdown-output "$case_root/failure_report.md"
