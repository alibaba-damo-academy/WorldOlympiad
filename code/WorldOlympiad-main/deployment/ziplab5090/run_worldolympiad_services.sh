#!/usr/bin/env bash
set -euo pipefail

workspace=${WORLDOLYMPIAD_WORKSPACE:-$(pwd)}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
evaluation_python=${WORLDEVAL_PYTHON:-$workspace/envs/worldolympiad/bin/python}
export PATH="$(dirname "$evaluation_python"):$PATH"
log_root=${1:?usage: run_worldolympiad_services.sh LOG_ROOT}
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
if [[ $gpu_count -eq 1 ]]; then
  default_qwen_gpu=0
  default_sam_gpu=0
  default_da3_gpu=0
  default_safe_max_used_mib=1024
else
  default_qwen_gpu=1
  default_sam_gpu=2
  default_da3_gpu=3
  default_safe_max_used_mib=512
fi
qwen_gpu=${WORLDEVAL_QWEN_GPU:-$default_qwen_gpu}
sam_gpu=${WORLDEVAL_SAM_GPU:-$default_sam_gpu}
da3_gpu=${WORLDEVAL_DA3_GPU:-$default_da3_gpu}
qwen_port=${WORLDEVAL_QWEN_PORT:-8108}
sam_port=${WORLDEVAL_SAM_PORT:-8110}
da3_port=${WORLDEVAL_DA3_PORT:-8112}
startup_attempts=${WORLDEVAL_SERVICE_STARTUP_ATTEMPTS:-480}
startup_interval_seconds=${WORLDEVAL_SERVICE_STARTUP_INTERVAL_SECONDS:-5}
safe_max_used_mib=${WORLDEVAL_SAFE_MAX_USED_MIB:-$default_safe_max_used_mib}
safe_max_util_percent=${WORLDEVAL_SAFE_MAX_UTIL_PERCENT:-50}
[[ $startup_attempts =~ ^[1-9][0-9]*$ ]] || {
  echo "WORLDEVAL_SERVICE_STARTUP_ATTEMPTS must be a positive integer" >&2
  exit 2
}
[[ $startup_interval_seconds =~ ^[1-9][0-9]*$ ]] || {
  echo "WORLDEVAL_SERVICE_STARTUP_INTERVAL_SECONDS must be a positive integer" >&2
  exit 2
}
[[ $safe_max_used_mib =~ ^[0-9]+$ ]] || {
  echo "WORLDEVAL_SAFE_MAX_USED_MIB must be a non-negative integer" >&2
  exit 2
}
[[ $safe_max_util_percent =~ ^[0-9]+$ ]] || {
  echo "WORLDEVAL_SAFE_MAX_UTIL_PERCENT must be a non-negative integer" >&2
  exit 2
}

bash "$world_repo/deployment/ziplab5090/preflight_worldolympiad_evaluation.sh"
mkdir -p "$log_root"

# Hold the same device-wide lease as the native model/reference jobs for the
# whole service lifetime.  Without it an evaluation started while a model run
# is finishing sees residual memory and refuses a correctly scheduled cell;
# with it heavy model and evaluator stages queue instead of racing.  Each
# distinct device is leased once (a second flock on a separate descriptor of
# the same file would self-deadlock).
declare -A leased_gpus=()
for gpu in "$qwen_gpu" "$sam_gpu" "$da3_gpu"; do
  [[ -n ${leased_gpus[$gpu]+set} ]] && continue
  leased_gpus[$gpu]=1
  lease=$workspace/outputs/.locks/robotics_gpu_$gpu.lock
  mkdir -p "$(dirname -- "$lease")"
  exec {lease_fd}>"$lease"
  echo "Waiting for GPU $gpu job lease" >&2
  flock "$lease_fd"
  echo "Acquired GPU $gpu job lease" >&2
done

for gpu in "$qwen_gpu" "$sam_gpu" "$da3_gpu"; do
  used=999999
  util=999999
  for _ in $(seq 1 "${WORLDEVAL_GPU_SAMPLE_ATTEMPTS:-6}"); do
    stats=$(nvidia-smi --id="$gpu" \
      --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    used=$(printf '%s' "$stats" | cut -d, -f1 | tr -d ' ')
    util=$(printf '%s' "$stats" | cut -d, -f2 | tr -d ' ')
    [[ $used -le $safe_max_used_mib && $util -le $safe_max_util_percent ]] && break
    sleep 5
  done
  if [[ $used -gt $safe_max_used_mib || $util -gt $safe_max_util_percent ]]; then
    echo "Evaluator GPU $gpu is not safely free: used=${used}MiB util=${util}%" >&2
    exit 2
  fi
done

"$evaluation_python" - "$qwen_port" "$sam_port" "$da3_port" <<'PY'
import socket
import sys

occupied = []
for raw_port in sys.argv[1:]:
    port = int(raw_port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            occupied.append(port)
if occupied:
    joined = ", ".join(map(str, occupied))
    raise SystemExit(f"Refusing to reuse occupied evaluator port(s): {joined}")
PY

children=()
cleanup() {
  local pid
  for pid in "${children[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
  done
  for pid in "${children[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

CUDA_DEVICE_ORDER=PCI_BUS_ID "$evaluation_python" \
  "$world_repo/batch_test/start_qwenvl_servers.py" \
  --worldeval-root "$world_repo" \
  --gpus "$qwen_gpu" --ports "$qwen_port" \
  --model "$world_repo/weights/QwenVL" --warmup \
  >"$log_root/qwenvl.log" 2>&1 &
children+=("$!")

qwen_url="http://127.0.0.1:$qwen_port"
for _ in $(seq 1 "$startup_attempts"); do
  if curl -fsS --max-time 2 "$qwen_url/health" >/dev/null; then break; fi
  kill -0 "${children[0]}" 2>/dev/null || { tail -n 80 "$log_root/qwenvl.log" >&2; exit 3; }
  sleep "$startup_interval_seconds"
done
curl -fsS --max-time 5 "$qwen_url/health" >/dev/null || {
  echo "QwenVL service did not become healthy" >&2
  exit 3
}

CUDA_DEVICE_ORDER=PCI_BUS_ID "$evaluation_python" \
  "$world_repo/batch_test/start_sam3_servers.py" \
  --worldeval-root "$world_repo" \
  --gpus "$sam_gpu" --ports "$sam_port" \
  --model "$world_repo/weights/sam3/sam3.pt" \
  --qwen-server-urls "$qwen_url" --max-frames 128 \
  >"$log_root/sam3.log" 2>&1 &
children+=("$!")

CUDA_DEVICE_ORDER=PCI_BUS_ID "$evaluation_python" \
  "$world_repo/batch_test/start_reward_3d_servers.py" \
  --worldeval-root "$world_repo" \
  --gpus "$da3_gpu" --ports "$da3_port" \
  --model "$world_repo/weights/da3" \
  --qwen-server-urls "$qwen_url" --no-lpips \
  >"$log_root/da3.log" 2>&1 &
children+=("$!")

for spec in "$sam_port:sam3" "$da3_port:da3"; do
  port=${spec%%:*}
  name=${spec##*:}
  for _ in $(seq 1 "$startup_attempts"); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; then break; fi
    sleep "$startup_interval_seconds"
  done
  curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >/dev/null || {
    echo "$name service did not become healthy" >&2
    tail -n 80 "$log_root/$name.log" >&2 || true
    exit 3
  }
done

touch "$log_root/ready"
while true; do
  for pid in "${children[@]}"; do
    kill -0 "$pid" 2>/dev/null || { echo "Evaluator service process exited: $pid" >&2; exit 4; }
  done
  sleep 5
done
