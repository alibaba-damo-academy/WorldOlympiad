#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PYTHON=${PYTHON:-python3}
cmd=${1:-all}

case "$cmd" in
  benchmark-check|benchmark-run|benchmark-score)
    action=${cmd#benchmark-}
    shift
    runtime_python="$ROOT/envs/autoresearch/bin/python"
    [[ -x $runtime_python ]] || runtime_python=$PYTHON
    exec "$runtime_python" "$ROOT/scripts/benchmark.py" "$action" "$@"
    ;;
  dataset-route)
    shift || true
    exec "$PYTHON" "$ROOT/scripts/hf_dataset_router.py" "$@"
    ;;
  configure)
    shift || true
    exec "$PYTHON" "$ROOT/scripts/configure.py" "$@"
    ;;
  set-api-key)
    shift || true
    exec "$PYTHON" "$ROOT/scripts/set_api_key.py" "$@"
    ;;
  sources)
    shift || true
    exec "$PYTHON" "$ROOT/scripts/setup_sources.py" "$@"
    ;;
  runtime-reuse)
    shift || true
    exec "$ROOT/scripts/reuse_server_runtime.sh" "$@"
    ;;
  rsi-ready)
    shift || true
    exec "$PYTHON" "$ROOT/scripts/prepare_rsi_base.py" "$@"
    ;;
  sources-check)
    exec "$PYTHON" "$ROOT/scripts/setup_sources.py" --check-only
    ;;
  sources-verify)
    exec "$PYTHON" "$ROOT/scripts/setup_sources.py" --verify-only
    ;;
  sources-unresolved)
    exec "$PYTHON" "$ROOT/scripts/setup_sources.py" --list-unresolved
    ;;
  weights-check)
    shift || true
    exec "$ROOT/scripts/setup_weights.sh" --check "$@"
    ;;
  weights-reuse)
    shift || true
    exec "$ROOT/scripts/setup_weights.sh" --reuse-existing "$@"
    ;;
  weights-fetch-aux)
    shift || true
    exec "$ROOT/scripts/setup_weights.sh" --download-aux-missing "$@"
    ;;
  robotwin-assets-check)
    exec "$PYTHON" "$ROOT/scripts/setup_robotwin_assets.py" --check-only
    ;;
  robotwin-assets-download)
    exec "$PYTHON" "$ROOT/scripts/setup_robotwin_assets.py" --download
    ;;
  runtime)
    shift || true
    exec "$ROOT/scripts/setup_runtime.sh" "$@"
    ;;
  preflight-preparation)
    shift || true
    exec "$ROOT/scripts/preflight.sh" preparation "$@"
    ;;
  preflight-evaluation)
    shift || true
    exec "$ROOT/scripts/preflight.sh" evaluation "$@"
    ;;
  preflight-scoring)
    shift || true
    exec "$ROOT/scripts/preflight.sh" scoring "$@"
    ;;
  run)
    shift || true
    exec "$ROOT/scripts/run_eval.sh" "$@"
    ;;
  all)
    shift || true
    shared_root=${WOV2_SHARED_WEIGHTS:-}
    if [[ $# -eq 2 && $1 == --shared-root ]]; then
      shared_root=$2
    elif [[ $# -gt 0 ]]; then
      echo "usage: ./setup.sh all [--shared-root EXISTING_WEIGHTS_DIR]" >&2
      exit 2
    fi
    [[ -n $shared_root ]] || {
      echo "set WOV2_SHARED_WEIGHTS to the existing data-disk weights directory; no download will be attempted" >&2
      exit 2
    }
    [[ -d $shared_root ]] || {
      echo "shared weights directory does not exist; source cloning has not started: $shared_root" >&2
      exit 2
    }
    "$PYTHON" "$ROOT/scripts/setup_sources.py"
    "$ROOT/scripts/setup_weights.sh" --reuse-existing --shared-root "$shared_root"
    "$ROOT/scripts/setup_runtime.sh"
    "$ROOT/scripts/preflight.sh" static
    ;;
  -h|--help|help)
    cat <<'HELP'
Usage: ./setup.sh [configure|set-api-key|dataset-route|all|sources [--only SOURCE]|sources-check|sources-verify|sources-unresolved|weights-check|weights-reuse|weights-fetch-aux|robotwin-assets-check|robotwin-assets-download|runtime-reuse|rsi-ready|runtime [--online-install]|preflight-preparation|preflight-evaluation|preflight-scoring|run|benchmark-check|benchmark-run|benchmark-score]

configure               Configure Agent GPT API (--agent-base-url / --agent-model) and tested-model profile.
set-api-key             Securely store WO_AGENT_API_KEY in .secrets/scenecode.env (mode 0600, no echo).
dataset-route           Route & materialize embodiment-matched cases from HuggingFace WorldOlympiad 2.0.
sources                 Materialize locked sources; accepts repeated --only SOURCE filters.
sources-unresolved      Show a bundled source without an upstream Git revision.
runtime-reuse           Link an existing server's verified Python environments and read-only assets.
rsi-ready               Create/verify a deterministic local Git baseline and RSI worktree.
weights-reuse           Link and validate an existing shared weights root; no downloads.
weights-fetch-aux        Explicitly allow fetching missing auxiliary weights only.
robotwin-assets-check     Check required RoboTwin embodiment/object data; no download.
robotwin-assets-download  Explicitly download/extract data after Hugging Face access is ready.
runtime                 Prepare separate project-local Python runtimes. Pass --online-install for explicit network bootstrap.
preflight-preparation   Check configuration, assets, API key presence and one-card gate.
preflight-evaluation    Check the full policy/simulator topology.
preflight-scoring       Check the one-card or split VLM/compute scoring topology.
run                     Start prepare, continue or full evaluation.
all                     Reuse existing weights, prepare sources/runtimes and verify the setup.

Set WOV2_SHARED_WEIGHTS to the existing data-disk cache for `all` or `weights-reuse`.
No setup command deletes existing workspaces, checkouts, outputs, environments or weights.
The tested-model checkpoint is never downloaded or replaced. GPU jobs are started only by `run`.
HELP
    ;;
  *)
    echo "unknown command: $cmd" >&2
    exit 2
    ;;
esac
