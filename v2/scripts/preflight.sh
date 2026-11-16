#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
phase=${1:-}
[[ -n $phase ]] || { echo "usage: preflight.sh static|runtimes|preparation|evaluation [options]" >&2; exit 2; }
shift
exec python3 "$ROOT/scripts/preflight.py" "$phase" "$@"
