#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
mode=${1:-}
case "$mode" in
  --check|--reuse-existing|--download-aux-missing) shift ;;
  *) echo "usage: setup_weights.sh (--check|--reuse-existing|--download-aux-missing) --shared-root PATH" >&2; exit 2 ;;
esac
shared_root=
while (($#)); do
  case "$1" in
    --shared-root) (($# >= 2)) || { echo "--shared-root needs a path" >&2; exit 2; }; shared_root=$2; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n $shared_root ]] || { echo "set --shared-root to the existing weights directory" >&2; exit 2; }
shared_root=$(realpath -e -- "$shared_root")
[[ -d $shared_root ]] || { echo "shared weights root is not a directory" >&2; exit 2; }
world_repo=$ROOT/code/WorldOlympiad-main
[[ -f $world_repo/deployment/ziplab5090/download_worldolympiad_weights.sh ]] || {
  echo "pinned WorldOlympiad source is missing; no weights were changed" >&2; exit 2;
}

link_or_verify() {
  local link=$1 target=$2
  if [[ -L $link ]]; then
    [[ $(realpath -e -- "$link") == "$target" ]] || {
      echo "existing symlink points elsewhere and was left untouched: $link" >&2; return 2;
    }
  elif [[ -e $link ]]; then
    [[ $(realpath -e -- "$link") == "$target" ]] || {
      echo "existing path differs from requested cache and was left untouched: $link" >&2; return 2;
    }
  else
    ln -s -- "$target" "$link"
    echo "linked existing cache: $link -> $target"
  fi
}

if [[ $mode != --check ]]; then
  link_or_verify "$ROOT/weights" "$shared_root"
  link_or_verify "$world_repo/weights" "$shared_root"
fi
export WORLDOLYMPIAD_WORKSPACE="$ROOT"
export WORLD_REPO="$world_repo"
export WORLDEVAL_MODEL_CACHE_ROOT="$shared_root"
downloader=$world_repo/deployment/ziplab5090/download_worldolympiad_weights.sh
if [[ $mode == --check || $mode == --reuse-existing ]]; then
  bash "$downloader" --check
else
  echo "Fetching missing auxiliary benchmark weights only; the configured tested-model checkpoint is not handled by this downloader."
  bash "$downloader"
fi
python3 "$ROOT/scripts/check_weights.py"
