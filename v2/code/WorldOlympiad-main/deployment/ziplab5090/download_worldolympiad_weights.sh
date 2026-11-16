#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: download_worldolympiad_weights.sh [--check]

Download or reuse the official WorldOlympiad evaluator weights under
$WORLDOLYMPIAD_WORKSPACE/weights, then link them into $WORLD_REPO/weights.

Environment:
  WORLDOLYMPIAD_WORKSPACE  Required WOV2 workspace directory.
  WORLD_REPO               Optional WorldOlympiad-main checkout path.
  WORLDEVAL_MODEL_CACHE_ROOT Optional weights root (defaults to workspace/weights).
  HF_CLI                   Optional path to the `hf` executable.
  UVX_BIN                  Optional path to `uvx` if `hf` is not installed.

Run with --check to validate files and links without downloading or changing them.
EOF
}

mode=download
case "${1:-}" in
  "") ;;
  --check) mode=check ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

workspace=${WORLDOLYMPIAD_WORKSPACE:-}
[[ -n $workspace ]] || {
  echo "Set WORLDOLYMPIAD_WORKSPACE to this user's WOV2 workspace." >&2
  exit 2
}
[[ -d $workspace ]] || {
  echo "Workspace directory does not exist: $workspace" >&2
  exit 2
}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
[[ -d $world_repo ]] || {
  echo "WorldOlympiad checkout does not exist: $world_repo" >&2
  exit 2
}

if [[ -n ${WORLDEVAL_MODEL_CACHE_ROOT:-} ]]; then
  weights_root=$WORLDEVAL_MODEL_CACHE_ROOT
else
  weights_root=$workspace/weights
fi
repo_weights=$world_repo/weights

clip_sha256=40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
clip_url="https://openaipublic.azureedge.net/clip/models/$clip_sha256/ViT-B-32.pt"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "Need sha256sum or shasum to validate CLIP weights." >&2
    return 2
  fi
}

validate_qwenvl() {
  local dir=$1
  local required
  for required in \
    config.json generation_config.json model.safetensors.index.json \
    chat_template.json merges.txt preprocessor_config.json \
    tokenizer.json tokenizer_config.json video_preprocessor_config.json vocab.json; do
    if [[ ! -s $dir/$required ]]; then
      echo "QwenVL is missing required file: $dir/$required" >&2
      return 1
    fi
  done
  python3 - "$dir" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
index_path = root / "model.safetensors.index.json"
try:
    index = json.loads(index_path.read_text())
    shard_names = sorted(set(index["weight_map"].values()))
except Exception as exc:
    print(f"Invalid QwenVL shard index {index_path}: {exc}", file=sys.stderr)
    raise SystemExit(1)

if not shard_names:
    print(f"QwenVL shard index contains no weight files: {index_path}", file=sys.stderr)
    raise SystemExit(1)

missing = [name for name in shard_names if not (root / name).is_file() or (root / name).stat().st_size == 0]
if missing:
    print("QwenVL is missing or has empty shard(s): " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
print(f"QwenVL shard index valid: {len(shard_names)} non-empty shards")
PY
}

validate_da3() {
  local dir=$1
  [[ -s $dir/config.json && -s $dir/model.safetensors ]] || {
    echo "DA3 requires non-empty config.json and model.safetensors under $dir" >&2
    return 1
  }
}

validate_sam3() {
  [[ -s $1/sam3.pt ]] || {
    echo "SAM3 requires a non-empty sam3.pt under $1" >&2
    return 1
  }
}

validate_clip() {
  local file=$1
  [[ -s $file ]] || {
    echo "CLIP weight is missing or empty: $file" >&2
    return 1
  }
  local actual
  actual=$(sha256_file "$file") || return
  [[ $actual == "$clip_sha256" ]] || {
    echo "CLIP checksum mismatch for $file" >&2
    echo "  expected: $clip_sha256" >&2
    echo "  actual:   $actual" >&2
    return 1
  }
}

report_paths() {
  printf 'Weights root: %s\n' "$weights_root"
  printf '  QwenVL: %s\n' "$weights_root/QwenVL"
  printf '  DA3:    %s\n' "$weights_root/da3"
  printf '  SAM3:   %s\n' "$weights_root/sam3/sam3.pt"
  printf '  CLIP:   %s\n' "$weights_root/clip/ViT-B-32.pt"
}

offer_local_staging_option() {
  cat >&2 <<'EOF'
If this machine cannot reach the required download host, you may choose to run
this same script on a machine that can, then rsync its workspace/weights/
directory here. The script does not switch machines or start that transfer.
See deployment/ziplab5090/README.md for the commands.
EOF
}

check_all() {
  local valid=0
  validate_qwenvl "$weights_root/QwenVL" || valid=1
  validate_da3 "$weights_root/da3" || valid=1
  validate_sam3 "$weights_root/sam3" || valid=1
  validate_clip "$weights_root/clip/ViT-B-32.pt" || valid=1
  if [[ $valid -ne 0 ]]; then
    report_paths >&2
    return 1
  fi

  for name in QwenVL da3 sam3 clip; do
    if [[ ! -e $repo_weights/$name ]]; then
      echo "Runtime link/path is missing: $repo_weights/$name" >&2
      return 1
    fi
    if [[ ! $repo_weights/$name -ef $weights_root/$name ]]; then
      echo "Runtime path does not point to the validated workspace weights: $repo_weights/$name" >&2
      return 1
    fi
  done
  echo "All evaluator weights and runtime links are valid."
}

if [[ $mode == check ]]; then
  check_all
  exit
fi

mkdir -p "$weights_root" "$repo_weights"

# Preserve already-downloaded files in the checkout by exposing them at the
# canonical workspace path instead of downloading a duplicate copy.
adopt_checkout_weight() {
  local name=$1
  local validator=$2
  local cache_path=$weights_root/$name
  local checkout_path=$repo_weights/$name
  if [[ ! -e $cache_path && ! -L $cache_path ]] && \
    "$validator" "$checkout_path" >/dev/null 2>&1; then
    ln -s "$checkout_path" "$cache_path"
    echo "reusing existing checkout weight: $checkout_path"
  fi
}
validate_checkout_clip() {
  validate_clip "$1/ViT-B-32.pt"
}
adopt_checkout_weight QwenVL validate_qwenvl
adopt_checkout_weight da3 validate_da3
adopt_checkout_weight sam3 validate_sam3
adopt_checkout_weight clip validate_checkout_clip

hf_cli=${HF_CLI:-}
if [[ -z $hf_cli ]]; then
  hf_cli=$(command -v hf || true)
fi
if [[ -z $hf_cli && -x $workspace/envs/lingbot/bin/hf ]]; then
  hf_cli=$workspace/envs/lingbot/bin/hf
fi
uvx_bin=${UVX_BIN:-}
if [[ -z $uvx_bin || ! -x $uvx_bin ]]; then
  uvx_bin=$(command -v uvx || true)
fi

run_hf() {
  if [[ -n $hf_cli && -x $hf_cli ]]; then
    "$hf_cli" "$@"
  elif [[ -n $uvx_bin && -x $uvx_bin ]]; then
    "$uvx_bin" hf "$@"
  else
    echo 'Hugging Face CLI is unavailable. Install hf or provide UVX_BIN.' >&2
    return 127
  fi
}

ensure_downloader() {
  if { [[ -n $hf_cli && -x $hf_cli ]]; } || { [[ -n $uvx_bin && -x $uvx_bin ]]; }; then
    return 0
  fi
  echo 'Install the Hugging Face CLI (hf) or uv (uvx) to download missing weights.' >&2
  return 2
}

prepare_sam3_access() {
  local sam_dir=$weights_root/sam3
  local sam_partial="$sam_dir.partial"
  local output
  local status

  [[ -s $sam_dir/sam3.pt ]] && return 0
  [[ ${sam3_preflight_ok:-0} == 1 ]] && return 0

  ensure_downloader || return
  if output=$(run_hf auth whoami 2>&1); then
    :
  else
    if printf '%s' "$output" | grep -Eqi 'ConnectError|ConnectionError|ReadTimeout|ConnectTimeout|NameResolution|network|timed out|temporary failure'; then
      echo "This machine cannot reach Hugging Face to check SAM3 access." >&2
      echo "You may choose to run this script on a machine that can reach Hugging Face, then transfer the weights; see deployment/ziplab5090/README.md." >&2
    else
    if [[ -n $hf_cli && -x $hf_cli ]]; then
      echo "On this machine run: $hf_cli auth login" >&2
    else
      echo "On this machine run: $uvx_bin hf auth login" >&2
    fi
      cat >&2 <<'EOF'
SAM3 requires an authenticated Hugging Face account with access to facebook/sam3.
Then accept/request access on:
  https://huggingface.co/facebook/sam3
The request may need approval. No large SAM3 file has been downloaded yet.
EOF
    fi
    return 4
  fi

  mkdir -p "$sam_partial"
  if output=$(run_hf download facebook/sam3 README.md --local-dir "$sam_partial" 2>&1); then
    sam3_preflight_ok=1
    echo "SAM3 access preflight passed for the current Hugging Face account."
    return 0
  else
    status=$?
  fi

  if printf '%s' "$output" | grep -Eqi 'GatedRepoError|gated repo|403|401|not authorized|request access|restricted access'; then
    echo "SAM3 access preflight was denied by Hugging Face (exit $status)." >&2
    cat >&2 <<'EOF'
The current account needs access to facebook/sam3. Log in with the account that
accepted the model terms and wait for approval if an access request is pending.
No large SAM3 file has been downloaded.
EOF
  elif printf '%s' "$output" | grep -Eqi 'ConnectError|ConnectionError|ReadTimeout|ConnectTimeout|NameResolution|network|timed out|temporary failure'; then
    echo "SAM3 access preflight could not reach Hugging Face (exit $status)." >&2
    cat >&2 <<'EOF'
Use a machine that can reach Hugging Face to run this same script, then transfer
the completed weights directory to this workspace. See deployment/ziplab5090/README.md.
No large SAM3 file has been downloaded.
EOF
  else
    echo "SAM3 access preflight failed for another reason (exit $status)." >&2
    echo 'Check `hf auth whoami`, the Hugging Face connection, and the SAM3 model page; no large SAM3 file has been downloaded.' >&2
  fi
  return 4
}

download_snapshot() {
  local repo_id=$1
  local name=$2
  local validator=$3
  local destination=$weights_root/$name
  local partial="$destination.partial"
  local attempt

  if "$validator" "$destination" >/dev/null 2>&1; then
    echo "reusing complete weight: $destination"
    return 0
  fi
  if [[ -e $destination || -L $destination ]]; then
    echo "Weight destination exists but is incomplete; preserving it: $destination" >&2
    echo "Move the incomplete directory aside yourself; any partial download at $partial will be retained and resumed." >&2
    return 3
  fi

  mkdir -p "$partial"
  for attempt in 1 2 3 4 5; do
    if run_hf download "$repo_id" --local-dir "$partial"; then
      if "$validator" "$partial"; then
        [[ ! -e $destination && ! -L $destination ]] || {
          echo "Weight destination appeared during download; preserving both paths: $destination" >&2
          return 3
        }
        mv "$partial" "$destination"
        echo "installed $repo_id at $destination"
        return 0
      fi
      echo "Downloaded snapshot is incomplete; partial files are retained for retry: $partial" >&2
      return 3
    fi
    if [[ $attempt -lt 5 ]]; then
      echo "Download failed; retrying $repo_id ($attempt/5). Partial files remain at $partial." >&2
      sleep 15
    fi
  done
  echo "Download failed after five attempts: $repo_id" >&2
  echo "Partial files remain at $partial; resolve network/access issues and rerun to resume." >&2
  offer_local_staging_option
  return 3
}

download_sam3() {
  local destination=$weights_root/sam3
  local partial="$destination.partial"
  local attempt
  if validate_sam3 "$destination" >/dev/null 2>&1; then
    echo "reusing complete weight: $destination"
    return 0
  fi
  if [[ -e $destination || -L $destination ]]; then
    echo "SAM3 destination exists but has no non-empty sam3.pt; preserving it: $destination" >&2
    echo "Move the incomplete directory aside yourself; any partial download at $partial will be retained and resumed." >&2
    return 3
  fi
  prepare_sam3_access || return
  mkdir -p "$partial"
  for attempt in 1 2 3 4 5; do
    if run_hf download facebook/sam3 sam3.pt --local-dir "$partial"; then
      if validate_sam3 "$partial"; then
        [[ ! -e $destination && ! -L $destination ]] || {
          echo "SAM3 destination appeared during download; preserving both paths: $destination" >&2
          return 3
        }
        mv "$partial" "$destination"
        echo "installed facebook/sam3 sam3.pt at $destination/sam3.pt"
        return 0
      fi
    fi
    if [[ $attempt -lt 5 ]]; then
      echo "SAM3 download failed; retrying ($attempt/5). Partial files remain at $partial." >&2
      sleep 15
    fi
  done
  echo "SAM3 download failed after five attempts; partial files remain at $partial." >&2
  echo "Check the HF login, SAM3 approval, and network, then rerun to resume." >&2
  offer_local_staging_option
  return 3
}

download_clip() {
  local destination=$weights_root/clip/ViT-B-32.pt
  local partial="$destination.partial"
  mkdir -p "$(dirname "$destination")"
  if [[ -s $destination ]]; then
    validate_clip "$destination" || {
      echo "Preserving the existing CLIP file; resolve the checksum mismatch before retrying." >&2
      return 3
    }
    echo "reusing complete weight: $destination"
    return 0
  fi
  if [[ -e $destination || -L $destination ]]; then
    echo "CLIP destination exists but is empty or invalid; preserving it: $destination" >&2
    return 3
  fi
  curl --fail --location --retry 5 --continue-at - \
    --output "$partial" "$clip_url" || {
    echo "CLIP download failed; partial file remains at $partial. Rerun to resume." >&2
    offer_local_staging_option
    return 3
  }
  validate_clip "$partial" || {
    echo "CLIP partial failed its checksum and was preserved: $partial" >&2
    return 3
  }
  [[ ! -e $destination && ! -L $destination ]] || {
    echo "CLIP destination appeared during download; preserving both paths: $destination" >&2
    return 3
  }
  mv "$partial" "$destination"
  echo "installed OpenAI CLIP ViT-B/32 at $destination"
}

link_weight() {
  local name=$1
  local source=$weights_root/$name
  local destination=$repo_weights/$name
  if [[ -e $destination || -L $destination ]]; then
    if [[ $destination -ef $source ]]; then
      return 0
    fi
    echo "Runtime path already exists and points elsewhere; preserving it: $destination" >&2
    return 3
  fi
  ln -s "$source" "$destination"
  echo "linked $destination -> $source"
}

echo "Preparing evaluator weights in the user workspace."
report_paths
command -v python3 >/dev/null 2>&1 || {
  echo "python3 is required to validate the QwenVL shard index." >&2
  exit 2
}

# Check SAM3 permission before starting any multi-gigabyte download.
if [[ ! -s $weights_root/sam3/sam3.pt && ! -s $repo_weights/sam3/sam3.pt ]]; then
  prepare_sam3_access || exit $?
fi

need_hf=0
validate_qwenvl "$weights_root/QwenVL" >/dev/null 2>&1 || need_hf=1
validate_da3 "$weights_root/da3" >/dev/null 2>&1 || need_hf=1
validate_sam3 "$weights_root/sam3" >/dev/null 2>&1 || need_hf=1
if [[ $need_hf == 1 ]]; then
  ensure_downloader || exit $?
fi
if [[ $need_hf == 1 && ${sam3_preflight_ok:-0} != 1 ]] && \
  { ! validate_qwenvl "$weights_root/QwenVL" >/dev/null 2>&1 || ! validate_da3 "$weights_root/da3" >/dev/null 2>&1; }; then
  qwen_partial="$weights_root/QwenVL.partial"
  mkdir -p "$qwen_partial"
  if ! run_hf download Qwen/Qwen3-VL-8B-Instruct README.md --local-dir "$qwen_partial"; then
    echo "Hugging Face connectivity preflight failed; large model downloads were not started." >&2
    offer_local_staging_option
    exit 4
  fi
  echo "Hugging Face connectivity preflight passed."
fi
download_snapshot Qwen/Qwen3-VL-8B-Instruct QwenVL validate_qwenvl
download_snapshot depth-anything/DA3NESTED-GIANT-LARGE-1.1 da3 validate_da3
download_sam3
download_clip

for name in QwenVL da3 sam3 clip; do
  link_weight "$name"
done

check_all
echo "WorldOlympiad evaluator weights are ready."
