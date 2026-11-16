#!/usr/bin/env bash
set -euo pipefail

workspace=${WORLDOLYMPIAD_WORKSPACE:-/mnt/data/v-wangchengzhi/worldolympiad2}
bootstrap_dir=${AUTORESEARCH_BOOTSTRAP_DIR:-$workspace/envs/bootstrap/autoresearch}
# The shared /mnt/data mount is excellent for durable artifacts but very slow
# for thousands of small Python package writes. Keep the reproducible inputs on
# /mnt/data and materialize the disposable runtime on node-local storage.
environment_dir=${AUTORESEARCH_ENV_DIR:-/tmp/worldolympiad-autoresearch-py311}
archive=$bootstrap_dir/python311.tar.gz
expected_sha256=b9fff092374acebc451bb881156afe6a58991213d25386358742fee296128909

if [[ -e "$environment_dir" ]]; then
  echo "Refusing to overwrite existing environment: $environment_dir" >&2
  exit 2
fi
printf '%s  %s\n' "$expected_sha256" "$archive" | sha256sum -c -
mkdir -p "$environment_dir"
tar -xzf "$archive" -C "$environment_dir" --strip-components=1
"$environment_dir/bin/python3" -m pip install \
  --no-index \
  --find-links "$bootstrap_dir/wheels" \
  -r "$workspace/code/WorldOlympiad-main/requirements-autoresearch.txt"
"$environment_dir/bin/python3" -c \
  'import cv2, numpy, pybullet, pydantic, scipy, trimesh; print("autoresearch dependencies OK")'
