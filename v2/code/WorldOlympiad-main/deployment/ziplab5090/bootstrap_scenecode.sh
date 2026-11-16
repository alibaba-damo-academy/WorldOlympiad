#!/usr/bin/env bash
set -euo pipefail

workspace=${WORLDOLYMPIAD_WORKSPACE:-/mnt/data/v-wangchengzhi/worldolympiad2}
scenecode_root=${SCENECODE_ROOT:-$workspace/code/SceneCode-main}
uv_bin=${UV_BIN:-$workspace/envs/bin/uv}
scenecode_env=${SCENECODE_ENV:-$workspace/envs/scenecode}
blender_parent=${BLENDER_PARENT:-$workspace/runtime}
blender_version=4.4.3
blender_archive=$blender_parent/blender-$blender_version-linux-x64.tar.xz
blender_dir=$blender_parent/blender-$blender_version-linux-x64
blender_sha256=8d3be07d2bc412b502c6bfe3cfe3e22195a4164076867da987ce148d73c27946

if [[ ! -x "$blender_dir/blender" ]]; then
  if [[ ! -f "$blender_archive" ]]; then
    curl --fail --location --output "$blender_archive" \
      "https://download.blender.org/release/Blender4.4/blender-$blender_version-linux-x64.tar.xz"
  fi
  printf '%s  %s\n' "$blender_sha256" "$blender_archive" | sha256sum -c -
  tar -xf "$blender_archive" -C "$blender_parent"
fi

export UV_PROJECT_ENVIRONMENT=$scenecode_env
cd "$scenecode_root"
# Do not disable TLS verification. If the host's Python package proxy presents
# an invalid certificate, stage a verified wheel mirror and set UV_FIND_LINKS.
"$uv_bin" sync --no-dev --frozen --native-tls \
  --extra physics --extra generation --extra retrieval
# Blender itself is installed separately below. The official bpy wheel supplies
# a matching mathutils module without forcing the source-only mathutils package
# (which would require system Eigen headers on the compute node).
"$uv_bin" pip install --python "$scenecode_env/bin/python" --no-deps bpy==4.5.4
"$scenecode_env/bin/python" -c \
  'import agents, bpy, coacd, drake, hydra, mathutils, torch; print("SceneCode dependencies OK", torch.__version__, bpy.app.version_string)'
"$blender_dir/blender" --version | head -2
