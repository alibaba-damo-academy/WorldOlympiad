#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
workspace=${WORLDOLYMPIAD_WORKSPACE:-$(cd -- "$script_dir/../../../.." && pwd -P)}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
bridge_root=${SCENECODE_PYBULLET_BRIDGE_ROOT:-$workspace/code/scenecode_pybullet_minimal}
patch_file=$world_repo/deployment/ziplab5090/patches/scenecode_pybullet_place_on_region.patch
output_patch=$world_repo/deployment/ziplab5090/patches/scenecode_pybullet_empty_output_dir.patch
host_frame_patch=$world_repo/deployment/ziplab5090/patches/scenecode_pybullet_host_link_frame.patch
articulated_driver=$bridge_root/scenecode_pybullet_bridge/articulated.py

driver=$bridge_root/scripts/run_droid_pick_place.py
graph=$bridge_root/scenecode_pybullet_bridge/droid_operator_graph.py
push_driver=$bridge_root/scenecode_pybullet_bridge/droid_push.py
pick_driver=$bridge_root/scenecode_pybullet_bridge/droid_pick_place.py
[[ -f $driver && -f $graph && -f $push_driver && -f $pick_driver \
  && -f $patch_file && -f $output_patch && -f $host_frame_patch && -f $articulated_driver ]] || {
  echo "SceneCode PyBullet bridge or compatibility patch is missing" >&2
  exit 2
}

if { grep -Fq '"covering", "on_region"' "$driver" \
     || grep -Fq '"on_region": "on_region"' "$driver"; } \
  && grep -Fq '"place": {"in_container", "on_region"}' "$graph" \
  && grep -Fq 'step.success_predicate not in TERMINAL_PREDICATES[step.terminal_operator]' "$graph"; then
  :
elif patch --dry-run --forward --batch -p1 -d "$bridge_root" < "$patch_file" >/dev/null 2>&1; then
  patch --forward --batch -p1 -d "$bridge_root" < "$patch_file"
else
  echo "SceneCode PyBullet on_region patch is neither applicable nor present" >&2
  exit 3
fi

if grep -Fq 'output.exists() and any(output.iterdir())' "$pick_driver" \
  && grep -Fq 'output.exists() and any(output.iterdir())' "$push_driver" \
  && grep -Fq 'output.exists() and any(output.iterdir())' "$graph"; then
  :
elif patch --dry-run --forward --batch -p1 -d "$bridge_root" < "$output_patch" >/dev/null 2>&1; then
  patch --forward --batch -p1 -d "$bridge_root" < "$output_patch"
else
  echo "SceneCode PyBullet empty-output patch is neither applicable nor present" >&2
  exit 3
fi

if grep -Fq 'host_root = measured_base_link_pose(p, self.host)' "$articulated_driver" \
  && grep -Fq '"link_position": host_root["position"]' "$articulated_driver"; then
  :
elif patch --dry-run --forward --batch --fuzz=0 -p1 -d "$bridge_root" < "$host_frame_patch" >/dev/null 2>&1; then
  patch --forward --batch --fuzz=0 -p1 -d "$bridge_root" < "$host_frame_patch"
else
  echo "SceneCode PyBullet host link-frame patch is neither applicable nor present" >&2
  exit 3
fi

python_bin=${AUTORESEARCH_PYTHON:-$workspace/envs/autoresearch/bin/python}
"$python_bin" -m py_compile "$driver" "$graph" "$pick_driver" "$push_driver" "$articulated_driver"
printf '%s\n' "$bridge_root"
