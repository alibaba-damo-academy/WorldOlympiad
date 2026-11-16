#!/usr/bin/env bash
set -euo pipefail

workspace=${WORLDOLYMPIAD_WORKSPACE:-/mnt/data/v-wangchengzhi/worldolympiad2}
world_repo=${WORLD_REPO:-$workspace/code/WorldOlympiad-main}
run_id=${1:?usage: prepare_worldolympiad_evaluation.sh RUN_ID WM_RUN CASE_SET PIPELINE}
wm_run=${2:?usage: prepare_worldolympiad_evaluation.sh RUN_ID WM_RUN CASE_SET PIPELINE}
case_set=${3:?usage: prepare_worldolympiad_evaluation.sh RUN_ID WM_RUN CASE_SET PIPELINE}
pipeline=${4:?usage: prepare_worldolympiad_evaluation.sh RUN_ID WM_RUN CASE_SET PIPELINE}

safe='^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$'
[[ $run_id =~ $safe && $wm_run =~ $safe && $case_set =~ $safe ]] || {
  echo "run, WM run, and case-set ids must be filesystem-safe" >&2
  exit 2
}
case "$pipeline" in
  cosmos-predict) model_id=cosmos ;;
  lingbot-world) model_id=lingbot_world ;;
  *) echo "Unsupported real World Model pipeline: $pipeline" >&2; exit 2 ;;
esac

archive="$workspace/outputs/scenecode_pybullet_cae_$run_id.tar.zst"
wm_root="$workspace/outputs/wm_runs/$wm_run"
output_root="$workspace/outputs/evaluation_cases/$case_set"
[[ -f $archive ]] || { echo "Certified-reference archive is missing: $archive" >&2; exit 2; }
[[ -d $wm_root ]] || { echo "World Model video directory is missing: $wm_root" >&2; exit 2; }
[[ -f $wm_root/wam_run_manifest.json ]] || {
  echo "Real WAM provenance manifest is missing: $wm_root/wam_run_manifest.json" >&2
  exit 2
}
[[ ! -e $output_root ]] || { echo "Refusing to overwrite evaluation case set: $output_root" >&2; exit 2; }

PYTHONPATH="$world_repo${PYTHONPATH:+:$PYTHONPATH}" \
  "${AUTORESEARCH_PYTHON:-$workspace/envs/autoresearch/bin/python}" \
  - "$wm_root/wam_run_manifest.json" <<'PY'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
if manifest.get("event_count") != 4:
    raise SystemExit("WAM manifest must contain exactly four events")
if manifest.get("leaderboard_eligible") is not False:
    raise SystemExit("synthetic Pilot WAM run must be marked leaderboard_eligible=false")
if manifest.get("evaluation_scope") != "pipeline_validation_cross_domain":
    raise SystemExit("unexpected WAM evaluation scope")
PY

scratch=$(mktemp -d "/tmp/worldolympiad-eval-${case_set}.XXXXXX")
cleanup() { rm -rf -- "$scratch"; }
trap cleanup EXIT
release_name="scenecode_pybullet_cae_$run_id"
members=()
for event in block_sort_tray cup_tip block_push_target stack_collapse; do
  for name in episode_spec.json benchmark_certificate.json rgb.mp4 \
    camera_calibration_info.json initial_depth.pt scene_data.yaml; do
    members+=("$release_name/pilot/episodes/$event/reference/$name")
  done
done
tar --zstd -xf "$archive" -C "$scratch" "${members[@]}"
release_root="$scratch/$release_name"
[[ -d $release_root/pilot/episodes ]] || { echo "Archive has no certified Pilot episodes" >&2; exit 3; }

PYTHONPATH="$world_repo${PYTHONPATH:+:$PYTHONPATH}" \
  "${AUTORESEARCH_PYTHON:-$workspace/envs/autoresearch/bin/python}" \
  -m autoresearch.worldolympiad_case \
  "$release_root" "$wm_root" "$output_root" --model-id "$model_id"
