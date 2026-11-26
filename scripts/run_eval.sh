#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
phase=
run_id=
goal_template=
while (($#)); do
  case "$1" in
    --phase) (($# >= 2)) || { echo "--phase requires prepare|continue|full" >&2; exit 2; }; phase=$2; shift 2 ;;
    --run) (($# >= 2)) || { echo "--run requires an id" >&2; exit 2; }; run_id=$2; shift 2 ;;
    --goal-template) (($# >= 2)) || { echo "--goal-template requires a file path" >&2; exit 2; }; goal_template=$2; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: ./scripts/run_eval.sh --phase prepare|continue|full --run RUN_ID [--goal-template GOAL.md]

prepare: create fresh task(s), generate SceneCode scene/reference (+ RSI repair), stop before model inference.
continue: reuse the verified task/reference and finish native episode, score and report.
full: perform all stages in one launch when the full resource topology is ready.
--goal-template: optional custom goal Markdown file (defaults to built-in reference/evaluation goals).

The adapter and RSI engineering tools remain enabled. Every run is preserved.
EOF
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ $phase == prepare || $phase == continue || $phase == full ]] || { echo "choose --phase prepare|continue|full" >&2; exit 2; }
[[ -n $run_id ]] || { echo "--run is required" >&2; exit 2; }
python3 "$ROOT/scripts/validate_run.py" "$([[ $phase == continue ]] && echo continue || echo fresh)" --run "$run_id"
python3 "$ROOT/scripts/prepare_rsi_base.py"
export WO_AGENT_ENV_FILE="${WO_AGENT_ENV_FILE:-$ROOT/.secrets/scenecode.env}"
if [[ $phase == prepare ]]; then
  "$ROOT/scripts/preflight.sh" preparation --run "$run_id"
elif [[ $phase == continue ]]; then
  "$ROOT/scripts/preflight.sh" evaluation --run "$run_id" --allow-same-run-queue
else
  "$ROOT/scripts/preflight.sh" evaluation --run "$run_id"
fi

WORLD=$ROOT/code/WorldOlympiad-main
PY=$ROOT/envs/autoresearch/bin/python
[[ -x $PY ]] || { echo "project-local Agent/worker Python is missing: $PY" >&2; exit 2; }
if [[ -z "${WO_FFMPEG:-}" ]]; then
  FFMPEG_PY=$ROOT/envs/lingbot/bin/python
  [[ -x $FFMPEG_PY ]] || { echo "project-local video runtime is missing: $FFMPEG_PY" >&2; exit 2; }
  WO_FFMPEG=$("$FFMPEG_PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())') || {
    echo "could not resolve ffmpeg from the project LingBot runtime" >&2; exit 2;
  }
fi
[[ -x $WO_FFMPEG ]] || { echo "configured WO_FFMPEG is missing or not executable" >&2; exit 2; }
export WO_FFMPEG
export WO_WORKSPACE="$ROOT" WORLDOLYMPIAD_WORKSPACE="$ROOT"
export WO_EVAL_CONFIG="$ROOT/config/two-interface.json"
export PYTHONPATH="$WORLD${PYTHONPATH:+:$PYTHONPATH}"
if [[ $phase == prepare ]]; then
  export WO_RESOURCE_PHASE=preparation
  tpl="${goal_template:-$WORLD/wo/agent/goal_user_reference.md}"
  "$PY" "$ROOT/scripts/prepare_scenecode_profile.py" --run "$run_id" \
    --goal-template "$tpl"
  export WO_SCENECODE_SOURCE_CONFIG_ROOT="$ROOT/outputs/wo_runs/$run_id/scenecode_profile"
  goal=$ROOT/outputs/evaluator/goals/${run_id}_reference.md
  extra=(--through-reference)
elif [[ $phase == continue ]]; then
  export WO_RESOURCE_PHASE=evaluation
  goal="${goal_template:-$ROOT/goals/continue_reference.md}"
  extra=()
else
  export WO_RESOURCE_PHASE=evaluation
  tpl="${goal_template:-$WORLD/wo/agent/goal_user_evaluation.md}"
  "$PY" "$ROOT/scripts/prepare_scenecode_profile.py" --run "$run_id" \
    --goal-template "$tpl" --goal-kind evaluation
  export WO_SCENECODE_SOURCE_CONFIG_ROOT="$ROOT/outputs/wo_runs/$run_id/scenecode_profile"
  goal=$ROOT/outputs/evaluator/goals/${run_id}_evaluation.md
  extra=()
fi
[[ -f $goal ]] || { echo "goal file is missing: $goal" >&2; exit 2; }
logdir=$ROOT/outputs/evaluator/launcher
mkdir -p "$logdir"
log="$logdir/${run_id}_${phase}_$(date -u +%Y%m%dT%H%M%SZ).log"
printf 'Starting phase=%s run=%s; adapter and RSI engineering remain enabled.\n' "$phase" "$run_id" | tee -a "$log"
cd "$WORLD"
"$PY" -m wo.agent.launch \
  --config "$ROOT/config/two-interface.json" \
  --run "$run_id" \
  --goal "$goal" \
  --rounds 0 \
  --min-gap-s 600 \
  ${extra[@]+"${extra[@]}"} 2>&1 | tee -a "$log"
