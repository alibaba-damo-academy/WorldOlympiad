# Pipeline Batch Evaluation

This document explains how to evaluate generated videos in
`outputs_batch/<domain>/` by pipeline. Use `evaluate_pipelines.py` as the
single batch entry point.

```bash
cd /path/to/OpenWorldLib
conda activate worldolympiad
```

## Case Layout

Each case directory should contain:

```text
outputs_batch/general/<case_id>/
  prompt.json
  ref_<case_id>.mp4
  <output_prefix>_gen_<case_id>.mp4
  <output_prefix>_gen_<case_id>_chunk_timestamps.json
  continuous_action_manifest_v13.json  # optional active CAE sidecar; robotics only
```

For example:

```text
outputs_batch/general/02BEoux44n8_part3/
  prompt.json
  ref_02BEoux44n8_part3.mp4
  cosmos_gen_02BEoux44n8_part3.mp4
  cosmos_gen_02BEoux44n8_part3_chunk_timestamps.json
```

Each manifest row evaluates one generated video. If one case has both
`cosmos_gen_*.mp4` and `longlive_gen_*.mp4`, evaluate those pipelines
separately so score files do not overwrite each other.

The active CAE sidecar has the exact name
`continuous_action_manifest_v13.json`. In active mode, the batch builder
requires any `continuous_action_manifest` pointer in `worldolympiad_case.json`
to name that exact file, then falls back to that exact filename only when the
pointer is absent. Historical CAE manifests are replay-only and are never
discovered through the old generic filename; select one explicitly with an
exact historical `--cae-pattern` filename. Retired v10 is rejected even when
selected explicitly. The canonical
metric contract is maintained in
[`docs/continuous_action_evaluation.md`](../docs/continuous_action_evaluation.md).
With default output naming, a resolved CAE manifest adds its metric suffix
(currently `_v13`) to the judge filename. Explicit output-name options remain
explicit overrides.

If the active sidecar is absent, the unified result records
`continuous_action.status=not_applicable`; the case is not skipped and CAE is
not treated as a zero score.

## Pipeline Names

Show the current aliases:

```bash
python worldeval/batch_test/evaluate_pipelines.py --list-pipelines
```

Common output prefixes:

| pipeline | output prefix |
| --- | --- |
| `cosmos-predict` | `cosmos` |
| `hunyuan-gamecraft` | `hunyuan_gamecraft` |
| `hunyuan-worldplay` | `hunyuan_worldplay` |
| `lingbot-world` | `lingbot_world` |
| `longlive` | `longlive` |
| `matrix-game2` | `matrix_game2` |
| `rolling-forcing` | `rolling_forcing` |
| `wow` | `wow` |
| `yume1p5` | `yume1p5` |

Default score file:

```text
<output_prefix>_judge_<case_id>.json
```

## Evaluate One Pipeline

```bash
python worldeval/batch_test/evaluate_pipelines.py \
  --domains general \
  --pipelines cosmos-predict \
  --gpu-slots 5,6,7 \
  --workers 3 \
  --qwen-server-urls http://127.0.0.1:8008,http://127.0.0.1:8009 \
  --sam3-server-urls http://127.0.0.1:8090 \
  --reward-3d-server-urls http://127.0.0.1:8092,http://127.0.0.1:8093 \
  --print-skipped
```

Evaluate several pipelines:

```bash
python worldeval/batch_test/evaluate_pipelines.py \
  --domains general gaming embodied \
  --pipelines cosmos-predict longlive matrix-game2 rolling-forcing wow \
  --gpu-slots 5,6,7 \
  --workers 3 \
  --qwen-server-urls http://127.0.0.1:8008,http://127.0.0.1:8009 \
  --sam3-server-urls http://127.0.0.1:8090 \
  --reward-3d-server-urls http://127.0.0.1:8092,http://127.0.0.1:8093
```

The evaluator skips already scored cases by default. Use `--force` to recompute
existing scores or `--limit N` to run only the first `N` pending cases for each
domain/pipeline pair.

## Custom File Names

If generated files do not use `<output_prefix>_gen_*.mp4`, pass explicit
patterns:

```bash
python worldeval/batch_test/evaluate_pipelines.py \
  --root outputs_batch/general \
  --domain-name general \
  --pipelines cosmos-predict \
  --gen-pattern 'cosmos_predict_*.mp4' \
  --chunk-pattern 'cosmos_predict_*_chunk_timestamps.json' \
  --output-name-template 'cosmos_predict_judge_{id}.json' \
  --gpu-slots 5 \
  --workers 1
```

## Outputs

For `general/cosmos-predict`, the main files are:

```text
batch_manifests/general_cosmos_all.jsonl
batch_manifests/general_cosmos_pending.jsonl
batch_logs/general_cosmos/summary_latest.jsonl
batch_logs/general_cosmos/summary_from_outputs.jsonl
batch_logs/general_cosmos/score_summary.json
batch_logs/general_cosmos/score_cases.csv
```

Use `--dry-run` before a large run to inspect the generated commands.
