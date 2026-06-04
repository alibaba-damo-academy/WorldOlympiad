# Batch Pipeline Evaluation

This directory contains the reusable batch evaluation tools for WorldEval. The
main entry point is `evaluate_pipelines.py`; the lower-level manifest,
scheduler, service, and summarization scripts are kept for direct debugging.

Activate the exported environment before running evaluation:

```bash
conda activate worldolympiad
```

## Quick Start

From the OpenWorldLib project root:

```bash
python worldeval/batch_test/evaluate_pipelines.py \
  --domains general gaming embodied \
  --pipelines cosmos-predict longlive matrix-game2 rolling-forcing wow \
  --gpu-slots 5,6,7 \
  --workers 3 \
  --qwen-server-urls http://127.0.0.1:8008,http://127.0.0.1:8009 \
  --sam3-server-urls http://127.0.0.1:8090 \
  --reward-3d-server-urls http://127.0.0.1:8092,http://127.0.0.1:8093 \
  --print-skipped
```

The script builds manifests, filters completed cases, runs
`batch_scheduler.py`, refreshes progress summaries, and writes aggregate score
files.

Outputs:

```text
batch_manifests/<domain>_<pipeline>_all.jsonl
batch_manifests/<domain>_<pipeline>_pending.jsonl
batch_logs/<domain>_<pipeline>/summary_latest.jsonl
batch_logs/<domain>_<pipeline>/summary_from_outputs.jsonl
batch_logs/<domain>_<pipeline>/score_summary.json
batch_logs/<domain>_<pipeline>/score_cases.csv
```

## Data Layout

Each case directory should contain:

```text
outputs_batch/<domain>/<case_id>/
  prompt.json
  ref_<case_id>.mp4
  <output_prefix>_gen_<case_id>.mp4
  <output_prefix>_gen_<case_id>_chunk_timestamps.json
```

Each generated pipeline is evaluated independently. For example,
`cosmos_gen_*.mp4` and `longlive_gen_*.mp4` should produce separate manifests
and separate score files.

## Useful Commands

List supported pipeline aliases:

```bash
python worldeval/batch_test/evaluate_pipelines.py --list-pipelines
```

Evaluate a single custom root:

```bash
python worldeval/batch_test/evaluate_pipelines.py \
  --root outputs_batch/general \
  --domain-name general \
  --pipelines cosmos-predict \
  --gpu-slots 5,6 \
  --workers 2 \
  --qwen-server-urls http://127.0.0.1:8008
```

Run only a small pending slice:

```bash
python worldeval/batch_test/evaluate_pipelines.py \
  --domains embodied gaming \
  --pipelines wow rolling-forcing matrix-game2 \
  --limit 40 \
  --gpu-slots 0,1,2,3 \
  --workers 4 \
  --qwen-server-urls http://127.0.0.1:8108,http://127.0.0.1:8109
```

Print commands without running scoring:

```bash
python worldeval/batch_test/evaluate_pipelines.py \
  --root outputs_batch/general \
  --domain-name general \
  --pipelines cosmos-predict \
  --gpu-slots 5 \
  --workers 1 \
  --dry-run
```

## Persistent Services

Start services in separate terminals and keep them running.

QwenVL:

```bash
python worldeval/batch_test/start_qwenvl_servers.py \
  --gpus 0,1 \
  --ports 8008,8009 \
  --model worldeval/weights/QwenVL \
  --warmup
```

SAM3:

```bash
python worldeval/batch_test/start_sam3_servers.py \
  --gpus 2 \
  --ports 8090 \
  --model worldeval/weights/sam3/sam3.pt \
  --qwen-server-urls http://127.0.0.1:8008,http://127.0.0.1:8009
```

DA3 / 3D reward:

```bash
python worldeval/batch_test/start_reward_3d_servers.py \
  --gpus 3,4 \
  --ports 8092,8093 \
  --model worldeval/weights/da3 \
  --qwen-server-urls http://127.0.0.1:8008,http://127.0.0.1:8009 \
  --no-lpips
```

Recommended 8-GPU layout:

```text
GPU 0-1: QwenVL services
GPU 2: SAM3 service
GPU 3-4: DA3 reward services
GPU 5-7: scoring workers
```

## Lower-Level Tools

- `make_manifest.py`: create JSONL manifests from case directories.
- `batch_scheduler.py`: run the single-video scorer with controlled workers.
- `summarize_scores.py`: aggregate per-case score JSON files into JSON/CSV.
- `start_qwenvl_servers.py`, `start_sam3_servers.py`,
  `start_reward_3d_servers.py`: persistent service launchers.
