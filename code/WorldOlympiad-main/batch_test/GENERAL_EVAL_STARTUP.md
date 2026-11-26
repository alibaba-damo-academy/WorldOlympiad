# General Batch Eval Startup Notes

This note records a six-GPU layout for evaluating `outputs_batch/general`.

## GPU Layout

```text
GPU 0: QwenVL service, port 8008
GPU 1: SAM3 service, port 8090
GPU 2: Reward 3D service, port 8092
GPU 3: Reward 3D service, port 8093
GPU 4-5: batch scoring workers
```

Activate the environment first:

```bash
conda activate worldolympiad
```

## 1. Start QwenVL

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID python worldeval/batch_test/start_qwenvl_servers.py \
  --gpus 0 \
  --ports 8008 \
  --model worldeval/weights/QwenVL \
  --warmup
```

## 2. Start SAM3

`start_sam3_servers.py` resolves numeric GPU IDs to UUIDs by default, so the
service can safely expose the selected physical GPU as `cuda:0` internally.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID python worldeval/batch_test/start_sam3_servers.py \
  --gpus 1 \
  --ports 8090 \
  --model worldeval/weights/sam3/sam3.pt \
  --qwen-server-urls http://127.0.0.1:8008 \
  --max-frames 128
```

Health check:

```bash
curl -s http://127.0.0.1:8090/health | python -m json.tool
```

## 3. Start Reward 3D / DA3

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID python worldeval/batch_test/start_reward_3d_servers.py \
  --gpus 2,3 \
  --ports 8092,8093 \
  --model worldeval/weights/da3 \
  --qwen-server-urls http://127.0.0.1:8008 \
  --no-lpips
```

## 4. Run General Pipelines

Default general core pipelines:

```text
cosmos-predict
rolling-forcing
matrix-game2
```

Command:

```bash
python worldeval/batch_test/evaluate_pipelines.py \
  --domains general \
  --pipelines cosmos-predict rolling-forcing matrix-game2 \
  --gpu-slots 4,5 \
  --workers 2 \
  --qwen-server-urls http://127.0.0.1:8008 \
  --sam3-server-urls http://127.0.0.1:8090 \
  --reward-3d-server-urls http://127.0.0.1:8092,http://127.0.0.1:8093 \
  --print-skipped
```

Run only one pipeline:

```bash
python worldeval/batch_test/evaluate_pipelines.py \
  --domains general \
  --pipelines cosmos-predict \
  --gpu-slots 4,5 \
  --workers 2 \
  --qwen-server-urls http://127.0.0.1:8008
```

The evaluator skips existing `*_judge_*.json` files by default. Add `--force`
only when the existing scores should be recomputed.

## 5. Summaries

Each pipeline writes:

```text
batch_logs/general_cosmos/score_summary.json
batch_logs/general_cosmos/score_cases.csv
batch_logs/general_rolling_forcing/score_summary.json
batch_logs/general_rolling_forcing/score_cases.csv
batch_logs/general_matrix_game2/score_summary.json
batch_logs/general_matrix_game2/score_cases.csv
```

The evaluator refreshes `summary_from_outputs.jsonl` from all manifest items, so
previously completed results are included in the aggregate summary.
