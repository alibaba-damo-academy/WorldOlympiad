<h1 align="center">WorldOlympiad: Can Your World Model Survive a Triathlon?</h1>

<p align="center">
  <a href="https://alibaba-damo-academy.github.io/WorldOlympiad"><img src="https://img.shields.io/badge/Project%20Page-000000?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2606.11129"><img src="https://img.shields.io/badge/Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://github.com/alibaba-damo-academy/WorldOlympiad"><img src="https://img.shields.io/badge/Code-24292F?style=for-the-badge&logo=github&logoColor=white" alt="Code"></a>
  <a href="https://huggingface.co/datasets/ziplab/WorldOlympiad"><img src="https://img.shields.io/badge/Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Dataset"></a>
  <a href="README_ZH.md"><img src="https://img.shields.io/badge/中文文档-1677FF?style=for-the-badge" alt="Chinese README"></a>
</p>

<p align="center">
  Yuke Zhao<sup>1,*</sup> &nbsp;
  Wangbo Zhao<sup>3,*</sup> &nbsp;
  Weijie Wang<sup>1,*</sup> &nbsp;
  Zeyu Zhang<sup>2,*,&dagger;</sup> &nbsp;
  Dakai An<sup>3</sup> &nbsp;
  Akide Liu<sup>4</sup> &nbsp;
  Yinghao Yu<sup>5</sup> &nbsp;
  Jiasheng Tang<sup>2,&Dagger;</sup> &nbsp;
  Fan Wang<sup>2</sup> &nbsp;
  Wei Wang<sup>3</sup> &nbsp;
  Bohan Zhuang<sup>1,&Dagger;</sup>
</p>

<p align="center">
  <sup>1</sup>Zhejiang University &nbsp;&nbsp;
  <sup>2</sup>DAMO Academy, Alibaba Group &nbsp;&nbsp;
  <sup>3</sup>The Hong Kong University of Science and Technology &nbsp;&nbsp;
  <sup>4</sup>Monash University &nbsp;&nbsp;
  <sup>5</sup>TRE, Alibaba Group
</p>

<p align="center">
  <sup>*</sup>Equal contribution &nbsp;&nbsp;
  <sup>&dagger;</sup>Project lead &nbsp;&nbsp;
  <sup>&Dagger;</sup>Corresponding authors
</p>

<p align="center">
  <b>Official repository for the WorldOlympiad paper.</b><br>
  A triathlon-style benchmark for physical faithfulness, geometric consistency, and interaction fidelity in video world models.
</p>

## Overview

WorldOlympiad diagnoses video-based world models beyond visual quality. It asks
whether generated long videos obey interpretable physical rules, preserve
coherent 3D structure, and follow controllable interactions across consecutive
chunks. The benchmark covers three downstream scenarios - gaming, robotics, and
general real-world videos - and evaluates representative long-video generation
pipelines through a unified automatic protocol.

## Contributions

- **Triathlon evaluation.** WorldOlympiad decomposes world-model evaluation into physical, geometric, and interactive tracks.
- **Multi-domain benchmark.** The benchmark contains 1,000 long videos: 400 robotics videos, 400 gaming videos, and 200 general real-world videos.
- **Interpretable automatic metrics.** The physical track uses object-centric segmentation and MLLM judges; the geometry track uses DA3/Gaussian-splatting diagnostics; the interaction track evaluates chunk-level and global prompt following.
- **Pipeline-scale diagnosis.** The official code evaluates multiple long-video generation pipelines and writes resumable per-case judge JSON files.

## Benchmark Design

| Track | What It Tests | Main Signals |
| --- | --- | --- |
| Physical | Whether generated behavior follows physical rules. | Mechanics, thermodynamics, and material-property compliance judged with SAM3-assisted object evidence and MLLM scoring. |
| Geometry | Whether the generated video preserves coherent 3D structure. | DA3 reconstruction quality, diagnostic meta-view consistency, and recovered camera-trajectory alignment. |
| Interaction | Whether the rollout follows long-horizon prompts. | Chunk-level instruction following, adjacent transition smoothness, full-video consistency, and CLIP semantic grounding. |

<p align="center">
  <img src="figure/data_overview.png" alt="WorldOlympiad data overview" width="95%">
</p>

The annotation pipeline identifies the main continuous execution interval,
splits each video into contiguous chunks, generates action/caption metadata,
and refines annotations with the full video context. These annotations are the
`prompt.json` files used by the interaction evaluator.

## Results And Diagnostics

WorldOlympiad is designed to expose failure modes rather than only produce one
aggregate leaderboard. The official paper reports score distributions, human
preference alignment, radar-style diagnostics, and qualitative failure cases.

<p align="center">
  <img src="figure/result_statistics.png" alt="WorldOlympiad result statistics" width="95%">
</p>

The score statistics summarize how current pipelines behave across the three
tracks. The important pattern is not only which model has the highest average
score, but where each model fails: some pipelines can maintain plausible visual
appearance while breaking physical rules, while others preserve local motion but
lose geometry or interaction consistency over longer rollouts. The qualitative
cases below are included to make these error modes concrete and inspectable.

<p align="center">
  <img src="figure/failure_case_study.png" alt="WorldOlympiad failure case study" width="95%">
</p>

## Repository Layout

```text
worldeval/
  batch_test/                 # batch manifests, scheduler, and service launchers
  scripts/                    # single-video scoring and preprocessing helpers
  physical/                   # SAM3-based physical preprocessing and judges
  3d_metrics/                 # DA3 / 3D reward scoring
  interaction/                # VLM and CLIP interaction scoring
  model/                      # VLM backends and local QwenVL server
  problem/                    # physical-rule question sets
  figure/                     # paper and README figures
  environment.yml             # exported conda environment, named worldolympiad
```

## Quick Start

Create the exported conda environment. The file was exported from the working
`world_eval` environment and renamed to `worldolympiad`.

```bash
cd worldeval
conda env create -f environment.yml
conda activate worldolympiad
```

If the environment already exists:

```bash
conda env update -n worldolympiad -f environment.yml --prune
```

### Depth Anything 3

WorldOlympiad's geometry track imports DA3 code from
`worldeval/Depth-Anything-3/src`, so the **DA3 repository must be cloned**:

```bash
cd worldeval
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

The `worldolympiad` environment already contains the DA3 runtime stack used by
this project, including `depth-anything-3`, `gsplat`, `torch`, `torchvision`,
and `xformers`. In the normal setup, you only need the DA3 source tree above
and the DA3 weights below. You do **not** need to recreate or install the full
environment from the DA3 repository.

```bash
hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 --local-dir ./weights/da3
```

Only use the following fallback if DA3 import fails inside `worldolympiad`:

```bash
cd worldeval/Depth-Anything-3
pip install -e .
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

### SAM3 And QwenVL

Download SAM3 weights:

```bash
cd worldeval
pip install modelscope
modelscope download --model facebook/sam3 --local_dir ./weights/sam3
```

Place your local QwenVL checkpoint under `worldeval/weights/QwenVL`, or pass a
different path to `start_qwenvl_servers.py --model`.

If you use OpenRouter or another OpenAI-compatible endpoint for VLM scoring,
create `worldeval/.env`:

```bash
base_url=https://openrouter.ai/api/v1
api_key=YOUR_API_KEY
```

## Evaluation Data Layout

Batch evaluation expects one directory per test case:

```text
outputs_batch/
  general/
    <case_id>/
      prompt.json
      ref_<case_id>.mp4
      <output_prefix>_gen_<case_id>.mp4
      <output_prefix>_gen_<case_id>_chunk_timestamps.json
  gaming/
    <case_id>/
      prompt.json
      ref_<case_id>.mp4
      <output_prefix>_gen_<case_id>.mp4
      <output_prefix>_gen_<case_id>_chunk_timestamps.json
  embodied/
    <case_id>/
      prompt.json
      ref_<case_id>.mp4
      <output_prefix>_gen_<case_id>.mp4
      <output_prefix>_gen_<case_id>_chunk_timestamps.json
```

`general`, `gaming`, and `embodied` are the default domain names. For a custom
case directory, use `--root <case_root> --domain-name <name>`.

### Required Files

- `prompt.json`: prompt metadata for the whole video or for each generated chunk.
- `ref_<case_id>.mp4`: reference video. The default matcher is `ref_*.mp4`.
- `<output_prefix>_gen_<case_id>.mp4`: generated video from one pipeline.
- `<output_prefix>_gen_<case_id>_chunk_timestamps.json`: generated-video chunk timing metadata.

The evaluator first looks for
`<generated-video-stem>_chunk_timestamps.json`. Score outputs are written next
to the videos:

```text
<output_prefix>_judge_<case_id>.json
```

Existing score files are skipped by default, so interrupted batches can resume
without recomputing completed cases.

### Pipeline Prefixes

| Pipeline | Output Prefix | Generated Video Example |
| --- | --- | --- |
| `cosmos-predict` | `cosmos` | `cosmos_gen_<case_id>.mp4` |
| `hunyuan-gamecraft` | `hunyuan_gamecraft` | `hunyuan_gamecraft_gen_<case_id>.mp4` |
| `hunyuan-worldplay` | `hunyuan_worldplay` | `hunyuan_worldplay_gen_<case_id>.mp4` |
| `lingbot-world` | `lingbot_world` | `lingbot_world_gen_<case_id>.mp4` |
| `longlive` | `longlive` | `longlive_gen_<case_id>.mp4` |
| `matrix-game2` | `matrix_game2` | `matrix_game2_gen_<case_id>.mp4` |
| `rolling-forcing` | `rolling_forcing` | `rolling_forcing_gen_<case_id>.mp4` |
| `wow` | `wow` | `wow_gen_<case_id>.mp4` |
| `yume1p5` | `yume1p5` | `yume1p5_gen_<case_id>.mp4` |

One case directory may contain outputs from multiple pipelines. Select the
pipelines to evaluate with `--pipelines`; each pipeline receives an independent
judge JSON.

### `prompt.json`

`prompt.json` may be a JSON list, or an object with a `chunks` or `prompts`
list. Each item may be a string or an object:

```json
[
  {
    "interval": "[00:00, 00:15)",
    "action": "turn left",
    "caption": "A vehicle drives through a cone-marked course."
  },
  {
    "interval": "[00:15, 00:30)",
    "action": "turn right",
    "caption": "The vehicle returns through the same course."
  }
]
```

The evaluator accepts `caption`, `prompt`, `text`, or `description` as the text
field. `action`, `interval`, and `chunk_index` are optional but recommended for
interaction scoring.

### Chunk Timestamp File

The timestamp file maps each prompt chunk to a generated-video time range:

```json
{
  "version": 1,
  "video_path": "outputs_batch/general/case1/cosmos_gen_case1.mp4",
  "fps": 28,
  "total_frames": 186,
  "duration_sec": 6.642857,
  "chunks": [
    {
      "chunk_index": 0,
      "source_interval": "[00:00, 00:15)",
      "frame_start": 0,
      "frame_end": 93,
      "generated_start_sec": 0.0,
      "generated_end_sec": 3.321428
    }
  ]
}
```

If file names differ from the default layout, pass explicit patterns with
`--gen-pattern`, `--chunk-pattern`, `--ref-pattern`, and
`--output-name-template`.

## Evaluate Pipelines

For large batches, start persistent services in separate terminals.

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

Run the unified batch script from the OpenWorldLib project root:

```bash
python worldeval/batch_test/evaluate_pipelines.py \
  --domains general gaming embodied \
  --pipelines cosmos-predict longlive matrix-game2 rolling-forcing wow \
  --gpu-slots 5,6,7 \
  --workers 3 \
  --qwen-server-urls http://127.0.0.1:8008,http://127.0.0.1:8009 \
  --sam3-server-urls http://127.0.0.1:8090 \
  --reward-3d-server-urls http://127.0.0.1:8092,http://127.0.0.1:8093 \
  --run-clip-interaction \
  --print-skipped
```

The script creates manifests under `batch_manifests/`, skips completed judge
JSON files unless `--force` is set, runs a controlled worker scheduler, and
writes logs plus score summaries under `batch_logs/`.

Useful options:

- `--list-pipelines`: show supported pipeline aliases and output prefixes.
- `--root <case_root> --domain-name <name>`: evaluate one custom case root.
- `--limit N`: evaluate at most `N` pending cases per domain/pipeline.
- `--force`: recompute existing score JSON files.
- `--dry-run`: print commands without running scoring.
- `--no-summarize`: skip aggregate score CSV/JSON generation.
- `--skip-pair gaming:lingbot-world`: omit a specific domain/pipeline pair.

## Single-Video Scoring

For debugging one case directly:

```bash
python worldeval/scripts/score_video_physical_3d.py \
  --video outputs_batch/general/case1/cosmos_gen_case1.mp4 \
  --gt-video outputs_batch/general/case1/ref_case1.mp4 \
  --prompt-json outputs_batch/general/case1/prompt.json \
  --chunk-json outputs_batch/general/case1/cosmos_gen_case1_chunk_timestamps.json \
  --output outputs_batch/general/case1/cosmos_judge_case1.json
```

## Citation

If you find this repository useful, please cite:

```bibtex
@misc{zhao2026worldolympiad,
  title         = {WorldOlympiad: Can Your World Model Survive a Triathlon?},
  author        = {Zhao, Yuke and Zhao, Wangbo and Wang, Weijie and Zhang, Zeyu and An, Dakai and Liu, Akide and Yu, Yinghao and Tang, Jiasheng and Wang, Fan and Wang, Wei and Zhuang, Bohan},
  year          = {2026},
  eprint        = {2606.11129},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2606.11129}
}
```

## Acknowledgements

WorldOlympiad builds on OpenWorldLib and several open-source model and metric
ecosystems, including Depth Anything 3, SAM3, QwenVL-compatible VLM backends,
CLIP-based semantic scoring, and Gaussian-splatting reconstruction tools.
