<h1 align="center">WorldOlympiad: Can Your World Model Survive a Triathlon?</h1>

<p align="center">
  <a href="https://alibaba-damo-academy.github.io/WorldOlympiad"><img src="https://img.shields.io/badge/Project%20Page-000000?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2606.11129"><img src="https://img.shields.io/badge/Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://github.com/alibaba-damo-academy/WorldOlympiad"><img src="https://img.shields.io/badge/Code-24292F?style=for-the-badge&logo=github&logoColor=white" alt="Code"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/English-1677FF?style=for-the-badge" alt="English README"></a>
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
  <sup>*</sup>共同一作 &nbsp;&nbsp;
  <sup>&dagger;</sup>项目负责人 &nbsp;&nbsp;
  <sup>&Dagger;</sup>通讯作者
</p>

<p align="center">
  <b>WorldOlympiad 论文官方仓库。</b><br>
  面向视频世界模型的三项全能式评测：物理一致性、3D 几何一致性与交互一致性。
</p>

## 项目概览

WorldOlympiad 用于诊断视频世界模型，而不只评估视频是否视觉上好看。它关注长视频生成结果是否遵守可解释的物理规律、是否维持一致的 3D 结构，以及是否能够在连续 chunk 中持续遵循可控交互指令。Benchmark 覆盖 gaming、robotics 和 general real-world videos 三类下游场景，并用统一的自动评测协议比较多个代表性长视频生成 pipeline。

## 主要贡献

- **三项全能式评测。** WorldOlympiad 将世界模型能力拆成 physical、geometry 和 interaction 三个互补赛道。
- **多领域 benchmark。** 测试集包含 1,000 个长视频：400 个 robotics 视频、400 个 gaming 视频和 200 个 general real-world 视频。
- **可解释自动指标。** Physical track 使用 object-centric segmentation 和 MLLM judge；geometry track 使用 DA3 / Gaussian-splatting 诊断；interaction track 评估 chunk-level 和全局 prompt following。
- **面向 pipeline 的诊断。** 官方代码支持对多个长视频生成 pipeline 进行批量测评，并生成可断点续跑的 per-case judge JSON。

## Benchmark 设计

| 赛道 | 评测目标 | 主要信号 |
| --- | --- | --- |
| Physical | 生成视频中的行为是否符合物理规律。 | 使用 SAM3 辅助的 object evidence 和 MLLM scoring，评测 mechanics、thermodynamics 和 material properties。 |
| Geometry | 生成视频是否保持一致的 3D 结构。 | DA3 重建质量、diagnostic meta-view consistency 和 recovered camera-trajectory alignment。 |
| Interaction | 长时序 rollout 是否遵循交互 prompt。 | Chunk-level instruction following、相邻 chunk transition smoothness、full-video consistency 和 CLIP semantic grounding。 |

<p align="center">
  <img src="figure/data_overview.png" alt="WorldOlympiad data overview" width="95%">
</p>

标注流程会先定位视频中的主要连续执行区间，将视频切分为连续 chunk，然后生成 action/caption 元数据，并结合整段视频上下文进行 refine。最终这些 annotations 会作为 interaction evaluator 使用的 `prompt.json`。

## 结果与诊断

WorldOlympiad 的目标不只是生成一个 aggregate leaderboard，而是帮助定位不同类型的失败模式。论文中展示了 score distributions、human preference alignment、radar-style diagnostics 和 qualitative failure cases。

<p align="center">
  <img src="figure/result_statistics.png" alt="WorldOlympiad result statistics" width="95%">
</p>

这些统计结果展示了当前长视频生成 pipeline 在三个赛道上的整体表现。这里更关键的不是单一平均分，而是不同模型的失败位置：有些 pipeline 能保持较好的视觉观感，却会违背物理规则；有些 pipeline 能维持局部运动，却会在长时序中丢失几何一致性或交互一致性。下面的案例图用于把这些错误模式具体化，方便进一步分析模型短板。

<p align="center">
  <img src="figure/failure_case_study.png" alt="WorldOlympiad failure case study" width="95%">
</p>

## 仓库结构

```text
worldeval/
  batch_test/                 # batch manifest、调度器和服务启动脚本
  scripts/                    # 单视频评分和预处理辅助脚本
  physical/                   # 基于 SAM3 的物理预处理与 judge
  3d_metrics/                 # DA3 / 3D reward scoring
  interaction/                # VLM 和 CLIP interaction scoring
  model/                      # VLM backends 和本地 QwenVL server
  problem/                    # 物理规则题库
  figure/                     # 论文与 README 图片
  environment.yml             # 导出的 conda 环境，环境名为 worldolympiad
```

## 快速开始

创建导出的 conda 环境。该环境文件来自已验证可运行的 `world_eval` 环境，并已改名为 `worldolympiad`。

```bash
cd worldeval
conda env create -f environment.yml
conda activate worldolympiad
```

如果环境已经存在：

```bash
conda env update -n worldolympiad -f environment.yml --prune
```

### Depth Anything 3

WorldOlympiad 的 geometry track 会从 `worldeval/Depth-Anything-3/src` 导入 DA3 代码，因此**必须 clone DA3 仓库**：

```bash
cd worldeval
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

当前 `worldolympiad` 环境已经包含本项目使用的 DA3 runtime stack，包括 `depth-anything-3`、`gsplat`、`torch`、`torchvision` 和 `xformers`。正常情况下，只需要准备上面的 DA3 源码目录和下面的 DA3 权重，不需要重新创建或安装 DA3 仓库里的完整依赖环境。

```bash
hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 --local-dir ./weights/da3
```

只有当 `worldolympiad` 环境里仍然无法 import DA3 时，才执行下面的 fallback：

```bash
cd worldeval/Depth-Anything-3
pip install -e .
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

### SAM3 与 QwenVL

下载 SAM3 权重：

```bash
cd worldeval
pip install modelscope
modelscope download --model facebook/sam3 --local_dir ./weights/sam3
```

将本地 QwenVL checkpoint 放到 `worldeval/weights/QwenVL`，或者在启动 `start_qwenvl_servers.py` 时通过 `--model` 传入其他路径。

如果使用 OpenRouter 或其他 OpenAI-compatible endpoint 进行 VLM scoring，需要创建 `worldeval/.env`：

```bash
base_url=https://openrouter.ai/api/v1
api_key=YOUR_API_KEY
```

## 测评数据如何排布

批量测评要求每个 case 一个目录：

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

`general`、`gaming` 和 `embodied` 是默认 domain 名称。如果需要评测自定义目录，可以使用 `--root <case_root> --domain-name <name>`。

### 必需文件

- `prompt.json`：整段视频或每个生成 chunk 的 prompt 元数据。
- `ref_<case_id>.mp4`：参考视频。默认匹配规则是 `ref_*.mp4`。
- `<output_prefix>_gen_<case_id>.mp4`：某个 pipeline 生成的视频。
- `<output_prefix>_gen_<case_id>_chunk_timestamps.json`：生成视频的 chunk 时间戳元数据。

评测器会优先查找 `<generated-video-stem>_chunk_timestamps.json`。评分结果写在同一个 case 目录下：

```text
<output_prefix>_judge_<case_id>.json
```

如果评分 JSON 已存在，批量测评默认会跳过该 case，因此中断后可以直接续跑。

### Pipeline 和文件名前缀

| Pipeline | Output Prefix | 生成视频文件示例 |
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

同一个 case 目录可以同时放多个 pipeline 的生成结果。通过 `--pipelines` 选择要评测的 pipeline，每个 pipeline 会生成独立的 judge JSON。

### `prompt.json`

`prompt.json` 可以是 JSON list，也可以是带有 `chunks` 或 `prompts` 字段的 JSON object。每个元素可以是字符串，也可以是 object：

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

文本字段可以使用 `caption`、`prompt`、`text` 或 `description`。`action`、`interval` 和 `chunk_index` 不是强制字段，但建议保留，因为它们会帮助 interaction scoring。

### Chunk Timestamp 文件

timestamp 文件用于说明每个 prompt chunk 对应生成视频中的哪一段：

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

如果文件命名不符合默认 layout，可以通过 `--gen-pattern`、`--chunk-pattern`、`--ref-pattern` 和 `--output-name-template` 显式指定。

## 批量测评 Pipeline

大规模批量测评建议先在不同终端启动持久化服务。

QwenVL：

```bash
python worldeval/batch_test/start_qwenvl_servers.py \
  --gpus 0,1 \
  --ports 8008,8009 \
  --model worldeval/weights/QwenVL \
  --warmup
```

SAM3：

```bash
python worldeval/batch_test/start_sam3_servers.py \
  --gpus 2 \
  --ports 8090 \
  --model worldeval/weights/sam3/sam3.pt \
  --qwen-server-urls http://127.0.0.1:8008,http://127.0.0.1:8009
```

DA3 / 3D reward：

```bash
python worldeval/batch_test/start_reward_3d_servers.py \
  --gpus 3,4 \
  --ports 8092,8093 \
  --model worldeval/weights/da3 \
  --qwen-server-urls http://127.0.0.1:8008,http://127.0.0.1:8009 \
  --no-lpips
```

8 卡机器推荐分配：

```text
GPU 0-1: QwenVL 服务
GPU 2: SAM3 服务
GPU 3-4: DA3 reward 服务
GPU 5-7: scoring worker
```

从 OpenWorldLib 项目根目录运行统一批量测评脚本：

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

该脚本会在 `batch_manifests/` 下创建 manifest；默认跳过已经完成的 judge JSON，除非传入 `--force`；随后调用受控 worker 调度器进行评分，并在 `batch_logs/` 下写入日志和汇总结果。

常用参数：

- `--list-pipelines`：查看支持的 pipeline aliases 和 output prefixes。
- `--root <case_root> --domain-name <name>`：评测一个自定义 case 根目录。
- `--limit N`：每个 domain/pipeline 最多测 `N` 个待测 case。
- `--force`：强制重算已有评分 JSON。
- `--dry-run`：只打印命令，不真正运行评分。
- `--no-summarize`：不生成汇总 CSV/JSON。
- `--skip-pair gaming:lingbot-world`：跳过某个 domain/pipeline 组合。

## 单视频评分

如果只想调试一个 case：

```bash
python worldeval/scripts/score_video_physical_3d.py \
  --video outputs_batch/general/case1/cosmos_gen_case1.mp4 \
  --gt-video outputs_batch/general/case1/ref_case1.mp4 \
  --prompt-json outputs_batch/general/case1/prompt.json \
  --chunk-json outputs_batch/general/case1/cosmos_gen_case1_chunk_timestamps.json \
  --output outputs_batch/general/case1/cosmos_judge_case1.json
```

## 引用

如果本仓库对你的研究有帮助，请引用：

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

## 致谢

WorldOlympiad 构建在 OpenWorldLib 以及多个开源模型和指标生态之上，包括 Depth Anything 3、SAM3、QwenVL-compatible VLM backends、CLIP semantic scoring 和 Gaussian-splatting reconstruction tools。
