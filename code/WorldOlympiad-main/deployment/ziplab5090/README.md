# deployment/ziplab5090

SceneCode 运行时的暂存与补丁，以及环境供给脚本。

**流水线在用**（`wo.tasks.scene` → `run_scenecode_case.sh`）
- `run_scenecode_case.sh`：跑一个场景
- `stage_scenecode_runtime.sh` `bootstrap_scenecode.sh`：把仓库里的补丁应用到 `runtime/scenecode-worldolympiad`
- `patches/scenecode_*.patch`：SceneCode 的补丁序列（例如图像生成走 Responses 接口的退避重试）

双接口评测从 profile 的 `tested_model.gpu` 选择 SceneCode 渲染卡；如需单独指定，可设置 `SCENECODE_RENDER_GPU`。GPU 隔离仍由 bubblewrap 执行，launcher 只将本轮 `workspace/tmp` 和 `outputs/robotics_scenecode_generations` 加入可写路径。

运行时补丁还用 `forkserver` 启动 SceneCode 的隔离 worker，并只在真正的 SceneCode 入口进程导入 `bpy`，避免 worker 重载入口时丢失 Blender 二进制模块。

## Evaluator weights on a user workspace

Set the workspace explicitly before running the evaluator-weight downloader. By default it stores weights under `workspace/weights` and links them into the code checkout at `weights/`, which is where the evaluator expects them. Set `WORLDEVAL_MODEL_CACHE_ROOT` when several code branches should share one weights directory; existing complete weights are reused, while incomplete destinations are preserved and reported instead of being overwritten.

The script needs `bash`, `python3`, and `curl`; missing HF weights are downloaded
with an installed `hf` CLI or `uvx hf`. Follow the [official HF CLI setup
guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli) if needed.
SAM3 login is done in that user's shell (for example, `hf auth login` or
`uvx hf auth login`). No GPU is used for downloading or validating weights.

```bash
WORLDOLYMPIAD_WORKSPACE=/path/to/worldolympiad2 \
WORLD_REPO=/path/to/worldolympiad2/code/WorldOlympiad-main \
WORLDEVAL_MODEL_CACHE_ROOT=/path/to/worldolympiad2/weights \
  bash deployment/ziplab5090/download_worldolympiad_weights.sh
```

The script downloads the complete `Qwen/Qwen3-VL-8B-Instruct` and
`depth-anything/DA3NESTED-GIANT-LARGE-1.1` repositories, `sam3.pt` from
`facebook/sam3`, and OpenAI CLIP ViT-B/32. It checks Qwen's required processor
files and every shard referenced by its safetensors index, checks DA3 and SAM3
files, and verifies CLIP's pinned SHA-256. It supports `sha256sum` on Linux and
`shasum` on macOS. Downloaded HF snapshots stay in `.partial` directories until
validation succeeds; rerunning the command can continue an interrupted HF
download. Existing incomplete final directories are never replaced.

SAM3 is gated on Hugging Face. If its weights are missing, the script checks
the current `hf` identity and tests access using the small model card before
starting the multi-gigabyte checkpoint. The user must log in and accept/request
access on the [SAM3 model page](https://huggingface.co/facebook/sam3). An access
request can require approval. `hf auth whoami` only confirms the logged-in
identity; it does not by itself prove SAM3 access. DA3 is CC BY-NC 4.0 and is
limited to non-commercial use.

Use `--check` to validate files and checkout links without downloading or
changing them:

```bash
WORLDOLYMPIAD_WORKSPACE=/path/to/worldolympiad2 \
WORLD_REPO=/path/to/worldolympiad2/code/WorldOlympiad-main \
WORLDEVAL_MODEL_CACHE_ROOT=/path/to/worldolympiad2/weights \
  bash deployment/ziplab5090/download_worldolympiad_weights.sh --check
```

If the current machine cannot reach Hugging Face, the script stops and offers a
local staging option; it does not switch machines or start a transfer on the
user's behalf. The user may choose to run the same script on a machine with HF
access, then sync the resulting `weights/` directory:

```bash
STAGE="$HOME/Downloads/WOV2-evaluator-weights"
mkdir -p "$STAGE"
WORLDOLYMPIAD_WORKSPACE="$STAGE" \
WORLD_REPO="/path/to/WorldOlympiad-main" \
  bash /path/to/WorldOlympiad-main/deployment/ziplab5090/download_worldolympiad_weights.sh

rsync -aP -e 'ssh -i ~/.ssh/<your_key> -o IdentitiesOnly=yes -p <port>' \
  "$STAGE/weights/" \
  <user>@<server>:/path/to/worldolympiad2/weights/
```

After transfer, run the downloader on the target workspace with
`WORLDEVAL_MODEL_CACHE_ROOT=/path/to/worldolympiad2/weights` once to establish the
checkout links, then run it with `--check`. HF credentials are kept in the
user's HF CLI login; never put an HF token in `two-interface.json` or chat.

**环境供给，仅作记录**
`bootstrap_autoresearch.sh` `bootstrap_robotwin_runtime.sh` `bootstrap_worldolympiad_evaluation.sh`
`configure_python_paths.sh` `download_*` `install_worldolympiad_gsplat.sh` `prepare_*`
`upgrade_worldolympiad_blackwell_runtime.sh` `stage_scenecode_pybullet_bridge.sh`。
这些写于旧服务器，路径与版本不保证与现状一致。

**保留，尚未接入**：`run_worldolympiad_{cae,evaluation,services}.sh`、`preflight_worldolympiad_evaluation.sh`
（LingBot-World2 物理赛道使用，见仓库根 `README.md`）。

旧的 `run_olympiad_forge.sh`、`run_autoresearch_*.sh`、`run_robotics_task_author.sh` 等入口已于 2026-10-06 清理。
