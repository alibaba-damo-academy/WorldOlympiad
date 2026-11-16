# WorldOlympiad2

机器人操作任务的世界模型评测基准：把任务描述变成 SceneCode 场景与 PyBullet 专家参考，让各个模型**在各自的原生仿真里**闭环执行，再用 Physical / Geometry / Interaction 三轨评分，并自动汇总成结果表。

> 2026-10-04 起按新设计重写，流水线在 `wo/` 包里。更早的 7 任务 × 2 模型批次、`olympiad_forge`、`robotics_agentic_harness` 等旧体系已在 2026-10-06 清理，需要时从 git 标签 `pre-prune-20261006` 恢复。

## 流水线

| 阶段 | 入口 | 做什么 |
|---|---|---|
| scene | `wo.tasks.scene` | `wo_task_spec.v1`（`wo/spec.py`）→ SceneCode 生成场景（LLM 约 30–70 分钟） |
| reference | `wo.expert.run` + `wo.render` | PyBullet 专家完成任务，Blender 回放出参考视频 |
| episode | `wo.models.cosmos` / `lingbot_va` / `gr00t` | 各模型在原生仿真中执行：Cosmos3 → Isaac/RoboLab，LingBot-VA → RoboTwin，GR00T N1.6 → Isaac |
| score | `wo.eval.case` + `scripts.score_robotics_cases` | **Physical** = 通用物理判官 + 任务探针；**Geometry** = 3D 重建；**Interaction** = 语义交互 + CAE v2 |
| report | `wo.eval.report` | 每个子项分数、按模型/按任务族汇总，输出 `results.json` / `results.md` |

产物布局：`outputs/wo_runs/<run>/{tasks,references,<case>/<model>_s<seed>,manifest.json,results.*}`。`manifest.json` 声明这次 run 要审计的 `models` 和 `seeds`，报告按它统计覆盖率，不再靠磁盘上残留的旧产物推断。

FastWAM 的环境已就绪，适配器尚未实现；N1.7 需要 HF token。

## 目录

| 路径 | 内容 |
|---|---|
| `wo/` | 流水线本体：`spec` `scene` `sim` `expert` `render*` `models` `eval` `tasks` `agent`，测试在 `wo/tests` |
| `wo/agent/` | SQLite 作业队列（`jobs.py`）+ 已注册阶段（`stages.py`）+ LLM 调度 agent（`run.py` `tools.py`），目标写在 `goal_v1.md`，领域知识在 `knowledge/` |
| `autoresearch/` | SceneCode 场景生成（`robotics_scene_prompt_compiler` `robotics_scenecode_case_generator` `generate_scene`）和评分支撑（`score_hierarchy` `full_track_report` `robotics_resource_lock`） |
| `scripts/` | 评分入口（`score_*`）、WorldIR → Isaac 的构建与闭环脚本（`build_task_graph_worldir` `run_cosmos3_*`）及相关工具 |
| `physical/` `problem/` `mechanics/` `thermotics/` `material/` | Physical 判官及其题库 |
| `3d_metrics/` | Geometry 评分（`reward_3d`） |
| `interaction/` `continuous_action/` | 语义交互与 CLIP；CAE v1 的计划/时间工具 |
| `model/` | VLM 客户端 |
| `deployment/` | SceneCode 运行时的补丁与暂存（`ziplab5090/stage_scenecode_runtime.sh` 等）、环境供给脚本、`upstream_lock.json` |
| `docs/` | 背景文档，见下 |

**不能随意改名或删除：** `autoresearch/scenecode_*.py`、`pybullet_frame_contract.py`、`robotwin_camera_alignment.py`、`scripts/robotwin_articulated_task.py`，它们被仓库**之外**的 SceneCode 运行时、pybullet 桥和 RoboTwin 环境直接 import，仓库内部看不到这些调用方。

## 保留但尚未接入新流水线

**LingBot-World2 物理赛道**（决定放到最后做）：`autoresearch/{lingbot_world2_adapter,lingbot_world2_harness,lingbot_world_physics,physical_reference,physical_apparatus,agentic_physics*}.py`、`scripts/{run_lingbot_world_physics*,prepare_lingbot_world_physics_plan,blender_physics_native_reference,build_physics_room_from_robotics_scene,score_physics_object_cae}.py`、`continuous_action/physics_*.py`、`batch_test/`、`docs/LINGBOT_WORLD_PHYSICS.md`。这是旧体系的实现，没有测试保护，其中 `batch_test/make_manifest.py` 有一处早已失效的 import；接入时按 `wo/agent` 的方式重做比修补更稳。

## 文档

`docs/WORLDOLYMPIAD_EVALUATION_AND_BASE.md`（原作评测定义）和 `docs/worldir_multi_simulator_architecture.md`（WorldIR 架构）写于重设计之前，作背景参考；与本文冲突时以本文和代码为准。

## 运行与开发

服务器为准，工作区 `/root/autodl-tmp/worldolympiad2`，主仓库 `code/WorldOlympiad-main`（分支 `shared-agent-loop`）。

双接口运行配置从 `config/two-interface.example.json` 复制：Agent API 密钥只放在私有环境文件，模型 checkpoint 和运行目录用相对 `workspace_root` 的路径。`tested_model.scenecode_source` 可指向现有 SceneCode checkout；默认是 `code/SceneCode-main`。启动预检会验证该目录的 `main.py`，缺少的 managed runtime 可在 scene 阶段暂存，已有但没有 WorldOlympiad 标记的目录会被拒绝。

```bash
# 作业 worker 与调度 agent（需要 .secrets/scenecode.env，并设 OPENAI_AGENTS_DISABLE_TRACING=1）
envs/autoresearch/bin/python -m wo.agent.jobs worker
envs/scenecode/bin/python -m wo.agent.run --run bench_v1 --goal wo/agent/goal_v1.md

# 专家回归（改 wo/expert、wo/sim 之后必跑；原来成功的用例不能退步）
envs/autoresearch/bin/python -m wo.expert.regress TAG
envs/autoresearch/bin/python -m wo.expert.regress --compare base TAG

# 测试：两个目录要分开跑（同一进程里 PyBullet 全局状态会让
# tests/test_pybullet_frame_contract.py 失败，这是既有的隔离问题，与代码改动无关）
envs/autoresearch/bin/python -m pytest wo/tests -q
envs/autoresearch/bin/python -m pytest tests -q
```

- agent 在 `code/wo-agent-worktree`（分支 `agent/dev`）里改代码，测试通过后 `promote` 回 `shared-agent-loop`。手动修改前先看 `git status` 和 agent 是否在跑，**不要用 scp 整文件覆盖服务器上的 `wo/`**，曾因此把已注册的模型表回滚成只剩 Cosmos。
- 模型注册表在 `wo/agent/stages.py` 的 `MODELS`，由 `wo/tests/test_model_registry.py` 锁定。
- 根文件系统只有 30 GB，`TMPDIR` 必须指向数据盘。
