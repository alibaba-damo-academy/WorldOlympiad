# 通用候选发布验证清单

状态：T1–T3 已于 2026-11-14 通过；T4 于 2026-11-16 在 6000D 完成同机干净环境安装、恢复和运行时验收，限制见 [T4 记录](t4-clean-install-20261116.md)；T5 依用户要求暂缓。以下状态以实际退出码和产物为准。

基线发布提交：b97979495f0cfb76915d1b34c367eeed3c2bcf67。T1 评测实现提交：cd31b87；T2 评测实现提交：03c294a。T4 测试源码提交：7518d72d995db702f4f0e336b73a539a094e400e，Git tree：65e6b261721d8f6896c9c5327f5cc8e504a0edc1。后续仅文档变动不影响实现指纹；代码、依赖或资源契约变动必须记录新版本并重验受影响步骤。

2026-11-16 后续变更：新用户接口接入原生模型注册表，并将外部模型生成/评分与 LingBot 部署 profile 解耦；本地 26 项针对性测试通过，包括非 LingBot 命令 adapter 实际启动、视频解码和评分汇总。上述 T1–T4 运行证据属于变更前实现；新增的其他模型路径尚无其自身权重/运行环境的 GPU 闭环证据，不能据此宣称五模型全部实测通过。

范围是用户自带 real/sim 参考数据、模型 adapter、生成或提交预测视频、评分和报告。测试不要求师兄或论文的固定样本清单，不要求整批长视频。当前内置运行验证对象为 LingBot-VA；其他模型通过各自 adapter 接入。

## 已有证据，不重复计为待办

| 项目 | 当前证据 |
|---|---|
| 定向回归 | 19 项通过，脚本语法和差异检查通过 |
| real/i2av 分步流程 | 真实 LingBot 生成 5 帧、10 FPS、720×540 视频；正式提交评分入口三轨完成，退出码 0，report.complete=true |
| 已完成评分的恢复执行 | 重复命令退出 0，无新增评分阶段，报告和原始收据不变 |
| 输入保护 | 原始参考/预测视频哈希不变，评分旁路产物留在评测私有目录 |
| 干净源码包 | 新目录解压、无项目环境/私有配置时，7 个源码包清单验证通过；这不等于干净依赖安装通过 |
| 原生闭环基线 | 6000D 冻结基线上已有原生任务到 report-v3 的完整运行证据；新 sim_rollout 封装已由 T2 独立验证 |

## 顺序与通过条件

| 步骤 | 测试内容 | 输入与执行边界 | 通过条件 | 当前状态 |
|---|---|---|---|---|
| T1 | 最终 benchmark-run 一条命令 real/i2av | 1 条合规 real 样例、真实三路初始图像、LingBot，先用 1 chunk；全新 run ID 和模型进程，不导入旧预测/评分 | 生成、裁剪、完整解码、三项真实视频评测、视频协议报告一次完成；退出 0、complete=true；输入绑定一致；本轮 GPU 子进程及服务退出 | **通过**：2026-11-14，run `release_step2_full_20261114`，提交 `cd31b87`，GPU 3，overall 0.602 |
| T2 | 新 sim_rollout 封装 | 1 条已通过证书的 WOV2 原生 reference；新场景检查、新原生 episode；先用 3–5 秒诊断窗口，不改任务目标 | scene_ready=true；执行了实际模型/仿真步；任务与 reference 绑定正确；导出 exterior_left 正确视角；视频可解码；视频协议三轨报告完成；资源释放 | **通过**：2026-11-14，run `release_step3_sim_verified_20261114`，提交 `03c294a`，GPU 3，overall 0.1396 |
| T3 | 用户批量提交、恢复与拒绝行为 | 用 T1/T2 的真实输出组成 2 案例提交，分别保留真实 real/sim 来源；不重新生成模型视频 | 两例各自完成且无漏项/重复；real/sim 计数和均值准确；恢复不重跑已完成阶段；非法或变化输入被拒绝；不损坏既有结果 | **通过**：2026-11-14，提交 `a1d016e`；受控中断后续跑及全新双案例命令均完成 |
| T4 | 依赖锁审计和干净安装 | 从源码包建立独立目录及全新项目环境；优先在另一台合规 NVIDIA 服务器做。可复用权重、数据、资产和下载缓存，禁止链接旧 envs 或旧 runtime overlay | 依赖定义与 bootstrap 一致，未锁项/版本冲突有明确处置；按文档能创建环境和构建扩展；运行时、权重、资产及 CUDA/渲染预检通过；无需修改系统配置或共享环境 | **限定通过**：2026-11-16，6000D 同机全新环境；GPU 争用中断后用固定 bootstrap 续装，五套环境、CUDA 扩展、SAPIEN 相机和 Blender 渲染通过。单命令无中断安装及完整传递依赖锁未证明；见 [T4 记录](t4-clean-install-20261116.md) |
| T5 | 新环境按文档跑用户流程 | 仅使用 T4 新环境；跑 1 条 real/i2av 一键短流程及 1 条小规格原生任务到报告流程 | 两条流程都按文档完成；路径绑定落在新工作区；视频协议报告和原生 report-v3 各自满足其契约；原生适用的轨迹指标不缺失；文档遗漏已修正并重验 | 依用户要求暂缓，不计为通过 |

### T1 / T2 命令形态

REAL_MANIFEST 和 SIM_MANIFEST 指向用户样例的 manifest；下面是命令形态，不在制定清单时执行。每一步启动前重新配置并核验 WOV2_GPU_ALLOWLIST。

```bash
./setup.sh benchmark-check --manifest "$REAL_MANIFEST"
./setup.sh benchmark-run --manifest "$REAL_MANIFEST" --model lingbot_va --mode i2av --run release_t1_real_UNIQUE --chunks 1 --timeout 7200

./setup.sh benchmark-check --manifest "$SIM_MANIFEST"
./setup.sh benchmark-run --manifest "$SIM_MANIFEST" --model lingbot_va --mode sim_rollout --run release_t2_sim_UNIQUE --timeout 7200
```

T2 的短窗口写入该轮 manifest 的 execution.window_s。sim 是数据来源标签，不代表所有仿真视频都能直接执行 sim_rollout：本轮内置适配器使用其支持的 WOV2 reference 契约；其他仿真格式需要对应 adapter。

### T3 必须覆盖的边界

1. 两个实际案例的 benchmark-score 批量运行，校验 case_id、视角、分组计数和均值。
2. 在仅属于本轮测试的进程中做一次受控中断，再用原输入恢复。必须先核对 PID/PGID、用户和工作区，不能中断其他人的进程；已完成案例复用收据，未完成案例保留历史后继续，不能残留 GPU/服务进程。
3. 对完整批量结果重复执行，确认无新增模型生成/评分阶段，报告和原始收据不变。
4. 在独立的坏输入副本上验证至少：缺失文件或重复 ID、预测视角不匹配、修改输入后复用旧 run ID。预期为明确的非零退出，已有成功结果和源数据不变。不得编辑原用户输入来制造此测试。

### T4 / T5 环境和部署边界

- 优先直接在新目标服务器执行 T4/T5，同时获得干净安装和跨服务器证据。若暂时只能在 6000D 独立目录执行，则仅标记同机干净环境通过，跨服务器验证继续保持待办。
- 可以利用已有下载缓存减少网络下载，但必须创建全新环境，并从锁定源码重新建立本轮 runtime；不能用旧环境链接代替安装测试。权重及 checkpoint 继续只读复用，不重新下载或替换。
- 主机预检分别记录 CUDA、Blender 实际 EGL/Mesa 后端及 SAPIEN Vulkan 相机读回。NVIDIA EGL 缺失时按受支持的 Mesa 路径验证；SAPIEN 的 Vulkan 依赖仍须满足。驱动组件或权限问题记录为外部前置条件，不自动改系统。
- 检查所有项目运行环境的 import/版本、CUDA 原生扩展构建和实际设备架构。若依赖锁不完整或安装脚本仍有固定主机路径，先定位并修复，再继续；不能靠手工补包后仍宣称文档一次通过。
- 新原生闭环需要的 API 由用户在终端手动配置，密钥不进入源码、命令参数、配置 JSON 或日志。real/i2av 本地生成不需要调用上层 Agent API。
- 原生流程保持 RSI 工具启用，policy_check 的 30 分钟超时不变。专家 reference 必须通过其证书；待测模型的 task_success=false 保留为模型结果，不修改标签。
- 若尚未选择跨服务器目标，5090 可作为后续目标；不能将旧版本的 5090 运行记录算成当前版本回归通过。

## 记录和执行纪律

- 每步开始前记录提交、实现/源码/依赖锁哈希、manifest/输入哈希、配置路径和资源白名单；不得记录凭据值。资源只在用户允许的 0–6 索引内查询和选择，每阶段重新预检，同时最多一个评测作业。
- benchmark 产物仍按既有 outputs/benchmark/RUN 保存。原生任务产物按 outputs/wo_runs/RUN 保存。统一执行记录放在 outputs/benchmark/release_validation_SESSION/，其中 inputs/ 放本轮自带样例描述，step-status.json 放每步状态，日志/退出码/版本/报告路径按 T1–T5 命名。数据与运行产物不进入 Git。
- 通过证据至少包括退出码、完整视频解码、任务/视角/哈希绑定、所需评测证据、最终报告状态和资源清理。视频协议按预先声明排除原生轨迹 CAE，原始评分和收据保持不变；原生报告的适用指标仍完整验证。
- 失败保留日志、部分产物和真实状态。代码问题先定位、最小修复、定向验证，再使用新 run ID 重试受影响步骤；环境或资源阻断先解决原因，不盲目重复提交。
- 后续执行按 T1→T2→T3→T4→T5 推进，每步记录结果后再继续。不因增加一个文档计划而重新运行已通过的 19 项测试；出现代码/依赖变化再做相关回归。
- T1–T3 已执行并保存退出码、生成记录、评分收据和报告；T4–T5 继续按顺序验证。没有重新下载权重、修改密钥或部署新服务器。发布源码包在测试收口后更新。

## T1 验收证据

- 执行：`outputs/benchmark/release_validation_20261111/step-2-full-20261114-exit.json`，单条 `benchmark-run` 退出码 0，耗时约 337 秒。
- 结果：`outputs/benchmark/release_step2_full_20261114/report.json`，`complete=true`，1 条 real 案例 `overall=0.602`；原始三轨评分收据和独立的视频协议分数均存在。
- 输入：reference SHA256 `7ab6bca0c0657ae48dc80de28ab0ba76e3e1097c7ae450951daffb819b4c41de` 未变化；生成视频 19,829 字节，SHA256 与生成记录相符，视角 `head_camera` 一致。
- 设备：SAM3 与 DA3 均在物理 GPU 3；DA3 日志为 `CUDA_VISIBLE_DEVICES=3`。结束后 GPU 0/3 无本轮显存占用，Qwen 服务端口 8008 已释放。
- 修复：`35dc777` 约束 DA3 设备；`cd31b87` 修复 Ultralytics SAM3 将逻辑 `cuda:0` 误当物理 GPU 0 的问题，并把该源码纳入运行指纹。相关评分拓扑 11 项与 SAM3 设备 2 项检查通过。

## T2 验收证据

- 执行：`outputs/benchmark/release_validation_20261111/step-3-sim-verified-20261114-exit.json`，一条 `benchmark-run --mode sim_rollout` 命令退出码 0，耗时约 470 秒。
- 场景：`scene_ready=true`；从已通过证书的 WOV2 reference 重建苹果和托盘，LingBot 原生 SAPIEN rollout 执行 75 个动作，符合 `window_s=5`、15 Hz 的上限。模型的 `task_success=false` 如实保留，不作为评测功能失败。
- 输出：`exterior_left.mp4` 为 320×180、15 FPS、76 帧，完整解码，生成收据 SHA256 匹配；原始参考视频 SHA256 `fa951469dc15528385cf3538ffc07c00de223e3a8f7da0ab7692e6c03c06fb85` 未改变。
- 评分：`outputs/benchmark/release_step3_sim_verified_20261114/report.json` 为 `complete=true`，physical、geometry、interaction 三轨状态均为 `scored`，overall 约 0.1396；视频协议不声称原生任务成功或轨迹 CAE。
- 环境与修复：GPU 3 的本轮进程结束后释放；SAPIEN 复用与主机驱动匹配的进程内 Vulkan 图形库，Blender 使用 Mesa。`3976aea` 修复短窗口传递，`18a1511` 更新源码锁及运行时装配说明，`58cea6b` 和 `03c294a` 修复 bundled ffmpeg 传递。两个窗口聚焦测试、6 个渲染相关测试、7 个源码 bundle 校验和 5 类 Python 环境预检通过。首轮和修复前失败尝试均保留在 `outputs/benchmark/release_validation_20261111/`。

## T3 验收证据

- 输入：`outputs/benchmark/release_validation_20261111/inputs/mixed_batch.json`（SHA256 `c2aa4805b9edb0023417a977aea463d59ff43c217b3e9ae004188cf227cf24e9`）和 `mixed_predictions_accepted_20261114.json`（SHA256 `9bcce976a668b41dd466678049a95cf4793f714fc2516c22c968deae1407903f`）仅引用 T1/T2 的原始输出；real/sim 参考视频及预测视频的哈希验后未变化。
- 受控中断：run `release_step4_batch_resumable_20261114` 在 real 已完成、sim 的真实评分日志开始后，只向本轮进程组发送 SIGINT，退出码 `-2`；sim 保留为 pending，GPU 3 与本轮子进程退出。相同清单、提交和 run ID 续跑退出码 0，`complete=true`。real 的原始评分、收据和视频协议分数 SHA256 不变，评分阶段仍只有 1 个；sim 保留中断尝试并完成新尝试。
- 独立完整批量：`outputs/benchmark/release_validation_20261111/step-4-batch-clean-20261114-exit.json` 记录单条 `benchmark-score` 退出码 0；`outputs/benchmark/release_step4_batch_clean_20261114/report.json` 中两例均 complete，三条轨道均 scored。real 与 sim 各 1 例，组均值分别为 0.602 和 0.1398，等于该报告的单例分数；各例只有 1 个评分阶段。此命令只评分既有视频，没有再次调用待测模型。
- 恢复后的完整批次重复执行退出 0，报告 SHA256 与两例评分阶段数量不变。独立坏输入副本的缺失视频、重复 ID、错误视角、修改提交后复用已完成 run ID 均退出 1；成功报告和原始视频哈希保持不变。记录在 `step-4-batch-repeat-20261114.json` 与 `step-4-bad-inputs.json`。
- 首次双案例尝试在案例切换时遇到上一评分进程退出后的短暂 GPU 利用率残留，按严格资源预检失败；`a1d016e` 增加仅针对无计算进程的有界读数回落等待，下一案例仍重新执行正式预检。`tests/test_benchmark.py` 共 21 项通过。该失败尝试保留在 `outputs/benchmark/release_step4_batch_interrupt_20261114/`，不计为通过。完成后无本轮残留进程；后续 GPU 3 被另一用户作业占用，本轮未操作其进程。

## 收口标准

T1–T3 通过：当前 6000D 的新增用户数据入口、仿真封装、批量与恢复行为完成验收。T4–T5 通过：文档安装流程和新环境完整运行获得证据。若 T4/T5 在第二台服务器完成，可据此增加该主机配置的迁移支持记录；否则不扩大兼容性声明。以上均不等于长时间压力测试或模型质量 benchmark，也不要求任何官方固定样本集。
