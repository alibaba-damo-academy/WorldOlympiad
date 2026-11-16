# Physical task → SceneCode → PyBullet reference → CAE 前处理

本链路按2026-09-08的实施要求统一使用 PyBullet。沿用现有 SceneCode 资产编译和 Blender 回放，不增加调研阶段提出的专项质量门槛。每个任务记录真实 PyBullet 轨迹，生成独立参考视频、标定深度和 Dream2Flow 物体运动文件。

当前展示版位于 `outputs/physical_reference_pybullet_20260908_v2_room`，保留原SceneCode的墙面、地板、边柜、挂画和灯具等静态背景。背景以固定物体进入PyBullet与Blender两侧，深度和目标框随场景重新生成；没有给视频贴一张背景图。

扩展事件原型位于 `outputs/physical_showcase_20260908_v3`：斜坡滚动、多米诺连锁、三球碰撞传递和落入浅盘。新一轮SceneCode在线生成仍返回API日限额耗尽，因此原型复用完整SceneCode房间，并用 `physical_apparatus.py` 按明确尺寸生成实验器具。`apparatus.json`、WorldIR和汇总均记录程序器具来源；它们不是官方SceneCode新生成资产。斜坡原型增加末端挡块以保留完整事件，碰撞链的主动球采用红色，属于公开候选的具体实验变体。

```bash
python3 scripts/runtime.py physical-reference -- \
  --output /root/autodl-tmp/worldolympiad2/outputs/physical_showcase_20260908_v3 \
  --stage reference \
  --source-world /root/autodl-tmp/worldolympiad2/outputs/scenecode_pybullet_cae_robotics_formal_v9_20260822/pilot/world_ir.json \
  --bindings /root/autodl-tmp/worldolympiad2/outputs/physical_reference_pybullet_20260908_v2_room/reuse_bindings.json \
  --tasks ramp_roll domino_chain collision_chain shallow_containment
```

多米诺各块、碰撞链各球均独立参与PyBullet动力学，全体状态和接触记录保存在轨迹中。当前CAE前处理分别追踪首块红色多米诺、主动红球、斜坡青球和浅盘紫球，尚不宣称完成所有对象的CAE评分。

## 典型任务prompt与SceneCode适配

候选见 `autoresearch/configs/physical_prompt_catalog_v1.json` 和 [可读prompt清单](PHYSICAL_PROMPT_CATALOG.md)。依据SceneCode自带示例及场景prompt生成规范，每条只用四句静态场景描述：房间和背景、完整物体、初始关系、光照。动态指令、释放高度、力、速度、接触系数留在独立execution记录，物体最终状态不写进SceneCode初始场景。

服务额度恢复后，可单独选择候选生成场景，例如：

```bash
python3 scripts/runtime.py physical-reference -- \
  --output /root/autodl-tmp/worldolympiad2/outputs/physical_ramp_candidate \
  --stage generate --prompt-id ramp_roll
```

这十个候选均是带来源的原创适配，尚未逐一生成。`--prompt-id`只承诺生成分支；斜坡、多米诺等新操作的执行绑定仍按清单标注，不复用不相干的六例执行器冒充完成。

## 服务器入口

在 `/root/autodl-tmp/worldolympiad2/code/WorldOlympiad-main` 执行：

```bash
python3 scripts/runtime.py physical-reference -- \
  --output /root/autodl-tmp/worldolympiad2/outputs/physical_reference_new_run \
  --stage all
```

`all` 依次调用官方 SceneCode、新场景对象绑定、PyBullet、Blender、CAE reference 前处理。SceneCode API 环境沿用服务器 `.secrets/scenecode.env`，不安装环境。完成的场景可以复用；生成服务报错会记录在 `generation_manifest.json`。更改实验输入/代码时使用新的输出目录。

当前接入六类：`free_fall`、`projectile`、`rebound`、`sliding`、`collision`、`support_tip`。用 `--tasks` 后接一个或多个名称选择子集。默认共用一套 SceneCode 实验室资产，逐任务重新设置初始状态和触发，六个任务各自独立仿真；不是六条旧视频的复制。

## 已有 SceneCode 资产的显式复用

```bash
python3 scripts/runtime.py physical-reference -- \
  --output /root/autodl-tmp/worldolympiad2/outputs/physical_reference_pybullet_20260908_v2_room \
  --stage reference \
  --source-world /root/autodl-tmp/worldolympiad2/outputs/scenecode_pybullet_cae_robotics_formal_v9_20260822/pilot/world_ir.json \
  --bindings /root/autodl-tmp/worldolympiad2/outputs/physical_reference_pybullet_20260908_v2_room/reuse_bindings.json
```

bindings明确指向来源 WorldIR 中的对象ID与实际 grounding 描述。2026-09-08首批运行遇到 SceneCode API 日限额耗尽，使用已有红色方块、木块、杯子和桌子继续接通后续；`scene_source.json` 标明复用来源，不把它当成新场景生成结果。新场景的在线生成/自动绑定分支仍需服务恢复后实跑确认。

还可分别运行 `--stage simulate`、`render`、`flow`、`status`。simulate需要提供来源与bindings；render和flow读取该运行目录的已有输入。任务级完成记录支持中断后继续。改变源视频后，flow会核对摘要并重新提取，不能将旧flow用于新视频。

## 产物与含义

每个 `tasks/<task>/reference` 保存：`rgb.mp4`、`camera_rgb.png`、`camera_calibration_info.json`、PyBullet深度、`state_trajectory.npz`、接触记录及 `dream2flow/object_flow_result.pt/.npz`。RGB由现有Blender回放脚本生成；物理状态与标定深度来自PyBullet。观察窗口4秒，15fps，共61帧含初始帧；不是事后拉伸或填充视频。

CAE前处理沿用现有 erosion=3、particle ratio=0.01、min particles=64、max particles=1024、CoTracker visibility=0.8，读取实际标定深度并从视频跟踪对象。模拟器状态没有冒充跟踪器输出。

**当前产物是reference一侧的CAE输入，不是双路CAE模型分数。** 被动落体、碰撞等实验没有机器人初始注册，不凭空制造机器人optimizer plan。接入模型视频及适用的动作评估协议后，才能产生对应比较结果；当前 `model_comparison_score` 保持null。公开汇报不添加二元任务成败。

## 环境归属

- `core / autoresearch`：任务编排、PyBullet、WorldIR编译、视频编码。
- `scenecode`：官方SceneCode生成。
- 现有Blender 4.4.3：参考RGB回放。
- `worldeval`：把PyBullet数组导出为现有CAE所需深度张量。
- `dream2flow`：现有SAM2/CoTracker和官方格式导出。

没有安装或合并环境。现有robotics 10×2交付及分数保持原版本。本轮最小测试覆盖相机投影和reference-only汇总语义，另在服务器实跑六例；共享Blender回放的坐标约定改动通过实际画面和PyBullet投影检查。
