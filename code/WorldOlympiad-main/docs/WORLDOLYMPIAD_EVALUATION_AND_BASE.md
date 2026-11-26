# WorldOlympiad 前作评测调研与 base 验收

更新：2026-10-01。依据本地论文 `2606.11129v2.pdf` 第 6-8、12、30 页，及
[官方代码](https://github.com/alibaba-damo-academy/WorldOlympiad) 的
`bb2e240918d15d822250a33b3504672be545d10d` 快照。此文明确区分论文定义、
公开实现和本项目的实际运行证据。

## 前作的评测设计

前作包含 1,000 个长视频：robotics 400、gaming 400、real-world 200，评测八条
视频生成 pipeline。数据先寻找连续执行片段，再做 chunk 切分、caption 和全视频
上下文修订；interaction 使用这些实际时间边界，而不是逐帧强迫模型对齐参考。

| 宏 track | 论文测量内容 | 实现证据与限制 |
| --- | --- | --- |
| Physical | 力学、热学、材料规则。先以 reference 判断规则是否相关，再评估生成视频是否遵守；相关规则在子类内、子类之间汇总 | MLLM 提取动态主体，SAM3 生成对象证据；核查 `physical/vlm_judge.py` 和规则问句。依赖时间采样，短暂接触／下落可能漏掉 |
| Geometry | Gaussian-splat 重建视频、诊断 meta-view、预测／参考相机轨迹；三项平均 | DA3 重建，动态前景去除／inpainting 后主要评估静态场景；不能当作机械臂或被操作物体的运动正确性 |
| Interaction | chunk 指令、相邻转场、全视频长程一致性，再融合 CLIP 语义辅助分 | `interaction/vlm_inter.py` 与 `interaction/clip_score.py`。不要求 reference／model 视频有相同长度 |

Geometry 的规范分数是三个 [0,1] 子分数的均值，不能把原始求和 [0,3] 直接
与其他 track 混合。相机路径会按参考运动类型选择平移／旋转权重；近静态相机
会惩罚重建抖动。它的相机路径对齐与本项目 CAE 的动作路径对齐是不同定义。

论文的 Interaction 定义为：三层 MLLM 分别归一化到 [0,1]，先对
chunk／transition／global 三层等权；CLIP 原始 cosine 经固定阈值 0.20、0.40
校准后，以 0.1 权重融合。总分对 Physical／Geometry／Interaction 等权。

## 必须记入实验协议的代码差异

官方快照 `interaction/vlm_inter.py` 将全部 chunk、transition 和 global 的
`overall` 放到同一个列表求均值。因此多 chunk 会改变三层的相对权重，与论文
先算每层均值再三层等权不同。单视频入口把 CLIP 保存在独立字段，默认配置和
总分路径不能直接证明已经按论文 10% 公式融合。

本项目既有 14 格来自实际运行的 judge 分数。本次 CAE 层级迁移保留这些已测量
基础分数，新增三轨聚合；不能称为按论文公式重新测量了 Interaction。
后续实验要分别标记 `release-code baseline`、`paper-formula baseline` 和新增
CAE 版本；缺少 CLIP 或三层证据的样本不能伪装为完整论文公式结果。

## CAE 的新归属

CAE 衡量完整动作轨迹的形状、方向／尺度和物理速度曲线与 reference 的一致性。
它不直接判断物理定律、碰撞接触或姿态旋转，因此放在 Interaction 的
`continuous_action` 子指标，而不是重复计入三个宏 track。

当前已选定并实现：

- Interaction = 0.5 × 原交互测量 + 0.5 × CAE。
- Overall = (Physical + Geometry + Interaction) / 3。
- CAE 的 trajectory shape 和 speed curve 保留为诊断分解，不再次参与总分。
- 缺少应有证据保持 pending；语义 N/A 才对剩余适用项归一化；真实零分保留。
- 原四轨报告保留为历史证据。新聚合用独立 schema、score policy 和输出路径。

已有 14 格的新聚合均分 0.52090273，Cosmos 0.53009853，LingBot 0.51170693。
这只是相同旧视频的聚合口径变化，不证明模型改善，也不等于修复后的新 base 数据。

## 七类 base 的验收

七类是经验样本，不按任务 ID 路由代码：pick/place、reorientation、articulation、
hanging、tool use、transfer、kettle-on-coaster。底层按操作、谓词、对象角色与模型
协议选择能力。

每个案例需要分别检查：任务语义／对象数量正确；reference 真实成功；原生模型
场景与已选 reference 对应；相机、材质、碰撞和初态正确；模型原生运行协议与
终止记录明确；三个宏 track 的适用证据齐全。模型没有完成任务的有效输出仍保留。

| 当前发现 | 处理与状态 |
| --- | --- |
| GPU 资源工具使用未定义变量 | 已修复；真实主控→工具→Research 专项委派完成 |
| glTF 只打包 JSON，遗漏外部 buffer／纹理 | 已修复为含依赖摘要的独立资源包；不同资产的同名 buffer 不会互相覆盖；消费端重载测试通过 |
| 多物体 native 轨迹借用 reference 首帧 | 已改为采集全部动态物体的 live PhysX 状态，并将这些状态加入 resume 验证；真实 901 帧倾倒运行记录了三个独立方块的位置、速度、姿态 |
| 原生材质退化为默认颜色 | 配置的 Blender 路径被正确使用；native USD 保留 glTF 材质。挂杯的真实 Isaac 输入已恢复蓝色杯子和黑色挂架 |
| hanging／tool-use／transfer 用 reference 时长终止且成功函数恒 false | 已接入原生目标测量与独立 60 秒 episode；保留真实失败，不再把 reference 时长到期当作任务完成 |

prompt 泛化要分开测：同义改写保持语义与物理输入；对象替换改变资产；布局与
数量变化改变场景。不能通过只改 JSON 中的文本就证明语言理解或模型泛化。
建议分阶段积累七类 × 至少三种 prompt 改写，再加入对象／布局变化和重复 seed。
这是生产验收矩阵，不是恢复已撤销的候选冻结／晋级系统。

## 论文需要的数据

1. 数据生产：各类请求数量、reference 成功率、失败 owner、自动修复成功率、
   单案耗时／调用成本，以及不可用样本的完整计数。
2. 模型结果：按任务、模型、prompt／对象／seed 分组，报告三个宏 track、
   Interaction 两个子指标、CAE shape／speed、观测覆盖和 native 终止记录。
3. 泛化与 RSI：未经修复／修复后、无经验复用／有复用的对照，后续新案例的
   实际复用轨迹；区分 Codex 人工工程修复与 agent 自主发现和解决。
4. 评测可信度：按子指标做人工一致性、权重／采样消融和 case 级误判分析。
   前作人工实验是 20 prompts、8 models、5 人、560 对比较、2,800 个标签；
   Spearman 0.95 是八个模型层面的结果，不能直接作为本项目 CAE 的验证。
5. 公平性：固定模型 checkpoint、协议、输入、相机、seed、评估器版本和采样。
   不同 embodiment 的模型先报告各自匹配参考的结果，跨模型归因明确场景差异。
   置信区间按独立 case／seed 而不是视频帧计算。

当前七类各一个基础样例与 14 格旧结果足以定位问题，仍不足以支持任意 prompt
泛化或完整自主 RSI 的论文结论。先验收 base，再持续生产可复算的新数据。

## 本轮实际运行结果

- 2026-10-01 全分环境回归：核心 887 passed / 1 skipped、原生依赖 1 passed；
  job-ID 恢复检查追加后 OlympiadForge 21 passed。
- 挂杯：材质保真的 USD 重建和真实 Isaac 原生输入渲染完成；杯子恢复蓝色，
  挂架恢复黑色。对照图保存在本地 `artifacts/robotics_base_20261001/`。
- 新 Cosmos pick/place run `base_pick_place_pbr_20261001`：197 帧、7 个模型窗口、
  native task_success=true、termination_reason=task_completed、quality_passed=true。
  reference/preflight/WAM/audit/pair/case 六阶段完成；评测另外留在 case 的
  `three_track_report_v1.json`。CAE=0.71934257，shape=0.85460697，
  speed=0.60548739，三轨总分=0.66619043。
- 新视频、旧 14 格三轨重聚合报告及输入对照图在同一本地结果目录。
- 六项新 Cosmos case 均已三轨／CAE 评分完成；柜门复用已独立复核的 v8 输出，
  LingBot 七项复用完整有效输出。全部 reference、视频、物理轨迹与标定未重建。
- 当前 14/14 格评分完整，均分 0.54120036；Cosmos 0.57069378，LingBot
  0.51170693。原生任务成功为 2/14（Cosmos 取放、LingBot 工具任务），其余是
  有效的模型失败。这不证明七类模型任务均达标，也不是任意 prompt 泛化结果。
- 新运行修复了输入材质和执行协议，与旧运行均分不能作为受控改进消融。
- 这些修复由 Codex 直接实施，不作为 agent 自主 RSI 证据。

## 2026-10-01 自定义原生任务协议更新

挂杯、冲击、倾倒已增加按对象角色／目标谓词选择的 native terminal measurements。
旧 reference 未重建；三份已成功参考轨迹分别通过新判据。新增共享 60 秒原生
episode 预算，不再从 reference 时长推导。实际模型未达成目标时正常保留 timeout
输出，不以假成功替代评测。

工具任务 `base_tool_use_native_v3_20261001` 已完成 901 帧／60 秒；抬升、抓持有
观测，但未出现合格冲击。native task_success=false，termination_reason=
`native_task_timeout`；WAM、audit、pair、case 完成，三轨与 CAE 也已完成。

Geometry／Interaction 的细粒度原始结果已按相同 cohort 导出，并记录子指标
scored／pending／N/A 数量和源文件哈希。样本总数不替代每列有效测量数。

新表的每模型七例均无 pending 子指标。热学各七例 N/A；材料规则 Cosmos
两例适用、LingBot 一例适用；LingBot 单语义 chunk 的 transition 七例 N/A。
JSON 保留 case／model／task 的完整分解和原始规则计数。

对应服务器目录 `outputs/robotics_base_20261001/`：`base_score_selection_v1.json`、
`base_three_track_suite_v1.json`、`base_fine_grained_report_v1.json`、
`base_fine_grained_tables_v1.md`。本地副本在 `artifacts/robotics_base_20261001/`。

两个模型使用各自 embodiment 和匹配 reference，不能把这个 cohort 标成前作
Table 10 那样的跨模型同一场景 subset。倾倒的既有 CAE 只描述其已选跟踪目标，
不能声称独立衡量了每个方块。后续论文数据需补组件／阶段范围、人评一致性、
多 prompt 和重复 seed 的证据。

## 原生目标与网格坐标的追加复核

材质 OBJ 导出原先使用 `forward_axis=NEGATIVE_Y`，相对 glTF／参考几何绕 Z
多转了 180°。改为 `Y`，实际 Blender 的非对称网格回归通过。原生显示 USD
本身方向正确，错误影响分析几何与目标判据；已生成的视频没有据此被删改。
修正后的挂杯几何能在原 ref 的 25.8 秒处识别穿柄，模型旧输出没有成功。

倾倒判据曾遗漏 source-container 的凸包，成功分支会报错；已补齐，并用已成功
的原参考轨迹验证。桌面区域不再继承“白杯放桌面中心”的特定样例判据；
现在按实际 named support 的碰撞几何、区域范围、释放和声明的直立条件判定。
复用相同 SDF 的两个对象不能仅凭文件哈希相同就合并，相关错误尝试未采用。

`native_goal_reassessment_v3.json` 记录六项原参考轨迹都通过修正后的目标测量。
旧取放模型在官方判据首次成功时已经结束，因此没有覆盖新判据要求的后续
稳定窗口；不能把后续帧缺失解释为模型必然失败。旧表的 2/14 是各旧运行实际
采用的 native runtime 结果，并不等于新目标判据下的统一成功率。

取放、瓶子扶正、壶放杯垫已使用当前目标语义完成三项新增验收与完整评分，
保持原 reference 不变。取放 227 帧后成功，总分 0.69718790、CAE 0.66072738；
扶正与杯垫任务各 901 帧／60 秒超时，总分分别为 0.59674401、0.58337281，
CAE 分别为 0.17846404、0.29963685。

扶正的末帧瓶身直立余弦为 0.0006022，仍近水平；杯垫任务末帧壶仍由双指
握持，壶与杯垫水平距离约 0.2006 米。它们是已测出的任务失败，不由视频总分
替代。新独立 cohort v2 为 14/14 评分完整，无 pending 细分列；Cosmos 均分
0.57208898、LingBot 0.51170693，整体 0.54189796。旧版本全部保留。

对应 `base_score_selection_v2.json`、`base_three_track_suite_v2.json`、
`base_fine_grained_report_v2.json`、`base_fine_grained_tables_v2.md`。
这组数据仍未证明跨 prompt 泛化或七类模型任务全部成功。


### 2026-10-01: equal-time acceptance, before further interface repairs

The user requires equal reference/imagination time for **every task**, including
future generalization tasks. The current acceptance order is time alignment,
then the outstanding native model/SceneCode interface repairs, one item at a
time. Existing valid reference assets and trajectories remain available.

Cosmos now binds the complete certified 15 Hz reference window before inference
and captures that entire window independently of success. The six rigid-task
outputs under `base_*_window_v2_20261001` and the native articulated task under
`robotics_base_20261001/articulated_time_window_v1` completed native validation.
First passing predicate and final predicate are separate diagnostics; these
runs do not establish seven-task model success or prompt generalization.

LingBot-RoboTwin retains native 10 Hz output. Its predeclared common prefix
ends on both the 15 Hz reference and 10 Hz prediction frame clocks, without
stretching either video or selecting a cutoff after seeing model success.
Original full expert videos and predicted latent chunks are retained. The
seven-case native acceptance job completed sequentially. All 14 native pairs
passed independent time acceptance.
The selected prefix may omit at most two terminal reference frames; the full
expert success certificate must not be represented as a newly measured
prefix-end goal certificate.

`time_window_acceptance_v1.json` independently checks artifact hashes,
predeclared window bindings and actual exported video duration for all 14
base cells. It separates time acceptance from native goal outcome. Historical
score tables remain labeled with their original mixed, independent-duration
protocols until new equal-time evaluations are complete.

SceneCode official audit is pinned to commit
`1c8e972d24c960a319b5df936ae552ad410beb85`. The official image generator uses
`gpt-image-2`; old robotics context-image flags were disabled. They are now
required for new generation. The actual SceneCode image endpoint generated the
oven reference, and its SHA-256 is bound to the image-conditioned articulated
planner. Nine parts were constructed. Complete asset export is still blocked
by a generated Blender material `NodeLink` lifetime error; no new oven physical
reference or RSI success is claimed. This remains a subsequent interface repair,
not a reason to declare time acceptance complete early.


Final time acceptance: `time_window_acceptance_v1.json` reports 14/14 validated
and `complete=true`. Full core verification: 906 passed, 3 skipped. An isolated
copy with a changed frozen producer snapshot was correctly rejected. Source
validation also now checks SceneCode stage digests and recomputes the native
asset dependency graph digest, rather than treating a derived digest as a file.
The next sequential repair is `oven_image_asset_material_api_v2`: only the
material-link control flow changes, while the generated image and checked
ObjectPlan are retained. This does not establish a physical oven reference.


The material API repair completed in `oven_image_asset_material_api_v2`.
Native SceneCode rechecked 9/9 cached part constructors, exported the complete
GLB, and generated the URDF from the original image-conditioned checked plan.
The source receipt binds the one-line `if` to `elif` change. An isolated
PyBullet kinematic diagnostic loaded the URDF, found exactly one revolute joint
on a horizontal bottom axis, and measured zero handle-to-door frame drift.
This is asset/interface validation, not successful manipulation: the native
URDF omits inertial entries and still needs the normal collision/inertia
materialization before any physical reference or RSI evaluation.
The node-link lifetime lesson is now included in all three SceneCode constructor
prompt profiles. The incremental runtime patch applied with zero fuzz.

The next sequential base repair is camera observability. Two candidates were
selected solely from the complete certified cabinet reference motion and its
handle geometry. They are being rendered in the original native simulator to
check actual occlusion. Neither candidate is a published reference/model pair;
a selected camera requires matching reference RGB, depth and calibration.


Native cabinet replay audit found two concrete remaining blockers. The close-up
candidate exposes the handle initially, but the door/handle leaves that plane
later in the motion, so it is not admitted as a matched reference/model camera.
The replay binding previously compared the original reference SDF hash with a
certified visual-only revision. The new replay check verifies the manifest and
alignment certificate and compares all non-visual SDF physics, with mesh URIs
replaced by mesh-content hashes. Four regression cases reject collision, mass,
and joint-limit changes; the nine targeted joint/replay tests passed.

With this binding fixed, all 249 reference transitions replayed without invalid
native constraints. Native maximum opening was 1.3120628835 rad (about 75.2
degrees), while the original PyBullet reference ended at 1.4354847061 rad
(about 82.2 degrees) and claimed success under a 0.90 travel-fraction executor.
Its frozen graph names `joint_at_open_limit` and declares terminal fraction 1.0;
the native task targets 1.5707963 rad with tolerance 0.08. The reference
materializer also has a stale 0.60 default in its execution-geometry metadata,
while the bridge CLI defaults to 0.90. These are identified contract/executor
mismatches, not permission to loosen the native predicate. No new cabinet camera
or successful native reference is admitted. The next base work is to align the
reference acceptance contract and physically verify a smooth native expert
before interpreting Cosmos cabinet failures or publishing new paper scores.
