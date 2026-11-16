# 面向汇报的 Physical 场景候选与 SceneCode prompt

这些是依据公开 benchmark 典型现象重新编写的候选，不是官方 prompt 原文，也不是已经生成的视频。现有六例补背景的视频另行提供。优先建议展示球撞玩具、斜坡滚动、多米诺、浅盘落入、叠放倒塌，视觉事件比单独方块移动更明确。

适配依据：仓库的 `scenecode-robotics-prompt-authoring` 规范，以及服务器 SceneCode 自带 `examples/prompts.csv`、`scenecode/prompts/data/robot_eval/scene_prompt_generator.yaml`。SceneCode 输入采用1–5句自然语言，先写房间及2–4个背景物件，再写准确数量、单体外观和初始支撑关系。动态指令与物理参数单独交给PyBullet。视觉材质与实际摩擦/恢复系数分开记录，不能靠“橡胶”一词确定数值。

下列每个 SceneCode prompt 都是4句，可以直接用于场景生成。这里的初始状态是资产生成后的静态布局；落体、抛体等在PyBullet的实验初始化阶段设置释放高度，随后才记录reference。不会要求SceneCode生成悬空且需要静态重力检查的物体，也不会把物体已经倒下、已落入容器写成初始布局。具体对象身份、数量和执行绑定仍需与生成结果对应。

公开来源中，Physics-IQ提供了落体、碰撞、斜坡、多米诺及容器等实际样例；PhyGenBench包含弹性现象；Physion提供支撑、滑动与结构稳定性场景。下文链接及样例定位方便核对，本表不报告模型成败。

## 双物体落体

来源启发：[Physics-IQ：0001 / ball-and-block-fall](https://github.com/google-deepmind/physics-IQ-benchmark/blob/main/descriptions/descriptions_original.csv)。

**SceneCode 场景描述**

A bright school science room, 4.8 m by 4.0 m, contains a central pale-oak demonstration table, a closed storage cabinet on the back wall, a cork noticeboard with plain colored cards, and a side window with a fabric blind. On the table are one matte red solid sphere 0.10 m in diameter and one matte blue solid cube 0.10 m on each side, each a separate opaque object. Initially, both objects rest separately near the middle of the tabletop with 0.30 m of clear space between them and an unobstructed area below the near edge. Soft diffuse illumination and matte materials keep the task objects distinct while preserving the furnished room background.

**动态指令**：Release both objects from the same height at the same time.

**PyBullet 绑定**：PyBullet places both objects 0.35 m above the support with zero velocity；release under gravity。

**接入状态**：new_multi_object_initialization_required；尚未生成该候选场景。

## 滚动物体撞击目标

来源启发：[Physics-IQ：0007 / ball-hits-duck](https://github.com/google-deepmind/physics-IQ-benchmark/blob/main/descriptions/descriptions_original.csv)。

**SceneCode 场景描述**

A warm modern family living room, 4.5 m by 4.0 m, has a low oak coffee table, a sage-green sofa against the far wall, a slim bookcase, and a framed geometric print. The coffee table holds one matte orange solid ball 0.10 m in diameter and one small yellow opaque duck-shaped toy with a broad stable base, 0.14 m long. Initially, the ball is on the left and the toy stands on the right, 0.35 m apart along a clear strip of tabletop. Soft diffuse illumination and matte materials keep the task objects distinct while preserving the furnished room background.

**动态指令**：Roll the orange ball toward the yellow toy.

**PyBullet 绑定**：ball and target at rest, horizontally separated；set ball linear velocity along the target direction。

**接入状态**：existing_collision_operator_new_assets_required；尚未生成该候选场景。

## 球体沿斜坡滚动

来源启发：[Physics-IQ：0019 / ball-ramp](https://github.com/google-deepmind/physics-IQ-benchmark/blob/main/descriptions/descriptions_original.csv)。

**SceneCode 场景描述**

A tidy university teaching laboratory, 5.0 m by 4.0 m, contains an ash-wood workbench, a metal storage cabinet along the far wall, a pinboard, and a high side window. On the workbench are one single-piece matte wood wedge with a flat sloping top, 0.50 m long, 0.18 m wide and 0.15 m high, and one solid teal sphere 0.08 m in diameter. Initially, the wedge rests with its low end toward the clear center of the bench and the sphere rests separately beside its high end. Soft diffuse illumination and matte materials keep the task objects distinct while preserving the furnished room background.

**动态指令**：Release the teal sphere at the upper end of the slope.

**PyBullet 绑定**：PyBullet positions sphere on the high end before release；gravity only。

**接入状态**：ramp_asset_binding_and_release_required；尚未生成该候选场景。

## 多米诺连锁倾倒

来源启发：[Physics-IQ：0034 / block-domino](https://github.com/google-deepmind/physics-IQ-benchmark/blob/main/descriptions/descriptions_original.csv)。

**SceneCode 场景描述**

A contemporary design classroom, 5.0 m by 4.0 m, has a long matte maple table, a low closed sideboard, a wall-mounted pinboard, and two simple ceiling pendants. Six separate solid wooden rectangular blocks stand on the tabletop, each 0.035 m thick, 0.07 m wide and 0.16 m tall, alternating red and blue. Initially, all six blocks stand upright in one straight row with 0.06 m gaps between neighboring blocks and clear space beyond the final block. Soft diffuse illumination and matte materials keep the task objects distinct while preserving the furnished room background.

**动态指令**：Tip the first block toward the remaining row.

**PyBullet 绑定**：upright row, distinct instance IDs；brief force on first block。

**接入状态**：multi_object_tracking_binding_required；尚未生成该候选场景。

## 落入浅盘

来源启发：[Physics-IQ：0013 / ball-in-basket](https://github.com/google-deepmind/physics-IQ-benchmark/blob/main/descriptions/descriptions_original.csv)。

**SceneCode 场景描述**

A sunlit home study, 4.2 m by 3.8 m, has a birch desk, a low bookshelf against the back wall, a fabric reading chair away from the desk, and a small framed print. The desk holds one matte purple solid sphere 0.08 m in diameter and one shallow cream-colored rectangular tray, 0.30 m long, 0.24 m wide and 0.04 m high, with a bottom, four low sides and an open empty interior. Initially, the sphere rests beside the empty tray with a 0.20 m gap, and the tray interior is unobstructed. Soft diffuse illumination and matte materials keep the task objects distinct while preserving the furnished room background.

**动态指令**：Release the purple sphere above the center of the tray.

**PyBullet 绑定**：PyBullet moves sphere above tray; keep tray walls low for visibility；release from rest。

**接入状态**：containment_asset_and_release_binding_required；尚未生成该候选场景。

## 多物体碰撞传递

来源启发：[Physics-IQ：0028 / ball-train](https://github.com/google-deepmind/physics-IQ-benchmark/blob/main/descriptions/descriptions_original.csv)。

**SceneCode 场景描述**

A clean maker studio, 4.8 m by 4.0 m, contains a broad pale-wood worktable, a closed tool cabinet against the far wall, a simple wall clock, and a diffuse ceiling panel. Three separate matte solid spheres, one orange, one blue and one yellow, each 0.09 m in diameter, rest on the table. Initially, the blue and yellow spheres are near one another along the same straight line, while the orange sphere sits 0.30 m to their left with a clear path toward them. Soft diffuse illumination and matte materials keep the task objects distinct while preserving the furnished room background.

**动态指令**：Roll the orange sphere toward the other two spheres.

**PyBullet 绑定**：three dynamic rigid bodies, distinct object IDs；initial velocity on orange sphere。

**接入状态**：multi_body_collision_and_tracking_required；尚未生成该候选场景。

## 弹性回弹

来源启发：[PhyGenBench：Elasticity / rubber ball](https://github.com/PhyGenBench/PhyGenBench/blob/main/PhyGenBench/prompts.json)。

**SceneCode 场景描述**

A bright indoor recreation room, 4.8 m by 4.2 m, has a light wooden demonstration table, a low bench along the rear wall, a closed equipment cabinet, and two warm ceiling lights. One opaque single-piece coral-colored rubber sphere 0.12 m in diameter rests on the smooth central tabletop. Initially, the sphere is still, with the surrounding tabletop clear and the background furniture behind the work area. Soft diffuse illumination and matte materials keep the task objects distinct while preserving the furnished room background.

**动态指令**：Release the coral sphere above the tabletop.

**PyBullet 绑定**：place sphere at a recorded release height；gravity; explicit contact restitution in physics config。

**接入状态**：existing_rebound_operator_sphere_asset_required；尚未生成该候选场景。

## 摩擦减速

来源启发：[Physion：Sliding scenario](https://physion-benchmark.github.io/)。

**SceneCode 场景描述**

A warm materials teaching room, 4.6 m by 4.0 m, contains a matte oak workbench, a rear storage sideboard, an acoustic wall panel, and a side window with a pale blind. One solid dark-blue wooden cuboid 0.16 m long, 0.10 m wide and 0.05 m high rests broad-face-down on the light workbench. Initially, the cuboid is near the left side, with a clear horizontal path across the middle of the bench. Soft diffuse illumination and matte materials keep the task objects distinct while preserving the furnished room background.

**动态指令**：Slide the dark-blue block across the workbench.

**PyBullet 绑定**：block at rest on flat support；specified initial velocity; friction stored numerically。

**接入状态**：existing_sliding_operator_new_asset_required；尚未生成该候选场景。

## 支撑边界与倾倒

来源启发：[Physion：Support scenario](https://physion-benchmark.github.io/)。

**SceneCode 场景描述**

A small architecture workshop, 4.8 m by 4.0 m, has a central birch worktable, a closed flat-file cabinet along the far wall, a framed plan without legible text, and a soft overhead light. One solid terracotta-colored rectangular block 0.24 m long, 0.10 m wide and 0.08 m high rests horizontally on the tabletop. Initially, the block is fully supported near the front edge, with its long axis perpendicular to that edge and the adjacent floor clear. Soft diffuse illumination and matte materials keep the task objects distinct while preserving the furnished room background.

**动态指令**：Move the block outward beyond its support boundary.

**PyBullet 绑定**：PyBullet sets the chosen overhang before free dynamics；release at specified support margin。

**接入状态**：overhang_initialization_required；尚未生成该候选场景。

## 叠放结构倒塌

来源启发：[Physion：Stability / stacked objects](https://physion-benchmark.github.io/)。

**SceneCode 场景描述**

A contemporary learning lounge, 4.8 m by 4.0 m, contains a central maple table, a low bookcase along the far wall, a muted green chair beside it, and a framed abstract print. Three separate matte solid cubes, one red, one green and one yellow, each 0.10 m on a side, form a straight upright stack on the tabletop. Initially, the red cube supports the green cube and the green cube supports the yellow cube, with the stack stable and open tabletop around it. Soft diffuse illumination and matte materials keep the task objects distinct while preserving the furnished room background.

**动态指令**：Push the lower cube sideways.

**PyBullet 绑定**：settled three-body stack；short lateral force on the lower cube。

**接入状态**：multi_body_stack_initialization_required；尚未生成该候选场景。
