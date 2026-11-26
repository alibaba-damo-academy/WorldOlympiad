# autoresearch

SceneCode 场景生成与评分支撑。流水线本体在 `wo/`，本目录只保留它调用的部分。

**场景生成**（`wo.tasks.scene` 调用）
`robotics_scene_prompt_compiler.py`（把任务编译成 SceneCode prompt，读 `skills/scenecode-robotics-prompt-authoring`）、
`robotics_scenecode_case_generator.py`（跑 SceneCode 并做任务物体盘点与恢复）、`generate_scene.py`、
`configs/scene_prompt_profiles_v1.json`。

**被仓库之外的代码直接 import，不能改名或删除**
`scenecode_*.py`（SceneCode 运行时）、`pybullet_frame_contract.py`（pybullet 桥）、
`robotwin_camera_alignment.py`（RoboTwin 相机）。

**评分支撑**
`score_hierarchy.py`（三轨汇总策略）、`full_track_report.py`、`robotics_resource_lock.py`（GPU 互斥）。

**保留，尚未接入新流水线**：LingBot-World2 物理赛道相关模块，见仓库根 `README.md`。

2026-10-06 之前这里还有一套自带的 agentic harness（`robotics_agentic_harness` 等）和它的 CLI，
已随 `olympiad_forge` 一起清理，标签 `pre-prune-20261006` 可恢复。
