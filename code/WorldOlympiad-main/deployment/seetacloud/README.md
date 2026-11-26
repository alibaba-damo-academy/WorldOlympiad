# deployment/seetacloud

当前服务器（SeetaCloud）的环境供给与两份被代码读取的配置。

- `bootstrap_workspace.sh` `bootstrap_model_runtimes.sh` `download_models.sh` `preflight_migration.sh`：
  当初搭建工作区与模型运行时的脚本，仅作记录，**不是流水线的一部分**，也不保证能在现状上原样重跑。
- `cosmos_task_graph_protocol.json`：被 `autoresearch/cosmos_task_graph_predicates.py` 读取。
- `robolab_simulation_profile.json`：被 `scripts/run_cosmos3_articulated_isaac.py` 读取。

流水线与调度见仓库根 `README.md` 和 `wo/agent/`。
