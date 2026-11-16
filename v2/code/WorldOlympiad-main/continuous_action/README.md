# Continuous Action Evaluation

一个实现、两项指标：完整轨迹形状与完整速度曲线。

完整定义、可见性语义与运行入口见 [评分说明](../docs/continuous_action_evaluation.md)。
当前代码为 `metrics.py`，冻结配置为 `current_metric.py`。
CAE 不包含覆盖率惩罚，不保留旧版公式或重放入口。
