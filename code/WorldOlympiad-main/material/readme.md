材料学相关的 pipeline 已经就绪，涵盖颜色混合、溶解、硬度/形变、可燃性四类问题，并将 VLM 判断与轻量视觉启发式结合。

## 目录
- `material/integrated_pipeline.py`：集成入口，过滤题目并调用各问题分析。
- `material/analysis.py`：视觉启发式（RGB 混色、区域消失、火焰检测、形变估计）与数据 URL 编码。
- `material/__init__.py`：便捷导出。

## 用法
从仓库根目录运行：
```bash
python material/integrated_pipeline.py \
  --video data/sample.mp4 \
  --prompt "Describe what happens in the video" \
  --model google/gemini-2.5-flash \
  --output-dir material/outputs
```

选项：
- `--force` 跳过筛题，直接处理全部材料学问题。
- `--model` 指定 OpenRouter 上的 VLM，默认 `google/gemini-2.5-flash`。

输出：
- `material/outputs/pipeline_result.json`：筛题结果与每个问题的汇总。
- `material/outputs/<question_id>/result_<question_id>.json`：包含启发式指标与 VLM 判定。

## 视觉启发式
- 颜色混合：提取首尾帧主色，用 RGB 平均预测混色，计算与终态色差得分。
- 溶解：Otsu 阈值估计显著区域面积，若末尾较开头收缩 >25% 则视作溶解。
- 可燃性：HSV 范围检测火焰色占比，作为燃烧线索。
- 硬度/形变：Canny 边缘差分估计形变程度，较大变化提示软/可折叠材料。

## 依赖
- Python 包：`opencv-python`, `numpy`, `requests`, `python-dotenv`（与项目一致）。
- SAM3 可按需接入，但当前启发式默认无需额外权重即可运行。确认 `.env` 中配置 `api_key` 供 OpenRouter 调用。
