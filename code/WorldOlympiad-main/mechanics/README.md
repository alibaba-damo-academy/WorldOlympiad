# Mechanics Analysis Pipeline

This directory contains tools for analyzing mechanics-related physics phenomena in videos through object segmentation and trajectory analysis.

## Overview

The mechanics pipeline extends the LV-Bench 2 evaluation system by:

1. **Filtering** relevant physics questions using LLM (via `problem_filter.py`)
2. **Extracting** object descriptions from video prompts
3. **Segmenting** objects using SAM3 with text prompts
4. **Tracking** object centroids across video frames
5. **Analyzing** trajectories for physics compliance (gravity, buoyancy, impact, etc.)
6. [Optional] **DA3 3D Augmentation**: Use Depth-Anything-3 to estimate per-frame depth and compute 3D world centroids for each tracked object

## Architecture

```
problem_filter.py (问题过滤)
        ↓
integrated_pipeline.py (集成管道)
        ↓
object_segmentation.py (对象分割)
    ↓                    ↓
SAM3 Segmentation    LLM Object Extraction
        ↓
trajectory_analysis.py (轨迹分析)
        ↓
Physics Compliance Results
```

## Files

### Core Modules

- **`integrated_pipeline.py`**: Complete end-to-end pipeline from filtering to analysis
- **`object_segmentation.py`**: SAM3-based object segmentation with text prompts
- **`trajectory_analysis.py`**: Physics-aware trajectory analysis
- **`da3_integration.py`**: DA3 depth + 3D centroid enrichment for trajectories

### Legacy/Test Files

- **`sam_test.py`**: Basic SAM3 usage example
- **`yolo_test.py`**: YOLO-based tracking example
- **`boxmot_test.py`**: BoxMOT tracking example
- **`mechanics.py`**: Empty placeholder

## Installation

Required dependencies:
```bash
pip install ultralytics opencv-python numpy
```

Required models:
- SAM3 model: `./weights/sam3/sam3.pt`
- YOLO models: `yolo11n.pt`, `yolo11m.pt` (optional, for legacy tracking)
- DA3: local repo under `Depth-Anything-3/` with a model id (e.g., `depth-anything/DA3NESTED-GIANT-LARGE`)

## Usage

### 1. Complete Integrated Pipeline (Recommended)

Process a video with automatic problem filtering and segmentation:

```bash
python mechanics/integrated_pipeline.py \
    --video data/6.mp4 \
    --prompt "A stone rolling down a hill" \
    --enable-da3
```

This will:
1. Filter to identify relevant mechanics questions (gravity, buoyancy, etc.)
2. For each question, extract objects and run SAM3 segmentation
3. Generate trajectory data and visualizations
4. Save all results to `mechanics/outputs/`

**Options:**
```bash
# Force processing all mechanics questions (skip filtering)
python mechanics/integrated_pipeline.py \
    --video data/6.mp4 \
    --prompt "A stone rolling" \
    --force \
    --enable-da3 --da3-model depth-anything/DA3NESTED-GIANT-LARGE --da3-device cuda:0

# Specify custom output directory
python mechanics/integrated_pipeline.py \
    --video data/6.mp4 \
    --prompt "A ball falling" \
    --output-dir mechanics/my_results

# Use different LLM model
python mechanics/integrated_pipeline.py \
    --video data/6.mp4 \
    --prompt "Water flowing" \
    --model "google/gemini-2.0-flash-exp"
```

### 2. Object Segmentation Only

Run just the segmentation pipeline without filtering:

```bash
python mechanics/object_segmentation.py \
    --video data/videos/buoyancy_self_forcing_2.mp4 \
    --prompt "A lemon floating in water" \
    --question-id buoyancy \
    --enable-da3
```

**With custom object prompts:**
```bash
python mechanics/object_segmentation.py \
    --video data/6.mp4 \
    --prompt "A ball colliding with a wall" \
    --question-id impact \
    --custom-prompts "ball" "wall" \
    --enable-da3
```

**Disable automatic object extraction:**
```bash
python mechanics/object_segmentation.py \
    --video data/6.mp4 \
    --prompt "Objects moving" \
    --question-id gravity \
    --no-extract  # Will use default prompt "object"
```

### 3. Trajectory Analysis Only

Analyze existing trajectory data:

```bash
python mechanics/trajectory_analysis.py \
    --trajectory-json mechanics/outputs/gravity/trajectories_gravity.json \
    --question-id gravity \
    --fps 25
```

**Save analysis results:**
```bash
python mechanics/trajectory_analysis.py \
    --trajectory-json mechanics/outputs/buoyancy/trajectories_buoyancy.json \
    --question-id buoyancy \
    --output mechanics/outputs/buoyancy/analysis.json
```

## Output Structure

After running the integrated pipeline, outputs are organized as follows:

```
mechanics/outputs/
├── pipeline_result.json          # Complete pipeline summary
├── gravity/                       # Per-question results
│   ├── sam3_results/             # SAM3 segmentation outputs
│   │   └── video.mp4             # Segmented video
│   ├── trajectories_gravity.json # Trajectory data (with `depths` and `points3d_world` when DA3 is enabled)
│   ├── visualization_gravity.mp4 # Visualization with centroids
│   └── result_gravity.json       # Question-specific results
├── buoyancy/
│   └── ...
└── impact/
    └── ...
```

### Output File Formats

**`trajectories_{question_id}.json`**:
```json
{
  "trajectories": [
    {
      "object_id": 0,
      "label": "ball",          # human-readable (from prompt or fallback)
      "raw_label": "person",    # SAM3 class/auto label if available
      "centroids": [[320.5, 240.3], [321.2, 245.8], ...],
      "frames": [0, 1, 2, ...],
      "bboxes": [[300, 220, 340, 260], ...],
      "depths": [2.31, 2.29, ...],
      "points3d_world": [[x, y, z], ...],
      "num_frames": 100
    }
  ]
}
```

**`pipeline_result.json`**:
```json
{
  "video_path": "data/6.mp4",
  "video_prompt": "A stone rolling",
  "filter_result": {
    "mechanics": ["gravity", "impact"],
    "thermotics": [],
    "material": []
  },
  "mechanics_results": [
    {
      "question_id": "gravity",
      "num_objects": 1,
      "trajectories": [...],
      "outputs": {
        "segmented_video": "...",
        "trajectory_json": "...",
        "visualization_video": "..."
      }
    }
  ],
  "summary": {
    "total_questions": 2,
    "successful": 2,
    "failed": 0
  }
}
```

## Physics Analyses

The trajectory analysis module evaluates different physics phenomena:

### Gravity (`question_id="gravity"`)

Checks if free-falling objects move downward with positive acceleration.

**Evaluation criteria:**
- ✓ Object falling with downward acceleration
- ✗ Object rising without support
- ~ Object stationary (neutral)

### Buoyancy (`question_id="buoyancy"`)

Checks if objects in fluids float or sink appropriately.

**Evaluation criteria:**
- ✓ Object stationary near surface (floating)
- ✓ Object sinking (higher density)
- ✗ Object rising rapidly

### Impact (`question_id="impact"`)

Detects collisions between objects.

**Evaluation criteria:**
- Detects when objects are within threshold distance
- Estimates relative velocities
- Reports collision frames

### Compression (`question_id="compression"`)

Analyzes object deformation (future implementation).

## Configuration

### SAM3 Model Path

Default: `./weights/sam3/sam3.pt`

Change via command line:
```bash
--sam3-model /path/to/your/sam3.pt
```

### Device Selection

Default: `cuda:1`

Modify in source files:
```python
# In object_segmentation.py, line ~200
overrides = dict(
    ...
    device='cuda:0',  # Change device here
    ...
)
```

### SAM3 Parameters

Adjust confidence and IoU thresholds:
```python
# In object_segmentation.py, segment_objects_sam3()
conf=0.5,  # Confidence threshold
iou=0.7,   # IoU threshold
```

## Examples

### Example 1: Gravity Analysis

```bash
# Video: Ball falling from height
python mechanics/integrated_pipeline.py \
    --video data/videos/gravity_test.mp4 \
    --prompt "A red ball falling from a table" \
    --force

# Expected output:
# - Detects "gravity" question
# - Segments "ball" object
# - Tracks downward trajectory
# - Confirms positive acceleration
# Result: gravity compliance = YES
```

### Example 2: Buoyancy Analysis

```bash
# Video: Lemon in water
python mechanics/integrated_pipeline.py \
    --video data/videos/buoyancy_self_forcing_2.mp4 \
    --prompt "A lemon floating on water surface"

# Expected output:
# - Detects "buoyancy" question
# - Segments "lemon" and "water"
# - Tracks minimal vertical motion
# Result: buoyancy compliance = YES (floating)
```

### Example 3: Collision Detection

```bash
# Video: Billiard balls colliding
python mechanics/integrated_pipeline.py \
    --video data/videos/impact_self_forcing_1.mp4 \
    --prompt "A cue ball striking billiard balls"

# Expected output:
# - Detects "impact" question
# - Segments multiple balls
# - Detects collision events
# Result: impact compliance = YES (collision detected)
```

## Integration with Main Pipeline

The mechanics pipeline integrates with the main LV-Bench 2 judge pipeline:

```python
# In judge_pipeline.py or custom script
from mechanics.integrated_pipeline import run_integrated_mechanics_pipeline

# 1. Filter mechanics questions
from problem.problem_filter import filter_problems_by_physics
filter_result = filter_problems_by_physics(video, model, prompt)

# 2. If mechanics questions exist, run segmentation
if filter_result.get("mechanics"):
    mechanics_result = run_integrated_mechanics_pipeline(
        video_path="video.mp4",
        video_prompt="...",
    )

# 3. Combine with LLM judge results
# Use trajectory data to augment physics evaluation
```

## Troubleshooting

### SAM3 Model Not Found

```
Error: model=./weights/sam3/sam3.pt not found
```

**Solution**: Download SAM3 model or update path:
```bash
--sam3-model /your/path/to/sam3.pt
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Use CPU: `device='cpu'` in code
2. Reduce image size: `imgsz=320` instead of 640
3. Disable half precision: `half=False`

### No Objects Detected

```
Warning: No objects detected in video
```

**Solutions**:
1. Use custom prompts: `--custom-prompts "ball" "cube"`
2. Lower confidence: Modify `conf=0.3` in code
3. Check video quality and object visibility

### Object Extraction Failed

```
Error extracting objects from prompt
```

**Solution**: Use `--no-extract` and provide `--custom-prompts`:
```bash
python mechanics/object_segmentation.py \
    --video video.mp4 \
    --prompt "..." \
    --no-extract \
    --custom-prompts "object1" "object2"
```

## Future Enhancements

Planned improvements:

1. **Deformation Analysis**: Quantify object shape changes for compression evaluation
2. **Multi-object Tracking**: Better association across frames using ReID
3. **Physics Metrics**: More sophisticated physics calculations (momentum, energy)
4. **Real-world Calibration**: Convert pixel measurements to real-world units
5. **Temporal Consistency**: Validate trajectory smoothness

## References

- SAM3: Segment Anything Model 3 (Ultralytics implementation)
- YOLO: YOLOv11 object detection
- BoxMOT: Multi-object tracking with ReID
