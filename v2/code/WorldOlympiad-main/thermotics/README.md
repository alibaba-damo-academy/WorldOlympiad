# Thermotics Analysis Pipeline

This directory contains tools for analyzing thermal physics phenomena (phase transitions) in videos through SAM3-based substance segmentation and state change detection.

**Supported Phase Transitions:**
1. **Melting** (固→液 / solid → liquid)
2. **Sublimation** (固→气 / solid → gas)
3. **Vaporization** (液→气 / liquid → gas)
4. **Condensation** (气→液 / gas → liquid)
5. **Deposition** (气→固 / gas → solid)
6. **Freezing** (液→固 / liquid → solid)

## Overview

The thermotics pipeline extends the LV-Bench 2 evaluation system by:

1. **Filtering** relevant thermal physics questions using LLM (via `problem_filter.py`)
2. **Extracting** substance descriptions from video prompts
3. **Segmenting** substances using SAM3 with text prompts
4. **Tracking** state changes across video frames through:
   - Area changes (expansion/shrinking)
   - Shape changes (fragmentation/coalescence)
   - Disappearance (evaporation/sublimation)
5. **Analyzing** thermal phenomena compliance (melting, evaporation, sublimation, etc.)

## Architecture

```
problem_filter.py (问题过滤)
        ↓
integrated_pipeline.py (集成管道)
        ↓
state_change_detection.py (状态变化检测)
    ↓                    ↓
SAM3 Segmentation    LLM Substance Extraction
        ↓
state_analysis.py (物理分析)
        ↓
Thermal Physics Compliance Results
```

## Key Differences from Mechanics Pipeline

| Aspect | Mechanics | Thermotics |
|--------|-----------|------------|
| **Focus** | Object motion (trajectory) | State changes (solid→liquid→gas) |
| **Tracking** | Centroid position over time | Area, shape, fragmentation over time |
| **Metrics** | Displacement, velocity, acceleration | Area change rate, disappearance, compactness |
| **Physics** | Gravity, buoyancy, impact | Melting, evaporation, sublimation |
| **Visualization** | Trajectory lines | Area change with color-coded states |

## Files

### Core Modules

- **`integrated_pipeline.py`**: Complete end-to-end pipeline from filtering to analysis
- **`state_change_detection.py`**: SAM3-based substance segmentation and state change tracking
- **`state_analysis.py`**: Physics compliance analysis from state change data (with frame downsampling)
- **`__init__.py`**: Module exports

## Installation

Required dependencies:
```bash
pip install ultralytics opencv-python numpy
```

Required models:
- SAM3 model: `./weights/sam3/sam3.pt`

## Usage

### 1. Complete Integrated Pipeline (Recommended)

Process a video with automatic problem filtering and state change detection:

```bash
python thermotics/integrated_pipeline.py \
    --video data/ice_melting.mp4 \
    --prompt "Ice cubes melting in warm water"
```

This will:
1. Filter to identify relevant thermotics questions (ice_melting, water_evaporation, etc.)
2. For each question, extract substances and run SAM3 segmentation
3. Track area changes, shape changes, and disappearance
4. Generate state change metrics and visualizations
5. Save all results to `thermotics/outputs/`

**Options:**
```bash
# Force processing all thermotics questions (skip filtering)
python thermotics/integrated_pipeline.py \
    --video data/sublimation.mp4 \
    --prompt "Dry ice sublimating" \
    --force

# Specify custom output directory
python thermotics/integrated_pipeline.py \
    --video data/wax_melting.mp4 \
    --prompt "Candle wax melting" \
    --output-dir thermotics/my_results

# Use different LLM model
python thermotics/integrated_pipeline.py \
    --video data/steam.mp4 \
    --prompt "Water boiling and producing steam" \
    --model "google/gemini-2.0-flash-exp"
```

### 2. State Change Detection Only

Run just the state change detection pipeline without filtering:

```bash
python thermotics/state_change_detection.py \
    --video data/ice_melting.mp4 \
    --prompt "Ice melting in water" \
    --question-id ice_melting
```

**With custom substance prompts:**
```bash
python thermotics/state_change_detection.py \
    --video data/evaporation.mp4 \
    --prompt "Water evaporating from a hot surface" \
    --question-id water_evaporation \
    --custom-prompts "water" "steam" "vapor"
```

**Disable automatic substance extraction:**
```bash
python thermotics/state_change_detection.py \
    --video data/sublimation.mp4 \
    --prompt "Dry ice turning to fog" \
    --question-id sublimation \
    --no-extract  # Will use default prompt "substance"
```

### 3. State Change Analysis Only

Analyze existing state change data to determine physics compliance:

```bash
python thermotics/state_analysis.py \
    --state-json thermotics/outputs/ice_melting/state_changes_ice_melting.json \
    --question-id ice_melting
```

**With frame downsampling (for large JSON files):**
```bash
# Analyze every 5th frame (reduces memory/computation)
python thermotics/state_analysis.py \
    --state-json thermotics/outputs/ice_melting/state_changes_ice_melting.json \
    --question-id ice_melting \
    --sample-rate 5
```

**Save analysis results:**
```bash
python thermotics/state_analysis.py \
    --state-json thermotics/outputs/sublimation/state_changes_sublimation.json \
    --question-id sublimation \
    --output thermotics/outputs/sublimation/analysis_result.json
```

## Output Structure

After running the integrated pipeline, outputs are organized as follows:

```
thermotics/outputs/
├── pipeline_result.json              # Complete pipeline summary
├── ice_melting/                       # Per-question results
│   ├── sam3_results/                 # SAM3 segmentation outputs
│   │   └── video.mp4                 # Segmented video
│   ├── state_changes_ice_melting.json # State change data
│   ├── visualization_ice_melting.mp4  # Visualization with metrics
│   └── result_ice_melting.json       # Question-specific results
├── water_evaporation/
│   └── ...
└── sublimation/
    └── ...
```

### Output File Formats

**`state_changes_{question_id}.json`**:
```json
{
  "state_changes": [
    {
      "object_id": 0,
      "label": "ice",
      "frames": [0, 1, 2, ...],
      "areas": [5000, 4800, 4500, ...],
      "centroids": [[320.5, 240.3], [321.2, 245.8], ...],
      "bboxes": [[300, 220, 340, 260], ...],
      "perimeters": [250.5, 248.2, ...],
      "compactness": [0.89, 0.87, ...],
      "num_components": [1, 1, 2, ...],
      "num_frames": 100,
      "area_change_rate": -50.2,
      "total_area_change": -35.5,
      "disappeared": false,
      "state_change_detected": true
    }
  ],
  "num_substances": 1,
  "note": "Tracks solid-liquid-gas state changes through area, shape, and fragmentation metrics"
}
```

**Metrics Explanation:**
- **`area_change_rate`**: Average area change per frame (pixels/frame)
  - Negative = shrinking (melting, evaporating)
  - Positive = expanding (freezing, condensing)
- **`total_area_change`**: Total percentage change from start to end
- **`disappeared`**: True if final area < 10% of initial area (complete evaporation/sublimation)
- **`state_change_detected`**: True if significant state change occurred (>20% area change OR disappeared OR fragmentation)
- **`compactness`**: Circularity metric (1.0 = perfect circle, lower = more irregular)
- **`num_components`**: Number of disconnected pieces (fragmentation indicator)

**`pipeline_result.json`**:
```json
{
  "video_path": "data/ice_melting.mp4",
  "video_prompt": "Ice melting in water",
  "filter_result": {
    "mechanics": [],
    "thermotics": ["ice_melting", "water_evaporation"],
    "material": []
  },
  "thermotics_results": [
    {
      "question_id": "ice_melting",
      "num_substances": 2,
      "state_changes": [
        {
          "label": "ice",
          "total_area_change": -45.3,
          "disappeared": false,
          "state_change_detected": true
        }
      ],
      "outputs": {
        "segmented_video": "...",
        "state_json": "...",
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

**`analysis_result.json`** (from state_analysis.py):
```json
{
  "question_id": "ice_melting",
  "compliant": true,
  "confidence": 0.95,
  "explanation": "Ice melting is physically plausible. Ice area decreased by 45.3% over 100 frames. Ice fragmented into smaller pieces as expected during melting. Water detected in video, consistent with melted ice.",
  "metrics": {
    "ice_detected": true,
    "area_trend": "decreasing",
    "total_area_change": -45.3,
    "disappeared": false,
    "fragmentation_increased": true,
    "num_frames": 100,
    "violations": []
  }
}
```

## State Change Analyses

The state analysis module (`state_analysis.py`) evaluates physics compliance for six phase transitions.

**Key Design**: Uses LLM-extracted substance labels (not keyword matching) to analyze state changes based on area trends.

### Melting (`question_id="melting"`)

Solid → Liquid transition (e.g., ice to water, wax melting).

**Analysis logic:**
- ✓ At least one substance should decrease (solid melting)
- ✓ >10% area decrease indicates expected melting
- ✓ Fragmentation is expected (solid breaking apart)
- ✓ Optionally, another substance may increase (liquid forming)

**Example:**
```bash
python thermotics/state_analysis.py \
    --state-json thermotics/outputs/melting/state_changes_melting.json \
    --question-id melting
```

### Sublimation (`question_id="sublimation"`)

Solid → Gas transition (e.g., dry ice to CO2 gas).

**Analysis logic:**
- ✓ Substance should rapidly decrease (>30%) or disappear
- ✓ Gas/vapor should appear or increase
- ✗ No intermediate liquid phase expected

### Vaporization (`question_id="vaporization"`)

Liquid → Gas transition (e.g., water evaporating, boiling).

**Analysis logic:**
- ✓ Substance should decrease (>5-15%) or disappear
- ✓ Vapor/gas may appear (increases confidence)
- ✗ Substance area increasing violates physics

### Condensation (`question_id="condensation"`)

Gas → Liquid transition (e.g., steam to water droplets, fog).

**Analysis logic:**
- ✓ At least one substance should increase (liquid forming)
- ✓ >15% area increase indicates condensation
- ✓ Fragmentation expected (liquid droplets forming)
- ✓ Optionally, gas may decrease

### Deposition (`question_id="deposition"`)

Gas → Solid transition (e.g., water vapor to frost, CO2 gas to dry ice).

**Analysis logic:**
- ✓ At least one substance should increase (>10-20%) - solid forming
- ✓ Gas may decrease
- ✗ No intermediate liquid phase expected

### Freezing (`question_id="freezing"`)

Liquid → Solid transition (e.g., water to ice, wax solidifying).

**Analysis logic:**
- ✓ Area may increase (e.g., ice expansion) or decrease (liquid contracting)
- ✓ >10% area change indicates freezing
- ✓ Compactness may increase (more regular shape)

## Visualization

The visualization videos use color-coded bounding boxes to indicate state changes:

- 🟢 **Green**: Stable (area change < ±20%)
- 🔵 **Blue**: Shrinking (area decrease > 20%)
- 🟠 **Orange**: Expanding (area increase > 20%)
- 🔴 **Red**: Disappeared (area < 10% of initial)

Each substance shows:
- Bounding box with color-coded state
- Current area (pixels)
- Total area change percentage

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
# In state_change_detection.py, segment_substances_sam3()
device='cuda:0',  # Change device here
```

### SAM3 Parameters

Adjust confidence and IoU thresholds:
```python
# In state_change_detection.py, segment_substances_sam3()
conf=0.4,  # Lower threshold for detecting fading substances
iou=0.7,   # IoU threshold
```

### State Change Detection Thresholds

```python
# In state_change_detection.py, after line ~350
# Adjust these thresholds:
disappeared = (final_area < initial_area * 0.1)  # 10% threshold
area_changed = abs(total_area_change) > 20  # 20% threshold
fragmentation_increased = (num_components[-1] > num_components[0] * 2)  # 2x threshold
```

## Examples

### Example 1: Ice Melting Analysis

```bash
# Video: Ice cubes in warm water
python thermotics/integrated_pipeline.py \
    --video data/ice_melting.mp4 \
    --prompt "Ice cubes melting in a glass of warm water"

# Expected output:
# - Detects "ice_melting" question
# - Segments "ice" and "water" substances
# - Tracks ice area decreasing over time
# - Shows negative area_change_rate
# Result: ice_melting compliance = YES
```

### Example 2: Water Evaporation Analysis

```bash
# Video: Water boiling
python thermotics/integrated_pipeline.py \
    --video data/boiling_water.mp4 \
    --prompt "Water boiling and producing steam"

# Expected output:
# - Detects "water_evaporation" question
# - Segments "water" (liquid) and "steam" (gas)
# - Tracks water area decreasing
# - May detect steam appearing
# Result: water_evaporation compliance = YES
```

### Example 3: Sublimation Analysis

```bash
# Video: Dry ice in water
python thermotics/integrated_pipeline.py \
    --video data/dry_ice.mp4 \
    --prompt "Dry ice sublimating in water, producing white fog"

# Expected output:
# - Detects "sublimation" question
# - Segments "dry ice" and "fog/vapor"
# - Tracks rapid area decrease of dry ice
# - Detects fog appearing
# Result: sublimation compliance = YES
```

### Example 4: Candle Wax Melting

```bash
# Video: Candle burning
python thermotics/integrated_pipeline.py \
    --video data/candle.mp4 \
    --prompt "Candle wax melting and dripping"

# Expected output:
# - Detects "thermal_state_change" question
# - Segments "wax" substance
# - Tracks shape changes (solid→liquid)
# - Detects wax flowing downward
# Result: thermal_state_change compliance = YES
```

## Integration with Main Pipeline

The thermotics pipeline integrates with the main LV-Bench 2 judge pipeline:

```python
# In judge_pipeline.py or custom script
from thermotics.integrated_pipeline import run_integrated_thermotics_pipeline

# 1. Filter thermotics questions
from problem.problem_filter import filter_problems_by_physics
filter_result = filter_problems_by_physics(video, model, prompt)

# 2. If thermotics questions exist, run state change detection
if filter_result.get("thermotics"):
    thermotics_result = run_integrated_thermotics_pipeline(
        video_path="video.mp4",
        video_prompt="...",
    )

# 3. Combine with LLM judge results
# Use state change metrics to augment physics evaluation
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

### No Substances Detected

```
Warning: No substances detected in video
```

**Solutions**:
1. Use custom prompts: `--custom-prompts "ice" "water" "steam"`
2. Lower confidence: Modify `conf=0.3` in code (default 0.4)
3. Check video quality and substance visibility
4. Ensure substances have clear boundaries

### Substance Extraction Failed

```
Error extracting substances from prompt
```

**Solution**: Use `--no-extract` and provide `--custom-prompts`:
```bash
python thermotics/state_change_detection.py \
    --video video.mp4 \
    --prompt "..." \
    --no-extract \
    --custom-prompts "ice" "water"
```

### No State Change Detected

```
state_change_detected: false
```

**Possible causes**:
1. Substance area change < 20% threshold
2. State change too subtle for detection
3. Video too short to capture full transition

**Solutions**:
1. Adjust thresholds in code (see Configuration section)
2. Use longer video with complete state transition
3. Check if substance was properly segmented

### Large JSON File / Memory Issues

```
MemoryError when loading state_changes_*.json
```

**Solution**: Use frame downsampling in state_analysis.py:
```bash
# Analyze every 5th frame instead of all frames
python thermotics/state_analysis.py \
    --state-json thermotics/outputs/ice_melting/state_changes_ice_melting.json \
    --question-id ice_melting \
    --sample-rate 5

# For very long videos, use higher sample rate (every 10th frame)
python thermotics/state_analysis.py \
    --state-json thermotics/outputs/water_evaporation/state_changes_water_evaporation.json \
    --question-id water_evaporation \
    --sample-rate 10
```

**Note**: Frame downsampling reduces memory usage and computation time while preserving the overall trend. A sample rate of 5-10 is typically sufficient for physics analysis.

## Technical Details

### State Change Detection Algorithm

The pipeline uses multiple metrics to detect state changes:

1. **Area Tracking**
   - Calculates total area (number of pixels) per frame
   - Computes area change rate and total percentage change
   - Detects expansion (freezing) vs shrinking (melting/evaporation)

2. **Shape Analysis**
   - Perimeter: Total boundary length
   - Compactness: 4π×area / perimeter² (1.0 = perfect circle)
   - Irregular shapes indicate state transitions

3. **Fragmentation Detection**
   - Counts number of disconnected regions
   - Increasing components → breaking apart (melting ice)
   - Decreasing components → coalescence (water droplets merging)

4. **Disappearance Detection**
   - Tracks if substance area drops below 10% of initial
   - Indicates complete phase transition (evaporation/sublimation)

### Physics Compliance Analysis Algorithm

The state_analysis.py module uses the following approach:

1. **Data Loading with Downsampling**
   - Loads JSON data from state_change_detection.py
   - Optionally downsamples frames (every Nth frame) for large datasets
   - Preserves overall trends while reducing memory usage

2. **Trend Analysis**
   - Computes linear regression on area values
   - Classifies trend as: "increasing", "decreasing", or "stable"
   - Used to detect expected vs unexpected behavior

3. **Question-Specific Checks**
   - Each question (ice_melting, water_evaporation, etc.) has custom physics rules
   - Checks for violations (e.g., ice expanding during melting)
   - Assigns confidence scores based on observed metrics

4. **Confidence Scoring**
   - Base confidence: 0.8-0.9
   - Adjusted based on evidence strength
   - Violations reduce confidence significantly
   - Additional evidence (e.g., vapor detection) increases confidence

5. **Result Generation**
   - Returns: compliant (bool), confidence (float), explanation (string), metrics (dict)
   - Compliant=True means physics is correct
   - Confidence ranges from 0.0 (certainly wrong) to 0.99 (very confident)

### Comparison with Mechanics Pipeline

| Metric | Mechanics | Thermotics |
|--------|-----------|------------|
| **Primary Data** | Centroid (x, y) | Area, shape, fragmentation |
| **Frame Analysis** | Position change | Area/shape change |
| **Movement Filter** | Min 20px displacement | N/A (tracks all substances) |
| **Physics Check** | Acceleration, velocity | Area change rate, disappearance |
| **Visualization** | Trajectory lines | Color-coded states |

## Future Enhancements

Planned improvements:

1. **Texture Analysis**: Detect state changes through texture/appearance changes
2. **Multi-phase Tracking**: Simultaneous tracking of all three phases (solid→liquid→gas)
3. **Temperature Inference**: Estimate relative temperature from visual cues
4. **Real-world Calibration**: Convert pixel measurements to actual volumes
5. **Temporal Smoothing**: Apply filtering to reduce noise in metrics
6. **Advanced Visualization**: Show area/shape graphs alongside video

## References

- SAM3: Segment Anything Model 3 (Ultralytics implementation)
- OpenCV: Computer vision library for image processing
- Thermal physics evaluation inspired by PhyGenBench methodology

## Comparison Table: Thermotics vs Mechanics

| Feature | Mechanics Pipeline | Thermotics Pipeline |
|---------|-------------------|---------------------|
| **Input Focus** | Moving objects | Substances changing state |
| **LLM Extraction** | "moving objects" | "substances undergoing state changes" |
| **Primary Metric** | Trajectory (position over time) | Area + shape over time |
| **Secondary Metrics** | Velocity, acceleration | Perimeter, compactness, fragmentation |
| **Filtering** | Movement threshold (20px) | No filtering (tracks all substances) |
| **Physics Questions** | gravity, buoyancy, impact, compression | ice_melting, water_evaporation, sublimation, thermal_state_change |
| **Visualization Color** | Green boxes only | Color-coded by state (green/blue/orange/red) |
| **Output Metrics** | Displacement, velocity | Area change rate, disappeared status |
| **Analysis Module** | trajectory_analysis.py | state_analysis.py |
| **Typical Use Case** | Ball falling, collision | Ice melting, steam rising |

## Quick Start Examples

### Complete Workflow: Ice Melting

```bash
# Step 1: Run detection and segmentation
python thermotics/integrated_pipeline.py \
    --video data/ice_melting.mp4 \
    --prompt "Ice cubes melting in warm water"

# Step 2: Analyze physics compliance
python thermotics/state_analysis.py \
    --state-json thermotics/outputs/ice_melting/state_changes_ice_melting.json \
    --question-id ice_melting \
    --output thermotics/outputs/ice_melting/analysis_result.json

# Expected output:
# Physics Compliant: YES ✓
# Confidence: 0.95
# Explanation: Ice melting is physically plausible...
```

### Complete Workflow: Water Evaporation (with downsampling)

```bash
# Step 1: Run detection (may produce large JSON for long videos)
python thermotics/state_change_detection.py \
    --video data/long_boiling_water.mp4 \
    --prompt "Water boiling for 5 minutes" \
    --question-id water_evaporation

# Step 2: Analyze with frame downsampling
python thermotics/state_analysis.py \
    --state-json thermotics/outputs/water_evaporation/state_changes_water_evaporation.json \
    --question-id water_evaporation \
    --sample-rate 10 \
    --output thermotics/outputs/water_evaporation/analysis_result.json

# Downsampling reduces processing time while preserving trends
```

---

For more information about the overall LV-Bench 2 system, see the main [README.md](../README.md) and [CLAUDE.md](../CLAUDE.md).
