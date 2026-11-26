---
name: wov2-hf-dataset-router
description: Route, filter, and materialize embodiment- and protocol-matched benchmark cases from the HuggingFace dataset Chengzhiwww/WorldOlympiad2.0 for any embodied world model (dual-arm vs. single-arm, sim-only vs. real+sim, constructed vs. public). Use whenever preparing benchmark data for a new or existing model evaluation.
---

# WorldOlympiad 2.0 HuggingFace Dataset Router

Official dataset repository: `https://huggingface.co/datasets/Chengzhiwww/WorldOlympiad2.0`

Use this skill **before** running `benchmark-check` or `benchmark-run` so every evaluated model receives **only** cases that match its robot embodiment, camera topology, and execution capability (never sending dual-arm ALOHA tasks to a single-arm Panda policy, or real-world videos to a simulator-only policy).

## 1. Classify the Model First

Determine two axes from the model's config, paper, or checkpoint:

1. **Arm Topology (`--arm`)**:
   - `dual`: Bimanual models (`ALOHA`, `ALOHA-AgileX`, `AgileX Cobot Magic`, `ARX5`) — e.g. `LingBot-VA`, `Fast-WAM`, `Long-WAM`.
   - `single`: Single-arm models (`Franka`, `Panda`, `DROID`) — e.g. `Cosmos3`, `GR00T N1.6-DROID`.
2. **Execution Capability (`--execution`)**:
   - `sim_and_real` *(default — alias `video_all`)*: **Hybrid simulation + real-world** — used by **all embodied world models except sim-only policies like `GR00T N1.6-DROID`** (e.g. `LingBot-VA`, `Fast-WAM`, `Long-WAM`, `Cosmos3`, and general new models). Rule: **run native `sim_rollout` whenever a runnable simulator exists (`domain=sim`), and fall back to `i2av` / video-conditioned rollout on real-world or non-executable videos (`domain=real`)**.
   - `sim_only`: Closed-loop policies that execute **only** inside a simulator environment (`sim_rollout`, e.g. `GR00T N1.6-DROID`) — excludes real-world recordings.
   - `constructed_only`: Direct in-repository canonical MP4 tasks (`Constructed/ARX5` or `Constructed/Panda`, 100 each) with verified `expected_reference_sha256` and zero upstream download dependencies — ideal for fast smoke tests and full constructed evaluation.
   - `real_only`: Real-world public benchmarks (`Table30`, `RoboCOIN`, `RoboMIND`) or the separate 300-episode bimanual `real_robot/` suite.

See [references/dataset-routing-matrix.md](references/dataset-routing-matrix.md) for the exact 1,100-case cohort breakdown and published model splits.

## 2. Route & Materialize via CLI

```bash
# 1. Inspect the full embodiment / domain / source breakdown from HuggingFace
./setup.sh dataset-route summary

# 2. Route hybrid sim + real cases for a dual-arm embodied world model (use sim when available, else real/i2av)
./setup.sh dataset-route route --arm dual --execution sim_and_real --out /tmp/dual_route.json

# 3. Route cases for a single-arm sim-only policy like GR00T (excludes real-world cases)
./setup.sh dataset-route route --arm single --execution sim_only --out /tmp/single_sim.json

# 4. Materialize verified constructed cases directly into a benchmark-ready wov2_benchmark.v1 manifest
./setup.sh dataset-route materialize \
  --arm dual --execution constructed_only --limit 5 \
  --out-dir outputs/datasets/dual_smoke

# 5. Validate the materialized manifest with the evaluator contract
./setup.sh benchmark-check --manifest outputs/datasets/dual_smoke/dataset.json
```

## 3. Strict Evaluation Boundary

- `materialize` extracts **only frame 0** (`initial_front.png`) for model conditioning and verifies `expected_reference_sha256` on every downloaded `reference.mp4`.
- **Never** pass `reference.mp4` or frames after `t = 0` into a model adapter.
- For public upstream-indexed cases (`RoboDojo`, `ManiSkill3`, `Table30`, `RoboCOIN`, `RoboMIND`), `materialize` writes `upstream_pending.json` with pinned `upstream_repo`, `upstream_revision`, and `upstream_media` pointers so the agent can fetch only the required upstream episodes cleanly.
- For the 300 newly collected bimanual real-robot demos (`real_robot/episodes.jsonl`), use the single-view byte-range reader `tools/fetch_real_video.py` from `Chengzhiwww/WorldOlympiad2.0`.
