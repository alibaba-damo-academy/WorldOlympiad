# Registered native model routing (2026-11-25)

The benchmark command adapter remains the route for a model that generates video
from initial inputs but has no compatible registered simulator. It is independent
of LingBot's tested-model profile. `benchmark-run --model` selects a registered
built-in model; its configured tested-model identity must match the request.

FastWAM and Long-WAM's registered RoboTwin paths now archive the verified WOV2
TaskSpec and frame-zero reference state, initialize the source-owned native
scene, and use each model's 14-D qpos policy protocol. Model and simulator
CUDA placement is selected by the resource gate. The resulting trajectory and
capture are bound to the reference before the baseline-view replay is scored.
The resource gate selects one idle allowlisted card when its free-memory
headroom covers both model and simulator thresholds; otherwise it selects a
second eligible idle card. An explicit simulator card is checked for its
selected topology. The default `simulator_gpu: auto` requires no manual card
assignment.
For qpos policies, a short `--window-s` bounds measured SAPIEN physics time
(the current action finishes before the bound is checked again). Native
trajectory frames sample real simulator state at roughly 15 Hz plus action
boundaries, with measured timestamps. The raw observation videos still sample
one frame per completed policy action and are diagnostic captures; scoring
uses the baseline-camera replay of the denser native trajectory.
The scene loader currently supports the stock `stack_blocks_two` native task
with a compatible single WOV2 `on` or `stacked` goal. This is a narrower
capability than arbitrary `sim` video input.

## Evidence

- 49 focused native adapter, qpos action, GPU routing, reference-scene and
  LingBot regression tests passed on the candidate. The 25 benchmark contract
  tests also passed. These tests do not execute a GPU model.
- On the 6000D host, a preserved WOV2 apple-on-tray reference passed certificate,
  TaskSpec and source-bound frame-zero archive checks in the isolated test
  workspace. The source-owned bridge command resolved to the configured
  RoboTwin Python. No source or checkpoint in the senior workspace was changed;
  model source was copied and weights were linked read-only.
- Long-WAM's official RoboTwin IDM entry point imported in the existing
  FastWAM Python environment, but this does not prove model loading or rollout.
- At the last hardware check, every 6000D GPU had another user's compute
  process. The project requires an allowlisted, idle card at stage admission.
  Consequently neither FastWAM nor Long-WAM has a new native GPU rollout or
  complete benchmark score from this patch yet. Do not describe the native
  route as end-to-end validated until both short reference-bound runs and
  their output videos complete on eligible hardware.

The isolated test workspace is
a local isolated validation workspace (not included in this release).
It preserves its configuration, binding probe and future attempt logs; it is
not a release artifact.

## Pending native validation for handoff

The earlier FastWAM 9-frame and Long-WAM COD 33-frame prediction videos came
from external video-generation adapters. Their predicted actions were not
executed in RoboTwin. The registered Long-WAM native path uses the IDM
checkpoint, which is distinct from the COD checkpoint used for that video.

For each registered FastWAM and Long-WAM IDM `sim_rollout`, a reviewer still
needs one short GPU run against a certified WOV2 reference. Check that policy
inference receives live simulator observations, qpos actions advance RoboTwin,
the recorded trajectory has increasing physical timestamps, baseline-camera
replay produces the scored video, and the episode/score receipts bind to the
same reference. A video-only benchmark run does not establish native task
success or native trajectory CAE.
