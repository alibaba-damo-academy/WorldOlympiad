# Historical recovery record: `lingbot_reference_20261101c`

Recorded (UTC): 2026-11-01T14:12:06Z

## Source and environment

- Candidate source root: `code/WorldOlympiad-main (candidate checkout; host path omitted)`.
- Current exact WorldOlympiad tree SHA-256: `bf22d093fa31a58a30f29d298aeab72363c486fc62fd124e2acf4a2edfac3fa5`; manifest SHA-256: `725a1b700f7da834415315332c76b697724f5d1cc6a580c569fe5f218771bfb1`.
- No old branch/workspace was rewritten. The preserved workspace Git repository was unborn with no refs or recoverable commit objects. The old source lock had declared `2b8309c3f072ae01bafc4fcaa694475a43e25161`; this is retained only as `previous_lock_commit_claim`, not asserted as verified history.
- Candidate reuses server Python environments through symlinks and the existing robot assets/weights. No weights or checkpoint were fetched.

## Recovered run evidence

- Run root: `historical server run workspace (absolute path omitted)`.
- Scene job #13: `done`; reference job #16: `done`.
- Native episode job #30: `done`, task_success `true`; it is hash-bound to `tasks/apple_to_coaster_onboarding_01.json` and the matching task graph / archived TaskSpec. TaskSpec SHA-256 and native trajectory `task_sha256` are both `1a41da4b6b6140dde705aa2354fd325b0b00ab0b25145a830e68244b9a4df3bf`.
- The user goal is `on(apple, coaster)`. The source adapter intentionally maps that exact two-object goal onto RoboTwin's `stack_blocks_two` control environment, with `apple` and `coaster` bound by ID and a native WOV2 contact/geometry predicate. The predicate evidence is true (`touching=true`, positive xy overlap); the control class name alone does not imply task mismatch.
- Episode receipt: `apple_to_coaster_onboarding_01/lingbot_va_s0/episode_reference_binding.json`; native receipt: `apple_to_coaster_onboarding_01/lingbot_va_s0/episode.json`; trajectory: `apple_to_coaster_onboarding_01/lingbot_va_s0/native_run/simulation/trajectory.json`.
- Videos: `apple_to_coaster_onboarding_01/lingbot_va_s0/capture/head.mp4`, `left_wrist.mp4`, and `right_wrist.mp4`. Receipt files and all reference-bound digests were verified on the server.
- Score jobs #31–33 failed during scorer/VLM startup or scoring subprocess execution. Earlier score #26 and report #28 belong to the older episode and are stale for #30. Fresh score and report-v3 are still required.

## Current resource gate

Latest fresh scoring preflight used the final candidate source/config and preserved original run database; it queried only RTX 5090 indices 0–6. Evidence: `historical candidate outputs/evaluator/resource_preflight/scoring-1793542326561732837.json (checkout path omitted)` (2026-11-01T14:12:06Z). Split scoring requires two distinct idle cards, each with at least 25,600 MiB free, utilization ≤5%, and no compute-app PID; eligibility was 0/2. GPUs 0 and 1 each had 32,107 MiB free but 100% utilization; GPUs 2–6 failed the utilization, compute-PID, and/or free-memory gate. The original run database had no queued/running jobs. The preflight marked the run blocked and submitted no score job.

Score jobs #31–33 all failed; each failure is retained in the original run workspace. Job #31's Qwen-VL warmup loaded the checkpoint but then OOMed with 9.5 MiB free while another process occupied 18.41 GiB. Job #32 also failed during Qwen-VL startup under GPU contention. Job #33 reached physical scoring, then SAM3 OOMed with 12.12 MiB free while other processes occupied the card. These are infrastructure resource failures, not evidence of an API or adapter failure. Current candidate code gates split scoring on the full two-card topology and rechecks before launch.

`results_v3.json` and `.md` are timestamped 2026-11-01T10:42:51Z, before episode job #30 finished at 11:35:52Z; they are stale for the successful #30 episode. The current episode case has no `three_track_receipt.json` or `score.json`, so a fresh score and report-v3 remain required.

The candidate separates lightweight SceneCode/reference admission (one idle card, ≥8,192 MiB free) from LingBot inference admission (two policy cards plus a distinct simulator card). Split scoring places Qwen-VL and physical/interaction/geometry workloads on distinct cards.

## Promoted candidate source revision

- RSI base commit: `1c36114555bb959064a0e0a2be22f31d319d35b5`, deterministically generated from the exact locked source snapshot; the original nested Git history was unavailable and is not claimed. The candidate RSI worktree is clean and based on this revision.
- The clean-clone source check exposed a cuRobo asset omitted by upstream `*.zip` ignore rules; `srl_ur10_bins.zip` is now included and verified against its manifest SHA-256.
- Regular-file modes in source manifests now match Git checkout semantics (0644/0755). A second clean clone verified all seven source bundles, initialized the RSI worktree without changing the tracked lock, and retained a clean outer Git worktree.
- Focused resource, scoring, source-manifest, RSI preparation, and config-path regression suite: 65 passed.
- A fresh clone at this commit reused the existing runtime/weights, staged source-bound runtime overlays, passed all five environment import/version probes and RoboTwin asset checks, and verified all seven source bundles. Active non-secret config uses the user-provided endpoint/model and existing checkpoint; mode is 0600. No API key was read, copied, or stored.

## Imported episode provenance

- The final candidate has an integrity snapshot at `outputs/wo_runs/lingbot_reference_20261101c/recovery_source_manifest.json` (2,250 files; 635,789,940 bytes), copied from the preserved source run with its old score `case/` excluded.
- The original reference certificate and episode receipt contain absolute paths to the original workspace. Relocated-copy binding checks therefore reject the copy as a scoring input. The copied evidence is preserved as a hash-verified snapshot only; the original reference and episode paths remain authoritative for scoring. Their receipts were not rewritten.
- The original episode's current `case/` has no valid three-track receipt or score. The candidate score stage archives that incomplete case under `.score_history` before rebuilding, preserving prior failure evidence.
- The final clone has no `.secrets/scenecode.env`; no API key was copied or read. The current score/report stages need no Agent API call. A fresh task/SceneCode/RSI launch requires the user to enter the key through `scripts/set_api_key.py`.

## Candidate deployment checks

- Candidate checkout: `candidate checkout (host path omitted)`; source and runtime setup passed in a fresh clone, separate from the preserved original run.
- The root preparation script creates its ignored `outputs/` target before adding the RSI worktree link, and rejects occupied paths without modifying them. Config path resolution preserves venv symlink entrypoints instead of collapsing them to the underlying Python binary.

## Completion state

The reference and native episode are verified; the score/report gates remain open. Existing job/scene/reference/episode evidence is preserved in the original workspace and linked from this record. A later successful score must consume episode #30's artifact and then regenerate report-v3; do not rerun the native episode unless the bound output is invalidated or a diagnosed infrastructure condition requires it.
