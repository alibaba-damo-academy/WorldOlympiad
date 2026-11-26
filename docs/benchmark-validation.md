# Benchmark interface validation

Initial scoring-interface validation was completed on 2026-11-11 using an RTX 6000D and NVIDIA driver 580.173.02, at implementation e34f51acbb93fbdc72770564a1f7571b59a918d3. This page preserves that snapshot; later code and end-to-end evidence are recorded in release-validation-plan.md.

## Accepted evidence

- 19 focused tests passed. Shell syntax and whitespace checks passed.
- Actual public case AIRBOT_MMK2_boxs_storage-observation.images.cam_high_rgb-episode_000000 used three real initial camera images and the configured LingBot-VA posttrain Robotwin checkpoint. Existing audited environments, weights and source data were reused; no model weights were downloaded.
- Native i2av generation completed in run lingbot_real_i2av_smoke_20261111b. Its adapter and locked source hashes match the accepted candidate. The selected head-camera output fully decoded as 5 frames, 10 FPS, 720 by 540 pixels, 0.5 seconds.
- The generated video was submitted through benchmark-score in run lingbot_real_i2av_score_smoke_20261111d. The command exited 0 and report.complete is true. Physical = 0.95, Geometry = 0.6548, video semantic Interaction = 0.20, equal-track overall = 0.6016.
- This is the explicitly declared video-only protocol. Native trajectory CAE and simulator task success are not applicable. The native raw score, including its native CAE pending state, remains unchanged; benchmark_score.json records the video-protocol composition and raw receipt hashes.
- Physical rule evidence, nonempty geometry artifacts, interaction judgments and complete CLIP chunk coverage were checked before declaring completion. A failed or missing judge is not a zero score.
- Source reference and prediction hashes remained unchanged. Evaluator intermediates were generated beside private regular video copies, not in the source dataset.
- Repeating the completed benchmark-score command exited 0, produced no additional judge stages and left the report and raw score unchanged.
- The working tree was clean. The source archive was verified against the implementation files and excludes environments, weights, private configuration and run data.

Runtime evidence is retained under outputs/benchmark/interface_validation_20261111/final-validation.json. The complete scored report is outputs/benchmark/lingbot_real_i2av_score_smoke_20261111d/report.json. Generation and scoring were validated here as the two-step workflow. The later fresh combined benchmark-run and sim_rollout results are recorded in release-validation-plan.md.

## Validation boundary

This initial test was a short integration check, not a model-quality result or a completed public benchmark. As of 2026-11-14, the fresh real/i2av command, sim_rollout command and two-case real/sim batch have passed on the RTX 6000D; see release-validation-plan.md for exact versions and receipts. A fresh 5090 regression and clean online installation remain separate validation tasks. A lab-specific official sample list is not a release requirement for this user-data interface. Existing runtime reuse is the installation path exercised here.

Earlier timeout, host-reboot interruption and failed scoring attempts remain in the runtime history. They are not counted as successful runs. The preserved original native RSI workflow remains separate from this video protocol.
