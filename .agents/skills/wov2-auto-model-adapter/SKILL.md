---
name: wov2-auto-model-adapter
description: High-level self-evolving meta-adapter skill that automatically profiles any user-supplied embodied world model, selects embodiment-matched data from Chengzhiwww/WorldOlympiad2.0 via wov2-hf-dataset-router, synthesizes and validates a native or external wov2_adapter.v1 adapter, and updates the reusable adapter registry with verified experience. Use whenever onboarding or evaluating a new model, checkpoint, or API.
---

# WOV2 Self-Evolving Auto Model Adapter

This is the high-level orchestration skill for onboarding **any** new embodied world model into WorldOlympiad 2.0 without manual per-model glue engineering. Work from the `v2` repository root. Read `CLAUDE.md`, `docs/benchmark.md`, and [references/adapter-registry.md](references/adapter-registry.md) before synthesizing a new adapter.

## End-to-End Autonomous Onboarding Loop

### Phase 1 — Profile the Target Model from Evidence
Inspect the model's inference entry point, config, checkpoint/API revision, and environment to record a concise **Model Protocol Profile**:
1. **Embodiment & Arm Topology**: `dual` (`ALOHA` / `ARX5` / `AgileX`) vs. `single` (`Franka` / `Panda` / `DROID`).
2. **Execution Mode**:
   - `i2av` (direct RGB / image-to-video or joint video-action generation from initial frame(s) + instruction), or
   - `sim_rollout` (closed-loop policy emitting action chunks executed inside an interactive simulator).
3. **Conditioning Inputs**: required camera views (`front`, `left_wrist`, `right_wrist`), optional proprioception/state (`conditioning_state`), native FPS, input/output resolution `(W, H)`, and horizon.
4. **Hardware & Runtime**: Python env path, GPU count, `minimum_free_mib`, and any private API env var (kept in ignored `.secrets/` — never committed).

### Phase 2 — Route Compatible Benchmark Data (`wov2-hf-dataset-router`)
Never evaluate a model on an incompatible embodiment or execution domain. Chain directly to `.agents/skills/wov2-hf-dataset-router/SKILL.md`:
- For fast onboarding smoke tests, materialize 1–3 verified canonical constructed cases from `Chengzhiwww/WorldOlympiad2.0`:
  ```bash
  ./setup.sh dataset-route materialize \
    --arm <dual|single> --execution constructed_only --limit 2 \
    --out-dir outputs/datasets/<model>_smoke
  ./setup.sh benchmark-check --manifest outputs/datasets/<model>_smoke/dataset.json
  ```
- For full evaluation, route the full cohort matching the model's capability (`--execution sim_and_real` for hybrid sim+real evaluation — use `sim_rollout` where available, else `i2av`; `--execution sim_only` for closed-loop simulator-only policies such as GR00T).

### Phase 3 — Reuse Registry Archetype or Synthesize Adapter
1. Check [references/adapter-registry.md](references/adapter-registry.md) first:
   - If the model matches a **registered native adapter** (`lingbot_va`, `fastwam`, `longwam`, `gr00t`), reuse the native route directly—do not re-wrap it.
   - If it shares an archetype with a previously adapted model (e.g., single-arm DROID video predictor like Cosmos3, or dual-arm video-action wrapper), clone that verified template from the registry.
2. Otherwise, read [references/adapter-contract.md](references/adapter-contract.md) and synthesize a minimal model-owned wrapper + `wov2_adapter.v1` JSON under `outputs/adapters/<model>/` (outside tracked evaluator source unless upstream integration is requested):
   - Preserve native preprocessing, native FPS, and native output decoding.
   - Read **only** `{input}` conditioning (`instruction`, initial `conditioning` images, optional `conditioning_state`); **never** read `reference_video` or future frames.
   - Write generated MP4 + `prediction.json` (`{"video": "<relative>.mp4", "view": "<output_view>"}`) matching `output_size_wh`.
   - If the model requires state or simulator access absent from the selected cases, **stop and report the exact gap** rather than fabricating inputs.

### Phase 4 — Progressive Validation Gate
Run in strict order; report the highest stage actually passed:
1. **Contract & Dry-Run Check**: Validate manifest (`./setup.sh benchmark-check`) and test wrapper CLI + `prediction.json` schema with a lightweight fixture before allocating GPU time.
2. **1-Case Real Model Smoke Run**: Run `./setup.sh benchmark-run --manifest <smoke>/dataset.json --adapter <adapter.json> --run <run_id> --generation-only` on an allowlisted idle GPU. Decode the output video and verify resolution, frame count, native FPS, view alignment, and non-trivial motion.
3. **Full Three-Track + CAE Scoring**: Only after real generation passes, run scoring (`benchmark-run` / `benchmark-score`) and inspect Physical ($P$), Geometry ($G$, $S_{\mathrm{path}}$), and Interaction ($I$, $S_{\mathrm{vel}}$) outputs.

### Phase 5 — Self-Evolving Skill Update (Persist Learned Experience)
Whenever a new model passes Phase 4 (or reveals a concrete protocol incompatibility), **update [references/adapter-registry.md](references/adapter-registry.md)** with a structured entry:
- Model family, revision/checkpoint identifier, embodiment (`dual`/`single`), execution mode (`i2av`/`sim_rollout`), native FPS & resolution.
- Exact camera/coordinate/action conversions and pitfalls resolved during debugging.
- Reusable `wov2_adapter.v1` command pattern and highest validated gate.

This turns every manual or agent-assisted adaptation into permanent, reusable knowledge so future models in the same family onboard automatically.
