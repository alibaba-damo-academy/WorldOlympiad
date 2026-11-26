---
name: wov2-benchmark-pipeline
description: Master orchestration skill for the WorldOlympiad 2.0 pipeline: (1) synthesizing & repairing physics-verified tasks via SceneCode + RSI, (2) selecting & evaluating built-in adapted models (lingbot_va, fastwam, longwam, cosmos3, gr00t), (3) evaluating on public HuggingFace WorldOlympiad2.0 data, and (4) onboarding new models. Use whenever the user asks to generate synthetic tasks, run evaluations, switch models, or execute end-to-end benchmark runs.
---

# WorldOlympiad 2.0 Pipeline Skill

Work from the `v2` repository root and follow `CLAUDE.md`. Always verify `WOV2_GPU_ALLOWLIST` is explicitly set by the user (or ask which GPU indices to authorize) before starting any GPU stage.

## Prerequisites Check

1. `config/two-interface.json` must exist for autonomous Harness runs (`./setup.sh run`) and built-in native adapters (`--model`). Create or update it with:
   ```bash
   ./setup.sh configure \
     --replace \
     --agent-base-url <HTTPS_URL_ENDING_IN_/v1> \
     --agent-model <GPT_MODEL_ID> \
     --tested-model <lingbot_va|fastwam|longwam|cosmos3|gr00t> \
     --checkpoint </abs/path/to/checkpoint>
   ```
2. `.secrets/scenecode.env` (mode `0600`) must contain `WO_AGENT_API_KEY` (set via `./setup.sh set-api-key`, never echoed or read into logs).

---

## Workflow 1 — Synthesize & Verify New Physics Tasks (Agent + SceneCode + RSI)

When the user asks to construct synthetic manipulation tasks (e.g., cabinet articulation, cup hanging, bottle uprighting, pick-and-place, tool use):

1. If the user wants a custom task family or task count, write a clean Goal Markdown file under `outputs/evaluator/custom_goals/<run_id>.md` following the rules in `code/WorldOlympiad-main/wo/agent/knowledge/benchmark.md` (ending with `USER_REFERENCE_READY` on its own final line for preparation-only runs).
2. Run preflight and launch `--phase prepare`:
   ```bash
   export WOV2_GPU_ALLOWLIST=<GPUS>
   ./setup.sh preflight-preparation
   # Default 1-task onboarding reference:
   ./setup.sh run --phase prepare --run <RUN_ID>
   # Or custom multi-task / family-specific goal:
   ./setup.sh run --phase prepare --run <RUN_ID> --goal-template outputs/evaluator/custom_goals/<run_id>.md
   ```
3. Inside `--phase prepare`, the GPT Benchmark Agent (`wo.agent.run`) autonomously executes `author_task` → `scene` (SceneCode) → `reference` (PyBullet/SAPIEN physics + Blender replay). If an asset collision or predicate bug occurs, its built-in RSI loop (`inspect_scene`, `view_image`, worktree `edit_file`, `run_tests`, `promote`) repairs the general capability before finishing.

---

## Workflow 2 — Select & Evaluate Built-In Adapted Models

Five models have built-in native adapters (`--tested-model`):
- Dual-arm (`ALOHA` / `ARX5`): `lingbot_va`, `fastwam`, `longwam`
- Single-arm (`Panda` / `DROID`): `cosmos3`, `gr00t` (`GR00T N1.6-DROID`, sim-only)

1. Point `config/two-interface.json` to the chosen `--tested-model` and `--checkpoint` (`./setup.sh configure --replace ...`).
2. To evaluate on tasks prepared in Workflow 1:
   ```bash
   export WOV2_GPU_ALLOWLIST=<GPUS>
   ./setup.sh preflight-evaluation
   ./setup.sh preflight-scoring
   ./setup.sh run --phase continue --run <RUN_ID>
   ```
   *(Or run synthesis + native rollout + three-track/CAE scoring in one pass via `./setup.sh run --phase full --run <RUN_ID>`.)*
3. Inspect final artifacts under `outputs/wo_runs/<RUN_ID>/` (`score.json`, `three_track_receipt.json`, `results.json`).

---

## Workflow 3 — Evaluate on Public HuggingFace Benchmark Data (`Chengzhiwww/WorldOlympiad2.0`)

Chain to [`.agents/skills/wov2-hf-dataset-router/SKILL.md`](../wov2-hf-dataset-router/SKILL.md):
1. Select `--arm dual|single` and `--execution sim_and_real` (default for all models except `GR00T N1.6-DROID`, which uses `--execution sim_only`: run `sim_rollout` whenever simulation is available, else `i2av`).
2. Materialize cases (`./setup.sh dataset-route materialize ...`) and verify with `./setup.sh benchmark-check --manifest <dataset.json>`.
3. Run evaluation via `./setup.sh benchmark-run` (with `--model` or `--adapter`) or score pre-generated videos via `./setup.sh benchmark-score --manifest <dataset.json> --predictions <predictions.json> --run <RUN_ID>`.

---

## Workflow 4 — Onboard & Auto-Adapt Any New Model

Chain directly to [`.agents/skills/wov2-auto-model-adapter/SKILL.md`](../wov2-auto-model-adapter/SKILL.md):
1. Profile the new model's real observation/action/video protocol from its codebase.
2. Route matched benchmark smoke & full cohorts via `wov2-hf-dataset-router`.
3. Synthesize `outputs/adapters/<model>/adapter.json` (`wov2_adapter.v1`), pass `--generation-only` smoke check, run full three-track + CAE scoring, and append the verified pattern to `.agents/skills/wov2-auto-model-adapter/references/adapter-registry.md`.
