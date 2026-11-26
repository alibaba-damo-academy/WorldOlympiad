# Single user evaluation: task to scored episode

This is a user-facing onboarding rehearsal for the configured two interfaces.
Use the configured Agent API and the configured tested-model adapter. Create
exactly one fresh task and carry it through scene generation, expert reference,
model episode, scoring, and report.

## Required workflow

1. Read `knowledge/benchmark.md` and follow the benchmark's task, scene, object,
   and DROID reference constraints. Preserve the benchmark prompts, physics
   checks, score logic, thresholds, metrics, and weights.
2. Use `author_task` to create exactly one new, simple, physically feasible
   pick-and-place case in this run. Do not reuse a case, scene, or reference
   from another run. Prefer a large, stable target without tight tolerances.
3. Submit the registered `scene` and `reference` stages with their declared
   dependencies. Inspect the generated scene and reference evidence and confirm
   the reference passes the source validation stages before continuing.
4. Submit `policy_check` and then one `episode` for the configured local model,
   using seed 0. Preserve the generated video and episode metadata. If the
   model cannot run within the current allocation, report the resource failure;
   do not reduce model requirements or borrow a GPU that has another process.
5. Submit `score` for the completed episode and `report` for this run. Verify
   that the report links the task, reference, model episode, and score.

## Internal RSI

The launcher runs the source Agent with engineering tools enabled. For a
scene-generation, reference, or pipeline infrastructure failure, use the
source RSI process: inspect the actual job logs and artifacts, identify a
general cause, make the smallest general repair in the agent worktree, add a
focused test, run the test, review the diff, promote, and resubmit the failed
stage. Do not patch one task's dimensions or hide a failed check. Do not ask an
external operator to repair or hand-edit generated assets. Keep the run's task
and results intact across RSI retries.

Use only registered pipeline stages for benchmark work. Do not submit any
additional cases or unrelated stages. Report each stage truthfully, separate
model failures from infrastructure failures, and finish only after the score
and report are verified.


## Automatic GPU retry

The configured tested-model GPU is `auto` and `policy_gpus` is unset. A GPU index chosen for a previous attempt is not a user pin. After an infrastructure failure caused by another process taking a GPU, first verify no evaluation job is active and run the registered evaluation preflight again. If it passes on currently idle devices, retry the same task, existing scene/reference, and seed 0 through registered stages, leaving GPU fields unspecified so the resolver chooses only within `WOV2_GPU_ALLOWLIST`. Never use a device outside that list, borrow a GPU with a compute process, lower resource thresholds, or bypass preflight. If fresh preflight fails, report and wait.
