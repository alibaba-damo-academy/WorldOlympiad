# Single user evaluation: create a fresh task and expert reference

This run is a user-facing onboarding rehearsal. The configured tested-model adapter is already registered. Use the upper Agent API to create exactly one new, simple, physically feasible task, then generate its SceneCode scene and a successful expert reference.

## Required stages

1. Read `knowledge/benchmark.md` and follow its task, scene, and DROID reference constraints.
2. Use `author_task` to create one new case in this run. Do not reuse a task, scene, or reference from another run. Prefer a basic single-object pick-and-place with a large, stable target and no tight geometric tolerances.
3. Submit the `scene` stage for that case, then submit `reference` with the scene job as a dependency. Inspect both job results and verify the fresh reference artifacts.
4. Stop after the reference is verified. Do not submit `policy_check`, `episode`, `score`, `report`, or `showcase` jobs in this goal.

The launcher will perform a separate tested-model resource check after the reference completes. It will report whether the configured GPU allocation can proceed to an episode.

Only after the new task, scene, and successful reference are verified, end the final response with this exact marker on its own final line:

`USER_REFERENCE_READY`
