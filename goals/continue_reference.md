# Continue the same fresh user evaluation to native scoring

The preparation phase already authored exactly one task in this run and produced its SceneCode scene and successful expert reference. Do not call `author_task`, create another case, regenerate the scene, or replace the reference. Use only the existing task and verified reference in this run.

Follow `code/WorldOlympiad-main/wo/agent/knowledge/benchmark.md`. Keep the registered LingBot-VA adapter and RSI engineering tools enabled. Use only registered stages and preserve every prior artifact. Confirm a successful registered `scene_check` for the existing reference before submitting the stock `policy_check` for the existing task, then one native LingBot-VA episode with seed 0, then scoring and the run report. Keep declared dependencies. Inspect each result and distinguish infrastructure failures from a measured model outcome.

For an adapter or pipeline infrastructure failure, RSI must inspect the actual log/artifacts, make the smallest general repair in its engineering worktree, add or extend focused tests, run them, review the diff, promote, and then retry the failed registered stage. Repeated repair/test/promote cycles are allowed; do not interpret the source's 60-turn-per-Agent-call bound as a total retry count. Each registered `policy_check` has a 30-minute job timeout. Do not lower resource thresholds, mask the adapter, or bypass resource preflight.

If a complete native episode returns `task_success=false`, preserve and score it as a valid measured result. Do not report infrastructure failure in place of a native model result. Finish only after the report links the same task, reference, episode and score.

Do not submit additional tasks, showcase jobs, scene/reference retries for unrelated cases, or any job outside this run's requested stages. Use only GPUs in the configured WOV2_GPU_ALLOWLIST.
