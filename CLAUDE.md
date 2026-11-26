# wov2-evaluator project rules

## Purpose

This is the user-facing entry repository for a reproducible WOV2 task-to-report evaluation. Preserve the behavior and provenance of the server-recovered WorldOlympiad pipeline; do not reconstruct a guessed evaluator.

## Layout

- Track the entry scripts, locks, docs, tests, and immutable per-file-manifest source snapshots under `code/`.
- Keep reusable user-facing Codex skills under `.agents/skills/<kebab-case-name>/SKILL.md`; put conditional detail in that skill's `references/`. Skills describe supported contracts and evidence, not private runtimes, credentials, weights, or generated adapters.
- `sources.lock.json` records the upstream URL/commit or archive provenance for every source snapshot. The DROID bridge is an explicitly provenance-limited bundle until its upstream source is found.
- `.secrets/`, `weights/`, `envs/`, generated `runtime/`, `cache/`, `outputs/`, and `tmp/` are runtime data and never enter Git.
- Preserve run outputs, round traces, job DBs, logs, videos, reports, shared runtime dependencies, and weights. Cleanup is a separate approved task.

## Source and environment setup

- Validate every source manifest before materialization or runtime staging. Never fetch floating branches, guess a revision, overwrite a divergent path, or discard a failed partial.
- Reuse existing benchmark weights and user-owned checkpoints in place. The tested-model checkpoint is never fetched or replaced. Missing or gated assets are reported; downloads require the explicit user command.
- Reuse existing server environments only through the audited link script. Do not mutate shared runtime directories. RoboTwin data assets are linked or downloaded into evaluator-owned paths, not copied into source.
- Create/reuse project-local environments only. RoboTwin and LingBot-VA use separate Python environments. Distinguish path presence from import/version probes, runtime markers, hashes, weight contracts, and fresh GPU preflight.
- SceneCode profiles are generated per run from the Agent API model and passed only from a path inside that run. Do not add secrets to profiles or logs.

## API, adapter and RSI

- The upper Agent API and local tested model are separate interfaces. Never put API keys in JSON, source files, logs, or commits. Load a private key environment file with mode `0600`; do not echo or inspect its value.
- Keep the registered tested-model adapter and RSI edit/test/promote tools enabled in user-loop runs. Do not pass `--no-engineering`.
- RSI repairs must be general, minimal, focused-tested, reviewed, and promoted. Preserve patch diffs and test/validation receipts. A focused test is not an end-to-end result.
- Keep each registered `policy_check` timeout at 30 minutes. A per-call Agent turn bound is not a total retry limit. Do not repeat an infrastructure-blocked job without diagnosing the blocker.

## GPU and queue

- This migration supports NVIDIA CUDA GPUs. A job requires an explicit comma-separated WOV2_GPU_ALLOWLIST of local nvidia-smi indices; fail closed when it is missing or malformed. Query and use only indices in that list.
- Resource preflight checks per-stage free-memory thresholds, utilization, compute PIDs, and required card topology. It may make a deterministic choice only within the user allowlist; it must not scan, preempt, or schedule outside that list.
- Blender rendering uses the host EGL stack independently of the CUDA GPU allowlist. Prefer a readable NVIDIA GLVND EGL vendor JSON for replay when available; otherwise preserve host vendor discovery. Document device permissions and verify the renderer on the target host.
- LingBot on GPUs up to 40 GiB requires two policy cards and a distinct simulator card; larger-memory topology follows the adapter's explicit contract. Enforce these thresholds.
- Run at most one evaluation job at a time. Check the queue and compute-app PIDs. Never stop or signal another workspace's process.
- Recheck resources immediately before each stage. If availability changes, stop only this run's process tree and preserve its artifacts and job history.

## Release and credentials

- Keep source commits, bundle manifests, runtime locks, config schema, and accepted evidence in the repository.
- The recovered WorldOlympiad bundle has no upstream `.git`; initialize the locked synthetic RSI baseline and isolated worktree with `./setup.sh rsi-ready`. Preserve its source manifest and promotion history; keep weights/DA3/outputs runtime links out of commits.
- Never commit secrets, weights, runtime binaries, caches, or run outputs. Never push or publish without explicit user authorization.

## User-supplied benchmark interface

- Store orchestration in scripts/benchmark*.py and its contract in docs/benchmark.md. Runtime manifests and results belong under outputs/benchmark/.
- Reference real/sim provenance is distinct from execution mode i2av/sim_rollout.
- Video adapters receive conditioning only. Future reference frames and evaluator captions stay on the scoring side.
- Bind dataset version, model, IDs, views and hashes before execution. Preserve partial attempts and reject changed completed inputs.
- Reuse existing scorer and GPU gates. Video-only reports cannot claim native task success or native trajectory CAE.
- Validate adapter output decoding and scoring handoff. Model execution and complete scoring require separate runtime evidence.

- Stage evaluator-owned regular video copies before scoring. Recovered scorers write adjacent caches; they must never resolve back into a read-only dataset or model-output tree.
- Freeze video-only applicability in run inputs before execution. Validate actual judge evidence before aggregation; preserve the native raw score and receipt. Native trajectory requirements remain part of the existing native protocol.

- Dataset membership is supplied by the caller manifest and frozen per run. Lab-specific or paper-specific sample lists are not required repository contents or a prerequisite for releasing the user-facing evaluator.
