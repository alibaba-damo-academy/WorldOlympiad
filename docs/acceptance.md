# Acceptance gates

A complete user-loop acceptance requires the following evidence for the same run ID, target case, model, and seed.

1. **Source:** `sources.lock.json` and all file/mode manifests verify. Each bundle states whether its upstream history is actually recoverable; a snapshot hash is not presented as a Git commit.
2. **Runtime/assets:** required source-bound runtime copies and import/version probes pass. Existing shared Python environments, Blender, robot assets, and benchmark assets are linked read-only/reused; clean-room installation is a separate gate.
3. **Weights/checkpoint:** the auxiliary weight checker validates existing shards/indexes without redownloading. Missing or gated weights are reported with the required user action. The local tested-model checkpoint passes its contract and is not fetched.
4. **Two interfaces:** Agent base URL and model ID are configured; a key is present only in the private environment file and never printed. The tested-model checkpoint/runtime is separately configured. API configuration alone is not evidence of generation.
5. **Task/SceneCode/RSI:** built-in task selection and RSI engineering remain enabled. SceneCode generation and its own internal repair produce a task-bound scene certificate. Every RSI repair is tested and promoted by the in-flow process.
6. **Reference:** a successful native expert trajectory and required reference video exist and match the source task spec/certificate.
   Scoring videos must use the exact reference view required by the baseline; a mismatched native `head` stream blocks scoring and cannot be relabeled as `exterior_left`.
7. **Resource/queue:** CUDA queries are restricted to the explicitly configured WOV2_GPU_ALLOWLIST of local NVIDIA indices. Preflight records the allowlisted per-card free/total memory, utilization, compute-app PIDs, selected indices, and phase. Preparation/evaluation require a readable NVIDIA EGL vendor JSON; the server must separately grant the process access to the NVIDIA/render devices needed by its driver stack and verify actual NVIDIA rendering. No physical GPU index is reserved for EGL. Episode admission requires two policy GPUs (≥30,720 MiB each) and one distinct simulator GPU (≥25,600 MiB); split scoring requires two distinct GPUs (≥25,600 MiB each). Eligible CUDA cards have utilization ≤5% and no compute PID. Fixture tests prove insufficient topologies block before job insertion. At most one evaluation job is active.
8. **Native episode:** receipt reference path and bound-file digests match the reference; the task-graph source-binding path/hash, archived TaskSpec, and native trajectory's TaskSpec SHA match. The adapter control-class label is interpreted according to its explicit task mapping. Native result, predicate evidence, task outcome, and video files are preserved. `task_success=false` is still a valid model measurement only after this identity and output contract passes.
9. **Score/report:** all required score tracks are complete with numeric values and no infrastructure `pending`; report-v3 marks the target row complete and contains an actual score. A prior report or a job return code alone is insufficient.

## Current recovered run

`docs/recovery-record.md` records a prior run and its present limitations. It is evidence for reuse/recovery, not a claim that the new score/report gate has already passed.
