# Model integration knowledge

Current entry points are the registered adapters in wo/agent/stages.py. Reuse them and
inspect current receipts before changing dependencies, rebuilding environments or adding
an adapter. Historical execution or loading success does not imply task success.
The benchmark agent uses gpt-5.6-sol; that is separate from the policies being evaluated.

| Stage model | Existing adapter | Protocol / identity |
| --- | --- | --- |
| cosmos3 | wo/models/cosmos.py | Cosmos3 Nano-DROID; native RoboLab/DROID execution. Report actual action and imagination artifacts separately when present. |
| lingbot_va | wo/models/lingbot_va.py | Native RoboTwin Aloha execution; use its existing EE-action decoder, not FastWAM's qpos protocol. |
| fastwam | wo/models/fastwam.py | Existing RoboTwin checkpoint, 14-D absolute joint targets, existing server/client and dataset statistics. |
| gr00t | wo/models/gr00t.py | GR00T N1.6-DROID. It is not the user-requested original GR00T N1 (arXiv:2503.14734). Original N1 remains separate unfinished work. |

FastWAM uses envs/fastwam-rt, wo/models/fastwam_server.py and fastwam_client.py. Its
robotwin_uncond_3cam_384 checkpoint still receives instruction conditioning. Do not
recreate the server, redownload installed statistics, or infer absent video from actions.
Use the native head/wrist camera configuration and the upstream policy preprocessing;
do not copy camera sizes or normalization assumptions from old harness notes.

The shared RoboTwin loader, reference-state binding, measured dynamics, visibility checks
and native physics clock already exist. Keep those paths for SceneCode assets and task
predicates. Do not replace them with case-specific task classes or scripted policy actions.
A passing scene_check is input/visibility evidence, not a successful manipulation rollout.

FastWAM defaults to stable native success or the native action limit. For an explicit
fixed-time diagnostic, submit episode with fixed_window=true and window_s specified.
Other adapters use their declared native protocols; record actual budgets and termination
rather than assuming identical wall time, physical time, action counts or video duration.
Follow the run manifest for models/seeds; do not expand a scoped diagnostic into a full matrix.

Preserve official observation/action conventions when reproducing an upstream baseline.
Alternative proprioception or layout controls belong in separately labelled diagnostics.
Do not silently alter model-specific object dimensions, scene appearance or success criteria
to obtain a pass. Native model failure is a reportable outcome after valid inputs are verified.

Useful local evidence (read outcomes, not only filenames):
- outputs/wo_runs/bench_v1/results.json: historical four-model matrix; retained unchanged.
- outputs/adapter_repair_20261011/causal_audit_summary.json: prior adapter/physical-contact audit.
- outputs/single_factor_ref_20261011/controlled_batch_result.json: controlled shape/material
  trials, the distinction between success reached and final state, and default native acceptance.
