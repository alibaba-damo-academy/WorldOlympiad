# Auto model adapter skill validation (2026-11-24)

This record covers `.agents/skills/wov2-auto-model-adapter/SKILL.md`. Two
different non-LingBot models completed held-out video generation and
three-track scoring by 2026-11-25. A third model completed generation but its
score was blocked by the shared evaluator service. These results do not imply
that every model is runnable.

## Checks completed

- `skill-creator/scripts/quick_validate.py` accepted the skill structure.
- `test_real_input_decodes_and_hashes`,
  `test_adapter_gets_conditioning_only_and_reuses_result`, and
  `test_external_adapter_reserves_declared_gpu_count` passed.
- `test_external_model_cli_generates_and_scores_without_lingbot_profile` and
  `test_registered_native_adapter_uses_bound_wov2_reference` passed. These are
  fixture and contract tests; they do not execute a real model or judge.

## Non-LingBot forward check: Cosmos3-Nano-Policy-DROID

The candidate registers `cosmos3` as a native episode adapter. Its supported
benchmark route is `sim_rollout` with a certified WOV2 simulation reference,
the Cosmos policy runtime, Isaac/RoboLab runtime, model checkpoint, VAE, and
simulator assets. A plain real or sim reference recording does not provide
that executable state. Selecting the registered native path, rather than
generating a generic `i2av` wrapper, is the skill's correct routing decision
for this model. See the [model card](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID)
and `code/WorldOlympiad-main/wo/models/cosmos.py`.

On the RTX 6000D, the pinned model and VAE files were audited and a separate
Python 3.13 / CUDA 12.8 environment passed CPU imports, including PyTorch
2.10.0+cu128, Cosmos Framework, and OpenPI client. The evaluated workspace
does not yet have the matching Isaac/RoboLab environment and assets required by
the native adapter. No Cosmos GPU generation or benchmark score was run.
Therefore the initial native-route check validated source and runtime
preparation plus routing, not native `sim_rollout` execution. A separate
unregistered-model check was then performed as described below.

## Held-out external adapter: real generation and score

In an isolated checkout, Cosmos-specific native registration and files were
removed from the candidate before applying the skill. A pinned public DROID
sample supplied synchronized camera frames and actual frame-zero joint and
gripper measurements. The evaluator gained an optional file-backed
`conditioning_state` input: it validates finite numeric content, hashes the
file into the immutable run inputs, and passes the measurements to the
external command adapter. The model-owned wrapper validated the DROID shape
and used the existing Cosmos inference bridge, checkpoint, and VAE on the
RTX 6000D shared disk. The candidate's `scripts/benchmark.py` and its 25
benchmark tests are byte-identical to those used by this held-out run.

`benchmark-run --generation-only` produced a new 16-frame, 15 fps prediction
video for one real case; its SHA-256 is
`9205ef162ce07a9e1f6622d99a6ebae019df1a7acb68646c78e5839ebbd99b2c`.
The same run resumed with three-track video scoring and wrote a complete
one-case `report.json` (SHA-256
`08be8bf9aef19010c16e49167f181b99bc70ea737934cafec5a2ad6b74964038`).
The reported overall 0.7003 is only an integration check: the sample's
published task label is blank, so its task prompt is not independently
verified. The external wrapper and this sample are test assets in the
isolated checkout, not new built-in model support in the release candidate.
Native Cosmos `sim_rollout` still requires the matching RoboLab/Isaac runtime,
assets, and executable reference.

## Second model check: FastWAM on RoboTwin data

On 2026-11-25 the skill was applied to FastWAM as a separate non-LingBot
model. Its pinned RoboTwin checkpoint and model runtime already existed in the
senior workspace; the run reused them read-only and downloaded no weights.
FastWAM requires three synchronized camera views and a measured 14-value
initial joint state. A RoboTwin `adjust_bottle` episode supplied both. The
preparation record stores the source HDF5 and task-description SHA-256. The
case manifest supplies nine sampled reference frames, the initial camera
images, and the measured state; the evaluator freezes their hashes in its run
inputs. The adapter receives only those initial inputs, never future reference
frames.

The skill selected an external `i2av` command adapter for this video case.
FastWAM's registered native episode adapter does not accept an arbitrary
reference recording as an executable WOV2 scene. The model-owned wrapper
loaded the checkpoint and generated a 9-frame, 12.5 fps, 320 × 384 prediction
video and a `[32, 14]` action array on RTX 6000D. The video SHA-256 is
`9e3fa0d7a16f8a1c27bac6e11ecbc5914ec9a9680f69c041a90680f946435fc3`.
The evaluator accepted the generated video and resumed the same run for
three-track scoring. `report.json` has `complete=true`, all three video tracks
scored, and overall `0.7722` (report SHA-256
`3d929bc361cab85233d45ed747c359b6960a5288059aaddf4354943e507f9d21`).
The score receipt binds the generated and reference video hashes. The source
HDF5 has no verified camera timestamps, so the numeric score is an adapter
integration check, not a model-quality measurement. The video-only protocol
does not claim native task success or trajectory CAE. This run also does not
validate FastWAM native `sim_rollout`.

## Third model check: Long-WAM RoboTwin2.0-COD

This test used the released Long-WAM COD checkpoint, distinct from the
registered Long-WAM IDM adapter, which returns actions without a clean video.
The skill therefore selected an external `i2av` command adapter. The model
source and 12 GB checkpoint were reused read-only from the senior workspace;
the checkpoint matched its published SHA-256
`25d310fd4586307ec728a1289b6f3b39dda84b51c0893b0b4691c098a7aa70cc`.
No new weights were downloaded.

The same RoboTwin source episode supplied three measured initial camera frames,
frame-zero 14-value joint state, and a separately versioned 33-frame reference.
`benchmark-check` and the adapter's conditioning-only CPU check passed. On RTX
6000D, the native COD inference generated four consecutive chunks: a 33-frame,
12.5 fps prediction video (SHA-256
`4d120996196df80a49a0134e66037c6aa192ee9e5d00717bdace1051eaaa2cfa`)
and a `[4, 32, 14]` predicted action array. The evaluator decoded and recorded
the video. The adapter received no future reference frames. After the measured
initial state, later states were predicted commands, not simulator feedback;
this is video generation evidence, not native task-success evidence.

The 33-frame three-track score did not complete: QwenVL object-extraction
requests repeatedly timed out after 240 seconds, used a generic-prompt
fallback, and left the report pending while its service accumulated stale
connections. A separate score-only check on the first 17 frames of the same
generated video also remained pending when a QwenVL worker stalled reading
shared NFS weight pages. Both evaluator jobs were stopped without deleting
their inputs, output, or logs. Consequently this third test validates real
multi-chunk model generation and the benchmark prediction handoff, but **not**
a complete Long-WAM score. The original HDF5 has no verified camera clock, so
any future numeric score on this derived case would remain an integration
check, not a model-quality claim.
