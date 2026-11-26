# External benchmark adapter contract

The implementation is defined by `scripts/benchmark.py` and
`docs/benchmark.md` in this candidate. Recheck those files if the candidate
revision changes.

`--adapter` points to JSON with `schema_version: "wov2_adapter.v1"`, a stable
`model` identifier, declared `modes`, positive integer `minimum_free_mib`,
optional positive `gpu_count` (default 1), and an argv `command` list. The
command must contain whole-element `{input}` and `{output}` placeholders; it
may contain whole-element `{gpu}` and `{gpus}`. There is no shell expansion.
The runner chooses only idle allowlisted physical GPUs, sets
`CUDA_VISIBLE_DEVICES` to those indices, and launches the command from the
adapter JSON's parent directory. Inside CUDA, selected devices are numbered
from zero. Do not treat `{gpu}` as a local CUDA ordinal.
The current contract reserves at least one GPU even for a remote API model;
report that limitation instead of inventing a memory requirement for a
GPU-free protocol.

`{input}` is a JSON file containing `case_id`, `instruction`, `mode`,
`conditioning` (role to absolute initial-image path), optional
`conditioning_state` (parsed finite numeric measurements from a case-bound
JSON file), `output_view`,
`output_size_wh`, and `chunks`. For `sim_rollout` it also contains `execution`,
which may contain a verified WOV2 `reference_dir` and `window_s`. The input
intentionally omits the future reference video and evaluation captions. A
wrapper may read only source-bound initial simulation state needed for its
declared protocol; it must not use future reference frames or trajectory as
model predictions.

Write a regular generated video inside `{output}` and a `prediction.json` there
with `{"video": "relative-video.mp4", "view": "output_view"}`. The reported
view must equal the requested view, and the decoded dimensions must equal
`output_size_wh`. The runner validates video decoding and hashes the output.
Do not point the receipt at the reference video, copy it as a prediction, or
write into the dataset or checkpoint directory.

For a model that needs extra proprioception, calibration, or simulation state,
verify that the selected case's certified initial state provides it and that
the current mode passes or permits reading it. Use the dataset's
`conditioning_state` file for numeric initial measurements; its content and
hash are bound to run inputs. An untracked model-specific sidecar would make
resume unsound. If the contract cannot do this, mark the case incompatible
and specify the required extension. A `sim` label or
reference recording by itself never supplies those measurements.

For the exact current commands and applicability limits, use
`docs/benchmark.md`. A registered native adapter uses the separate tested-model
profile and its model-specific runtime; do not wrap it as an external command
solely because its name differs from LingBot.
