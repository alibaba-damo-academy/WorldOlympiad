# WorldOlympiad v2 evaluation workflow

This repository packages the recovered WOV2 task-to-report workflow: task selection → SceneCode scene generation and repair → expert reference → native LingBot-VA episode/video → three-track scoring → report-v3.

**Status:** the RTX 6000D native LingBot workflow and real/sim user benchmark paths completed on the frozen pre-adapter revision. The current adapter extension has passed local contract and non-LingBot command-adapter integration tests; its additional native model paths have not yet run with their model-specific weights and runtimes. A separate clean-environment installation on 6000D passed after GPU-contention interruption and controlled recovery. A fresh clone still needs host prerequisites, model/data assets, and credentials. This is not yet an approved public release; remaining boundaries are in [docs/migration-v1.md](docs/migration-v1.md) and the [T4 installation record](docs/t4-clean-install-20261116.md).

## Migration v1 scope

The generic profile targets Linux x86_64 with NVIDIA CUDA GPUs supported by the pinned PyTorch and simulator stack, CUDA toolkit 12.8, separate Python 3.10/3.11 environments, a user-supplied tested-model checkpoint where the native adapter requires one, and an HTTPS Agent API for RSI runs. Set `WOV2_GPU_ALLOWLIST` to the local `nvidia-smi` indices this run may query or use; resource checks stay inside that list. LingBot functional and clean-environment evidence comes from RTX 6000D before the adapter extension; the earlier RTX 5090 full workflow used a previous candidate. AMD/ROCm is not supported by this profile. Exact requirements, stage admission thresholds, dependency sources, and remaining validation gates are in [docs/migration-v1.md](docs/migration-v1.md). The files runtime.lock.json, sources.lock.json, assets.lock.json, and bundled project lockfiles record the key pinned inputs; they do not lock every transitive wheel.

The source verifier is portable and does not need GPU access or credentials:

~~~bash
./setup.sh sources-verify
~~~

For a host that already has a compatible WOV2 runtime and data root, reuse the existing environments and weights explicitly:

~~~bash
./setup.sh runtime-reuse --server-root /path/to/compatible/WOV2
./setup.sh weights-reuse --shared-root /path/to/weights
./setup.sh robotwin-assets-check
./setup.sh runtime --check-only
~~~

For a clean host, set `WOV2_GPU_ALLOWLIST` first, install Python 3.10, Python 3.11, uv, and CUDA 12.8, then run `./setup.sh runtime --online-install`. This explicit mode downloads packages; the tested-model checkpoint and benchmark datasets are still supplied separately. The legacy offline mode continues to require a pinned Python archive and wheel cache that are not included. See the migration guide for limitations and host requirements.

## Configuration and run

The Agent API and tested model are separate interfaces. Configure an already-existing checkpoint and HTTPS endpoint, then enter the API key through the no-echo prompt:

~~~bash
python3 scripts/configure.py \
  --agent-base-url https://HOST/v1 \
  --agent-model MODEL_ID \
  --checkpoint /absolute/path/to/checkpoint
python3 scripts/set_api_key.py
~~~

The key is kept in ignored .secrets/scenecode.env with mode 0600. Never put it in JSON, source files, command arguments, logs, or commits. The tested-model checkpoint is not downloaded or replaced by setup.

Run the admission checks before work. Set the allowlist in the same shell. Preparation and evaluation warn if the NVIDIA EGL vendor JSON is unavailable. Replay then uses host vendor discovery, which may select Mesa. Verify Blender rendering on the target host before a full run.

~~~bash
# Replace these with the local nvidia-smi indices you authorize; no physical index is reserved for EGL.
export WOV2_GPU_ALLOWLIST=2,3,4
# Optional when the NVIDIA GLVND vendor JSON is outside its standard path:
# export WOV2_EGL_VENDOR_JSON=/path/to/10_nvidia.json
./setup.sh preflight-preparation
./setup.sh preflight-evaluation
./setup.sh preflight-scoring
~~~

Then use ./setup.sh run --phase prepare --run RUN_ID followed by ./setup.sh run --phase continue --run RUN_ID. Every launched stage rechecks its GPU topology. At most one evaluation job may be active. No setup or retry path removes run evidence.

## Current workflow evidence

The RTX 6000D run `lingbot-6000d-user-flow-20261110c` completed scene generation, successful reference, native scene check, LingBot-VA episode, scoring and report-v3 using compatible reused environments. The model executed 800 actions and returned `task_success=false`. The report-v3 overall score was 0.3579. Blender replay used host EGL discovery and SAPIEN required matching NVIDIA Vulkan userspace libraries. T1–T3 then passed the real/i2av, sim_rollout, and mixed-batch user interfaces before the adapter extension. T4 established five new environments on 6000D and passed CUDA and renderer checks after controlled recovery. The adapter extension adds model-independent scoring and multi-GPU command adapters, and routes registered models through their original native episode adapter for WOV2 sim references. Only the interface tests have run for non-LingBot models; RTX 5090 regression and T5 full workflow on the new environments remain unverified. See the migration guide for the exact deployment boundary.

## Data and source handling

envs/, runtime/, weights/, cache/, .secrets/, outputs/, tmp/, and logs/ are machine-local data and are not committed. Bundled source snapshots are verified against file/mode manifests; an absent upstream Git history is recorded as unavailable rather than guessed. Preserve source locks and run receipts.

## Credits and provenance

Chengzhi Wang developed the WOV2 evaluation source on which this candidate is based. SimerWhite prepared this portable repository and its user-facing benchmark interface. The separately bundled third-party projects retain their own authorship and license terms; their source origins are recorded in `sources.lock.json`. Git commit authorship for this packaging does not replace attribution for the underlying work.

The repository contains third-party source snapshots with different license terms. Do not publish it publicly until the missing license/provenance items in the migration guide are resolved.

## User-supplied real/sim datasets

Users provide their own real/sim reference datasets and initial model inputs, then generate predictions through a built-in or user-supplied adapter or submit existing prediction videos. The built-in native sim_rollout can select a registered model when its WOV2 reference and model-specific runtime are available; arbitrary public videos use i2av or a submitted prediction. Use benchmark-check, benchmark-run and benchmark-score for this workflow. No lab-specific or paper-specific sample list is required. See [the benchmark contract](docs/benchmark.md) for model adapters, input manifests, result submissions, GPU gates and validation scope.

Accepted scope and runtime evidence for the benchmark interface are recorded in [docs/benchmark-validation.md](docs/benchmark-validation.md).
