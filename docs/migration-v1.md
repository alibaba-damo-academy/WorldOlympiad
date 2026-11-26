# Migration v1: host, dependencies, and readiness

## Target and current readiness

The target profile is Linux x86_64 with NVIDIA CUDA GPUs supported by the pinned PyTorch and simulator stack. The runtime queries only indices named in `WOV2_GPU_ALLOWLIST`; AMD/ROCm is outside this profile. The RTX 6000D validation run `lingbot-6000d-user-flow-20261110c` completed scene generation, expert reference, native camera validation, LingBot-VA seed-0 inference, three-track scoring and report-v3. The run reused compatible server Python environments, weights and RoboTwin assets. Blender replay used host Mesa EGL discovery; SAPIEN cameras used a process-local NVIDIA Vulkan driver bundle matching host driver 580.173.02.

The run executed 800 policy actions and returned `task_success=false`. Report-v3 records the episode as complete with Physical 0.431775, Geometry 0.3904607489, Interaction 0.2515158653 and overall 0.3579172047. Task, reference, episode and score provenance are recorded in the report. Generated evidence remains external to Git.

This candidate incorporates the tested RSI repairs for overwritten glTF visual buffers, single-run evaluation scope, explicit fixed-table binding, native table-height alignment, allocated scene-check GPU selection, and QwenVL readiness probing. The source snapshot is derived from tested nested source commit `52c9a15`; WorldOlympiad-main provenance remains recorded in sources.lock.json.

| Validation | Status |
|---|---|
| Full workflow on RTX 6000D with reused compatible runtimes | Completed with model task_success=false |
| RTX 5090 full workflow | Completed on an earlier candidate; regression of this frozen candidate pending |
| Source snapshot verification | Per-file manifests and source lock |
| Dependency reproducibility | Key versions and source revisions pinned; some direct/transitive packages and wheel hashes remain unpinned. SceneCode/DA3 metadata exceptions are recorded in [T4 evidence](https://github.com/alibaba-damo-academy/WorldOlympiad/blob/v2/docs/t4-clean-install-20261116.md) |
| Clean online installation | 6000D clean environments and runtime checks passed after GPU-contention recovery; uninterrupted one-command install is unverified. See [T4 evidence](https://github.com/alibaba-damo-academy/WorldOlympiad/blob/v2/docs/t4-clean-install-20261116.md) |
| User-interface adapter extension | Local non-LingBot command-adapter generation/score integration and registered native-adapter routing tests passed; additional native models have not been run with their own weights/runtimes on GPU |
| Public GitHub release | Requires licensing/provenance and public-tree review |

## Host requirements

| Component | Requirement for v1 |
|---|---|
| OS/architecture | Ubuntu 24.04 LTS, Linux x86_64 is the validated target profile |
| GPU | NVIDIA CUDA devices supported by the locked PyTorch/native-extension stack; set an explicit `WOV2_GPU_ALLOWLIST`. LingBot on devices up to 40 GiB currently needs two policy GPUs plus a separate simulator GPU. |
| Driver/CUDA | NVIDIA driver compatible with the locked CUDA 12.8 runtime; the configured server uses driver 580.173.02 and CUDA toolkit 12.8. Set `CUDA_HOME` or put CUDA 12.8 `nvcc` on `PATH`. |
| Python | Separate Python 3.11 and 3.10 runtimes. SceneCode, WorldEval, and autoresearch use 3.11; RoboTwin and LingBot-VA use 3.10 |
| Environment tool | uv for runtime creation and package installation |
| Build tools | C/C++ compiler, make, CMake, and Git for CUDA/native extension builds and source operations |
| Rendering | A working host EGL stack is required. NVIDIA EGL is preferred for replay when configured. Host vendor discovery may select Mesa software rendering. Smoke-test Blender on each server. |
| Network/access | HTTPS access to the configured Agent API and package/model sources; Hugging Face access for gated or large benchmark/RoboTwin assets as applicable |
| Storage | Repository checkout plus separate space for Python environments, Blender, model weights/checkpoints, RoboTwin assets, caches, and generated outputs. These data are not included in the source tree |

### RoboTwin and SAPIEN Vulkan

RoboTwin's native cameras require a working NVIDIA Vulkan userspace driver. If the host has a matching driver bundle outside system paths, export `VK_DRIVER_FILES` and `VK_ICD_FILENAMES` with the path to its NVIDIA ICD JSON, and include the bundle's driver library directory in `LD_LIBRARY_PATH` for the evaluation process. The ICD must resolve to the matching NVIDIA driver library. Check SAPIEN renderer initialization and camera readback as the evaluation user. Blender replay can use Mesa EGL independently of this requirement.

### NVIDIA EGL host configuration

Blender uses the host's EGL default display. The evaluator does not bind this display to a CUDA index, and WOV2_GPU_ALLOWLIST applies only to CUDA resource queries/selections. There is no universal requirement to reserve physical GPU 0. For NVIDIA hardware rendering, the administrator should:

1. Install the NVIDIA driver components that provide libEGL, GLVND, the NVIDIA EGL vendor JSON (commonly /usr/share/glvnd/egl_vendor.d/10_nvidia.json), and the library named by that JSON. If the JSON is elsewhere, set WOV2_EGL_VENDOR_JSON to its path.
2. Grant the evaluation user access to the NVIDIA device files and, where that driver/container path requires them, DRM render nodes. Use the host's group/udev/container-device policy; device-node permissions vary by installation. Do not assume CUDA access alone proves EGL access.
3. As the same user and inside the same container/job environment, run eglinfo -B (or EGL_PLATFORM=surfaceless eglinfo -B) and record whether the vendor/renderer is NVIDIA or Mesa llvmpipe. Then run the evaluator's reference-render smoke and inspect its Blender log and video. A readable JSON file alone does not prove the driver can create a hardware EGL context.

The frozen implementation before the adapter extension passed T1–T3 user-facing paths and T4 clean-environment installation on RTX 6000D. The additional adapter paths have local integration evidence only. An earlier candidate completed a full workflow on RTX 5090; this implementation has not been rerun there. The seven-card count and the lab's local indices are host facts, not evaluator requirements. Runtime queries are gated by the user's `WOV2_GPU_ALLOWLIST`, not by the server's card model or count.

## GPU admission thresholds

Admission is checked immediately before each stage. Only CUDA devices in WOV2_GPU_ALLOWLIST are queried or used; eligible cards must have utilization at most 5% and no compute-app PID. Preparation/evaluation warn if the NVIDIA EGL vendor JSON is unavailable and continue with the CUDA admission checks. EGL device selection remains the host's responsibility and is not inferred from a CUDA index.

| Stage | Required topology |
|---|---|
| Scene preparation/reference | 1 eligible card with at least 8,192 MiB free |
| LingBot native episode | 2 distinct policy cards, each with at least 30,720 MiB free, plus 1 distinct simulator card with at least 25,600 MiB free |
| Split scoring | 2 distinct cards, each with at least 25,600 MiB free: Qwen-VL service on one; physical/interaction/geometry and CLIP/SAM/DA compute on the other |
| Report-v3 | CPU only |

These are admission thresholds, not a promise that every task or model fits or succeeds. Thresholds are based on the current LingBot and scoring path; a different adapter or checkpoint may need a different resource contract. The actual host smoke test must confirm the selected device's CUDA architecture and renderer support.

## Dependency inventory and source of truth

runtime.lock.json maps each runtime to its Python version, source, lock/requirements inputs, and key versions. Do not merge the environments: RoboTwin and LingBot-VA intentionally use separate Python 3.10 environments, while SceneCode, WorldEval, and autoresearch use Python 3.11.

| Runtime | Python | Dependency definition in this checkout | Important external runtime/data |
|---|---:|---|---|
| autoresearch / RSI tools | 3.11 | code/WorldOlympiad-main/requirements-autoresearch.txt; direct packages are pinned | Requires cache/bootstrap/autoresearch/python311.tar.gz (SHA-256 b9fff092374acebc451bb881156afe6a58991213d25386358742fee296128909) and a local wheel cache |
| SceneCode | 3.11 | code/SceneCode/uv.lock with frozen uv sync; bpy==4.5.4 is installed separately | Blender 4.4.3 Linux x64 archive, SHA-256 8d3be07d2bc412b502c6bfe3cfe3e22195a4164076867da987ce148d73c27946; source-bound runtime overlay |
| WorldEval/scoring | 3.11 | Torch 2.7.1 + cu128 and requirements-worldeval.txt | CLIP is pinned by the root installer to ultralytics/CLIP commit 7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40; DA3 source and scoring/model weights are separate |
| RoboTwin | 3.10 | runtime.lock.json plus code/WorldOlympiad-main/deployment/ziplab5090/bootstrap_robotwin_runtime.sh; cuRobo source is vendored at code/curobo-0.7.8 | Torch 2.9.0 + cu128, CUDA 12.8 compiler, native extensions built for the selected device architecture, RoboTwin assets from the explicit asset flow |
| LingBot-VA | 3.10 | runtime.lock.json and direct package list in scripts/setup_runtime.sh | Torch 2.9.0 + cu128, local model checkpoint, patched LingBot runtime source |
| DROID robot asset | N/A | assets.lock.json and locks/asset-manifests/ | Generated URDF bundle is included and hash-checked; the optional Cosmos Isaac path has a separate USD requirement below |

code/WorldOlympiad-main/requirements-worldeval.txt, code/RoboTwin/scripts/requirements.txt, and code/lingbot-va/requirements.txt are useful source requirements but are not all installer lockfiles. The root installer now overrides the vendored WorldEval bootstrap default and installs the pinned CLIP commit recorded in runtime.lock.json. The shared server runtime contains the matching CLIP source copy and a .wov2_source.json record with that commit and archive SHA-256; the copied tree has no Git metadata. Some direct and transitive package versions are still unpinned, so every resolved wheel is not yet locked for reproducible installation.

The tracked source snapshots occupy about 163 MB before Git history/checkout overhead. They are cloned as repository content; Python environments, Blender, weights, checkpoints, and benchmark datasets remain external.

### Process-local Vulkan configuration

For a compatible user-owned NVIDIA graphics bundle, configure the evaluation shell before preflight and launch:

~~~bash
export WOV2_NVIDIA_GRAPHICS_ROOT=/absolute/path/to/matching-nvidia-graphics
export LD_LIBRARY_PATH="$WOV2_NVIDIA_GRAPHICS_ROOT/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_DRIVER_FILES=/absolute/path/to/nvidia_icd.local.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
~~~

The ICD JSON must point to a readable NVIDIA Vulkan library, such as `libGLX_nvidia.so.0` in that bundle. Its dependent libraries must resolve inside the same process. The bundle must match the host kernel driver. Obtain it through the host administrator or the corresponding NVIDIA package distribution. It is external to this repository. Validate SAPIEN camera capture with the actual evaluation account and then run registered `scene_check` before policy evaluation.

## First-clone flow

1. Clone this repository and verify the immutable source snapshots:

   ~~~bash
   git clone <repository-url> WOV2
   cd WOV2
   ./setup.sh sources-verify
   ~~~

2. Choose one setup route:
   - Existing compatible server data: use runtime-reuse with the existing WOV2 root and weights-reuse with an existing weights root; then run `./setup.sh runtime` to stage this checkout's source-bound RoboTwin, LingBot and SceneCode runtimes without downloading packages. Run the static checks afterward. `runtime-reuse` alone does not create `runtime/scenecode`.
   - Clean machine: install the host prerequisites and required RoboTwin assets, then run the explicit online bootstrap. It installs packages from configured indexes and may take substantial time and bandwidth:

     ~~~bash
     export WOV2_GPU_ALLOWLIST=2,3,4  # replace with the local NVIDIA indices you authorize
     # export WOV2_EGL_VENDOR_JSON=/path/to/10_nvidia.json  # only if outside the standard path
     PYTHON310=/path/to/python3.10 PYTHON311=/path/to/python3.11 ./setup.sh runtime --online-install
     ~~~

     Set `CUDA_HOME` to CUDA 12.8 before this command. If the toolkit places headers and libraries under `targets/x86_64-linux`, add its `include` directory to `CPATH` and `lib` directory to `LIBRARY_PATH` and `LD_LIBRARY_PATH` for this shell. On hosts without a system NVIDIA Vulkan userspace driver, apply the process-local Vulkan configuration above **before installation**: RoboTwin's bootstrap creates a SAPIEN renderer during its smoke test. Keep one allowlisted GPU idle throughout installation; the installer rechecks at several stages and preserves partial environments if another job takes the card. This release has no automatic mixed-environment resume command; the T4 record describes the controlled recovery used on 6000D. The tested-model checkpoint and benchmark datasets remain separate inputs. The legacy offline mode still requires the pinned Python archive and wheel cache; neither is included, and the archive source is not recorded.

3. Obtain RoboTwin assets through the explicit asset-download flow if they are not already available; obtain the tested-model checkpoint separately. The setup scripts do not fetch or replace the tested-model checkpoint.

4. Configure the HTTPS Agent endpoint/model and existing checkpoint with scripts/configure.py, then set WO_AGENT_API_KEY through scripts/set_api_key.py. Keep .secrets/scenecode.env local and mode 0600.

5. Pass ./setup.sh runtime --check-only, the relevant asset/weight checks, and the stage-specific preflights before launching an evaluation.

## Evaluator weights and model adapters

Auxiliary scoring weights are separate from the tested-model checkpoint. The repository has explicit download, reuse, and check commands:

~~~bash
./setup.sh weights-check --shared-root /path/to/weights
./setup.sh weights-fetch-aux --shared-root /path/to/weights
~~~

The auxiliary bundle consists of Qwen3-VL-8B-Instruct, DA3 Nested Giant Large 1.1, SAM3, and OpenAI CLIP ViT-B/32. The downloader validates the Qwen shard index, DA3 files, SAM3 file presence, and CLIP SHA-256. Before starting any large download, it checks access to facebook/sam3 through the logged-in Hugging Face account. If approval has not been granted, the command reports the required access and stops; no SAM3 checkpoint is redistributed in this repository. Reuse already provisioned weights with the check/reuse path. A failed weight check blocks scoring preflight.

The local tested-model checkpoint is a separate input and is never fetched by the auxiliary downloader. The tested clean-install runtime profile remains LingBot-VA. scripts/configure.py now accepts --tested-model for the five registered native episode adapters, but setup.sh installs only the LingBot model runtime and scripts/check_weights.py validates only its checkpoint contract. Other registered adapters require their own model-specific environments, weights and assets at the locations their source modules expect. The profile checkpoint path alone does not redirect every upstream loader. External command adapters and submitted prediction videos use the scoring runtime without a LingBot tested-model profile.

The source-level episode registry currently lists cosmos3, lingbot_va, gr00t, fastwam, and longwam. The Agent reads the registry from wo/agent/stages.py, and its status tool reloads that registry after code promotion. This is explicit registration, not automatic checkpoint detection: a new policy needs an adapter module, a registry entry, its own runtime/weights contract and preflight support. The other preset adapters use distinct protocols and external assets; their presence in source does not mean they are ready in this LingBot-focused migration profile. In particular, this checkout labels GR00T as N1.6-DROID, not the original GR00T N1.

## Optional DROID USD input

wo.models.cosmos expects cache/official_assets/robolab/f72d39445ab699c9cfd9bf551e8d21b6075293b9/franka_robotiq_2f_85_flattened.usd with SHA-256 f555695465687548a1bd31b5e3f30385182d476a67c17080b7820ad0ef747e41. A complete matching USD was found in the external RoboLab-isaac51 migration tree; the expected cache file is absent from this checkout. This asset is not needed by the completed default LingBot-VA/RoboTwin episode path. The source tree carries Apache-2.0 licensing, but this USD's file-specific distribution rights and immutable source revision have not been independently established; it is not bundled here.

## Before a public GitHub release

1. **Resolve redistribution rights.** There is no root project license. code/WorldOlympiad-main and the DROID bridge snapshot have no license file in this checkout. Establish permission/license terms for the recovered and locally modified WorldOlympiad source before publishing it. Keep third-party notices for all bundled assets and snapshots.
2. **Verify and lock clean installation.** The explicit online bootstrap path is implemented, but has not been exercised on a fresh environment. Validate it, then pin the remaining direct/transitive packages. The legacy offline path still needs its absent Python archive and wheel cache.
3. **Prove portability from a clean clone.** On a second NVIDIA CUDA host with a different GPU model or compute capability, test source verification, runtime installation, CUDA extension builds, Blender/EGL rendering, assets/checkpoints, all preflights, and one small end-to-end run. Fix or explicitly parameterize remaining host-specific paths in vendored sources and generated asset receipts; do not rewrite source snapshots without regenerating their manifests.
4. **Resolve optional DROID USD packaging.** Either verify the USD's rights/provenance and add it to an appropriate hash-locked optional asset bundle, or keep the Cosmos path documented as requiring a separately obtained asset.
5. **Review public-tree contents.** Confirm ignored server outputs, credentials, model weights, private configs, and machine-local files are absent; audit retained historical logs/paths and every bundled source license.

The candidate has been pushed to the public `v2` branch for collaborator review.
That push does not complete the licensing, redistribution, or deployment gates
above. See [`THIRD_PARTY_NOTICES.md`](https://github.com/alibaba-damo-academy/WorldOlympiad/blob/v2/THIRD_PARTY_NOTICES.md).
