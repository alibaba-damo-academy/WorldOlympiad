# Existing benchmark input and model results

The reference dataset may contain real recordings and simulated demonstrations. Both provide benchmark inputs and evaluation reference. The generated model result is the scoring target.

## Dataset contract

Use schema_version "wov2_benchmark.v1", dataset_id, version, and nonempty cases. Paths are relative to their containing JSON file.

Example:
    {
      "schema_version": "wov2_benchmark.v1",
      "dataset_id": "my_benchmark", "version": "1",
      "cases": [{
        "case_id": "case_001", "domain": "real",
        "instruction": "Place the apple on the tray.",
        "reference_video": "reference.mp4",
        "reference_view": "head_camera",
        "conditioning": {
          "head_camera": "initial/head.png",
          "left_camera": "initial/left.png",
          "right_camera": "initial/right.png"
        }
      }]
    }

For domain sim, execution.reference_dir may point to an existing verified WOV2 simulator reference and execution.window_s may request a short diagnostic rollout. Reference domain and execution mode are separate fields. A generic simulation dataset requires a compatible simulator adapter; its recording alone does not define a runnable simulation.

Models that require proprioception may add `"conditioning_state": "initial_state.json"`
to a case. This relative file must contain a nonempty JSON object of finite
numeric measurements, such as `{"joint_position": [seven measured values],
"gripper_position": 0.0}`. The evaluator validates the file and binds its
SHA-256 and values into immutable run inputs. The external adapter receives
the parsed object in `conditioning_state`; it must validate the model-specific
shape, units, and camera alignment. Use measured initial state from the same
episode and time as the conditioning images. Keep the source record and
extraction receipt with the dataset. No state is inferred from a reference
video or filled with defaults.

Optional caption_chunks entries contain numeric start_s, end_s, caption. The default is the sample instruction over the complete reference duration. Source intervals map proportionally to the generated duration following the recovered paper_three_track_batch protocol. Videos are not temporally cropped or retimed.

Generation adapters receive initial conditioning images, optional file-backed
initial state, instruction, output view and size, and the requested horizon.
Evaluation captions and future reference video are excluded from i2av inputs.
The user is responsible for supplying actual permitted initial measurements.
sim_rollout receives execution metadata for resetting the source-bound
simulation.

## Generation adapters

Built-in lingbot_va i2av calls the pinned upstream robotwin_i2av configuration with the configured checkpoint. It requires actual head, left and right conditioning views. It crops the official T-shaped composite to the selected camera and resizes spatially to the evaluation reference, following the recovered video_batch layout. It never creates missing camera views. One chunk is the short-integration default; benchmark runs must explicitly specify their intended horizon. The launcher closes its distributed process group on both success and failure. --timeout covers import, checkpoint loading, inference and shutdown; slow shared storage may need --timeout 7200. A timeout remains a failed generation attempt even if an intermediate video exists.

Built-in sim_rollout uses the registered native episode adapter for the configured tested model and exports the reference-bound exterior_left evaluation video. The reference must belong to a completed WOV2 native run with a passing certificate and admission evidence. The built-in LingBot path retains its audited scene check. FastWAM and Long-WAM use their own 14-D qpos protocols through the same reference-bound RoboTwin scene loader; the current loader admits the stock `stack_blocks_two` environment with a compatible one-step WOV2 `on`/`stacked` goal. They do not accept an arbitrary sim recording as an executable scene. Resource preflight automatically selects one allowlisted GPU when it has combined model-and-simulator headroom, or two eligible GPUs otherwise. Other simulation protocols use command adapters. An existing native adapter module does not install that model's checkpoint, runtime, simulator assets, or driver stack.

A command adapter JSON contains schema_version "wov2_adapter.v1", model, modes, minimum_free_mib, command, and optional gpu_count (default 1). The command is an argv array with whole elements {input}, {output}, and optionally {gpu} or {gpus}. No shell interpolation is used. Input points to a conditioning-only JSON, output to a private attempt directory. The adapter writes prediction.json containing video and view, with paths relative to that receipt. Its checkpoint and runtime are separately provisioned. The runner reserves gpu_count eligible allowlisted GPUs and sets CUDA_VISIBLE_DEVICES to their physical indices in order; the framework sees local ordinals starting at cuda:0. {gpu} is the first physical index and {gpus} is the comma-separated physical list. Supported modes must be declared explicitly. This external adapter path does not require a LingBot checkpoint or tested-model profile.

Example adapter:
    {
      "schema_version": "wov2_adapter.v1",
      "model": "my_model", "modes": ["i2av"],
      "minimum_free_mib": 24576, "gpu_count": 1,
      "command": ["/path/to/python", "/path/to/model_adapter.py",
                  "--input", "{input}", "--output", "{output}", "--gpu", "{gpu}"]
    }

## Commands

For a built-in model, configure its existing checkpoint and compatible runtime using the normal repository flow. For a command adapter, provision that model's runtime and checkpoint yourself; no LingBot checkpoint, Agent API, or tested-model profile is needed for benchmark-run or benchmark-score. The scoring runtime, scoring weights, and allowlisted GPUs are still required for a real score. Run CPU checks before GPU work.

    ./setup.sh benchmark-check --manifest /path/to/dataset.json
    ./setup.sh benchmark-run --manifest /path/to/dataset.json --model lingbot_va --mode i2av --run test_model --case case_001 --chunks 1 --generation-only
    ./setup.sh benchmark-run --manifest /path/to/dataset.json --adapter /path/to/adapter.json --mode i2av --run custom --case case_001

For an already provisioned registered native model and its original completed WOV2 reference, use --model MODEL --mode sim_rollout. A public sim video without that reference is still handled through i2av, a compatible external command adapter, or benchmark-score.

benchmark-run generates then scores by default. --generation-only stops after predictions.json. GPUs use the explicit WOV2_GPU_ALLOWLIST and existing idle-device checks, topology and leases. The runner serializes benchmark work within this workspace and refuses an active native evaluation queue. The built-in adapter uses the configured runtime and model paths; no checkpoint downloads occur.

For a registered native sim_rollout, configure the matching tested_model.adapter and checkpoint in the non-secret profile; the repository's configure.py accepts --tested-model for the registered names. Only LingBot's full runtime is installed by the default setup profile. Other registered models require their own documented runtimes, weights and simulator assets at the paths expected by their source adapters; the profile checkpoint field alone does not redirect every upstream loader. The external command-adapter and submitted-video paths select scoring GPUs independently of the tested-model profile. Use --score-mode, --score-gpu and --score-compute-gpu when the automatic scoring topology is unsuitable; these options are bound into the run inputs.

## Submit already generated videos

A submission contains schema_version "wov2_predictions.v1", dataset_id, dataset_version, model, predictions. Entries contain case_id, video, view. For the selected cohort, missing/duplicate/unknown IDs and wrong views or dimensions fail before GPU scoring.

    {
      "schema_version": "wov2_predictions.v1",
      "dataset_id": "my_benchmark", "dataset_version": "1",
      "model": "my_model",
      "predictions": [{"case_id": "case_001", "video": "generated.mp4", "view": "head_camera"}]
    }

    ./setup.sh benchmark-score --manifest /path/to/dataset.json --predictions /path/to/predictions.json --run submitted --case case_001

## Output and resume

Outputs live in outputs/benchmark/RUN separately from native RSI evaluation runs. inputs.json binds the manifest, hashes, model, mode, configured checkpoint/runtime, implementation hashes and cohort. Scheduling GPU choices may change on resume. Treat checkpoints and command-adapter implementations as immutable versioned artifacts: provision a new path and run ID when replacing their contents. Private attempt directories preserve interrupted generation. Scoring uses byte-for-byte regular video copies inside its own case directory: the recovered scorers create adjacent intermediates, so symlinks to source videos are rejected. Original dataset and prediction files remain untouched. Completed generation is reused only after its video hash is verified.

The existing three-track score_robotics_cases runner handles the paired videos, caption intervals, local weights and service lifecycle. report.json records complete/failed/pending cases, domain, input hashes, raw score paths, track scores and real/sim summaries. It declares simulator task success and native trajectory CAE not applicable for this video-scoring protocol, in inputs.json before execution. Native raw scores and receipts remain unchanged. benchmark_score.json uses the existing score hierarchy with this explicit applicability declaration; the semantic Interaction submetric supplies the applicable Interaction measurement. Missing video-judge evidence remains a failure, including physical judge errors, absent geometry artifacts, incomplete interaction judgments or incomplete CLIP coverage. Simulation-native report-v3 remains available through the existing native pipeline. Additional benchmark trajectory protocols require explicit adapter and scorer support.

A complete report requires all selected cases to have numeric three-track scores. Failures remain failures; missing outputs are not zero scores. Resume with unchanged inputs and flags. A changed dataset or model protocol requires a new run ID.

Users supply their own real or simulated reference data through the versioned manifest. The repository provides input validation, model-adapter contracts, generation, submitted-video scoring and reports. No lab-specific or paper-specific sample list is required to install, use or release this evaluator. Dataset membership and hashes are frozen for each user run. A published benchmark may separately provide its own sample manifest; user videos, weights and generated outputs remain runtime data outside the source repository.

## Validation boundary

CPU tests use actual short encoded videos and a fixture adapter/scorer to test integration contracts, not model quality or judge outputs. Acceptance of a real model adapter requires a short native generation and output decode. Acceptance of scoring requires a real short paired-video three-track run and complete report. Long video rendering and the full benchmark cohort are separate validation steps.
