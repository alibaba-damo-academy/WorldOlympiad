# Frozen-input Phase 1: native view, score, and report

This phase validates the post-reference half without re-running task selection, SceneCode, expert reference, or the three-card LingBot policy episode. The exact frozen source is `lingbot_reference_candidate_20261101a`, case `apple_to_coaster_onboarding_20261102`, model `lingbot_va`, seed `0`: Scene job `#8`, reference job `#10`, and native episode job `#14`. Import is permitted only when the SQLite records are still successful and the task, reference certificate, episode metadata, reference binding, task graph, capture metadata, and native trajectory hashes match.

The import creates a new run, records the candidate Git commit and source job/artifact identities, links the TaskSpec and expert reference read-only, and exposes the episode in the new run while preserving its original metadata as `episode_original.json`. It does not alter the old run. In particular, the recorded `task_success=false` is a native result and must remain unchanged.

The missing `exterior_left` evaluator view is rendered from the imported native trajectory using the baseline `wo.render_plan` and `wo.blender_replay` camera/renderer path. It is stored under the new episode's `native_replay/`, never under the frozen reference. Native head/wrist captures are retained. The score input must declare `matched_single_physical_view_v2`, with reference and generated view IDs both equal to `exterior_left`.

The current server's headless Blender EEVEE path uses the EGL default display; this wrapper therefore fails closed unless render GPU index `0` is requested. A fresh preflight must show physical GPU index `0` is allowlisted and free before this view-render job starts. Do not infer EGL device selection from `CUDA_VISIBLE_DEVICES`; do not render on another GPU through this path.

Acceptance requires all of the following for the declared case/model/seed row:

- the frozen source import and TaskSpec/reference/episode bindings verify;
- the native trajectory is preserved and the replay video decodes and matches the reference view;
- baseline `physical`, `geometry`, and `interaction` tracks are scored, including `interaction.continuous_action` from `wo/eval/cae_episodes.py` with source `simulator_execution`;
- the report-v3 row is `complete`, has a numeric aggregate, has no pending required tracks, and its CAE action-trajectory and action-speed components are scored;
- report coverage has no missing references, episodes, or pending matrix entries.

A successful process return code or focused test alone is not end-to-end acceptance. A model may have `task_success=false`; that is accepted as a measured outcome when the native trajectory and evidence are valid and every required score/report component completes.

## Renderer EGL host configuration

The headless Blender renderer uses the host-configured EGL default display. This is independent of the CUDA devices selected through WOV2_GPU_ALLOWLIST; the evaluator does not infer or claim a physical EGL GPU index. Configure the host's NVIDIA GLVND vendor JSON and referenced EGL libraries, grant the run user the device-node access required by that driver/container setup, and set WOV2_EGL_VENDOR_JSON when the vendor JSON is outside /usr/share/glvnd/egl_vendor.d/10_nvidia.json. The stage preflight checks that this JSON exists and is readable.

Before evaluation on a new host, run eglinfo -B (or EGL_PLATFORM=surfaceless eglinfo -B) as the run user and confirm the NVIDIA vendor/renderer, then complete a reference-render smoke and inspect the output. A file-presence check does not prove hardware EGL works. Keep the CUDA allowlist scoped to cards the user authorizes for CUDA work; the host must configure the EGL default so rendering uses an intended, accessible device.
