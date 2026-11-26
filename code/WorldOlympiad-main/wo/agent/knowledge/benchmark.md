# Benchmark design and task authoring

Use the current user goal and declared run manifest. The historical bench_v1 has 24 cases,
11 family labels and 5 execution primitives; a scoped pilot need not recreate that matrix.
Use wo/spec.py and the registered stages as the supported interface, not old TODO notes.

## Author new evaluation data

- Reuse the existing family labels: pick_place, stacking, insertion, hanging,
  reorientation, push, articulation, tool_use, covering, transfer and clutter. Household
  assets do not create a new family; add a family only when explicitly planned.
- initial_container declares physical containment INSIDE another object, not resting ON
  a table/support. Omit it for objects on a tabletop. Use only predicate parameters
  interpreted by wo/predicates.py; do not invent goal flags such as resting or held_by_robot.
  Stable release uses the existing step constraints and native success checks.
- Prefer recognizable household assets: apples, bananas, cups, bowls, utensils, cabinets
  and ovens. Keep realistic proportions, stable initial placement and an unsolved goal.
  Give destinations physical geometry (a coaster, tray or mat with thickness), not an
  unmodelled colored goal region.
- Before generating assets, describe the intended graspable part and approach directions,
  dimensions in metres or centimetres, and whether precise orientation is an intended
  challenge. Use existing object descriptions/geometry_constraints and agent notes;
  do not add mandatory TaskSpec fields or insert privileged grasp poses into model inputs.
- Check the graspable part against the actual participating grippers, including approach
  clearance. A handle needs a real finger passage behind its bar and clearance from
  the panel/roof along the actual finger sweep; a bright strip alone is insufficient.
  A banana's length, a cup handle and a cabinet's full width are different
  quantities: do not uniformly shrink a whole assembly to fit its handle into a gripper.
  Reuse wo/expert/grasps.py and embodiment.py; they already check geometric clearance.
- For basic pick-and-place, prefer compact shapes with margin across plausible grasp
  directions, rather than merely one feasible narrow-direction grasp. For the current
  roughly 9 cm opening, compact 5-6 cm objects are a provisional authoring starting point,
  NOT a universal size threshold, a required diameter, or a proven optimum for all models.
  Keep elongated objects and handle/precision tasks when those skills are intended.
  Record intended difficulty before model evaluation; do not reclassify failures afterward.
- Check the graspable part in the actual native camera frames at model input resolution.
  Existing scene_check framing/segmentation proves visibility, not sufficient visual detail
  or full task reachability. Inspect measured pixel extent and native finger/contact geometry
  for small objects; do not invent a universal minimum centimetre or pixel threshold.
- A reference demonstrates feasibility for its own embodiment; it does not prove every
  policy robot has the same grasp options. Check native reach/approach/clearance when the
  embodiment changes, and use native expert validation where available. Missing validation
  remains explicitly pending. Baseline policy failure alone does not invalidate a scene.
- Select evaluation data by declared task difficulty, physical feasibility and input quality,
  not whether a particular evaluated model succeeds. Use a separate development run for
  repairs; freeze the evaluation set before comparing models. Preserve hard but feasible
  cases and all historical failures. Use the same asset dimensions and task predicates for
  models compared on a case; record necessary robot/camera/protocol adaptations explicitly.

## Execution and bounded repairs

Pipeline: TaskSpec -> scene -> reference (physical execution and matching replay) ->
scene_check/native input verification -> episode -> score -> report. Reuse existing jobs
and valid bound references. author_task publishes into the run's active tasks: keep
unselected drafts outside a live run's tasks directory, and isolate development variants.

Change one authored factor per diagnostic candidate. Preserve old assets/references/results;
update visual and collision geometry together and record derived mass/inertia/pose changes.
Compare the resulting initial state and native layout too: automatic scaling, placement or
camera fitting can introduce extra changes. Keep the goal and success predicate fixed.
Do not replace a failed score with a successful variant or bypass reference/input gates.

Respect the declared protocol. FastWAM's existing default stops on stable native success
or the native action limit; explicit fixed-window controls remain available. Do not impose
60 seconds, two seeds, or the reference duration on every run. Distinguish a stable success
reached during a rollout from its final state after additional actions. Keep action-policy
execution and video-imagination protocols distinct; do not invent missing imagination video.

## Current reporting contract

Use wo/eval/report_v3.py rather than rebuilding metric weights in a prompt or script:
- Physical is unchanged. Geometry equally combines reconstruction, meta-view, camera
  trajectory and CAE action trajectory.
- Interaction equally combines chunk, transition, global, CLIP auxiliary and action speed.
  Action speed is one submetric, not half the Interaction score.
- Overall equally combines Physical, Geometry and Interaction. Report applicability,
  subset, N, NA, pending measurements, protocol and source evidence. Missing measurements
  or infrastructure errors are not legal NA or model failures; job done is not score complete.
- Prefer compact aggregate tables; retain case-level evidence in artifacts rather than
  filling user summaries with every case. Preserve historical v2 results and diagnostics.

The 2026-10-12 FastWAM experiment is one-case evidence: reducing the red block's long axis
from about 12.2 to 5.5 cm succeeded at seed0; changing only height or roughness did not.
The unchanged registered adapter also succeeded. This motivates checking grasp direction
and shape, not hardcoding that case or concluding that all model failures have this cause.
Evidence: outputs/single_factor_ref_20261011/controlled_batch_result.json.


Reference attempt provenance: the `.history` path printed at the start of a reference job contains the preceding attempt, not that job's result. Diagnose the exact job log and its current certificate/task snapshot/source bindings first; if setup failed before a certificate existed, do not assign a recursively found older certificate to the failed job or retry unchanged assets on that assumption.


Actual asset geometry: use the read-only `wo.tasks.asset_geometry` CLI in the
existing SceneCode Python environment. From the worktree run
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/worldolympiad2/envs/scenecode/bin/python -m wo.tasks.asset_geometry --sdf ABSOLUTE_SDF_PATH`.
It reports per-link glTF/GLB visual bounds in source-model metres AFTER declared scales,
plus the SDF hash. Use the exact current asset and its front direction when
comparing bar rear to panel front; requested prose and unscaled ObjectPlan
numbers are not measurements. Check the final room instance scale separately.
These bounds do not certify contact clearance, graspability, robot reach,
execution, or native-camera visibility; keep the existing physical checks.
Primitive/OBJ visual geometry is explicitly unsupported by this reader; a tool error is missing measurement, never a failed task score or legal NA.
