Produce the first complete WorldOlympiad results table.

Deliverables (run `bench_v1`, everything under outputs/wo_runs/bench_v1/):
1. Task coverage: at least 2 cases for EVERY task family in knowledge/benchmark.md (22+ cases). 10 legacy
   cases are already in tasks/ with SceneCode generations - reuse the ones that fit; author the rest
   (explicit sizes, DROID envelope) and generate their scenes (stage "scene", runs ~30-70 min each, run many
   in parallel). Every case needs a successful expert reference (stage "reference"); when the expert fails,
   fix the general capability (e.g. implement the reorient primitive), not the case.
2. Models: cosmos3 (ready), lingbot_va, gr00t (current N1.6-DROID baseline, labeled distinctly from the user-requested original GR00T N1, arXiv:2503.14734), fastwam.
   Build each missing adapter, prove it with a positive control, register it, then run every case it can
   execute with 1 seed (window 60 s). Missing adapters/importers/measurements and failed scene
   initialization remain pending engineering work. NA requires an explicit intrinsic simulator or
   action/embodiment protocol limitation with its reason and evidence; never silently drop a row.
3. Scoring: stage "score" for every episode; CAE v2 from simulator execution for WAMs.
4. Report: extend wo/eval/report.py so results.json / results.md contain, per case x model x seed: task
   success, every sub-metric of every track (physical: per-question compliance/confidence and dimension
   scores; geometry: its components; interaction: semantic score + CAE v2 total, EE/object, track/speed),
   macro track scores and the total; plus per-model and per-family aggregates. Then stage "report".

Completion is the declared bench_v1 matrix: all 24 references and all 96 case x model x seed-0 rows,
with complete scores or justified applicability exclusions and a fresh report whose coverage.matrix_complete
is true. Include per-metric subset/N/NA and source evidence, and state the current 60-second model-window
and CAE v2 alignment protocol. End with GOAL COMPLETE only after this audit. This completes this table,
not automatically the broader user goals: explicitly distinguish remaining original-N1 or oven-loop work
from the N1.6-DROID baseline and this matrix.

Priority: first make sure ALL expert references succeed (fix any remaining failed references before running
more model episodes). The user explicitly requested this.
