# Restore the baseline matched-view scoring contract

- Candidate source snapshot: WorldOlympiad-main, tree SHA-256 `8f80b784c215cf5038df76d9cf815f77ba77941f29809b9e32be7918799abf01`.
- Change: require the exact reference camera view when resolving generated episode video; reject head-only fallback for an `exterior_left` reference.
- Scope: video selection and its focused contract tests only; no task, predicate, track, metric, threshold, or weighting changes.
- Validation: `wo/tests/test_eval_case_native_videos.py` accepts exact matching views and rejects a mismatched camera.
- Baseline comparison: local `WorldOlympiad-main/wo/eval/case.py` uses strict requested-view lookup.
