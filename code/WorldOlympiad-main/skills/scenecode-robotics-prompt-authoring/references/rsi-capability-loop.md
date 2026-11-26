# Research and reusable capability improvement

OlympiadForge keeps one shared loop for robotics→WAM and physics→WM:

```text
case evidence → diagnose owner → retrieve prior lessons → focused repair
→ relevant tests → retry owning stage → record result → reuse on a later case
```

Use `inspect_research_cases`, `inspect_experience`, and `inspect_rsi_traces`
to find retained failures and related successful work. Read relevant source
with `inspect_harness_source`; apply an evidence-backed patch through
`apply_harness_patch`. Failed tests roll back the change. After the owning
stage runs again, attach its real result using `record_repair_outcome`.

A reusable lesson records the original failure, hypothesis, affected code,
tests, stage result and applicability tags. Later matching tasks retrieve it
by stage, object roles, operations and predicates. Record whether the later
task actually benefited; retrieval alone does not prove transfer.

SceneCode generation, asset reuse, camera framing, physical execution, model
adapters, scheduling and completion decisions can all improve through this
process. Ordinary repairs do not require frozen splits, isolated candidate
search or automatic promotion. Broaden relevant regression for changes to
model protocols or evaluators. Preserve valid poor model results as evidence;
low scores alone are not harness failures.
