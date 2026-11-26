---
name: robotics-eval-task-authoring
description: "Author diverse, executable robotics evaluation task data for the WorldOlympiad agentic harness. Use when generating a batch from a high-level evaluation request; do not use for SceneCode prose, simulator execution, or model rollout."
---

# Robotics Eval Task Authoring

Turn the requested batch size, target WAM, and optional user brief into typed
task data. Do not create a task-specific skill, select from a fixed case list,
or encode simulator implementation details in prose.

The current automatic author schema supports rigid-object cases. Check the
live author and adapter schemas before proposing articulated components; their
existence in the reference executor does not imply the author can emit them.

Within that schema, each case must contain exactly one primary manipuland, one reachable primary
support, and any destination or fixture required by the action. Use distinct,
complete singular noun phrases for object appearances. Make task objects large
enough to remain clearly visible in the target WAM's native cameras and give
the manipuland strong color/texture contrast against the support and target.
Keep the initial state concrete, physically plausible, reachable, and unsolved.
The instruction is one short imperative with no duration, camera, scoring, or
controller language.

Choose object colors and materials together with the support and immediate
background, not independently. Encode clear color/lightness separation in the
authoritative `visual_description` fields and a coherent setting in
`scene_brief`; naming different objects is insufficient. Respect explicit user
appearance and choose compatible surrounding materials. Treat object-appropriate
materials and common plausible colors/finishes as equally necessary as contrast;
do not invent an implausible color or excessive saturation just for separation.
If needed, choose a more suitable background or support within the task instead.
For household tasks, use plausible furniture materials, including natural wood where appropriate;
do not impose a fixed palette or furniture recipe on other contexts. Authored
contrast remains a requirement to verify in actual contextual RGB renders.

Give every object a short code-style `object_id` matching
`[A-Za-z][A-Za-z0-9_.-]*`, such as `orange_cylinder` or `target_mat`. Every
`subjects`, `target`, predicate `subject`, and predicate `object` value must be
the exact `object_id` of a declared object—not its visual description, category,
or natural-language name. Categories are also code-style symbols. Roles are
exactly `manipuland`, `destination`, or `support`.

Choose only action operators and terminal predicates advertised in the runtime
capability data supplied with the request. The action program and terminal
predicate must describe the same goal relation. Use one deterministic terminal
predicate per case. Do not add `released`, contact-only, or other auxiliary
predicates unless the capability data explicitly requires them.

Preserve the requested action as well as its endpoint. An unsupported requested
interaction is a capability gap; never hide it behind a supported action with
the same final relation. Grasping a container and pouring its contents cannot be
represented as picking one content item. Striking with a tool cannot become
pressing the target directly, and a required orientation change must be encoded
in the program constraints and its terminal check. Match the actual grasped
subject, tool, container and every declared content instance. The shared
robotics_task_intent reviewer independently compares the full instruction,
roles, program and predicates before publication. Rejections must drive bounded
author/compiler repair while the requested instruction remains unchanged.

For trajectory evaluation, the primary manipuland must remain visible through
the terminal state. Prefer low, flat, high-contrast planar destinations for
place or push tasks. A stack target needs a broad visible top. An insertion
fixture needs a large visible opening and must leave the inserted object
observable. Preserve the requested task even when it involves a container or drawer.
Choose geometry and views that expose the tracked action; if visibility cannot
be certified, report that input failure rather than silently substituting a
different relation.

Create real semantic diversity across a suite: vary the manipulation relation,
object geometry, materials, colors, and workspace context where compatible.
Changing only a seed, name, color, or layout is not a new task. Conversely, do
not force every case to use a different controller family when that would make
the suite less executable.

Return only the requested strict JSON object. Do not return case ids, seeds,
render sizes, rollout lengths, WAM prompts, thresholds, recipes, or explanatory
text; the harness owns those fields.

For project articulated tasks, apply the coded gates in
[robotics input contracts](../../docs/ROBOTICS_INPUT_CONTRACTS.md). Use a short
object-specific goal; distinguish drawer layer or handle identity. Geometric
projection is not proof of no occlusion: review rendered initial views. Use
automatic reference-prefix pose selection only with a valid reference, and
label it as a changed initial condition. Preserve technically valid failures.
