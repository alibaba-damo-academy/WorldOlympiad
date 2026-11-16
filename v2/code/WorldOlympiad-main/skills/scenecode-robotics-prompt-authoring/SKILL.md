---
name: scenecode-robotics-prompt-authoring
description: "Compile or repair concise SceneCode prose from selected typed task facts and geometry constraints. Use for scene prompts and asset descriptions; task-suite authoring belongs to robotics-eval-task-authoring."
---

# SceneCode Robotics Prompt Authoring

Use SceneCode's own prompt idiom: a short, concrete description of a room and
its objects in their initial positions. This follows the upstream examples in
`examples/prompts.csv` and the rules and few-shot examples in
`scenecode/prompts/data/robot_eval/scene_prompt_generator.yaml`. The skill
defines syntax and semantic ownership only. It never supplies a case catalog,
object names, colors, room theme, or final task content.

## Model-authored task fields

Consume the selected typed task before compiling prose; use the task-authoring
skill when a new task suite is needed:

1. `scene_brief`: one or two natural sentences naming the setting, primary
   support, required task objects, and useful spatial context. Write like a
   SceneCode sample, not like an instruction manual or validator checklist.
2. `visual_description`: one singular noun phrase per object. Include only
   identity-bearing appearance and construction facts that must survive asset
   generation: color, material, shape, opening/handle/hinge, and explicitly
   absent parts when relevant. Describe a complete object, not a Blender recipe.
3. `initial_state`: concrete physical relations before action. State what is on,
   beside, inside, attached to, raised, closed, face-up, or otherwise installed.
   It must be task-ready and unsolved.
4. `instruction`: one short imperative. Do not add rollout duration, camera
   text, success explanation, or controller implementation.
5. `action_program` and `success_predicates`: typed execution/evaluation data;
   never encode these by changing prose or a case name.

For video-trajectory CAE cases, preserve primary-manipuland observability through
the terminal state. Prefer low, flat, high-contrast planar destinations when
they express the intended task. Do not place the tracked object inside a tall
box, bin, opaque enclosure, or other destination that hides its terminal path.
For requested containment or articulated tasks, preserve the task and certify
adequate trajectory visibility in the selected views. Do not replace the task
merely to match a convenient camera.

## Compiled SceneCode prompt

Compile the task into one to five natural-language sentences in this order:

1. room function, metric size, style, and the primary support;
2. complete required objects and their initial support/host relations;
3. reachable, separated layout and a clear manipulation corridor;
4. exact unsolved initial state;
5. one compact lighting/readability sentence.

Do not echo the robot instruction, success predicate, controller, model name,
goal relation, operator geometry formula, render resolution, visibility
threshold, stage ownership, or admission policy
into the SceneCode prompt. Those stay in typed harness records and downstream
validators. The scene prompt only creates an enabling, non-goal scene, exactly
as SceneCode's official task prompt generator does.

`scene_brief` guides task generation and review but need not be copied verbatim
when the compiler has a more compact sentence using the typed objects. Preserve
its meaning, not duplicate prose.

Stage isolation is a runtime concern, not prompt prose. The robotics SceneCode
profile disables stage-level repair critiques because they can reinterpret a
goal relation as a request to pre-solve the static scene, or try to repair
objects outside their tool scope. Asset-level plan checking, mesh/VLM validation,
collision checking, and placement remain enabled, and the completed scene must
still pass the harness inventory, identity, unsolved-state, physics, geometry,
and visibility gates.

Carry each exact `visual_description` once in the object sentence. Do not repeat
it as a category, an attribute list, a construction recipe, and a negative
constraint. Category is typing only. Do not ask SceneCode to judge a later-stage
manipuland during the furniture stage.

Identity-bearing structural words must describe visible geometry, not merely an
asset label. The generic compiler may clarify `open-top` or `hollow` with one
short clause requiring a bottom, four enclosing side walls, a visibly
unobstructed empty interior, and no top panel; likewise, a `through-hole` must
be visibly open end to end. These
clarifications come from the selected description itself and are not case
recipes or task-goal prose.

## Object/background contrast

Plan each task object's appearance jointly with its support, destination and
immediate background. The manipulated object and target must be readily
distinguishable through color and lightness under the scene's actual lighting.
Do not accept similar gray or beige materials that blend together merely
because the objects have different labels or recognizable standalone shapes.
Visual separation and plausible appearance are joint requirements. Unless the
task explicitly requests an unusual finish, choose materials, colors and
coatings commonly plausible for that object: natural-looking wood grain,
credible metal finishes, or familiar household enamel/plastic colors. Do not
use implausible recoloring or exaggerated saturation merely to increase
contrast. When plausible object colors blend with their surroundings, first
consider compatible support/background materials or lighting. Household
furniture may use natural wood when it fits the task, without imposing wood
or a fixed color recipe on unrelated scenes.

Carry the chosen colors and materials in the authoritative task descriptions
and their owning SceneCode requests. Preserve explicitly requested appearance;
resolve poor separation through other compatible scene choices or the existing
task-repair path, rather than silently changing task identity.

Inspect actual RGB renders with the objects together, including their nearby
background and the robot at the selected native views. Require both readable
object/background separation and material/color plausibility for the declared
objects; reject a visually distinct but implausibly colored result. Check initial,
manipulation and terminal phases when reviewing a reference. Identity contact
sheets, semantic masks and pixel counts alone do not prove visual separation.
Before a full render, use contextual stills when available to catch blending
early; verify the completed video as well.

If the object blends into its surroundings, withhold visual-quality acceptance,
record the affected object/background pair and rendered evidence, and return
that feedback to the owning autonomous repair stage. Adjust compatible
appearance or lighting, rerender and inspect again. Keep the existing identity,
physical and native-view checks; do not use post-render recoloring or overlays
to manufacture passing evidence.

## Geometry-safe language

- Say `single-piece` or `one coherent object` when disconnected components
  would contradict identity.
- `symmetric` describes appearance; it must not imply a Mirror construction.
  A complete cylinder, plate, lid, button, peg, bowl, or other monolithic shape
  is already symmetric. Only an explicitly modeled half/side may request a
  mirrored counterpart.
- Derive paired geometry from typed operators, never a case name: insert binds
  subject cross-section to clear aperture; cover binds lid to opening; stack
  binds footprint to support area. Reject a visually correct pair that cannot
  physically execute the selected operator.
- Avoid arbitrary decoration, glossy/transparent task surfaces, printed text,
  proxy robots, robot-zone markers, plinths, and pre-solved layouts unless the
  selected task explicitly requires them.

## Admission

Reject a task prompt before SceneCode when it loses an exact appearance phrase,
contradicts an initial relation, introduces a goal-state relation as an initial
fact, contains timing/model-adapter details, or expands into repetitive policy
prose. Prompt admission does not certify generated geometry; final identity and
physics gates remain mandatory.

After rendering, require deterministic frame-zero semantic masks from the
renderer. Every selected adapter checks all required task-object instances and
the end effector in its own native views. A visibility failure routes back to
asset scale, contrast, layout, or camera framing; it never creates a case skill
or a fixed object recipe. Do not use post-render resizing to make the gate pass.
For trajectory-scored tasks, admission also requires the primary manipuland to
remain observable at the terminal state; a high-wall destination may not pass
merely because the object was visible at frame zero.

Before SceneCode's VLM may accept a structure-bearing generated mesh, run the
deterministic triangle gate implied by the exact description. An open vessel
must expose an open end, a bottom, clear interior, and enclosure in both side
axes; a through-hole must have a clear center channel inside a surrounding
solid body. Return failed check names to SceneCode's existing per-item router
retry. This retries only the rejected asset while preserving already-valid
siblings and the room/furniture checkpoint. If all per-item attempts fail, keep
the case unpublished and retain only the clean pre-manipuland checkpoint.

Typed geometry_constraints are part of task identity. Propagate component
constraints into the owning host asset request. Local contact-frame metadata
belongs to binding/physics data, not prose that substitutes for verification.

## RSI feedback loop

When a new task fails, treat the failure as evidence about a reusable capability,
not as permission to add a case-specific recipe. Read the run ledger and RSI
traces, classify the failure as prompt, asset geometry, scale, camera,
articulation, operator, or evaluator, and state one falsifiable hypothesis.
The hypothesis may change a generic size heuristic, camera envelope,
articulation contract, or operator adapter; it may not hard-code a task id,
coordinates, or a previously solved terminal state.

Inspect prior lessons and apply one focused repair to the owning shared
capability. Run the relevant tests, retry the affected stage, and record the
actual result in its repair case card. Failed tests roll back the patch.
Retrieve the lesson on later matching tasks and record whether reuse succeeds.
Broaden relevant regression for changes to evaluators or model protocols;
ordinary repairs do not require an isolated candidate or frozen split plan.
Preserve original case evidence and published references.

For SceneCode specifically, RSI may refine the typed prompt compiler's generic
geometry wording, object-count normalization, reachable layout, or view
envelope. SceneCode remains a black-box generator: do not patch its source or
teach the loop a task-specific prompt recipe. A generated scene still needs the
normal identity, physics, visibility, provenance, and native-adapter gates.

For the shared research, repair, and reuse workflow, read
[references/rsi-capability-loop.md](references/rsi-capability-loop.md).
