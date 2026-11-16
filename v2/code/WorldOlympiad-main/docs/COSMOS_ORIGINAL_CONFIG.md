# Cosmos3: original inference and simulation settings

The current default is the released Nano RoboLab serving configuration, not a
prompt format inferred from a separate training recipe. The original public
Nano example does not force JSON; the Edge example explicitly does. See the
[official serving cookbook](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/generator/action/run_policy_with_cosmos_framework.md).

`launch_cosmos3_robolab_server.py` defaults to `--prompt-format upstream`.
Explicit `--prompt-format json` and `--prompt-format plain` are available for
declared comparisons. Each response reports the formatter actually present in
the service; each window saves this attestation. Reports must not hardcode
`format_prompt_as_json=True`, and an episode cannot mix formats between windows.

The released prediction settings remain unchanged: 4 denoising steps, guidance
3, shift 5, 15 Hz, 32 future joint-position actions, 7 joint radians plus the
official binary gripper conversion. The seed is declared before inference;
the upstream request RNG advances normally. The input is 640×540 RGB, with a
360-row wrist panel and two 320×180 exterior panels. The 528-row decoded output
has a bottom crop; the panel split remains at row 360.

## One simulation profile

The three project Cosmos runners now default to
`deployment/seetacloud/robolab_simulation_profile.json`. It was exported from an
actual original RoboLab run on the existing Isaac 5.1 installation and retains
the original GPU PhysX settings, solver limits, material defaults, rendering
configuration and buffer sizes. Its source export digest is recorded.

- Physics device: `cuda:0`.
- Physics timestep: 1/120 s; 8 substeps per 15 Hz control step.
- Maximum position solver iterations: 32.
- Default prediction execution horizon: 32.

`autoresearch/cosmos_simulation_profile.py` constructs the Isaac configuration
from that record. A changed timestep that would produce 20 Hz control is
rejected. New captures record the profile digest, and resume rejects a profile
change.

Historical CPU behavior remains available only through
`--legacy-cpu-physics`. For the old forced-JSON inference behavior, also start
the service with `--prompt-format json`. Use a separate output directory for a
reproduction; never overwrite the original capture or silently resume it with
different physics. Older incomplete captures without per-window formatter
evidence cannot be treated as newly attested runs.

`--execute-actions-per-window 8` is a diagnostic option in the articulated
runner. Its 2026-09-10 control did not improve opening the cabinet. It is not
the default or the official evaluation profile.

## What alignment establishes

Using the same inference and simulation settings does not make custom SceneCode
geometry, object size, placement, lighting or task difficulty identical to a
RoboLab scene. Those remain explicit task data. In particular, the historical
cup scene obscures much of the cup in the initial wrist image and makes it
small in exterior views. Seven cup variants do not constitute seven diverse
object/scene tests.

On 2026-09-10 the same recorded door reference completed in original GPU
physics (approximately 69° opening, released and stable), while the model
trial opened only approximately 8.4°. A complete unmodified original
BananaInBowlTask episode also failed at grasping/placement. These observations
do not justify claiming that the checkpoint is broken or that every failure
has been fixed. The official overall default-instruction success rate is
36.8% across RoboLab-120, not a guarantee for any individual episode:
[official leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html).

The custom cup-in-tray task completed twice with the original GPU profile,
in 163 and 497 actions, whereas the same task/prompt/model seed under the old
CPU profile timed out after 750 actions. Fourteen identity/inference fields
matched in the controlled CPU/GPU-profile comparison. Both repetitions used
the same model seed and profile; their initial images differed slightly and
the action sequences were not bitwise identical. These two successes establish
a practical improvement, not a general success-rate estimate. The exported
raw imagination was inspected and shows approach, pickup and transport into
the tray; no execution video was substituted.

All diagnostic trials are separate from historical v15 scores and videos.
Detailed experiment records are under server `outputs/model_repair_20260910`.
