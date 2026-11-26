# Source and license inventory

This inventory describes the source snapshot in this directory. Each bundled
project retains its own license; no project-wide license is granted by this
file. The exact source revisions are recorded in `sources.lock.json`.

| Bundled path | Source | License included in the snapshot | Relevant condition |
| --- | --- | --- | --- |
| `code/SceneCode/` | [wangpuyi/SceneCode](https://github.com/wangpuyi/SceneCode) | [`code/SceneCode/LICENSE`](code/SceneCode/LICENSE), MIT | Preserve its copyright and license notice. |
| `code/RoboTwin/` | [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | [`code/RoboTwin/LICENSE`](code/RoboTwin/LICENSE), MIT | Preserve its copyright and license notice. |
| `code/lingbot-va/` | [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va) | [`code/lingbot-va/LICENSE.txt`](code/lingbot-va/LICENSE.txt), Apache-2.0 | Retain notices and mark modifications where required. Model checkpoints are separate. |
| `code/Depth-Anything-3/` | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | [`code/Depth-Anything-3/LICENSE`](code/Depth-Anything-3/LICENSE), Apache-2.0 | Retain notices and mark modifications where required. Model checkpoints are separate. |
| `code/curobo-0.7.8/` | [NVlabs/curobo at the pinned commit](https://github.com/NVlabs/curobo/tree/d64c4b005459db10c5dd867d8b30a87d5bda9bdb) | [`code/curobo-0.7.8/LICENSE`](code/curobo-0.7.8/LICENSE), NVIDIA License; [`LICENSE_ASSETS`](code/curobo-0.7.8/LICENSE_ASSETS) covers additional assets | The pinned license limits use of the work and derivatives to non-commercial research or evaluation, except for NVIDIA and its affiliates. Redistribution must retain the license and notices. This is not an unrestricted open-source grant. |
| `assets/robot_assets/` | Generated robot asset bundle described in `assets.lock.json` | PickNik BSD-3-Clause notice is included as `PICKNIK_LICENSE` | The bundle also records pybullet and RoboLab inputs. Their complete redistribution terms have not been established here. |

## Project-owned and recovered code

The upstream [WorldOlympiad 1.0 repository](https://github.com/alibaba-damo-academy/WorldOlympiad)
has no project-level `LICENSE` in its `main` branch. This snapshot's
`code/WorldOlympiad-main/` and `code/scenecode_pybullet_minimal/` also have no
license file. Public availability, Git commit authorship and push permission do
not by themselves establish terms for reuse or redistribution. The project
maintainers and relevant rights holders need to choose and authorize a license
for their code. No license can be inferred from WorldOlympiad 1.0.

The generated robot asset bundle and any separately obtained weights, datasets,
simulator assets or model checkpoints need their own provenance and terms.
Adding a project-level license later must not erase or override the bundled
projects' licenses, especially cuRobo's non-commercial restriction.
The current cuRobo `main` branch may display different license terms; the
license in this pinned snapshot governs this bundled copy.
