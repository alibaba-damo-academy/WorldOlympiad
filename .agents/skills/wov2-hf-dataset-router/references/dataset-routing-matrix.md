# WorldOlympiad 2.0 Dataset Routing Matrix (`Chengzhiwww/WorldOlympiad2.0`)

Total benchmark pool: **1,100 cases** (900 public + 200 constructed), plus **300 separate real-robot bimanual demonstrations** (`real_robot/episodes.jsonl`, AgileX Piper, 10 tasks × 30 episodes × 3 views @ 10 fps).

## Cohort-to-Model Routing Table

| Model Archetype | `--arm` | `--execution` | Public Cohort (400) | Constructed Cohort (100) | Paper Reference Models |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Dual-Arm Hybrid (Sim when available, else Real)** | `dual` | `sim_and_real` | `RoboDojo` (300 sim, ALOHA-AgileX) + `Table30` (60 real, ALOHA) + `RoboCOIN` (40 real, AgileX Cobot Magic) | `Constructed/ARX5` (100 sim, 20 × 5 families) | `LingBot-VA`, `Fast-WAM`, `Long-WAM` |
| **Single-Arm Hybrid (Sim when available, else Real)** | `single` | `sim_and_real` | `ManiSkill3` (300 sim, Panda) + `RoboMIND` (65 real, Franka) + `Table30` (35 real, Franka) | `Constructed/Panda` (100 sim, 20 × 5 families) | `Cosmos3` |
| **Single-Arm Sim-Only Closed-Loop Policy** | `single` | `sim_only` | `ManiSkill3` (400 sim, Panda — executable trajectories) | `Constructed/Panda` (100 sim, 20 × 5 families) | `GR00T N1.6-DROID` |
| **Fast Smoke / In-Repo Constructed Only** | `dual` or `single` | `constructed_only` | *(skipped — zero upstream downloads needed)* | `Constructed/ARX5` or `Constructed/Panda` (canonical MP4s in HF repo) | Any new model during adapter onboarding |

## Full 1,100-Case Breakdown by Source, Embodiment, and Task Family

| Subset | Source | Domain | Embodiment | Arm | Task Families & Counts | Total |
| :--- | :--- | :---: | :--- | :---: | :--- | :---: |
| `public` | `RoboDojo` | `sim` | `ALOHA-AgileX` | `dual` | `grasp_manipulate` (120), `insert_align` (120), `stack` (60) | 300 |
| `public` | `Table30` | `real` | `ALOHA` | `dual` | `insert_align` (40), `stack` (20) | 60 |
| `public` | `RoboCOIN` | `real` | `AgileX Cobot Magic` | `dual` | `grasp_manipulate` (40) | 40 |
| `public` | `ManiSkill3` | `sim` | `Panda` | `single` | `grasp_manipulate` (160), `insert_align` (160), `stack` (80) | 400 |
| `public` | `RoboMIND` | `real` | `Franka` | `single` | `grasp_manipulate` (5), `insert_align` (40), `stack` (20) | 65 |
| `public` | `Table30` | `real` | `Franka` | `single` | `grasp_manipulate` (35) | 35 |
| `constructed` | `Constructed` | `sim` | `ARX5` | `dual` | `cabinet_articulation` (20), `cup_hanging` (20), `cup_uprighting` (20), `tissue_box_pushing` (20), `tool_manipulation` (20) | 100 |
| `constructed` | `Constructed` | `sim` | `Panda` | `single` | `cabinet_articulation` (20), `cup_hanging` (20), `cup_uprighting` (20), `tissue_box_pushing` (20), `tool_manipulation` (20) | 100 |

## Published Model Splits (`benchmark/splits/<model>.jsonl`)

- `fastwam`, `lingbot_va`, `longwam_cod`: shared 500-case dual-arm cohort (`--model-split lingbot_va`)
- `cosmos3`: 500-case single-arm video cohort (300 ManiSkill3 + 100 real + 100 Constructed/Panda)
- `gr00t_n1.6_droid`: 500-case single-arm sim-only cohort (400 ManiSkill3 + 100 Constructed/Panda)
