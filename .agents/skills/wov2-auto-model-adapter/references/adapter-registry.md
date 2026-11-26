# WorldOlympiad 2.0 Verified Adapter Registry (Self-Evolving Memory)

This registry stores verified model protocol profiles, dataset routes, and reusable adapter archetypes. The `wov2-auto-model-adapter` skill reads this file **before** creating a new adapter and appends newly validated models after Phase 4 passes.

---

## Verified Paper Archetypes

| Model | Route | Arm | Execution | HF Dataset Route (`dataset-route`) | Native FPS / Protocol Notes |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **LingBot-VA** | Registered Native (`sim_rollout` / `i2av`) | `dual` | `sim_and_real` | `--arm dual --execution sim_and_real` (`--model-split lingbot_va`) | 10 fps joint video-action; 3-view ALOHA/ARX5 conditioning; native RoboTwin adapter for WOV2 constructed scenes. |
| **Fast-WAM** | Registered Native / External `wov2_adapter.v1` | `dual` | `sim_and_real` | `--arm dual --execution sim_and_real` (`--model-split fastwam`) | Dual-arm 14-D qpos / fast video-action rollout; native RoboTwin bridge + baseline-camera replay. |
| **Long-WAM** | Registered Native / External `wov2_adapter.v1` | `dual` | `sim_and_real` | `--arm dual --execution sim_and_real` (`--model-split longwam_cod`) | 48-step observation history; IDM native simulator route distinct from COD video-only route. |
| **Cosmos3 (Nano-Policy-DROID)** | Registered Native / External `wov2_adapter.v1` | `single` | `sim_and_real` | `--arm single --execution sim_and_real` (`--model-split cosmos3`) | Direct RGB-conditioned video / DROID execution; preserves Panda exterior view; native FPS retained without resampling. |
| **GR00T N1.6-DROID** | Closed-Loop Sim Policy (`sim_rollout`) | `single` | `sim_only` | `--arm single --execution sim_only` (`--model-split gr00t_n1.6_droid`) | Action-chunk policy executed in ManiSkill3 / PyBullet-SAPIEN; requires executable sim environment (real-world videos excluded). |

---

## Reusable External Adapter Template (`wov2_adapter.v1`)

Place adapter JSON + wrapper under `outputs/adapters/<model_id>/`:

```json
{
  "schema_version": "wov2_adapter.v1",
  "model": "my_new_model_v1",
  "modes": ["i2av"],
  "minimum_free_mib": 16000,
  "gpu_count": 1,
  "command": [
    "/path/to/model_env/bin/python",
    "run_adapter.py",
    "--input", "{input}",
    "--output", "{output}"
  ]
}
```

Minimal wrapper contract (`run_adapter.py`):
1. Read JSON from `--input`: `case_id`, `instruction`, `conditioning["front"]` (initial PNG), `output_view`, `output_size_wh`.
2. Run the model's native inference on the initial image + instruction (never reading reference videos).
3. Save the generated video to `<output>/generated.mp4` matching `output_size_wh` (`[W, H]`).
4. Write `<output>/prediction.json`:
   ```json
   {
     "video": "generated.mp4",
     "view": "front"
   }
   ```

---

## Learned Community & Self-Evolved Entries

*(When `wov2-auto-model-adapter` validates a new model through Phase 4, append its profile, conversions, and validation receipt here.)*
