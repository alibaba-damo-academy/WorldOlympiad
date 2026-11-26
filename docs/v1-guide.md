# WorldOlympiad 1.0 Quickstart Guide

The full **WorldOlympiad 1.0** codebase (*Can Your World Model Survive a Triathlon?*, [arXiv:2606.11129](https://arxiv.org/abs/2606.11129)) is maintained on the dedicated **`v1`** branch.

## Checkout `v1`

```bash
git clone -b v1 https://github.com/alibaba-damo-academy/WorldOlympiad.git
cd WorldOlympiad
```

Or browse online: [github.com/alibaba-damo-academy/WorldOlympiad/tree/v1](https://github.com/alibaba-damo-academy/WorldOlympiad/tree/v1)

## Included Modules (`v1` branch)

- `physical/`, `mechanics/`, `material/`, `thermotics/` — Physical faithfulness judges
- `3d_metrics/` — 3D Gaussian-splat reconstruction, meta-view & camera alignment
- `interaction/` — Chunk, transition, global & CLIP interaction fidelity
- `judge_pipeline.py` & `batch_test/` — Unified evaluation entry points

See the full installation and dataset instructions in [`v1/README.md`](https://github.com/alibaba-damo-academy/WorldOlympiad/blob/v1/README.md) or [`v1/README_ZH.md`](https://github.com/alibaba-damo-academy/WorldOlympiad/blob/v1/README_ZH.md).
