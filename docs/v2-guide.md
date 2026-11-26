# WorldOlympiad 2.0 Quickstart Guide

The full **WorldOlympiad 2.0** codebase (*Evaluating Embodied World Models with a Self-Evolving Agent Harness and Continuous Action Evaluation*) is maintained on the dedicated **`v2`** branch.

## Checkout `v2`

```bash
git clone -b v2 https://github.com/alibaba-damo-academy/WorldOlympiad.git
cd WorldOlympiad
```

Or browse online: [github.com/alibaba-damo-academy/WorldOlympiad/tree/v2](https://github.com/alibaba-damo-academy/WorldOlympiad/tree/v2)

## Verify Sources & Run Tests

```bash
# Verify SHA-256 tree integrity of all pinned source bundles (portable, no GPU needed)
./setup.sh sources-verify

# Configure agent API endpoint & tested model checkpoint
python3 scripts/configure.py \
  --agent-base-url https://HOST/v1 \
  --agent-model MODEL_ID \
  --checkpoint /absolute/path/to/checkpoint
python3 scripts/set_api_key.py
```

## Detailed Documentation

- [Benchmark Contract & Adapter Specification](benchmark.md)
- [Migration & Environment Requirements](migration-v1.md)
- [Acceptance Criteria](acceptance.md)
- [Native Multi-Model Validation](native-multimodel-validation.md)
- Full branch README: [`v2/README.md`](https://github.com/alibaba-damo-academy/WorldOlympiad/blob/v2/README.md)
