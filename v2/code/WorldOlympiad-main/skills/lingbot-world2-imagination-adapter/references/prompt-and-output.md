# Prompt and output contract

The event prompt must be one paragraph and state, in order:

1. the visible subject and scene relation;
2. the immediate physical trigger;
3. the ordered event chain and terminal visible state.

Do not append `camera remains stationary`, `background remains stationary`, or their combined form. Camera stability is already encoded by the camera condition and these phrases suppressed motion in the diagnosed case.

Example: `A red rubber ball is visible above a wooden table. The ball is released immediately. It falls straight down, strikes the table, bounces once, and settles.`

Published outputs are immutable per request identity. The harness stages raw tensor, MP4 and five sampled frames, verifies finite tensor shape, then fully decodes the MP4 to confirm dimensions, frame count and FPS before atomically publishing `artifacts/` and `run_manifest.json`.
