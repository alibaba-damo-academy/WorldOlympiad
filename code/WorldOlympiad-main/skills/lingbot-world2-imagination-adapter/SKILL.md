---
name: lingbot-world2-imagination-adapter
description: Run the pinned LingBot-World2 14B causal-fast first-frame, text, and camera-conditioned imagination adapter. Use for World2 open-loop video generation and its artifact checks; do not use it as LingBot-VA control or claim closed-loop robotics execution.
---

# LingBot-World2 Imagination Adapter

Use `lingbot_world2_14b_causal_fast` only through the typed adapter. It is a fourth model profile with protocol `open_loop_first_frame_text_camera_video`; it is not one of the LingBot-VA action/control paths.

## Workflow

1. Read [native-interface.md](references/native-interface.md) and [prompt-and-output.md](references/prompt-and-output.md).
2. Prepare or validate a versioned `input_contract.json` with `python -m autoresearch.lingbot_world2_adapter`.
3. Invoke only the typed `run_lingbot_world2` tool with the frozen input contract, a new or resumable `run_id`, one inspected GPU, and the exact required protocol.
4. Wait for the managed job. Accept success only when `run_manifest.json` is complete and the result reports finite raw tensor plus a full frame-count/FPS/dimension decode.
5. Reuse a complete identical request. Do not retry unchanged valid poor output, change prompt/model/duration through the run tool, or claim physical success from video appearance alone.

The adapter is additive. Do not route this profile through `robotics_agentic_harness` or `lingbot_robotwin_harness`, and do not change the semantics of the three existing WAM profiles.
