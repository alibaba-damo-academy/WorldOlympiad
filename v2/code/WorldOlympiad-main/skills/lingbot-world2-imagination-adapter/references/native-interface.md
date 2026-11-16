# Native interface

- Model: `Robbyant/lingbot-world-v2-14b-causal-fast`
- Wan task: `i2v-A14B`
- Inference mode: `causal_fast`
- Input: one native 832×464 RGB frame, one event prompt, fixed-camera `poses.npy`, and calibrated `intrinsics.npy`
- Output: 16 FPS RGB video; frame count is fixed by complete causal chunks
- Sampling: shift 10.0, chunk size 4, timesteps `[0, 250, 500, 750]`, local attention 18, sink size 6

The checkpoint, upstream commit, environment, config, adapter and runner are identity-bound. The run tool intentionally exposes no checkpoint, prompt, duration, shift, frame-count or encoder controls.

This is open-loop visual imagination. It has no action feedback, simulator state feedback, robot-controller authority or terminal-predicate certificate.
