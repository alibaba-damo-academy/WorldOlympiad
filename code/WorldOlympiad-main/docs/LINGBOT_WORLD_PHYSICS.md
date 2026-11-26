# LingBot-World v2 1.3B physics pilot

Use the official `Robbyant/lingbot-world-v2-1.3b-causal-fast` release for a
first-frame/text/camera-conditioned physics diagnostic. This is a video world
model, not an interface for numerical forces, masses or simulator state tensors.

## Server runtime

- Workspace: `/root/autodl-tmp/worldolympiad2`.
- Official code: `code/lingbot-world-v2`, commit `1895d300d8ac936401689b26389f51cbd36530eb`.
- Weights: `models/lingbot-world-v2-1.3b-causal-fast`.
- Runtime: `envs/lingbot-world-v2/bin/python`.
- The venv has its own Transformers 4.51.3, SciPy and FlashAttention 2.8.3.
  It reads the existing `envs/lingbot` Torch/CUDA runtime through a .pth entry;
  it does not alter that environment. Preserve the pinned shared Torch 2.9.0+cu128 runtime.
- GPU jobs use the existing workspace-wide `exclusive_gpu_job(outputs, "0")` lease.

ModelScope provided the six DiT shards plus the shared 14B-release T5, VAE and
tokenizer assets (18.729 GB total). Each file's official SHA256 is recorded and
verified. Only shared assets are taken from the 14B package, not its DiT weights.
The nominal 1.3B release contains 1,709,502,016 state elements with camera modules;
all 1,071 checkpoint tensors matched the released task architecture.

Sources: [official repository](https://github.com/Robbyant/lingbot-world-v2),
[ModelScope 1.3B](https://modelscope.cn/models/Robbyant/lingbot-world-v2-1.3b-causal-fast).

## Native input contract

`autoresearch/lingbot_world_physics.py` prepares version
`lingbot_world_v2_native_render_physics.v2`:

1. Render the existing SceneCode assets and saved PyBullet trajectory directly
   at **832x464** with `scripts/blender_physics_native_reference.py`.
   The wrapper uses the shared Blender replay without changing it and verifies
   actual Blender projections against the declared intrinsics.
2. Retain the physical camera pose and vertical FOV. The wider native aspect
   expands horizontal FOV; this is a fresh render, not a stretched image.
3. Reject a first frame whose size would change under the released model's
   own dimension calculation. The nominal 832x480 input evaluates to 832x464
   in the current floating-point/floor implementation. 832x464 is a fixed point.
4. Save the first RGB unchanged. Model input and output are both 832x464;
   same-size upstream bicubic interpolation is verified pixel-identical.
5. Supply exact identity OpenCV c2w poses relative to camera zero for every frame.
   Keep the original absolute calibration in provenance. Exact identity prevents
   tiny floating-point translations from being normalized into artificial motion.
6. Encode intrinsics.npy as rows [fx,fy,cx,cy] in the upstream's expected
   virtual **832x480 pixel units**, then verify the upstream transformation
   recovers the actual render intrinsics. This changes coordinate units only,
   not RGB pixels.
7. Pass a natural-language paragraph describing the scene and initial trigger.
   Never pass reference future RGB, target trajectories or future object states.
   Any requested numeric initial speed is text conditioning, not guaranteed control.

## Timing and released settings

The four saved physics references each retain 61 frames at 15 fps (4 s between
first and last frame). At the released 16 fps and 4-latent-frame chunk size,
the smallest complete causal generation covering that interval is 77 frames
(4.75 s between first and last frame; 4.8125 s media duration).
Neither video is stretched, looped or truncated to match the other.

The pilot uses the public `WanI2VCausal.generate` defaults:
causal_fast, flow shift 5.0, timestep indices [0,250,500,750],
local attention 18, sink 6, seed 42, one GPU, no FSDP or Ulysses,
and original model weights. The private fast method has different unused
default indices; the public entry point governs this run.

## Reproduction

After preparing a fresh plan and verifying weights/runtime:

```bash
envs/lingbot-world-v2/bin/python code/WorldOlympiad-main/scripts/run_lingbot_world_physics.py \
  --workspace /root/autodl-tmp/worldolympiad2 \
  --upstream-root /root/autodl-tmp/worldolympiad2/code/lingbot-world-v2 \
  --checkpoint /root/autodl-tmp/worldolympiad2/models/lingbot-world-v2-1.3b-causal-fast \
  --plan /root/autodl-tmp/worldolympiad2/outputs/physics_lingbot_world_v2_20260911/pilot_plan.json \
  --output /root/autodl-tmp/worldolympiad2/outputs/physics_lingbot_world_v2_20260911/NEW_TRIAL
```

The runner checks the completed download, architecture and CUDA receipts;
records the exact prompt and conditioning hashes before calling the model;
preserves raw prediction tensors; and fully decodes each native output.
The official lake example is a separate installation control using its released
input and normal upstream preprocessing. The four physics cases use direct
native renders with no image resize.

## First pilot and limits

The official lake control and four physics cases completed. Each physics clip
took about 12 seconds and peaked at 26.51 GiB of allocated GPU memory.
All clips contain 77 finite, decodable frames at 16 fps.

Qualitative findings on the saved first-seed outputs:
- Free fall: the block remains suspended and gains a thin generated line.
- Collision: the red block moves but visibly changes shape; contact response
  differs from the reference.
- Ramp roll: motion begins late, deviates from the slope, and the ball becomes
  unidentifiable near the end.
- Domino chain: the row remains upright through the clip.

These are observations on this four-case pilot, not a general benchmark ranking
or a proof that prompt/distribution differences are irrelevant.
The first-frame global RGB MAE is about 1.86–2.09 on the 0–255 scale, but background
dominates that statistic. It is not sufficient alone to validate small objects.

Passive physics has no initial robot/grasp registration. Do not manufacture a
robotics EE-trajectory CAE score. The current pilot leaves numeric physics scores
null; use an explicitly defined passive-physics evaluation protocol for later scoring.
