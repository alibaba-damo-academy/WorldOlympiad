"""
Headline: Generate one FLUX image from a single text prompt.

What this script does:
- Loads a local FLUX.2 model checkpoint with `diffusers.Flux2Pipeline`.
- Generates exactly one image from the input prompt.
- Saves the image to the target output directory with an auto-generated filename.

Usage:
python generate_flux_from_prompt.py \
  --prompt "your prompt text" \
  --output_dir "/path/to/output_dir"

Input:
- prompt (str): text prompt used to generate the image.
- output_dir (str): directory path where the generated image is saved.

Output:
- One PNG file in `output_dir`, filename format:
  `flux_YYYYMMDD_HHMMSS_seed{SEED}.png`
- Console logs for loading/generation timing and runtime status.
"""

import time
import argparse
from pathlib import Path

import torch
from diffusers import Flux2Pipeline


MODEL_PATH = "$FLUX_MODEL_PATH"

HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 4.0
SEED = 41

def build_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    t0 = time.time()
    pipe = Flux2Pipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        local_files_only=True,
        device_map="balanced" if device == "cuda" else None,
    )
    t1 = time.time()
    print(f"Pipeline loaded in {t1 - t0:.2f} seconds.")

    if device != "cuda":
        pipe.to(device)

    return pipe, device


def generate_image(pipe, prompt, output_dir):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"flux_{timestamp}_seed{SEED}.png"

    if save_path.exists():
        print(f"{save_path.name} already exists, skipping")
        return False

    item_start = time.time()
    print(f"Generating image -> {save_path}")

    full_prompt = prompt
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    with torch.inference_mode():
        image = pipe(
            prompt=full_prompt,
            height=HEIGHT,
            width=WIDTH,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            max_sequence_length=256,
            generator=generator,
        ).images[0]

    image.save(save_path)

    del image
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    item_end = time.time()
    print(f"Image finished in {item_end - item_start:.2f} seconds.")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one FLUX image from a prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the generated image will be saved.",
    )
    return parser.parse_args()


def main(prompt, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe, device = build_pipeline()
    print(f"Using device: {device}")

    total_start = time.time()
    success = generate_image(pipe, prompt, output_dir)

    total_end = time.time()

    if success:
        print(
            f"Generated 1 image in {total_end - total_start:.2f} seconds."
        )
    else:
        print("No new images generated.")

    print(f"Total elapsed time: {total_end - total_start:.2f} seconds.")


if __name__ == "__main__":
    args = parse_args()
    main(args.prompt, args.output_dir)