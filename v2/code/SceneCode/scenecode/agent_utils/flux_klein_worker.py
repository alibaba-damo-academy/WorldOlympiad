import argparse
import gc
import logging
import os
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from diffusers import Flux2KleinPipeline
from PIL import Image

from scenecode.agent_utils.gpu_diagnostics import describe_gpu_snapshot

LOGGER = logging.getLogger("scenecode.flux_klein_worker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLUX.2-klein image generation")
    parser.add_argument("--mode", choices=["generate", "edit"], required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--reference-image-path")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def build_pipeline(model_path: str) -> tuple[Flux2KleinPipeline, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
        device_map=None,
        low_cpu_mem_usage=False,
    )
    pipe = pipe.to(device)
    return pipe, device


def cleanup_worker_resources(
    pipe: Flux2KleinPipeline | None = None,
    generated_image: Image.Image | None = None,
    reference_image: Image.Image | None = None,
) -> None:
    if generated_image is not None:
        try:
            generated_image.close()
        except Exception:
            pass
    if reference_image is not None:
        try:
            reference_image.close()
        except Exception:
            pass

    if pipe is not None:
        del pipe
    if generated_image is not None:
        del generated_image
    if reference_image is not None:
        del reference_image

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


def main() -> None:
    _configure_logging()
    args = parse_args()
    LOGGER.info(
        "Starting FLUX worker pid=%s mode=%s output=%s",
        os.getpid(),
        args.mode,
        args.output_path,
    )
    LOGGER.info(describe_gpu_snapshot("FLUX worker start"))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipe: Flux2KleinPipeline | None = None
    generated_image: Image.Image | None = None
    reference_image: Image.Image | None = None

    try:
        pipe, _ = build_pipeline(args.model_path)
        LOGGER.info(describe_gpu_snapshot("FLUX worker after pipeline load"))

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        call_kwargs = {
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "max_sequence_length": args.max_sequence_length,
            "generator": generator,
        }

        if args.mode == "edit":
            if not args.reference_image_path:
                raise ValueError("--reference-image-path is required for edit mode")
            with Image.open(args.reference_image_path) as source_image:
                reference_image = source_image.convert("RGB")
            call_kwargs["image"] = reference_image

        with torch.inference_mode():
            generated_image = pipe(**call_kwargs).images[0]

        generated_image.save(output_path)
        LOGGER.info("Saved FLUX output to %s", output_path)
    except Exception:
        LOGGER.exception("FLUX worker failed")
        LOGGER.info(describe_gpu_snapshot("FLUX worker exception"))
        raise
    finally:
        cleanup_worker_resources(
            pipe=pipe,
            generated_image=generated_image,
            reference_image=reference_image,
        )
        LOGGER.info(describe_gpu_snapshot("FLUX worker cleanup"))


if __name__ == "__main__":
    main()
