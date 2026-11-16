import base64
import logging
import os

from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np

from agents import RunConfig, set_default_openai_client
from openai import AsyncOpenAI, OpenAI
from PIL import Image

DEFAULT_OPENAI_API_BASE = os.environ.get(
    "OPENAI_API_BASE", "https://api.openai.com/v1"
)
_DEFAULT_ASYNC_CLIENTS: dict[str, AsyncOpenAI] = {}
console_logger = logging.getLogger(__name__)


def _coerce_usage_int(value: Any) -> int:
    """Safely coerce usage counters to ints, defaulting unknown values to zero."""

    return value if isinstance(value, int) else 0


def _get_config_value(config: Any, *keys: str) -> Any:
    """Fetch a nested value from a dict-like or attribute-based config."""
    current = config
    for key in keys:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def get_openai_api_base(cfg: Any | None = None, api_base: str | None = None) -> str:
    """Resolve the OpenAI API base from explicit input or config."""
    if api_base:
        return api_base

    cfg_api_base = _get_config_value(cfg, "openai", "api_base")
    if cfg_api_base:
        return str(cfg_api_base)

    direct_api_base = _get_config_value(cfg, "api_base")
    if direct_api_base:
        return str(direct_api_base)

    return DEFAULT_OPENAI_API_BASE


def create_openai_client(
    cfg: Any | None = None, api_base: str | None = None
) -> OpenAI:
    """Create a synchronous OpenAI client pinned to the configured API base."""
    return OpenAI(base_url=get_openai_api_base(cfg=cfg, api_base=api_base))


def create_async_openai_client(
    cfg: Any | None = None, api_base: str | None = None
) -> AsyncOpenAI:
    """Create an async OpenAI client pinned to the configured API base."""
    return AsyncOpenAI(base_url=get_openai_api_base(cfg=cfg, api_base=api_base))


def create_run_config(
    api_base: str | None = None, call_model_input_filter: Any | None = None
) -> RunConfig:
    """Create a RunConfig and ensure default async client uses configured base URL."""
    resolved_api_base = get_openai_api_base(api_base=api_base)
    default_client = _DEFAULT_ASYNC_CLIENTS.get(resolved_api_base)
    if default_client is None:
        default_client = create_async_openai_client(api_base=resolved_api_base)
        _DEFAULT_ASYNC_CLIENTS[resolved_api_base] = default_client

    set_default_openai_client(default_client)
    return RunConfig(call_model_input_filter=call_model_input_filter)


async def close_cached_async_openai_clients() -> None:
    """Close and clear cached async OpenAI clients."""
    cached_clients = list(_DEFAULT_ASYNC_CLIENTS.values())
    _DEFAULT_ASYNC_CLIENTS.clear()

    for client in cached_clients:
        try:
            await client.aclose()
        except Exception as exc:
            console_logger.warning(
                "Failed to close cached async OpenAI client during cleanup: %s", exc
            )


def extract_openai_usage_stats(
    response: Any, api: Literal["responses", "chat_completions"]
) -> dict[str, int]:
    """Normalize OpenAI SDK usage objects to a single logging schema."""

    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input": 0,
            "output": 0,
            "reasoning": 0,
            "cached": 0,
            "total": 0,
        }

    if api == "responses":
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        return {
            "input": _coerce_usage_int(getattr(usage, "input_tokens", 0)),
            "output": _coerce_usage_int(getattr(usage, "output_tokens", 0)),
            "reasoning": _coerce_usage_int(
                getattr(output_details, "reasoning_tokens", 0)
            ),
            "cached": _coerce_usage_int(getattr(input_details, "cached_tokens", 0)),
            "total": _coerce_usage_int(getattr(usage, "total_tokens", 0)),
        }

    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "input": _coerce_usage_int(getattr(usage, "prompt_tokens", 0)),
        "output": _coerce_usage_int(getattr(usage, "completion_tokens", 0)),
        "reasoning": _coerce_usage_int(
            getattr(completion_details, "reasoning_tokens", 0)
        ),
        "cached": _coerce_usage_int(getattr(prompt_details, "cached_tokens", 0)),
        "total": _coerce_usage_int(getattr(usage, "total_tokens", 0)),
    }


def log_openai_usage(
    response: Any,
    *,
    component: str,
    api: Literal["responses", "chat_completions"],
    model: str,
) -> None:
    """Log normalized usage for a successful OpenAI SDK response."""

    usage = extract_openai_usage_stats(response=response, api=api)
    input_tokens = _coerce_usage_int(usage.get("input", 0))
    output_tokens = _coerce_usage_int(usage.get("output", 0))
    reasoning_tokens = _coerce_usage_int(usage.get("reasoning", 0))
    cached_tokens = _coerce_usage_int(usage.get("cached", 0))
    total_tokens = _coerce_usage_int(usage.get("total", 0))
    console_logger.info(
        f"[OPENAI_USAGE] component={component}, "
        f"api={api}, "
        f"model={model}, "
        f"input={input_tokens:,}, "
        f"output={output_tokens:,}, "
        f"reasoning={reasoning_tokens:,}, "
        f"cached={cached_tokens:,}, "
        f"total={total_tokens:,}"
    )


def encode_image_to_base64(image: np.ndarray | str | Path) -> str:
    """Encodes an image to a base64 string.

    Args:
        image: Either a numpy array of shape (H, W, 3) in RGB format, a path string,
            or a Path object to an image file.

    Returns:
        str: The base64 encoded image string.
    """
    if isinstance(image, (str, Path)):
        # Read image directly from path.
        with Image.open(image) as img:
            # Convert to RGB in case it's not.
            img = img.convert("RGB")
            # Save to bytes.
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        # Convert numpy array to PIL Image.
        img = Image.fromarray(image)
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
