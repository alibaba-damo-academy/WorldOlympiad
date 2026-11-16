"""Lightweight LLM utilities for robot_eval module.

Provides simple async structured LLM calls without Agent SDK overhead.
Uses OpenAI's structured output API for guaranteed schema compliance.
"""

import logging
import os

from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from scenecode.utils.openai import create_async_openai_client, get_openai_api_base

console_logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Lazy-initialized clients keyed by endpoint and API key.
_clients: dict[tuple[str, str | None], AsyncOpenAI] = {}


def _get_client(api_base: str | None = None) -> AsyncOpenAI:
    """Get or create the async OpenAI client."""
    resolved_api_base = get_openai_api_base(api_base=api_base)
    cache_key = (resolved_api_base, os.environ.get("OPENAI_API_KEY"))
    if cache_key not in _clients:
        _clients[cache_key] = create_async_openai_client(api_base=resolved_api_base)
    return _clients[cache_key]


async def close_cached_clients() -> None:
    """Close and clear cached async clients used by robot_eval helpers."""
    cached_clients = list(_clients.values())
    _clients.clear()

    for client in cached_clients:
        try:
            await client.aclose()
        except Exception as exc:
            console_logger.warning(
                "Failed to close cached robot_eval OpenAI client during cleanup: %s",
                exc,
            )


async def structured_llm_call(
    model: str,
    system_prompt: str,
    user_input: str,
    output_type: type[T],
    api_base: str | None = None,
) -> T:
    """Make an async LLM call with structured Pydantic output.

    Uses OpenAI's structured output API which constrains the LLM to only
    produce valid schema-compliant output. Simpler than Agent SDK for
    single-turn LLM calls without tools.

    Args:
        model: Model name (e.g., "gpt-5.2").
        system_prompt: System instructions for the LLM.
        user_input: User message/query.
        output_type: Pydantic model class for structured output.

    Returns:
        Parsed Pydantic model instance.

    Raises:
        OpenAI API errors if the call fails.
    """
    client = _get_client(api_base=api_base)

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        response_format=output_type,
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(
            f"Failed to parse structured output. "
            f"Refusal: {response.choices[0].message.refusal}"
        )

    return parsed
