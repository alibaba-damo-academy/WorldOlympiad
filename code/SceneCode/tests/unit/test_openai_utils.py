"""Unit tests for shared OpenAI helper utilities."""

import pytest

from unittest.mock import AsyncMock

from scenecode.utils.openai import (
    DEFAULT_OPENAI_API_BASE,
    close_cached_async_openai_clients,
    create_run_config,
    extract_openai_usage_stats,
)


def test_create_run_config_uses_openai_base_url(monkeypatch):
    """Shared RunConfig helper should configure default client with base URL."""
    created_clients = []
    configured_clients = []

    def fake_create_async_openai_client(cfg=None, api_base=None):
        client = object()
        created_clients.append((api_base, client))
        return client

    def fake_set_default_openai_client(client):
        configured_clients.append(client)

    monkeypatch.setattr("scenecode.utils.openai._DEFAULT_ASYNC_CLIENTS", {})
    monkeypatch.setattr(
        "scenecode.utils.openai.create_async_openai_client",
        fake_create_async_openai_client,
    )
    monkeypatch.setattr(
        "scenecode.utils.openai.set_default_openai_client",
        fake_set_default_openai_client,
    )

    first_run_config = create_run_config()
    second_run_config = create_run_config()

    assert first_run_config is not None
    assert second_run_config is not None
    assert len(created_clients) == 1
    assert created_clients[0][0] == DEFAULT_OPENAI_API_BASE
    assert configured_clients == [created_clients[0][1], created_clients[0][1]]


def test_create_run_config_preserves_input_filter(monkeypatch):
    """Shared RunConfig helper should preserve input filter and api_base."""
    marker = object()
    created_clients = []
    configured_clients = []

    def fake_create_async_openai_client(cfg=None, api_base=None):
        client = object()
        created_clients.append((api_base, client))
        return client

    def fake_set_default_openai_client(client):
        configured_clients.append(client)

    monkeypatch.setattr("scenecode.utils.openai._DEFAULT_ASYNC_CLIENTS", {})
    monkeypatch.setattr(
        "scenecode.utils.openai.create_async_openai_client",
        fake_create_async_openai_client,
    )
    monkeypatch.setattr(
        "scenecode.utils.openai.set_default_openai_client",
        fake_set_default_openai_client,
    )

    run_config = create_run_config(
        api_base="https://api.example.com/v1",
        call_model_input_filter=marker,
    )

    assert run_config.call_model_input_filter is marker
    assert run_config is not None
    assert created_clients == [("https://api.example.com/v1", configured_clients[0])]


def test_extract_openai_usage_stats_for_responses_api():
    """Responses usage should map to the normalized logging schema."""

    class InputDetails:
        cached_tokens = 12

    class OutputDetails:
        reasoning_tokens = 34

    class Usage:
        input_tokens = 100
        input_tokens_details = InputDetails()
        output_tokens = 50
        output_tokens_details = OutputDetails()
        total_tokens = 150

    class Response:
        usage = Usage()

    assert extract_openai_usage_stats(Response(), api="responses") == {
        "input": 100,
        "output": 50,
        "reasoning": 34,
        "cached": 12,
        "total": 150,
    }


def test_extract_openai_usage_stats_for_chat_completions_api():
    """Chat completion usage should map to the normalized logging schema."""

    class PromptDetails:
        cached_tokens = 7

    class CompletionDetails:
        reasoning_tokens = 9

    class Usage:
        prompt_tokens = 80
        prompt_tokens_details = PromptDetails()
        completion_tokens = 20
        completion_tokens_details = CompletionDetails()
        total_tokens = 100

    class Response:
        usage = Usage()

    assert extract_openai_usage_stats(Response(), api="chat_completions") == {
        "input": 80,
        "output": 20,
        "reasoning": 9,
        "cached": 7,
        "total": 100,
    }


def test_extract_openai_usage_stats_defaults_missing_fields_to_zero():
    """Missing usage details should safely fall back to zero."""

    class Response:
        usage = None

    assert extract_openai_usage_stats(Response(), api="responses") == {
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cached": 0,
        "total": 0,
    }


@pytest.mark.asyncio
async def test_close_cached_async_openai_clients_closes_and_clears_cache(monkeypatch):
    """Cached async clients should be closed and removed during cleanup."""
    client_one = AsyncMock()
    client_two = AsyncMock()
    cache = {"one": client_one, "two": client_two}

    monkeypatch.setattr("scenecode.utils.openai._DEFAULT_ASYNC_CLIENTS", cache)

    await close_cached_async_openai_clients()

    client_one.aclose.assert_awaited_once()
    client_two.aclose.assert_awaited_once()
    assert cache == {}
