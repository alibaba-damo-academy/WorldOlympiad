"""Unit tests for robot_eval structured LLM utilities."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic import BaseModel

from scenecode.robot_eval import llm_utils


class DummyOutput(BaseModel):
    value: str


@pytest.fixture(autouse=True)
def clear_client_cache():
    """Reset cached clients between tests."""
    llm_utils._clients.clear()
    yield
    llm_utils._clients.clear()


def test_get_client_reuses_same_api_base(monkeypatch):
    """Repeated lookups with the same api_base should reuse the cached client."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch(
        "scenecode.robot_eval.llm_utils.create_async_openai_client"
    ) as mock_create_client:
        client = MagicMock()
        mock_create_client.return_value = client

        first = llm_utils._get_client(api_base="https://api.example.com/v1")
        second = llm_utils._get_client(api_base="https://api.example.com/v1")

    assert first is client
    assert second is client
    mock_create_client.assert_called_once_with(api_base="https://api.example.com/v1")


def test_get_client_separates_different_api_bases(monkeypatch):
    """Different api_base values should not share the same cached client."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch(
        "scenecode.robot_eval.llm_utils.create_async_openai_client"
    ) as mock_create_client:
        client_one = MagicMock()
        client_two = MagicMock()
        mock_create_client.side_effect = [client_one, client_two]

        first = llm_utils._get_client(api_base="https://api.one/v1")
        second = llm_utils._get_client(api_base="https://api.two/v1")

    assert first is client_one
    assert second is client_two
    assert mock_create_client.call_count == 2


@pytest.mark.asyncio
async def test_structured_llm_call_passes_api_base(monkeypatch):
    """Structured LLM calls should use the configured api_base."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    parsed_output = DummyOutput(value="ok")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(parsed=parsed_output, refusal=None)
            )
        ]
    )

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=response)

    with patch(
        "scenecode.robot_eval.llm_utils.create_async_openai_client",
        return_value=mock_client,
    ) as mock_create_client:
        result = await llm_utils.structured_llm_call(
            model="gpt-5.2",
            system_prompt="system",
            user_input="user",
            output_type=DummyOutput,
            api_base="https://api.example.com/v1",
        )

    assert result == parsed_output
    mock_create_client.assert_called_once_with(api_base="https://api.example.com/v1")


@pytest.mark.asyncio
async def test_close_cached_clients_closes_and_clears_cache():
    """Cached robot_eval clients should be closed and removed during cleanup."""
    client_one = MagicMock()
    client_one.aclose = AsyncMock()
    client_two = MagicMock()
    client_two.aclose = AsyncMock()
    llm_utils._clients[("https://api.one/v1", "k1")] = client_one
    llm_utils._clients[("https://api.two/v1", "k2")] = client_two

    await llm_utils.close_cached_clients()

    client_one.aclose.assert_awaited_once()
    client_two.aclose.assert_awaited_once()
    assert llm_utils._clients == {}
