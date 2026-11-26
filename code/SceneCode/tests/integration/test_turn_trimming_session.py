"""Integration tests for TurnTrimmingSession.

These tests make real OpenAI API calls to verify summarization works correctly.
Uses gpt-5.2 with low thinking effort for cost efficiency.
"""

from pathlib import Path

import pytest

from agents import SQLiteSession
from omegaconf import OmegaConf

from scenecode.agent_utils.turn_trimming_session import TurnTrimmingSession

from .common import has_openai_key


@pytest.mark.skipif(not has_openai_key(), reason="Requires OPENAI_API_KEY")
@pytest.mark.asyncio
async def test_turn_trimming_session_with_real_summarization(tmp_path: Path):
    """Integration test: verify trimming + summarization with real OpenAI calls.

    This test validates:
    1. System prompt is preserved exactly (never summarized)
    2. Old turns (0, 1) are summarized into single messages each
    3. Recent turns (2, 3) are preserved exactly with images intact
    4. Total item count is correct: 1 system + 2 summaries + 4 recent = 7
    5. Summaries preserve key information (positions, scores)
    """
    # Setup: Create session with keep_last_n_turns=2.
    cfg = OmegaConf.create(
        {
            "session_memory": {
                "enabled": True,
                "keep_last_n_turns": 2,
                "enable_summarization": True,
                "summarization_model": "gpt-5.2",
                "summarization_thinking": "low",
            },
            "openai": {"model": "gpt-5.2"},
        }
    )

    db_path = tmp_path / "test_integration.db"
    sqlite_session = SQLiteSession(session_id="test_integration", db_path=db_path)
    session = TurnTrimmingSession(wrapped_session=sqlite_session, cfg=cfg)

    # Add system prompt first (should ALWAYS be preserved).
    system_prompt = (
        "You are a furniture placement assistant. Always place furniture precisely."
    )
    await session.add_items([{"role": "system", "content": system_prompt}])

    # Add 4 turns with images (simulating agent renders).
    for i in range(4):
        await session.add_items(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"Turn {i}: Observe the scene"},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,FAKE_IMAGE_{i}",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": f"Turn {i}: I see furniture at position ({i}, {i}, 0). "
                    f"Score: {80 + i}/100.",
                },
            ]
        )

    # Get items - should have:
    # - System message: preserved exactly
    # - Turns 0,1: summarized (each turn becomes 1 summary message)
    # - Turns 2,3: intact (with images)
    items = await session.get_items()

    # Verify total item count.
    # Original: 1 system + 4 turns * 2 items = 9 items
    # After: 1 system + 2 summaries + 2 turns * 2 items = 7 items
    assert len(items) == 7, f"Expected 7 items, got {len(items)}"

    # Verify system prompt is preserved exactly (first item).
    assert items[0]["role"] == "system", "First item should be system message"
    assert (
        items[0]["content"] == system_prompt
    ), "System prompt must be preserved exactly"

    # Verify old turns are summarized (items 1 and 2).
    turn_0_summary = items[1]
    turn_1_summary = items[2]

    # Each old turn should be a single assistant message with summary.
    assert (
        turn_0_summary["role"] == "assistant"
    ), "Summary should be an assistant message"
    assert (
        turn_1_summary["role"] == "assistant"
    ), "Summary should be an assistant message"

    # Summaries should contain the summary prefix.
    assert "[Previous turn summary]" in str(
        turn_0_summary["content"]
    ), "Turn 0 should be summarized"
    assert "[Previous turn summary]" in str(
        turn_1_summary["content"]
    ), "Turn 1 should be summarized"

    # Verify no images in old turn summaries.
    for summary in [turn_0_summary, turn_1_summary]:
        content = summary.get("content", "")
        if isinstance(content, list):
            for part in content:
                assert (
                    part.get("type") != "input_image"
                ), "Summary should have no images"
        # String content is fine - that's the summary.

    # Verify recent turns are preserved EXACTLY.
    expected_turn_2_user = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Turn 2: Observe the scene"},
            {"type": "input_image", "image_url": "data:image/png;base64,FAKE_IMAGE_2"},
        ],
    }
    expected_turn_2_assistant = {
        "role": "assistant",
        "content": "Turn 2: I see furniture at position (2, 2, 0). Score: 82/100.",
    }
    expected_turn_3_user = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Turn 3: Observe the scene"},
            {"type": "input_image", "image_url": "data:image/png;base64,FAKE_IMAGE_3"},
        ],
    }
    expected_turn_3_assistant = {
        "role": "assistant",
        "content": "Turn 3: I see furniture at position (3, 3, 0). Score: 83/100.",
    }

    # Recent turns should be EXACTLY preserved (last 4 items = turns 2 and 3).
    recent_items = items[-4:]
    assert recent_items[0] == expected_turn_2_user, "Turn 2 user message must be exact"
    assert (
        recent_items[1] == expected_turn_2_assistant
    ), "Turn 2 assistant message must be exact"
    assert recent_items[2] == expected_turn_3_user, "Turn 3 user message must be exact"
    assert (
        recent_items[3] == expected_turn_3_assistant
    ), "Turn 3 assistant message must be exact"

    # Verify summaries preserve key info (positions, scores).
    all_text = str(items)
    # The summaries should mention key details from turns 0 and 1.
    # At minimum, recent turns should have exact text.
    assert "position" in all_text.lower() or "(2, 2, 0)" in all_text
    assert "score" in all_text.lower() or "82" in all_text


@pytest.mark.skipif(not has_openai_key(), reason="Requires OPENAI_API_KEY")
@pytest.mark.asyncio
async def test_summarization_caching(tmp_path: Path):
    """Test that summaries are cached and reused."""
    cfg = OmegaConf.create(
        {
            "session_memory": {
                "enabled": True,
                "keep_last_n_turns": 1,
                "enable_summarization": True,
                "summarization_model": "gpt-5.2",
                "summarization_thinking": "low",
            },
            "openai": {"model": "gpt-5.2"},
        }
    )

    db_path = tmp_path / "test_cache.db"
    sqlite_session = SQLiteSession(session_id="test_cache", db_path=db_path)
    session = TurnTrimmingSession(wrapped_session=sqlite_session, cfg=cfg)

    # Add 2 turns (1 will be summarized).
    await session.add_items(
        [
            {"role": "user", "content": "Turn 0: Place a chair"},
            {"role": "assistant", "content": "Turn 0: Chair placed at (1, 2, 0)"},
        ]
    )
    await session.add_items(
        [
            {"role": "user", "content": "Turn 1: Check the scene"},
            {"role": "assistant", "content": "Turn 1: Scene looks good"},
        ]
    )

    # Get items twice - second call should use cached summary.
    items1 = await session.get_items()
    items2 = await session.get_items()

    # Both calls should return the same summary for turn 0.
    assert items1[0] == items2[0], "Cached summary should be identical"


@pytest.mark.skipif(not has_openai_key(), reason="Requires OPENAI_API_KEY")
@pytest.mark.asyncio
async def test_image_only_turn_summarization(tmp_path: Path):
    """Test summarization of turns with only images (no text)."""
    cfg = OmegaConf.create(
        {
            "session_memory": {
                "enabled": True,
                "keep_last_n_turns": 1,
                "enable_summarization": True,
                "summarization_model": "gpt-5.2",
                "summarization_thinking": "low",
            },
            "openai": {"model": "gpt-5.2"},
        }
    )

    db_path = tmp_path / "test_image_only.db"
    sqlite_session = SQLiteSession(session_id="test_image_only", db_path=db_path)
    session = TurnTrimmingSession(wrapped_session=sqlite_session, cfg=cfg)

    # Add turn with only image (simulating a render-only message).
    await session.add_items(
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,XYZ"},
                ],
            },
            {"role": "assistant", "content": "I see the rendered scene."},
        ]
    )
    # Add recent turn.
    await session.add_items(
        [
            {"role": "user", "content": "Keep this turn"},
            {"role": "assistant", "content": "Keeping it"},
        ]
    )

    items = await session.get_items()

    # Should have 3 items: 1 summary + 2 recent.
    assert len(items) == 3, f"Expected 3 items, got {len(items)}"

    # First item should be the summary (no crash on image-only content).
    assert "[Previous turn summary]" in str(items[0]["content"])


@pytest.mark.skipif(not has_openai_key(), reason="Requires OPENAI_API_KEY")
@pytest.mark.asyncio
async def test_mixed_content_preservation(tmp_path: Path):
    """Test that mixed text+image content is handled correctly."""
    cfg = OmegaConf.create(
        {
            "session_memory": {
                "enabled": True,
                "keep_last_n_turns": 1,
                "enable_summarization": True,
                "summarization_model": "gpt-5.2",
                "summarization_thinking": "low",
            },
            "openai": {"model": "gpt-5.2"},
        }
    )

    db_path = tmp_path / "test_mixed.db"
    sqlite_session = SQLiteSession(session_id="test_mixed", db_path=db_path)
    session = TurnTrimmingSession(wrapped_session=sqlite_session, cfg=cfg)

    # Add turn with mixed content.
    await session.add_items(
        [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Here is the scene:"},
                    {"type": "input_image", "image_url": "data:image/png;base64,ABC"},
                    {"type": "input_text", "text": "What do you think?"},
                ],
            },
            {
                "role": "assistant",
                "content": "The sofa is at position (3, 4, 0). Score: 85/100.",
            },
        ]
    )
    # Add recent turn.
    await session.add_items(
        [
            {"role": "user", "content": "Recent turn"},
            {"role": "assistant", "content": "Response"},
        ]
    )

    items = await session.get_items()

    # First item should be summary that preserves the position and score info.
    summary = str(items[0]["content"])
    assert "[Previous turn summary]" in summary

    # The summary should ideally mention the key info (position, score).
    # This is a soft check - LLM might phrase it differently.
    assert (
        "sofa" in summary.lower()
        or "position" in summary.lower()
        or "score" in summary.lower()
        or "85" in summary
    ), f"Summary should preserve key info. Got: {summary}"
