"""Unit tests for TurnTrimmingSession.

Tests the core logic of turn parsing, image detection/stripping, and session
behavior without making actual OpenAI API calls.
"""

import unittest

import pytest

from agents import SQLiteSession
from omegaconf import OmegaConf
from unittest.mock import AsyncMock, MagicMock, patch

from scenecode.agent_utils.turn_trimming_session import (
    Turn,
    TurnTrimmingSession,
    _contains_base64_image,
    _extract_text_from_turn,
    _is_image_content,
    _is_system_message,
    _is_user_message,
    _parse_turns,
    _strip_base64_from_string,
    _strip_images_from_item,
)


class TestTurnParsing(unittest.TestCase):
    """Tests for turn parsing logic."""

    def test_parse_turns_empty(self):
        """Empty items list returns empty turns."""
        turns, first_turn_start = _parse_turns([])
        self.assertEqual(turns, [])
        self.assertEqual(first_turn_start, 0)

    def test_parse_turns_single_turn(self):
        """Single user message + response forms one turn."""
        items = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        turns, first_turn_start = _parse_turns(items)

        self.assertEqual(len(turns), 1)
        self.assertEqual(first_turn_start, 0)
        self.assertEqual(turns[0].start_index, 0)
        self.assertEqual(turns[0].end_index, 2)
        self.assertEqual(len(turns[0].items), 2)

    def test_parse_turns_multiple_turns(self):
        """Multiple user messages create separate turns."""
        items = [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Turn 3"},
        ]
        turns, first_turn_start = _parse_turns(items)

        self.assertEqual(len(turns), 3)
        self.assertEqual(first_turn_start, 0)
        self.assertEqual(turns[0].items, items[0:2])
        self.assertEqual(turns[1].items, items[2:4])
        self.assertEqual(turns[2].items, items[4:5])

    def test_parse_turns_with_preamble(self):
        """Items before first user message are preamble."""
        items = [
            {"role": "system", "content": "System prompt"},
            {"role": "developer", "content": "Dev message"},
            {"role": "user", "content": "First turn"},
            {"role": "assistant", "content": "Response"},
        ]
        turns, first_turn_start = _parse_turns(items)

        self.assertEqual(len(turns), 1)
        self.assertEqual(first_turn_start, 2)  # User message at index 2
        self.assertEqual(turns[0].start_index, 2)

    def test_is_user_message(self):
        """Correctly identifies user messages."""
        self.assertTrue(_is_user_message({"role": "user", "content": "test"}))
        self.assertFalse(_is_user_message({"role": "assistant", "content": "test"}))
        self.assertFalse(_is_user_message({"role": "system", "content": "test"}))

    def test_is_system_message(self):
        """Correctly identifies system messages."""
        self.assertTrue(_is_system_message({"role": "system", "content": "test"}))
        self.assertFalse(_is_system_message({"role": "user", "content": "test"}))
        self.assertFalse(_is_system_message({"role": "assistant", "content": "test"}))


class TestImageDetection(unittest.TestCase):
    """Tests for image content detection."""

    def test_is_image_content_input_image_type(self):
        """Detects input_image type."""
        self.assertTrue(_is_image_content({"type": "input_image", "image_url": "..."}))

    def test_is_image_content_image_url_type(self):
        """Detects image_url type."""
        self.assertTrue(_is_image_content({"type": "image_url", "url": "..."}))

    def test_is_image_content_image_type(self):
        """Detects image type."""
        self.assertTrue(_is_image_content({"type": "image"}))

    def test_is_image_content_data_url(self):
        """Detects data URL in image_url field."""
        self.assertTrue(
            _is_image_content({"image_url": "data:image/png;base64,ABC123"})
        )

    def test_is_image_content_http_url(self):
        """Detects HTTP URL in image_url field."""
        self.assertTrue(
            _is_image_content({"image_url": "https://example.com/image.png"})
        )

    def test_is_image_content_nested_url(self):
        """Detects nested image_url object."""
        self.assertTrue(
            _is_image_content(
                {"image_url": {"url": "data:image/png;base64,ABC123", "detail": "auto"}}
            )
        )

    def test_is_image_content_text_type(self):
        """Text content is not an image."""
        self.assertFalse(_is_image_content({"type": "input_text", "text": "Hello"}))
        self.assertFalse(_is_image_content({"type": "text", "text": "Hello"}))


class TestImageStripping(unittest.TestCase):
    """Tests for image stripping logic."""

    def test_strip_images_pure_image_message(self):
        """Message with only images gets all replaced."""
        item = {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,ABC"},
                {"type": "input_image", "image_url": "data:image/png;base64,DEF"},
            ],
        }
        result = _strip_images_from_item(item)

        self.assertEqual(result["role"], "user")
        self.assertEqual(len(result["content"]), 2)
        self.assertEqual(result["content"][0]["type"], "input_text")
        self.assertEqual(result["content"][0]["text"], "[image removed]")
        self.assertEqual(result["content"][1]["text"], "[image removed]")

    def test_strip_images_mixed_message(self):
        """Mixed content preserves text, replaces images."""
        item = {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Look at this:"},
                {"type": "input_image", "image_url": "data:image/png;base64,ABC"},
                {"type": "input_text", "text": "What do you see?"},
            ],
        }
        result = _strip_images_from_item(item)

        self.assertEqual(len(result["content"]), 3)
        self.assertEqual(result["content"][0]["text"], "Look at this:")
        self.assertEqual(result["content"][1]["text"], "[image removed]")
        self.assertEqual(result["content"][2]["text"], "What do you see?")

    def test_strip_images_string_content(self):
        """String content is returned unchanged."""
        item = {"role": "assistant", "content": "Plain text response"}
        result = _strip_images_from_item(item)
        self.assertEqual(result, item)

    def test_strip_images_non_dict_item(self):
        """Non-dict items are returned unchanged."""
        item = "just a string"
        result = _strip_images_from_item(item)
        self.assertEqual(result, item)

    def test_strip_images_from_tool_output_with_base64(self):
        """Tool output with base64 image (like observe_scene) has it stripped."""
        # Simulate observe_scene tool output containing ToolOutputImage data.
        # Real format: ToolOutputImage(image_url=f"data:image/png;base64,{img_base64}")
        base64_img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ" + "A" * 500
        item = {
            "type": "function_call_output",
            "call_id": "call_observe_123",
            "output": f'[{{"type": "image_url", "image_url": "data:image/png;base64,{base64_img}"}}]',
        }
        result = _strip_images_from_item(item)

        self.assertEqual(result["type"], "function_call_output")
        self.assertEqual(result["call_id"], "call_observe_123")
        self.assertNotIn(base64_img, result["output"])
        self.assertIn("[base64 image removed]", result["output"])

    def test_strip_images_tool_output_without_base64(self):
        """Tool output without base64 is returned unchanged."""
        item = {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "Simple text result",
        }
        result = _strip_images_from_item(item)
        self.assertEqual(result["output"], "Simple text result")

    def test_strip_images_from_list_tool_output(self):
        """Tool output with list of ToolOutputImage items has images stripped.

        This is the actual format from observe_scene: a list of dicts with
        type='input_image' and image_url containing base64 data.
        """
        base64_img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ" + "A" * 500
        item = {
            "type": "function_call_output",
            "call_id": "call_observe_123",
            "output": [
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{base64_img}",
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{base64_img}",
                },
                {"type": "input_text", "text": "Scene description"},
            ],
        }
        result = _strip_images_from_item(item)

        self.assertEqual(result["type"], "function_call_output")
        self.assertEqual(result["call_id"], "call_observe_123")
        # Images should be replaced with placeholders.
        self.assertEqual(len(result["output"]), 3)
        self.assertEqual(
            result["output"][0], {"type": "input_text", "text": "[image removed]"}
        )
        self.assertEqual(
            result["output"][1], {"type": "input_text", "text": "[image removed]"}
        )
        # Non-image items preserved.
        self.assertEqual(
            result["output"][2], {"type": "input_text", "text": "Scene description"}
        )


class TestBase64Detection(unittest.TestCase):
    """Tests for base64 image detection."""

    def test_contains_base64_data_url(self):
        """Detects data:image URL pattern (ToolOutputImage format)."""
        text = '{"image_url": "data:image/png;base64,iVBORw0KGgo..."}'
        self.assertTrue(_contains_base64_image(text))

    def test_no_base64_in_normal_text(self):
        """Normal text without base64 returns False."""
        text = "Just some normal text with no images"
        self.assertFalse(_contains_base64_image(text))

    def test_partial_pattern_not_matched(self):
        """Partial patterns (only data:image or only base64) not matched."""
        # Only data:image without base64
        self.assertFalse(_contains_base64_image("data:image/png without base64"))
        # Only base64 without data:image
        self.assertFalse(_contains_base64_image("some base64, encoded text"))


class TestBase64Stripping(unittest.TestCase):
    """Tests for base64 stripping from strings."""

    def test_strip_data_url(self):
        """Data URLs are replaced with placeholder."""
        base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        text = f'{{"image_url": "data:image/png;base64,{base64_data}"}}'
        result = _strip_base64_from_string(text)
        self.assertIn("[base64 image removed]", result)
        self.assertNotIn(base64_data, result)

    def test_preserves_normal_text(self):
        """Normal text is preserved."""
        text = "This is normal text with {json: 'data'}"
        result = _strip_base64_from_string(text)
        self.assertEqual(result, text)

    def test_strips_multiple_images(self):
        """Multiple images in same output are all stripped."""
        base64_a = "iVBORw0KGgoAAAAA"
        base64_b = "iVBORw0KGgoAAAAB"
        text = (
            f'[{{"image_url": "data:image/png;base64,{base64_a}"}}, '
            f'{{"image_url": "data:image/jpeg;base64,{base64_b}"}}]'
        )
        result = _strip_base64_from_string(text)
        self.assertEqual(result.count("[base64 image removed]"), 2)
        self.assertNotIn(base64_a, result)
        self.assertNotIn(base64_b, result)


class TestTextExtraction(unittest.TestCase):
    """Tests for text extraction from turns."""

    def test_extract_text_from_turn_simple(self):
        """Extracts text from simple turn."""
        turn = Turn(
            start_index=0,
            end_index=2,
            items=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )
        text = _extract_text_from_turn(turn)

        self.assertIn("user: Hello", text)
        self.assertIn("assistant: Hi there", text)

    def test_extract_text_from_turn_with_images(self):
        """Images are replaced with placeholder in extraction."""
        turn = Turn(
            start_index=0,
            end_index=1,
            items=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "See this:"},
                        {"type": "input_image", "image_url": "data:image/png;base64,X"},
                    ],
                },
            ],
        )
        text = _extract_text_from_turn(turn)

        self.assertIn("See this:", text)
        self.assertIn("[image]", text)

    def test_extract_text_from_turn_with_tool_output(self):
        """Tool outputs are included in extraction."""
        turn = Turn(
            start_index=0,
            end_index=2,
            items=[
                {"role": "user", "content": "Call a tool"},
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "Tool result",
                },
            ],
        )
        text = _extract_text_from_turn(turn)

        self.assertIn("tool_output(call_123): Tool result", text)

    def test_extract_text_strips_base64_from_tool_output(self):
        """Tool outputs with base64 string have it stripped before extraction."""
        base64_img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ" + "A" * 500
        turn = Turn(
            start_index=0,
            end_index=2,
            items=[
                {"role": "user", "content": "Observe the scene"},
                {
                    "type": "function_call_output",
                    "call_id": "call_observe",
                    "output": f'{{"image_url": "data:image/png;base64,{base64_img}"}}',
                },
            ],
        )
        text = _extract_text_from_turn(turn)

        # Should not contain the base64 data (would cause 110K+ tokens).
        self.assertNotIn(base64_img, text)
        # Should contain placeholder instead.
        self.assertIn("[base64 image removed]", text)

    def test_extract_text_handles_list_tool_output_with_images(self):
        """Tool outputs with list of images show count placeholder.

        This is the actual format from observe_scene: a list of ToolOutputImage.
        """
        base64_img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ" + "A" * 500
        turn = Turn(
            start_index=0,
            end_index=2,
            items=[
                {"role": "user", "content": "Observe the scene"},
                {
                    "type": "function_call_output",
                    "call_id": "call_observe",
                    "output": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_img}",
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_img}",
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_img}",
                        },
                    ],
                },
            ],
        )
        text = _extract_text_from_turn(turn)

        # Should not contain the base64 data.
        self.assertNotIn(base64_img, text)
        # Should show image count placeholder.
        self.assertIn("[3 images]", text)


class TestTurnTrimmingSessionUnit:
    """Unit tests for TurnTrimmingSession (mocked OpenAI)."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temp directory for test databases."""
        return tmp_path

    @pytest.fixture
    def cfg(self):
        """Create test configuration."""
        return OmegaConf.create(
            {
                "session_memory": {
                    "enabled": True,
                    "keep_last_n_turns": 2,
                    "enable_summarization": False,  # Disabled for unit tests
                    "summarization_model": "gpt-5.2",
                    "summarization_thinking": "low",
                },
                "openai": {"model": "gpt-5.2"},
            }
        )

    @pytest.mark.asyncio
    async def test_get_items_keeps_recent_turns(self, temp_dir, cfg):
        """Last N turns are preserved intact."""
        db_path = temp_dir / "test_session.db"
        sqlite_session = SQLiteSession(session_id="test", db_path=db_path)
        session = TurnTrimmingSession(sqlite_session, cfg)

        # Add 3 turns (keep_last_n_turns=2, so turn 0 should be trimmed).
        await session.add_items(
            [
                {"role": "user", "content": "Turn 0"},
                {"role": "assistant", "content": "Response 0"},
            ]
        )
        await session.add_items(
            [
                {"role": "user", "content": "Turn 1"},
                {"role": "assistant", "content": "Response 1"},
            ]
        )
        await session.add_items(
            [
                {"role": "user", "content": "Turn 2"},
                {"role": "assistant", "content": "Response 2"},
            ]
        )

        items = await session.get_items()

        # Last 2 turns should be preserved exactly.
        recent_items = items[-4:]  # Last 4 items = 2 turns
        assert recent_items[0]["content"] == "Turn 1"
        assert recent_items[1]["content"] == "Response 1"
        assert recent_items[2]["content"] == "Turn 2"
        assert recent_items[3]["content"] == "Response 2"

    @pytest.mark.asyncio
    async def test_get_items_strips_old_images(self, temp_dir, cfg):
        """Old turn images are stripped when summarization disabled."""
        db_path = temp_dir / "test_session.db"
        sqlite_session = SQLiteSession(session_id="test", db_path=db_path)
        session = TurnTrimmingSession(sqlite_session, cfg)

        # Add 3 turns with images.
        for i in range(3):
            await session.add_items(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": f"Turn {i}"},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,IMG{i}",
                            },
                        ],
                    },
                    {"role": "assistant", "content": f"Response {i}"},
                ]
            )

        items = await session.get_items()

        # Turn 0 should have images stripped.
        old_turn_items = items[:2]  # First 2 items = turn 0
        old_user_msg = old_turn_items[0]
        # Check that images are replaced with placeholder.
        found_placeholder = False
        for part in old_user_msg["content"]:
            if part.get("type") == "input_text":
                if "[image removed]" in part.get("text", ""):
                    found_placeholder = True
                    break

        if not found_placeholder:
            # If no placeholder found, check there are no images.
            has_image = any(
                part.get("type") == "input_image" for part in old_user_msg["content"]
            )
            assert not has_image, "Old turn should have images removed"

    @pytest.mark.asyncio
    async def test_system_message_preserved(self, temp_dir, cfg):
        """System messages are never trimmed."""
        db_path = temp_dir / "test_session.db"
        sqlite_session = SQLiteSession(session_id="test", db_path=db_path)
        session = TurnTrimmingSession(sqlite_session, cfg)

        system_prompt = "You are a helpful assistant."
        await session.add_items([{"role": "system", "content": system_prompt}])

        # Add 5 turns to exceed keep_last_n_turns.
        for i in range(5):
            await session.add_items(
                [
                    {"role": "user", "content": f"Turn {i}"},
                    {"role": "assistant", "content": f"Response {i}"},
                ]
            )

        items = await session.get_items()

        # System message should be first and preserved exactly.
        assert items[0]["role"] == "system"
        assert items[0]["content"] == system_prompt

    @pytest.mark.asyncio
    async def test_fewer_turns_than_n_all_kept(self, temp_dir, cfg):
        """When turns < N, all are kept intact."""
        db_path = temp_dir / "test_session.db"
        sqlite_session = SQLiteSession(session_id="test", db_path=db_path)
        session = TurnTrimmingSession(sqlite_session, cfg)

        # Add only 1 turn (keep_last_n_turns=2).
        await session.add_items(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Only turn"},
                        {"type": "input_image", "image_url": "data:image/png;base64,X"},
                    ],
                },
                {"role": "assistant", "content": "Response"},
            ]
        )

        items = await session.get_items()

        # Should have 2 items, with image preserved.
        assert len(items) == 2
        user_msg = items[0]
        has_image = any(
            part.get("type") == "input_image" for part in user_msg["content"]
        )
        assert has_image, "Single turn should keep images"

    @pytest.mark.asyncio
    async def test_empty_session(self, temp_dir, cfg):
        """Empty session returns empty list."""
        db_path = temp_dir / "test_session.db"
        sqlite_session = SQLiteSession(session_id="test", db_path=db_path)
        session = TurnTrimmingSession(sqlite_session, cfg)

        items = await session.get_items()
        assert items == []

    @pytest.mark.asyncio
    async def test_clear_session(self, temp_dir, cfg):
        """Clear removes all items and summaries."""
        db_path = temp_dir / "test_session.db"
        sqlite_session = SQLiteSession(session_id="test", db_path=db_path)
        session = TurnTrimmingSession(sqlite_session, cfg)

        await session.add_items(
            [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ]
        )

        await session.clear_session()
        items = await session.get_items()

        assert items == []

    @pytest.mark.asyncio
    async def test_pop_item_delegates(self, temp_dir, cfg):
        """pop_item delegates to wrapped session."""
        db_path = temp_dir / "test_session.db"
        sqlite_session = SQLiteSession(session_id="test", db_path=db_path)
        session = TurnTrimmingSession(sqlite_session, cfg)

        await session.add_items(
            [
                {"role": "user", "content": "First"},
                {"role": "user", "content": "Second"},
            ]
        )

        popped = await session.pop_item()
        assert popped["content"] == "Second"

        remaining = await session.get_items()
        assert len(remaining) == 1

    def test_summary_client_uses_configured_api_base(self, temp_dir, cfg):
        """Summarization client is created with the configured api_base."""
        cfg.openai.api_base = "https://api.example.com/v1"
        db_path = temp_dir / "test_session.db"
        sqlite_session = SQLiteSession(session_id="test", db_path=db_path)

        with patch(
            "scenecode.agent_utils.turn_trimming_session.create_async_openai_client"
        ) as mock_create_client:
            mock_client = MagicMock()
            mock_create_client.return_value = mock_client

            session = TurnTrimmingSession(sqlite_session, cfg)
            client = session._get_openai_client()

        assert client == mock_client
        mock_create_client.assert_called_once_with(api_base=cfg.openai.api_base)

    @pytest.mark.asyncio
    async def test_summarize_turn_omits_reasoning_argument(self, temp_dir, cfg):
        """Turn summarization should not send reasoning to the Responses API."""
        db_path = temp_dir / "test_session.db"
        sqlite_session = SQLiteSession(session_id="test", db_path=db_path)
        session = TurnTrimmingSession(sqlite_session, cfg)

        mock_response = MagicMock()
        mock_response.output_text = "Summarized turn"
        mock_client = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)
        session._openai_client = mock_client

        turn = Turn(
            start_index=0,
            end_index=2,
            items=[
                {"role": "user", "content": "Please summarize this turn."},
                {"role": "assistant", "content": "Here is the turn content."},
            ],
        )

        with patch(
            "scenecode.agent_utils.turn_trimming_session.prompt_registry.get_prompt",
            return_value="Summarize the turn.",
        ), patch(
            "scenecode.agent_utils.turn_trimming_session.log_openai_usage"
        ) as mock_log_usage:
            summary = await session._summarize_turn(turn=turn, turn_number=0)

        assert summary == "Summarized turn"
        call_kwargs = mock_client.responses.create.await_args.kwargs
        assert call_kwargs["model"] == cfg.session_memory.summarization_model
        assert call_kwargs["instructions"] == "Summarize the turn."
        assert "reasoning" not in call_kwargs
        mock_log_usage.assert_called_once_with(
            response=mock_response,
            component="turn_summarization",
            api="responses",
            model=cfg.session_memory.summarization_model,
        )

    @pytest.mark.asyncio
    async def test_close_is_noop_without_client(self, temp_dir, cfg):
        """close() should be a no-op when summarization client was never created."""
        db_path = temp_dir / "test_session.db"
        sqlite_session = SQLiteSession(session_id="test", db_path=db_path)
        session = TurnTrimmingSession(sqlite_session, cfg)

        await session.close()

        assert session._openai_client is None

    @pytest.mark.asyncio
    async def test_close_closes_client_once(self, temp_dir, cfg):
        """close() should release the cached summarization client and be idempotent."""
        db_path = temp_dir / "test_session.db"
        sqlite_session = SQLiteSession(session_id="test", db_path=db_path)
        session = TurnTrimmingSession(sqlite_session, cfg)
        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()
        session._openai_client = mock_client

        await session.close()
        await session.close()

        mock_client.aclose.assert_awaited_once()
        assert session._openai_client is None


if __name__ == "__main__":
    unittest.main()
