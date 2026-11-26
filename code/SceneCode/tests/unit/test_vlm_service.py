import shutil
import tempfile
import unittest

from pathlib import Path
from unittest.mock import Mock, patch

from scenecode.agent_utils.vlm_service import VLMService
from scenecode.utils.openai import DEFAULT_OPENAI_API_BASE


class TestVLMService(unittest.TestCase):
    """Test VLMService class contracts."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_openai_client = Mock()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("scenecode.utils.openai.OpenAI")
    def test_vlm_initialization(self, mock_openai_class):
        """Test VLMService initializes OpenAI client properly."""
        mock_openai_class.return_value = self.mock_openai_client

        vlm_service = VLMService()

        # Verify VLMService was initialized.
        self.assertIsNotNone(vlm_service)
        self.assertEqual(vlm_service.client, self.mock_openai_client)

        # Verify OpenAI client was created.
        mock_openai_class.assert_called_once_with(
            base_url=DEFAULT_OPENAI_API_BASE
        )

    @patch("scenecode.utils.openai.OpenAI")
    def test_create_completion_basic(self, mock_openai_class):
        """Test create_completion with basic parameters for standard models."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        # Mock the chat completions API for standard models.
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Test response content"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        vlm_service = VLMService()

        messages = [{"role": "user", "content": "Test message"}]
        model = "gpt-4o"
        # Verbosity is ignored for non-reasoning models (Chat Completions API).
        result = vlm_service.create_completion(
            model=model,
            messages=messages,
            usage_label="test.chat.basic",
            reasoning_effort="medium",
            verbosity="low",
        )

        # Verify result.
        self.assertEqual(result, "Test response content")

        # Verify OpenAI chat API was called for standard model.
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["model"], model)

    @patch("scenecode.utils.openai.OpenAI")
    def test_create_completion_omits_reasoning_effort_and_verbosity_from_request(
        self, mock_openai_class
    ):
        """Reasoning-model requests should omit reasoning and verbosity fields."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        # Mock the responses API.
        mock_response = Mock()
        mock_response.output_text = "Reasoning response"
        mock_response.usage = None
        mock_openai_client.responses.create.return_value = mock_response

        vlm_service = VLMService()

        messages = [{"role": "user", "content": "Complex reasoning task"}]
        model = "gpt-5"
        reasoning_effort = "high"
        verbosity = "low"
        result = vlm_service.create_completion(
            model=model,
            messages=messages,
            usage_label="test.responses.basic",
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )

        # Verify result.
        self.assertEqual(result, "Reasoning response")

        # Verify reasoning and verbosity were not forwarded.
        call_args = mock_openai_client.responses.create.call_args
        self.assertEqual(call_args[1]["model"], model)
        self.assertNotIn("reasoning", call_args[1])
        self.assertNotIn("text", call_args[1])

    @patch("scenecode.utils.openai.OpenAI")
    def test_create_completion_with_json_format(self, mock_openai_class):
        """Test create_completion with JSON response format."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        # Mock the chat completions API for standard models.
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '{"result": "json_response"}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        vlm_service = VLMService()

        model = "gpt-4o"
        messages = [{"role": "user", "content": "Return JSON"}]
        result = vlm_service.create_completion(
            model=model,
            messages=messages,
            usage_label="test.chat.json",
            reasoning_effort="medium",
            verbosity="low",
            response_format={"type": "json_object"},
        )

        # Verify result.
        self.assertEqual(result, '{"result": "json_response"}')

        # Verify chat API was called with JSON format for standard model.
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["model"], model)
        self.assertEqual(call_args[1]["response_format"], {"type": "json_object"})

    @patch("scenecode.utils.openai.OpenAI")
    def test_error_handling_for_api_failures(self, mock_openai_class):
        """Test handling of OpenAI API errors."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        # Mock chat API to raise an error for standard models.
        mock_openai_client.chat.completions.create.side_effect = Exception(
            "API rate limit exceeded"
        )

        vlm_service = VLMService()

        # Test that API errors are propagated.
        messages = [{"role": "user", "content": "Test"}]
        with self.assertRaises(Exception) as context:
            vlm_service.create_completion(
                model="gpt-4o",
                messages=messages,
                usage_label="test.chat.error",
                reasoning_effort="medium",
                verbosity="low",
            )

        self.assertIn("API rate limit exceeded", str(context.exception))

    @patch("scenecode.utils.openai.OpenAI")
    def test_message_conversion_to_responses_format(self, mock_openai_class):
        """Test that messages work correctly for reasoning models with images."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        # Mock the responses API for reasoning models.
        mock_response = Mock()
        mock_response.output_text = "Converted response"
        mock_response.usage = None
        mock_openai_client.responses.create.return_value = mock_response

        vlm_service = VLMService()

        # Test with image content in messages for reasoning model.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,..."},
                    },
                ],
            }
        ]

        # Test with reasoning model (gpt-5) for image conversion.
        result = vlm_service.create_completion(
            model="gpt-5",
            messages=messages,
            usage_label="test.responses.image",
            reasoning_effort="medium",
            verbosity="low",
        )

        # Verify conversion worked and responses API was called.
        self.assertEqual(result, "Converted response")
        mock_openai_client.responses.create.assert_called_once()

        # Verify input was converted for responses API.
        call_args = mock_openai_client.responses.create.call_args
        self.assertIn("input", call_args[1])

    @patch("scenecode.utils.openai.OpenAI")
    def test_vision_detail_parameter_chat_completions(self, mock_openai_class):
        """Test that vision_detail parameter is added to image_url objects for Chat
        API."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        # Mock the chat completions API for standard models.
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Vision response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None
        mock_openai_client.chat.completions.create.return_value = mock_response

        vlm_service = VLMService()

        # Test with image content in messages for standard model.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,..."},
                    },
                ],
            }
        ]

        result = vlm_service.create_completion(
            model="gpt-4o",
            messages=messages,
            usage_label="test.chat.image",
            reasoning_effort="medium",
            verbosity="low",
            vision_detail="high",
        )

        # Verify result.
        self.assertEqual(result, "Vision response")

        # Verify chat API was called.
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args

        # Verify that detail parameter was added to image_url.
        messages_sent = call_args[1]["messages"]
        image_content = None
        for item in messages_sent[0]["content"]:
            if item["type"] == "image_url":
                image_content = item
                break

        self.assertIsNotNone(image_content)
        self.assertEqual(image_content["image_url"]["detail"], "high")

    @patch("scenecode.utils.openai.OpenAI")
    def test_vision_detail_parameter_responses_api(self, mock_openai_class):
        """Test vision_detail parameter handling for Responses API (reasoning models)."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        # Mock the responses API for reasoning models.
        mock_response = Mock()
        mock_response.output_text = "Reasoning vision response"
        mock_response.usage = None
        mock_openai_client.responses.create.return_value = mock_response

        vlm_service = VLMService()

        # Test with image content in messages for reasoning model.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,..."},
                    },
                ],
            }
        ]

        result = vlm_service.create_completion(
            model="gpt-5",
            messages=messages,
            usage_label="test.responses.vision",
            reasoning_effort="medium",
            verbosity="low",
            vision_detail="high",
        )

        # Verify result.
        self.assertEqual(result, "Reasoning vision response")

        # Verify responses API was called.
        mock_openai_client.responses.create.assert_called_once()

        # Verify the detail parameter was included in the Responses API format.
        call_args = mock_openai_client.responses.create.call_args
        input_messages = call_args[1]["input"]
        self.assertIsNotNone(input_messages)

        # Find the image content in the converted format.
        image_content = None
        for msg in input_messages:
            if "content" in msg and isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "input_image":
                        image_content = item
                        break

        self.assertIsNotNone(image_content)
        self.assertEqual(image_content["detail"], "high")


    @patch("scenecode.utils.openai.OpenAI")
    def test_create_completion_logs_usage_for_responses_api(self, mock_openai_class):
        """Responses API calls should emit normalized usage logs."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        mock_response = Mock()
        mock_response.output_text = "Reasoning response"
        mock_response.usage = Mock(
            input_tokens=123,
            input_tokens_details=Mock(cached_tokens=11),
            output_tokens=45,
            output_tokens_details=Mock(reasoning_tokens=6),
            total_tokens=168,
        )
        mock_openai_client.responses.create.return_value = mock_response

        vlm_service = VLMService()

        with self.assertLogs("scenecode.utils.openai", level="INFO") as logs:
            result = vlm_service.create_completion(
                model="gpt-5",
                messages=[{"role": "user", "content": "Test"}],
                usage_label="test.responses.logging",
                reasoning_effort="medium",
                verbosity="low",
            )

        self.assertEqual(result, "Reasoning response")
        self.assertTrue(
            any(
                "[OPENAI_USAGE] component=test.responses.logging, api=responses, "
                "model=gpt-5, input=123, output=45, reasoning=6, cached=11, total=168"
                in entry
                for entry in logs.output
            )
        )

    @patch("scenecode.utils.openai.OpenAI")
    def test_create_completion_logs_usage_for_chat_completions(self, mock_openai_class):
        """Chat Completions API calls should emit normalized usage logs."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Chat response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock(
            prompt_tokens=90,
            prompt_tokens_details=Mock(cached_tokens=5),
            completion_tokens=30,
            completion_tokens_details=Mock(reasoning_tokens=2),
            total_tokens=120,
        )
        mock_openai_client.chat.completions.create.return_value = mock_response

        vlm_service = VLMService()

        with self.assertLogs("scenecode.utils.openai", level="INFO") as logs:
            result = vlm_service.create_completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test"}],
                usage_label="test.chat.logging",
                reasoning_effort="medium",
                verbosity="low",
            )

        self.assertEqual(result, "Chat response")
        self.assertTrue(
            any(
                "[OPENAI_USAGE] component=test.chat.logging, "
                "api=chat_completions, model=gpt-4o, input=90, output=30, "
                "reasoning=2, cached=5, total=120" in entry
                for entry in logs.output
            )
        )

    @patch("scenecode.utils.openai.OpenAI")
    def test_create_completion_logs_usage_before_empty_responses_error(
        self, mock_openai_class
    ):
        """Usage should still be logged before empty response validation fails."""
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client

        mock_output = Mock()
        mock_output.type = "message"
        mock_output.content = []
        mock_response = Mock()
        mock_response.output_text = ""
        mock_response.output = [mock_output]
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_response.status = "completed"
        mock_response.usage = Mock(
            input_tokens=40,
            input_tokens_details=None,
            output_tokens=10,
            output_tokens_details=None,
            total_tokens=50,
        )
        mock_openai_client.responses.create.return_value = mock_response

        vlm_service = VLMService()

        with self.assertLogs("scenecode.utils.openai", level="INFO") as logs:
            with self.assertRaises(RuntimeError):
                vlm_service.create_completion(
                    model="gpt-5",
                    messages=[{"role": "user", "content": "Test"}],
                    usage_label="test.responses.empty",
                    reasoning_effort="medium",
                    verbosity="low",
                )

        self.assertTrue(
            any(
                "[OPENAI_USAGE] component=test.responses.empty, api=responses, "
                "model=gpt-5, input=40, output=10, reasoning=0, cached=0, total=50"
                in entry
                for entry in logs.output
            )
        )


if __name__ == "__main__":
    unittest.main()
