import base64
import os
import tempfile
import unittest

from pathlib import Path

from openai import OpenAI


class TestResponsesImageGenerationSmoke(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("OPENAI_API_KEY"),
        "Requires OPENAI_API_KEY",
    )
    def test_generate_single_image(self):
        client = OpenAI(
            base_url="https://api.openai.com/v1",
        )

        response = client.responses.create(
            model="gpt-5",
            input="Generate an image of gray tabby cat hugging an otter with an orange scarf",
            tools=[{"type": "image_generation"}],
        )

        image_data = [
            output.result
            for output in response.output
            if output.type == "image_generation_call"
        ]

        self.assertTrue(image_data, "No image_generation_call output found")

        image_bytes = base64.b64decode(image_data[0])
        self.assertGreater(len(image_bytes), 0, "Decoded image is empty")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "otter.png"
            out.write_bytes(image_bytes)

            self.assertTrue(out.exists(), "Image file was not created")
            self.assertGreater(out.stat().st_size, 0, "Image file is empty")


if __name__ == "__main__":
    unittest.main(verbosity=2)

# python -m unittest -v ./tests/unit/gpt_image_generation.py
