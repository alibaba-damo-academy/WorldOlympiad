"""
Docstring for model.test

测试Openrouter的模型调用
"""

import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY_REF = os.getenv("api_key")

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY_REF}",
    "Content-Type": "application/json"
}
import json
import base64
from pathlib import Path
def encode_video_to_base64(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY_REF}",
    "Content-Type": "application/json"
}
# Read and encode the video
video_path = "data/6.mp4"
# video_path = "data/videos/compression_self_forcing_1.mp4"
base64_video = encode_video_to_base64(video_path)
data_url = f"data:video/mp4;base64,{base64_video}"

text = """
Is the floating phenomenon of this lemon natural?
"""
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": text
            },
            {
                "type": "video_url",
                "video_url": {
                    "url": data_url
                }
            }
        ]
    }
]

payload = {
    "model": "qwen/qwen3.5-9b",
    "messages": messages
}
response = requests.post(url, headers=headers, json=payload)
print(response.text)

