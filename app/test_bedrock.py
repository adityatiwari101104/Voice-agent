import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

if not api_key:
    raise RuntimeError("Bedrock API key not found")

url = "https://bedrock-runtime.ap-south-1.amazonaws.com/model/anthropic.claude-sonnet-4-6/converse"

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "text": "Say hello in one sentence."
                }
            ]
        }
    ]
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

response = requests.post(
    url,
    headers=headers,
    json=payload
)

print("STATUS:", response.status_code)
print(response.text)