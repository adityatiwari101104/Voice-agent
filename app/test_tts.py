import os

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

api_key = os.getenv("SARVAM_API_KEY")

if not api_key:
    raise RuntimeError("SARVAM_API_KEY not found in .env")

client = SarvamAI(
    api_subscription_key=api_key,
)

response = client.text_to_speech.convert_stream(
    text="नमस्ते! मैं कक्षा दसवीं के बोर्ड के प्रश्न पत्र बना लेता हूँ। बताइए, मैं आपकी कैसे मदद करूँ?",
    language_code="hi-IN",
    speaker="ishita",
    model="bulbul:v3",
    pace=1,
    speech_sample_rate=22050,
    output_audio_codec="mp3",
)

with open("speech.mp3", "wb") as f:
    for chunk in response:
        f.write(chunk)

print("Speech generated successfully: speech.mp3")