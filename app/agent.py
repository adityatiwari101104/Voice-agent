import os
from dotenv import load_dotenv
from openai import OpenAI

from .schemas import PaperRequest

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY not found in .env")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

MODEL_ID = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """
You are the requirement-understanding component of an AI question paper generator.

Convert the teacher's natural-language request into a structured
question-paper specification.

Extract:
- class level
- subject
- chapters or topics
- total marks
- duration if mentioned
- difficulty
- question sections

Rules:
1. Do not invent information that the teacher did not provide.
2. If duration is not mentioned, set duration_minutes to null.
3. Preserve the teacher's requested number of questions.
4. Preserve the requested marks per question.
5. Return data matching the provided PaperRequest schema.
"""

def create_paper_request(user_input: str) -> PaperRequest:
    response = client.responses.parse(
        model=MODEL_ID,
        instructions=SYSTEM_PROMPT,
        input=user_input,
        text_format=PaperRequest
    )

    return response.output_parsed