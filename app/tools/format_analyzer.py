import os
import json

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY not found in .env")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

MODEL_ID = "openai/gpt-oss-20b"


class QuestionPattern(BaseModel):
    question_range: str
    question_types: list[str]
    marks_distribution: list[int]
    notes: list[str]


class BoardSection(BaseModel):
    name: str
    subject_area: str | None
    question_range: str
    patterns: list[QuestionPattern]


class PaperFormat(BaseModel):
    exam_name: str | None
    board_name: str | None
    class_level: str | None
    subject: str | None
    duration_minutes: int | None
    total_marks: int | None

    header_structure: list[str]
    general_instructions: list[str]

    total_questions: int | None
    sections: list[BoardSection]

    question_numbering_style: str
    marks_display_style: str
    options_style: str
    internal_choice_style: str | None

    layout_notes: list[str]


FORMAT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "exam_name": {"type": ["string", "null"]},
        "board_name": {"type": ["string", "null"]},
        "class_level": {"type": ["string", "null"]},
        "subject": {"type": ["string", "null"]},
        "duration_minutes": {"type": ["integer", "null"]},
        "total_marks": {"type": ["integer", "null"]},

        "header_structure": {
            "type": "array",
            "items": {"type": "string"}
        },

        "general_instructions": {
            "type": "array",
            "items": {"type": "string"}
        },

        "total_questions": {
            "type": ["integer", "null"]
        },

        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "subject_area": {"type": ["string", "null"]},
                    "question_range": {"type": "string"},
                    "patterns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "question_range": {
                                    "type": "string"
                                },
                                "question_types": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "marks_distribution": {
                                    "type": "array",
                                    "items": {
                                        "type": "integer"
                                    }
                                },
                                "notes": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            },
                            "required": [
                                "question_range",
                                "question_types",
                                "marks_distribution",
                                "notes"
                            ]
                        }
                    }
                },
                "required": [
                    "name",
                    "subject_area",
                    "question_range",
                    "patterns"
                ]
            }
        },

        "question_numbering_style": {
            "type": "string"
        },

        "marks_display_style": {
            "type": "string"
        },

        "options_style": {
            "type": "string"
        },

        "internal_choice_style": {
            "type": ["string", "null"]
        },

        "layout_notes": {
            "type": "array",
            "items": {"type": "string"}
        }
    },

    "required": [
        "exam_name",
        "board_name",
        "class_level",
        "subject",
        "duration_minutes",
        "total_marks",
        "header_structure",
        "general_instructions",
        "total_questions",
        "sections",
        "question_numbering_style",
        "marks_display_style",
        "options_style",
        "internal_choice_style",
        "layout_notes"
    ]
}


SYSTEM_PROMPT = """
You are a question-paper format analyzer.

Analyze the supplied REAL question paper reference.

Do NOT generate questions.

The purpose is to extract the actual board-paper pattern so that a new
paper can reproduce its structure and layout while using completely new
questions.

IMPORTANT:

The paper may contain different marks per question inside the same section.
DO NOT assume one marks_each value for an entire section.

Extract:

- exam name
- board
- class
- subject
- duration
- total marks
- total number of questions
- header structure
- general instructions
- sections
- question ranges
- question types
- marks distribution
- internal choices
- option style
- numbering style
- useful layout details

For each pattern, explicitly identify the question range.

Examples:

"Questions 1-20"
"Questions 21-24"
"Questions 25-29"

For marks_distribution, list the marks of the questions in that pattern.

Use null when information genuinely cannot be determined.

Do NOT mistake the statement "39 questions in 3 sections" as meaning
that every section contains 39 questions.

Return ONLY the requested JSON.
"""


def analyze_format(reference_pages: list[str]) -> PaperFormat:
    selected_pages = []

    for page in reference_pages:
        if page.strip():
            selected_pages.append(page)

        if len(selected_pages) >= 6:
            break

    reference_text = "\n\n".join(selected_pages)
    reference_text = reference_text[:6000]

    response = client.chat.completions.create(
        model=MODEL_ID,
        reasoning_effort="low",
        max_completion_tokens=2500,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    "Analyze this actual board question paper reference:\n\n"
                    + reference_text
                )
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "paper_format",
                "strict": True,
                "schema": FORMAT_SCHEMA
            }
        }
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("Format analyzer returned empty output")

    return PaperFormat.model_validate(
        json.loads(content)
    )