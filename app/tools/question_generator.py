import os
import json

from dotenv import load_dotenv
from openai import OpenAI, APIError
from pydantic import BaseModel

from ..schemas import PaperRequest

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY not found in .env")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

MODEL_ID = "openai/gpt-oss-20b"


class Question(BaseModel):
    question: str
    options: list[str] | None
    answer: str


class GeneratedSection(BaseModel):
    type: str
    count: int
    marks_each: int
    questions: list[Question]


class GeneratedPaper(BaseModel):
    sections: list[GeneratedSection]


SECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "type": {
            "type": "string"
        },
        "count": {
            "type": "integer"
        },
        "marks_each": {
            "type": "integer"
        },
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question": {
                        "type": "string"
                    },
                    "options": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "string"
                        }
                    },
                    "answer": {
                        "type": "string"
                    }
                },
                "required": [
                    "question",
                    "options",
                    "answer"
                ]
            }
        }
    },
    "required": [
        "type",
        "count",
        "marks_each",
        "questions"
    ]
}


SYSTEM_PROMPT = """
You are a question-generation component for a school question-paper generator.

Generate ONLY the requested section.

Rules:
1. Generate exactly the requested number of questions.
2. Use exactly the requested marks per question.
3. Follow the class, subject, chapters, and difficulty.
4. Do not add extra questions.
5. Do not omit questions.
6. For MCQ sections, every question must have exactly four options.
7. For non-MCQ sections, options must be null.
8. Provide the correct answer for every question.
9. Do not generate explanations outside the JSON.
10. Return only the requested JSON structure.
11. Write all mathematical notation in PLAIN TEXT, not LaTeX. Use
    "x^2", "sqrt(2)", "a/b" instead of backslash commands like
    "\\frac", "\\sqrt", "\\neq". Backslashes and LaTeX markup break
    JSON formatting -- never use them.
12. Keep answers concise (a few sentences or a short worked solution).
    Do not write long multi-paragraph proofs.
"""


def _is_case_study(section_type: str) -> bool:
    """Detect case-study/case-based sections regardless of exactly how
    board_pattern's LLM extraction phrased the type string (e.g. 'Case
    Based', 'Case Study-Based', 'Case Study' have all shown up)."""

    return "case" in section_type.lower()


CASE_STUDY_INSTRUCTIONS = """
This is a CASE STUDY section. Each "question" is actually one
self-contained case study, not a single question. For each one:

1. Write a short realistic passage/scenario (2-4 sentences) relevant
   to the class, subject and chapters given -- something a student
   would need to read and apply concepts to.
2. Follow it with 2-3 clearly labeled sub-parts: (i), (ii), (iii).
3. Give each sub-part its own mark value in brackets, e.g. "[1 mark]".
   The sub-part marks MUST sum to exactly the marks_each value given
   for this section. A common CBSE pattern for a 4-mark case study is
   sub-parts worth 1, 1, and 2 marks -- use that split if it fits,
   otherwise choose a sensible split that still sums correctly.
4. Put all of this (passage + labeled sub-parts + their marks) into
   the single "question" text field, formatted for readability with
   line breaks.
5. In the "answer" field, give labeled answers matching each sub-part,
   e.g. "(i) ... (ii) ... (iii) ...".
6. options must be null.
"""


def _generate_section_once(
    paper_request: PaperRequest,
    section_type: str,
    count: int,
    marks_each: int
) -> GeneratedSection:

    case_study_block = (
        CASE_STUDY_INSTRUCTIONS if _is_case_study(section_type) else ""
    )

    prompt = f"""
Generate exactly {count} questions for this section.

Section type: {section_type}
Marks per question: {marks_each}

Class: {paper_request.class_level}
Subject: {paper_request.subject}
Chapters: {", ".join(paper_request.chapters)}
Difficulty: {paper_request.difficulty}
{case_study_block}
Return:
- type = "{section_type}"
- count = {count}
- marks_each = {marks_each}
- exactly {count} questions
"""

    response = client.chat.completions.create(
        model=MODEL_ID,
        reasoning_effort="medium",
        max_completion_tokens=2500,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "generated_section",
                "strict": True,
                "schema": SECTION_SCHEMA
            }
        }
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            f"Question generator returned empty output for section: {section_type}"
        )

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON generated for section '{section_type}':\n{content}"
        ) from exc

    section = GeneratedSection.model_validate(data)

    if section.type != section_type:
        raise ValueError(
            f"Expected section '{section_type}', got '{section.type}'"
        )

    if section.count != count:
        raise ValueError(
            f"Expected {count} questions in '{section_type}', got {section.count}"
        )

    if section.marks_each != marks_each:
        raise ValueError(
            f"Expected {marks_each} marks each in '{section_type}', "
            f"got {section.marks_each}"
        )

    if len(section.questions) < count:
        raise ValueError(
            f"Expected at least {count} actual questions in "
            f"'{section_type}', got {len(section.questions)}"
        )

    return section


def _generate_section(
    paper_request: PaperRequest,
    section_type: str,
    count: int,
    marks_each: int,
    max_attempts: int = 3
) -> GeneratedSection:
    """Generate a section, retrying on validation failures since the
    model occasionally drifts on the question count.

    If a successful attempt overshot the requested count, trim the
    extras rather than treating it as a failure -- the surplus
    questions are still individually valid, so trimming is safe.
    Undershooting the count can't be fixed by trimming, so that keeps
    triggering a retry until attempts run out.
    """

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            section = _generate_section_once(
                paper_request=paper_request,
                section_type=section_type,
                count=count,
                marks_each=marks_each
            )

            if len(section.questions) > count:
                section.questions = section.questions[:count]
                section.count = count

            return section

        except (ValueError, RuntimeError, APIError) as exc:
            last_error = exc

    raise RuntimeError(
        f"Question generation for section '{section_type}' failed "
        f"validation after {max_attempts} attempts: {last_error}"
    )


def generate_questions(paper_request: PaperRequest) -> GeneratedPaper:
    sections = []

    for requested_section in paper_request.sections:
        section = _generate_section(
            paper_request=paper_request,
            section_type=requested_section.type,
            count=requested_section.count,
            marks_each=requested_section.marks_each
        )

        sections.append(section)

    return GeneratedPaper(sections=sections)