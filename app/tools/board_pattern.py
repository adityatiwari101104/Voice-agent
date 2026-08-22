import os
import re
import json
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from .reference_loader import load_reference

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY not found in .env")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

MODEL_ID = "openai/gpt-oss-20b"


@dataclass
class QuestionPattern:
    question_range: str
    question_type: str
    marks_each: int
    count: int
    notes: list[str]


@dataclass
class BoardPattern:
    exam_name: str
    board_name: str
    class_level: str
    subject: str
    duration_minutes: int
    total_marks: int
    total_questions: int
    patterns: list[QuestionPattern]
    general_instructions: list[str]
    header_structure: list[str]


class MarkingSchemePattern(BaseModel):
    question_range: str
    question_type: str
    marks_each: int
    count: int
    notes: list[str]


class MarkingSchemeExtraction(BaseModel):
    patterns: list[MarkingSchemePattern]


PATTERN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question_range": {"type": "string"},
                    "question_type": {"type": "string"},
                    "marks_each": {"type": "integer"},
                    "count": {"type": "integer"},
                    "notes": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": [
                    "question_range",
                    "question_type",
                    "marks_each",
                    "count",
                    "notes"
                ]
            }
        }
    },
    "required": ["patterns"]
}


INSTRUCTIONS_SYSTEM_PROMPT = """
You are analyzing the "General Instructions" section of an official
board question paper (NOT the marking scheme, NOT the solutions).

These instructions typically state the paper's structure explicitly,
for example:

"In Section A, Questions no 1 to 20 are multiple choice questions
(MCQs) of 1 mark each."

"In Section B, Questions no 21 to 25 are Very Short Answer type
questions, carrying 02 marks each."

"In Section E, Questions no 36 to 38 are case study based questions
carrying 4 marks each."

Extract ONE pattern entry per such explicit statement:
- question_range (e.g. "1-20" or "36-38")
- question_type (use the paper's own description, e.g. "MCQ",
  "Very Short Answer", "Short Answer", "Long Answer", "Case Based")
- marks_each (the marks per question stated for that range)
- count (number of questions in that range)
- notes (anything relevant, e.g. "internal choice provided")

IMPORTANT:
- Only extract patterns that are EXPLICITLY stated with a question
  range and a marks value. Do not guess or invent a pattern for a
  section that isn't described this way.
- Different sections almost always have different marks values.
  Do not assume every section shares the same marks_each.
- If the instructions do not describe the structure this way at all,
  return an empty patterns list rather than guessing.

Return ONLY the requested JSON.
"""


MARKING_SCHEME_SYSTEM_PROMPT = """
You are analyzing an official board exam MARKING SCHEME document.

Your job is to extract the exact question-numbering-to-marks pattern.

A marking scheme lists every question number along with how many marks
it is worth. Group consecutive question numbers that share the same
marks value and same question type into a single pattern entry.

IMPORTANT:

A real board paper usually has DIFFERENT mark values in different
question ranges (for example questions 1-20 worth 1 mark each,
questions 21-24 worth 2 marks each, questions 25-29 worth 3 marks each,
questions 30-33 worth 4 marks each, questions 34-36 worth 5 marks each).

Do NOT assume the whole paper uses one mark value.

The text is extracted from a PDF and may have scattered or
out-of-order numbers from multi-column marking annotations. Use the
"Section" headers and question numbers as your primary anchors, and
the surrounding context (marks distribution statements, question
groupings) to infer the pattern. If a question's exact marks cannot be
determined confidently, use your best reasonable estimate based on
the section it falls in rather than omitting it.

For question_type, classify based on marks and structure:
- 1 mark -> "MCQ" or "Very Short Answer" (use context to decide)
- 2 marks -> "Short Answer"
- 3 marks -> "Short Answer"
- 4 marks -> "Case Based" or "Source Based" (use context)
- 5 marks or more -> "Long Answer"

question_range should look like "1-20" for a range, or "21" for a
single question.

count is the number of questions in that range.

Return ONLY the requested JSON.
"""


def _extract_total_marks(text: str) -> int | None:
    match = re.search(
        r"Max\.?\s*Marks\s*:\s*(\d+)",
        text,
        re.IGNORECASE
    )

    return int(match.group(1)) if match else None


def _extract_duration(text: str) -> int | None:
    match = re.search(
        r"Time\s*Allowed\s*:\s*(\d+)\s*hours?",
        text,
        re.IGNORECASE
    )

    return int(match.group(1)) * 60 if match else None


def _extract_total_questions(text: str) -> int | None:
    match = re.search(
        r"consists\s+of\s+(\d+)\s+questions",
        text,
        re.IGNORECASE
    )

    return int(match.group(1)) if match else None


def _looks_like_exam_question(segment: str) -> bool:
    """True if a segment looks like an actual exam question rather
    than a general instruction -- i.e. it has MCQ-style lettered
    options. Used to cut off spillover once instruction numbering
    (1. 2. 3...) runs into the exam's own question numbering, which
    reuses the same "N." pattern and would otherwise get misread as
    more instructions."""

    return bool(
        re.search(r"\(A\).*?\(B\).*?\(C\).*?\(D\)", segment, re.DOTALL)
    )


def _extract_instructions(text: str) -> list[str]:
    idx = text.lower().find("general instructions")

    if idx == -1:
        return []

    # General Instructions blocks are short (CBSE SQPs: ~7-9 numbered
    # items). We deliberately do NOT stop at the first mention of
    # "Section A" -- the instructions themselves say things like
    # "In Section A, Questions no 1 to 20 are..." which would trip a
    # boundary regex looking for "Section A". Instead, grab a generous
    # window and split on the numbered-list markers ("1.", "2.", etc)
    # that structure the instructions themselves.
    window = text[idx:idx + 2200]
    window = re.sub(r"(?i)^general instructions\s*:?", "", window, count=1)

    matches = list(re.finditer(r"(?:^|\s)(\d{1,2})[\.\)]\s+", window))

    if not matches:
        lines = [line.strip() for line in window.splitlines() if line.strip()]
        return lines[:12]

    instructions = []

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(window)
        segment = re.sub(r"\s+", " ", window[start:end]).strip()

        if not segment:
            continue

        # The exam's actual questions reuse "1." "2." numbering, which
        # our regex can't tell apart from instruction numbering. Once
        # we hit something that looks like a real question (lettered
        # MCQ options), stop -- everything from here on is exam
        # content, not instructions.
        if _looks_like_exam_question(segment):
            break

        # Strip trailing spillover like "(Section A) Section A
        # consists of 20 questions... Q.No. Questions Marks" that can
        # get glued onto the last real instruction before the exam
        # content begins.
        segment = re.sub(
            r"\(Section.*$", "", segment, flags=re.IGNORECASE | re.DOTALL
        ).strip()

        if segment:
            instructions.append(segment)

    return instructions


def _analyze_general_instructions(instructions_text: str) -> list[QuestionPattern]:
    if not instructions_text.strip():
        return []

    # This text is short and clean (a handful of numbered sentences),
    # so no truncation/rate-limit concerns here.
    response = client.chat.completions.create(
        model=MODEL_ID,
        reasoning_effort="low",
        max_completion_tokens=1200,
        messages=[
            {
                "role": "system",
                "content": INSTRUCTIONS_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    "Analyze these general instructions from an "
                    "official board question paper:\n\n"
                    + instructions_text
                )
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "instructions_patterns",
                "strict": True,
                "schema": PATTERN_SCHEMA
            }
        }
    )

    content = response.choices[0].message.content

    if not content:
        return []

    extraction = MarkingSchemeExtraction.model_validate(json.loads(content))

    return [
        QuestionPattern(
            question_range=pattern.question_range,
            question_type=pattern.question_type,
            marks_each=pattern.marks_each,
            count=pattern.count,
            notes=pattern.notes
        )
        for pattern in extraction.patterns
    ]


def _analyze_marking_scheme(marking_scheme_text: str) -> list[QuestionPattern]:
    # Fallback path only. Kept small to stay well under Groq's free-tier
    # 8000 TPM rate limit -- this text is noisy (scattered partial-mark
    # fragments from a multi-column PDF layout) so we lean on the model's
    # own judgement rather than trying to feed it the whole document.
    truncated_text = marking_scheme_text[:6000]

    response = client.chat.completions.create(
        model=MODEL_ID,
        reasoning_effort="low",
        max_completion_tokens=1500,
        messages=[
            {
                "role": "system",
                "content": MARKING_SCHEME_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    "Analyze this official marking scheme document:\n\n"
                    + truncated_text
                )
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "marking_scheme_patterns",
                "strict": True,
                "schema": PATTERN_SCHEMA
            }
        }
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("Marking scheme analysis returned empty output")

    extraction = MarkingSchemeExtraction.model_validate(json.loads(content))

    return [
        QuestionPattern(
            question_range=pattern.question_range,
            question_type=pattern.question_type,
            marks_each=pattern.marks_each,
            count=pattern.count,
            notes=pattern.notes
        )
        for pattern in extraction.patterns
    ]


def build_board_pattern(
    sqp_url: str,
    marking_scheme_url: str,
    board_name: str,
    class_level: str,
    subject: str
) -> BoardPattern:
    """
    Build a BoardPattern from a dynamically discovered SQP and marking
    scheme. board_name / class_level / subject should come from the
    teacher's request (via PaperRequest), not be hardcoded.
    """

    sqp_text = load_reference(sqp_url)

    total_marks = _extract_total_marks(sqp_text)
    duration_minutes = _extract_duration(sqp_text)
    total_questions = _extract_total_questions(sqp_text)
    instructions = _extract_instructions(sqp_text)

    # Primary source: the SQP's own General Instructions, which state
    # the section/marks structure explicitly and cleanly. Far more
    # reliable than reconstructing it from noisy marking-scheme text.
    patterns = _analyze_general_instructions("\n".join(instructions))

    if not patterns:
        # Fallback: attempt to derive the pattern from the marking
        # scheme instead. Noisier and more expensive, used only if the
        # instructions didn't yield anything usable.
        marking_scheme_text = load_reference(marking_scheme_url)
        patterns = _analyze_marking_scheme(marking_scheme_text)

    if not patterns:
        raise RuntimeError(
            "Could not extract the question/marks pattern from either "
            "the official general instructions or the marking scheme. "
            "This board/subject's SQP may use a non-standard format."
        )

    if total_marks is None:
        total_marks = sum(p.marks_each * p.count for p in patterns)

    if total_questions is None:
        total_questions = sum(p.count for p in patterns)

    if duration_minutes is None:
        duration_minutes = 180

    header_structure = [
        f"{subject.upper()}",
        "SAMPLE QUESTION PAPER",
        f"CLASS \u2013 {class_level}",
        f"Max. Marks: {total_marks}",
        f"Time Allowed: {duration_minutes // 60} hours"
    ]

    return BoardPattern(
        exam_name="SAMPLE QUESTION PAPER",
        board_name=board_name,
        class_level=class_level,
        subject=subject,
        duration_minutes=duration_minutes,
        total_marks=total_marks,
        total_questions=total_questions,
        patterns=patterns,
        general_instructions=instructions,
        header_structure=header_structure
    )


def get_board_pattern(
    sqp_url: str,
    marking_scheme_url: str,
    board_name: str,
    class_level: str,
    subject: str
) -> dict:
    return asdict(
        build_board_pattern(
            sqp_url,
            marking_scheme_url,
            board_name,
            class_level,
            subject
        )
    )