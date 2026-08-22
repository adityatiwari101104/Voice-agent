from ..schemas import PaperRequest, Section
from .question_generator import generate_questions


paper_request = PaperRequest(
    class_level="10",
    subject="Physics",
    chapters=["Light and Human Eye"],
    total_marks=50,
    duration_minutes=None,
    difficulty="medium",
    sections=[
        Section(type="MCQ", count=10, marks_each=1),
        Section(type="two-mark", count=5, marks_each=2),
        Section(type="five-mark", count=4, marks_each=5)
    ]
)


result = generate_questions(paper_request)

print("\nQUESTION GENERATOR TEST\n")

total_questions = 0

for section in result.sections:
    print(
        f"{section.type}: "
        f"{len(section.questions)} generated | "
        f"count={section.count} | "
        f"marks_each={section.marks_each}"
    )

    total_questions += len(section.questions)

print(f"\nTotal questions generated: {total_questions}")

if total_questions != 19:
    raise RuntimeError(
        f"Expected 19 questions, got {total_questions}"
    )

print("\nQuestion generation passed.")